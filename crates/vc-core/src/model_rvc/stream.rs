use anyhow::{anyhow, Result};

use crate::dsp;

use super::shape::{
    feature_len_for_samples, keep_tail_in_place, samples_between_rates, tensor_rt_convert_size_16k,
    Rounding, EMBEDDER_SAMPLE_RATE, RMVPE_FRAME_SAMPLES_16K,
};
use super::speech_activity::{SpeechActivityFeatureAccumulator, SpeechActivityFeatures};

pub(super) const VOLUME_DECAY: f32 = 0.97;

// Speech-preserving denoiser mixing runs on the fixed 16 kHz model timeline.
// Evidence is refreshed every 10 ms, while the actual mix moves per sample so a
// frame decision cannot create a step in either model input.
const SPEECH_MIX_FRAME_SAMPLES: usize = RMVPE_FRAME_SAMPLES_16K;
const SPEECH_MIX_SIGNAL_FLOOR: f32 = 1.0e-5;
const SPEECH_MIX_CONTENT_MAX_REDUCTION: f32 = 0.78;
const SPEECH_MIX_RMVPE_MAX_REDUCTION: f32 = 0.18;
// At 16 kHz these one-pole coefficients are roughly a 2 ms protection attack
// and a 52 ms recovery. Protection must arrive within a consonant, while the
// denoised share should return slowly enough to avoid pumping between frames.
const SPEECH_MIX_FAST_ALPHA: f32 = 0.03;
const SPEECH_MIX_SLOW_ALPHA: f32 = 0.0012;

// This state is owned by the model worker, not the realtime audio callback.
// Keep resizing and resampling work here so callback code remains queue-only.

/// Fixed sample delay used to align a residual ContentVec branch with a
/// fixed-delay denoiser. Configuration happens during model load or a worker
/// mode switch; `process_in_place` itself does not allocate or block.
#[derive(Default)]
pub(super) struct SampleDelay {
    samples: Vec<f32>,
    next: usize,
}

impl SampleDelay {
    pub(super) fn configure(&mut self, delay_samples: usize) {
        if self.samples.len() != delay_samples {
            self.samples.resize(delay_samples, 0.0);
        } else {
            self.samples.fill(0.0);
        }
        self.next = 0;
    }

    pub(super) fn reset(&mut self) {
        self.samples.fill(0.0);
        self.next = 0;
    }

    #[cfg(feature = "gtcrn")]
    pub(super) fn delay_samples(&self) -> usize {
        self.samples.len()
    }

    pub(super) fn process_in_place(&mut self, audio: &mut [f32]) {
        if self.samples.is_empty() {
            return;
        }
        for sample in audio {
            std::mem::swap(&mut self.samples[self.next], sample);
            self.next += 1;
            if self.next == self.samples.len() {
                self.next = 0;
            }
        }
    }
}

pub(super) struct RvcStreamInput {
    pub(super) convert_size: usize,
    pub(super) out_size: usize,
    pub(super) volume: f32,
    // RMS of the new 16 kHz pitch increment. RMVPE and silence detection use
    // the configured raw/denoised blend; ContentVec may use a separate blend.
    pub(super) input_rms: f32,
    /// Number of new frames appended to the raw RMVPE timeline this update.
    /// The pipeline uses it to draw raw-F0 speech evidence from the newest
    /// increment only, never from old rolling context.
    pub(super) new_feature_frames: usize,
    /// A device-rate or input-branch change cleared the model timeline. The
    /// pipeline must reset all worker-owned activity state at this boundary.
    pub(super) stream_restarted: bool,
    /// Acoustic measurements of the newest configured RMVPE branch. These are
    /// calculated once on the worker and reused by adaptive output muting.
    pub(super) speech_features: SpeechActivityFeatures,
}

/// Worker-owned timing and alignment inputs for one stream update. Grouping
/// these values keeps the dual ContentVec/RMVPE entry point explicit without
/// growing a fragile positional-argument list as denoiser handling grows. It
/// is copied on the worker stack and never allocates on the audio path.
#[derive(Clone, Copy)]
pub(super) struct StreamInputTiming {
    pub(super) sample_rate: u32,
    pub(super) crossfade_and_search_samples: usize,
    pub(super) volume_excluded_samples: usize,
    pub(super) extra_convert_samples: usize,
    pub(super) denoiser_content_mix: f32,
    pub(super) denoiser_rmvpe_mix: f32,
}

impl RvcStreamState {
    /// Tail of the ContentVec 16 kHz rolling signal, resampled to the RVC output
    /// rate, for the RMS-mix reference. This follows the content branch so a
    /// residual-preserving denoiser mix does not make output leveling chase a
    /// different signal. Callers pass `input_sample_rate = EMBEDDER_SAMPLE_RATE`.
    pub(super) fn output_reference_audio<'a>(
        &'a self,
        input_sample_rate: u32,
        output_sample_rate: u32,
        output_samples: usize,
        scratch: &'a mut Vec<f32>,
    ) -> Result<&'a [f32]> {
        scratch.clear();
        if self.audio_16k_buffer.is_empty()
            || output_samples == 0
            || input_sample_rate == 0
            || output_sample_rate == 0
        {
            return Ok(scratch.as_slice());
        }

        let input_samples = samples_between_rates(
            output_samples,
            output_sample_rate,
            input_sample_rate,
            Rounding::Ceil,
        )
        .max(1);
        let start = self.audio_16k_buffer.len().saturating_sub(input_samples);
        let input_tail = &self.audio_16k_buffer[start..];
        if input_sample_rate == output_sample_rate && input_tail.len() >= output_samples {
            return Ok(&input_tail[input_tail.len() - output_samples..]);
        }

        if input_sample_rate == output_sample_rate {
            scratch.extend_from_slice(input_tail);
        } else {
            let reference = dsp::resample_mono(
                input_tail,
                input_sample_rate as usize,
                output_sample_rate as usize,
            )?;
            scratch.extend_from_slice(&reference);
        }
        keep_tail_in_place(scratch, output_samples);
        left_pad_to_len_in_place(scratch, output_samples);

        Ok(scratch.as_slice())
    }
}

fn left_pad_to_len_in_place(values: &mut Vec<f32>, len: usize) {
    if values.len() >= len {
        return;
    }
    let old_len = values.len();
    let pad = len - old_len;
    values.resize(len, 0.0);
    values.copy_within(0..old_len, pad);
    values[..pad].fill(0.0);
}

pub(super) struct RvcStreamState {
    pub(super) audio_buffer: Vec<f32>,
    /// ContentVec input: raw/denoised blend that preserves residual speech
    /// detail when a denoiser is active.
    pub(super) audio_16k_buffer: Vec<f32>,
    /// RMVPE input: raw/denoised blend used for voiced/unvoiced decisions.
    /// Both pitch buffers stay on the same 16 kHz sample grid.
    pub(super) pitch_16k_buffer: Vec<f32>,
    /// Raw branch aligned to the denoiser output. It is kept only while a
    /// separate pitch path is active so live RMVPE mix changes can be applied
    /// to each new increment without allocating a temporary buffer.
    rmvpe_raw_16k_buffer: Vec<f32>,
    pub(super) pitchf_buffer: Vec<f32>,
    /// Unconsumed 16 kHz samples on the 10 ms F0 grid.  Counting frames from
    /// each resampler increment independently would floor away a fractional
    /// remainder at every callback (most visible at 44.1 kHz).
    feature_sample_remainder_16k: usize,
    pub(super) prev_vol: f32,
    pub(super) prev_silence: bool,
    pub(super) sample_rate: u32,
    /// The RVC model's native output rate (from metadata `samplingRate`, default
    /// `RVC_SAMPLE_RATE`). Fixed per model — distinct from `sample_rate`, which is
    /// the device/input rate. Sizes `out_size` and the RVC-domain conversions.
    pub(super) rvc_sample_rate: u32,
    /// Fixed ContentVec waveform length selected while the pipeline loaded.
    /// `None` preserves timing-derived behaviour for tests and legacy callers.
    /// When present, it is the exact profile/CUDA-graph input size and only
    /// enlarges the rolling left context; it never changes per callback.
    contentvec_context_samples_16k: Option<usize>,
    pub(super) resampler_16k: Option<dsp::StreamingResampleMono>,
    pub(super) pitch_resampler_16k: Option<dsp::StreamingResampleMono>,
    // Align raw ContentVec samples with GTCRN's fixed-delay output before the
    // two are mixed. Do not remove this as an apparent duplicate delay: mixing
    // delayed GTCRN output with undelayed speech causes audible echo/phase smear.
    content_delay_16k: SampleDelay,
    // GTCRN input denoiser applied to each new 16 kHz pitch increment before it
    // is independently mixed into the ContentVec and RMVPE branches. `Some`
    // only when the pipeline was built with `load_with_gtcrn`. At 16 kHz the
    // adapter's resamplers are bypass, so only its frame FIFO + fixed delay run.
    #[cfg(feature = "gtcrn")]
    pub(super) gtcrn: Option<crate::denoise::GtcrnDenoiser>,
    // Scalar-only analysis/envelope state for the aligned raw/denoised mix. It
    // belongs to the conversion worker and is reset with the stream timeline;
    // do not move it into a callback or turn its 10 ms analysis into allocation.
    speech_preserving_mix: SpeechPreservingMix,
    // Keep acoustic frame phase and the previous sample across arbitrary input
    // increments. Recreating this on every callback makes activity decisions
    // depend on chunk boundaries and can drop a quiet consonant at a split.
    speech_feature_accumulator: SpeechActivityFeatureAccumulator,
}

impl RvcStreamState {
    pub(super) fn new(rvc_sample_rate: u32) -> Self {
        Self {
            audio_buffer: Vec::new(),
            audio_16k_buffer: Vec::new(),
            pitch_16k_buffer: Vec::new(),
            rmvpe_raw_16k_buffer: Vec::new(),
            pitchf_buffer: Vec::new(),
            feature_sample_remainder_16k: 0,
            prev_vol: 0.0,
            prev_silence: false,
            sample_rate: 0,
            rvc_sample_rate,
            contentvec_context_samples_16k: None,
            resampler_16k: None,
            pitch_resampler_16k: None,
            content_delay_16k: SampleDelay::default(),
            #[cfg(feature = "gtcrn")]
            gtcrn: None,
            speech_preserving_mix: SpeechPreservingMix::default(),
            speech_feature_accumulator: SpeechActivityFeatureAccumulator::default(),
        }
    }

    pub(super) fn set_contentvec_context_samples_16k(&mut self, samples: usize) {
        self.contentvec_context_samples_16k = Some(samples);
    }

    /// Return the load-time ContentVec waveform context, if one was selected.
    ///
    /// A denoiser/passthrough reset replaces transient stream history, but this
    /// value is part of the TensorRT/profile contract and must survive it.
    pub(super) fn contentvec_context_samples_16k(&self) -> Option<usize> {
        self.contentvec_context_samples_16k
    }

    pub(super) fn generate_input(
        &mut self,
        new_audio: &[f32],
        sample_rate: u32,
        crossfade_and_search_samples: usize,
        volume_excluded_samples: usize,
        extra_convert_samples: usize,
    ) -> Result<RvcStreamInput> {
        // The no-denoiser path must stay single-resample: ContentVec and RMVPE
        // consume the same 16 kHz increment, so copy that increment into the
        // pitch history instead of running a second streaming resampler.
        self.generate_input_inner(
            new_audio,
            None,
            StreamInputTiming {
                sample_rate,
                crossfade_and_search_samples,
                volume_excluded_samples,
                extra_convert_samples,
                denoiser_content_mix: 0.0,
                // The field is unused when both model inputs share one path;
                // keep the no-denoiser helper on the legacy full-pitch value.
                denoiser_rmvpe_mix: 1.0,
            },
        )
    }

    /// Append one chunk to separate ContentVec and RMVPE streams. `content_audio`
    /// is the gain-scaled source that retains articulation; `pitch_audio` is the
    /// source sent through the configured denoiser. The blend is applied only to
    /// the new 16 kHz increment so old history is never mixed twice.
    pub(super) fn generate_input_with_pitch(
        &mut self,
        content_audio: &[f32],
        pitch_audio: &[f32],
        timing: StreamInputTiming,
    ) -> Result<RvcStreamInput> {
        self.generate_input_inner(content_audio, Some(pitch_audio), timing)
    }

    fn generate_input_inner(
        &mut self,
        content_audio: &[f32],
        pitch_audio: Option<&[f32]>,
        timing: StreamInputTiming,
    ) -> Result<RvcStreamInput> {
        if let Some(pitch_audio) = pitch_audio {
            if content_audio.len() != pitch_audio.len() {
                return Err(anyhow!(
                    "dual-path input lengths differ: content={} pitch={}",
                    content_audio.len(),
                    pitch_audio.len()
                ));
            }
        }
        let separate_pitch_path = pitch_audio.is_some();
        // A path change also restarts both resamplers. Otherwise a newly-created
        // pitch resampler would begin at a different filter phase than the
        // existing ContentVec stream. Public denoiser switches additionally
        // reset the outer chunk smoother before another chunk is emitted.
        let input_path_changed = self.pitch_resampler_16k.is_some() != separate_pitch_path;
        let stream_restarted = self.sample_rate != timing.sample_rate
            || self.resampler_16k.is_none()
            || input_path_changed;
        if stream_restarted {
            self.audio_buffer.clear();
            self.audio_16k_buffer.clear();
            self.pitch_16k_buffer.clear();
            self.rmvpe_raw_16k_buffer.clear();
            self.pitchf_buffer.clear();
            self.feature_sample_remainder_16k = 0;
            self.content_delay_16k.reset();
            self.speech_preserving_mix.reset();
            self.speech_feature_accumulator.reset();
            self.prev_vol = 0.0;
            self.prev_silence = false;
            self.sample_rate = timing.sample_rate;
            self.resampler_16k = Some(dsp::StreamingResampleMono::new(
                timing.sample_rate as usize,
                EMBEDDER_SAMPLE_RATE as usize,
            )?);
            self.pitch_resampler_16k = if separate_pitch_path {
                Some(dsp::StreamingResampleMono::new(
                    timing.sample_rate as usize,
                    EMBEDDER_SAMPLE_RATE as usize,
                )?)
            } else {
                None
            };
            // A device sample-rate change restarts the stream; reset GTCRN's
            // fixed-delay/cache state so it does not emit pre-restart audio.
            #[cfg(feature = "gtcrn")]
            if let Some(gtcrn) = self.gtcrn.as_mut() {
                gtcrn.reset()?;
            }
        }

        self.audio_buffer.extend_from_slice(content_audio);
        let new_16k_start = self.audio_16k_buffer.len();
        self.resampler_16k
            .as_mut()
            .ok_or_else(|| anyhow!("16kHz stream resampler is not initialized"))?
            .process_into(content_audio, &mut self.audio_16k_buffer)?;
        let new_pitch_16k_start = self.pitch_16k_buffer.len();
        if let Some(pitch_audio) = pitch_audio {
            self.pitch_resampler_16k
                .as_mut()
                .ok_or_else(|| anyhow!("16kHz pitch resampler is not initialized"))?
                .process_into(pitch_audio, &mut self.pitch_16k_buffer)?;
        } else {
            self.pitch_16k_buffer
                .extend_from_slice(&self.audio_16k_buffer[new_16k_start..]);
        }
        let content_16k_len = self.audio_16k_buffer.len() - new_16k_start;
        let pitch_16k_len = self.pitch_16k_buffer.len() - new_pitch_16k_start;
        if content_16k_len != pitch_16k_len {
            return Err(anyhow!(
                "dual-path resamplers produced different lengths: content={} pitch={}",
                content_16k_len,
                pitch_16k_len
            ));
        }
        // Rubato keeps pending input and output delay internally, so the
        // emitted count can differ from a rounded device-rate estimate. Drive
        // the feature/F0 timeline from the actual increment; otherwise a
        // 44.1 kHz stream slowly accumulates phantom frames and drifts away
        // from the ContentVec/RMVPE waveform timeline.
        let new_audio_16k_samples = content_16k_len;
        let feature_sample_total = self
            .feature_sample_remainder_16k
            .saturating_add(content_16k_len);
        let new_feature_len = feature_sample_total / RMVPE_FRAME_SAMPLES_16K;
        self.feature_sample_remainder_16k = feature_sample_total % RMVPE_FRAME_SAMPLES_16K;
        // GTCRN denoises exactly the new 16 kHz increment, in place, BEFORE the
        // windowing below. Guardrail: process only the increment, never the
        // re-windowed `audio_16k_buffer`, so its length — and thus
        // `new_audio_16k_samples` and the ContentVec/F0 window length — stays
        // unchanged. The fixed delay is internal (adds latency, never shifts the
        // sample grid). Each sample is denoised exactly once, at append time.
        #[cfg(feature = "gtcrn")]
        if let Some(gtcrn_delay_samples) = self.gtcrn.as_ref().map(|gtcrn| gtcrn.latency_samples())
        {
            // `set_gtcrn` configures this at load/hot-switch time. Keep the
            // assertion so future changes cannot silently mix different
            // timelines; reconfiguring here would allocate in the hot worker
            // path and would also lose residual alignment mid-stream.
            debug_assert_eq!(self.content_delay_16k.delay_samples(), gtcrn_delay_samples);
            self.gtcrn
                .as_mut()
                .expect("GTCRN was present while reading its delay")
                .process_in_place(&mut self.pitch_16k_buffer[new_pitch_16k_start..])?;
            self.content_delay_16k
                .process_in_place(&mut self.audio_16k_buffer[new_16k_start..]);
        }
        // Save the aligned raw increment before ContentVec's independent mix
        // mutates `audio_16k_buffer`. This copy is worker-owned and reused over
        // the rolling window; it keeps RMVPE automation allocation-free.
        let new_rmvpe_raw_16k_start = self.rmvpe_raw_16k_buffer.len();
        if separate_pitch_path {
            self.rmvpe_raw_16k_buffer
                .extend_from_slice(&self.audio_16k_buffer[new_16k_start..]);
        }
        if separate_pitch_path {
            // Analyze both aligned branches before either is mixed. The helper
            // keeps only scalar state, works on the new increment exactly once,
            // and writes both model inputs with click-free per-sample envelopes.
            self.speech_preserving_mix.process_in_place(
                &self.rmvpe_raw_16k_buffer[new_rmvpe_raw_16k_start..],
                &mut self.audio_16k_buffer[new_16k_start..],
                &mut self.pitch_16k_buffer[new_pitch_16k_start..],
                timing.denoiser_content_mix,
                timing.denoiser_rmvpe_mix,
            );
        }
        // Silence detection follows the configured RMVPE branch, while
        // ContentVec receives its own residual-preserving blend above. Analyze
        // only this new increment: old rolling context must not keep an output
        // gate open after the speaker has stopped.
        let speech_features = self
            .speech_feature_accumulator
            .observe(&self.pitch_16k_buffer[new_pitch_16k_start..]);
        let input_rms = speech_features.rms;
        self.pitchf_buffer
            .extend(std::iter::repeat_n(0.0, new_feature_len));

        let extra_16k_samples = samples_between_rates(
            timing.extra_convert_samples,
            self.rvc_sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let volume_excluded_16k_samples = samples_between_rates(
            timing.volume_excluded_samples,
            self.rvc_sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let timing_convert_size_16k = tensor_rt_convert_size_16k(
            content_audio.len(),
            timing.sample_rate,
            timing.crossfade_and_search_samples,
            timing.extra_convert_samples,
            self.rvc_sample_rate,
        );
        let convert_size_16k = self
            .contentvec_context_samples_16k
            .unwrap_or(timing_convert_size_16k);
        if convert_size_16k < timing_convert_size_16k {
            return Err(anyhow!(
                "fixed ContentVec context {convert_size_16k} is smaller than this chunk needs {timing_convert_size_16k}"
            ));
        }
        let convert_size = samples_between_rates(
            convert_size_16k,
            EMBEDDER_SAMPLE_RATE,
            timing.sample_rate,
            Rounding::Ceil,
        );
        // A custom T adds only *left* history. Keep the emitted tail based on
        // the timing-derived window so the chunk converter's output cadence and
        // latency do not grow with the selected TensorRT profile.
        let out_size = samples_between_rates(
            timing_convert_size_16k.saturating_sub(extra_16k_samples),
            EMBEDDER_SAMPLE_RATE,
            self.rvc_sample_rate,
            Rounding::Floor,
        );
        let out_size = out_size.max(1);
        let feature_size = feature_len_for_samples(convert_size_16k, EMBEDDER_SAMPLE_RATE);

        // Left-pad with zeros in place (reusing the buffers) when a chunk arrives
        // before enough history has accumulated — startup and just after a
        // passthrough->RVC switch resets the state.
        left_pad_to_len_in_place(&mut self.audio_buffer, convert_size);
        left_pad_to_len_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        left_pad_to_len_in_place(&mut self.pitch_16k_buffer, convert_size_16k);
        if separate_pitch_path {
            left_pad_to_len_in_place(&mut self.rmvpe_raw_16k_buffer, convert_size_16k);
        }
        left_pad_to_len_in_place(&mut self.pitchf_buffer, feature_size);

        keep_tail_in_place(&mut self.audio_buffer, convert_size);
        keep_tail_in_place(&mut self.audio_16k_buffer, convert_size_16k);
        keep_tail_in_place(&mut self.pitch_16k_buffer, convert_size_16k);
        if separate_pitch_path {
            keep_tail_in_place(&mut self.rmvpe_raw_16k_buffer, convert_size_16k);
        }
        keep_tail_in_place(&mut self.pitchf_buffer, feature_size);

        // Volume envelope memory on the 16 kHz timeline (same signal as
        // ContentVec/F0), the new-increment region minus the excluded tail. The
        // crop is at the back of the buffer, so in steady state it avoids the
        // front zero pad; at stream start it may dip into the pad exactly as the
        // former device-rate crop did (parity, not a regression).
        let crop_len_16k = new_audio_16k_samples + volume_excluded_16k_samples;
        let crop_end_16k = volume_excluded_16k_samples;
        let volume = if crop_len_16k > crop_end_16k && self.audio_16k_buffer.len() >= crop_len_16k {
            let end = self.audio_16k_buffer.len().saturating_sub(crop_end_16k);
            let start = self.audio_16k_buffer.len().saturating_sub(crop_len_16k);
            dsp::rms(&self.audio_16k_buffer[start..end])
        } else {
            0.0
        };
        // Keep a short memory of previous chunk loudness so envelope-based
        // output shaping does not collapse instantly between adjacent chunks.
        let volume = volume.max(self.prev_vol * VOLUME_DECAY);
        self.prev_vol = volume;

        Ok(RvcStreamInput {
            convert_size,
            out_size,
            volume,
            input_rms,
            new_feature_frames: new_feature_len,
            stream_restarted,
            speech_features,
        })
    }

    pub(super) fn update_pitchf_from_estimator_window(
        &mut self,
        f0: &[f32],
        window_start_samples_16k: usize,
    ) {
        let dst_start = window_start_samples_16k / RMVPE_FRAME_SAMPLES_16K;
        if dst_start >= self.pitchf_buffer.len() {
            return;
        }
        let n = (self.pitchf_buffer.len() - dst_start).min(f0.len());
        if n == 0 {
            return;
        }
        // RMVPE and FCPE exports may emit one center-padded frame past
        // `(samples / hop)`. Copy from the front and let any trailing frame fall
        // off while preserving the absolute offset of a tail-only F0 window.
        self.pitchf_buffer[dst_start..dst_start + n].copy_from_slice(&f0[..n]);
    }

    /// Return the newest RMVPE-branch samples after denoising/mixing. The
    /// shared output-side VAD must consume this exact tail rather than old
    /// rolling context, otherwise a previous voiced phrase can hold the idle
    /// gate open after the speaker stops.
    pub(super) fn newest_pitch_audio(&self, sample_count: usize) -> &[f32] {
        let start = self.pitch_16k_buffer.len().saturating_sub(sample_count);
        &self.pitch_16k_buffer[start..]
    }

    /// Return raw estimator frames belonging to the newest stream increment. This
    /// is intentionally a tail view of `pitchf_buffer`: all callers must use it
    /// only after `update_pitchf_from_estimator_window` has installed fresh F0.
    pub(super) fn newest_raw_pitchf(&self, frame_count: usize) -> &[f32] {
        if frame_count == 0 {
            return &[];
        }
        let start = self.pitchf_buffer.len().saturating_sub(frame_count);
        &self.pitchf_buffer[start..]
    }

    /// Install or remove GTCRN off the audio callback. The raw 16 kHz delay is
    /// reset together with the model so its first residual frame is aligned with
    /// GTCRN's startup zeros rather than stale pre-switch audio.
    #[cfg(feature = "gtcrn")]
    pub(super) fn set_gtcrn(&mut self, gtcrn: Option<crate::denoise::GtcrnDenoiser>) {
        let delay_samples = gtcrn
            .as_ref()
            .map(|denoiser| denoiser.latency_samples())
            .unwrap_or(0);
        self.content_delay_16k.configure(delay_samples);
        self.gtcrn = gtcrn;
    }
}

#[derive(Default)]
struct SpeechPreservingMix {
    previous_raw_rms: Option<f32>,
    previous_denoised_rms: Option<f32>,
    previous_raw_sample: Option<f32>,
    // Number of samples already consumed in the current 10 ms analysis frame.
    // Device chunks rarely land on this boundary, so resetting `chunks(160)`
    // at every call makes transient protection move with callback scheduling.
    frame_phase: usize,
    content_mix: Option<f32>,
    rmvpe_mix: Option<f32>,
}

impl SpeechPreservingMix {
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Adapt the user-selected denoised shares without changing either branch's
    /// length or alignment. A frame must contain an energy rise before residual
    /// or spectral-shape evidence can reduce a mix; stationary room noise thus
    /// returns to the configured base instead of leaking through the raw path.
    fn process_in_place(
        &mut self,
        raw: &[f32],
        content: &mut [f32],
        pitch: &mut [f32],
        content_base_mix: f32,
        rmvpe_base_mix: f32,
    ) {
        debug_assert_eq!(raw.len(), content.len());
        debug_assert_eq!(raw.len(), pitch.len());
        let content_base_mix = normalized_content_mix(content_base_mix);
        let rmvpe_base_mix = normalized_rmvpe_mix(rmvpe_base_mix);

        let mut offset = 0usize;
        while offset < raw.len() {
            let remaining_in_frame = SPEECH_MIX_FRAME_SAMPLES - self.frame_phase;
            let take = remaining_in_frame.min(raw.len() - offset);
            let raw_frame = &raw[offset..offset + take];
            let content_frame = &mut content[offset..offset + take];
            let pitch_frame = &mut pitch[offset..offset + take];
            let evidence =
                MixFrameEvidence::measure(raw_frame, pitch_frame, self.previous_raw_sample);
            let raw_rise = energy_rise_strength(evidence.raw_rms, self.previous_raw_rms);
            let denoised_rise =
                energy_rise_strength(evidence.denoised_rms, self.previous_denoised_rms);
            let residual_strength = smoothstep(evidence.residual_ratio, 0.08, 0.55);

            // A real onset normally rises in both aligned branches. If a
            // denoiser removes the onset, a large residual supplies the missing
            // evidence instead; a raw-only stationary level does neither after
            // its first frame.
            let aligned_rise = raw_rise * (0.45 + 0.55 * denoised_rise.max(residual_strength));
            let zcr_shape = smoothstep(evidence.zero_crossing_rate, 0.08, 0.28);
            let difference_shape = smoothstep(evidence.first_difference_ratio, 0.22, 0.90);
            let transient_shape = zcr_shape.max(difference_shape);
            let content_protection =
                (aligned_rise * residual_strength * (0.35 + 0.65 * transient_shape))
                    .clamp(0.0, 1.0);

            // RMVPE benefits from denoising on fricatives/noise. Retain only a
            // small raw share at a low-ZCR, low-difference voiced onset, where a
            // denoiser can otherwise erase the first pitch periods.
            let low_zcr = 1.0 - smoothstep(evidence.zero_crossing_rate, 0.06, 0.20);
            let periodic_shape = 1.0 - smoothstep(evidence.first_difference_ratio, 0.28, 1.10);
            let rmvpe_protection =
                (aligned_rise * residual_strength * low_zcr * (0.60 + 0.40 * periodic_shape))
                    .clamp(0.0, 1.0);

            let content_target =
                content_base_mix * (1.0 - SPEECH_MIX_CONTENT_MAX_REDUCTION * content_protection);
            let rmvpe_target =
                rmvpe_base_mix * (1.0 - SPEECH_MIX_RMVPE_MAX_REDUCTION * rmvpe_protection);

            for ((&raw_sample, content_sample), pitch_sample) in raw_frame
                .iter()
                .zip(content_frame.iter_mut())
                .zip(pitch_frame.iter_mut())
            {
                let denoised_sample = *pitch_sample;
                let content_mix =
                    smooth_mix_sample(&mut self.content_mix, content_base_mix, content_target);
                let rmvpe_mix =
                    smooth_mix_sample(&mut self.rmvpe_mix, rmvpe_base_mix, rmvpe_target);
                *content_sample = raw_sample * (1.0 - content_mix) + denoised_sample * content_mix;
                *pitch_sample = raw_sample * (1.0 - rmvpe_mix) + denoised_sample * rmvpe_mix;
            }

            self.previous_raw_rms = Some(evidence.raw_rms);
            self.previous_denoised_rms = Some(evidence.denoised_rms);
            self.previous_raw_sample = evidence.last_raw_sample;

            self.frame_phase = (self.frame_phase + take) % SPEECH_MIX_FRAME_SAMPLES;
            offset += take;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MixFrameEvidence {
    raw_rms: f32,
    denoised_rms: f32,
    residual_ratio: f32,
    zero_crossing_rate: f32,
    first_difference_ratio: f32,
    last_raw_sample: Option<f32>,
}

impl MixFrameEvidence {
    fn measure(raw: &[f32], denoised: &[f32], previous_raw_sample: Option<f32>) -> Self {
        debug_assert_eq!(raw.len(), denoised.len());
        if raw.is_empty() {
            return Self::default();
        }

        let mut raw_energy = 0.0f64;
        let mut denoised_energy = 0.0f64;
        let mut residual_energy = 0.0f64;
        let mut difference_energy = 0.0f64;
        let mut crossings = 0usize;
        let mut previous = previous_raw_sample.filter(|sample| sample.is_finite());

        for (&raw, &denoised) in raw.iter().zip(denoised) {
            let raw = finite_or_zero(raw);
            let denoised = finite_or_zero(denoised);
            raw_energy += f64::from(raw) * f64::from(raw);
            denoised_energy += f64::from(denoised) * f64::from(denoised);
            let residual = raw - denoised;
            residual_energy += f64::from(residual) * f64::from(residual);
            if let Some(previous) = previous {
                let difference = raw - previous;
                difference_energy += f64::from(difference) * f64::from(difference);
                if raw * previous < 0.0 {
                    crossings = crossings.saturating_add(1);
                }
            }
            previous = Some(raw);
        }

        let samples = raw.len() as f64;
        let raw_rms = (raw_energy / samples).sqrt() as f32;
        let denoised_rms = (denoised_energy / samples).sqrt() as f32;
        let residual_rms = (residual_energy / samples).sqrt() as f32;
        let difference_rms = (difference_energy / samples).sqrt() as f32;
        let raw_reference = raw_rms.max(SPEECH_MIX_SIGNAL_FLOOR);
        Self {
            raw_rms,
            denoised_rms,
            residual_ratio: (residual_rms / raw_reference).clamp(0.0, 2.0),
            zero_crossing_rate: (crossings as f32 / raw.len() as f32).clamp(0.0, 1.0),
            first_difference_ratio: (difference_rms / raw_reference).clamp(0.0, 2.0),
            last_raw_sample: previous,
        }
    }
}

fn energy_rise_strength(current_rms: f32, previous_rms: Option<f32>) -> f32 {
    let Some(previous_rms) = previous_rms.filter(|value| value.is_finite()) else {
        // Stream startup has no baseline. Treat it as neutral so an already-on
        // stationary fan does not momentarily force raw ContentVec input.
        return 0.0;
    };
    let current_rms = finite_or_zero(current_rms).max(0.0);
    let previous_rms = previous_rms.max(SPEECH_MIX_SIGNAL_FLOOR);
    let ratio_strength = smoothstep(current_rms / previous_rms, 1.30, 2.50);
    let audible_strength = smoothstep(current_rms, 0.000_3, 0.003);
    ratio_strength * audible_strength
}

fn smooth_mix_sample(current: &mut Option<f32>, base: f32, target: f32) -> f32 {
    let target = target.clamp(0.0, base);
    let value = current.get_or_insert(base);
    // The user's base is an upper bound, not merely a target. In particular,
    // base=0 must keep its documented exact-raw meaning after live automation;
    // adaptive recovery is only allowed to approach the base from below.
    *value = value.min(base);
    let alpha = if target < *value {
        SPEECH_MIX_FAST_ALPHA
    } else {
        SPEECH_MIX_SLOW_ALPHA
    };
    *value = (*value + alpha * (target - *value)).clamp(0.0, 1.0);
    *value
}

fn smoothstep(value: f32, low: f32, high: f32) -> f32 {
    if !value.is_finite() || value <= low {
        return 0.0;
    }
    if value >= high {
        return 1.0;
    }
    let t = (value - low) / (high - low);
    t * t * (3.0 - 2.0 * t)
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn normalized_content_mix(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn normalized_rmvpe_mix(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
fn blend_content_in_place(content: &mut [f32], denoised: &[f32], denoiser_content_mix: f32) {
    debug_assert_eq!(content.len(), denoised.len());
    let mix = normalized_content_mix(denoiser_content_mix);
    if mix <= f32::EPSILON {
        return;
    }
    if (mix - 1.0).abs() <= f32::EPSILON {
        content.copy_from_slice(denoised);
        return;
    }
    let raw_weight = 1.0 - mix;
    for (content, denoised) in content.iter_mut().zip(denoised) {
        *content = *content * raw_weight + *denoised * mix;
    }
}

#[cfg(test)]
fn blend_denoised_with_raw_in_place(denoised: &mut [f32], raw: &[f32], denoiser_rmvpe_mix: f32) {
    debug_assert_eq!(denoised.len(), raw.len());
    // Full denoising is the compatibility default. A malformed live value
    // therefore falls back to the existing RMVPE behavior rather than exposing
    // raw noise unexpectedly.
    let mix = normalized_rmvpe_mix(denoiser_rmvpe_mix);
    if mix <= f32::EPSILON {
        denoised.copy_from_slice(raw);
        return;
    }
    if (mix - 1.0).abs() <= f32::EPSILON {
        return;
    }
    let raw_weight = 1.0 - mix;
    for (denoised, raw) in denoised.iter_mut().zip(raw) {
        *denoised = *raw * raw_weight + *denoised * mix;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        blend_content_in_place, blend_denoised_with_raw_in_place, smooth_mix_sample,
        RvcStreamState, SampleDelay, SpeechPreservingMix, SPEECH_MIX_FRAME_SAMPLES,
    };

    fn alternating(amplitude: f32) -> Vec<f32> {
        (0..SPEECH_MIX_FRAME_SAMPLES)
            .map(|index| {
                if index.is_multiple_of(2) {
                    amplitude
                } else {
                    -amplitude
                }
            })
            .collect()
    }

    fn tone(amplitude: f32, frequency_hz: f32) -> Vec<f32> {
        (0..SPEECH_MIX_FRAME_SAMPLES)
            .map(|index| {
                let phase = index as f32 * std::f32::consts::TAU * frequency_hz / 16_000.0;
                amplitude * phase.sin()
            })
            .collect()
    }

    fn scaled(input: &[f32], scale: f32) -> Vec<f32> {
        input.iter().map(|sample| sample * scale).collect()
    }

    #[test]
    fn fixed_contentvec_context_is_retained_by_stream_state() {
        let mut state = RvcStreamState::new(48_000);
        assert_eq!(state.contentvec_context_samples_16k(), None);
        state.set_contentvec_context_samples_16k(9_920);
        assert_eq!(state.contentvec_context_samples_16k(), Some(9_920));
    }

    #[test]
    fn feature_frames_follow_emitted_resampler_samples_at_44100() {
        let mut state = RvcStreamState::new(48_000);
        let mut emitted_samples = 0usize;
        let mut emitted_frames = 0usize;
        let input = vec![0.0; 441]; // 10 ms at 44.1 kHz.

        for _ in 0..120 {
            let update = state
                .generate_input(&input, 44_100, 0, 0, 0)
                .expect("streaming resampler accepts 44.1 kHz input");
            emitted_samples += update.speech_features.samples;
            emitted_frames += update.new_feature_frames;
        }

        // Per-callback flooring would lose the fractional 44.1k/16k ratio;
        // the stateful remainder must account for exactly every 160 emitted
        // samples on the RMVPE grid.
        assert_eq!(emitted_frames, emitted_samples / SPEECH_MIX_FRAME_SAMPLES);
        assert!(emitted_samples > 16_000);
    }

    fn process_mix(
        state: &mut SpeechPreservingMix,
        raw: &[f32],
        denoised: &[f32],
        content_base: f32,
        rmvpe_base: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut content = raw.to_vec();
        let mut pitch = denoised.to_vec();
        state.process_in_place(raw, &mut content, &mut pitch, content_base, rmvpe_base);
        (content, pitch)
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1.0e-6, "maximum error was {max_error}");
    }

    #[test]
    fn sample_delay_preserves_order_across_chunks() {
        let mut delay = SampleDelay::default();
        delay.configure(2);
        let mut first = [1.0, 2.0, 3.0];
        delay.process_in_place(&mut first);
        assert_eq!(first, [0.0, 0.0, 1.0]);
        let mut second = [4.0, 5.0];
        delay.process_in_place(&mut second);
        assert_eq!(second, [2.0, 3.0]);
    }

    #[test]
    fn content_blend_zero_keeps_raw_signal() {
        let mut content = [1.0, -2.0, 0.5];
        blend_content_in_place(&mut content, &[4.0, 8.0, -3.0], 0.0);
        assert_eq!(content, [1.0, -2.0, 0.5]);
    }

    #[test]
    fn content_blend_one_replaces_with_denoised_signal() {
        let mut content = [1.0, -2.0, 0.5];
        blend_content_in_place(&mut content, &[4.0, 8.0, -3.0], 1.0);
        assert_eq!(content, [4.0, 8.0, -3.0]);
    }

    #[test]
    fn content_blend_is_linear_and_bounded() {
        let mut content = [0.0, 2.0, -4.0];
        blend_content_in_place(&mut content, &[4.0, 6.0, 0.0], 0.25);
        assert_eq!(content, [1.0, 3.0, -3.0]);

        let mut over = [1.0];
        blend_content_in_place(&mut over, &[3.0], 4.0);
        assert_eq!(over, [3.0]);
        let mut under = [1.0];
        blend_content_in_place(&mut under, &[3.0], -2.0);
        assert_eq!(under, [1.0]);
    }

    #[test]
    fn content_blend_non_finite_mix_is_a_raw_fallback() {
        let mut content = [1.0, -2.0];
        blend_content_in_place(&mut content, &[4.0, 8.0], f32::NAN);
        assert_eq!(content, [1.0, -2.0]);
    }

    #[test]
    fn rmvpe_blend_zero_uses_raw_signal() {
        let mut denoised = [4.0, 8.0, -3.0];
        blend_denoised_with_raw_in_place(&mut denoised, &[1.0, -2.0, 0.5], 0.0);
        assert_eq!(denoised, [1.0, -2.0, 0.5]);
    }

    #[test]
    fn rmvpe_blend_one_keeps_denoised_signal() {
        let mut denoised = [4.0, 8.0, -3.0];
        blend_denoised_with_raw_in_place(&mut denoised, &[1.0, -2.0, 0.5], 1.0);
        assert_eq!(denoised, [4.0, 8.0, -3.0]);
    }

    #[test]
    fn rmvpe_blend_interpolates_and_defaults_to_denoised_for_non_finite() {
        let mut denoised = [4.0, 8.0, -3.0];
        blend_denoised_with_raw_in_place(&mut denoised, &[0.0, 4.0, 1.0], 0.25);
        assert_eq!(denoised, [1.0, 5.0, 0.0]);

        let mut malformed = [4.0];
        blend_denoised_with_raw_in_place(&mut malformed, &[0.0], f32::NAN);
        assert_eq!(malformed, [4.0]);
    }

    #[test]
    fn stationary_noise_keeps_the_configured_base_mixes() {
        let raw = alternating(0.02);
        let denoised = scaled(&raw, 0.5);
        let expected_content: Vec<f32> = raw
            .iter()
            .zip(&denoised)
            .map(|(raw, denoised)| raw * 0.6 + denoised * 0.4)
            .collect();
        let expected_pitch: Vec<f32> = raw
            .iter()
            .zip(&denoised)
            .map(|(raw, denoised)| raw * 0.2 + denoised * 0.8)
            .collect();
        let mut state = SpeechPreservingMix::default();

        for _ in 0..4 {
            let (content, pitch) = process_mix(&mut state, &raw, &denoised, 0.4, 0.8);
            assert_close(&content, &expected_content);
            assert_close(&pitch, &expected_pitch);
        }
        assert_eq!(state.content_mix, Some(0.4));
        assert_eq!(state.rmvpe_mix, Some(0.8));
    }

    #[test]
    fn zero_base_is_exact_raw_after_live_mix_changes() {
        let raw = alternating(0.02);
        let denoised = scaled(&raw, 0.25);
        let mut state = SpeechPreservingMix::default();
        let _ = process_mix(&mut state, &raw, &denoised, 1.0, 1.0);

        let (content, pitch) = process_mix(&mut state, &raw, &denoised, 0.0, 0.0);

        assert_eq!(content, raw);
        assert_eq!(pitch, raw);
        assert_eq!(state.content_mix, Some(0.0));
        assert_eq!(state.rmvpe_mix, Some(0.0));
    }

    #[test]
    fn high_zcr_transient_protects_content_but_keeps_rmvpe_denoised() {
        let quiet = alternating(0.000_5);
        let quiet_denoised = scaled(&quiet, 0.9);
        let transient = alternating(0.12);
        let transient_denoised = scaled(&transient, 0.1);
        let mut state = SpeechPreservingMix::default();
        let _ = process_mix(&mut state, &quiet, &quiet_denoised, 1.0, 1.0);

        let (content, pitch) = process_mix(&mut state, &transient, &transient_denoised, 1.0, 1.0);

        assert!(state.content_mix.is_some_and(|mix| mix < 0.30));
        assert!(state.rmvpe_mix.is_some_and(|mix| mix > 0.99));
        let content_error = content
            .iter()
            .zip(&transient)
            .map(|(content, raw)| (content - raw).abs())
            .sum::<f32>();
        let denoised_error = transient_denoised
            .iter()
            .zip(&transient)
            .map(|(denoised, raw)| (denoised - raw).abs())
            .sum::<f32>();
        assert!(content_error < denoised_error * 0.45);
        assert_close(&pitch, &transient_denoised);
    }

    #[test]
    fn low_zcr_voiced_onset_only_slightly_reduces_rmvpe_mix() {
        let quiet = tone(0.000_5, 200.0);
        let quiet_denoised = scaled(&quiet, 0.9);
        let onset = tone(0.15, 200.0);
        let onset_denoised = scaled(&onset, 0.2);
        let mut state = SpeechPreservingMix::default();
        let _ = process_mix(&mut state, &quiet, &quiet_denoised, 1.0, 1.0);

        let _ = process_mix(&mut state, &onset, &onset_denoised, 1.0, 1.0);

        let content_mix = state.content_mix.expect("content mix initialized");
        let rmvpe_mix = state.rmvpe_mix.expect("RMVPE mix initialized");
        assert!(rmvpe_mix > 0.80 && rmvpe_mix < 0.95, "mix={rmvpe_mix}");
        assert!(rmvpe_mix > content_mix);
    }

    #[test]
    fn mix_envelope_protects_faster_than_it_recovers() {
        let mut protecting = Some(1.0);
        let protected = smooth_mix_sample(&mut protecting, 1.0, 0.0);
        let mut recovering = Some(0.0);
        let recovered = smooth_mix_sample(&mut recovering, 1.0, 1.0);

        assert!(1.0 - protected > recovered);
        assert!(protected > 0.0, "protection must not make a hard step");
        assert!(recovered < 1.0, "recovery must not make a hard step");
    }

    #[test]
    fn ten_ms_frame_partitioning_is_streaming_deterministic() {
        let quiet = alternating(0.000_5);
        let transient = alternating(0.12);
        let steady = alternating(0.12);
        let raw: Vec<f32> = quiet
            .iter()
            .chain(&transient)
            .chain(&steady)
            .copied()
            .collect();
        let denoised: Vec<f32> = scaled(&quiet, 0.9)
            .into_iter()
            .chain(scaled(&transient, 0.1))
            .chain(scaled(&steady, 0.1))
            .collect();

        let mut whole_state = SpeechPreservingMix::default();
        let (whole_content, whole_pitch) = process_mix(&mut whole_state, &raw, &denoised, 1.0, 1.0);

        let mut split_state = SpeechPreservingMix::default();
        let mut split_content = Vec::new();
        let mut split_pitch = Vec::new();
        for (raw_frame, denoised_frame) in raw
            .chunks(SPEECH_MIX_FRAME_SAMPLES)
            .zip(denoised.chunks(SPEECH_MIX_FRAME_SAMPLES))
        {
            let (content, pitch) =
                process_mix(&mut split_state, raw_frame, denoised_frame, 1.0, 1.0);
            split_content.extend(content);
            split_pitch.extend(pitch);
        }

        assert_eq!(split_content, whole_content);
        assert_eq!(split_pitch, whole_pitch);
        assert_eq!(split_state.content_mix, whole_state.content_mix);
        assert_eq!(split_state.rmvpe_mix, whole_state.rmvpe_mix);
    }
}

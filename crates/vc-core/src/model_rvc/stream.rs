use anyhow::{anyhow, Result};

use crate::dsp;

use super::shape::{
    feature_len_for_samples, keep_tail_in_place, samples_between_rates, tensor_rt_convert_size_16k,
    Rounding, EMBEDDER_SAMPLE_RATE, RMVPE_FRAME_SAMPLES_16K,
};

pub(super) const VOLUME_DECAY: f32 = 0.97;

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
    pub(super) prev_vol: f32,
    pub(super) prev_silence: bool,
    pub(super) sample_rate: u32,
    /// The RVC model's native output rate (from metadata `samplingRate`, default
    /// `RVC_SAMPLE_RATE`). Fixed per model — distinct from `sample_rate`, which is
    /// the device/input rate. Sizes `out_size` and the RVC-domain conversions.
    pub(super) rvc_sample_rate: u32,
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
}

impl RvcStreamState {
    pub(super) fn new(rvc_sample_rate: u32) -> Self {
        Self {
            audio_buffer: Vec::new(),
            audio_16k_buffer: Vec::new(),
            pitch_16k_buffer: Vec::new(),
            rmvpe_raw_16k_buffer: Vec::new(),
            pitchf_buffer: Vec::new(),
            prev_vol: 0.0,
            prev_silence: false,
            sample_rate: 0,
            rvc_sample_rate,
            resampler_16k: None,
            pitch_resampler_16k: None,
            content_delay_16k: SampleDelay::default(),
            #[cfg(feature = "gtcrn")]
            gtcrn: None,
        }
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
        if self.sample_rate != timing.sample_rate
            || self.resampler_16k.is_none()
            || input_path_changed
        {
            self.audio_buffer.clear();
            self.audio_16k_buffer.clear();
            self.pitch_16k_buffer.clear();
            self.rmvpe_raw_16k_buffer.clear();
            self.pitchf_buffer.clear();
            self.content_delay_16k.reset();
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

        let new_audio_16k_samples = samples_between_rates(
            content_audio.len(),
            timing.sample_rate,
            EMBEDDER_SAMPLE_RATE,
            Rounding::Floor,
        );
        let new_feature_len = feature_len_for_samples(new_audio_16k_samples, EMBEDDER_SAMPLE_RATE);
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
        // Blend only the new increment. A finite clamp keeps malformed live
        // automation from producing NaNs in the embedder input.
        blend_content_in_place(
            &mut self.audio_16k_buffer[new_16k_start..],
            &self.pitch_16k_buffer[new_pitch_16k_start..],
            timing.denoiser_content_mix,
        );
        // RMVPE can independently retain raw articulation. The raw branch was
        // aligned above, so this is a sample-for-sample mix with no phase smear.
        if separate_pitch_path {
            blend_denoised_with_raw_in_place(
                &mut self.pitch_16k_buffer[new_pitch_16k_start..],
                &self.rmvpe_raw_16k_buffer[new_rmvpe_raw_16k_start..],
                timing.denoiser_rmvpe_mix,
            );
        }
        // Silence detection follows the configured RMVPE branch, while
        // ContentVec receives its own residual-preserving blend above.
        let input_rms = dsp::rms(&self.pitch_16k_buffer[new_pitch_16k_start..]);
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
        let convert_size_16k = tensor_rt_convert_size_16k(
            content_audio.len(),
            timing.sample_rate,
            timing.crossfade_and_search_samples,
            timing.extra_convert_samples,
            self.rvc_sample_rate,
        );
        let convert_size = samples_between_rates(
            convert_size_16k,
            EMBEDDER_SAMPLE_RATE,
            timing.sample_rate,
            Rounding::Ceil,
        );
        let out_size = samples_between_rates(
            convert_size_16k.saturating_sub(extra_16k_samples),
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
        })
    }

    pub(super) fn update_pitchf_from_rmvpe_window(
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
        // RMVPE emits one center-padded frame past `(samples / hop)` for the
        // upstream bucket sizes. Copy from the front and let any trailing frame
        // fall off, matching the full-window WebUI assignment above while
        // preserving the absolute frame offset of a tail-only RMVPE window.
        self.pitchf_buffer[dst_start..dst_start + n].copy_from_slice(&f0[..n]);
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

fn blend_content_in_place(content: &mut [f32], denoised: &[f32], denoiser_content_mix: f32) {
    debug_assert_eq!(content.len(), denoised.len());
    let mix = if denoiser_content_mix.is_finite() {
        denoiser_content_mix.clamp(0.0, 1.0)
    } else {
        0.0
    };
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

fn blend_denoised_with_raw_in_place(denoised: &mut [f32], raw: &[f32], denoiser_rmvpe_mix: f32) {
    debug_assert_eq!(denoised.len(), raw.len());
    // Full denoising is the compatibility default. A malformed live value
    // therefore falls back to the existing RMVPE behavior rather than exposing
    // raw noise unexpectedly.
    let mix = if denoiser_rmvpe_mix.is_finite() {
        denoiser_rmvpe_mix.clamp(0.0, 1.0)
    } else {
        1.0
    };
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
    use super::{blend_content_in_place, blend_denoised_with_raw_in_place, SampleDelay};

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
}

//! Worker-owned adaptive source-activity detection for output silence suppression.
//!
//! The primary evidence is a small worker-owned Silero neural VAD when the
//! `silero-vad` feature is enabled. Its acoustic fallback learns the stationary
//! RMS floor of the branch RMVPE receives, then combines energy above that floor
//! with zero-crossing shape and pre-continuity F0 evidence. F0 is intentionally
//! never sufficient on its own: RMVPE can assign pitch to a fan, hum, or music
//! bleed. With stabilization enabled this mask has already passed the waveform
//! periodicity check. The detector only decides whether generated output is safe to mute; it
//! never skips model inference, so rolling RVC, RMVPE, and denoiser state stay
//! continuous while the microphone is idle.

#[cfg(feature = "silero-vad")]
use silero_vad_pure::{SampleRate, SileroVad};

/// The pitch branch is always on the 16 kHz RVC timeline. Ten milliseconds is a
/// useful, allocation-free analysis granularity for stationary-noise estimates.
const ENERGY_FRAME_SAMPLES: usize = 160;
const MIN_NOISE_FLOOR_RMS: f32 = 0.000_05;
const PITCH_EVIDENCE_VOICED_RATIO: f32 = 0.12;
const PITCH_ENERGY_SPEECH_RATIO: f32 = 1.25;
const OPENING_PITCH_ENERGY_SPEECH_RATIO: f32 = 1.70;
const STATIONARY_ENERGY_VARIATION: f32 = 0.08;
const STATIONARY_PITCH_ENERGY_SPEECH_RATIO: f32 = 2.80;
const ENERGY_SPEECH_RATIO: f32 = 1.55;
const STRONG_ENERGY_SPEECH_RATIO: f32 = 2.35;
const STATIONARY_STRONG_ENERGY_SPEECH_RATIO: f32 = 3.25;
const HIGH_ZERO_CROSSING_RATE: f32 = 0.34;
const MODULATED_ENERGY_VARIATION: f32 = 0.18;
const TRANSIENT_ENERGY_VARIATION: f32 = 0.45;
const HANGOVER_SAMPLES_16K: usize = 2_880; // 180 ms
#[cfg(feature = "silero-vad")]
const NEURAL_VAD_FRAME_SAMPLES_16K: usize = 512; // 32 ms
const NEURAL_VAD_MEAN_SPEECH_PROBABILITY: f32 = 0.60;
const NEURAL_VAD_PEAK_SPEECH_PROBABILITY: f32 = 0.78;
const NEURAL_VAD_REJECT_SPEECH_PROBABILITY: f32 = 0.18;
const NEURAL_VAD_MIN_RMS: f32 = 0.0001;
// The configured noise-gate threshold is an idle calibration hint, not a
// minimum microphone loudness. These paths intentionally use the learned
// floor so a quiet but real onset can reopen an output gate whose UI threshold
// is set for a noisier room.
const VOICE_ONSET_ENERGY_RATIO: f32 = 1.20;
const VOICE_SHAPED_ENERGY_RATIO: f32 = 1.15;
const STARTUP_SPEECH_FLOOR_FRACTION: f32 = 0.25;
// A chunk-level detector can miss a short syllable hidden inside a long
// increment.  Require a conservative ambient-shaped signal before allowing
// the output envelope to mute a whole increment; this is a safety valve, not
// an additional speech detector.
const SAFE_MUTE_ENERGY_RATIO: f32 = 1.80;
const SAFE_MUTE_PEAK_RATIO: f32 = 4.0;
const SAFE_MUTE_MAX_VARIATION: f32 = 0.30;

/// Summary of the neural VAD frames completed by one RVC input increment.
///
/// Keeping this scalar-only lets the acoustic decision and its tests stay
/// independent from the neural runtime. The VAD itself keeps its 32 ms partial
/// frame and recurrent state on the same conversion worker as this detector.
#[derive(Clone, Copy, Debug, Default)]
struct NeuralVadEvidence {
    mean_probability: f32,
    max_probability: f32,
    frames: usize,
}

impl NeuralVadEvidence {
    #[cfg(feature = "silero-vad")]
    fn record(&mut self, probability: f32) {
        let probability = finite_unit(probability);
        self.mean_probability = (self.mean_probability * self.frames as f32 + probability)
            / (self.frames.saturating_add(1) as f32);
        self.max_probability = self.max_probability.max(probability);
        self.frames = self.frames.saturating_add(1);
    }

    fn indicates_speech(self) -> bool {
        self.frames > 0
            && (self.mean_probability >= NEURAL_VAD_MEAN_SPEECH_PROBABILITY
                || self.max_probability >= NEURAL_VAD_PEAK_SPEECH_PROBABILITY)
    }

    fn rejects_weak_pitch(self) -> bool {
        // One 32 ms VAD frame can land on a word boundary. Wait for two quiet
        // frames before vetoing only the weak RMVPE-pitch branch; energy and
        // transient evidence still preserve unvoiced consonants and word starts.
        self.frames >= 2 && self.max_probability <= NEURAL_VAD_REJECT_SPEECH_PROBABILITY
    }
}

/// Stateful 16 kHz Silero adapter. It turns arbitrary RVC chunk lengths into
/// fixed 32 ms frames without allocating after construction.
#[cfg(feature = "silero-vad")]
struct NeuralVad {
    inner: SileroVad,
    frame: [f32; NEURAL_VAD_FRAME_SAMPLES_16K],
    frame_len: usize,
}

#[cfg(feature = "silero-vad")]
impl NeuralVad {
    fn new() -> Option<Self> {
        let inner = SileroVad::new(SampleRate::Hz16000).ok()?;
        (inner.chunk_size() == NEURAL_VAD_FRAME_SAMPLES_16K).then_some(Self {
            inner,
            frame: [0.0; NEURAL_VAD_FRAME_SAMPLES_16K],
            frame_len: 0,
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.frame.fill(0.0);
        self.frame_len = 0;
    }

    fn observe(&mut self, audio: &[f32]) -> NeuralVadEvidence {
        let mut evidence = NeuralVadEvidence::default();
        let mut remaining = audio;
        while !remaining.is_empty() {
            let take = (NEURAL_VAD_FRAME_SAMPLES_16K - self.frame_len).min(remaining.len());
            for (destination, source) in self.frame[self.frame_len..self.frame_len + take]
                .iter_mut()
                .zip(&remaining[..take])
            {
                *destination = finite_or_zero(*source);
            }
            self.frame_len += take;
            remaining = &remaining[take..];

            if self.frame_len == NEURAL_VAD_FRAME_SAMPLES_16K {
                match self.inner.process(&self.frame) {
                    Ok(probability) => evidence.record(probability),
                    // The embedded model only rejects a malformed frame, which
                    // this adapter cannot produce. Fall back to the acoustic VAD
                    // rather than propagating a worker-only model fault into the
                    // realtime output path.
                    Err(_) => self.reset(),
                }
                self.frame_len = 0;
            }
        }
        evidence
    }
}

/// Allocation-free measurements of the newest RMVPE input increment.
///
/// The stream state calculates these after the configured raw/denoised blend,
/// so the detector sees exactly the signal used for voiced/unvoiced estimation.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SpeechActivityFeatures {
    pub(super) rms: f32,
    pub(super) peak: f32,
    pub(super) zero_crossing_rate: f32,
    /// Standard deviation of 10 ms RMS values divided by their mean.
    pub(super) frame_energy_variation: f32,
    pub(super) samples: usize,
}

/// Streaming accumulator for the acoustic measurements used by the output
/// silence gate.
///
/// `SpeechActivityFeatures::from_audio` is intentionally kept as a stateless
/// helper for callers/tests that already own a complete window.  The realtime
/// stream, however, receives arbitrary device chunks.  Restarting the
/// previous-sample and 10 ms-frame state at every chunk makes ZCR and frame
/// energy variation depend on callback boundaries (44.1 kHz is especially
/// prone to this).  This scalar-only accumulator keeps the phase and one
/// partial frame on the conversion worker; it never allocates in `observe`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SpeechActivityFeatureAccumulator {
    previous_sample: f32,
    has_previous: bool,
    frame_sum_squares: f64,
    frame_len: usize,
    // Include the immediately preceding completed frame when the next frame
    // lands in a different callback.  This keeps a rise straddling a callback
    // visible without retaining a growing history or allocating a frame list.
    previous_frame_rms: Option<f32>,
}

impl SpeechActivityFeatureAccumulator {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    /// Measure one newly emitted 16 kHz increment while retaining frame phase
    /// across calls.  `samples` and RMS describe only this increment; frame
    /// variation is computed from complete 10 ms frames ending in it.  A
    /// trailing partial frame is deliberately retained for the next call so a
    /// split buffer cannot manufacture a second frame boundary.
    pub(super) fn observe(&mut self, audio: &[f32]) -> SpeechActivityFeatures {
        if audio.is_empty() {
            return SpeechActivityFeatures::default();
        }

        let mut sum_squares = 0.0f64;
        let mut peak = 0.0f32;
        let mut crossings = 0usize;
        let mut frame_rms_sum = 0.0f64;
        let mut frame_rms_sum_squares = 0.0f64;
        let mut frame_count = 0usize;

        if let Some(previous_frame_rms) = self.previous_frame_rms {
            push_frame_rms(
                f64::from(previous_frame_rms)
                    * f64::from(previous_frame_rms)
                    * ENERGY_FRAME_SAMPLES as f64,
                ENERGY_FRAME_SAMPLES,
                &mut frame_rms_sum,
                &mut frame_rms_sum_squares,
                &mut frame_count,
            );
        }

        for &sample in audio {
            let sample = finite_or_zero(sample);
            let sample_square = f64::from(sample) * f64::from(sample);
            sum_squares += sample_square;
            peak = peak.max(sample.abs());

            if self.has_previous && (sample >= 0.0) != (self.previous_sample >= 0.0) {
                crossings = crossings.saturating_add(1);
            }
            self.previous_sample = sample;
            self.has_previous = true;

            self.frame_sum_squares += sample_square;
            self.frame_len += 1;
            if self.frame_len == ENERGY_FRAME_SAMPLES {
                let frame_rms = (self.frame_sum_squares / self.frame_len as f64).sqrt() as f32;
                push_frame_rms(
                    self.frame_sum_squares,
                    self.frame_len,
                    &mut frame_rms_sum,
                    &mut frame_rms_sum_squares,
                    &mut frame_count,
                );
                self.previous_frame_rms = Some(frame_rms);
                self.frame_sum_squares = 0.0;
                self.frame_len = 0;
            }
        }

        let rms = (sum_squares / audio.len() as f64).sqrt() as f32;
        let frame_energy_variation = if frame_count > 1 {
            let count = frame_count as f64;
            let mean = frame_rms_sum / count;
            let variance = (frame_rms_sum_squares / count - mean * mean).max(0.0);
            (variance.sqrt() / mean.max(f64::from(MIN_NOISE_FLOOR_RMS))) as f32
        } else {
            0.0
        };

        SpeechActivityFeatures {
            rms,
            peak,
            zero_crossing_rate: crossings as f32 / audio.len() as f32,
            frame_energy_variation,
            samples: audio.len(),
        }
    }
}

impl SpeechActivityFeatures {
    #[cfg(test)]
    pub(super) fn from_audio(audio: &[f32]) -> Self {
        if audio.is_empty() {
            return Self::default();
        }

        let mut sum_squares = 0.0f64;
        let mut peak = 0.0f32;
        let mut crossings = 0usize;
        let mut previous = 0.0f32;
        let mut has_previous = false;
        let mut frame_sum = 0.0f64;
        let mut frame_len = 0usize;
        let mut frame_rms_sum = 0.0f64;
        let mut frame_rms_sum_squares = 0.0f64;
        let mut frame_count = 0usize;

        for &sample in audio {
            let sample = finite_or_zero(sample);
            let sample_square = f64::from(sample) * f64::from(sample);
            sum_squares += sample_square;
            frame_sum += sample_square;
            frame_len += 1;
            peak = peak.max(sample.abs());
            if has_previous && (sample >= 0.0) != (previous >= 0.0) {
                crossings = crossings.saturating_add(1);
            }
            previous = sample;
            has_previous = true;

            if frame_len == ENERGY_FRAME_SAMPLES {
                push_frame_rms(
                    frame_sum,
                    frame_len,
                    &mut frame_rms_sum,
                    &mut frame_rms_sum_squares,
                    &mut frame_count,
                );
                frame_sum = 0.0;
                frame_len = 0;
            }
        }
        if frame_len > 0 {
            push_frame_rms(
                frame_sum,
                frame_len,
                &mut frame_rms_sum,
                &mut frame_rms_sum_squares,
                &mut frame_count,
            );
        }

        let rms = (sum_squares / audio.len() as f64).sqrt() as f32;
        let frame_energy_variation = if frame_count > 1 {
            let count = frame_count as f64;
            let mean = frame_rms_sum / count;
            let variance = (frame_rms_sum_squares / count - mean * mean).max(0.0);
            (variance.sqrt() / mean.max(f64::from(MIN_NOISE_FLOOR_RMS))) as f32
        } else {
            0.0
        };

        Self {
            rms,
            peak,
            zero_crossing_rate: crossings as f32 / audio.len() as f32,
            frame_energy_variation,
            samples: audio.len(),
        }
    }
}

fn push_frame_rms(
    sum_squares: f64,
    samples: usize,
    sum: &mut f64,
    sum_squares_total: &mut f64,
    count: &mut usize,
) {
    let rms = (sum_squares / samples as f64).sqrt();
    *sum += rms;
    *sum_squares_total += rms * rms;
    *count = count.saturating_add(1);
}

/// Stateful speech-activity estimate used only by the output-side silence gate.
///
/// It lives in `RvcPipeline`, which is owned by a conversion worker. Do not move
/// this state to an audio callback: even though this implementation is small and
/// allocation-free, inference and all state ownership intentionally remain on
/// the worker side of the shared conversion pipeline.
pub(super) struct SpeechActivityDetector {
    noise_floor_rms: f32,
    initialized: bool,
    active: bool,
    hangover_samples: usize,
    #[cfg(feature = "silero-vad")]
    neural_vad: Option<NeuralVad>,
}

impl Default for SpeechActivityDetector {
    fn default() -> Self {
        Self {
            noise_floor_rms: MIN_NOISE_FLOOR_RMS,
            initialized: false,
            active: false,
            hangover_samples: 0,
            #[cfg(feature = "silero-vad")]
            neural_vad: NeuralVad::new(),
        }
    }
}

impl SpeechActivityDetector {
    pub(super) fn reset(&mut self) {
        self.noise_floor_rms = MIN_NOISE_FLOOR_RMS;
        self.initialized = false;
        self.active = false;
        self.hangover_samples = 0;
        #[cfg(feature = "silero-vad")]
        if let Some(neural_vad) = self.neural_vad.as_mut() {
            // Reset recurrent state in place. Rebuilding Silero here would
            // allocate model buffers on a live worker toggle or stream restart.
            neural_vad.reset();
        }
    }

    /// Observe one newest 16 kHz increment and return whether it contains or is
    /// adjacent to speech. `raw_voiced_ratio` is deliberately taken before F0
    /// continuity/post-processing (but after optional waveform validation): gap
    /// interpolation must not create false speech evidence in a quiet room.
    /// `audio_16k` is the same RMVPE branch used to calculate
    /// `features`; it is consumed only by the worker-owned neural VAD.
    pub(super) fn observe(
        &mut self,
        features: SpeechActivityFeatures,
        audio_16k: &[f32],
        raw_voiced_ratio: f32,
        configured_threshold: f32,
    ) -> bool {
        #[cfg(feature = "silero-vad")]
        let neural_evidence = self
            .neural_vad
            .as_mut()
            .map(|neural_vad| neural_vad.observe(audio_16k))
            .unwrap_or_default();
        #[cfg(not(feature = "silero-vad"))]
        let neural_evidence = {
            let _ = audio_16k;
            NeuralVadEvidence::default()
        };
        self.observe_with_neural_evidence(
            features,
            raw_voiced_ratio,
            configured_threshold,
            neural_evidence,
        )
    }

    /// Return whether a conversion increment has the conservative shape of
    /// ambient noise and may be muted as a whole.  The primary activity
    /// decision still comes from [`observe`]; this guard only prevents a long
    /// chunk containing a brief/quiet consonant from being discarded when the
    /// aggregate VAD happened to stay closed.
    pub(super) fn safe_to_mute(&self, features: SpeechActivityFeatures) -> bool {
        let rms = finite_non_negative(features.rms);
        let peak = finite_non_negative(features.peak);
        let variation = finite_non_negative(features.frame_energy_variation);
        let floor = self.noise_floor_rms.max(MIN_NOISE_FLOOR_RMS);
        let energy_ratio = rms / floor;
        let peak_ratio = peak / floor;
        energy_ratio <= SAFE_MUTE_ENERGY_RATIO
            && peak_ratio <= SAFE_MUTE_PEAK_RATIO
            && variation <= SAFE_MUTE_MAX_VARIATION
    }

    fn observe_with_neural_evidence(
        &mut self,
        features: SpeechActivityFeatures,
        raw_voiced_ratio: f32,
        configured_threshold: f32,
        neural_evidence: NeuralVadEvidence,
    ) -> bool {
        let rms = finite_non_negative(features.rms);
        let peak = finite_non_negative(features.peak);
        let zero_crossing_rate = finite_unit(features.zero_crossing_rate);
        let frame_energy_variation = finite_non_negative(features.frame_energy_variation);
        let raw_voiced_ratio = finite_unit(raw_voiced_ratio);
        let configured_threshold = finite_non_negative(configured_threshold);
        let starting = !self.initialized;
        let crest_factor = peak / rms.max(MIN_NOISE_FLOOR_RMS);
        let stationary = frame_energy_variation < STATIONARY_ENERGY_VARIATION;
        let startup_speech_shape = neural_evidence.indicates_speech()
            || (!stationary
                && (frame_energy_variation >= MODULATED_ENERGY_VARIATION
                    || (crest_factor >= 2.5 && frame_energy_variation >= 0.12)))
            || (raw_voiced_ratio >= PITCH_EVIDENCE_VOICED_RATIO && !stationary);

        if starting {
            // The first chunk is a calibration sample, not a reason to mute.
            // If the stream starts while the user is already speaking, learning
            // the full speech RMS as "noise" would poison the floor and mute the
            // next chunk. Seed a conservative quarter-floor for speech-shaped
            // input; stationary room noise still gets its true initial floor.
            self.noise_floor_rms = if startup_speech_shape {
                (rms * STARTUP_SPEECH_FLOOR_FRACTION).max(MIN_NOISE_FLOOR_RMS)
            } else {
                rms.max(MIN_NOISE_FLOOR_RMS)
            };
            self.initialized = true;
        }

        // Treat the user-selected silence threshold as a lower reference for
        // the conservative noise/energy paths. Speech-shaped onset and
        // high-confidence neural evidence use their own learned-floor paths
        // below, so a high idle threshold cannot mute a quiet human voice.
        let calibrated_floor = configured_threshold.max(MIN_NOISE_FLOOR_RMS);
        let noise_reference = self.noise_floor_rms.max(calibrated_floor);
        let energy_ratio = rms / noise_reference.max(MIN_NOISE_FLOOR_RMS);
        let opening_from_idle = !starting && !self.active && self.hangover_samples == 0;
        // An already-open gate gets a little more tolerance for a quiet voiced
        // tail. Opening from idle is deliberately stricter because that is where
        // a false RMVPE F0 can otherwise turn a steady room tone back into RVC
        // output for a whole chunk.
        let pitch_energy_ratio = if opening_from_idle {
            OPENING_PITCH_ENERGY_SPEECH_RATIO
        } else {
            PITCH_ENERGY_SPEECH_RATIO
        };
        let pitch_energy_ratio = if stationary {
            pitch_energy_ratio.max(STATIONARY_PITCH_ENERGY_SPEECH_RATIO)
        } else {
            pitch_energy_ratio
        };
        let pitch_evidence =
            raw_voiced_ratio >= PITCH_EVIDENCE_VOICED_RATIO && energy_ratio >= pitch_energy_ratio;
        // Silero is the only evidence with a learned speech/no-speech model.
        // Do not gate a high-confidence result by the user-selected absolute
        // RMS threshold: quiet speech can legitimately sit below that value.
        // Keep a tiny absolute floor so numerical silence cannot open the gate.
        let neural_speech_evidence =
            neural_evidence.indicates_speech() && rms >= NEURAL_VAD_MIN_RMS;
        let noise_like = zero_crossing_rate >= HIGH_ZERO_CROSSING_RATE
            && frame_energy_variation < MODULATED_ENERGY_VARIATION;
        // A low-ZCR hum resembles a long voiced vowel in both F0 and zero
        // crossings. Require modulation before ordinary energy can open the
        // gate; a genuinely loud stationary vowel still qualifies through the
        // stricter pitch/strong-energy paths above.
        let shaped_speech = !stationary
            && (zero_crossing_rate < HIGH_ZERO_CROSSING_RATE
                || frame_energy_variation >= MODULATED_ENERGY_VARIATION);
        let energy_evidence = energy_ratio >= ENERGY_SPEECH_RATIO && shaped_speech;
        // A real consonant/vowel onset can be quieter than the configured gate
        // threshold, but it still rises above the learned ambient floor and
        // has temporal shape. This path is deliberately unavailable to a
        // stationary fan/hum, which remains protected by the stricter paths.
        let learned_energy_ratio = rms / self.noise_floor_rms.max(MIN_NOISE_FLOOR_RMS);
        let shaped_pitch_evidence = raw_voiced_ratio >= PITCH_EVIDENCE_VOICED_RATIO
            && !stationary
            && learned_energy_ratio >= VOICE_SHAPED_ENERGY_RATIO;
        // A high-energy fricative can have a noise-like zero-crossing rate. Let
        // it open the gate only when it clearly exceeds the learned ambient
        // floor. Stationary fan/traffic noise needs a larger margin than a
        // modulated consonant so a level change cannot repeatedly reset silence.
        let strong_energy_ratio = if stationary {
            STATIONARY_STRONG_ENERGY_SPEECH_RATIO
        } else {
            STRONG_ENERGY_SPEECH_RATIO
        };
        let strong_energy_evidence = energy_ratio >= strong_energy_ratio;
        let transient_evidence = learned_energy_ratio >= VOICE_ONSET_ENERGY_RATIO
            && zero_crossing_rate < 0.45
            && (frame_energy_variation >= TRANSIENT_ENERGY_VARIATION
                || (crest_factor >= 3.0 && frame_energy_variation >= 0.25));
        let speech_evidence = neural_speech_evidence
            || (!neural_evidence.rejects_weak_pitch() && pitch_evidence)
            || (!neural_evidence.rejects_weak_pitch() && shaped_pitch_evidence)
            || energy_evidence
            || strong_energy_evidence
            || transient_evidence;

        // Learn quickly while below the current floor and slowly as ambient
        // noise rises. Uncorroborated pre-continuity F0 remains eligible for floor learning;
        // otherwise a fan that starts after speech could keep the output gate
        // permanently open merely because RMVPE assigned it a pitch.
        let learns_noise = (!speech_evidence && !shaped_pitch_evidence && !transient_evidence)
            || (!pitch_evidence && noise_like);
        if learns_noise {
            let alpha = if rms > self.noise_floor_rms {
                0.10
            } else {
                0.25
            };
            self.noise_floor_rms =
                ema(self.noise_floor_rms, rms, alpha).clamp(MIN_NOISE_FLOOR_RMS, 1.0);
        }

        if speech_evidence {
            self.active = true;
            self.hangover_samples = HANGOVER_SAMPLES_16K;
            return true;
        }

        if starting {
            self.active = true;
            self.hangover_samples = HANGOVER_SAMPLES_16K;
            return true;
        }

        let elapsed = features.samples.max(1);
        if self.hangover_samples > elapsed {
            self.hangover_samples -= elapsed;
            self.active = true;
        } else {
            self.hangover_samples = 0;
            self.active = false;
        }
        self.active
    }
}

const OUTPUT_GATE_OPEN_MS: f32 = 8.0;
const OUTPUT_GATE_CLOSE_MS: f32 = 50.0;

/// Cross-chunk gain envelope for the output-side silence suppressor.
///
/// It intentionally lives next to the source VAD in the shared pipeline rather
/// than a frontend. GUI/CLI, WAV conversion, and VST3 then all fade identical
/// model-domain output, while their audio callbacks remain queue-only.
pub(super) struct OutputSilenceEnvelope {
    gain: f32,
}

impl Default for OutputSilenceEnvelope {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl OutputSilenceEnvelope {
    pub(super) fn reset(&mut self) {
        self.gain = 1.0;
    }

    pub(super) fn is_silent(&self) -> bool {
        self.gain <= f32::EPSILON
    }

    /// Apply a click-free open/close envelope and return the resulting RMS.
    ///
    /// This is called on the conversion worker after RVC inference. Do not move
    /// it to an audio callback: the callback only consumes the already-shaped
    /// output ring, and this loop must remain coupled to the model timeline.
    pub(super) fn apply_and_rms(
        &mut self,
        audio: &mut [f32],
        audible: bool,
        sample_rate: u32,
    ) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        if !self.gain.is_finite() {
            self.gain = 1.0;
        }
        let target = if audible { 1.0 } else { 0.0 };
        let duration_ms = if audible {
            OUTPUT_GATE_OPEN_MS
        } else {
            OUTPUT_GATE_CLOSE_MS
        };
        let ramp_samples = ((sample_rate as f32 * duration_ms / 1_000.0).round() as usize).max(1);
        let step = 1.0 / ramp_samples as f32;
        let mut sum_squares = 0.0f64;

        for sample in audio.iter_mut() {
            if target > self.gain {
                self.gain = (self.gain + step).min(target);
            } else if target < self.gain {
                self.gain = (self.gain - step).max(target);
            }
            *sample *= self.gain;
            sum_squares += f64::from(*sample) * f64::from(*sample);
        }

        (sum_squares / audio.len() as f64).sqrt() as f32
    }
}

fn ema(previous: f32, current: f32, alpha: f32) -> f32 {
    previous + (current - previous) * alpha
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_non_negative(value: f32) -> f32 {
    finite_or_zero(value).max(0.0)
}

fn finite_unit(value: f32) -> f32 {
    finite_or_zero(value).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observe(detector: &mut SpeechActivityDetector, audio: &[f32], voiced_ratio: f32) -> bool {
        // These acoustic unit tests deliberately bypass the model runtime so
        // they exercise stable decision boundaries regardless of a third-party
        // neural VAD weight update. Integration uses `observe()` above.
        detector.observe_with_neural_evidence(
            SpeechActivityFeatures::from_audio(audio),
            voiced_ratio,
            0.01,
            NeuralVadEvidence::default(),
        )
    }

    fn alternating_noise(amplitude: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| {
                if index.is_multiple_of(2) {
                    amplitude
                } else {
                    -amplitude
                }
            })
            .collect()
    }

    fn tonal_noise(amplitude: f32, samples: usize) -> Vec<f32> {
        // 200 Hz completes two cycles per 10 ms analysis frame, so its frame
        // energy is intentionally stationary like a mains/fan hum.
        (0..samples)
            .map(|index| {
                let phase = index as f32 * std::f32::consts::TAU * 200.0 / 16_000.0;
                phase.sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn stationary_noise_becomes_inactive_after_adaptation() {
        let noise = alternating_noise(0.015, 1_600);
        let mut detector = SpeechActivityDetector::default();

        // Startup intentionally preserves one chunk so an initial syllable is
        // never muted while the floor is being seeded.
        assert!(observe(&mut detector, &noise, 0.0));
        // The detector's 180 ms hangover may span more than one short test
        // increment; once it expires, stationary noise remains inactive.
        assert!(observe(&mut detector, &noise, 0.0));
        assert!(!observe(&mut detector, &noise, 0.0));
        for _ in 0..8 {
            assert!(!observe(&mut detector, &noise, 0.0));
        }
    }

    #[test]
    fn voiced_speech_activates_even_after_noise_calibration() {
        let noise = alternating_noise(0.012, 1_600);
        let speech: Vec<f32> = (0..1_600)
            .map(|index| (index as f32 * 0.11).sin() * 0.05)
            .collect();
        let mut detector = SpeechActivityDetector::default();

        let _ = observe(&mut detector, &noise, 0.0);
        let _ = observe(&mut detector, &noise, 0.0);
        assert!(observe(&mut detector, &speech, 0.45));
    }

    #[test]
    fn speech_at_start_does_not_poison_the_noise_floor() {
        // A stream can be opened while the speaker is already talking. The
        // initial envelope makes this a non-stationary voice-shaped chunk, so
        // the detector must keep subsequent sustained chunks audible instead
        // of learning the speech RMS as the room floor.
        let speech: Vec<f32> = (0..1_600)
            .map(|index| {
                let attack = (index as f32 / 400.0).min(1.0);
                (index as f32 * 0.11).sin() * 0.05 * attack
            })
            .collect();
        let mut detector = SpeechActivityDetector::default();

        for _ in 0..6 {
            assert!(observe(&mut detector, &speech, 0.65));
        }
    }

    #[test]
    fn quiet_shaped_onset_reopens_below_absolute_gate_threshold() {
        let noise = alternating_noise(0.004, 1_600);
        let speech: Vec<f32> = (0..1_600)
            .map(|index| {
                let attack = (index as f32 / 500.0).min(1.0);
                (index as f32 * 0.11).sin() * 0.008 * attack
            })
            .collect();
        let mut detector = SpeechActivityDetector::default();

        for _ in 0..4 {
            let _ = observe(&mut detector, &noise, 0.0);
        }
        // RMS remains below the configured 0.01 threshold, but the learned
        // floor and temporal onset shape provide enough speech evidence.
        assert!(SpeechActivityFeatures::from_audio(&speech).rms < 0.01);
        assert!(observe(&mut detector, &speech, 0.35));
    }

    #[test]
    fn high_confidence_neural_speech_is_not_blocked_by_gate_threshold() {
        let quiet_noise = SpeechActivityFeatures {
            rms: 0.002,
            peak: 0.003,
            zero_crossing_rate: 0.10,
            frame_energy_variation: 0.04,
            samples: 1_600,
        };
        let mut detector = SpeechActivityDetector::default();
        // Startup grace plus the 180 ms hangover protect the first short
        // increments; wait until the detector has actually closed.
        for _ in 0..3 {
            let _ = detector.observe_with_neural_evidence(
                quiet_noise,
                0.0,
                0.01,
                NeuralVadEvidence::default(),
            );
        }
        assert!(!detector.observe_with_neural_evidence(
            quiet_noise,
            0.0,
            0.01,
            NeuralVadEvidence::default(),
        ));

        let neural_speech = NeuralVadEvidence {
            mean_probability: 0.88,
            max_probability: 0.94,
            frames: 2,
        };
        assert!(detector.observe_with_neural_evidence(quiet_noise, 0.0, 0.01, neural_speech,));
    }

    #[test]
    fn false_rmvpe_pitch_from_stationary_tone_does_not_reopen_the_gate() {
        let room_tone = tonal_noise(0.015, 1_600);
        let mut detector = SpeechActivityDetector::default();

        // Startup grace and the hangover may preserve the first two short
        // chunks, but false F0 alone must not keep refreshing that hangover.
        assert!(observe(&mut detector, &room_tone, 0.95));
        assert!(observe(&mut detector, &room_tone, 0.95));
        assert!(!observe(&mut detector, &room_tone, 0.95));
        for _ in 0..8 {
            assert!(!observe(&mut detector, &room_tone, 0.95));
        }
    }

    #[test]
    fn louder_stationary_tone_with_false_pitch_adapts_without_reopening() {
        let quiet_room_tone = tonal_noise(0.010, 1_600);
        let fan_turning_on = tonal_noise(0.030, 1_600);
        let mut detector = SpeechActivityDetector::default();

        let _ = observe(&mut detector, &quiet_room_tone, 0.0);
        let _ = observe(&mut detector, &quiet_room_tone, 0.0);
        assert!(!observe(&mut detector, &quiet_room_tone, 0.0));

        // The louder tone is below the stationary speech margin. Its false F0
        // must not freeze the learned floor or make the output flap back on.
        for _ in 0..8 {
            assert!(!observe(&mut detector, &fan_turning_on, 0.95));
        }
    }

    #[test]
    fn hangover_keeps_short_pause_active() {
        let speech: Vec<f32> = (0..160)
            .map(|index| (index as f32 * 0.11).sin() * 0.05)
            .collect();
        let silence = [0.0; 160];
        let mut detector = SpeechActivityDetector::default();

        assert!(observe(&mut detector, &speech, 0.50));
        for _ in 0..10 {
            assert!(observe(&mut detector, &silence, 0.0));
        }
    }

    #[test]
    fn sustained_silence_closes_after_hangover() {
        let speech: Vec<f32> = (0..160)
            .map(|index| (index as f32 * 0.11).sin() * 0.05)
            .collect();
        let silence = [0.0; 160];
        let mut detector = SpeechActivityDetector::default();

        assert!(observe(&mut detector, &speech, 0.50));
        for _ in 0..17 {
            assert!(observe(&mut detector, &silence, 0.0));
        }
        assert!(!observe(&mut detector, &silence, 0.0));
    }

    #[test]
    fn neural_rejection_blocks_false_pitch_without_blocking_energy_fallbacks() {
        let quiet_noise = SpeechActivityFeatures {
            rms: 0.01,
            peak: 0.015,
            zero_crossing_rate: 0.50,
            frame_energy_variation: 0.10,
            samples: 1_600,
        };
        let false_pitch_noise = SpeechActivityFeatures {
            rms: 0.018,
            peak: 0.025,
            zero_crossing_rate: 0.50,
            frame_energy_variation: 0.10,
            samples: 1_600,
        };
        let mut detector = SpeechActivityDetector::default();

        assert!(detector.observe_with_neural_evidence(
            quiet_noise,
            0.0,
            0.01,
            NeuralVadEvidence::default(),
        ));
        assert!(detector.observe_with_neural_evidence(
            quiet_noise,
            0.0,
            0.01,
            NeuralVadEvidence::default(),
        ));
        assert!(!detector.observe_with_neural_evidence(
            quiet_noise,
            0.0,
            0.01,
            NeuralVadEvidence::default(),
        ));

        let neural_rejection = NeuralVadEvidence {
            mean_probability: 0.05,
            max_probability: 0.10,
            frames: 2,
        };
        // This chunk has enough energy/F0 for the weak pitch path, but its
        // high-ZCR shape is not energy speech. Two neural no-speech frames must
        // keep it muted rather than let RMVPE reopen the output.
        assert!(!detector.observe_with_neural_evidence(
            false_pitch_noise,
            0.60,
            0.01,
            neural_rejection,
        ));

        let neural_speech = NeuralVadEvidence {
            mean_probability: 0.80,
            max_probability: 0.85,
            frames: 2,
        };
        assert!(detector.observe_with_neural_evidence(false_pitch_noise, 0.0, 0.01, neural_speech,));
    }

    #[test]
    fn safe_mute_guard_rejects_a_chunk_with_a_transient_peak() {
        let noise = alternating_noise(0.01, 1_600);
        let mut detector = SpeechActivityDetector::default();
        let _ = observe(&mut detector, &noise, 0.0);
        let _ = observe(&mut detector, &noise, 0.0);
        let _ = observe(&mut detector, &noise, 0.0);

        let mut mixed = noise.clone();
        mixed[800..820].fill(0.08);
        let features = SpeechActivityFeatures::from_audio(&mixed);
        assert!(!detector.safe_to_mute(features));
    }

    #[test]
    fn streaming_features_preserve_zero_crossing_phase() {
        let alternating = alternating_noise(0.02, 160);
        let mut accumulator = SpeechActivityFeatureAccumulator::default();
        let first = accumulator.observe(&alternating[..80]);
        let second = accumulator.observe(&alternating[80..]);

        // The first sample of the second increment is compared with the last
        // sample of the first one. A stateless per-chunk measurement would
        // miss that crossing and report the lower 79/80 ratio again.
        assert!(second.zero_crossing_rate > first.zero_crossing_rate);
        assert_eq!(second.samples, 80);
    }

    #[test]
    fn streaming_features_keep_energy_frame_phase_across_callbacks() {
        let mut audio = vec![0.01; 160];
        audio.extend(std::iter::repeat_n(0.10, 160));
        let expected = SpeechActivityFeatures::from_audio(&audio);
        assert!(expected.frame_energy_variation > 0.5);

        let mut accumulator = SpeechActivityFeatureAccumulator::default();
        let mut newest = SpeechActivityFeatures::default();
        for chunk in audio.chunks(80) {
            newest = accumulator.observe(chunk);
        }

        // The second completed frame is measured together with the previous
        // frame even though the callback split occurs halfway through each.
        assert!(newest.frame_energy_variation > 0.5);
        assert!(newest.rms > 0.09);
    }

    #[test]
    fn output_silence_envelope_fades_across_chunk_boundaries() {
        let mut envelope = OutputSilenceEnvelope::default();
        let mut closing = vec![1.0; 4_800]; // 100 ms at 48 kHz.
        let closing_rms = envelope.apply_and_rms(&mut closing, false, 48_000);

        assert!(closing[0] > 0.0 && closing[0] < 1.0);
        assert_eq!(closing.last().copied(), Some(0.0));
        assert!(closing_rms > 0.0);
        assert!(envelope.is_silent());

        let mut opening = vec![1.0; 480]; // 10 ms at 48 kHz.
        let opening_rms = envelope.apply_and_rms(&mut opening, true, 48_000);
        assert!(opening[0] > 0.0);
        assert!(*opening.last().unwrap() > opening[0]);
        assert!(opening_rms > 0.0);
        assert!(!envelope.is_silent());
    }

    #[cfg(feature = "silero-vad")]
    #[test]
    fn neural_vad_preserves_partial_frames_and_reports_finite_probabilities() {
        let mut vad = NeuralVad::new().expect("embedded Silero VAD loads");
        let first = vad.observe(&[0.0; 256]);
        assert_eq!(first.frames, 0);

        let second = vad.observe(&[0.0; 256]);
        assert_eq!(second.frames, 1);
        assert!((0.0..=1.0).contains(&second.mean_probability));
        assert!((0.0..=1.0).contains(&second.max_probability));
    }
}

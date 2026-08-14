//! Short, bounded input-voice calibration for front ends.
//!
//! The accumulator intentionally records only level and F0-availability
//! statistics, never audio samples. A realtime host can feed it from its worker
//! thread and discard the raw microphone data immediately after each chunk.

use crate::model_rvc::{
    DEFAULT_DENOISER_CONTENT_MIX, DEFAULT_DENOISER_RMVPE_MIX, DEFAULT_F0_THRESHOLD, DEFAULT_PROTECT,
};

/// Default length used by the standalone microphone calibration flow.
///
/// Twelve seconds leaves room for normal speech, a short pause, and natural
/// level changes. Eight seconds was enough to set gain, but made the noise and
/// F0-confidence percentiles unnecessarily sensitive to a single utterance.
pub const DEFAULT_VOICE_CALIBRATION_DURATION_MS: u32 = 12_000;
/// Keep a calibration long enough to contain both speech and a short pause.
pub const MIN_VOICE_CALIBRATION_DURATION_MS: u32 = 2_000;
/// This is a setup action, not a recorder. Bound its duration and histogram
/// accumulation so a malformed frontend request cannot retain worker state.
pub const MAX_VOICE_CALIBRATION_DURATION_MS: u32 = 20_000;

const FRAME_MS: u32 = 20;
const DB_FLOOR: f32 = -96.0;
const HISTOGRAM_BINS: usize = 96;
const TARGET_SPEECH_RMS: f32 = 0.06;
const BASELINE_CHUNK_MS: u32 = 450;
const BASELINE_EXTRA_CONVERT_MS: u32 = 2_120;

// These remain inside the common conversion-timing bounds. They deliberately
// move only within a small high-quality range: calibration is meant to tune an
// established RVC session, not silently turn it into a high-latency offline
// conversion profile.
const NOISY_CHUNK_MS: u32 = 500;
const NOISY_EXTRA_CONVERT_MS: u32 = 2_300;
const UNSTABLE_F0_CHUNK_MS: u32 = 550;
const UNSTABLE_F0_EXTRA_CONVERT_MS: u32 = 2_400;
const DEGRADED_CHUNK_MS: u32 = 600;
const DEGRADED_EXTRA_CONVERT_MS: u32 = 2_500;

/// A privacy-preserving summary of the microphone signal captured during a
/// calibration pass. Levels are un-gained device input values, so callers can
/// safely derive an absolute input-gain recommendation from them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceCalibrationProfile {
    pub captured_ms: u32,
    pub frame_count: u32,
    pub speech_frame_ratio: f32,
    pub noise_floor_rms: f32,
    pub speech_rms: f32,
    pub signal_to_noise_db: f32,
    pub peak: f32,
    pub clipped_sample_ratio: f32,
    /// RMVPE voiced-frame ratio reported by the shared RVC pipeline. It is zero
    /// for a passthrough-only calibration where no F0 model is loaded.
    pub f0_voiced_ratio: f32,
    pub f0_frame_count: u32,
}

impl VoiceCalibrationProfile {
    /// Derive conservative live conversion settings from the measured input.
    ///
    /// These recommendations deliberately do not choose `pitch_shift`: an RVC
    /// ONNX export does not describe the target voice's natural F0 range, so
    /// guessing a gendered/octave shift from the source voice would make the
    /// conversion less reliable. The target-model choice remains explicit.
    pub fn recommendation(self, has_feature_index: bool) -> VoiceCalibrationRecommendation {
        let speech_rms = finite_positive(self.speech_rms);
        let peak = finite_positive(self.peak);
        let desired_gain = if speech_rms > 0.0 {
            TARGET_SPEECH_RMS / speech_rms
        } else {
            1.0
        };
        // Input gain runs before every denoiser branch. Reserve headroom for
        // speech transients instead of chasing the RMS target into clipping.
        let clip_limited_gain = if peak > 0.0 { 0.92 / peak } else { 4.0 };
        let input_gain = desired_gain.min(clip_limited_gain).clamp(0.10, 4.0);

        let noise_floor = finite_positive(self.noise_floor_rms) * input_gain;
        let speech_after_gain = speech_rms * input_gain;
        // The gate detects instantaneous envelope amplitude, not frame RMS.
        // A 2x noise-floor threshold has enough margin for steady ambience while
        // the speech-relative cap avoids chewing quiet consonants.
        let speech_cap = (speech_after_gain * 0.35).max(0.001);
        let gate_threshold = (noise_floor * 2.0).clamp(0.001, speech_cap.min(0.12));
        let noisy = (self.signal_to_noise_db.is_finite() && self.signal_to_noise_db < 18.0)
            || (self.noise_floor_rms.is_finite() && self.noise_floor_rms > 0.004);
        let very_noisy = self.signal_to_noise_db.is_finite() && self.signal_to_noise_db < 12.0;
        let clipped = self.clipped_sample_ratio.is_finite() && self.clipped_sample_ratio >= 0.002;
        // Only interpret a low voiced ratio when both a pitch model and a
        // meaningful amount of speech were observed. A zero frame count means
        // the current session has no F0 model, not that the microphone failed.
        let f0_is_weak = self.f0_frame_count >= 20
            && self.speech_frame_ratio >= 0.25
            && self.f0_voiced_ratio.is_finite()
            && self.f0_voiced_ratio < 0.20;
        let degraded = very_noisy || clipped;
        let (chunk_ms, extra_convert_ms, protect_transition_ms) = if degraded {
            (DEGRADED_CHUNK_MS, DEGRADED_EXTRA_CONVERT_MS, 40)
        } else if f0_is_weak {
            (UNSTABLE_F0_CHUNK_MS, UNSTABLE_F0_EXTRA_CONVERT_MS, 35)
        } else if noisy {
            (NOISY_CHUNK_MS, NOISY_EXTRA_CONVERT_MS, 25)
        } else {
            (BASELINE_CHUNK_MS, BASELINE_EXTRA_CONVERT_MS, 15)
        };

        VoiceCalibrationRecommendation {
            input_gain,
            gate_threshold,
            prefer_noise_gate: noisy && self.speech_frame_ratio >= 0.10,
            // Keep a raw ContentVec residual for articulation. In noisy rooms a
            // slightly larger cleaned share reduces environmental consonant-like
            // artifacts without returning to the old all-denoised path.
            denoiser_content_mix: if degraded {
                0.45
            } else if noisy {
                0.35
            } else {
                DEFAULT_DENOISER_CONTENT_MIX
            },
            // A clean recording can keep a small raw RMVPE share for subtle
            // vibrato and high-register detail. As noise or clipping rises,
            // move progressively toward the fully cleaned pitch branch.
            denoiser_rmvpe_mix: if degraded {
                DEFAULT_DENOISER_RMVPE_MIX
            } else if noisy {
                0.90
            } else {
                0.80
            },
            // Retrieval is most useful on clean voiced content. Back it off when
            // the measured SNR is weak so environmental features are not pulled
            // toward the target index.
            index_rate: if has_feature_index {
                if degraded {
                    0.45
                } else if noisy {
                    0.60
                } else if f0_is_weak {
                    0.65
                } else {
                    0.75
                }
            } else {
                0.0
            },
            // Lower values retain more original ContentVec on unvoiced frames.
            // This protects fricatives in noisy input while preserving the
            // standard RVC setting for a clean microphone.
            protect: if degraded {
                0.22
            } else if noisy || f0_is_weak {
                0.28
            } else {
                DEFAULT_PROTECT
            },
            // Weak but otherwise clean voices benefit from retaining RMVPE's
            // low-confidence voiced frames. Do not lower the threshold in a
            // noisy profile, where those frames are more likely false positives.
            f0_threshold: if degraded {
                0.05
            } else if noisy {
                0.04
            } else if f0_is_weak {
                0.02
            } else {
                DEFAULT_F0_THRESHOLD
            },
            // `protect_transition_ms`, `chunk_ms`, and `extra_convert_ms` are
            // load-time controls. They are selected together here so a frontend
            // cannot accidentally combine a long protect ramp with a short
            // context profile after a calibration restart.
            protect_transition_ms,
            chunk_ms,
            extra_convert_ms,
        }
    }
}

/// Settings derived from [`VoiceCalibrationProfile`] that are safe to apply to
/// the existing shared RVC configuration surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceCalibrationRecommendation {
    pub input_gain: f32,
    pub gate_threshold: f32,
    pub prefer_noise_gate: bool,
    pub denoiser_content_mix: f32,
    pub denoiser_rmvpe_mix: f32,
    pub index_rate: f32,
    pub protect: f32,
    pub f0_threshold: f32,
    pub protect_transition_ms: u32,
    pub chunk_ms: u32,
    pub extra_convert_ms: u32,
}

/// Fixed-memory level/F0 accumulator for one microphone calibration pass.
///
/// Feed `observe_audio()` with the raw worker-side input before input gain or
/// denoising. Call `observe_f0()` after the shared conversion pipeline returns
/// its chunk stats. This split keeps the source-level profile independent of
/// whichever denoiser was selected before calibration.
#[derive(Debug)]
pub struct VoiceCalibrationAccumulator {
    sample_rate: u32,
    target_samples: u64,
    observed_samples: u64,
    frame_samples: usize,
    frame_len: usize,
    frame_sum_squares: f64,
    histogram: [u32; HISTOGRAM_BINS],
    peak: f32,
    clipped_samples: u64,
    f0_weighted_voiced_frames: f64,
    f0_frames: u64,
    finalized: bool,
}

impl VoiceCalibrationAccumulator {
    pub fn new(sample_rate: u32, duration_ms: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        let duration_ms = duration_ms.clamp(
            MIN_VOICE_CALIBRATION_DURATION_MS,
            MAX_VOICE_CALIBRATION_DURATION_MS,
        );
        let target_samples = u64::from(sample_rate) * u64::from(duration_ms) / 1_000;
        Self {
            sample_rate,
            target_samples: target_samples.max(1),
            observed_samples: 0,
            frame_samples: 0,
            frame_len: ((u64::from(sample_rate) * u64::from(FRAME_MS)) / 1_000).max(1) as usize,
            frame_sum_squares: 0.0,
            histogram: [0; HISTOGRAM_BINS],
            peak: 0.0,
            clipped_samples: 0,
            f0_weighted_voiced_frames: 0.0,
            f0_frames: 0,
            finalized: false,
        }
    }

    /// Consume up to the remaining calibration duration and return that sample
    /// count. The caller uses the count to proportionally weight F0 stats from
    /// the associated inference chunk when the final chunk is only partly used.
    pub fn observe_audio(&mut self, samples: &[f32]) -> usize {
        if self.finalized || self.observed_samples >= self.target_samples {
            return 0;
        }
        let remaining = (self.target_samples - self.observed_samples) as usize;
        let count = samples.len().min(remaining);
        for &sample in &samples[..count] {
            let sample = if sample.is_finite() { sample } else { 0.0 };
            let abs = sample.abs();
            self.peak = self.peak.max(abs);
            if abs >= 0.98 {
                self.clipped_samples += 1;
            }
            self.frame_sum_squares += f64::from(sample) * f64::from(sample);
            self.frame_samples += 1;
            if self.frame_samples == self.frame_len {
                self.commit_frame();
            }
        }
        self.observed_samples += count as u64;
        if self.observed_samples == self.target_samples && self.frame_samples > 0 {
            self.commit_frame();
        }
        count
    }

    /// Incorporate the RVC-aligned F0 statistic for a chunk whose raw audio was
    /// partially or fully observed. `observed_fraction` is normally 1.0, and is
    /// smaller only for the final capture chunk.
    pub fn observe_f0(&mut self, voiced_ratio: f32, pitch_frames: usize, observed_fraction: f32) {
        if self.finalized
            || pitch_frames == 0
            || !matches!(
                observed_fraction.partial_cmp(&0.0),
                Some(std::cmp::Ordering::Greater)
            )
        {
            return;
        }
        let weighted_frames = ((pitch_frames as f32 * observed_fraction.clamp(0.0, 1.0)).round()
            as u64)
            .min(pitch_frames as u64);
        self.f0_frames += weighted_frames;
        self.f0_weighted_voiced_frames +=
            f64::from(voiced_ratio.clamp(0.0, 1.0)) * weighted_frames as f64;
    }

    pub fn is_complete(&self) -> bool {
        self.observed_samples >= self.target_samples
    }

    pub fn captured_ms(&self) -> u32 {
        ((self.observed_samples * 1_000) / u64::from(self.sample_rate)).min(u64::from(u32::MAX))
            as u32
    }

    pub fn target_ms(&self) -> u32 {
        ((self.target_samples * 1_000) / u64::from(self.sample_rate)).min(u64::from(u32::MAX))
            as u32
    }

    /// Finalize the profile. The method is idempotent so a host can safely
    /// publish the result once a chunk reaches the capture boundary.
    pub fn finish(&mut self) -> VoiceCalibrationProfile {
        self.finalized = true;
        let frame_count = self
            .histogram
            .iter()
            .map(|&count| u64::from(count))
            .sum::<u64>();
        if frame_count == 0 {
            return VoiceCalibrationProfile {
                captured_ms: self.captured_ms(),
                peak: self.peak,
                clipped_sample_ratio: self.clipped_ratio(),
                ..VoiceCalibrationProfile::default()
            };
        }

        let noise_db = histogram_percentile_db(&self.histogram, 0.20);
        let speech_db = histogram_percentile_db(&self.histogram, 0.85).max(noise_db);
        let speech_threshold_db = noise_db + (speech_db - noise_db).max(6.0) * 0.5;
        let speech_frames = self
            .histogram
            .iter()
            .enumerate()
            .filter_map(|(bin, &count)| {
                (histogram_bin_db(bin) >= speech_threshold_db).then_some(u64::from(count))
            })
            .sum::<u64>();

        VoiceCalibrationProfile {
            captured_ms: self.captured_ms(),
            frame_count: frame_count.min(u64::from(u32::MAX)) as u32,
            speech_frame_ratio: (speech_frames as f32 / frame_count as f32).clamp(0.0, 1.0),
            noise_floor_rms: db_to_rms(noise_db),
            speech_rms: db_to_rms(speech_db),
            signal_to_noise_db: (speech_db - noise_db).max(0.0),
            peak: self.peak,
            clipped_sample_ratio: self.clipped_ratio(),
            f0_voiced_ratio: if self.f0_frames == 0 {
                0.0
            } else {
                (self.f0_weighted_voiced_frames / self.f0_frames as f64) as f32
            }
            .clamp(0.0, 1.0),
            f0_frame_count: self.f0_frames.min(u64::from(u32::MAX)) as u32,
        }
    }

    fn commit_frame(&mut self) {
        if self.frame_samples == 0 {
            return;
        }
        let rms = (self.frame_sum_squares / self.frame_samples as f64).sqrt() as f32;
        self.histogram[rms_histogram_bin(rms)] += 1;
        self.frame_samples = 0;
        self.frame_sum_squares = 0.0;
    }

    fn clipped_ratio(&self) -> f32 {
        if self.observed_samples == 0 {
            0.0
        } else {
            (self.clipped_samples as f64 / self.observed_samples as f64) as f32
        }
    }
}

fn finite_positive(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn rms_histogram_bin(rms: f32) -> usize {
    let db = 20.0 * finite_positive(rms).max(1e-6).log10();
    ((db.floor() as i32 - DB_FLOOR as i32).clamp(0, HISTOGRAM_BINS as i32 - 1)) as usize
}

fn histogram_bin_db(bin: usize) -> f32 {
    DB_FLOOR + bin.min(HISTOGRAM_BINS - 1) as f32 + 0.5
}

fn histogram_percentile_db(histogram: &[u32; HISTOGRAM_BINS], percentile: f32) -> f32 {
    let total = histogram.iter().map(|&count| u64::from(count)).sum::<u64>();
    if total == 0 {
        return DB_FLOOR;
    }
    let target = ((total as f32 * percentile.clamp(0.0, 1.0)).ceil() as u64).max(1);
    let mut cumulative = 0u64;
    for (bin, &count) in histogram.iter().enumerate() {
        cumulative += u64::from(count);
        if cumulative >= target {
            return histogram_bin_db(bin);
        }
    }
    histogram_bin_db(HISTOGRAM_BINS - 1)
}

fn db_to_rms(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_separates_speech_from_room_noise() {
        let mut accumulator = VoiceCalibrationAccumulator::new(1_000, 2_000);
        let mut audio = Vec::new();
        audio.extend(std::iter::repeat_n(0.005, 500));
        audio.extend(std::iter::repeat_n(0.08, 1_500));

        assert_eq!(accumulator.observe_audio(&audio), audio.len());
        accumulator.observe_f0(0.7, 200, 1.0);
        assert!(accumulator.is_complete());
        let profile = accumulator.finish();

        assert_eq!(profile.captured_ms, 2_000);
        assert!(profile.noise_floor_rms < 0.01);
        assert!(profile.speech_rms > 0.05);
        assert!(profile.signal_to_noise_db > 15.0);
        assert!(profile.speech_frame_ratio > 0.5);
        assert!((profile.f0_voiced_ratio - 0.7).abs() < 0.01);
    }

    #[test]
    fn accumulator_stops_at_requested_duration() {
        let mut accumulator = VoiceCalibrationAccumulator::new(1_000, 2_000);
        let audio = vec![0.1; 3_000];

        assert_eq!(accumulator.observe_audio(&audio), 2_000);
        assert_eq!(accumulator.observe_audio(&audio), 0);
        assert_eq!(accumulator.captured_ms(), 2_000);
    }

    #[test]
    fn recommendation_preserves_headroom_and_backs_off_noisy_retrieval() {
        let profile = VoiceCalibrationProfile {
            speech_frame_ratio: 0.7,
            noise_floor_rms: 0.02,
            speech_rms: 0.08,
            signal_to_noise_db: 11.0,
            peak: 0.99,
            ..VoiceCalibrationProfile::default()
        };

        let recommendation = profile.recommendation(true);

        assert!(recommendation.input_gain <= 0.93);
        assert!(recommendation.prefer_noise_gate);
        assert_eq!(recommendation.index_rate, 0.45);
        assert_eq!(recommendation.protect, 0.22);
        assert_eq!(recommendation.denoiser_rmvpe_mix, 1.0);
        assert_eq!(recommendation.f0_threshold, 0.05);
        assert_eq!(recommendation.protect_transition_ms, 40);
        assert_eq!(recommendation.chunk_ms, 600);
        assert_eq!(recommendation.extra_convert_ms, 2_500);
    }

    #[test]
    fn recommendation_adds_context_for_a_clean_but_weak_f0_signal() {
        let profile = VoiceCalibrationProfile {
            speech_frame_ratio: 0.7,
            noise_floor_rms: 0.001,
            speech_rms: 0.06,
            signal_to_noise_db: 32.0,
            peak: 0.4,
            f0_voiced_ratio: 0.10,
            f0_frame_count: 100,
            ..VoiceCalibrationProfile::default()
        };

        let recommendation = profile.recommendation(true);

        assert_eq!(recommendation.index_rate, 0.65);
        assert_eq!(recommendation.protect, 0.28);
        assert_eq!(recommendation.f0_threshold, 0.02);
        assert_eq!(recommendation.protect_transition_ms, 35);
        assert_eq!(recommendation.chunk_ms, 550);
        assert_eq!(recommendation.extra_convert_ms, 2_400);
    }

    #[test]
    fn recommendation_uses_the_high_quality_baseline_for_a_clean_stable_voice() {
        let profile = VoiceCalibrationProfile {
            speech_frame_ratio: 0.7,
            noise_floor_rms: 0.001,
            speech_rms: 0.06,
            signal_to_noise_db: 32.0,
            peak: 0.4,
            f0_voiced_ratio: 0.75,
            f0_frame_count: 100,
            ..VoiceCalibrationProfile::default()
        };

        let recommendation = profile.recommendation(true);

        assert_eq!(recommendation.denoiser_content_mix, 0.25);
        assert_eq!(recommendation.denoiser_rmvpe_mix, 0.80);
        assert_eq!(recommendation.protect_transition_ms, 15);
        assert_eq!(recommendation.chunk_ms, 450);
        assert_eq!(recommendation.extra_convert_ms, 2_120);
    }

    #[test]
    fn default_calibration_window_is_long_enough_for_multiple_utterances() {
        assert_eq!(DEFAULT_VOICE_CALIBRATION_DURATION_MS, 12_000);
    }

    #[test]
    fn recommendation_disables_index_without_an_index_file() {
        let profile = VoiceCalibrationProfile {
            speech_rms: 0.06,
            peak: 0.2,
            signal_to_noise_db: 35.0,
            ..VoiceCalibrationProfile::default()
        };

        assert_eq!(profile.recommendation(false).index_rate, 0.0);
    }
}

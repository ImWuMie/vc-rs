//! Conservative online tuning for the shared realtime conversion pipeline.
//!
//! This module intentionally works from lightweight acoustic observations, not
//! from a claimed ASR result. The automatic language profile is therefore a
//! high-confidence heuristic with a neutral fallback; users needing a fixed
//! Chinese, English, or Japanese profile can select one explicitly. It only
//! changes knobs that are safe to update between conversion chunks. Chunk size,
//! extra context, model paths, and denoiser topology remain reload-scoped.

use crate::model_rvc::{
    LiveParams, DEFAULT_F0_THRESHOLD, MAX_DENOISER_CONTENT_MIX, MAX_DENOISER_RMVPE_MIX,
    MAX_PROTECT, MAX_PROTECT_TRANSITION_MS,
};

const AUTO_SWITCH_CHUNKS: u8 = 8;
const MIN_AUTOMATIC_SCORE: f32 = 0.58;
const MIN_AUTOMATIC_MARGIN: f32 = 0.10;
const SILENCE_SUPPRESSOR_NOISE_FLOOR: f32 = 0.0008;
const SILENCE_SUPPRESSOR_NOISE_PRESSURE: f32 = 0.70;
const DYNAMIC_PITCH_SPEECH_RATIO: f32 = 0.30;
const DYNAMIC_PITCH_VARIATION_SEMITONES: f32 = 0.20;
const DYNAMIC_SPEECH_CREST_FACTOR: f32 = 2.25;
const DYNAMIC_PITCH_ENERGY_RATIO: f32 = 1.35;

/// Dynamic tuning selection. `Auto` uses a deliberately conservative acoustic
/// profile estimate; it does not assert that it has performed text-level
/// language recognition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DynamicTuningMode {
    #[default]
    Off = 0,
    Auto = 1,
    Chinese = 2,
    English = 3,
    Japanese = 4,
}

impl DynamicTuningMode {
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Auto,
            2 => Self::Chinese,
            3 => Self::English,
            4 => Self::Japanese,
            _ => Self::Off,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    const fn fixed_profile(self) -> Option<DynamicLanguageProfile> {
        match self {
            Self::Chinese => Some(DynamicLanguageProfile::Chinese),
            Self::English => Some(DynamicLanguageProfile::English),
            Self::Japanese => Some(DynamicLanguageProfile::Japanese),
            Self::Off | Self::Auto => None,
        }
    }
}

/// The active profile selected by a fixed mode or by Auto's acoustic heuristic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DynamicLanguageProfile {
    #[default]
    Neutral,
    Chinese,
    English,
    Japanese,
}

impl DynamicLanguageProfile {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Chinese => "Chinese",
            Self::English => "English",
            Self::Japanese => "Japanese",
        }
    }
}

/// Stable worker-to-frontend diagnostic state for dynamic tuning.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicTuningSnapshot {
    pub mode: DynamicTuningMode,
    pub profile: DynamicLanguageProfile,
    /// `1.0` for an explicitly selected profile. In Auto mode this is only the
    /// acoustic-profile confidence, not a probability from a speech recognizer.
    pub confidence: f32,
    pub noise_floor_rms: f32,
    pub estimated_snr_db: f32,
}

/// Per-chunk acoustics collected on the conversion worker. No state here owns
/// an audio buffer, so publishing an observation requires no callback-side work
/// or allocation.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicTuningObservation {
    pub input_rms: f32,
    pub input_peak: f32,
    pub zero_crossing_rate: f32,
    pub voiced_ratio: f32,
    pub pitch_variation_semitones: f32,
}

impl DynamicTuningObservation {
    /// Measure raw device-rate input before gain or denoising. The worker already
    /// owns this chunk, so this remains outside the real-time callback path.
    pub fn from_audio(audio: &[f32], voiced_ratio: f32, pitch_variation_semitones: f32) -> Self {
        if audio.is_empty() {
            return Self {
                voiced_ratio: sanitized_unit(voiced_ratio),
                pitch_variation_semitones: sanitized_non_negative(pitch_variation_semitones),
                ..Self::default()
            };
        }

        let mut sum_squares = 0.0f64;
        let mut peak = 0.0f32;
        let mut crossings = 0usize;
        let mut previous = 0.0f32;
        let mut has_previous = false;
        for &sample in audio {
            let sample = if sample.is_finite() { sample } else { 0.0 };
            sum_squares += f64::from(sample) * f64::from(sample);
            peak = peak.max(sample.abs());
            if has_previous && (sample >= 0.0) != (previous >= 0.0) {
                crossings = crossings.saturating_add(1);
            }
            previous = sample;
            has_previous = true;
        }

        Self {
            input_rms: (sum_squares / audio.len() as f64).sqrt() as f32,
            input_peak: peak,
            zero_crossing_rate: crossings as f32 / audio.len() as f32,
            voiced_ratio: sanitized_unit(voiced_ratio),
            pitch_variation_semitones: sanitized_non_negative(pitch_variation_semitones),
        }
    }
}

/// Stateful, worker-owned dynamic overlay. It deliberately keeps adjustments
/// relative to the frontend's latest base [`LiveParams`], so changing a manual
/// slider takes effect immediately instead of fighting a slow feedback loop.
#[derive(Clone, Debug)]
pub struct DynamicTuner {
    mode: DynamicTuningMode,
    profile: DynamicLanguageProfile,
    pending_profile: DynamicLanguageProfile,
    pending_profile_chunks: u8,
    confidence: f32,
    speech_rms: f32,
    noise_floor_rms: f32,
    zero_crossing_rate: f32,
    voiced_ratio: f32,
    pitch_variation_semitones: f32,
    estimated_snr_db: f32,
    noise_pressure: f32,
    // The pipeline resets its activity detector when this live flag changes.
    // Once automatic mode has seen enough room-noise evidence to enable it,
    // keep it enabled for this mode/session rather than flapping around the
    // SNR boundary and repeatedly re-entering the detector's startup grace.
    auto_silence_suppressor_latched: bool,
    input_gain_delta: f32,
    f0_threshold_delta: f32,
    index_rate_delta: f32,
    protect_delta: f32,
    protect_transition_delta_ms: i32,
    denoiser_content_delta: f32,
    denoiser_rmvpe_delta: f32,
}

impl Default for DynamicTuner {
    fn default() -> Self {
        Self {
            mode: DynamicTuningMode::Off,
            profile: DynamicLanguageProfile::Neutral,
            pending_profile: DynamicLanguageProfile::Neutral,
            pending_profile_chunks: 0,
            confidence: 0.0,
            speech_rms: 0.0,
            noise_floor_rms: 0.0,
            zero_crossing_rate: 0.0,
            voiced_ratio: 0.0,
            pitch_variation_semitones: 0.0,
            estimated_snr_db: 60.0,
            noise_pressure: 0.0,
            auto_silence_suppressor_latched: false,
            input_gain_delta: 0.0,
            f0_threshold_delta: 0.0,
            index_rate_delta: 0.0,
            protect_delta: 0.0,
            protect_transition_delta_ms: 0,
            denoiser_content_delta: 0.0,
            denoiser_rmvpe_delta: 0.0,
        }
    }
}

impl DynamicTuner {
    /// Return the worker-side parameters for the next conversion chunk. Only
    /// live-safe values are overlaid; topology and buffer-shape settings stay
    /// in the frontend's explicit Apply/restart flow.
    pub fn live_params(&mut self, mode: DynamicTuningMode, base: LiveParams) -> LiveParams {
        self.sync_mode(mode);
        if !mode.is_enabled() {
            return base;
        }

        let mut tuned = base;
        tuned.input_gain =
            (finite_or(base.input_gain, 1.0) * (1.0 + self.input_gain_delta)).clamp(0.0, 12.0);
        tuned.f0_threshold = (finite_or(base.f0_threshold, DEFAULT_F0_THRESHOLD)
            * (1.0 + self.f0_threshold_delta))
            .clamp(0.001, 0.5);
        // Never make an index appear merely because dynamic mode was enabled.
        // A missing index and an intentional zero Index rate must remain a strict
        // no-retrieval path.
        if base.index_rate > f32::EPSILON {
            tuned.index_rate = (base.index_rate + self.index_rate_delta).clamp(0.0, 1.0);
        }
        tuned.protect = (finite_or(base.protect, 0.5) + self.protect_delta).clamp(0.0, MAX_PROTECT);
        let transition = base.protect_transition_ms as i32 + self.protect_transition_delta_ms;
        tuned.protect_transition_ms = transition.clamp(0, MAX_PROTECT_TRANSITION_MS as i32) as u32;
        tuned.denoiser_content_mix = (finite_or(base.denoiser_content_mix, 0.25)
            + self.denoiser_content_delta)
            .clamp(0.0, MAX_DENOISER_CONTENT_MIX);
        tuned.denoiser_rmvpe_mix = (finite_or(base.denoiser_rmvpe_mix, 1.0)
            + self.denoiser_rmvpe_delta)
            .clamp(0.0, MAX_DENOISER_RMVPE_MIX);

        // Dynamic mode is explicitly opt-in. Once it sees a meaningful ambient
        // floor, it can enable the output-only suppressor, never the exclusive
        // input-gate denoiser, so a selected stateful denoiser stays intact.
        tuned.silence_gate_enabled =
            base.silence_gate_enabled || self.auto_silence_suppressor_latched;
        if tuned.silence_gate_enabled {
            let adaptive_floor = self.noise_floor_rms * (2.0 + self.noise_pressure);
            tuned.noise_gate_threshold = finite_or(base.noise_gate_threshold, 0.01)
                .max(adaptive_floor)
                .clamp(0.0001, 0.5);
        }
        tuned
    }

    /// Feed measurements from a completed conversion chunk. The resulting
    /// overlay applies on the next chunk, preserving the model's current chunk
    /// timeline and avoiding mid-inference parameter changes.
    pub fn observe(&mut self, mode: DynamicTuningMode, observation: DynamicTuningObservation) {
        self.sync_mode(mode);
        if !mode.is_enabled() {
            return;
        }

        let observation = DynamicTuningObservation {
            input_rms: sanitized_non_negative(observation.input_rms),
            input_peak: sanitized_non_negative(observation.input_peak),
            zero_crossing_rate: sanitized_unit(observation.zero_crossing_rate),
            voiced_ratio: sanitized_unit(observation.voiced_ratio),
            pitch_variation_semitones: sanitized_non_negative(
                observation.pitch_variation_semitones,
            ),
        };
        let crest_factor = observation.input_peak / observation.input_rms.max(0.0001);
        let pitch_has_speech_shape = observation.pitch_variation_semitones
            >= DYNAMIC_PITCH_VARIATION_SEMITONES
            || crest_factor >= DYNAMIC_SPEECH_CREST_FACTOR
            || (self.noise_floor_rms > 0.0
                && observation.input_rms
                    >= self.noise_floor_rms.max(0.0001) * DYNAMIC_PITCH_ENERGY_RATIO);
        // Do not allow a stable F0 alone to seed the speech estimate. A fan or
        // mains hum can look fully voiced to RMVPE, and misclassifying it here
        // prevents the dynamic layer from learning the room floor that enables
        // its output suppressor.
        let likely_speech = (observation.voiced_ratio >= DYNAMIC_PITCH_SPEECH_RATIO
            && pitch_has_speech_shape)
            || (self.speech_rms > 0.0
                && observation.input_rms >= self.noise_floor_rms.max(0.0001) * 3.0
                && observation.input_rms >= self.speech_rms * 0.40);

        if likely_speech {
            self.speech_rms = ema(self.speech_rms, observation.input_rms, 0.16);
            self.zero_crossing_rate = ema(
                self.zero_crossing_rate,
                observation.zero_crossing_rate,
                0.20,
            );
            self.voiced_ratio = ema(self.voiced_ratio, observation.voiced_ratio, 0.20);
            self.pitch_variation_semitones = ema(
                self.pitch_variation_semitones,
                observation.pitch_variation_semitones,
                0.20,
            );
        } else {
            self.noise_floor_rms = ema(self.noise_floor_rms, observation.input_rms, 0.15);
        }
        // Do not seed the room floor from a first spoken chunk. Doing so would
        // manufacture a 0 dB SNR at startup and make a clean Chinese/Japanese
        // voice look noisy. Until a non-speech observation arrives, unknown
        // noise is treated as no evidence for a noise-driven overlay.

        self.estimated_snr_db = estimate_snr_db(self.speech_rms, self.noise_floor_rms);
        self.noise_pressure = ((20.0 - self.estimated_snr_db) / 25.0).clamp(0.0, 1.0);
        if self.noise_pressure >= SILENCE_SUPPRESSOR_NOISE_PRESSURE
            && self.noise_floor_rms >= SILENCE_SUPPRESSOR_NOISE_FLOOR
        {
            self.auto_silence_suppressor_latched = true;
        }
        if let Some(profile) = mode.fixed_profile() {
            self.profile = profile;
            self.pending_profile = profile;
            self.pending_profile_chunks = AUTO_SWITCH_CHUNKS;
            self.confidence = 1.0;
        } else {
            self.update_auto_profile(likely_speech);
        }
        self.update_overlay();
    }

    pub fn snapshot(&self) -> DynamicTuningSnapshot {
        DynamicTuningSnapshot {
            mode: self.mode,
            profile: self.profile,
            confidence: self.confidence.clamp(0.0, 1.0),
            noise_floor_rms: self.noise_floor_rms.max(0.0),
            estimated_snr_db: self.estimated_snr_db,
        }
    }

    fn sync_mode(&mut self, mode: DynamicTuningMode) {
        if self.mode == mode {
            return;
        }
        *self = Self {
            mode,
            profile: mode
                .fixed_profile()
                .unwrap_or(DynamicLanguageProfile::Neutral),
            confidence: if mode.fixed_profile().is_some() {
                1.0
            } else {
                0.0
            },
            ..Self::default()
        };
    }

    fn update_auto_profile(&mut self, likely_speech: bool) {
        if self.mode != DynamicTuningMode::Auto || !likely_speech {
            return;
        }
        let (candidate, confidence) = classify_acoustic_profile(
            self.voiced_ratio,
            self.zero_crossing_rate,
            self.pitch_variation_semitones,
        );
        if candidate != self.pending_profile {
            self.pending_profile = candidate;
            self.pending_profile_chunks = 1;
        } else {
            self.pending_profile_chunks = self.pending_profile_chunks.saturating_add(1);
        }
        if self.pending_profile_chunks >= AUTO_SWITCH_CHUNKS {
            self.profile = candidate;
        }
        self.confidence = if self.profile == candidate {
            confidence
        } else {
            0.0
        };
    }

    fn update_overlay(&mut self) {
        let (
            profile_index,
            profile_protect,
            profile_content,
            profile_rmvpe,
            profile_f0,
            profile_transition,
        ) = match self.profile {
            // Tonal speech benefits from a slightly lower F0 confidence
            // cutoff; all changes remain bounded and are blended over time.
            DynamicLanguageProfile::Chinese => (-0.03, -0.04, 0.00, 0.05, -0.12, 5),
            // English's denser fricative/consonant runs retain more raw
            // ContentVec detail and use a gentler Protect boundary.
            DynamicLanguageProfile::English => (0.00, -0.10, -0.10, -0.03, 0.08, 10),
            DynamicLanguageProfile::Japanese => (-0.02, -0.06, -0.05, 0.03, -0.08, 5),
            DynamicLanguageProfile::Neutral => (0.0, 0.0, 0.0, 0.0, 0.0, 0),
        };
        let noise = self.noise_pressure;
        let gain_target = if self.speech_rms > 0.001 {
            (0.04 / self.speech_rms).clamp(0.92, 1.08) - 1.0
        } else {
            0.0
        };
        slew(&mut self.input_gain_delta, gain_target, 0.01);
        slew(
            &mut self.f0_threshold_delta,
            profile_f0 + noise * 0.25,
            0.015,
        );
        slew(
            &mut self.index_rate_delta,
            profile_index - noise * 0.30,
            0.02,
        );
        slew(
            &mut self.protect_delta,
            profile_protect - noise * 0.10,
            0.015,
        );
        slew(
            &mut self.denoiser_content_delta,
            profile_content + noise * 0.20,
            0.02,
        );
        slew(
            &mut self.denoiser_rmvpe_delta,
            profile_rmvpe + noise * 0.15,
            0.02,
        );
        slew_i32(
            &mut self.protect_transition_delta_ms,
            profile_transition + (noise * 10.0).round() as i32,
            2,
        );
    }
}

fn classify_acoustic_profile(
    voiced_ratio: f32,
    zero_crossing_rate: f32,
    pitch_variation_semitones: f32,
) -> (DynamicLanguageProfile, f32) {
    let voiced = sanitized_unit(voiced_ratio);
    let zcr = (sanitized_unit(zero_crossing_rate) / 0.25).clamp(0.0, 1.0);
    let pitch = ((sanitized_non_negative(pitch_variation_semitones) - 0.25) / 1.5).clamp(0.0, 1.0);
    let chinese = 0.40 * voiced + 0.45 * pitch + 0.15 * (1.0 - zcr);
    let english = 0.60 * zcr + 0.25 * (1.0 - voiced) + 0.15 * (1.0 - pitch);
    let japanese = 0.55 * voiced
        + 0.25 * (1.0 - zcr)
        + 0.20 * (1.0 - ((pitch - 0.45).abs() / 0.45).clamp(0.0, 1.0));
    let mut ranked = [
        (DynamicLanguageProfile::Chinese, chinese),
        (DynamicLanguageProfile::English, english),
        (DynamicLanguageProfile::Japanese, japanese),
    ];
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let (profile, score) = ranked[0];
    let margin = score - ranked[1].1;
    if score >= MIN_AUTOMATIC_SCORE && margin >= MIN_AUTOMATIC_MARGIN {
        (profile, (score * margin / 0.35).clamp(0.0, 1.0))
    } else {
        (DynamicLanguageProfile::Neutral, 0.0)
    }
}

fn estimate_snr_db(speech_rms: f32, noise_floor_rms: f32) -> f32 {
    if speech_rms > 0.0 && noise_floor_rms > 0.0 {
        (20.0 * (speech_rms / noise_floor_rms).log10()).clamp(-20.0, 60.0)
    } else {
        60.0
    }
}

fn ema(previous: f32, value: f32, alpha: f32) -> f32 {
    if previous <= 0.0 || !previous.is_finite() {
        value
    } else {
        previous + (value - previous) * alpha
    }
}

fn slew(value: &mut f32, target: f32, max_step: f32) {
    let target = if target.is_finite() { target } else { 0.0 };
    *value += (target - *value).clamp(-max_step, max_step);
}

fn slew_i32(value: &mut i32, target: i32, max_step: i32) {
    *value += (target - *value).clamp(-max_step, max_step);
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn sanitized_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn sanitized_unit(value: f32) -> f32 {
    sanitized_non_negative(value).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> LiveParams {
        LiveParams {
            f0_threshold: 0.05,
            input_gain: 1.0,
            index_rate: 0.75,
            protect: 0.33,
            protect_transition_ms: 15,
            denoiser_content_mix: 0.25,
            denoiser_rmvpe_mix: 0.85,
            ..LiveParams::default()
        }
    }

    #[test]
    fn fixed_chinese_profile_preserves_tonal_f0_with_bounded_changes() {
        let mut tuner = DynamicTuner::default();
        let observation = DynamicTuningObservation {
            input_rms: 0.04,
            input_peak: 0.12,
            zero_crossing_rate: 0.05,
            voiced_ratio: 0.90,
            pitch_variation_semitones: 1.6,
        };
        for _ in 0..24 {
            tuner.observe(DynamicTuningMode::Chinese, observation);
        }
        let tuned = tuner.live_params(DynamicTuningMode::Chinese, baseline());

        assert_eq!(tuner.snapshot().profile, DynamicLanguageProfile::Chinese);
        assert!(tuned.f0_threshold < baseline().f0_threshold);
        assert!(tuned.denoiser_rmvpe_mix > baseline().denoiser_rmvpe_mix);
        assert!(tuned.protect <= baseline().protect);
    }

    #[test]
    fn auto_mode_requires_stable_high_confidence_evidence_before_switching() {
        let mut tuner = DynamicTuner::default();
        let english_like = DynamicTuningObservation {
            input_rms: 0.04,
            input_peak: 0.10,
            zero_crossing_rate: 0.25,
            voiced_ratio: 0.30,
            pitch_variation_semitones: 0.10,
        };
        for _ in 0..AUTO_SWITCH_CHUNKS.saturating_sub(1) {
            tuner.observe(DynamicTuningMode::Auto, english_like);
        }
        assert_eq!(tuner.snapshot().profile, DynamicLanguageProfile::Neutral);

        tuner.observe(DynamicTuningMode::Auto, english_like);
        let snapshot = tuner.snapshot();
        assert_eq!(snapshot.profile, DynamicLanguageProfile::English);
        assert!(snapshot.confidence > 0.0);
    }

    #[test]
    fn disabled_mode_returns_the_manual_baseline_unchanged() {
        let mut tuner = DynamicTuner::default();
        let baseline = baseline();
        tuner.observe(
            DynamicTuningMode::Chinese,
            DynamicTuningObservation {
                input_rms: 0.04,
                ..DynamicTuningObservation::default()
            },
        );
        let tuned = tuner.live_params(DynamicTuningMode::Off, baseline);

        assert_eq!(tuned.f0_threshold, baseline.f0_threshold);
        assert_eq!(tuned.index_rate, baseline.index_rate);
        assert_eq!(tuned.protect, baseline.protect);
        assert_eq!(tuned.denoiser_content_mix, baseline.denoiser_content_mix);
    }

    #[test]
    fn noisy_auto_mode_can_enable_only_the_output_silence_suppressor() {
        let mut tuner = DynamicTuner::default();
        let noise = DynamicTuningObservation {
            input_rms: 0.02,
            input_peak: 0.04,
            zero_crossing_rate: 0.1,
            voiced_ratio: 0.0,
            pitch_variation_semitones: 0.0,
        };
        let speech = DynamicTuningObservation {
            input_rms: 0.025,
            input_peak: 0.10,
            zero_crossing_rate: 0.25,
            voiced_ratio: 0.35,
            pitch_variation_semitones: 0.1,
        };
        for _ in 0..10 {
            tuner.observe(DynamicTuningMode::Auto, noise);
        }
        for _ in 0..10 {
            tuner.observe(DynamicTuningMode::Auto, speech);
        }
        let tuned = tuner.live_params(DynamicTuningMode::Auto, baseline());

        assert!(tuned.silence_gate_enabled);
        assert!(!tuned.noise_gate_enabled);
        assert!(tuned.noise_gate_threshold >= baseline().noise_gate_threshold);
    }

    #[test]
    fn false_pitch_room_tone_is_learned_as_noise_and_auto_gate_stays_latched() {
        let mut tuner = DynamicTuner::default();
        let tonal_room_noise = DynamicTuningObservation {
            input_rms: 0.02,
            input_peak: 0.028,
            zero_crossing_rate: 0.05,
            voiced_ratio: 0.95,
            pitch_variation_semitones: 0.01,
        };
        for _ in 0..10 {
            tuner.observe(DynamicTuningMode::Auto, tonal_room_noise);
        }
        assert!(tuner.snapshot().noise_floor_rms > 0.018);
        assert!(
            !tuner
                .live_params(DynamicTuningMode::Auto, baseline())
                .silence_gate_enabled
        );

        // A speech-shaped chunk can establish the SNR estimate and enable the
        // automatic output gate even in the presence of the learned room tone.
        let speech = DynamicTuningObservation {
            input_rms: 0.025,
            input_peak: 0.10,
            zero_crossing_rate: 0.25,
            voiced_ratio: 0.35,
            pitch_variation_semitones: 0.10,
        };
        for _ in 0..10 {
            tuner.observe(DynamicTuningMode::Auto, speech);
        }
        assert!(
            tuner
                .live_params(DynamicTuningMode::Auto, baseline())
                .silence_gate_enabled
        );

        // A later clean speech run lowers instantaneous noise pressure. The
        // gate must remain enabled so the pipeline does not reset its detector
        // and briefly leak idle RVC noise on every SNR-boundary transition.
        let clean_speech = DynamicTuningObservation {
            input_rms: 0.15,
            input_peak: 0.40,
            zero_crossing_rate: 0.18,
            voiced_ratio: 0.70,
            pitch_variation_semitones: 0.80,
        };
        for _ in 0..32 {
            tuner.observe(DynamicTuningMode::Auto, clean_speech);
        }
        assert!(
            tuner
                .live_params(DynamicTuningMode::Auto, baseline())
                .silence_gate_enabled
        );
    }
}

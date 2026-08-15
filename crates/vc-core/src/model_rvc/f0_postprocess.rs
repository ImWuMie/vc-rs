//! Lightweight F0 post-processing for the RVC pipeline.
//!
//! Operates on the RVC-aligned `pitchf` (same frame grid as the ContentVec
//! features and the coarse `pitch`). The pipeline extracts RMVPE F0 with a
//! `0.0` pitch shift, so the array reaching this module is raw / natural F0:
//! range clamp, octave correction and the median filter all act on natural F0,
//! and pitch shift is applied here exactly once at the very end. Coarse `pitch`
//! is recomputed by the caller from the final `pitchf` (`coarse_pitch_into`).
//!
//! Real-time note: this runs on the worker thread, never the audio callback.
//! Scratch buffers are reused so a steady-state chunk does no heap allocation.

/// `0.0` marks an unvoiced frame throughout this module.
const UNVOICED: f32 = 0.0;

/// Octave-jump detection tolerances (ratio space).
///
/// A frame is only treated as a single-frame octave error when its neighbours
/// are themselves close (so we do not flatten a genuine pitch glide) and the
/// centre sits near an exact 2x / 0.5x of the neighbour average.
const LR_NEAR_RATIO_TOL: f32 = 0.2;
const OCTAVE_RATIO_TOL: f32 = 0.25;

/// Waveform checks are intentionally conservative. A low score is strong
/// evidence that RMVPE found pitch in aperiodic noise; a middling score is not
/// enough to discard a breathy or weak voiced frame.
const PERIODICITY_WINDOW_RADIUS_SAMPLES_16K: usize = 320;
const PERIODICITY_MIN_OVERLAP_SAMPLES_16K: usize = 160;
const MIN_WAVEFORM_PERIODICITY: f32 = 0.16;

/// Full continuity across a long zero run turns pauses and unvoiced consonants
/// into synthetic voiced sound. Two missing 10 ms frames are safe to repair;
/// longer gaps need stable pitch support on both sides and are capped at 50 ms.
const DIRECT_GAP_FILL_FRAMES: usize = 2;
const CONTINUITY_MAX_GAP_FRAMES: usize = 5;
const CONTINUITY_LOCAL_SEMITONES: f32 = 1.5;
const CONTINUITY_BOUNDARY_SEMITONES: f32 = 3.0;

#[derive(Clone, Debug)]
pub struct F0PostprocessConfig {
    pub enabled: bool,

    /// Reject voiced RMVPE frames that have no matching periodic support in the
    /// 16 kHz waveform. This is a local autocorrelation check, not an RMVPE
    /// posterior probability, and runs before retrieval/protect decisions.
    pub waveform_periodicity_validation: bool,

    pub min_f0_hz: f32,
    pub max_f0_hz: f32,

    pub remove_short_voiced_islands: bool,
    pub max_voiced_island_frames: usize,

    pub fill_short_unvoiced_gaps: bool,
    pub max_unvoiced_gap_frames: usize,

    /// Fill bounded internal zero runs. The implementation always repairs gaps
    /// up to two frames, then requires stable pitch evidence on both sides, and
    /// never crosses `max_unvoiced_gap_frames`. Leading/trailing runs stay
    /// unvoiced because their off-window context is unknown.
    pub interpolate_internal_unvoiced_gaps: bool,

    pub fix_octave_jumps: bool,
    /// Correct a final-frame octave outlier when the preceding two frames agree.
    /// A rolling stream has no right neighbour for its newest F0 frame, so this
    /// bounded look-behind handles the otherwise unresolved chunk-edge case.
    pub fix_trailing_octave_jumps: bool,

    pub median_filter: bool,
    pub median_filter_radius: usize,

    /// When true, saturate voiced frames into `min_f0_hz..=max_f0_hz` *after*
    /// pitch shift (out-of-range values are clamped to the bound, kept voiced).
    /// Deliberately different from the pre-shift invalid step, which *zeroes*
    /// out-of-range frames: zeroing a shifted-up high note would silence it.
    /// Near-existing behaviour `false` is the default.
    pub clamp_after_pitch_shift: bool,
}

impl Default for F0PostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            waveform_periodicity_validation: false,

            min_f0_hz: 50.0,   // matches existing coarse f0_min
            max_f0_hz: 1100.0, // matches existing coarse f0_max

            remove_short_voiced_islands: true,
            max_voiced_island_frames: 1,

            fill_short_unvoiced_gaps: true,
            max_unvoiced_gap_frames: 2,
            interpolate_internal_unvoiced_gaps: false,

            fix_octave_jumps: true,
            fix_trailing_octave_jumps: false,

            median_filter: true,
            median_filter_radius: 1,

            clamp_after_pitch_shift: false,
        }
    }
}

impl F0PostprocessConfig {
    /// The compatibility-oriented F0-continuity treatment. Keep the other
    /// corrective filters off: they are independent policy choices and must not
    /// silently alter vibrato or genuine octave transitions when a front-end
    /// asks only for dropout interpolation.
    pub fn continuity(enabled: bool) -> Self {
        Self {
            enabled,
            waveform_periodicity_validation: false,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            interpolate_internal_unvoiced_gaps: true,
            max_unvoiced_gap_frames: CONTINUITY_MAX_GAP_FRAMES,
            fix_octave_jumps: false,
            median_filter: false,
            ..Self::default()
        }
    }

    /// Build the shared realtime F0 policy from independently visible controls.
    ///
    /// `continuity` preserves the established MXGF-compatible interpolation
    /// behavior. `stabilization` adds only temporal evidence checks around that
    /// policy: a one-frame voiced island is discarded, isolated octave mistakes
    /// are repaired, and a three-frame voiced median removes impulsive F0 noise.
    /// It does not claim access to an RMVPE probability, which the current model
    /// session API does not expose.
    pub fn continuity_with_stabilization(continuity: bool, stabilization: bool) -> Self {
        let mut config = Self::continuity(continuity);
        if !stabilization {
            return config;
        }

        // Stabilization can stand alone when users disable full internal-gap
        // interpolation. In that case, retain only short dropouts so silence and
        // stop consonants are not turned into a long synthetic pitch contour.
        config.enabled = true;
        config.waveform_periodicity_validation = true;
        config.remove_short_voiced_islands = true;
        config.max_voiced_island_frames = 1;
        if !continuity {
            config.fill_short_unvoiced_gaps = true;
            config.max_unvoiced_gap_frames = 2;
        }
        config.fix_octave_jumps = true;
        config.fix_trailing_octave_jumps = true;
        config.median_filter = true;
        config.median_filter_radius = 1;
        config
    }
}

pub struct F0Postprocessor {
    config: F0PostprocessConfig,
    /// Sorting window for the median filter (reused per frame).
    median_scratch: Vec<f32>,
    /// Snapshot of the array as it enters the median pass. The median reads from
    /// this copy and writes only to the output, so a corrected frame can never
    /// feed the next frame's window (which would make the filter order-dependent).
    median_input_scratch: Vec<f32>,
}

impl F0Postprocessor {
    pub fn new(config: F0PostprocessConfig) -> Self {
        Self {
            config,
            median_scratch: Vec::new(),
            median_input_scratch: Vec::new(),
        }
    }

    // Accessors for the future runtime/UI wiring (mirrors `set_pitch_shift` &c.).
    // Not used by the engine yet; that wiring is a separate task.
    #[allow(dead_code)]
    pub fn config(&self) -> &F0PostprocessConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub fn set_config(&mut self, config: F0PostprocessConfig) {
        self.config = config;
    }

    /// Validate thresholded RMVPE F0 against the exact 16 kHz waveform window
    /// used by the estimator.
    ///
    /// The output remains natural/unshifted F0. Keeping this pass before the
    /// rolling F0 timeline means speech activity, adaptive retrieval, Protect,
    /// and synthesis all see the same corrected voiced/unvoiced decision. The
    /// caller owns and reuses `output`; no allocation belongs in an audio
    /// callback (this method runs only on the conversion worker).
    pub fn validate_raw_pitchf_into(
        &self,
        input_pitchf: &[f32],
        audio_16k: &[f32],
        output: &mut Vec<f32>,
    ) {
        output.clear();
        output.extend_from_slice(input_pitchf);
        if !self.config.waveform_periodicity_validation {
            return;
        }

        for (frame_index, f0) in output.iter_mut().enumerate() {
            if !f0.is_finite() || *f0 < self.config.min_f0_hz || *f0 > self.config.max_f0_hz {
                *f0 = UNVOICED;
                continue;
            }
            if *f0 <= UNVOICED {
                *f0 = UNVOICED;
                continue;
            }

            if waveform_periodicity_16k(audio_16k, frame_index, *f0)
                .is_some_and(|periodicity| periodicity < MIN_WAVEFORM_PERIODICITY)
            {
                *f0 = UNVOICED;
            }
        }
    }

    /// Post-process aligned raw `pitchf` and apply pitch shift exactly once.
    ///
    /// `input_pitchf` is the RVC-aligned, *un-shifted* (natural) F0 and is never
    /// mutated; the result is written to `output`.
    ///
    /// Always call this, even when post-processing is disabled: pitch shift is
    /// applied here (RMVPE extract receives `0.0`), so skipping the call would
    /// drop the shift entirely. When disabled, the smoothing/invalid steps are
    /// skipped and only the shift (+ optional post-shift clamp) is applied,
    /// which is bit-equivalent to the previous "shift in extract" path for a
    /// static pitch shift.
    pub fn process_pitchf_into(
        &mut self,
        input_pitchf: &[f32],
        pitch_shift_semitones: f32,
        output: &mut Vec<f32>,
    ) {
        output.clear();
        output.extend_from_slice(input_pitchf);

        if self.config.enabled {
            self.remove_invalid(output);
            if self.config.remove_short_voiced_islands {
                self.remove_short_voiced_islands(output);
            }
            if self.config.interpolate_internal_unvoiced_gaps {
                self.interpolate_internal_unvoiced_gaps(output);
            } else if self.config.fill_short_unvoiced_gaps {
                self.fill_short_unvoiced_gaps(output);
            }
            if self.config.fix_octave_jumps {
                self.fix_octave_jumps(output);
                if self.config.fix_trailing_octave_jumps {
                    self.fix_trailing_octave_jump(output);
                }
            }
            if self.config.median_filter && self.config.median_filter_radius > 0 {
                self.median_filter(output);
            }
        }

        // Pitch shift, applied exactly once for both enabled and disabled modes.
        if pitch_shift_semitones != 0.0 {
            let factor = 2.0_f32.powf(pitch_shift_semitones / 12.0);
            for f0 in output.iter_mut() {
                if *f0 > UNVOICED {
                    *f0 *= factor;
                }
            }
        }

        if self.config.clamp_after_pitch_shift {
            let (min, max) = (self.config.min_f0_hz, self.config.max_f0_hz);
            for f0 in output.iter_mut() {
                if *f0 > UNVOICED {
                    *f0 = f0.clamp(min, max);
                }
            }
        }
    }

    /// Step 3: NaN/inf, non-positive, and out-of-[min,max] frames become unvoiced.
    /// Runs on natural (pre-shift) F0.
    fn remove_invalid(&self, pitchf: &mut [f32]) {
        let (min, max) = (self.config.min_f0_hz, self.config.max_f0_hz);
        for f0 in pitchf.iter_mut() {
            if !f0.is_finite() || *f0 <= UNVOICED || *f0 < min || *f0 > max {
                *f0 = UNVOICED;
            }
        }
    }

    /// Step 4: zero voiced runs of `<= max_voiced_island_frames` that are
    /// surrounded by unvoiced on both sides. Runs touching either edge are not
    /// "islands" (their off-window context is unknown) and are kept.
    fn remove_short_voiced_islands(&self, pitchf: &mut [f32]) {
        let max_len = self.config.max_voiced_island_frames;
        let n = pitchf.len();
        let mut i = 0;
        while i < n {
            if pitchf[i] > UNVOICED {
                let start = i;
                while i < n && pitchf[i] > UNVOICED {
                    i += 1;
                }
                let end = i; // exclusive
                let touches_edge = start == 0 || end == n;
                if !touches_edge && end - start <= max_len {
                    pitchf[start..end].fill(UNVOICED);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Step 5: fill unvoiced runs of `<= max_unvoiced_gap_frames` that are
    /// bounded by voiced frames on both sides, using log-F0 linear interpolation
    /// between the bounding values. Leading/trailing gaps are left unvoiced.
    fn fill_short_unvoiced_gaps(&self, pitchf: &mut [f32]) {
        let max_len = self.config.max_unvoiced_gap_frames;
        let n = pitchf.len();
        let mut i = 0;
        while i < n {
            if pitchf[i] <= UNVOICED {
                let start = i;
                while i < n && pitchf[i] <= UNVOICED {
                    i += 1;
                }
                let end = i; // exclusive; first voiced frame after the gap (or n)
                             // Bounded on both sides => start > 0 (voiced at start-1) and
                             // end < n (voiced at end). Both bounds are guaranteed voiced.
                if start > 0 && end < n && end - start <= max_len {
                    let log_left = pitchf[start - 1].ln();
                    let log_right = pitchf[end].ln();
                    let steps = (end - start + 1) as f32;
                    for (k, idx) in (start..end).enumerate() {
                        let t = (k + 1) as f32 / steps;
                        pitchf[idx] = (log_left + (log_right - log_left) * t).exp();
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    /// Fill only short, reliable internal unvoiced runs. Unbounded interpolation
    /// is tempting on a rolling window, but it voices complete pauses, breaths,
    /// and fricatives whenever a later vowel supplies the missing right bound.
    /// This pass therefore repairs one/two-frame dropouts directly, admits up to
    /// the configured cap only when both bounds have stable voiced support, and
    /// never extrapolates an edge run.
    fn interpolate_internal_unvoiced_gaps(&self, pitchf: &mut [f32]) {
        let n = pitchf.len();
        let mut i = 0;
        while i < n {
            if pitchf[i] <= UNVOICED {
                let start = i;
                while i < n && pitchf[i] <= UNVOICED {
                    i += 1;
                }
                let end = i;
                let gap_frames = end - start;
                if start > 0
                    && end < n
                    && gap_frames <= self.config.max_unvoiced_gap_frames
                    && (gap_frames <= DIRECT_GAP_FILL_FRAMES
                        || has_reliable_gap_bounds(pitchf, start, end))
                {
                    interpolate_log_f0_gap(pitchf, start, end);
                }
            } else {
                i += 1;
            }
        }
    }

    /// Step 6: correct isolated single-frame ~2x / ~0.5x octave jumps only.
    /// Requires left/center/right all voiced and left close to right, so a
    /// genuine sustained octave change (two adjacent shifted frames) is kept.
    fn fix_octave_jumps(&self, pitchf: &mut [f32]) {
        let n = pitchf.len();
        if n < 3 {
            return;
        }
        for i in 1..n - 1 {
            let left = pitchf[i - 1];
            let center = pitchf[i];
            let right = pitchf[i + 1];
            if left <= UNVOICED || center <= UNVOICED || right <= UNVOICED {
                continue;
            }
            if (left / right - 1.0).abs() > LR_NEAR_RATIO_TOL {
                continue;
            }
            let reference = 0.5 * (left + right);
            let ratio = center / reference;
            if (ratio - 2.0).abs() <= OCTAVE_RATIO_TOL {
                pitchf[i] = center * 0.5;
            } else if (ratio - 0.5).abs() <= OCTAVE_RATIO_TOL {
                pitchf[i] = center * 2.0;
            }
        }
    }

    /// Correct a potential last-frame octave error using two preceding voiced
    /// frames as temporal evidence. This is intentionally stricter than the
    /// interior rule: a real octave transition is allowed whenever the previous
    /// two frames do not already agree, preventing a new sustained note from
    /// being flattened merely because it begins at a chunk boundary.
    fn fix_trailing_octave_jump(&self, pitchf: &mut [f32]) {
        let n = pitchf.len();
        if n < 3 {
            return;
        }
        let before = pitchf[n - 3];
        let left = pitchf[n - 2];
        let center = pitchf[n - 1];
        if before <= UNVOICED || left <= UNVOICED || center <= UNVOICED {
            return;
        }
        if (before / left - 1.0).abs() > LR_NEAR_RATIO_TOL {
            return;
        }
        let reference = 0.5 * (before + left);
        let ratio = center / reference;
        if (ratio - 2.0).abs() <= OCTAVE_RATIO_TOL {
            pitchf[n - 1] = center * 0.5;
        } else if (ratio - 0.5).abs() <= OCTAVE_RATIO_TOL {
            pitchf[n - 1] = center * 2.0;
        }
    }

    /// Step 7: log-F0 median filter over voiced frames only.
    ///
    /// Unvoiced (`0.0`) frames stay unvoiced and are never mixed into a window.
    /// Reads from a snapshot (`median_input_scratch`) taken before the pass so
    /// the filter is order-independent; writes results into `pitchf`.
    fn median_filter(&mut self, pitchf: &mut [f32]) {
        let radius = self.config.median_filter_radius;
        let n = pitchf.len();
        let input = &mut self.median_input_scratch;
        input.clear();
        input.extend_from_slice(pitchf);
        let window = &mut self.median_scratch;
        for i in 0..n {
            if input[i] <= UNVOICED {
                continue;
            }
            if radius == 1 {
                // A three-frame median is meant to remove an impulse, not the
                // first frame of a real pitch transition. Require both adjacent
                // voiced frames to agree before replacing the centre; this also
                // makes the quality preset preserve deliberate octave changes.
                if i == 0 || i + 1 >= n {
                    continue;
                }
                let left = input[i - 1];
                let right = input[i + 1];
                if left <= UNVOICED
                    || right <= UNVOICED
                    || (left / right - 1.0).abs() > LR_NEAR_RATIO_TOL
                {
                    continue;
                }
            }
            window.clear();
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(n);
            for &v in &input[lo..hi] {
                if v > UNVOICED {
                    window.push(v);
                }
            }
            // The center is always voiced and included, so the window is never
            // empty. With no voiced neighbours the median equals the center and
            // the value is left unchanged.
            window.sort_by(f32::total_cmp);
            let m = window.len();
            pitchf[i] = if m % 2 == 1 {
                window[m / 2]
            } else {
                // Even effective count: average the two middle frames in log
                // space (the "log-F0" part that actually matters for medians).
                let a = window[m / 2 - 1];
                let b = window[m / 2];
                (0.5 * (a.ln() + b.ln())).exp()
            };
        }
    }
}

pub(super) fn waveform_periodicity_16k(
    audio_16k: &[f32],
    frame_index: usize,
    f0_hz: f32,
) -> Option<f32> {
    if audio_16k.is_empty() || !f0_hz.is_finite() || f0_hz <= 0.0 {
        return None;
    }

    let expected_lag = (16_000.0 / f0_hz).round() as usize;
    if expected_lag < 2 {
        return None;
    }
    let center = frame_index.saturating_mul(160).min(audio_16k.len());
    let start = center.saturating_sub(PERIODICITY_WINDOW_RADIUS_SAMPLES_16K);
    let end = center
        .saturating_add(PERIODICITY_WINDOW_RADIUS_SAMPLES_16K)
        .min(audio_16k.len());
    let window = &audio_16k[start..end];
    let search_radius = ((expected_lag as f32 * 0.04).ceil() as usize).clamp(1, 8);
    let first_lag = expected_lag.saturating_sub(search_radius).max(2);
    let last_lag = expected_lag.saturating_add(search_radius);
    let mut best: Option<f32> = None;

    for lag in first_lag..=last_lag {
        let Some(overlap) = window.len().checked_sub(lag) else {
            continue;
        };
        if overlap < PERIODICITY_MIN_OVERLAP_SAMPLES_16K {
            continue;
        }
        let correlation = normalized_lag_correlation(window, lag, overlap);
        best = Some(best.map_or(correlation, |value| value.max(correlation)));
    }
    best.map(|value| value.clamp(-1.0, 1.0))
}

fn normalized_lag_correlation(window: &[f32], lag: usize, overlap: usize) -> f32 {
    let left = &window[..overlap];
    let right = &window[lag..lag + overlap];
    let count = overlap as f64;
    let left_mean = left
        .iter()
        .map(|sample| f64::from(finite_or_zero(*sample)))
        .sum::<f64>()
        / count;
    let right_mean = right
        .iter()
        .map(|sample| f64::from(finite_or_zero(*sample)))
        .sum::<f64>()
        / count;
    let mut covariance = 0.0f64;
    let mut left_energy = 0.0f64;
    let mut right_energy = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        let left = f64::from(finite_or_zero(left)) - left_mean;
        let right = f64::from(finite_or_zero(right)) - right_mean;
        covariance += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        (covariance / denominator) as f32
    }
}

fn has_reliable_gap_bounds(pitchf: &[f32], start: usize, end: usize) -> bool {
    if start < 2 || end + 1 >= pitchf.len() {
        return false;
    }
    let left_support = pitchf[start - 2];
    let left = pitchf[start - 1];
    let right = pitchf[end];
    let right_support = pitchf[end + 1];
    [left_support, left, right, right_support]
        .into_iter()
        .all(|f0| f0.is_finite() && f0 > UNVOICED)
        && pitch_distance_semitones(left_support, left) <= CONTINUITY_LOCAL_SEMITONES
        && pitch_distance_semitones(right, right_support) <= CONTINUITY_LOCAL_SEMITONES
        && pitch_distance_semitones(left, right) <= CONTINUITY_BOUNDARY_SEMITONES
}

pub(super) fn pitch_distance_semitones(left: f32, right: f32) -> f32 {
    12.0 * (left / right).log2().abs()
}

fn interpolate_log_f0_gap(pitchf: &mut [f32], start: usize, end: usize) {
    let log_left = pitchf[start - 1].ln();
    let log_right = pitchf[end].ln();
    let steps = (end - start + 1) as f32;
    for (offset, value) in pitchf[start..end].iter_mut().enumerate() {
        let progress = (offset + 1) as f32 / steps;
        *value = (log_left + (log_right - log_left) * progress).exp();
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_all_off() -> F0PostprocessConfig {
        F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            interpolate_internal_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        }
    }

    fn run(cfg: F0PostprocessConfig, input: &[f32], shift: f32) -> Vec<f32> {
        let mut p = F0Postprocessor::new(cfg);
        let mut out = Vec::new();
        p.process_pitchf_into(input, shift, &mut out);
        out
    }

    fn validate(cfg: F0PostprocessConfig, input: &[f32], audio_16k: &[f32]) -> Vec<f32> {
        let processor = F0Postprocessor::new(cfg);
        let mut out = Vec::new();
        processor.validate_raw_pitchf_into(input, audio_16k, &mut out);
        out
    }

    fn sine_16k(frequency_hz: f32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|sample| {
                (std::f32::consts::TAU * frequency_hz * sample as f32 / 16_000.0).sin() * 0.2
            })
            .collect()
    }

    fn aperiodic_noise(samples: usize) -> Vec<f32> {
        let mut state = 0x6d2b_79f5u32;
        (0..samples)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.2
            })
            .collect()
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-2, "expected ~{b}, got {a}");
    }

    // 1. disabled mode: no smoothing, shift applied once, input untouched.
    #[test]
    fn disabled_passthrough_and_shift() {
        let cfg = F0PostprocessConfig::default(); // enabled = false
        let input = vec![0.0, 220.0, 0.0, 440.0];

        let out = run(cfg.clone(), &input, 0.0);
        assert_eq!(out, input, "shift 0 must be identity");

        let up = run(cfg.clone(), &input, 12.0);
        approx(up[1], 440.0);
        approx(up[3], 880.0);
        assert_eq!(up[0], 0.0);
        assert_eq!(up[2], 0.0);

        let down = run(cfg, &input, -12.0);
        approx(down[1], 110.0);
        approx(down[3], 220.0);
    }

    // 2. pitch shift applied exactly once (not double): raw 220 + 12 => 440.
    #[test]
    fn pitch_shift_single_application() {
        // Even with smoothing enabled, a clean steady tone shifts once.
        let out = run(F0PostprocessConfig::default(), &[220.0], 12.0);
        approx(out[0], 440.0);
        assert!((out[0] - 880.0).abs() > 1.0, "must not double-apply");

        let out_enabled = run(cfg_all_off(), &[220.0, 220.0, 220.0], 12.0);
        for v in out_enabled {
            approx(v, 440.0);
        }
    }

    // 3. invalid / out-of-range -> unvoiced.
    #[test]
    fn invalid_removal() {
        let cfg = cfg_all_off();
        let input = vec![f32::NAN, f32::INFINITY, -10.0, 0.0, 30.0, 2000.0, 220.0];
        let out = run(cfg, &input, 0.0);
        assert_eq!(out[0], 0.0); // NaN
        assert_eq!(out[1], 0.0); // inf
        assert_eq!(out[2], 0.0); // negative
        assert_eq!(out[3], 0.0); // zero
        assert_eq!(out[4], 0.0); // below min (50)
        assert_eq!(out[5], 0.0); // above max (1100)
        approx(out[6], 220.0); // in range
    }

    // 4. short voiced island removal, edges preserved.
    #[test]
    fn short_voiced_island_removal() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: true,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        assert_eq!(
            run(cfg.clone(), &[0.0, 0.0, 220.0, 0.0, 0.0], 0.0),
            vec![0.0; 5]
        );
        // Leading/trailing voiced runs touch the edge and are kept.
        let edge = run(cfg, &[220.0, 0.0, 0.0, 220.0], 0.0);
        approx(edge[0], 220.0);
        approx(edge[3], 220.0);
    }

    // 5. short unvoiced gap fill (log-linear), edges not filled.
    #[test]
    fn short_unvoiced_gap_fill() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: true,
            fix_octave_jumps: false,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        let out = run(cfg.clone(), &[220.0, 0.0, 0.0, 240.0], 0.0);
        let (ll, lr) = (220.0_f32.ln(), 240.0_f32.ln());
        approx(out[1], (ll + (lr - ll) * (1.0 / 3.0)).exp());
        approx(out[2], (ll + (lr - ll) * (2.0 / 3.0)).exp());

        // Leading/trailing gaps stay unvoiced.
        let edge = run(cfg, &[0.0, 220.0, 240.0, 0.0], 0.0);
        assert_eq!(edge[0], 0.0);
        assert_eq!(edge[3], 0.0);
    }

    #[test]
    fn continuity_repairs_short_internal_gaps_in_log_space_but_not_edges() {
        let cfg = F0PostprocessConfig::continuity(true);
        let out = run(cfg, &[0.0, 100.0, 0.0, 0.0, 110.0, 0.0], 0.0);
        let (left, right) = (100.0_f32.ln(), 110.0_f32.ln());
        assert_eq!(out[0], 0.0);
        approx(out[2], (left + (right - left) / 3.0).exp());
        approx(out[3], (left + (right - left) * 2.0 / 3.0).exp());
        assert_eq!(out[5], 0.0);
    }

    #[test]
    fn continuity_requires_stable_bounds_for_medium_gaps() {
        let cfg = F0PostprocessConfig::continuity(true);
        let stable = run(
            cfg.clone(),
            &[100.0, 100.0, 0.0, 0.0, 0.0, 105.0, 105.0],
            0.0,
        );
        assert!(stable[2..5].iter().all(|f0| *f0 > 0.0));

        let unstable = run(cfg, &[100.0, 130.0, 0.0, 0.0, 0.0, 200.0, 160.0], 0.0);
        assert_eq!(&unstable[2..5], &[0.0; 3]);
    }

    #[test]
    fn continuity_never_voices_a_long_pause() {
        let cfg = F0PostprocessConfig::continuity(true);
        let out = run(
            cfg,
            &[100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 105.0, 105.0],
            0.0,
        );
        assert_eq!(&out[2..8], &[0.0; 6]);
    }

    #[test]
    fn disabled_continuity_is_an_identity_without_pitch_shift() {
        let input = [100.0, 0.0, 500.0];
        assert_eq!(
            run(F0PostprocessConfig::continuity(false), &input, 0.0),
            input
        );
    }

    #[test]
    fn waveform_periodicity_keeps_supported_pitch_and_rejects_mismatch() {
        let cfg = F0PostprocessConfig::continuity_with_stabilization(false, true);
        let audio = sine_16k(200.0, 1_600);
        let supported = validate(cfg.clone(), &[200.0; 10], &audio);
        assert!(supported.iter().all(|f0| *f0 == 200.0));

        let mismatched = validate(cfg, &[320.0; 10], &audio);
        assert!(mismatched.iter().all(|f0| *f0 == 0.0));
    }

    #[test]
    fn waveform_periodicity_rejects_pitch_on_flat_audio() {
        let cfg = F0PostprocessConfig::continuity_with_stabilization(false, true);
        let out = validate(cfg, &[220.0; 8], &[0.0; 1_280]);
        assert!(out.iter().all(|f0| *f0 == 0.0));
    }

    #[test]
    fn waveform_periodicity_rejects_aperiodic_noise_pitch() {
        let cfg = F0PostprocessConfig::continuity_with_stabilization(false, true);
        let out = validate(cfg, &[220.0; 12], &aperiodic_noise(1_920));
        assert!(out.iter().filter(|f0| **f0 == 0.0).count() >= 11);
    }

    #[test]
    fn disabled_waveform_validation_is_an_exact_passthrough() {
        let cfg = F0PostprocessConfig::continuity(true);
        let input = [f32::NAN, -20.0, 0.0, 220.0, 2_000.0];
        let out = validate(cfg, &input, &[0.0; 800]);
        assert!(out[0].is_nan());
        assert_eq!(&out[1..], &input[1..]);
    }

    // 6. octave jump correction, only for isolated near-2x/0.5x with close sides.
    #[test]
    fn octave_jump_correction() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: true,
            median_filter: false,
            ..F0PostprocessConfig::default()
        };
        let up = run(cfg.clone(), &[220.0, 221.0, 440.0, 222.0, 221.0], 0.0);
        approx(up[2], 220.0);
        let down = run(cfg.clone(), &[220.0, 221.0, 110.0, 222.0, 221.0], 0.0);
        approx(down[2], 220.0);
        // Left and right not close => not an octave error, keep as-is.
        let glide = run(cfg, &[220.0, 440.0, 330.0], 0.0);
        approx(glide[1], 440.0);
    }

    #[test]
    fn stabilized_policy_repairs_interior_and_trailing_octave_outliers() {
        let cfg = F0PostprocessConfig::continuity_with_stabilization(false, true);
        assert!(cfg.enabled);
        assert!(cfg.fix_octave_jumps);
        assert!(cfg.fix_trailing_octave_jumps);
        assert!(cfg.median_filter);
        assert!(cfg.fill_short_unvoiced_gaps);

        let out = run(cfg, &[220.0, 220.0, 440.0, 220.0, 440.0], 0.0);
        approx(out[2], 220.0);
        approx(out[4], 220.0);
    }

    #[test]
    fn stabilized_policy_keeps_a_real_sustained_octave_change() {
        let cfg = F0PostprocessConfig::continuity_with_stabilization(false, true);
        let out = run(cfg, &[220.0, 220.0, 440.0, 440.0, 440.0], 0.0);
        approx(out[0], 220.0);
        approx(out[1], 220.0);
        approx(out[2], 440.0);
        approx(out[3], 440.0);
        approx(out[4], 440.0);
    }

    // 7. median filter: stable 3-point, unvoiced not mixed, order-independent.
    #[test]
    fn median_filter_behaviour() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: true,
            median_filter_radius: 1,
            ..F0PostprocessConfig::default()
        };
        // Single spike between equal neighbours -> pulled to the neighbour value.
        let out = run(cfg.clone(), &[200.0, 800.0, 200.0], 0.0);
        approx(out[1], 200.0);
        // Order independence: the corrected [1] must not feed [2]'s window.
        // input snapshot keeps [2]'s window = {800(orig),200} plus its center.
        let series = run(cfg.clone(), &[300.0, 300.0, 600.0, 300.0, 300.0], 0.0);
        approx(series[2], 300.0);
        // Unvoiced stays unvoiced and is not averaged in.
        let with_unvoiced = run(cfg, &[0.0, 220.0, 0.0], 0.0);
        assert_eq!(with_unvoiced[0], 0.0);
        assert_eq!(with_unvoiced[2], 0.0);
        approx(with_unvoiced[1], 220.0);
    }

    // post-shift clamp saturates (does not zero) out-of-range voiced frames.
    #[test]
    fn post_shift_clamp_saturates() {
        let cfg = F0PostprocessConfig {
            enabled: true,
            remove_short_voiced_islands: false,
            fill_short_unvoiced_gaps: false,
            fix_octave_jumps: false,
            median_filter: false,
            clamp_after_pitch_shift: true,
            ..F0PostprocessConfig::default()
        };
        // 1000 Hz shifted up an octave -> 2000 Hz, saturated to max (1100), kept voiced.
        let out = run(cfg, &[1000.0], 12.0);
        approx(out[0], 1100.0);
    }
}

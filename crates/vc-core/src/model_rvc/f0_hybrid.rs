//! Reliability fusion for the optional RMVPE + FCPE pitch path.
//!
//! The community FCPE ONNX exposes thresholded F0 only, not its raw voicing
//! confidence. Fusion therefore uses agreement, waveform periodicity, and local
//! contour continuity instead of pretending that zero/nonzero is a calibrated
//! probability. Keep this worker-side and allocation-reusing: model inference,
//! fusion, and all scratch buffers must stay out of the audio callback.

use anyhow::{bail, Result};

use super::f0_postprocess::{pitch_distance_semitones, waveform_periodicity_16k};

const UNVOICED: f32 = 0.0;
const MIN_PLAUSIBLE_F0_HZ: f32 = 35.0;
const MAX_PLAUSIBLE_F0_HZ: f32 = 1_600.0;
const AGREEMENT_SEMITONES: f32 = 0.75;
const NEIGHBOR_SEMITONES: f32 = 1.75;
const CONTINUITY_SEMITONES: f32 = 3.0;
const DISAGREEMENT_PERIODICITY_MARGIN: f32 = 0.07;
const MIN_DISAGREEMENT_PERIODICITY: f32 = 0.52;
const STRONG_LONE_PERIODICITY: f32 = 0.70;
const RMVPE_LONE_PERIODICITY: f32 = 0.55;
const FCPE_LONE_PERIODICITY: f32 = 0.60;
// The reference streaming implementation reports roughly 32 future 10 ms
// frames before FCPE predictions become fully settled. vc-rs deliberately does
// not add 320 ms of hidden latency, so agreement remains usable at the right
// edge while a conflicting/lone FCPE candidate needs stronger evidence there.
const FCPE_UNSETTLED_TAIL_FRAMES: usize = 32;

#[derive(Default)]
pub(super) struct HybridF0Fusion {
    fused: Vec<f32>,
}

impl HybridF0Fusion {
    pub(super) fn fuse<'a>(
        &'a mut self,
        rmvpe: &[f32],
        fcpe: &[f32],
        audio_16k: &[f32],
    ) -> Result<&'a [f32]> {
        if rmvpe.len() != fcpe.len() {
            bail!(
                "RMVPE/FCPE frame mismatch on the shared 10 ms grid: rmvpe={} fcpe={}",
                rmvpe.len(),
                fcpe.len()
            );
        }

        self.fused.clear();
        self.fused
            .reserve(rmvpe.len().saturating_sub(self.fused.capacity()));
        let mut last_voiced = UNVOICED;
        let mut unvoiced_run = 0usize;

        for frame in 0..rmvpe.len() {
            let rmvpe_f0 = normalize_candidate(rmvpe[frame]);
            let fcpe_f0 = normalize_candidate(fcpe[frame]);
            let fcpe_unsettled = rmvpe.len().saturating_sub(frame) <= FCPE_UNSETTLED_TAIL_FRAMES;
            let continuity_reference =
                (last_voiced > UNVOICED && unvoiced_run <= 2).then_some(last_voiced);

            let fused = match (is_voiced(rmvpe_f0), is_voiced(fcpe_f0)) {
                (true, true)
                    if pitch_distance_semitones(rmvpe_f0, fcpe_f0) <= AGREEMENT_SEMITONES =>
                {
                    log_hz_mean(rmvpe_f0, fcpe_f0)
                }
                (true, true) => choose_disagreement(
                    audio_16k,
                    frame,
                    rmvpe_f0,
                    fcpe_f0,
                    continuity_reference,
                    fcpe_unsettled,
                ),
                (true, false) => choose_lone_candidate(
                    audio_16k,
                    frame,
                    rmvpe_f0,
                    false,
                    continuity_reference,
                    has_neighbor_support(rmvpe, fcpe, frame, rmvpe_f0),
                    false,
                ),
                (false, true) => choose_lone_candidate(
                    audio_16k,
                    frame,
                    fcpe_f0,
                    true,
                    continuity_reference,
                    has_neighbor_support(rmvpe, fcpe, frame, fcpe_f0),
                    fcpe_unsettled,
                ),
                (false, false) => UNVOICED,
            };

            self.fused.push(fused);
            if is_voiced(fused) {
                last_voiced = fused;
                unvoiced_run = 0;
            } else {
                unvoiced_run = unvoiced_run.saturating_add(1);
            }
        }

        Ok(self.fused.as_slice())
    }
}

fn choose_disagreement(
    audio_16k: &[f32],
    frame: usize,
    rmvpe: f32,
    fcpe: f32,
    previous: Option<f32>,
    fcpe_unsettled: bool,
) -> f32 {
    if let Some(previous) = previous {
        let rmvpe_distance = pitch_distance_semitones(previous, rmvpe);
        let fcpe_distance = pitch_distance_semitones(previous, fcpe);
        if rmvpe_distance <= CONTINUITY_SEMITONES
            && fcpe_distance > rmvpe_distance + AGREEMENT_SEMITONES
        {
            return rmvpe;
        }
        if fcpe_distance <= CONTINUITY_SEMITONES
            && rmvpe_distance > fcpe_distance + AGREEMENT_SEMITONES
        {
            return fcpe;
        }
    }

    let rmvpe_periodicity = waveform_periodicity_16k(audio_16k, frame, rmvpe).unwrap_or(0.0);
    let fcpe_periodicity = waveform_periodicity_16k(audio_16k, frame, fcpe).unwrap_or(0.0);
    if rmvpe_periodicity >= MIN_DISAGREEMENT_PERIODICITY
        && rmvpe_periodicity >= fcpe_periodicity + DISAGREEMENT_PERIODICITY_MARGIN
    {
        return rmvpe;
    }
    if !fcpe_unsettled
        && fcpe_periodicity >= MIN_DISAGREEMENT_PERIODICITY
        && fcpe_periodicity >= rmvpe_periodicity + DISAGREEMENT_PERIODICITY_MARGIN
    {
        return fcpe;
    }

    // At the right edge FCPE has insufficient future context. Preserve a
    // waveform-supported RMVPE contour there instead of allowing an unstable
    // FCPE disagreement to punch a long unvoiced gap into the current chunk.
    if fcpe_unsettled && rmvpe_periodicity >= RMVPE_LONE_PERIODICITY {
        return rmvpe;
    }
    UNVOICED
}

fn choose_lone_candidate(
    audio_16k: &[f32],
    frame: usize,
    candidate: f32,
    is_fcpe: bool,
    previous: Option<f32>,
    neighbor_support: bool,
    fcpe_unsettled: bool,
) -> f32 {
    let periodicity = waveform_periodicity_16k(audio_16k, frame, candidate).unwrap_or(0.0);
    let continuous = previous.is_some_and(|previous| {
        pitch_distance_semitones(previous, candidate) <= CONTINUITY_SEMITONES
    });
    let minimum = if is_fcpe {
        FCPE_LONE_PERIODICITY + if fcpe_unsettled { 0.08 } else { 0.0 }
    } else {
        RMVPE_LONE_PERIODICITY
    };
    if periodicity >= minimum
        && (continuous || neighbor_support || periodicity >= STRONG_LONE_PERIODICITY)
    {
        candidate
    } else {
        UNVOICED
    }
}

fn has_neighbor_support(rmvpe: &[f32], fcpe: &[f32], frame: usize, candidate: f32) -> bool {
    let start = frame.saturating_sub(1);
    let end = (frame + 2).min(rmvpe.len());
    (start..end).filter(|&index| index != frame).any(|index| {
        [rmvpe[index], fcpe[index]].into_iter().any(|neighbor| {
            let neighbor = normalize_candidate(neighbor);
            is_voiced(neighbor)
                && pitch_distance_semitones(candidate, neighbor) <= NEIGHBOR_SEMITONES
        })
    })
}

fn normalize_candidate(value: f32) -> f32 {
    if value.is_finite() && (MIN_PLAUSIBLE_F0_HZ..=MAX_PLAUSIBLE_F0_HZ).contains(&value) {
        value
    } else {
        UNVOICED
    }
}

fn is_voiced(value: f32) -> bool {
    value > UNVOICED
}

fn log_hz_mean(left: f32, right: f32) -> f32 {
    (0.5 * (left.ln() + right.ln())).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f32, frames: usize) -> Vec<f32> {
        let samples = frames * 160 + 320;
        (0..samples)
            .map(|sample| {
                (std::f32::consts::TAU * frequency * sample as f32 / 16_000.0).sin() * 0.2
            })
            .collect()
    }

    #[test]
    fn voiced_agreement_is_averaged_in_log_hz() {
        let mut fusion = HybridF0Fusion::default();
        let out = fusion.fuse(&[220.0], &[222.0], &[]).unwrap();
        assert!((out[0] - (220.0f32 * 222.0).sqrt()).abs() < 0.001);
    }

    #[test]
    fn non_finite_and_nonpositive_fcpe_are_unvoiced() {
        let mut fusion = HybridF0Fusion::default();
        let out = fusion
            .fuse(&[0.0, 0.0, 0.0], &[f32::NAN, -1.0, f32::INFINITY], &[])
            .unwrap();
        assert_eq!(out, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn continuity_resolves_an_octave_disagreement() {
        let mut fusion = HybridF0Fusion::default();
        let out = fusion
            .fuse(
                &[220.0, 220.0, 440.0, 220.0],
                &[220.0, 220.0, 220.0, 220.0],
                &sine(220.0, 4),
            )
            .unwrap();
        assert!((out[2] - 220.0).abs() < 0.001);
    }

    #[test]
    fn supported_single_backend_dropout_stays_voiced() {
        let mut fusion = HybridF0Fusion::default();
        let out = fusion
            .fuse(
                &[220.0, 220.0, 220.0, 220.0],
                &[220.0, 0.0, 220.0, 220.0],
                &sine(220.0, 4),
            )
            .unwrap();
        assert!(out[1] > 0.0);
    }

    #[test]
    fn unsupported_lone_pitch_on_silence_is_rejected() {
        let mut fusion = HybridF0Fusion::default();
        let out = fusion
            .fuse(&[220.0, 220.0, 220.0], &[0.0, 0.0, 0.0], &[0.0; 960])
            .unwrap();
        assert_eq!(out, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn mismatched_frame_grids_are_rejected() {
        let error = HybridF0Fusion::default()
            .fuse(&[220.0], &[220.0, 220.0], &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("frame mismatch"));
    }
}

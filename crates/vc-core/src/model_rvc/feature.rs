use anyhow::{bail, Context, Result};

#[derive(Default)]
pub(super) struct FeatureTensor {
    pub(super) data: Vec<f32>,
    pub(super) shape: Vec<i64>,
}

impl FeatureTensor {
    /// Preserve the current tensor before RVC index retrieval mutates it. Both
    /// vectors retain capacity across chunks, so enabling consonant protection
    /// does not add a per-chunk allocation to the worker path.
    pub(super) fn copy_from(&mut self, source: &Self) {
        self.data.clear();
        self.data.extend_from_slice(&source.data);
        self.shape.clear();
        self.shape.extend_from_slice(&source.shape);
    }

    pub(super) fn repeat_frames(&mut self, factor: usize) -> Result<()> {
        if factor <= 1 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        // Repeat each frame `factor` times in place to reuse the buffer
        // allocated by `extract` instead of building a fresh Vec every chunk.
        // Walk frames back-to-front: frame `f`'s destination blocks start at
        // `f * factor * channels >= f * channels`, so a backward pass never
        // overwrites a source frame that has not been copied yet.
        let old_len = self.data.len();
        self.data.resize(old_len * factor, 0.0);
        for frame in (0..frames).rev() {
            let src = frame * channels;
            for repeat in (0..factor).rev() {
                let dst = (frame * factor + repeat) * channels;
                self.data.copy_within(src..src + channels, dst);
            }
        }
        self.shape[1] = (frames * factor) as i64;
        Ok(())
    }

    pub(super) fn trim_front_frames(&mut self, frames_to_drop: usize) -> Result<()> {
        if frames_to_drop == 0 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        if frames_to_drop >= frames {
            return Ok(());
        }
        let sample_offset = frames_to_drop * channels;
        self.data.drain(..sample_offset);
        self.shape[1] = (frames - frames_to_drop) as i64;
        Ok(())
    }

    /// Blend retrieved features with their pre-retrieval originals on unvoiced
    /// frames. This is RVC's `protect` behavior: voiced frames use the indexed
    /// result, while F0 <= 0 retains `protect` of the indexed frame and
    /// `1 - protect` of the original ContentVec frame. It runs after the normal
    /// two-times frame expansion, so its F0 mask shares the generator's 10 ms
    /// grid exactly.
    ///
    /// `transition_frames == 0` is the exact standard-RVC binary mask. A
    /// positive value eases *adjacent voiced frames* from the unvoiced protect
    /// weight back to full retrieval. The unvoiced frame itself stays fully
    /// protected, which avoids trading consonant clarity for a smoother feature
    /// boundary.
    #[allow(dead_code)]
    pub(super) fn protect_unvoiced_frames(
        &mut self,
        original: &Self,
        natural_pitchf: &[f32],
        protect: f32,
        transition_frames: usize,
    ) -> Result<()> {
        self.protect_unvoiced_frames_with_adaptive(
            original,
            natural_pitchf,
            protect,
            transition_frames,
            None,
            0,
        )
    }

    /// Apply Protect with optional per-ContentVec-frame confidence scales from
    /// adaptive index retrieval. The scale is intentionally limited to
    /// unvoiced frames: voiced transition easing already has its own temporal
    /// rule, and multiplying it by retrieval confidence would make sustained
    /// vowels lose too much target timbre. `repeated_frame_offset` is measured
    /// on the final 10 ms grid and accounts for the silence-front crop applied
    /// after ContentVec's 2x expansion.
    pub(super) fn protect_unvoiced_frames_with_adaptive(
        &mut self,
        original: &Self,
        natural_pitchf: &[f32],
        protect: f32,
        transition_frames: usize,
        adaptive_protect_scales: Option<&[f32]>,
        repeated_frame_offset: usize,
    ) -> Result<()> {
        // Upstream RVC reserves 0.5 as the disabled state. Keeping that exact
        // boundary matters for compatibility with its GUI/config presets.
        if !protect.is_finite() || protect >= 0.5 {
            return Ok(());
        }
        if self.shape.len() != 3 || original.shape != self.shape {
            bail!("retrieved and original feature tensors must have the same rank-3 shape");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1
            || self.data.len() != frames * channels
            || original.data.len() != self.data.len()
        {
            bail!("invalid feature tensor data for RVC protect");
        }
        if natural_pitchf.len() != frames {
            bail!(
                "RVC protect pitch frame count {} does not match feature frame count {}",
                natural_pitchf.len(),
                frames
            );
        }
        let protect = protect.clamp(0.0, 0.5);
        for (frame_index, ((retrieved, original), pitchf)) in self
            .data
            .chunks_exact_mut(channels)
            .zip(original.data.chunks_exact(channels))
            .zip(natural_pitchf)
            .enumerate()
        {
            let retrieved_weight = if is_unvoiced(*pitchf) {
                let raw_frame_index = frame_index.saturating_add(repeated_frame_offset) / 2;
                let adaptive_scale = adaptive_protect_scales
                    .and_then(|scales| scales.get(raw_frame_index))
                    .copied()
                    .filter(|scale| scale.is_finite())
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                protect * adaptive_scale
            } else {
                voiced_retrieval_weight(natural_pitchf, frame_index, protect, transition_frames)
            };
            if retrieved_weight < 1.0 {
                let original_weight = 1.0 - retrieved_weight;
                for (retrieved, original) in retrieved.iter_mut().zip(original) {
                    *retrieved = *retrieved * retrieved_weight + *original * original_weight;
                }
            }
        }
        Ok(())
    }
}

fn is_unvoiced(pitchf: f32) -> bool {
    !pitchf.is_finite() || pitchf <= 0.0
}

/// Return the indexed-feature share for a voiced frame near an unvoiced run.
///
/// The lookup is deliberately bounded by `transition_frames`, so the worker
/// does a fixed small amount of extra work per 10 ms feature frame and needs no
/// extra scratch allocation. A denominator of `frames + 1` makes the step from
/// an unvoiced frame no larger than any later ramp step.
fn voiced_retrieval_weight(
    natural_pitchf: &[f32],
    frame_index: usize,
    protect: f32,
    transition_frames: usize,
) -> f32 {
    if transition_frames == 0 {
        return 1.0;
    }

    for distance in 1..=transition_frames {
        let has_unvoiced_left =
            frame_index >= distance && is_unvoiced(natural_pitchf[frame_index - distance]);
        let has_unvoiced_right = natural_pitchf
            .get(frame_index + distance)
            .is_some_and(|pitchf| is_unvoiced(*pitchf));
        if has_unvoiced_left || has_unvoiced_right {
            let progress = distance as f32 / (transition_frames + 1) as f32;
            return protect + (1.0 - protect) * progress;
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_frames_duplicates_each_frame_in_order() {
        // 3 frames x 2 channels.
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![1, 3, 2],
        };
        tensor.repeat_frames(2).unwrap();
        assert_eq!(
            tensor.data,
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(tensor.shape, vec![1, 6, 2]);
    }

    #[test]
    fn repeat_frames_factor_three() {
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2],
        };
        tensor.repeat_frames(3).unwrap();
        assert_eq!(tensor.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(tensor.shape, vec![1, 3, 2]);
    }

    #[test]
    fn repeat_frames_factor_one_is_noop() {
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
        };
        tensor.repeat_frames(1).unwrap();
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.shape, vec![1, 2, 2]);
    }

    #[test]
    fn protect_keeps_original_features_on_unvoiced_frames() {
        let original = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
        };
        let mut retrieved = FeatureTensor {
            data: vec![10.0, 20.0, 30.0, 40.0],
            shape: vec![1, 2, 2],
        };
        retrieved
            .protect_unvoiced_frames(&original, &[100.0, 0.0], 0.25, 0)
            .unwrap();
        assert_eq!(retrieved.data, vec![10.0, 20.0, 9.75, 13.0]);
    }

    #[test]
    fn protect_half_is_the_standard_rvc_disabled_value() {
        let original = FeatureTensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2],
        };
        let mut retrieved = FeatureTensor {
            data: vec![10.0, 20.0],
            shape: vec![1, 1, 2],
        };
        retrieved
            .protect_unvoiced_frames(&original, &[0.0], 0.5, 0)
            .unwrap();
        assert_eq!(retrieved.data, vec![10.0, 20.0]);
    }

    #[test]
    fn protect_transition_eases_only_adjacent_voiced_frames() {
        let original = FeatureTensor {
            data: vec![0.0; 6],
            shape: vec![1, 6, 1],
        };
        let mut retrieved = FeatureTensor {
            data: vec![1.0; 6],
            shape: vec![1, 6, 1],
        };

        retrieved
            .protect_unvoiced_frames(&original, &[100.0, 100.0, 0.0, 100.0, 100.0, 100.0], 0.0, 2)
            .unwrap();

        let expected = [2.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (actual, expected) in retrieved.data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn zero_protect_transition_keeps_the_standard_binary_mask() {
        let original = FeatureTensor {
            data: vec![0.0; 3],
            shape: vec![1, 3, 1],
        };
        let mut retrieved = FeatureTensor {
            data: vec![1.0; 3],
            shape: vec![1, 3, 1],
        };

        retrieved
            .protect_unvoiced_frames(&original, &[100.0, 0.0, 100.0], 0.0, 0)
            .unwrap();

        assert_eq!(retrieved.data, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn adaptive_protect_maps_repeated_frames_to_raw_content_frames() {
        let original = FeatureTensor {
            data: vec![0.0; 4],
            shape: vec![1, 4, 1],
        };
        let mut retrieved = FeatureTensor {
            data: vec![1.0; 4],
            shape: vec![1, 4, 1],
        };

        // Frames 0/1 map to raw ContentVec frame 0, while frames 2/3 map to
        // raw frame 1. A one-frame front crop shifts that mapping by one 10 ms
        // slot and is intentionally included in this regression test.
        retrieved
            .protect_unvoiced_frames_with_adaptive(
                &original,
                &[0.0, 0.0, 0.0, 0.0],
                0.5 - f32::EPSILON,
                0,
                Some(&[0.2, 0.8]),
                1,
            )
            .unwrap();

        // `(frame + 1) / 2` maps [0, 1, 2, 3] to [0, 1, 1, 2]; missing raw
        // frame 2 falls back to an unscaled Protect value of 0.5.
        let expected = [0.1, 0.4, 0.4, 0.5];
        for (actual, expected) in retrieved.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }
}

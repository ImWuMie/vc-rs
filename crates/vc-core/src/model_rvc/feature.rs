use anyhow::{bail, Context, Result};

#[derive(Default)]
pub(super) struct FeatureTensor {
    pub(super) data: Vec<f32>,
    pub(super) shape: Vec<i64>,
}

impl FeatureTensor {
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
}

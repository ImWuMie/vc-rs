//! Official DeepFilterNet3 streaming denoiser adapter.
//!
//! The upstream `libDF` runtime owns the STFT, three ONNX graphs, recurrent
//! state, deep filtering, and iSTFT. This module only validates its streaming
//! contract and adapts one 48 kHz hop to the shared [`FixedDelayAdapter`]. The
//! model archive is supplied separately so vc-rs never embeds or silently
//! substitutes third-party weights.

use std::path::Path;

use anyhow::{bail, Context, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{ArrayView2, ArrayViewMut2};

use crate::denoise_config::{MAX_DFN3_ATTENUATION_LIMIT_DB, MAX_DFN3_POST_FILTER_BETA};

use super::{FixedDelayAdapter, FrameDenoiser};

#[derive(Clone, Copy, Debug)]
pub struct DeepFilterNet3Config<'a> {
    pub model_path: &'a Path,
    /// Maximum noise attenuation. `0` is effectively bypass; `100` is full
    /// model suppression. 12-24 dB is usually safer before voice conversion.
    pub attenuation_limit_db: f32,
    /// Upstream post-filter beta. Zero disables it; values up to 0.1 are useful.
    pub post_filter_beta: f32,
}

struct DeepFilterNet3FrameProcessor {
    model: DfTract,
    // `DfTract::init` resets spectral buffers but not all Tract recurrent
    // states. A pristine clone is the only complete, inexpensive stream reset.
    pristine: DfTract,
    input_scratch: Vec<f32>,
    sample_rate: usize,
    frame_size: usize,
    delay_frames: usize,
}

impl DeepFilterNet3FrameProcessor {
    fn new(config: DeepFilterNet3Config<'_>) -> Result<Self> {
        if !config.attenuation_limit_db.is_finite()
            || !(0.0..=MAX_DFN3_ATTENUATION_LIMIT_DB).contains(&config.attenuation_limit_db)
        {
            bail!(
                "DeepFilterNet3 attenuation limit must be finite and in 0..={MAX_DFN3_ATTENUATION_LIMIT_DB} dB"
            );
        }
        if !config.post_filter_beta.is_finite()
            || !(0.0..=MAX_DFN3_POST_FILTER_BETA).contains(&config.post_filter_beta)
        {
            bail!(
                "DeepFilterNet3 post-filter beta must be finite and in 0..={MAX_DFN3_POST_FILTER_BETA}"
            );
        }
        if !config.model_path.is_file() {
            bail!(
                "DeepFilterNet3 model archive not found: {}",
                config.model_path.display()
            );
        }

        let runtime = RuntimeParams::default()
            .with_atten_lim(config.attenuation_limit_db)
            .with_post_filter(config.post_filter_beta);
        let params = DfParams::new(config.model_path.to_path_buf()).with_context(|| {
            format!(
                "failed to read DeepFilterNet3 model archive {}",
                config.model_path.display()
            )
        })?;
        let model =
            DfTract::new(params, &runtime).context("failed to initialize DeepFilterNet3")?;
        if model.sr == 0 || model.hop_size == 0 || model.fft_size < model.hop_size {
            bail!(
                "invalid DeepFilterNet3 streaming shape: sr={} fft={} hop={}",
                model.sr,
                model.fft_size,
                model.hop_size
            );
        }
        let reconstruction_samples = model.fft_size - model.hop_size;
        let delay_frames = reconstruction_samples.div_ceil(model.hop_size) + model.lookahead;
        let sample_rate = model.sr;
        let frame_size = model.hop_size;
        let pristine = model.clone();
        Ok(Self {
            model,
            pristine,
            input_scratch: vec![0.0; frame_size],
            sample_rate,
            frame_size,
            delay_frames,
        })
    }
}

impl FrameDenoiser for DeepFilterNet3FrameProcessor {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn frame_size(&self) -> usize {
        self.frame_size
    }

    fn output_delay_frames(&self) -> usize {
        self.delay_frames
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), self.frame_size);
        debug_assert_eq!(output.len(), self.frame_size);
        for (dst, src) in self.input_scratch.iter_mut().zip(input) {
            *dst = if src.is_finite() { *src } else { 0.0 };
        }
        let noisy = ArrayView2::from_shape((1, self.frame_size), &self.input_scratch)
            .context("invalid DeepFilterNet3 input frame")?;
        let enhanced = ArrayViewMut2::from_shape((1, self.frame_size), output)
            .context("invalid DeepFilterNet3 output frame")?;
        self.model
            .process(noisy, enhanced)
            .context("DeepFilterNet3 frame inference failed")?;
        for sample in output {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // This runs on the conversion worker only during a route/mode reset,
        // never on an audio callback. Cloning preserves loaded plans/weights
        // while replacing every recurrent tensor and spectral history.
        self.model = self.pristine.clone();
        self.input_scratch.fill(0.0);
        Ok(())
    }
}

pub struct DeepFilterNet3Denoiser {
    inner: FixedDelayAdapter<DeepFilterNet3FrameProcessor>,
}

// `DfTract` stores Tract tensors in `Rc` and therefore does not automatically
// implement Send. A denoiser is nevertheless moved, never shared: the loader
// constructs it before publication, then ownership crosses once into the sole
// conversion worker and all processing/reset calls require `&mut self`. Do not
// add Sync or expose cloned model state; that would invalidate this confinement
// argument and make non-atomic Tract internals unsafe.
unsafe impl Send for DeepFilterNet3Denoiser {}

impl DeepFilterNet3Denoiser {
    pub fn new(config: DeepFilterNet3Config<'_>, device_sample_rate: u32) -> Result<Self> {
        Ok(Self {
            inner: FixedDelayAdapter::new(
                DeepFilterNet3FrameProcessor::new(config)?,
                device_sample_rate,
            )?,
        })
    }

    pub fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    pub fn process_in_place(&mut self, samples: &mut [f32]) -> Result<()> {
        self.inner.process_in_place(samples)
    }

    pub fn process_finite(
        input: &[f32],
        sample_rate: u32,
        config: DeepFilterNet3Config<'_>,
    ) -> Result<Vec<f32>> {
        Self::new(config, sample_rate)?.inner.process_finite(input)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::denoise_config::{DEFAULT_DFN3_ATTENUATION_LIMIT_DB, DEFAULT_DFN3_POST_FILTER_BETA};

    #[test]
    #[ignore = "requires VC_RS_DFN3_MODEL pointing at an official DeepFilterNet3 archive"]
    fn official_archive_initializes_and_processes_audio() -> Result<()> {
        let model_path = std::env::var_os("VC_RS_DFN3_MODEL")
            .map(PathBuf::from)
            .expect("set VC_RS_DFN3_MODEL to an official DeepFilterNet3 .tar.gz archive");
        let mut denoiser = DeepFilterNet3Denoiser::new(
            DeepFilterNet3Config {
                model_path: &model_path,
                attenuation_limit_db: DEFAULT_DFN3_ATTENUATION_LIMIT_DB,
                post_filter_beta: DEFAULT_DFN3_POST_FILTER_BETA,
            },
            48_000,
        )?;
        assert!(denoiser.latency_samples() > 0);

        // Ten 20 ms device blocks exercise multiple native model hops while
        // retaining the same non-uniform worker-facing block shape used by the
        // realtime pipeline. The assertion deliberately checks only stream
        // safety, not a fragile model-quality threshold.
        let mut audio = (0..9_600)
            .map(|sample| (sample as f32 * 0.013).sin() * 0.05)
            .collect::<Vec<_>>();
        for block in audio.chunks_mut(960) {
            denoiser.process_in_place(block)?;
        }
        assert!(audio.iter().all(|sample| sample.is_finite()));
        denoiser.reset()?;
        Ok(())
    }
}

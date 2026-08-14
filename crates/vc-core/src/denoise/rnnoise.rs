//! RNNoise [`FrameDenoiser`] backed by the pure-Rust `nnnoiseless` port.
//!
//! RNNoise consumes 480-sample frames at 48 kHz in i16-scale `f32`. The PCM
//! scaling that the surrounding adapter used to do inline now lives here, so the
//! adapter stays model-agnostic. The fixed-delay streaming, resampling, and
//! frame accumulation are all in [`super::adapter`].

use anyhow::Result;
use nnnoiseless::DenoiseState;

use super::adapter::{FixedDelayAdapter, FrameDenoiser};

const RNNOISE_SAMPLE_RATE: usize = 48_000;
const PCM_SCALE: f32 = 32_768.0;

struct RnnoiseFrameProcessor {
    state: Box<DenoiseState<'static>>,
    scratch_in: [f32; DenoiseState::FRAME_SIZE],
    scratch_out: [f32; DenoiseState::FRAME_SIZE],
}

impl RnnoiseFrameProcessor {
    fn new() -> Self {
        let mut processor = Self {
            state: DenoiseState::new(),
            scratch_in: [0.0; DenoiseState::FRAME_SIZE],
            scratch_out: [0.0; DenoiseState::FRAME_SIZE],
        };
        processor.warmup();
        processor
    }

    fn warmup(&mut self) {
        // Discard a silent warmup frame, not the caller's first speech frame.
        // This avoids nnnoiseless' documented fade-in artifact without losing
        // the first 10 ms of real input.
        let silent = [0.0; DenoiseState::FRAME_SIZE];
        let mut output = [0.0; DenoiseState::FRAME_SIZE];
        self.state.process_frame(&mut output, &silent);
    }
}

impl FrameDenoiser for RnnoiseFrameProcessor {
    fn sample_rate(&self) -> usize {
        RNNOISE_SAMPLE_RATE
    }

    fn frame_size(&self) -> usize {
        DenoiseState::FRAME_SIZE
    }

    fn output_delay_frames(&self) -> usize {
        // RNNoise is sample-aligned: its output frame corresponds to the input
        // frame, with no overlap-add reconstruction delay.
        0
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        for (dst, src) in self.scratch_in.iter_mut().zip(input) {
            let finite = if src.is_finite() { *src } else { 0.0 };
            *dst = (finite.clamp(-1.0, 1.0) * PCM_SCALE).clamp(-32_768.0, 32_767.0);
        }
        self.state
            .process_frame(&mut self.scratch_out, &self.scratch_in);
        for (dst, src) in output.iter_mut().zip(&self.scratch_out) {
            *dst = if src.is_finite() {
                (*src / PCM_SCALE).clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = DenoiseState::new();
        self.warmup();
        Ok(())
    }
}

/// Fixed-delay RNNoise input denoiser. Thin wrapper preserving the historical
/// public API over the shared [`FixedDelayAdapter`].
pub struct RnnoiseDenoiser {
    inner: FixedDelayAdapter<RnnoiseFrameProcessor>,
}

impl RnnoiseDenoiser {
    pub fn new(sample_rate: u32) -> Result<Self> {
        Ok(Self {
            inner: FixedDelayAdapter::new(RnnoiseFrameProcessor::new(), sample_rate)?,
        })
    }

    pub fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    /// Process arbitrary device-rate input while preserving the per-call length.
    pub fn process_in_place(&mut self, samples: &mut [f32]) -> Result<()> {
        self.inner.process_in_place(samples)
    }

    /// Denoise finite input, remove the streaming delay, and preserve its length.
    pub fn process_finite(input: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let mut denoiser = Self::new(sample_rate)?;
        denoiser.inner.process_finite(input)
    }

    /// Discard fixed-delay and recurrent state without reconstructing the
    /// adapter. Route changes call this only on the conversion worker, never in
    /// the audio callback.
    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_calls_preserve_length_at_common_rates() {
        for rate in [44_100, 48_000, 96_000] {
            let mut denoiser = RnnoiseDenoiser::new(rate).unwrap();
            for len in [1, 127, 480, 777, 2205, 4800] {
                let mut audio = vec![0.0; len];
                denoiser.process_in_place(&mut audio).unwrap();
                assert_eq!(audio.len(), len);
                assert!(audio.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn finite_processing_preserves_length() {
        for rate in [44_100, 48_000, 96_000] {
            for len in [0, 1, 127, 1000, rate as usize / 3] {
                let input = vec![0.0; len];
                let output = RnnoiseDenoiser::process_finite(&input, rate).unwrap();
                assert_eq!(output.len(), input.len());
                assert!(output.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn chunk_partition_does_not_reset_rnnoise_state() {
        let input: Vec<f32> = (0..48_000).map(|i| 0.4 * (i as f32 * 0.04).sin()).collect();
        let mut whole = input.clone();
        RnnoiseDenoiser::new(48_000)
            .unwrap()
            .process_in_place(&mut whole)
            .unwrap();

        let mut split_denoiser = RnnoiseDenoiser::new(48_000).unwrap();
        let mut split = Vec::with_capacity(input.len());
        for chunk in input.chunks(777) {
            let mut output = chunk.to_vec();
            split_denoiser.process_in_place(&mut output).unwrap();
            split.extend_from_slice(&output);
        }
        assert_eq!(whole, split);
    }

    #[test]
    fn finite_processing_keeps_non_silent_signal() {
        let input: Vec<f32> = (0..48_000).map(|i| 0.4 * (i as f32 * 0.04).sin()).collect();
        let output = RnnoiseDenoiser::process_finite(&input, 48_000).unwrap();
        assert!(crate::dsp::rms(&output) > 0.01);
    }
}

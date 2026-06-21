//! Streaming STFT analysis + iSTFT synthesis for the GTCRN denoiser.
//!
//! GTCRN consumes one 257-bin complex spectrum per 256-sample hop at 16 kHz and
//! returns an enhanced spectrum of the same shape. We own the time↔frequency
//! transform here (the streaming graph only does the spectral mapping), matching
//! the model's training-time framing exactly:
//!
//! - `n_fft = 512`, `hop = 256`, `win_length = 512`
//! - analysis **and** synthesis windows are the **square-root Hann** window
//!   (`hann_window(512).pow(0.5)` in the upstream export). Their product is a
//!   periodic Hann window, which sums to a constant 1 at 50 % overlap (hop =
//!   n_fft / 2), so analysis→synthesis with an unmodified spectrum reconstructs
//!   the input exactly (the COLA condition), delayed by exactly one hop.
//!
//! These constants are GTCRN-training facts (Phase 0); changing them is an
//! audio-quality change, not cleanup. `realfft` does not normalize, so the
//! inverse is scaled by `1 / n_fft` here.

use std::sync::Arc;

use anyhow::Result;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// FFT size of the GTCRN streaming STFT.
pub(crate) const GTCRN_N_FFT: usize = 512;
/// Hop / frame size in samples (the [`FrameDenoiser`](super::adapter::FrameDenoiser)
/// frame size for GTCRN).
pub(crate) const GTCRN_HOP: usize = 256;
/// Number of one-sided frequency bins (`n_fft / 2 + 1`).
pub(crate) const GTCRN_BINS: usize = GTCRN_N_FFT / 2 + 1;
/// Inherent overlap-add reconstruction delay, in whole hops. With
/// `win_length − hop = 256 = 1 hop`, analysis→synthesis delays the signal by
/// exactly one frame; this is the STFT's contribution to `output_delay_frames`.
pub(crate) const GTCRN_RECON_DELAY_FRAMES: usize = 1;

/// Periodic square-root Hann window of length `n` (matches
/// `torch.hann_window(n).pow(0.5)`, which defaults to `periodic=True`).
fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            hann.max(0.0).sqrt()
        })
        .collect()
}

/// Streaming STFT/iSTFT pair holding the rolling analysis window and the
/// overlap-add accumulator. One instance maps a continuous 16 kHz mono stream to
/// per-hop spectra and back; state is continuous across calls.
pub(crate) struct StftStreamer {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    // Most recent `n_fft` input samples (the current analysis frame).
    frame: Vec<f32>,
    // `n_fft`-length overlap-add accumulator; its first hop is emitted each step.
    ola: Vec<f32>,
    // Scratch reused across calls so steady-state processing does not allocate.
    time_scratch: Vec<f32>,
    spectrum_scratch: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    inv_norm: f32,
}

impl StftStreamer {
    pub(crate) fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(GTCRN_N_FFT);
        let inverse = planner.plan_fft_inverse(GTCRN_N_FFT);
        let fft_scratch_len = forward.get_scratch_len().max(inverse.get_scratch_len());

        Self {
            analysis_window: sqrt_hann(GTCRN_N_FFT),
            synthesis_window: sqrt_hann(GTCRN_N_FFT),
            frame: vec![0.0; GTCRN_N_FFT],
            ola: vec![0.0; GTCRN_N_FFT],
            time_scratch: vec![0.0; GTCRN_N_FFT],
            spectrum_scratch: vec![Complex::new(0.0, 0.0); GTCRN_BINS],
            fft_scratch: vec![Complex::new(0.0, 0.0); fft_scratch_len],
            inv_norm: 1.0 / GTCRN_N_FFT as f32,
            forward,
            inverse,
        }
    }

    /// Push one hop of new samples, returning the windowed analysis spectrum
    /// (`GTCRN_BINS` bins) for the most recent `n_fft`-sample window.
    ///
    /// The returned slice is borrowed scratch; copy it (or its enhanced form)
    /// before the next call.
    pub(crate) fn analyze(&mut self, hop: &[f32]) -> Result<&[Complex<f32>]> {
        debug_assert_eq!(hop.len(), GTCRN_HOP);
        // Slide the analysis window left by one hop and append the new samples,
        // so each call advances the frame by exactly `hop`.
        self.frame.copy_within(GTCRN_HOP.., 0);
        let tail = &mut self.frame[GTCRN_N_FFT - GTCRN_HOP..];
        for (dst, src) in tail.iter_mut().zip(hop) {
            *dst = if src.is_finite() { *src } else { 0.0 };
        }
        for (dst, (sample, win)) in self
            .time_scratch
            .iter_mut()
            .zip(self.frame.iter().zip(&self.analysis_window))
        {
            *dst = sample * win;
        }
        self.forward.process_with_scratch(
            &mut self.time_scratch,
            &mut self.spectrum_scratch,
            &mut self.fft_scratch,
        )?;
        Ok(&self.spectrum_scratch)
    }

    /// Synthesize one hop of output from an enhanced spectrum via iFFT, the
    /// synthesis window, and overlap-add. `spectrum.len() == GTCRN_BINS`,
    /// `out_hop.len() == GTCRN_HOP`.
    pub(crate) fn synthesize(
        &mut self,
        spectrum: &[Complex<f32>],
        out_hop: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(spectrum.len(), GTCRN_BINS);
        debug_assert_eq!(out_hop.len(), GTCRN_HOP);
        self.spectrum_scratch.copy_from_slice(spectrum);
        // The inverse real FFT assumes the DC and Nyquist bins are purely real;
        // drop any imaginary part the model may have left there.
        if let Some(dc) = self.spectrum_scratch.first_mut() {
            dc.im = 0.0;
        }
        if let Some(nyq) = self.spectrum_scratch.last_mut() {
            nyq.im = 0.0;
        }
        self.inverse.process_with_scratch(
            &mut self.spectrum_scratch,
            &mut self.time_scratch,
            &mut self.fft_scratch,
        )?;
        for (acc, (sample, win)) in self
            .ola
            .iter_mut()
            .zip(self.time_scratch.iter().zip(&self.synthesis_window))
        {
            *acc += sample * win * self.inv_norm;
        }
        out_hop.copy_from_slice(&self.ola[..GTCRN_HOP]);
        // Shift the accumulator down by one hop; clear the freshly exposed tail.
        self.ola.copy_within(GTCRN_HOP.., 0);
        self.ola[GTCRN_N_FFT - GTCRN_HOP..].fill(0.0);
        Ok(())
    }

    /// Restore the post-construction state (silent history). FFT plans and
    /// windows are stateless, so only the rolling buffers reset.
    pub(crate) fn reset(&mut self) {
        self.frame.fill(0.0);
        self.ola.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32;
                0.5 * (t * 0.013).sin() + 0.25 * (t * 0.071 + 1.0).sin()
            })
            .collect()
    }

    #[test]
    fn sqrt_hann_product_satisfies_cola() {
        // analysis * synthesis = periodic Hann; overlapped halves sum to 1.
        let w = sqrt_hann(GTCRN_N_FFT);
        for i in 0..GTCRN_HOP {
            let sum = w[i] * w[i] + w[i + GTCRN_HOP] * w[i + GTCRN_HOP];
            assert!((sum - 1.0).abs() < 1e-5, "i={i} sum={sum}");
        }
    }

    #[test]
    fn bins_and_constants_match_gtcrn_contract() {
        assert_eq!(GTCRN_N_FFT, 512);
        assert_eq!(GTCRN_HOP, 256);
        assert_eq!(GTCRN_BINS, 257);
        assert_eq!(GTCRN_RECON_DELAY_FRAMES, 1);
    }

    #[test]
    fn identity_round_trip_reconstructs_after_one_hop_delay() {
        // analysis -> synthesis with an unmodified spectrum must reproduce the
        // input, delayed by exactly one hop (the COLA reconstruction delay).
        let hops = 64;
        let input = signal(hops * GTCRN_HOP);
        let mut stft = StftStreamer::new();
        let mut output = Vec::with_capacity(input.len());
        let mut spec = vec![Complex::new(0.0, 0.0); GTCRN_BINS];
        let mut out_hop = vec![0.0f32; GTCRN_HOP];

        for hop in input.chunks(GTCRN_HOP) {
            spec.copy_from_slice(stft.analyze(hop).unwrap());
            stft.synthesize(&spec, &mut out_hop).unwrap();
            output.extend_from_slice(&out_hop);
        }

        let delay = GTCRN_RECON_DELAY_FRAMES * GTCRN_HOP;
        // Compare the steady-state region (skip the first frame's priming and the
        // last partial frame that has not received its second overlap yet).
        let mut max_err = 0.0f32;
        for i in delay..input.len() - GTCRN_HOP {
            max_err = max_err.max((output[i] - input[i - delay]).abs());
        }
        assert!(max_err < 1e-4, "max reconstruction error {max_err}");
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn streaming_is_deterministic_and_reset_repeats_startup() {
        let input = signal(40 * GTCRN_HOP);

        let run = |stft: &mut StftStreamer| {
            let mut spec = vec![Complex::new(0.0, 0.0); GTCRN_BINS];
            let mut out_hop = vec![0.0f32; GTCRN_HOP];
            let mut out = Vec::with_capacity(input.len());
            for hop in input.chunks(GTCRN_HOP) {
                spec.copy_from_slice(stft.analyze(hop).unwrap());
                stft.synthesize(&spec, &mut out_hop).unwrap();
                out.extend_from_slice(&out_hop);
            }
            out
        };

        let mut a = StftStreamer::new();
        let first = run(&mut a);
        a.reset();
        let second = run(&mut a);
        let fresh = run(&mut StftStreamer::new());

        assert_eq!(first, second);
        assert_eq!(first, fresh);
    }
}

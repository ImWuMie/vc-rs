//! Low-latency WebRTC-style spectral noise suppression.
//!
//! This is intentionally an in-tree Rust implementation rather than the
//! `webrtc-audio-processing` C++ wrapper. That wrapper's bundled build assumes
//! a Unix Meson/Ninja toolchain and `.a` archives, which is not a dependable
//! dependency for the project's Windows/MSVC distributions. The processor
//! keeps the useful WebRTC NS shape: 10 ms frames, a decision-directed Wiener
//! gain, speech-presence-aware noise tracking, and conservative time/frequency
//! gain smoothing. All frame buffers are allocated at construction.

use std::sync::Arc;

use anyhow::Result;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::denoise_config::WebRtcSuppressionLevel;

use super::{FixedDelayAdapter, FrameDenoiser};

const SAMPLE_RATE: usize = 48_000;
const HOP: usize = SAMPLE_RATE / 100;
const N_FFT: usize = HOP * 2;
const BINS: usize = N_FFT / 2 + 1;
const RECON_DELAY_FRAMES: usize = 1;
const EPSILON: f32 = 1.0e-12;

impl WebRtcSuppressionLevel {
    fn minimum_gain(self) -> f32 {
        // Maximum attenuation of 6/12/18/24 dB respectively. A non-zero floor
        // is important before ContentVec: hard spectral holes damage consonants
        // more than a small residual noise bed does.
        match self {
            Self::Low => 0.501_187_2,
            Self::Moderate => 0.251_188_64,
            Self::High => 0.125_892_53,
            Self::VeryHigh => 0.063_095_73,
        }
    }
}

fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            hann.max(0.0).sqrt()
        })
        .collect()
}

struct WebRtcFrameProcessor {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    analysis_frame: Vec<f32>,
    ola: Vec<f32>,
    time_scratch: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    smoothed_power: Vec<f32>,
    noise_power: Vec<f32>,
    minimum_power: Vec<f32>,
    previous_posterior_snr: Vec<f32>,
    gain: Vec<f32>,
    target_gain: Vec<f32>,
    frequency_smoothed_gain: Vec<f32>,
    minimum_gain: f32,
    frame_count: usize,
}

impl WebRtcFrameProcessor {
    fn new(level: WebRtcSuppressionLevel) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(N_FFT);
        let inverse = planner.plan_fft_inverse(N_FFT);
        let fft_scratch_len = forward.get_scratch_len().max(inverse.get_scratch_len());
        Self {
            forward,
            inverse,
            window: sqrt_hann(N_FFT),
            analysis_frame: vec![0.0; N_FFT],
            ola: vec![0.0; N_FFT],
            time_scratch: vec![0.0; N_FFT],
            spectrum: vec![Complex::new(0.0, 0.0); BINS],
            fft_scratch: vec![Complex::new(0.0, 0.0); fft_scratch_len],
            smoothed_power: vec![EPSILON; BINS],
            noise_power: vec![EPSILON; BINS],
            minimum_power: vec![f32::MAX; BINS],
            previous_posterior_snr: vec![1.0; BINS],
            gain: vec![1.0; BINS],
            target_gain: vec![1.0; BINS],
            frequency_smoothed_gain: vec![1.0; BINS],
            minimum_gain: level.minimum_gain(),
            frame_count: 0,
        }
    }

    fn analyze(&mut self, input: &[f32]) -> Result<()> {
        self.analysis_frame.copy_within(HOP.., 0);
        for (dst, src) in self.analysis_frame[N_FFT - HOP..].iter_mut().zip(input) {
            *dst = if src.is_finite() { *src } else { 0.0 };
        }
        for (dst, (sample, window)) in self
            .time_scratch
            .iter_mut()
            .zip(self.analysis_frame.iter().zip(&self.window))
        {
            *dst = sample * window;
        }
        self.forward.process_with_scratch(
            &mut self.time_scratch,
            &mut self.spectrum,
            &mut self.fft_scratch,
        )?;
        Ok(())
    }

    fn suppress(&mut self) {
        // Reset the rolling minimum every second. Retaining a small bias above
        // the current smoothed spectrum lets the estimator follow a rising fan
        // or HVAC floor without interpreting stable vowels as noise instantly.
        if self.frame_count > 0 && self.frame_count.is_multiple_of(100) {
            for (minimum, smooth) in self.minimum_power.iter_mut().zip(&self.smoothed_power) {
                *minimum = (*smooth * 1.25).max(EPSILON);
            }
        }

        for bin in 0..BINS {
            let value = self.spectrum[bin];
            let power = (value.re * value.re + value.im * value.im).max(EPSILON);
            let smooth = if self.frame_count == 0 {
                power
            } else {
                0.8 * self.smoothed_power[bin] + 0.2 * power
            };
            self.smoothed_power[bin] = smooth;
            self.minimum_power[bin] = self.minimum_power[bin].min(smooth);

            if self.frame_count == 0 {
                self.noise_power[bin] = power;
            }
            let noise = self.noise_power[bin].max(EPSILON);
            let posterior = (power / noise).clamp(0.0, 1.0e4);
            let prior = (0.98 * self.gain[bin] * self.gain[bin] * self.previous_posterior_snr[bin]
                + 0.02 * (posterior - 1.0).max(0.0))
            .clamp(0.0, 1.0e4);

            let prior_db = 10.0 * (prior + EPSILON).log10();
            let speech_probability =
                (1.0 / (1.0 + (-(prior_db + 2.0) / 3.0).exp())).clamp(0.0, 1.0);
            let noise_alpha = 0.82 + 0.175 * speech_probability;
            let tracked_floor = (self.minimum_power[bin] * 1.5).max(EPSILON);
            let noise_observation = if speech_probability < 0.5 {
                power.min(tracked_floor)
            } else {
                tracked_floor
            };
            self.noise_power[bin] =
                (noise_alpha * noise + (1.0 - noise_alpha) * noise_observation).max(EPSILON);

            // Square-root Wiener gain is intentionally less destructive than
            // the power-domain form. Strong posterior-SNR bins receive extra
            // transient protection so plosives and fricatives stay intelligible.
            let mut target = (prior / (1.0 + prior)).sqrt().max(self.minimum_gain);
            if posterior > 20.0 {
                target = target.max(0.85);
            } else if posterior > 8.0 {
                target = target.max(0.60);
            }
            self.target_gain[bin] = target;
            self.previous_posterior_snr[bin] = posterior;
        }

        // A three-bin frequency smoother prevents isolated musical-noise tones.
        // DC and Nyquist keep their own estimates because they have one neighbor.
        self.frequency_smoothed_gain[0] = self.target_gain[0];
        for bin in 1..BINS - 1 {
            self.frequency_smoothed_gain[bin] = 0.25 * self.target_gain[bin - 1]
                + 0.5 * self.target_gain[bin]
                + 0.25 * self.target_gain[bin + 1];
        }
        self.frequency_smoothed_gain[BINS - 1] = self.target_gain[BINS - 1];

        for bin in 0..BINS {
            let target = self.frequency_smoothed_gain[bin];
            let smoothing = if target > self.gain[bin] { 0.35 } else { 0.82 };
            let gain = (smoothing * self.gain[bin] + (1.0 - smoothing) * target)
                .clamp(self.minimum_gain, 1.0);
            self.gain[bin] = gain;
            self.spectrum[bin] *= gain;
        }
        self.frame_count = self.frame_count.saturating_add(1);
    }

    fn synthesize(&mut self, output: &mut [f32]) -> Result<()> {
        self.spectrum[0].im = 0.0;
        self.spectrum[BINS - 1].im = 0.0;
        self.inverse.process_with_scratch(
            &mut self.spectrum,
            &mut self.time_scratch,
            &mut self.fft_scratch,
        )?;
        let inverse_scale = 1.0 / N_FFT as f32;
        for (acc, (sample, window)) in self
            .ola
            .iter_mut()
            .zip(self.time_scratch.iter().zip(&self.window))
        {
            *acc += sample * window * inverse_scale;
        }
        output.copy_from_slice(&self.ola[..HOP]);
        self.ola.copy_within(HOP.., 0);
        self.ola[N_FFT - HOP..].fill(0.0);
        Ok(())
    }

    fn clear(&mut self) {
        self.analysis_frame.fill(0.0);
        self.ola.fill(0.0);
        self.smoothed_power.fill(EPSILON);
        self.noise_power.fill(EPSILON);
        self.minimum_power.fill(f32::MAX);
        self.previous_posterior_snr.fill(1.0);
        self.gain.fill(1.0);
        self.target_gain.fill(1.0);
        self.frequency_smoothed_gain.fill(1.0);
        self.frame_count = 0;
    }
}

impl FrameDenoiser for WebRtcFrameProcessor {
    fn sample_rate(&self) -> usize {
        SAMPLE_RATE
    }

    fn frame_size(&self) -> usize {
        HOP
    }

    fn output_delay_frames(&self) -> usize {
        RECON_DELAY_FRAMES
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), HOP);
        debug_assert_eq!(output.len(), HOP);
        self.analyze(input)?;
        self.suppress();
        self.synthesize(output)
    }

    fn reset(&mut self) -> Result<()> {
        self.clear();
        Ok(())
    }
}

/// Fixed-delay, arbitrary-device-rate WebRTC-style input denoiser.
pub struct WebRtcDenoiser {
    inner: FixedDelayAdapter<WebRtcFrameProcessor>,
}

impl WebRtcDenoiser {
    pub fn new(device_sample_rate: u32, level: WebRtcSuppressionLevel) -> Result<Self> {
        Ok(Self {
            inner: FixedDelayAdapter::new(WebRtcFrameProcessor::new(level), device_sample_rate)?,
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
        level: WebRtcSuppressionLevel,
    ) -> Result<Vec<f32>> {
        Self::new(sample_rate, level)?.inner.process_finite(input)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_noise(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_215.0 - 0.5) * 0.08
            })
            .collect()
    }

    #[test]
    fn finite_output_preserves_length_and_stays_finite() {
        let input = deterministic_noise(SAMPLE_RATE * 2);
        for level in [
            WebRtcSuppressionLevel::Low,
            WebRtcSuppressionLevel::Moderate,
            WebRtcSuppressionLevel::High,
            WebRtcSuppressionLevel::VeryHigh,
        ] {
            let output = WebRtcDenoiser::process_finite(&input, 48_000, level).unwrap();
            assert_eq!(output.len(), input.len());
            assert!(output.iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn stronger_levels_reduce_stationary_noise_more() {
        let input = deterministic_noise(SAMPLE_RATE * 3);
        let low =
            WebRtcDenoiser::process_finite(&input, 48_000, WebRtcSuppressionLevel::Low).unwrap();
        let high = WebRtcDenoiser::process_finite(&input, 48_000, WebRtcSuppressionLevel::VeryHigh)
            .unwrap();
        let tail = SAMPLE_RATE;
        assert!(crate::dsp::rms(&high[tail..]) < crate::dsp::rms(&low[tail..]));
    }

    #[test]
    fn reset_reproduces_startup_timeline() {
        let input = deterministic_noise(SAMPLE_RATE);
        let mut denoiser = WebRtcDenoiser::new(48_000, WebRtcSuppressionLevel::Moderate).unwrap();
        let mut first = input.clone();
        denoiser.process_in_place(&mut first).unwrap();
        denoiser.reset().unwrap();
        let mut second = input;
        denoiser.process_in_place(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

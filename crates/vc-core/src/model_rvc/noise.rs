//! Timeline-stable latent noise for RVC exports that expose the VITS `z`
//! (`rnd`) tensor as a generator input. The tensor layout is
//! `[1, channels, frames]`, so contiguous memory is channel-major.
//!
//! Rolling inference replays most feature frames on every chunk. A stateful
//! random stream would assign different noise to those replayed frames and can
//! make the generator's overlap sound grainy or unstable. This module instead
//! maps `(seed, channel, absolute_frame)` directly to a standard-normal value.
//! It is deterministic, independent of call order, and shared by every backend.

/// Fixed seed for all RVC backends. The value spells `RVCRND` in ASCII, which
/// keeps captures reproducible while the absolute frame coordinate prevents
/// adjacent chunks from repeating the same window.
pub(super) const RVC_RND_SEED: u64 = 0x0000_5256_4352_4e44;

/// Counter-based standard-normal generator for an RVC `rnd` tensor.
#[derive(Clone, Copy)]
pub(super) struct GaussianNoise {
    seed: u64,
}

impl GaussianNoise {
    pub(super) const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Fill a channel-major `[1, channels, frames]` window. Frames with the
    /// same absolute coordinate receive bit-identical values even when their
    /// local index changes in a later rolling window.
    pub(super) fn fill_window(
        &self,
        out: &mut [f32],
        channels: usize,
        frames: usize,
        window_start_frame: i64,
    ) {
        assert_eq!(
            out.len(),
            channels
                .checked_mul(frames)
                .expect("RVC rnd shape overflow"),
            "RVC rnd output length must match [channels, frames]"
        );
        let frames_i64 = i64::try_from(frames).expect("RVC rnd frame count must fit i64");
        window_start_frame
            .checked_add(frames_i64)
            .expect("RVC rnd absolute frame range overflow");

        for channel in 0..channels {
            let channel_offset = channel * frames;
            let mut local_frame = 0;
            while local_frame < frames {
                let absolute_frame = window_start_frame + local_frame as i64;
                let (even, odd) = self.normal_pair(channel as u64, absolute_frame.div_euclid(2));
                if absolute_frame.rem_euclid(2) == 0 {
                    out[channel_offset + local_frame] = even;
                    local_frame += 1;
                    if local_frame < frames {
                        out[channel_offset + local_frame] = odd;
                        local_frame += 1;
                    }
                } else {
                    out[channel_offset + local_frame] = odd;
                    local_frame += 1;
                }
            }
        }
    }

    fn normal_pair(&self, channel: u64, absolute_pair: i64) -> (f32, f32) {
        // Pair adjacent absolute frames so one Box-Muller transform supplies
        // both values without making the result depend on window boundaries.
        let counter = self.seed
            ^ channel.wrapping_mul(0xd2b7_4407_b1ce_6e93)
            ^ (absolute_pair as u64).wrapping_mul(0xca5a_8263_9512_1157);
        let u1 = 1.0 - unit_f64(splitmix64(counter ^ 0x8cb9_2baa_3f3d_8dd7));
        let u2 = unit_f64(splitmix64(counter ^ 0x9e37_79b9_7f4a_7c15));
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        ((radius * angle.cos()) as f32, (radius * angle.sin()) as f32)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn unit_f64(bits: u64) -> f64 {
    // The top 53 bits produce [0, 1). `normal_pair` reflects the first uniform
    // into (0, 1], so Box-Muller never evaluates ln(0).
    (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(noise: GaussianNoise, channels: usize, frames: usize, start: i64) -> Vec<f32> {
        let mut out = vec![0.0; channels * frames];
        noise.fill_window(&mut out, channels, frames, start);
        out
    }

    #[test]
    fn overlapping_windows_reuse_each_channel_frame_exactly() {
        let noise = GaussianNoise::new(RVC_RND_SEED);
        let channels = 4;
        let frames = 19;
        let shift = 7;
        let first = window(noise, channels, frames, -11);
        let second = window(noise, channels, frames, -11 + shift as i64);

        for channel in 0..channels {
            let first_base = channel * frames;
            let second_base = channel * frames;
            assert_eq!(
                &first[first_base + shift..first_base + frames],
                &second[second_base..second_base + frames - shift]
            );
        }
    }

    #[test]
    fn same_seed_and_timeline_are_stable_across_instances() {
        let first = window(GaussianNoise::new(1234), 3, 31, 98);
        let second = window(GaussianNoise::new(1234), 3, 31, 98);
        assert_eq!(first, second);
    }

    #[test]
    fn different_absolute_positions_do_not_repeat() {
        let noise = GaussianNoise::new(42);
        assert_ne!(window(noise, 2, 64, 0), window(noise, 2, 64, 64));
    }

    #[test]
    fn channel_major_layout_keeps_coordinate_values() {
        let noise = GaussianNoise::new(9);
        let full = window(noise, 3, 8, 20);
        for channel in 0..3 {
            let one = window(noise, channel + 1, 1, 25);
            assert_eq!(full[channel * 8 + 5], one[channel]);
        }
    }

    #[test]
    fn distribution_is_roughly_standard_normal() {
        let values = window(GaussianNoise::new(7), 192, 521, -137);
        let n = values.len() as f64;
        let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.05, "mean {mean} not near 0");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance {variance} not near 1"
        );
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn caller_owned_scratch_capacity_is_reused() {
        let noise = GaussianNoise::new(7);
        let mut scratch = Vec::new();
        scratch.resize(192 * 100, 0.0);
        noise.fill_window(&mut scratch, 192, 100, 0);
        let capacity = scratch.capacity();
        scratch.resize(192 * 80, 0.0);
        noise.fill_window(&mut scratch, 192, 80, 20);
        assert_eq!(scratch.capacity(), capacity);
    }
}

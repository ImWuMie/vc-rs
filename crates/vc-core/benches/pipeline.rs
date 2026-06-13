//! Microbenchmarks for the `vc-core` CPU hot paths.
//!
//! These cover the pure-CPU stages a realtime chunk passes through on the
//! worker thread — resampling, RMS/volume shaping, the SOLA/PSOLA cross-fade
//! join, and the input noise gate. They deliberately avoid ONNX model inference
//! (`RvcPipeline::process`): the model stages are non-deterministic and need GPU
//! SDKs, whereas everything here is deterministic and runs with no GPU stack
//! (`cargo bench -p vc-core`). The `dsp`/`sola` public API is what the realtime
//! worker actually spends its non-inference time in, so regressions here map
//! directly to added per-chunk latency.
//!
//! Input sizes mirror real usage: chunk lengths are quoted in 48 kHz model-rate
//! samples (the RVC output domain), and resampling benches use the 48k<->16k
//! conversions the embedder/F0 front-end performs every chunk.

use divan::{black_box, Bencher};
use vc_core::dsp::{self, NoiseGate, RmsMixScratch, StreamingResampleMono};
use vc_core::sola::SmoothingKind;
use vc_core::sola::{model_domain_chunk_smoother, prepare_model_output, ChunkSmootherConfig};

fn main() {
    divan::main();
}

// Representative chunk lengths at the 48 kHz model sample rate: ~100 ms and
// ~250 ms, the range a realtime block typically falls in.
const CHUNK_48K_100MS: usize = 4_800;
const CHUNK_48K_250MS: usize = 12_000;

/// Deterministic voiced-ish test signal: a 220 Hz tone plus a quieter harmonic,
/// so SOLA cross-correlation and the RMS envelope see realistic structure
/// instead of a pure sine. Amplitude stays well inside [-1, 1].
fn synthetic_signal(len: usize, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    (0..len)
        .map(|i| {
            let t = i as f32 / sr;
            0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect()
}

mod dsp_bench {
    use super::*;

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn rms(bencher: Bencher, len: usize) {
        let signal = synthetic_signal(len, 48_000);
        bencher.bench_local(|| dsp::rms(black_box(&signal)));
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn compute_rms_envelope_into(bencher: Bencher, len: usize) {
        let signal = synthetic_signal(len, 48_000);
        let mut out = Vec::new();
        bencher.bench_local(|| {
            dsp::compute_rms_envelope_into(black_box(&signal), 48_000, &mut out);
            black_box(&out);
        });
    }

    // Volume-following mix of the model output toward the input reference RMS;
    // runs once per chunk. `rms_mix_rate = 0.5` exercises the full envelope path
    // (0.0/1.0 short-circuit). Scratch is reused, matching the worker.
    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn apply_rms_mix_with_scratch(bencher: Bencher, len: usize) {
        let reference = synthetic_signal(len, 48_000);
        let mut scratch = RmsMixScratch::default();
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| template.clone())
            .bench_local_values(|mut output| {
                dsp::apply_rms_mix_with_scratch(
                    black_box(&reference),
                    &mut output,
                    48_000,
                    0.5,
                    &mut scratch,
                );
                black_box(output);
            });
    }

    // One-shot resampler (allocates a fresh FFT plan + output each call) for both
    // front-end directions: 48k->16k feeds the embedder/F0, 16k->48k and the
    // model->output conversions go the other way.
    #[divan::bench(args = [(48_000, 16_000), (16_000, 48_000)])]
    fn resample_mono(bencher: Bencher, (from, to): (usize, usize)) {
        let signal = synthetic_signal(from, from as u32); // ~1 s buffer
        bencher.bench_local(|| dsp::resample_mono(black_box(&signal), from, to).unwrap());
    }

    // Streaming resampler reusing its plan/scratch across chunks — the path the
    // worker actually drives. Each iteration pushes one 100 ms chunk through a
    // persistent resampler.
    #[divan::bench(args = [(48_000, 16_000), (16_000, 48_000)])]
    fn streaming_resample_chunk(bencher: Bencher, (from, to): (usize, usize)) {
        let chunk = synthetic_signal(from / 10, from as u32); // ~100 ms
        let mut resampler = StreamingResampleMono::new(from, to).unwrap();
        let mut out = Vec::new();
        // Warm the internal pending buffer so we measure steady-state cost.
        resampler.process_into(&chunk, &mut out).unwrap();
        bencher.bench_local(|| {
            out.clear();
            resampler.process_into(black_box(&chunk), &mut out).unwrap();
            black_box(&out);
        });
    }

    // SOLA offset search: normalized cross-correlation over a crossfade-sized
    // reference against a candidate with the search window appended. Sizes are
    // the CLI defaults at 48 kHz (85 ms crossfade, 12 ms search).
    #[divan::bench]
    fn sola_offset(bencher: Bencher) {
        let crossfade = dsp::chunk_samples_for_rate(48_000, 85);
        let search = dsp::chunk_samples_for_rate(48_000, 12);
        let reference = synthetic_signal(crossfade, 48_000);
        let candidate = synthetic_signal(crossfade + search, 48_000);
        bencher
            .bench_local(|| dsp::sola_offset(black_box(&candidate), black_box(&reference), search));
    }

    #[divan::bench]
    fn sola_offset_with_threshold(bencher: Bencher) {
        let crossfade = dsp::chunk_samples_for_rate(48_000, 85);
        let search = dsp::chunk_samples_for_rate(48_000, 12);
        let reference = synthetic_signal(crossfade, 48_000);
        let candidate = synthetic_signal(crossfade + search, 48_000);
        bencher.bench_local(|| {
            dsp::sola_offset_with_threshold(
                black_box(&candidate),
                black_box(&reference),
                search,
                1e-4,
            )
        });
    }

    #[divan::bench]
    fn crossfade(bencher: Bencher) {
        let len = dsp::chunk_samples_for_rate(48_000, 85);
        let prev_tail = synthetic_signal(len, 48_000);
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| template.clone())
            .bench_local_values(|mut current| {
                dsp::crossfade(black_box(&prev_tail), &mut current);
                black_box(current);
            });
    }

    // Input denoise gate, run in place over a chunk at the input sample rate.
    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn noise_gate_process_in_place(bencher: Bencher, len: usize) {
        let template = synthetic_signal(len, 48_000);
        bencher
            .with_inputs(|| {
                (
                    NoiseGate::new(48_000.0, 0.01, 5.0, 50.0, 0.0),
                    template.clone(),
                )
            })
            .bench_local_values(|(mut gate, mut buf)| {
                gate.process_in_place(&mut buf);
                black_box(buf);
            });
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn i16_to_f32_into(bencher: Bencher, len: usize) {
        let input: Vec<i16> = (0..len).map(|i| (i as i16).wrapping_mul(37)).collect();
        let mut out = vec![0.0f32; len];
        bencher.bench_local(|| {
            dsp::i16_to_f32_into(black_box(&input), &mut out);
            black_box(&out);
        });
    }

    #[divan::bench(args = [CHUNK_48K_100MS, CHUNK_48K_250MS])]
    fn f32_to_i16_into(bencher: Bencher, len: usize) {
        let input = synthetic_signal(len, 48_000);
        let mut out = vec![0i16; len];
        bencher.bench_local(|| {
            dsp::f32_to_i16_into(black_box(&input), &mut out);
            black_box(&out);
        });
    }
}

mod sola_bench {
    use super::*;

    fn smoother_config(kind: SmoothingKind, output_chunk_samples: usize) -> ChunkSmootherConfig {
        // Mirrors the CLI's ChunkOutputConfig defaults (crates/vc-cli/src/engine.rs):
        // SOLA/PSOLA run in the 48 kHz model domain with output also at 48 kHz, so
        // no resampling masks the join cost.
        ChunkSmootherConfig {
            kind,
            output_chunk_samples,
            output_sample_rate: 48_000,
            model_sample_rate: 48_000,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
        }
    }

    // Full chunk-join path: `prepare_model_output` runs the SOLA/PSOLA offset
    // search + cross-fade and fits the result to the output chunk. This is the
    // closest model-free stand-in for a realtime chunk's post-inference cost.
    // The smoother is primed once so we measure steady-state joins; the model
    // audio/pitchf and the output buffer are reused like the realtime worker.
    #[divan::bench(args = [SmoothingKind::Sola, SmoothingKind::Psola])]
    fn prepare_model_output_chunk(bencher: Bencher, kind: SmoothingKind) {
        let chunk = CHUNK_48K_100MS;
        let mut joiner = model_domain_chunk_smoother(smoother_config(kind, chunk));
        // The model output is longer than the chunk so the joiner has a search
        // window; pitchf gives a stable voiced F0 so PSOLA takes its pitch path.
        let raw_len = chunk + dsp::chunk_samples_for_rate(48_000, 97);
        let audio = synthetic_signal(raw_len, 48_000);
        let pitchf = vec![220.0f32; 256];
        joiner.prime_model_output(&audio, &pitchf);

        bencher
            .with_inputs(Vec::<f32>::new)
            .bench_local_values(|mut out| {
                let sola_offset = prepare_model_output(
                    &audio,
                    &pitchf,
                    48_000,
                    48_000,
                    chunk,
                    black_box(&mut joiner),
                    None,
                    &mut out,
                )
                .unwrap();
                black_box((sola_offset, out));
            });
    }
}

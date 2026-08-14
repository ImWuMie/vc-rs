# Architecture

## Purpose

This document describes the conceptual architecture of `vc-rs`: how audio moves
through the realtime engine, how the RVC pipeline is staged, and why chunk
smoothing is separated from the audio callback. Concrete commands, local model
paths, and smoke-test recipes belong in `README.md` or local scripts instead.

CLI, GUI, VST3, and WAV conversion share the same audio-I/O-agnostic conversion
components wherever their hosting constraints permit. Front-ends adapt device,
host, or file I/O to those components; they should not own separate inference,
chunk-conversion, smoothing, or output-assembly implementations.

## Module Boundaries

- `vc-core`: shared audio-I/O-agnostic conversion components, including
  `RvcPipeline`, `ChunkConverter`, DSP, and SOLA/PSOLA smoothing.
- `vc-app`: shared standalone realtime runtime for CLI and GUI, including device
  I/O, bounded queues, worker orchestration, and metrics. The audio host is the
  `AudioHost` enum (cpal-`HostId`-aligned: `Wasapi`/`Asio`/`CoreAudio`/`Alsa`/`Jack`)
  and is chosen **per direction** (`input_host`/`output_host` in `RealtimeConfig`),
  so input and output may use different hosts — they are independent streams and
  clock domains, already resampled between by the engine. Every host except WASAPI
  *exclusive* goes through the shared cpal stream path (they differ only by which
  cpal host the device comes from); WASAPI exclusive uses the bespoke `wasapi_audio`
  path until cpal gains exclusive mode. Hosts unavailable on the running
  platform/build (e.g. ASIO without the `asio` feature) error at open time. cpal
  loads a single ASIO driver globally, so ASIO-on-both-directions shares one driver.
- `vc-cli`: CLI arguments, validation, realtime runtime control, and WAV file
  adaptation to the shared conversion components.
- `vc-gui`: GUI state and controls that configure the `vc-app` runtime.
- `vc-vst3`: DAW host adaptation, audio-callback ring-buffer I/O, worker
  scheduling, plugin state, and host latency reporting.

Changes to chunk sizing, model context, smoothing, or output latency usually
cross the front-end worker runtimes, `model_rvc`, `sola`, and `dsp`; review them
together.

## Shared Conversion Paths

All conversion modes should use the shared `vc-core` model and chunk-conversion
components. Inference, model streaming state, output shaping, and SOLA/PSOLA
joining must not be reimplemented in a front-end merely because its audio source
or scheduler differs.

CLI and GUI additionally share `vc-app` because both own standalone audio
devices. VST3 cannot use that device runtime because the DAW owns its audio
callback and requires plugin-specific state and latency reporting. VST3 should
still adapt host audio to the shared conversion components and keep its distinct
worker and buffering behavior narrowly scoped to host integration.

WAV conversion is an offline adapter around the same `RvcPipeline` and
`ChunkConverter` path used for realtime conversion. Offline processing may use
different scheduling, prime the smoother, pad a partial input chunk, and collect
the final tail explicitly. Those differences must not become a separate model,
chunk-conversion, smoothing, or output-shaping path.

When a hosting constraint requires a front-end-specific behavior, document the
constraint near the implementation and preserve the shared path for all
unaffected stages.

## Realtime Topology

```mermaid
flowchart LR
    mic["Input device"] --> in_cb["Input audio callback"]
    in_cb --> in_ring["Input ring buffer"]
    in_ring --> worker["Worker thread"]
    worker --> model["RVC pipeline"]
    model --> smooth["SOLA / PSOLA smoother"]
    smooth --> out_ring["Output ring buffer"]
    out_ring --> out_cb["Output audio callback"]
    out_cb --> speaker["Output device"]
```

The audio callbacks are intentionally small. They move samples through bounded
ring buffers and emit silence on underrun; they do not run ONNX inference,
perform chunk smoothing, write files, or log directly. Anything that can block,
allocate heavily, or take model-scale CPU/GPU time is kept on the worker side.

The threads carrying those callbacks run at OS real-time priority. The bespoke
WASAPI path and the worker self-boost via `thread-priority`; the cpal-driven
paths (WASAPI-shared, ASIO, …) get the same treatment from cpal's `realtime`
feature (enabled on the `cpal` dependency in `vc-app`), which promotes its
internal stream threads — Windows MMCSS "Pro Audio", else
`THREAD_PRIORITY_TIME_CRITICAL`. This pulls `audio_thread_priority` (MPL-2.0),
allowed crate-scoped in `deny.toml` since it ships unmodified and statically
linked.

The worker owns chunk accumulation, model inference, output smoothing,
resampling back to the device rate, and metrics updates. If inference falls
behind, bounded queues make the failure mode explicit: input overrun drops new
input samples, output underrun emits silence, and output overflow drops newly
produced samples rather than blocking the realtime callback.

Standalone sessions with a complete model set keep both RVC and passthrough
routes available on the worker. The live passthrough flag is sampled once per
input chunk. Passthrough stops invoking RVC inference and applies input gain,
the configured input denoiser, device-rate resampling, and output gain. When
conversion resumes, the worker clears stale RVC rolling context and smoother
history before processing the next chunk. Model-free sessions expose only the
passthrough route.

## Chunk Lifecycle

Realtime audio arrives in device callback-sized blocks, but the model operates
on larger logical chunks. The worker accumulates input samples until one model
chunk is available, then sends that chunk through the RVC pipeline.

The RVC pipeline does not treat each chunk as isolated audio. It keeps streaming
state for recent input, 16 kHz resampled audio, content features, and F0 frames.
Each inference window includes the current chunk plus enough recent context and
extra output allowance for smoothing. The model output is then trimmed to the
tail that corresponds to the current chunk and the smoother search window.

This lifecycle preserves three invariants:

- The output smoother emits a fixed number of device-rate samples per input
  chunk.
- Feature frames, continuous F0, coarse pitch, and model output must refer to
  the same time window.
- The realtime callback sees only queued samples, never model-domain state.

## RVC Pipeline

```mermaid
flowchart TD
    input["Device-rate mono chunk"] --> gain["Input gain"]
    gain --> raw16["Residual-preserving 16 kHz branch"]
    gain --> denoise["Off / Gate / RNNoise / WebRTC / DFN3<br/>(device rate)"]
    denoise --> pitch16["Pitch 16 kHz branch"]
    pitch16 --> gtcrn["GTCRN or passthrough"]
    raw16 --> blend["ContentVec blend<br/>(raw + denoised share)"]
    gtcrn --> blend
    raw16 --> rmvpeblend["RMVPE blend<br/>(raw + denoised share)"]
    gtcrn --> rmvpeblend
    blend --> embed["Content embedder"]
    rmvpeblend --> f0["F0 estimator / RMVPE"]
    embed --> retrieve["Optional IVF-Flat feature retrieval"]
    retrieve --> feats["Content feature 2x upsampling"]
    f0 --> rawf0["Natural F0 alignment"]
    feats --> protect["Optional consonant protection + boundary easing<br/>(raw-F0 unvoiced frames)"]
    rawf0 --> protect
    protect --> rvc["RVC generator"]
    rawf0 --> pitchf["F0 continuity + pitch shift"]
    pitchf --> coarse["Coarse pitch bins"]
    pitchf --> rvc
    coarse --> rvc
    rvc --> tail["Select stable output tail"]
    tail --> level["RMS/envelope/gain shaping"]
    level --> join["SOLA or PSOLA chunk join"]
    join --> device["Device-rate output chunk"]
```

Standalone RNNoise (48 kHz) runs at the **device rate**, after input gain and on
the RMVPE branch. Its fixed-delay adapter preserves the input sample count for
every worker call while retaining recurrent and resampler state across chunks.
The residual ContentVec branch remains unprocessed except for a matching delay,
then receives the configured denoised share after both branches reach 16 kHz.

**GTCRN (16 kHz) is the exception to the device-rate rule.** It denoises the new
16 kHz increment *inside* `generate_input` — reusing the resample the pipeline
already does into `audio_16k_buffer`, before that increment is windowed — so the
realtime hot path pays no extra round-trip resample and the model sees native
16 kHz. It shares the same fixed-delay `FrameDenoiser` adapter as RNNoise (at
16 kHz the adapter's resamplers are bypass), preserves the per-call 16 kHz sample
count, and never shifts the feature/F0 grid. RVC-path **input RMS and silence
detection use the configured RMVPE branch**, while volume-envelope memory and
the RMS-mix reference follow the residual-preserving ContentVec blend. With
denoising off, both histories are filled from one 16 kHz resample; a second
resampler is used only when the device-rate branches differ. The passthrough
route keeps a separate device-rate GTCRN instance (its resamplers engage). GTCRN
ships in standalone packages: Windows ML uses ORT CPU for the tiny graph, while
TensorRT uses a native TensorRT engine so the TensorRT package remains ORT-free.
VST3 enables the in-tree WebRTC suppressor, but intentionally does not enable
or ship RNNoise, GTCRN, or DeepFilterNet3 model data. `package.ps1 -DeepFilterNet3`
is an external-model opt-in package variant: it includes DFN3 runtime code and
matching license notices, but never the model archive.

For Gate, RNNoise, WebRTC, GTCRN, and DeepFilterNet3, `denoiser_content_mix` and
`denoiser_rmvpe_mix` are live worker-side controls:
for ContentVec, `0` sends raw input, `0.25` is the residual-preserving default,
and `1` uses the fully denoised path. RMVPE's control uses the same `0..=1`
meaning, but defaults to `1.0` for compatibility with the former full-denoise
pitch path. RNNoise and GTCRN have fixed output delay, so both raw branches pass
through matching preallocated alignment before mixing; this prevents an earlier
voice from being combined with the cleaned signal. All branch buffers,
resampling, and denoiser work stay on the conversion worker; callbacks continue
to move samples only through lock-free queues.

Changing denoiser mode restarts its model-side rolling windows and matching
delay lines, and the owning front-end discards the SOLA/PSOLA join history at the
same boundary. This prevents pre-switch audio from being concatenated or
crossfaded with a differently delayed post-switch timeline.

Conceptually, RVC conversion has three model-facing inputs:

- Content features describe what is being spoken while discarding much of the
  source speaker identity.
- F0/pitch describes the melody of voiced speech and supports pitch shifting.
- Speaker/model conditioning selects the target voice inside the RVC model.

The content embedder and F0 estimator operate on the same 16 kHz context window.
Content features are upsampled by repeating each frame twice, matching the RVC
pipeline convention that expands the content-feature frame rate before
generation. F0 is then length-matched to the resulting feature frame count and
kept both as continuous `pitchf` and quantized coarse pitch. Misaligning these
streams usually sounds like timing drift, pitch lag, or unstable consonants, so
frame-grid changes should be treated as audio-quality changes, not cleanup.

When configured, the pipeline reads a standard RVC
`added_IVF*_Flat_*.index` FAISS `IndexIVFFlat` file during model construction.
It rejects unsupported index families, unpopulated `trained_*.index` files, and
feature widths that do not match the generator (v1 = 256, v2 = 768). The worker
searches its reusable IVF scratch buffers for the eight nearest vectors and
applies upstream RVC inverse-squared-distance blending before feature doubling.
`index_rate=0` is an exact no-op. If `protect < 0.5`, the worker also retains the
pre-retrieval ContentVec tensor and, after the same trim/repeat transformation,
mixes it back only for raw-F0-unvoiced frames. It intentionally uses raw,
unshifted F0 so continuity/pitch-shift postprocessing cannot turn a consonant
into a retrieval-voiced frame. Index file I/O, parsing, retrieval, and this
extra scratch memory are all outside the audio callback.

`protect_transition_ms` is a vc-rs extension, not an MXGF/upstream-RVC setting.
At its default of zero it takes the exact binary upstream Protect path. A
positive value is rounded up to the 10 ms generator feature grid and blends the
nearby *voiced* frames progressively from the Protect mix back to full retrieval;
the raw-F0-unvoiced frame itself remains protected. The bounded 0..100 ms scan
uses the existing worker-owned tensors and no extra scratch allocation. It is
skipped entirely without an index, with `index_rate=0`, or with `protect=0.5`.

The default RMVPE confidence threshold is 0.03. Optional F0 continuity runs
after alignment on the worker and linearly interpolates only zero runs bounded
by voiced frames; leading/trailing runs stay unvoiced to avoid extrapolating a
breath or consonant beyond the known streaming window. Coarse pitch is derived
from the resulting continuous F0 so the generator inputs remain consistent.

After generation, the output may be shaped by volume envelope, RMS mixing, and
manual or automatic gain. These operations happen before chunk joining so the
smoother compares and crossfades audio at the level that will actually be
played.

## SOLA

SOLA, Similarity Overlap-Add, is used to hide discontinuities between
independently generated chunks. Even when two chunks represent adjacent input
audio, the generated waveform can be shifted by a few samples at the boundary.
Naively concatenating those chunks can produce clicks, combing, or a rough
phasiness.

The smoother keeps a short tail from the previous emitted chunk as a reference.
For the next generated candidate, the worker asks the model for extra samples
around the boundary. SOLA searches within that extra range for the offset whose
overlap is most similar to the reference, cuts the candidate at that offset, and
crossfades the overlap. The emitted chunk length stays fixed; only the boundary
position inside the candidate moves.

SOLA must stay on the worker side. It needs model-output history, extra model
samples, correlation search, and crossfade buffers. Moving it into the audio
callback would put search work and allocation pressure on the realtime path.

## PSOLA

PSOLA, Pitch-Synchronous Overlap-Add, is the pitch-aware variant used here when
the current output has stable voiced F0. Instead of accepting any high-similarity
offset, it estimates the current pitch period from `pitchf` and prefers offsets
that align the overlap near pitch-period boundaries.

This is useful for sustained vowels and other voiced regions, where a boundary
that cuts across the waveform period can sound unstable even if the generic
SOLA score is acceptable. When F0 is missing, unvoiced, too unstable, or outside
the supported range, PSOLA falls back to normal SOLA. That fallback is important:
forcing pitch-synchronous alignment on noisy consonants or silence usually makes
the boundary worse.

## Latency Trade-offs

End-to-end latency is the sum of device buffering, input chunk accumulation,
model inference time, smoothing/search allowance, output buffering, and any
resampling delay. Reducing one term often increases pressure elsewhere.

Smaller chunks reduce chunking latency but increase scheduling overhead and make
the model pipeline more sensitive to inference spikes. Larger chunks are easier
for the model and smoother but add startup and interactive latency. Extra model
output gives SOLA/PSOLA more room to find a clean join, but it also increases
the amount of audio processed per chunk.

The architecture therefore treats latency-sensitive code as a boundary:
callbacks are realtime-safe sample movers, while the worker is the only place
that may spend time on inference, smoothing, diagnostics, and file-oriented
debug output.

## WAV Mode

WAV conversion uses the shared `RvcPipeline`, `ChunkConverter`, and smoother so
audio-quality changes can be tested deterministically without device scheduling
noise. It can prime the smoother, pad the final partial input chunk, and handle
the final output tail explicitly because it is not constrained by callback
deadlines. A difference between WAV and realtime output should be explained by
buffering, scheduling, padding, or final-tail handling rather than by a separate
conversion path.

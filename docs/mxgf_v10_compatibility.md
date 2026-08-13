# MXGF v10.1 Compatibility Audit

This note records what was checked in the publicly readable MXGF v10.1
directory and how the compatible part is handled by `vc-rs`.

## Findings

- The web page advertises a new v4/f048k pretrained RVC base. The readable
  update package contains `f0G48k_mxgf.pth`, `f0D48k_mxgf.pth`, a v2 48 kHz
  config, training entry points, and an RMVPE checkpoint.
- The MXGF generator config keeps the standard RVC v2 tensor contract
  (`filter_channels=768`, `gin_channels=256`, 48 kHz upsampling). Its material
  change is `spk_embed_dim: 308`, compared with the stock v2 config's 109.
- The MXGF RMVPE checkpoint has the same 741 tensors and 90,472,917 parameters
  as the installed full-precision checkpoint. The older backup is the same
  network in FP16, so this is a precision/asset change, not a new F0 graph.
- The locally configured MXGF realtime path uses a 0.03 RMVPE confidence
  threshold, interpolates every zero-valued F0 frame, and retains about 2.12 s
  of history. The compared vc-rs GUI configuration used 0.3, no continuity
  interpolation, and 333 ms of extra context. These settings, rather than a new
  pitch network, explain the more continuous upward-shifted vowels and tails.
- The local MXGF checkpoint and the compared `keruan4.onnx` generator have at
  least 247 directly corresponding tensors with byte-identical values. The
  exporter/runtime representation differs, but there is no evidence that a
  better target-speaker generator caused the observed A/B difference.
- CUDA Graph capture, fixed-shape TensorRT execution, F0 post-processing,
  RMS/envelope shaping, model pools, and shared WAV/realtime conversion already
  exist in `vc-rs`; they were not duplicated from the Python front-end.

## Implemented Compatibility

`vc-core` now reads the RVC generator's `emb_g.weight` initializer dimensions
while loading or inspecting an ONNX model. The count is retained as pipeline
metadata and live Speaker ID updates are clamped in constant time. The GUI
shows the active model's actual range, while VST3 exposes the static 0..307
range required by DAW parameter automation and lets the shared core clamp
smaller models.

The shared conversion pipeline now defaults RMVPE to 0.03 and provides an
explicit F0-continuity mode. Continuity linearly fills internal unvoiced runs
bounded by valid F0 on both sides; it deliberately keeps leading and trailing
runs unvoiced because a streaming window cannot safely distinguish an edge
dropout from breath, silence, or a consonant. This is slightly more conservative
than MXGF's `numpy.interp`, which also extends the nearest voiced value over
both edges.

GUI, CLI, WAV conversion, and VST3 all configure this same core
post-processor. The standalone GUI also has a high-quality preset using the
audited MXGF realtime values: 450 ms chunks, 2120 ms extra context, 0.03 F0
threshold, and continuity enabled. Existing GUI settings containing exactly the
old 0.3 default migrate to 0.03; deliberately chosen custom thresholds remain
unchanged.

The initializer parser reads only protobuf names and dimensions. It skips raw
weight bytes, runs only on the load/inspect path, and does not add any model
weights, local paths, or MXGF runtime components to this repository.

## Not Vendored

MXGF's proprietary/third-party pretrained weights, RMVPE files, launcher,
license service, obfuscated modules, and machine-specific packaging are not
copied into `vc-rs`. Users may point `vc-rs` at an ONNX export they are
licensed to use; distribution remains governed by `docs/distribution.md` and
the model's own license.

The Python callback implementation was also not copied. It performs model
inference, GPU work, logging, and RMS analysis directly from the sounddevice
callback. vc-rs retains its bounded ring-buffer and worker topology, so the
audio callbacks remain free of model inference, blocking I/O, and locks.

## Validation

- A deterministic two-second 48 kHz fixture was converted on CPU with the
  pre-change and current CLI using the same stock 109-speaker RVC model. The
  model's internal random ONNX nodes were seeded in a temporary validation copy
  so the comparison measured code changes rather than generator randomness.
  The outputs were sample-identical (`max_abs=0`, `relative_rms=0`,
  `log_spectral_distance=0 dB`).
- A temporary validation model expanded `emb_g.weight` from 109 to 308 rows by
  repeating rows from the local validation model. `inspect` reported IDs
  `0..307`, CPU WAV
  conversion completed with Speaker ID 307, and an out-of-range ID 999 produced
  the same samples as 307, proving the shared pipeline clamp. No generated
  model or audio fixture is retained in the repository.
- A 4.16 s local vocal clip was converted through the shared offline pipeline
  with the same `keruan4.onnx`, +12 semitones, unit gains, no denoiser, and a
  temporary model copy whose three internal ONNX random nodes had fixed seeds.
  Repeating the baseline was sample-identical (`max_abs=0`, spectral distance
  `0 dB`). On the emitted F0 tail, lowering only the threshold changed the
  voiced ratio from 0.2321 to 0.2518; enabling only continuity changed it to
  0.2482; the full 0.03 + continuity + 2120 ms preset reached 0.2643 (13.9%
  relatively more voiced frames than baseline). On CPU, average inference time
  per 450 ms chunk increased from about 230 ms at 333 ms context to about
  363-369 ms at 2120 ms context. Audio and seeded model artifacts remained in a
  temporary directory and are not part of the repository.
- The full high-quality configuration was also run through the release native
  TensorRT backend. After the one-time engine build, cached execution averaged
  19.688 ms of inference per 450 ms chunk and reported the same 0.2643 emitted-
  tail voiced ratio. The generated engine cache and output WAV remain outside
  the repository.
- Native TensorRT execution now writes ContentVec, RMVPE, and RVC results
  directly into the worker-owned buffers already used by the shared pipeline.
  Each loaded engine also retains its 64 KiB native error buffer and ContentVec
  retains its C input name, eliminating the former per-inference output and
  error-buffer allocations. A release conversion of a temporary 4.388 s WAV
  through `xiran.onnx` completed 10 high-quality chunks at 20.024 ms average
  inference time: 7.174 ms ContentVec, 6.206 ms RMVPE, and 6.080 ms RVC. A
  repeat native run was byte-identical. Temporary audio and logs remain outside
  the repository.

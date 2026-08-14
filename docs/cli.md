# vc-rs CLI reference (`vc-rs.exe`)

> [日本語](cli_ja.md) | English

`vc-rs.exe` is the CLI bundled with the GUI + CLI package. Everyday voice
conversion is fully covered by [`vc-gui.exe`](../README.en.md); the CLI adds the
things the **GUI does not do**:

- **Batch WAV-file conversion** (`wav`) — the GUI is real-time only.
- **Diagnostics and model inspection** (`doctor` / `devices` / `inspect`).
- **Listing and installing Windows ML execution providers (EPs)**
  (`windowsml-eps`).
- **Engine-cache management** (`engine-cache`).
- **Automation/scripting** and the finer DSP/audio parameters the GUI keeps
  pinned (WASAPI exclusive, `psola`, `--rms-mix-rate`, and more).

The GUI and CLI share the same inference pipeline, so settings you dial in from
the CLI reproduce identically in the GUI.

## Setup

1. Extract the GUI + CLI package zip (**keep the DLLs in the same folder as
   `vc-gui.exe` / `vc-rs.exe`**).
2. Open PowerShell in that folder.
3. Fetch the embedder + F0 models (below). Supply your own RVC voice-conversion
   model as `.onnx`, or export a compatible trained RVC `.pth` checkpoint with
   `export-pth` before loading it.

```powershell
pwsh .\download-models.ps1
```

This downloads `.\assets\content_vec_500.onnx` and `.\assets\rmvpe.onnx`.

> For requirements (Windows App SDK Runtime / NVIDIA driver) and how to pick a
> package, see [`README.en.md`](../README.en.md).

## Commands

```powershell
.\vc-rs.exe --help
```

| Command | Purpose |
| --- | --- |
| `doctor` | Diagnose runtime dependencies and device visibility needed to run |
| `devices` | List audio input/output devices |
| `inspect` | Show ONNX model inputs, outputs, and metadata (backend-independent) |
| `export-pth` | Offline-export a compatible trained RVC `.pth` generator checkpoint to ONNX |
| `run` | Real-time microphone-to-speaker conversion |
| `wav` | WAV-file to WAV-file conversion (same pipeline, deterministic testing) |
| `windowsml-eps` | List/install Windows ML catalog EPs (windowsml package only) |
| `engine-cache` | Inspect/clear the GPU engine cache |

### Diagnostics

```powershell
.\vc-rs.exe doctor
```

### List devices

```powershell
.\vc-rs.exe devices                      # all hosts available on this platform
.\vc-rs.exe devices --audio-backend wasapi # WASAPI devices only
.\vc-rs.exe devices --audio-backend asio # ASIO drivers (asio build only)
```

`--audio-backend` accepts `all` (default) or a host token
(`wasapi`/`asio`/`coreaudio`/`alsa`/`jack`). ASIO is only listed on a build made
with `--features asio` and with an ASIO driver installed; hosts not available on
the running platform/build are reported as such.

### Inspect a model

```powershell
.\vc-rs.exe inspect --model <your-rvc-model>.onnx
```

`inspect` is backend-independent and prints the ONNX model's inputs, outputs,
and metadata.

### Export a trained PTH checkpoint

RVC `.pth` files are PyTorch checkpoints and are not used directly by the
real-time engine. `export-pth` runs once outside the audio pipeline, exports a
standard ONNX generator, and validates its RVC inputs, output, F0 metadata, and
speaker embedding contract before making the output available.

```powershell
.\vc-rs.exe export-pth `
    --model .\voice.pth `
    --output .\voice.onnx `
    --rvc-root D:\path\to\your\trusted-rvc-installation `
    --python D:\path\to\your\trusted-rvc-installation\runtime\python.exe `
    --trust-rvc-root
```

`--rvc-root` must contain `infer\lib\infer_pack\models_onnx.py` and match the
checkpoint's RVC architecture. `--trust-rvc-root` is required because exporting
imports that local installation's Python model code. Use only models and RVC
installations you trust and are licensed to use. PyTorch loads the checkpoint
with `weights_only=True`.

The MXGF `f0G...pth` / `f0D...pth` files are **training base checkpoints**, not
target voice models, so the command rejects them. Train a voice from the base
first, then export the resulting compact `assets\weights\*.pth` checkpoint.

### Real-time conversion

```powershell
.\vc-rs.exe run --model <your-rvc-model>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --index-path <matching-added-IVF.index> --index-rate 0.65 --protect 0.33 --protect-transition-ms 20 `
    --input "Microphone" --output "Speakers" `
    --chunk-ms 500 --extra-convert-ms 100 `
    --provider windowsml --speaker-id 0
```

Pass a substring of the names shown by `devices` to `--input`/`--output`. On the
tensorrt package use `--provider tensorrt`.

### WAV-file conversion

Not available in the GUI. Useful for batch processing and for deterministic
verification of setting changes.

```powershell
.\vc-rs.exe wav --model <your-rvc-model>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input input.wav --output out.wav `
    --provider windowsml --speaker-id 0
```

## Tuning real-time settings

Balance dropouts, latency, and CPU/GPU load with `--chunk-ms` and
`--extra-convert-ms`.

- `--chunk-ms`: how much audio is processed per pass. Increase it
  (`500` → `750` → `1000`) when you hear dropouts or load spikes. Larger is more
  stable but adds input-to-output latency. GPU execution can often use smaller
  values.
- `--extra-convert-ms`: extra leading/trailing context handed to the conversion.
  Larger can be more stable but costs more. Start around `100` ms.

When tuning, the safe order is to **first find a value with no dropouts, then
lower `--chunk-ms`** to reduce latency.

## Key conversion parameters

- `--speaker-id 0`: speaker ID for multi-speaker models (default: 0).
- `--index-path <PATH>` (or `--index <PATH>`): optional RVC FAISS retrieval
  index. Use the training result named `added_IVF*_Flat_*.index`, never the
  unpopulated `trained_*.index`. Its feature width must match the generator:
  RVC v1 indexes are 256-dimensional and v2 indexes are 768-dimensional.
  The file is decoded when the model loads, not by the audio callback.
- `--index-rate 0.0..1.0`: blends ContentVec with the retrieved target-speaker
  features. `0.0` is an exact no-retrieval path; start around `0.5` to `0.75`.
  Higher values can improve target timbre but can make articulation less natural
  with a sparse, noisy, or mismatched index.
- `--protect 0.0..0.5`: standard RVC consonant protection for unvoiced F0
  frames when retrieval is enabled. `0.33` is the upstream default; `0.5`
  disables protection. Lower values preserve more original ContentVec in breaths
  and consonants. It has no effect when `--index-rate 0` or no index is loaded.
- `--protect-transition-ms 0..100`: optional vc-rs smoothing around a Protect
  boundary. `0` (the default) is exact standard RVC behavior. `20` ms is a
  useful starting point when full or high index retrieval leaves an obvious
  consonant edge: it gradually restores the retrieved features on the nearby
  voiced frames while leaving the actual raw-F0-unvoiced frame protected. It
  only applies when an index is loaded, `--index-rate` is nonzero, and
  `--protect < 0.5`.
- `--pitch-shift 0.0`: shift F0 in semitones (default: 0.0). `12.0` is one octave
  up, `-12.0` one octave down.
- `--input-gain 1.0` / `--output-gain 1.0`: input/output gain (default: 1.0).
  Raise when too quiet; raising too far clips.
- `--monitor-output <name>`: enable a monitor output — a second output device
  (on the same backend as `--output`) playing the converted signal with its own
  gain, e.g. headphones while the primary output feeds a stream or DAW. Pass `""`
  for the system default device. `--monitor-gain 1.0` sets the monitor gain
  (default: 1.0). Not supported with an ASIO output host.
- `--denoiser off|noise-gate|rnnoise|webrtc|gtcrn|deep-filter-net3`: exclusive input denoiser selection.
  RNNoise uses an embedded model. GTCRN requires `--gtcrn-model <dir>` pointing
  at a directory containing `gtcrn_stream.onnx` (download with
  `download-models.ps1 -Gtcrn`). WebRTC is built in and accepts
  `--webrtc-suppression-level low|moderate|high|very-high`. DeepFilterNet3
  requires `--deepfilternet3-model <archive>` (download with
  `download-models.ps1 -DeepFilterNet3`). The old `--noise-gate` flag remains
  as an alias.
- `--denoiser-content-mix 0.0..1.0`: share of the fully denoised branch mixed
  into ContentVec. `0.0` preserves the raw articulation branch, `0.25` is the
  recommended starting point, and `1.0` restores the legacy fully-denoised
  ContentVec path. The setting is ignored when denoising is off. RNNoise and
  GTCRN automatically delay the raw branch before mixing so their fixed output
  latency cannot produce an echo.
- `--denoiser-rmvpe-mix 0.0..1.0`: independent denoised share sent to RMVPE.
  `1.0` (the default) preserves the historical fully denoised pitch input;
  `0.0` uses the delay-aligned raw input. Intermediate values linearly blend
  the two branches. The setting is ignored when denoising is off.
- `--silence-threshold 0.0001`: threshold below which input is treated as
  silence.
- `--rms-mix-rate <0.0-1.0>`: closer to 0.0 follows the input's loudness
  dynamics, closer to 1.0 keeps the model output's loudness (default: 0.0).

Other options the GUI keeps pinned are also available from the CLI:
`--smoother sola|psola`, `--sola-search-ms`, `--crossfade-ms`,
`--rvc-output-tail-discard-ms`, `--gpu-priority normal|high`,
`--gpu-device-id <ID>` (CUDA/native TensorRT only), and the WASAPI exclusive
controls (`--wasapi-exclusive*`, `--wasapi-buffer-ms`).
See `--help` for the full list and defaults.

## Audio backends

`--audio-backend` selects the OS audio host for both directions (tokens match
cpal's host names); it defaults to the platform's native host (WASAPI on Windows,
CoreAudio on macOS, ALSA on Linux):

- `wasapi`: Windows WASAPI. Shared mode by default; add `--wasapi-exclusive*` for
  exclusive mode (with `--wasapi-buffer-ms` tuning), which routes through the
  bespoke low-latency path.
- `asio`: ASIO (low-latency audio interfaces). Only on a build made with
  `--features asio`; see [`scripts/README.md`](../scripts/README.md) for the ASIO
  SDK + LLVM build setup.
- `coreaudio` / `alsa` / `jack`: macOS / Linux hosts (for future cross-platform
  builds; `jack` needs `--features jack`). Selecting a host not available on the
  running platform/build errors with a hint.

Input and output can use **different** hosts with `--input-backend` /
`--output-backend`, which override `--audio-backend` for that direction:

```powershell
# Capture via WASAPI, play out through an ASIO interface.
.\vc-rs.exe run ... --input-backend wasapi --output-backend asio --output "<ASIO driver>"
```

ASIO loads a single driver globally, so when **both** directions are ASIO they
must name the same driver. ASIO buffer size is set in the driver's own control
panel, not by `--wasapi-buffer-ms`.

`wav --denoiser rnnoise`, `wav --denoiser webrtc`, and `wav --denoiser gtcrn`
compensate each denoiser's
fixed streaming delay and keep the output WAV at the original sample count.
RNNoise and GTCRN are available only in the standalone CLI/GUI packages, not
VST3. WebRTC is available in standalone and VST3 packages. GTCRN uses ORT CPU
in Windows ML packages and native TensorRT in TensorRT packages. DeepFilterNet3
is opt-in: pass `-DeepFilterNet3` to the package script to include its runtime
code, while its external archive is never part of a package.

## Windows ML execution providers (windowsml package)

With `--provider windowsml`, the windowsml package prefers a Windows ML catalog
EP, falling back to DirectML and finally CPU. To force a specific EP use
`windowsml-nvtrtx` / `windowsml-qnn` / `windowsml-openvino` / `windowsml-migraphx`
/ `windowsml-vitisai` (no fallback — it errors if the EP is not installed/ready).

Check and install catalog EPs from the CLI:

```powershell
.\vc-rs.exe windowsml-eps list
.\vc-rs.exe windowsml-eps install            # auto-select the best EP
.\vc-rs.exe windowsml-eps install --provider nvtrtx --yes
```

## TensorRT execution (tensorrt package)

The tensorrt package runs on the **bundled TensorRT runtime**, so nothing beyond
the NVIDIA driver needs installing.

> ⚠️ TensorRT builds an engine **on first run and whenever the model or input
> shape changes**, which can make startup very slow. Subsequent runs reuse the
> cached engine and start faster.

For detailed performance characteristics see
[`tensorrt_performance_ja.md`](tensorrt_performance_ja.md).

## Engine-cache management

Engines built by TensorRT (tensorrt package) and by Windows ML TensorRT-RTX
(`windowsml-nvtrtx`) are stored under `%LOCALAPPDATA%\vc-rs\tensorrt-cache` and
shared by both backends (override the location with `VC_RS_TENSORRT_CACHE_DIR`).
Inspect the location/size and clear the cache from the CLI:

```powershell
.\vc-rs.exe engine-cache info          # location, total size, per-model breakdown
.\vc-rs.exe engine-cache clear         # delete all (with confirmation)
.\vc-rs.exe engine-cache clear --yes   # delete all without confirmation
```

The cache is regenerable derived data — deleting it just rebuilds on the next
model load (only that run is slow again).

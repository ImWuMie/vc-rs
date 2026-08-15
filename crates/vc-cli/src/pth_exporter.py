"""Export one trained RVC generator checkpoint without using pickle globals.

This script is embedded in vc-rs and written to a temporary file by
`export-pth`. It intentionally imports the architecture from the explicitly
trusted local RVC installation rather than vendoring any upstream code or
weights into this project.
"""

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs768NSFsid,
)


def fail(message):
    raise RuntimeError(message)


def load_checkpoint(path):
    # RVC's compact target-voice checkpoints are tensor dictionaries. Loading
    # only weights avoids pickle global execution from untrusted checkpoints.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        fail(
            "the selected Python/PyTorch does not support weights_only=True; "
            "use PyTorch 2.0 or newer instead of falling back to unsafe pickle loading"
        )
    if not isinstance(checkpoint, dict):
        fail("expected a trained RVC checkpoint dictionary")
    required = {"config", "weight", "version", "f0", "sr"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        fail(
            "this is not an exported target-voice RVC generator checkpoint; "
            f"missing {', '.join(missing)} (available keys: {keys}). "
            "Training base checkpoints such as f0G/f0D are only training initialization weights; "
            "train a voice first and export its compact assets/weights/*.pth file."
        )
    if not checkpoint["f0"]:
        fail("only F0-guided RVC generator checkpoints are supported")
    if checkpoint["version"] not in ("v1", "v2"):
        fail(f"unsupported RVC checkpoint version: {checkpoint['version']!r}")
    if not isinstance(checkpoint["config"], (list, tuple)) or len(checkpoint["config"]) != 18:
        fail("RVC checkpoint config must contain the standard 18 generator fields")
    if not isinstance(checkpoint["weight"], dict):
        fail("RVC checkpoint weight field is not a tensor dictionary")
    return checkpoint


class GeneratorExport(nn.Module):
    """Expose RVC's normal infer path as the five-input vc-rs ONNX contract.

    `models_onnx` exposes latent noise as an input and, with current PyTorch,
    produces an attention graph whose dynamic reshape is invalid at non-example
    frame counts. The normal RVC inference graph samples that noise internally,
    which matches the VCClient-style exports accepted by all vc-rs backends.
    """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, feats, p_len, pitch, pitchf, sid):
        audio, _, _ = self.generator.infer(feats, p_len, pitch, pitchf, sid)
        return audio


def export(model_path, output_path, fixed_frames=None):
    checkpoint = load_checkpoint(model_path)
    weights = checkpoint["weight"]
    embedding = weights.get("emb_g.weight")
    if embedding is None or embedding.ndim != 2 or embedding.shape[0] < 1:
        fail("RVC checkpoint has no valid emb_g.weight speaker embedding")

    config = list(checkpoint["config"])
    # The compact checkpoint's embedding tensor is authoritative; RVC's own
    # exporter makes the same adjustment for models trained with a custom count.
    config[-3] = int(embedding.shape[0])
    version = checkpoint["version"]
    feature_channels = 256 if version == "v1" else 768

    torch.manual_seed(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fixed_frames is None:
        # Preserve the existing generic export byte-for-byte in spirit: it uses
        # the normal infer path, samples latent noise inside the graph, traces at
        # 200 frames, and marks only its historical time axes dynamic.
        generator_type = (
            SynthesizerTrnMs256NSFsid
            if version == "v1"
            else SynthesizerTrnMs768NSFsid
        )
        generator = generator_type(*config, is_half=False)
        generator.load_state_dict(weights, strict=False)
        generator.eval()
        model = GeneratorExport(generator).eval()

        frames = 200
        feats = torch.rand(1, frames, feature_channels)
        p_len = torch.tensor([frames], dtype=torch.long)
        pitch = torch.randint(low=5, high=255, size=(1, frames), dtype=torch.long)
        pitchf = torch.rand(1, frames)
        speaker = torch.zeros(1, dtype=torch.long)
        torch.onnx.export(
            model,
            (feats, p_len, pitch, pitchf, speaker),
            output_path,
            dynamic_axes={"feats": [1], "pitch": [1], "pitchf": [1]},
            # The legacy exporter emits invalid dynamic attention reshapes without
            # folding. Keep this enabled for compatibility with existing exports.
            do_constant_folding=True,
            opset_version=20,
            verbose=False,
            input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
            output_names=["audio"],
        )
    else:
        # RVC WebUI's legacy dynamic export traces attention at one Python T and
        # only relabels the public axes. TensorRT then sees internally fixed
        # reshapes when vc-rs requests another fixed profile. Trace the WebUI
        # six-input graph at the exact runtime T and leave every axis static.
        from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

        generator = SynthesizerTrnMsNSFsidM(
            *config, version=version, is_half=False
        )
        generator.load_state_dict(weights, strict=False)
        generator.eval()

        frames = fixed_frames
        phone = torch.rand(1, frames, feature_channels, dtype=torch.float32)
        phone_lengths = torch.tensor([frames], dtype=torch.int64)
        pitch = torch.randint(
            low=5, high=255, size=(1, frames), dtype=torch.int64
        )
        pitchf = torch.rand(1, frames, dtype=torch.float32)
        ds = torch.zeros(1, dtype=torch.int64)
        inter_channels = int(config[2])
        rnd = torch.rand(
            1, inter_channels, frames, dtype=torch.float32
        )
        torch.onnx.export(
            generator,
            (phone, phone_lengths, pitch, pitchf, ds, rnd),
            output_path,
            # Keep the legacy tracer explicitly: its Python shape decisions are
            # intentional here because this artifact is dedicated to one T.
            dynamo=False,
            do_constant_folding=True,
            opset_version=20,
            verbose=False,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
            output_names=["audio"],
        )

    # Preserve the source model's F0/sample-rate facts so vc-rs can size the
    # shared streaming pipeline correctly for 32/40/48 kHz exports.
    graph = onnx.load_model(output_path)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    sample_rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(
        checkpoint["sr"], checkpoint["sr"]
    )
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        fail(f"unsupported RVC checkpoint sample rate: {checkpoint['sr']!r}")
    metadata["metadata"] = json.dumps(
        {"f0": 1, "samplingRate": sample_rate}, separators=(",", ":")
    )
    del graph.metadata_props[:]
    for key, value in metadata.items():
        entry = graph.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(graph)
    onnx.save_model(graph, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int)
    arguments = parser.parse_args()
    if arguments.frames is not None and arguments.frames <= 0:
        parser.error("--frames must be greater than zero")
    export(arguments.model, arguments.output, arguments.frames)


if __name__ == "__main__":
    main()

# Third-Party Notices

`vc-rs` is an independently written Rust implementation for RVC-compatible
ONNX runtime behavior. The implementation and compatibility checks were
informed by public RVC ecosystem projects, but this repository does not vendor
their source files or RVC/voice-conversion pretrained model weights.

If a future change copies, translates, or includes substantial portions of code
from these or other third-party projects, keep the corresponding upstream
copyright and license notices with that code.

## RVC WebUI

- Repository: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
- License: MIT
- Upstream license copyright notices include liujing04, 源文雨, and Ftps.

## VCClient / w-okada voice-changer

- Repository: <https://github.com/w-okada/voice-changer>
- License: MIT
- Upstream license copyright notices include Wataru Okada, Isle Tennos,
  Jaehyeon Kim, liujing04, 源文雨, and yxlllc.

## Applio

- Repository: <https://github.com/IAHispano/Applio>
- License: MIT
- Upstream license copyright notices include AI Hispano.

## Silero VAD

- Pure-Rust runtime: <https://github.com/huanglizhuo/silero-vad-rust>
- VAD model/weights source: <https://github.com/snakers4/silero-vad>
- License: MIT

The `silero-vad-pure` dependency embeds generic Silero voice-activity-detection
weights in the executable. They are not RVC, ContentVec, RMVPE, or target voice
conversion weights. The dependency's full MIT notice is included in generated
distribution license material.

## External Model Weights

RVC, ContentVec, RMVPE, retrieval-index, and target voice model weights are not
included in this repository. The optional `download-models.ps1` helper downloads
third-party model files from
<https://huggingface.co/wok000/weights_gpl>. Those downloaded files are outside
the scope of this repository's MIT License; review the upstream model license
before using, modifying, or redistributing them.

The optional FCPE hybrid path likewise expects a user-supplied ONNX file. The
reference community export is published at
<https://huggingface.co/gzivdo/fcpe-onnx> and is marked MIT by its publisher.
vc-rs does not download, embed, or redistribute those FCPE weights; verify the
model card and any derivative-weight terms before packaging a model yourself.

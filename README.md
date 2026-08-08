# vc-rs

> [日本語](README.ja.md) | [English](README.en.md) | 简体中文

> [!IMPORTANT]
> **本项目基于上游 [`shirohata/vc-rs`](https://github.com/shirohata/vc-rs)**。
> 所有版权与致谢归原作者所有。
>
> **Base project:** https://github.com/shirohata/vc-rs

`vc-rs` 是一个 Rust 编写的 **RVC 声音转换应用**。可以将麦克风输入或 WAV 文件,
通过 ONNX 格式的 RVC 模型转换为另一种声音。有以下 3 种使用方式:

- **GUI 版(`vc-gui.exe`)** — 用于独立使用的桌面应用。**大多数用户仅需这一个。**
- **附带的 CLI(`vc-rs.exe`)** — 随 GUI 版附带的命令行工具,可用于 WAV 文件批量
  转换、诊断、Windows ML EP 管理、自动化等 GUI 没有的功能。详见
  [`docs/cli.md`](docs/cli.md)。
- **VST3 插件版(`vc-vst3.vst3`)** — 在 DAW 中加载使用的插件。

我们提供预编译的 Windows 包。**无需从源码构建。** 下载解压、准备模型即可立即
使用。

> 想从源码构建的开发者请参阅 [`docs/development_ja.md`](docs/development_ja.md)
> (日文)。内部设计见 [`docs/architecture.md`](docs/architecture.md)。

## 特点

- **原生 Rust 实现** — 无需 Python / PyTorch 运行时。没有 GC 停顿和解释器开销,
  实时处理时间的最差情况也能保持稳定。发行包也很轻量(windowsml 版仅数 MB)。
- **不易断音的实时设计** — 音频回调只用无锁方式搬运样本,将推理等重处理隔离到
  独立线程。即使负载升高也不会阻塞回调,失败时以丢输入/静音的方式继续运行。
- **广泛的 GPU 支持与最速模式** — Windows ML(DirectML)可在 NVIDIA 以外的 GPU
  上运行,也可选择原生 TensorRT 实现 NVIDIA GPU 的最快执行。
- **WAV 模式与实时模式走同一路径** — 可确定性地验证音质调校的效果。

> 转换本身的音质,只要使用相同的 RVC 模型,就与其他工具本质相同。vc-rs 的优势在
> 于实时场景的**稳定性(不易断音、易于压低延迟)**与**轻量、易上手**。

## 下载

最新版请从 GitHub 的 **[Releases](https://github.com/shirohata/vc-rs/releases)**
获取。发行包面向 Windows (x64),根据用途和环境分为以下 4 种:

| 包 | 形态 | 后端 | 适用环境 | 体积 | 需要准备 |
| --- | --- | --- | --- | --- | --- |
| `vc-rs-windowsml-…zip` | GUI + CLI | Windows ML | 大多数 GPU(含非 NVIDIA) | 小(数 MB) | Windows App SDK 运行时 |
| `vc-rs-tensorrt-…zip` | GUI + CLI | TensorRT | NVIDIA GPU | 大(约 1.9 GB) | 最新 NVIDIA 驱动 |
| `vc-vst3-windowsml-…zip` | VST3 插件 | Windows ML | 大多数 GPU(含非 NVIDIA) | 小 | Windows App SDK 运行时 |
| `vc-vst3-tensorrt-…zip` | VST3 插件 | TensorRT | NVIDIA GPU | 大(约 1.9 GB) | 最新 NVIDIA 驱动 |

**如何选择:**

- 想先试试的话选 **windowsml 版**。下载轻量,非 NVIDIA GPU 也能通过 DirectML
  运行。
- **拥有 NVIDIA GPU 并追求最速**的话选 **tensorrt 版**。下载较大,首次启动时引擎
  构建需要时间,之后会很快。
- 独立使用选 **GUI + CLI 版**,在 DAW 中唱歌或直播使用选 **VST3 版**。自动化和
  WAV 批量转换可使用 GUI 附带的 CLI。

## 需要准备

### windowsml 版

- 请安装 **Windows App SDK 运行时(2.x 系)**。它提供 ONNX Runtime 和 DirectML。
  请从 Microsoft 的
  [Windows App SDK 下载页面](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
  安装最新稳定版的 **Runtime(运行时)安装程序**。

### tensorrt 版

- **最新的 NVIDIA GPU 驱动**。TensorRT 本体 DLL 已随包附带,无需另行安装 CUDA
  或 TensorRT。

### 通用:模型文件

`vc-rs` 不附带模型。需要自己准备以下 3 个:

1. **RVC 声音转换模型**(`.onnx`) — 要转换的目标声音的模型。**仅支持 ONNX 格式**。
   无法直接加载 `.pth`(请先用 RVC 系工具或 VCClient 等预先转换为 `.onnx`)。
2. **嵌入提取模型**(ContentVec, `content_vec_500.onnx`)
3. **F0 估计模型**(RMVPE, `rmvpe.onnx`)

2 和 3 可通过附带的 `download-models.ps1` 获取(见下方"准备模型")。

## 使用方法(GUI 版)

1. 解压下载的 zip(**DLL 请保持与 `vc-gui.exe` 放在同一文件夹**)。
2. 通过下方"准备模型"获取嵌入、F0 模型。
3. 启动 `vc-gui.exe`。

### 准备模型

获取嵌入、F0 模型(在解压后的文件夹中执行)。

```powershell
pwsh .\download-models.ps1
```

会下载 `.\assets\content_vec_500.onnx` 和 `.\assets\rmvpe.onnx`。RVC 声音转换
模型(`.onnx`)请另行准备。

> 这些模型由第三方分发(分发方标注 GPL-3.0),不适用于 `vc-rs` 本体的 MIT
> License。使用、修改、再分发时请遵守分发方的许可证。详见 `download-models.ps1`
> 内的说明。

### 界面操作

1. **Models** — 通过 **Browse** 指定 RVC 模型、嵌入(ContentVec)、F0(RMVPE)的
   各个 `.onnx`。
2. **Model pool (live switch)** — 点击 **Add model…** 把额外的 RVC 模型加入
   实时切换池。模型在**后台加载**(列表实时显示加载进度),加载完成后点
   **Switch** 即可在运行中**实时切换模型,无需重启**。添加过的模型与最后激活的
   模型会持久化,下次启动后自动恢复。
3. **Provider** — 选择后端(windowsml 版:`windowsml` / `windowsml-directml` /
   `windowsml-nvtrtx` / `windowsml-cpu` / `cpu`;tensorrt 版:`tensorrt`)。也可选择
   **GPU Priority** 和用于 CUDA / TensorRT 的 **GPU Device ID**。
4. **Audio** — 选择输入、输出设备(**Refresh devices** 可重新获取)。留空则使用
   "System default"。勾选 **Enable monitor output** 可让第二个输出设备(例如
   耳机)同时播放转换后的信号,再选 **Monitor device**。运行中更改设备会**实时
   生效**:采样率相同的切换秒切、不重启会话,采样率变化时自动重启会话。
5. **Engine configuration (Apply to restart)** — 设置 **Chunk ms** /
   **Extra convert ms**(见下方"实时设置的调整")。这些改动,连同基础模型路径和
   Provider 一样,都需要按下 **Apply / Start** 才生效。
6. 按下 **Apply / Start** 应用并开始。通过 **Stop** 停止。
7. **Live parameters**(Pitch shift / Speaker ID / Input gain / Output gain /
   Monitor gain)实时生效。
8. **Input denoiser** — 在 `off` / `noise-gate` / `rnnoise` / `gtcrn` 之间
   **运行中实时切换**(详见下文);noise-gate 的阈值等参数也实时生效。
9. **Telemetry** 显示推理时间、输入输出 RMS、overrun/underrun,可确认断音和负载
   状况(推理时间超过 chunk 预算时会以颜色警告)。

设置会自动保存(`%APPDATA%\vc-rs\gui.toml`),下次启动时恢复。模型 3 件已加载时,
可在运行中通过 **Passthrough** 在块边界实时切换。Passthrough 期间停止 RVC 推理,
切回 RVC 时丢弃旧流上下文后重新开始。无模型的纯 passthrough 仍可使用,但该会话
中无法实时切换到 RVC。

输入噪声抑制可在 **Input denoiser** 中**运行中实时切换** `off` /
`noise-gate` / `rnnoise` / `gtcrn`:前三种在后台即时构建并热切换;GTCRN 首次切换
时在后台加载引擎(加载期间继续使用之前的降噪器),完成后自动热切换。Passthrough
也应用 Input gain、当前选择的输入噪声抑制、Output gain。RNNoise 使用内嵌模型,
无需额外模型。VST3 版不包含这些输入噪声抑制。

输入噪声抑制的定位:

| 模式 | 开销 | 质量 | 备注 |
| --- | --- | --- | --- |
| Noise Gate | 极小 | 仅阈值门限 | 内嵌 |
| RNNoise | 低 | 保守 | 内嵌、48 kHz |
| **GTCRN** | **低** | **良好** | **仅 standalone 版、16 kHz、需要模型** |

GTCRN 是超轻量(约 48K 参数)的语音增强模型,CPU 上也能实时运行。**standalone
(CLI/GUI)版**可用,Windows ML 版以 ONNX Runtime CPU、TensorRT 版以 native
TensorRT 运行。VST3 版不包含。固定延迟约 48 ms(16 kHz 的 STFT 重构 + 适配器
FIFO)。模型可通过 `download-models.ps1 -Gtcrn` 获取到 `assets\gtcrn\`,在 GUI 的
**GTCRN model dir** 或 CLI 的 `--gtcrn-model <dir>` 中指定。

## 实时设置的调整

通过 **Chunk ms** 和 **Extra convert ms** 平衡断音、延迟、CPU/GPU 负载。

- **Chunk ms**:每次处理汇总的音频长度。出现断音或负载升高时加大
  (`500` → `750` → `1000`)。越大越稳定,但输入到输出的体感延迟也增加。GPU 执行
  时有时可以用更小的值。
- **Extra convert ms**:传递给转换的前后文长度。加大有时会更稳定,但负载也增加。
  先尝试从 `100` ms 附近开始。

调整设置时,安全的做法是**先找到一个不断音的值,然后再缩小 Chunk ms 来降低延迟**。
Pitch / Speaker / Input、Output 增益可随时通过 Live parameters 调整。

## 附带的 CLI(进阶)

GUI + CLI 版附带 CLI `vc-rs.exe`。通常的转换用 GUI 即可完成,但 CLI 可用于
**GUI 没有的功能**。

- **WAV 文件批量转换**(GUI 仅实时转换)。
- **诊断、模型检查**(`doctor` / `devices` / `inspect`)。
- **Windows ML 执行提供程序(EP)的确认、安装**、**引擎缓存管理**。
- **自动化、脚本化**,以及调整 GUI 固定住的细微 DSP / 音频参数。

用法和命令列表请参阅 [`docs/cli.md`](docs/cli.md)。

## 使用方法(VST3 插件版)

1. 解压 zip,将 `vc-vst3-windowsml.vst3` 或 `vc-vst3-tensorrt.vst3` 复制到 VST3
   标准文件夹。
   - Windows:`%CommonProgramFiles%\VST3\`(例如 `C:\Program Files\Common Files\VST3`)
2. 在解压后的文件夹中执行 `pwsh .\download-models.ps1`,将嵌入、F0 模型获取到
   `.\assets\`(**请在解压后的文件夹中执行,而非安装目录**)。
3. 在 DAW 中加载插件并打开编辑器画面。
   - 通过 **Browse** 指定 RVC 模型、嵌入(ContentVec)、F0(RMVPE)的各个 `.onnx`。
   - 选择**后端**(windowsml 版:`windowsml` / `windowsml-directml` / `cpu`;
     tensorrt 版:`tensorrt`)。
   - 设置 **chunk size**(ms)(越大越稳定,但延迟增加)。
   - 按下 **Load / Reload** 应用。模型、后端、chunk 的更改在按下此按钮前不会生效。
   - Pitch / Speaker / Input、Output 增益实时生效,可作为 DAW 参数进行自动化和
     保存。

模型路径和设置按项目/预置保存。详见
[`crates/vc-vst3/README.md`](crates/vc-vst3/README.md)。

## 关于 TensorRT(tensorrt 版)

tensorrt 版使用**附带的 TensorRT 运行时**进行 GPU 执行,因此除 NVIDIA 驱动外
无需额外安装。

> ⚠️ TensorRT 在**首次启动或模型、输入形状改变时**会生成引擎,因此启动可能非常
> 耗时。第二次以后会复用引擎缓存,启动变快。

引擎缓存的位置、大小确认与清除(CLI 的 `engine-cache`)、详细性能特性请参阅
[`docs/cli.md`](docs/cli.md) 和
[`docs/tensorrt_performance_ja.md`](docs/tensorrt_performance_ja.md)(日文)。

## 故障排除 / FAQ

**Q. windowsml 版无法启动 / 模型加载失败**
A. 请确认已安装 **Windows App SDK 运行时(2.x 系)**(见"需要准备")。可通过附带
CLI 的 `.\vc-rs.exe doctor` 诊断运行所需的依赖。

**Q. 运行 exe 时出现 SmartScreen 警告**
A. 发行二进制未做代码签名,因此 Windows 可能会发出警告。请确认内容后,选择
"更多信息"→"仍要运行"。

**Q. VST3 版在 DAW 中崩溃**
A. 请确认插件文件夹中是否有 `onnxruntime_providers_cuda.dll` 等多余的 ONNX
Runtime 提供程序 DLL 混入。windowsml 版的捆绑包不包含 ONNX Runtime / DirectML /
CUDA 的 DLL(由系统的 Windows App SDK 运行时提供)。直接解压发行 zip 不会混入,但
从旧构建复制时请删除。

**Q. 无法加载 `.pth` 模型**
A. RVC 声音转换模型**仅支持 `.onnx`**。请先用 RVC 系工具或 VCClient 等转换为
ONNX。

**Q. 实时断音 / 延迟大**
A. 请参阅"实时设置的调整"。先加大 Chunk ms 止住断音,再压低延迟。

## 辅助脚本

`download-models.ps1` 是可选辅助脚本。它会从
[`wok000/weights_gpl`](https://huggingface.co/wok000/weights_gpl) 下载第三方参考
用 ONNX 模型(ContentVec / RMVPE)。获取的模型不包含在 `vc-rs` 本体内,也不适用本
仓库的 MIT License(分发方标注 GPL-3.0)。

## Acknowledgements

- 本实现参考了 RVC 系 OSS 实现的经验,尤其从 Applio、VCClient、RVC WebUI 的设计
  和实现技巧中学到了很多。
- 相关的 third-party notice 汇总在
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## License

MIT License(参见 [`LICENSE`](LICENSE))。关于外部项目与模型文件的注意事项请参阅
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

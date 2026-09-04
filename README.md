<div align="center">

# Text-to-Speech-TTS-ONNX

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED.svg)](https://onnxruntime.ai)

[English](#english) · [Models](#models) · [Performance](#performance)
<br>
[中文](#chinese) · [模型](#models) · [性能](#performance)

</div>

---

<a id="english"></a>
## English

**Text-to-speech, voice cloning, voice design, and vocoding with ONNX Runtime.**

Export and run TTS models using text and optional prompt audio. The repository contains export targets across the model families below, each with `Export`, `Inference`, and `Optimize` scripts.

The [shared model table](#models) lists public ONNX audio rates. Exporters handle native-rate conversion, while mode-dependent inputs are configured in the inference script.

The [shared performance table](#performance) reports RTF, calculated as synthesis time divided by audio duration. Lower is better. Unless noted, tests use a six-second reference and generate about 15 words. BigVGAN uses a `(1, 100, 512)` mel input.

### Audio Contract

- Set public audio I/O with `IN_SAMPLE_RATE`, `OUT_SAMPLE_RATE`, `IN_AUDIO_DTYPE`, and `OUT_AUDIO_DTYPE` in each exporter; KaniTTS and BigVGAN expose output settings only.
- Audio tensors support `F16`, `F32`, and `INT16`; floating point uses `[-1, 1]`, while `INT16` uses PCM amplitude.
- Resampling runs inside the ONNX graph with `torch.nn.functional.interpolate`.
- `Metadata.onnx` stores the fixed package contract; runtime controls remain in the inference script. Inference validates metadata, graph layout, and referenced files.
- Streaming exporters require each codec or latent frame to map to a whole number of output samples.
- Raon-OpenTTS is English-only in this package and requires reference audio with matching reference text.

---

<a id="chinese"></a>
## 中文

**基于 ONNX Runtime 的语音合成、声音克隆、声音设计与神经声码器。**

输入文本及可选提示音频即可导出并运行 TTS 模型。本项目包含下表所列模型家族的导出目标，每个目标均提供 `Export`、`Inference` 与 `Optimize` 脚本。

[共用模型表](#models)列出公开 ONNX 音频采样率。导出脚本负责原生采样率转换，模式相关输入则在推理脚本中配置。

[共用性能表](#performance)中的 RTF 为合成耗时除以音频时长，数值越低越好。除特别注明外，测试使用六秒参考音频并生成约 15 个单词。BigVGAN 使用 `(1, 100, 512)` mel 输入。

### 音频约定

- 在导出脚本中通过 `IN_SAMPLE_RATE`、`OUT_SAMPLE_RATE`、`IN_AUDIO_DTYPE` 与 `OUT_AUDIO_DTYPE` 设置公开音频输入输出；KaniTTS 与 BigVGAN 仅提供输出设置。
- 音频张量支持 `F16`、`F32` 与 `INT16`；浮点使用 `[-1, 1]`，`INT16` 使用 PCM 幅值。
- 重采样通过 `torch.nn.functional.interpolate` 在 ONNX 图内完成。
- `Metadata.onnx` 保存固定模型包约定，运行时配置保留在推理脚本中。推理时会校验元数据、图布局与引用文件。
- 流式导出要求每个编解码器帧或潜变量帧对应整数个输出采样点。
- 本项目中的 Raon-OpenTTS 仅支持英文，并且需要带有匹配文本的参考音频。


---

<a id="models"></a>
## Supported Models<br>支持的模型

| Capability<br>能力 | Model<br>模型 | Audio I/O<br>音频输入输出 | Prompt Audio<br>提示音频 | Code<br>代码 | Source<br>来源 |
| --- | --- | :---: | :---: | --- | --- |
| Neural vocoder<br>神经声码器 | BigVGAN V2 | Mel → 24 kHz | No<br>否 | [`BigVGAN`](./BigVGAN) | [GitHub](https://github.com/NVIDIA/BigVGAN) |
| Voice cloning · voice design · voice direction<br>声音克隆 · 声音设计 · 语音指导 | Breeze TTS 2 | 24 → 24 kHz | Mode-dependent<br>取决于模式 | [`Breeze_TTS/v2`](./Breeze_TTS/v2) | [GitHub](https://github.com/breezeblue-ai/breeze-tts) |
| Voice cloning<br>声音克隆 | F5-TTS | 24 → 24 kHz | Required<br>必需 | [`F5_TTS`](./F5_TTS) | [GitHub](https://github.com/SWivid/F5-TTS) |
| Clone · voice design · speech editing<br>克隆 · 声音设计 · 语音编辑 | FireRedTTS3 | 24 → 24 kHz | Mode-dependent<br>取决于模式 | [`FireRedTTS/v3`](./FireRedTTS/v3) | [GitHub](https://github.com/FireRedTeam/FireRedTTS3) |
| Voice cloning<br>声音克隆 | IndexTTS 1.5 | 24 → 24 kHz | Supported<br>支持 | [`Index_TTS/v1.5`](./Index_TTS/v1.5) | [GitHub](https://github.com/index-tts/index-tts) |
| Voice cloning · emotion control<br>声音克隆 · 情感控制 | IndexTTS 2 | 22.05 → 22.05 kHz | Supported<br>支持 | [`Index_TTS/v2`](./Index_TTS/v2) | [GitHub](https://github.com/index-tts/index-tts) |
| Voice cloning · emotion control<br>声音克隆 · 情感控制 | IndexTTS 2.5 | 22.05 → 22.05 kHz | Supported<br>支持 | [`Index_TTS/v2`](./Index_TTS/v2) | [GitHub](https://github.com/index-tts/index-tts) |
| Text-to-speech<br>语音合成 | Inflect | Text → 24 kHz | No<br>否 | [`Inflect`](./Inflect) | [GitHub](https://github.com/owenawsong/Inflect) |
| Text-to-speech<br>语音合成 | KaniTTS | Text → 22.05 kHz | No<br>否 | [`Kani_TTS`](./Kani_TTS) | [GitHub](https://github.com/nineninesix-ai/kani-tts) |
| Voice cloning · continuation<br>声音克隆 · 续写 | MOSS-TTS Nano | 48 → 48 kHz | Mode-dependent<br>取决于模式 | [`MOSS_TTS`](./MOSS_TTS) | [GitHub](https://github.com/OpenMOSS/MOSS-TTS) |
| Clone · custom voice · voice design<br>克隆 · 定制音色 · 声音设计 | Qwen3-TTS | 24 → 24 kHz | Mode-dependent<br>取决于模式 | [`Qwen_TTS`](./Qwen_TTS) | [GitHub](https://github.com/QwenLM/Qwen3-TTS) |
| English voice cloning<br>英文声音克隆 | Raon-OpenTTS-0.3B | 16 → 16 kHz | Required<br>必需 | [`Raon_OpenTTS`](./Raon_OpenTTS) | [GitHub](https://github.com/krafton-ai/Raon-OpenTTS) |
| English voice cloning<br>英文声音克隆 | Raon-OpenTTS-1B | 16 → 16 kHz | Required<br>必需 | [`Raon_OpenTTS`](./Raon_OpenTTS) | [GitHub](https://github.com/krafton-ai/Raon-OpenTTS) |
| Voice cloning<br>声音克隆 | VoxCPM 1.5 | 44.1 → 44.1 kHz | Supported<br>支持 | [`VoxCPM/v1.5`](./VoxCPM/v1.5) | [ModelScope](https://www.modelscope.cn/models/OpenBMB/VoxCPM1.5) |
| Clone · continuation · voice design<br>克隆 · 续写 · 声音设计 | VoxCPM 2 | 16 → 48 kHz | Mode-dependent<br>取决于模式 | [`VoxCPM/v2`](./VoxCPM/v2) | [ModelScope](https://www.modelscope.cn/models/OpenBMB/VoxCPM2) |
| Voice cloning · dialogue<br>声音克隆 · 对话 | ZipVoice | 24 → 24 kHz | Required<br>必需 | [`ZipVoice`](./ZipVoice) | [GitHub](https://github.com/k2-fsa/ZipVoice) |


---

<a id="performance"></a>
## Performance<br>性能

| OS<br>系统 | Device<br>设备 | Backend<br>后端 | Model<br>模型 | Precision<br>精度 | Time (s)<br>耗时（秒） | RTF |
| --- | --- | --- | --- | :---: | :---: | :---: |
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | F5-TTS · NFE=32 | f32 | 180 | 60 |
| Ubuntu 24.04 | NVIDIA GeForce MX150 | GPU | F5-TTS · NFE=32 | f32 | 62 | 21 |
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | IndexTTS | f32 | 18 | 6 |
| Ubuntu 24.04 | NVIDIA GeForce MX150 | GPU | BigVGAN V2 24khz_100band_256x | f16 | 4.6 | 1.53 |
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | KaniTTS | q8f32 | 8.4 | 1.4 |
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | KaniTTS | q4f32 | 5.2 | 0.87 |
| Ubuntu 24.04 | Intel Core i3-12300 | CPU | VoxCPM 1.5 | q8f32 | 9 | 1.5 |
| Ubuntu 24.04 | NVIDIA GeForce RTX 5060 Ti | GPU | VoxCPM 1.5 | f16 | 1.03 | 0.17 |
| Ubuntu 24.04 | Intel Core i3-12300 | CPU | Qwen3-TTS-0.6B-Base | q8f32 | 19 | 3.1 |
| Ubuntu 24.04 | Intel Core i3-12300 | CPU | VoxCPM 2 | q8f32 | 23 | 3.8 |
| Ubuntu 24.04 | NVIDIA GeForce RTX 5060 Ti | GPU | VoxCPM 2 | f16 | 2.05 | 0.34 |
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | FireRedTTS3 | q8f32 | 31.1 | 5.2 |

---

<div align="center">
<sub><b>Text-to-Speech-TTS-ONNX</b> · ONNX Runtime</sub>
<br>
<sub><a href="https://github.com/DakeQQ?tab=repositories">github.com/DakeQQ</a></sub>
</div>
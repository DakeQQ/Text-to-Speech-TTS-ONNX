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
- Hojo-TTS-Light-40M (V1/V2) exports `Hojo_LM.onnx` and `Hojo_Vocoder.onnx`; 80M adds `Hojo_ReferencePreprocess.onnx` for voice cloning with a matching reference transcript. Both support `<zh_{syllable}{tone}>` pronunciation overrides. Preset voices are rows of the voice archive: both graphs take a scalar `int32` `voice` input that gathers `speaker_embeds` (LM) and `speaker_vecs` (vocoder); clone packages have no `voice` input. Export `VOICES` / `--voices` keeps all voices or a unique, ordered subset given as 0-based indices or names; the package `voices` metadata defines the runtime indices. Select a voice with `VOICE_ID` (index or name), `--voice-id` or `--voice` (mutually exclusive), or per call with `voice_id=` / `voice=`; out-of-range or unknown voices list the indexed IDs, and language is not guessed from voice names. Generation stops at the first speech-end token.
- Audio8-TTS-Preview-0.1B voice cloning requires reference text that matches the reference audio. The prompt and the generated codec frames share a 2048-token context. The weights use the [Audio8 Community License v1.0](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b/blob/main/LICENSE) (revenue-capped) and are not included in this repository. Streaming requires a whole number of output samples per codec frame, so use an `OUT_SAMPLE_RATE` that is a multiple of 11025 Hz (44.1 kHz or 22.05 kHz).

### Hojo Package Contract

Re-export and re-optimize older Hojo packages for the current contract. `Hojo_Metadata.onnx` is required: consumers do not infer missing settings or parse the weighted LM to find metadata. Required fields describe `tts_mode`, `graph_layout`, audio rates/dtypes, vocabulary and contiguous audio-token bounds, attention dimensions, `hidden_size`, `num_layers`, `max_positions`, `rope_theta`, prompt-token IDs, and the exact graph/tokenizer filenames. Preset packages also require unique `voices`, speaker-placeholder IDs and `spk_slots`; clone packages require reference-token IDs and the reference-preprocess filename. Shared model/data filenames must appear together and both files must exist.

`Hojo_LM.onnx` always uses `float16` KV cache I/O (`past_key_values.*` and `present.*`), casting back to `float32` in F32/quantized graphs. Q8 has FP32 tensor activations, but the configured `ACCURACY_LEVEL=4` permits internal int8 activation quantization in MatMulNBits; it must not be described as pure FP32 arithmetic. The specialized `--precision F16` path keeps the sign-sensitive fine encoder in FP32 and the 80M decoder/reference preprocess in FP32, while using a half-precision LM and mixed 40M decoder. Common per-model precision plans are separate choices. Vocoders store only tokenizer-derived audio embedding rows. RoPE tables retain checkpoint-rounded inverse frequencies. Standard `RMSNormalization` makes optimized graphs checkable without test-only schemas; use a runtime supporting the emitted opset. The R3 80M package keeps the Transformer in F32 and explicitly selects a Q8 output head; its strict long-context gates remain 0.97 greedy agreement and 0.01 hidden error.

`--bundle-shared` (or `BUNDLE_SHARED_INITIALIZERS=True`) deduplicates multi-element floating-point and large tensors into one element-aligned shared blob, mapped once by the CPU runtime; byte- and dtype-identical audio embedding subranges can alias the LM table, and normalization weights are recorded in `shared_initializer_prepack_excluded` so ORT norm kernels prepack safely. Numerical gates are precision-specific: F32/F16 LM defaults require greedy agreement at least 0.97, hidden relative error at most 0.01, and clean EOS termination on unpunctuated Chinese reference prompts; Q8 defaults are 0.85 and 0.05. Vocoder SNR thresholds depend on the plan; use `--min-vocoder-snr-db 45` to require 45 dB explicitly. Sampler parity must be 100%. Preset packages retain a scalar `int32` `voice` read only by `Gather`; switching voices must change LM hidden states and waveforms. Threshold flags configure the gates; `--skip-verify` skips numerical checks, not structural checks. Reference codes must remain identical. Export and optimization both stage and validate before atomic publication; failed optimization preserves an existing package.

The canonical runtime offers opt-in `stream_audio()` in addition to unchanged sentence-level `stream()`, and loads either `tokenizer.json` or packaged `hojo_bpe.txt`/`hojo_added.txt` tables. `--advanced` (default on) adds a single-token decode graph with flattened MQA layout folding, symmetric FP32 LM attention/activation stabilization (`--stabilize-lm-fp32`, keeping weighted MatMuls, norms, `*_rope_add`, KV caches, and `hidden` in FP16), an in-graph counter RNG (SplitMix64), an FP16 streaming vocoder (`Hojo_Fine_Stream.onnx` + `Hojo_Decoder_Stream.onnx`) with FP32 internal iSTFT/OLA and a single FP16→FP32 playback boundary cast on `pcm`, topology-verified streaming geometry metadata (`fine_history_frames`, `stream_decoder_delay_frames`), precomputed speaker/coarse projections, and reference-text prefix reuse; whole-utterance `Hojo_Vocoder.onnx` is omitted by default in `--advanced` mode to prune ~12.6 MB of dead shared weights (pass `--keep-whole-vocoder` to retain it). Standard ORT builds default to `--kv-update concat` with zero native custom-op dependencies; `--kv-update inplaced_kv` emits the static BNSH `com.custom.llm::KVCacheUpdate` graph layout (verifiable/runnable with `--custom-ops-library` / `HOJO_KV_OPS_LIBRARY`), and `--export-tokenizer-tables` emits deterministic `hojo_bpe.txt`, `hojo_added.txt`, `hojo_chars.txt`, and `golden_prompt_ids.txt`. Stateful rendering requires 24 kHz F32 output and the CPU provider; select `--vocoder-backend onnx` for the whole-sentence ONNX path when `Hojo_Vocoder.onnx` is present. `--head-precision Q4/Q8` and `--audio-only-head` are explicit experiments, not hidden default vocabulary changes.

---

<a id="chinese"></a>
## 中文

**基于 ONNX Runtime 的语音合成、声音克隆、声音设计与神经声码器。**

输入文本及可选提示音频即可导出并运行 TTS 模型。本项目包含下表所列模型家族的导出目标，每个目标均提供 `Export`、`Inference` 与 `Optimize` 脚本。

[共用模型表](#models)列出公开 ONNX 音频采样率。导出脚本负责原生采样率转换，模式相关输入则在推理脚本中配置。

[共用性能表](#performance)中的 RTF 为合成耗时除以音频时长，数值越低越好。除特别注明外，测试使用六秒参考音频并生成约 15 个单词。BigVGAN 使用 `(1, 100, 512)` mel 输入。

### 音频约定

- 在导出脚本中通过 `IN_SAMPLE_RATE`、`OUT_SAMPLE_RATE`、`IN_AUDIO_DTYPE` 与 `OUT_AUDIO_DTYPE` 设置公开音频输入输出；KaniTTS 与 BigVGAN 仅提供输出设置。
- 音频张量支持 `F16`、`F32` 与 `INT16`; 浮点使用 `[-1, 1]`，`INT16` 使用 PCM 幅值。
- 重采样通过 `torch.nn.functional.interpolate` 在 ONNX 图内完成。
- `Metadata.onnx` 保存固定模型包约定，运行时配置保留在推理脚本中。推理时会校验元数据、图布局与引用文件。
- 流式导出要求每个编解码器帧或潜变量帧对应整数个输出采样点。
- 本项目中的 Raon-OpenTTS 仅支持英文，并且需要带有匹配文本的参考音频。
- Hojo-TTS-Light-40M（V1/V2）导出 `Hojo_LM.onnx` 与 `Hojo_Vocoder.onnx`；80M 额外导出 `Hojo_ReferencePreprocess.onnx`，声音克隆需要与参考音频匹配的文本。两者均支持 `<zh_{syllable}{tone}>` 拼音纠音标记。预置音色即音色文件中的行：两个图都接收标量 `int32` 输入 `voice`，分别从 `speaker_embeds`（LM）和 `speaker_vecs`（声码器）中 Gather；克隆包没有 `voice` 输入。导出时 `VOICES` / `--voices` 保留全部音色，或按给定顺序保留以 0 起始索引或名称指定的不重复子集；运行时索引以模型包元数据 `voices` 为准。通过 `VOICE_ID`（索引或名称）、`--voice-id` 或 `--voice`（二者互斥）选择音色，也可在调用时传入 `voice_id=` / `voice=`；越界或未知音色会列出带索引的可用 ID，不再根据音色名称猜测语言。生成在第一个语音结束 token 处停止。
- Audio8-TTS-Preview-0.1B 的声音克隆需要与参考音频匹配的参考文本。提示词与生成的编解码器帧共享 2048 个 token 的上下文。模型权重适用 [Audio8 Community License v1.0](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.1b/blob/main/LICENSE)（设有营收上限），本仓库不包含权重。流式解码要求每个编解码器帧对应整数个输出采样点，因此 `OUT_SAMPLE_RATE` 应为 11025 Hz 的整数倍（44.1 kHz 或 22.05 kHz）。

### Hojo 模型包约定

旧 Hojo 模型包需重新导出并优化，以采用当前约定。必须包含 `Hojo_Metadata.onnx`：使用方不会推断缺失设置，也不会解析带权重的 LM 来读取元数据。必需字段描述 `tts_mode`、`graph_layout`、音频采样率/类型、词表及连续音频 token 范围、注意力维度、`hidden_size`、`num_layers`、`max_positions`、`rope_theta`、提示 token ID，以及确切的模型/分词器文件名。预置音色包还需唯一的 `voices`、说话人占位 ID 与 `spk_slots`；克隆包需参考 token ID 和参考预处理文件名。共享模型/数据文件名必须同时出现，且两个文件都必须存在。

`Hojo_LM.onnx` 始终使用 `float16` KV 缓存输入输出，在 F32/量化图中再 Cast 回 `float32`。Q8 的张量激活为 FP32，但当前 `ACCURACY_LEVEL=4` 允许 MatMulNBits 内部进行 int8 激活量化，不能称为纯 FP32 计算。专用 `--precision F16` 路径将 LM 保留为半精度，将符号敏感的 fine 编码器及 80M 解码器/参考预处理保留为 FP32，40M 解码器采用混合精度；逐模型精度方案是不同选择。声码器只存储音频 token 对应的嵌入行。RoPE 保留检查点舍入后的逆频率。标准 `RMSNormalization` 无需测试专用 schema，但运行时须支持导出的 opset。R3 80M 包保留 F32 Transformer，并显式选择 Q8 输出头；长上下文仍须满足 0.97 贪心一致率和 0.01 隐藏状态误差门槛。

`--bundle-shared`（或 `BUNDLE_SHARED_INITIALIZERS=True`）将多元素浮点及大张量按元素对齐存入单个共享文件，由 CPU 运行时单次映射；数据类型和字节完全匹配的音频嵌入子区间可与 LM 表建立别名，归一化权重记录在 `shared_initializer_prepack_excluded` 中以保障 ORT 归一化算子安全预打包。数值阈值取决于精度：F32/F16 LM 默认要求贪心一致率至少 0.97、隐藏状态相对误差至多 0.01，并在无标点中文参考提示上正常输出 EOS；Q8 默认分别为 0.85 和 0.05。声码器 SNR 阈值取决于方案，可用 `--min-vocoder-snr-db 45` 明确要求 45 dB；采样器必须达到 100% 一致。预置音色保留仅由 `Gather` 读取的标量 `int32` 输入 `voice`，切换音色须同时改变隐藏状态和波形。阈值参数可配置；`--skip-verify` 只跳过数值检查。参考音频码须完全相同。导出和优化均先暂存验证再原子发布；验证失败时保留已有模型包。

标准运行时提供可选 `stream_audio()`，原句级 `stream()` 行为不变，并同时支持 `tokenizer.json` 与打包文本词表（`hojo_bpe.txt`/`hojo_added.txt`）。`--advanced`（默认开启）新增带布局折叠的单 token MQA 解码图、对称 FP32 LM 注意力/激活稳定化（`--stabilize-lm-fp32`，保持带权重的 MatMul、归一化、`*_rope_add`、KV 缓存与 `hidden` 为 FP16）、图内计数器 RNG（SplitMix64）、FP16 流式声码器（`Hojo_Fine_Stream.onnx` + `Hojo_Decoder_Stream.onnx`，内部 iSTFT/OLA 为 FP32 并在 `pcm` 末端附加单个 FP16→FP32 播放边界 Cast）、拓扑校验的流式几何元数据（`fine_history_frames`、`stream_decoder_delay_frames`）、说话人/音频嵌入投影预计算及参考文本前缀复用；`--advanced` 模式下默认省略整句 `Hojo_Vocoder.onnx` 以裁剪约 12.6 MB 冗余共享权重（可通过 `--keep-whole-vocoder` 保留）。标准 ORT 构建默认采用 `--kv-update concat` 无需自定义原生算子库；`--kv-update inplaced_kv` 可导出静态 BNSH `com.custom.llm::KVCacheUpdate` 原地 KV 图（配合 `--custom-ops-library` / `HOJO_KV_OPS_LIBRARY` 验证与运行），`--export-tokenizer-tables` 可确定性导出 `hojo_bpe.txt`、`hojo_added.txt`、`hojo_chars.txt` 与 `golden_prompt_ids.txt`。有状态渲染要求 24 kHz F32 输出与 CPU 执行提供器，保留 `Hojo_Vocoder.onnx` 时可用 `--vocoder-backend onnx` 选择整句 ONNX 路径。`--head-precision Q4/Q8`、`--audio-only-head` 是显式实验，不会静默缩减默认词表。

---

<a id="models"></a>
## Supported Models<br>支持的模型

| Capability<br>能力 | Model<br>模型 | Audio I/O<br>音频输入输出 | Prompt Audio<br>提示音频 | Code<br>代码 | Source<br>来源 |
| --- | --- | :---: | :---: | --- | --- |
| Text-to-speech · voice cloning<br>语音合成 · 声音克隆 | Audio8-TTS-Preview-0.1B | 44.1 → 44.1 kHz | Mode-dependent<br>取决于模式 | [`Audio8_TTS`](./Audio8_TTS) | [GitHub](https://github.com/Audio8-AI/Audio8_TTS) |
| Neural vocoder<br>神经声码器 | BigVGAN V2 | Mel → 24 kHz | No<br>否 | [`BigVGAN`](./BigVGAN) | [GitHub](https://github.com/NVIDIA/BigVGAN) |
| Voice cloning · voice design · voice direction<br>声音克隆 · 声音设计 · 语音指导 | Breeze TTS 2 | 24 → 24 kHz | Mode-dependent<br>取决于模式 | [`Breeze_TTS/v2`](./Breeze_TTS/v2) | [GitHub](https://github.com/breezeblue-ai/breeze-tts) |
| Voice cloning<br>声音克隆 | F5-TTS | 24 → 24 kHz | Required<br>必需 | [`F5_TTS`](./F5_TTS) | [GitHub](https://github.com/SWivid/F5-TTS) |
| Clone · voice design · speech editing<br>克隆 · 声音设计 · 语音编辑 | FireRedTTS3 | 24 → 24 kHz | Mode-dependent<br>取决于模式 | [`FireRedTTS/v3`](./FireRedTTS/v3) | [GitHub](https://github.com/FireRedTeam/FireRedTTS3) |
| Text-to-speech · preset voices<br>语音合成 · 预置音色 | Hojo-TTS-Light-40M | Text → 24 kHz | No<br>否 | [`Hojo_TTS`](./Hojo_TTS) | [GitHub](https://github.com/HojoAI/Hojo-TTS-Light) |
| Voice cloning<br>声音克隆 | Hojo-TTS-Light-80M | 24 → 24 kHz | Required<br>必需 | [`Hojo_TTS`](./Hojo_TTS) | [GitHub](https://github.com/HojoAI/Hojo-TTS-Light) |
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
| Ubuntu 24.04 | Intel Core i7-1165G7 | CPU | Audio8-TTS-Preview-0.1B | q8f32 · f16 KV | 21.5 | 2.7 |

---

<div align="center">
<sub><b>Text-to-Speech-TTS-ONNX</b> · ONNX Runtime</sub>
<br>
<sub><a href="https://github.com/DakeQQ?tab=repositories">github.com/DakeQQ</a></sub>
</div>
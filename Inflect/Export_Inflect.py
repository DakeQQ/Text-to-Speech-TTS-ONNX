"""Export the Inflect v2 neural path as deeply optimized ONNX graphs.

The public frontend remains Python/eSpeak-ng.  The neural path is split at the
duration expansion so the runner can build a compact frame-to-token index.  The
decode graph generates scaled latent noise internally, while still avoiding
VITS's dense ``[frames, tokens]`` alignment matrix.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import runpy
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from Shared_Weights import (
	SHARED_DATA_NAME,
	SHARED_MODEL_NAME,
	bundle_shared_initializers,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path.home() / "Downloads" / "Inflect-Micro-v2"  # [Inflect-Micro-v2, Inflect-Nano-v2]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Inflect_ONNX"
OPSET_VERSION = 20
DURATION_MODEL_NAME = "Inflect_Duration.onnx"
DECODE_MODEL_NAME = "Inflect_Decode.onnx"
METADATA_MODEL_NAME = "Inflect_Metadata.onnx"
MAX_FRAMES = 4000
MAX_TOKENS = 4000
FADE_MILLISECONDS = 5.0
OUT_SAMPLE_RATE = 24000
OUT_AUDIO_DTYPE = "F32"  	# F16 | F32 | INT16


OUTPUT_AUDIO_DTYPES = {"F16", "F32", "INT16"}
MODEL_SIGNATURES = {
	(128, 72, 384, 192, 2): "nano",
	(192, 96, 768, 320, 3): "micro",
}


def _copy_tensor(value: Tensor) -> Tensor:
	return value.detach().float().contiguous().clone()


class ChannelLayerNorm(nn.Module):
	"""Layer normalization over channels without layout transposes."""

	def __init__(self, source: nn.Module, *, affine: bool = True) -> None:
		super().__init__()
		self.eps = float(source.eps)
		self.affine = affine
		if affine:
			self.register_buffer("weight", _copy_tensor(source.gamma).view(1, -1, 1))
			self.register_buffer("bias", _copy_tensor(source.beta).view(1, -1, 1))

	def forward(self, values: Tensor) -> Tensor:
		mean = values.mean(dim=1, keepdim=True)
		centered = values - mean
		inverse_std = torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + self.eps)
		normalized = centered * inverse_std
		if self.affine:
			return normalized * self.weight + self.bias
		return normalized


class FusedRelativeAttention(nn.Module):
	"""Mask-free batch-one self-attention with fused and pre-scaled QKV."""

	def __init__(self, source: nn.Module) -> None:
		super().__init__()
		self.num_heads = int(source.n_heads)
		self.head_dim = int(source.k_channels)

		query_scale = self.head_dim**-0.5
		query_weight = _copy_tensor(source.conv_q.weight) * query_scale
		query_bias = _copy_tensor(source.conv_q.bias) * query_scale
		self.register_buffer(
			"qkv_weight",
			torch.cat(
				[
					query_weight,
					_copy_tensor(source.conv_k.weight),
					_copy_tensor(source.conv_v.weight),
				],
				dim=0,
			),
		)
		self.register_buffer(
			"qkv_bias",
			torch.cat(
				[
					query_bias,
					_copy_tensor(source.conv_k.bias),
					_copy_tensor(source.conv_v.bias),
				],
				dim=0,
			),
		)
		self.register_buffer("output_weight", _copy_tensor(source.conv_o.weight))
		self.register_buffer("output_bias", _copy_tensor(source.conv_o.bias))
		relative_table = torch.cat(
			(
				_copy_tensor(source.emb_rel_k[0]),
				_copy_tensor(source.emb_rel_v[0]),
			),
			dim=1,
		)
		self.register_buffer("relative_table", F.pad(relative_table, (0, 0, 1, 1)))

	def forward(
		self,
		values: Tensor,
		relative_index: Tensor,
		query_index: Tensor,
		key_index: Tensor,
		value_index: Tensor,
	) -> Tensor:
		qkv = F.conv1d(values, self.qkv_weight, self.qkv_bias)
		qkv = qkv.view(3, self.num_heads, self.head_dim, -1)
		query = qkv[query_index].transpose(1, 2)
		key = qkv[key_index]
		value = qkv[value_index].transpose(1, 2)
		relative_key, relative_value = self.relative_table[relative_index].split(
			self.head_dim,
			dim=2,
		)
		scores = torch.matmul(query, key)
		relative_scores = torch.bmm(
			query.transpose(0, 1),
			relative_key.transpose(1, 2),
		).transpose(0, 1)
		scores = scores + relative_scores
		attention = torch.softmax(scores, dim=2)

		output = torch.matmul(attention, value)
		relative_output = torch.bmm(
			attention.transpose(0, 1),
			relative_value,
		).transpose(0, 1)
		output = output + relative_output
		output = output.transpose(1, 2).reshape(
			1,
			self.num_heads * self.head_dim,
			-1,
		)
		return F.conv1d(output, self.output_weight, self.output_bias)


class OptimizedEncoderLayer(nn.Module):
	def __init__(self, encoder: nn.Module, index: int) -> None:
		super().__init__()
		attention = encoder.attn_layers[index]
		feed_forward = encoder.ffn_layers[index]
		self.attention = FusedRelativeAttention(attention)
		self.norm_after_attention = ChannelLayerNorm(encoder.norm_layers_1[index])
		self.norm_after_ffn = ChannelLayerNorm(encoder.norm_layers_2[index])
		self.padding = int(feed_forward.kernel_size) // 2
		self.register_buffer("ffn_weight_1", _copy_tensor(feed_forward.conv_1.weight))
		self.register_buffer("ffn_bias_1", _copy_tensor(feed_forward.conv_1.bias))
		self.register_buffer("ffn_weight_2", _copy_tensor(feed_forward.conv_2.weight))
		self.register_buffer("ffn_bias_2", _copy_tensor(feed_forward.conv_2.bias))

	def forward(
		self,
		values: Tensor,
		relative_index: Tensor,
		query_index: Tensor,
		key_index: Tensor,
		value_index: Tensor,
	) -> Tensor:
		values = self.norm_after_attention(
			values
			+ self.attention(
				values,
				relative_index,
				query_index,
				key_index,
				value_index,
			)
		)
		hidden = F.conv1d(
			values,
			self.ffn_weight_1,
			self.ffn_bias_1,
			padding=self.padding,
		)
		hidden = torch.relu(hidden)
		hidden = F.conv1d(
			hidden,
			self.ffn_weight_2,
			self.ffn_bias_2,
			padding=self.padding,
		)
		return self.norm_after_ffn(values + hidden)


class InflectDuration(nn.Module):
	"""Text encoder and duration predictor with no padded batch dimensions."""

	def __init__(self, model: nn.Module) -> None:
		super().__init__()
		text_encoder = model.enc_p
		duration = model.dp

		self.inter_channels = int(model.inter_channels)
		self.register_buffer(
			"embedding",
			_copy_tensor(text_encoder.emb.weight)
			.transpose(0, 1)
			.unsqueeze(0)
			.contiguous()
			* math.sqrt(text_encoder.hidden_channels),
		)
		self.layers = nn.ModuleList(
			[
				OptimizedEncoderLayer(text_encoder.encoder, index)
				for index in range(text_encoder.encoder.n_layers)
			]
		)
		window_sizes = {
			int(layer.window_size)
			for layer in text_encoder.encoder.attn_layers
		}
		window_size = window_sizes.pop()
		positions = torch.arange(MAX_TOKENS, dtype=torch.int16)
		relative_offset = positions.unsqueeze(0) - positions.unsqueeze(1)
		self.register_buffer(
			"relative_index",
			torch.clamp(
				relative_offset + window_size + 1,
				min=0,
				max=2 * window_size + 2,
			),
		)
		self.register_buffer("query_index", torch.tensor(0, dtype=torch.int32))
		self.register_buffer("key_index", torch.tensor(1, dtype=torch.int32))
		self.register_buffer("value_index", torch.tensor(2, dtype=torch.int32))
		self.register_buffer("stats_weight", _copy_tensor(text_encoder.proj.weight))
		self.register_buffer("stats_bias", _copy_tensor(text_encoder.proj.bias))

		self.duration_norm_1 = ChannelLayerNorm(duration.norm_1)
		self.duration_norm_2 = ChannelLayerNorm(duration.norm_2, affine=False)
		self.register_buffer("duration_weight_1", _copy_tensor(duration.conv_1.weight))
		self.register_buffer("duration_bias_1", _copy_tensor(duration.conv_1.bias))
		self.register_buffer("duration_weight_2", _copy_tensor(duration.conv_2.weight))
		self.register_buffer("duration_bias_2", _copy_tensor(duration.conv_2.bias))
		duration_output_weight = _copy_tensor(duration.proj.weight)
		duration_output_scale = _copy_tensor(duration.norm_2.gamma).view(1, -1, 1)
		duration_output_shift = _copy_tensor(duration.norm_2.beta).view(1, -1, 1)
		self.register_buffer(
			"duration_output_weight",
			duration_output_weight * duration_output_scale,
		)
		self.register_buffer(
			"duration_output_bias",
			_copy_tensor(duration.proj.bias)
			+ (duration_output_weight * duration_output_shift).sum(dim=(1, 2)),
		)
		self.duration_padding_1 = int(duration.conv_1.padding[0])
		self.duration_padding_2 = int(duration.conv_2.padding[0])

	def forward(self, token_ids: Tensor, speed: Tensor) -> tuple[Tensor, Tensor]:
		values = torch.index_select(self.embedding, 2, token_ids)
		token_count = token_ids.size(0)
		relative_index = self.relative_index[:token_count, :token_count].int()
		for layer in self.layers:
			values = layer(
				values,
				relative_index,
				self.query_index,
				self.key_index,
				self.value_index,
			)

		statistics = F.conv1d(values, self.stats_weight, self.stats_bias)
		statistics = statistics.transpose(1, 2).reshape(-1, 2 * self.inter_channels)
		means, log_scales = statistics.split(self.inter_channels, dim=1)
		priors = torch.cat((means, torch.exp(log_scales)), dim=1)

		hidden = F.conv1d(
			values,
			self.duration_weight_1,
			self.duration_bias_1,
			padding=self.duration_padding_1,
		)
		hidden = self.duration_norm_1(torch.relu(hidden))
		hidden = F.conv1d(
			hidden,
			self.duration_weight_2,
			self.duration_bias_2,
			padding=self.duration_padding_2,
		)
		hidden = self.duration_norm_2(torch.relu(hidden))
		log_duration = F.conv1d(
			hidden,
			self.duration_output_weight,
			self.duration_output_bias,
		).view(-1)
		durations = torch.ceil(torch.exp(log_duration) / speed).to(torch.int32)
		return priors, durations


class FrozenConv1d(nn.Module):
	def __init__(
		self,
		source: nn.Conv1d,
		*,
		reverse_inputs: bool = False,
		reverse_outputs: bool = False,
		weight_scale: float = 1.0,
	) -> None:
		super().__init__()
		weight = _copy_tensor(source.weight) * weight_scale
		bias = _copy_tensor(source.bias) if source.bias is not None else None
		if reverse_inputs:
			weight = weight.flip(1)
		if reverse_outputs:
			weight = weight.flip(0)
			if bias is not None:
				bias = bias.flip(0)
		self.register_buffer("weight", weight.contiguous())
		self.register_buffer("bias", bias.contiguous() if bias is not None else None)
		self.stride = tuple(source.stride)
		self.padding = tuple(source.padding)
		self.dilation = tuple(source.dilation)
		self.groups = int(source.groups)

	def forward(self, values: Tensor) -> Tensor:
		return F.conv1d(
			values,
			self.weight,
			self.bias,
			self.stride,
			self.padding,
			self.dilation,
			self.groups,
		)


class FrozenConvTranspose1d(nn.Module):
	def __init__(self, source: nn.ConvTranspose1d, *, weight_scale: float = 1.0) -> None:
		super().__init__()
		self.register_buffer("weight", _copy_tensor(source.weight) * weight_scale)
		self.register_buffer(
			"bias",
			_copy_tensor(source.bias) if source.bias is not None else None,
		)
		self.stride = tuple(source.stride)
		self.padding = tuple(source.padding)
		self.output_padding = tuple(source.output_padding)
		self.groups = int(source.groups)
		self.dilation = tuple(source.dilation)

	def forward(self, values: Tensor) -> Tensor:
		return F.conv_transpose1d(
			values,
			self.weight,
			self.bias,
			self.stride,
			self.padding,
			self.output_padding,
			self.groups,
			self.dilation,
		)


class MaskFreeWaveNet(nn.Module):
	"""Inference-only WaveNet block with zero conditioning and masks removed."""

	def __init__(self, source: nn.Module) -> None:
		super().__init__()
		self.hidden_channels = int(source.hidden_channels)
		self.input_layers = nn.ModuleList(FrozenConv1d(layer) for layer in source.in_layers)
		self.skip_layers = nn.ModuleList(
			FrozenConv1d(layer) for layer in source.res_skip_layers
		)

	def forward(self, values: Tensor) -> Tensor:
		skip: Tensor | None = None
		last_index = len(self.input_layers) - 1
		for index, (input_layer, skip_layer) in enumerate(
			zip(self.input_layers, self.skip_layers, strict=True)
		):
			projected = input_layer(values)
			projected_tanh, projected_sigmoid = projected.split(
				self.hidden_channels,
				dim=1,
			)
			activation = torch.tanh(projected_tanh)
			activation = activation * torch.sigmoid(projected_sigmoid)
			residual_skip = skip_layer(activation)
			if index != last_index:
				residual, current_skip = residual_skip.split(
					self.hidden_channels,
					dim=1,
				)
				values = values + residual
			else:
				current_skip = residual_skip
			skip = current_skip if skip is None else skip + current_skip
		return skip


class MeanOnlyCoupling(nn.Module):
	"""One inverse mean-only coupling, optionally with channel flips folded in."""

	def __init__(self, source: nn.Module, *, folded_flip: bool) -> None:
		super().__init__()
		self.pre = FrozenConv1d(source.pre, reverse_inputs=folded_flip)
		self.network = MaskFreeWaveNet(source.enc)
		self.post = FrozenConv1d(source.post, reverse_outputs=folded_flip)

	def forward(self, values: Tensor) -> Tensor:
		return self.post(self.network(self.pre(values)))


class FlipFreeInverseFlow(nn.Module):
	"""Four inverse couplings with all channel reversal nodes eliminated."""

	def __init__(self, source: nn.Module) -> None:
		super().__init__()
		couplings = list(source.flows[::2])
		self.half_channels = int(couplings[0].half_channels)
		self.coupling_3 = MeanOnlyCoupling(couplings[3], folded_flip=True)
		self.coupling_2 = MeanOnlyCoupling(couplings[2], folded_flip=False)
		self.coupling_1 = MeanOnlyCoupling(couplings[1], folded_flip=True)
		self.coupling_0 = MeanOnlyCoupling(couplings[0], folded_flip=False)

	def forward(self, values: Tensor) -> Tensor:
		first, second = values.split(self.half_channels, dim=1)
		first = first - self.coupling_3(second)
		second = second - self.coupling_2(first)
		first = first - self.coupling_1(second)
		second = second - self.coupling_0(first)
		return torch.cat((first, second), dim=1)


class FrozenResBlock(nn.Module):
	def __init__(self, source: nn.Module) -> None:
		super().__init__()
		self.first = nn.ModuleList(FrozenConv1d(layer) for layer in source.convs1)
		self.second = nn.ModuleList(FrozenConv1d(layer) for layer in source.convs2)

	def forward(self, values: Tensor) -> Tensor:
		for first, second in zip(self.first, self.second, strict=True):
			residual = first(F.leaky_relu(values, 0.1))
			residual = second(F.leaky_relu(residual, 0.1))
			values = values + residual
		return values


class FrozenGenerator(nn.Module):
	"""Weight-norm-free Inflect HiFi-GAN generator."""

	def __init__(self, source: nn.Module) -> None:
		super().__init__()
		branch_scale = 1.0 / int(source.num_kernels)
		self.pre = FrozenConv1d(source.conv_pre)
		self.upsamples = nn.ModuleList(
			FrozenConvTranspose1d(
				layer,
				weight_scale=1.0 if index == 0 else branch_scale,
			)
			for index, layer in enumerate(source.ups)
		)
		self.resblocks = nn.ModuleList(FrozenResBlock(layer) for layer in source.resblocks)
		self.post = FrozenConv1d(source.conv_post, weight_scale=branch_scale)
		self.num_kernels = int(source.num_kernels)

	def forward(self, values: Tensor) -> Tensor:
		values = self.pre(values)
		for stage, upsample in enumerate(self.upsamples):
			values = upsample(F.leaky_relu(values, 0.1))
			branch_start = stage * self.num_kernels
			mixed = self.resblocks[branch_start](values)
			for branch in range(1, self.num_kernels):
				mixed = mixed + self.resblocks[branch_start + branch](values)
			values = mixed
		return torch.tanh(self.post(F.leaky_relu(values)))


class ONNXDynamicThreeWaySplit(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		values: Tensor,
		edge_size: int,
	) -> tuple[Tensor, Tensor, Tensor]:
		middle_size = values.size(0) - 2 * edge_size
		return values.split((edge_size, middle_size, edge_size), dim=0)

	@staticmethod
	def symbolic(graph, values, edge_size):
		zero = graph.op(
			"Constant",
			value_t=torch.tensor([0], dtype=torch.int64),
		)
		edge = graph.op(
			"Constant",
			value_t=torch.tensor([edge_size], dtype=torch.int64),
		)
		double_edge = graph.op(
			"Constant",
			value_t=torch.tensor([2 * edge_size], dtype=torch.int64),
		)
		sample_count = graph.op("Gather", graph.op("Shape", values), zero, axis_i=0)
		middle = graph.op("Sub", sample_count, double_edge)
		split_sizes = graph.op("Concat", edge, middle, edge, axis_i=0)
		return graph.op("Split", values, split_sizes, axis_i=0, outputs=3)


class InflectDecode(nn.Module):
	"""Gather duration-expanded priors, apply inverse flow, and synthesize audio."""

	def __init__(
		self,
		model: nn.Module,
		model_sample_rate: int,
		output_sample_rate: int,
		output_audio_dtype: str,
		max_frames: int = 4000,
		fade_samples: int = 120,
	) -> None:
		super().__init__()
		output_audio_dtype = output_audio_dtype.upper()
		upsample_factor = math.prod(int(rate) for rate in model.upsample_rates)
		self.flow = FlipFreeInverseFlow(model.flow)
		self.generator = FrozenGenerator(model.dec)
		self.channels = int(model.inter_channels)
		self.max_frames = int(max_frames)
		self.fade_samples = int(fade_samples)
		self.output_resample_scale = float(output_sample_rate / model_sample_rate)
		self.output_audio_dtype = output_audio_dtype
		fade_in = torch.linspace(0.0, 1.0, self.fade_samples, dtype=torch.float32)
		self.register_buffer("fade_in", fade_in)
		self.register_buffer("fade_out", fade_in.flip(0).contiguous())

	def latent(
		self,
		priors: Tensor,
		frame_to_token: Tensor,
		variation: Tensor,
	) -> Tensor:
		frame_priors = priors[frame_to_token]
		means, scales = frame_priors.split(self.channels, dim=1)
		scaled_noise = torch.randn_like(means) * variation
		latent = means + scaled_noise * scales
		latent = latent.transpose(0, 1).unsqueeze(0)
		return self.flow(latent)

	def forward(
		self,
		priors: Tensor,
		frame_to_token: Tensor,
		variation: Tensor,
	) -> Tensor:
		latent = self.latent(priors, frame_to_token, variation)
		waveform = self.generator(latent[:, :, : self.max_frames]).view(-1)
		fade_in, waveform, fade_out = ONNXDynamicThreeWaySplit.apply(
			waveform,
			self.fade_samples,
		)
		waveform = torch.cat(
			(
				fade_in * self.fade_in,
				waveform,
				fade_out * self.fade_out,
			)
		)
		if self.output_resample_scale != 1.0:
			waveform = F.interpolate(
				waveform.view(1, 1, -1),
				scale_factor=self.output_resample_scale,
				mode="linear",
				align_corners=False,
				recompute_scale_factor=False,
			).view(-1)
		if self.output_audio_dtype == "INT16":
			return (waveform * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
		if self.output_audio_dtype == "F16":
			return waveform.half()
		return waveform


class MetadataCarrier(nn.Module):
	def forward(self, marker: Tensor) -> Tensor:
		return marker


def _detect_model_family(config: dict) -> str:
	model_config = config["model"]
	signature = (
		int(model_config["inter_channels"]),
		int(model_config["hidden_channels"]),
		int(model_config["filter_channels"]),
		int(model_config["upsample_initial_channel"]),
		int(model_config["n_layers_q"]),
	)
	return MODEL_SIGNATURES[signature]


def load_inflect_model(model_dir: Path) -> tuple[nn.Module, dict, list[str], str]:
	model_dir = model_dir.expanduser().resolve()
	runtime_dir = model_dir / "runtime"
	sys.path.insert(0, str(runtime_dir))

	SynthesizerTrn = importlib.import_module("models").SynthesizerTrn

	config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
	model_family = _detect_model_family(config)
	print(f"[Model] detected Inflect {model_family.title()} v2")
	symbols = runpy.run_path(str(runtime_dir / "text" / "symbols.py"))["symbols"]
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")
		model = SynthesizerTrn(
			len(symbols),
			config["data"]["filter_length"] // 2 + 1,
			config["train"]["segment_size"] // config["data"]["hop_length"],
			**config["model"],
		).float().eval()

	checkpoint = torch.load(model_dir / "model.pth", map_location="cpu", weights_only=False)
	model.load_state_dict(checkpoint["model"], strict=True)

	with contextlib.redirect_stdout(io.StringIO()):
		model.dec.remove_weight_norm()
	for flow in model.flow.flows:
		encoder = getattr(flow, "enc", None)
		if encoder is not None and hasattr(encoder, "remove_weight_norm"):
			encoder.remove_weight_norm()
	return model, config, list(symbols), model_family


def _set_metadata(path: Path, metadata: dict[str, str]) -> None:
	import onnx

	model = onnx.load(str(path), load_external_data=False)
	existing = {property_.key: property_ for property_ in model.metadata_props}
	for key, value in metadata.items():
		if key in existing:
			existing[key].value = str(value)
		else:
			model.metadata_props.add(key=str(key), value=str(value))
	onnx.save(model, str(path))


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for chunk in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _export_graph(
	module: nn.Module,
	arguments: tuple[Tensor, ...],
	path: Path,
	input_names: list[str],
	output_names: list[str],
	dynamic_axes: dict[str, dict[int, str]],
) -> None:
	torch.onnx.export(
		module,
		arguments,
		str(path),
		export_params=True,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
		do_constant_folding=True,
		keep_initializers_as_inputs=False,
		opset_version=OPSET_VERSION,
		training=torch.onnx.TrainingMode.EVAL,
		dynamo=False,
	)


def _export_inflect_package(
	model_dir: Path,
	output_dir: Path,
	max_frames: int,
	raw_dir: Path,
) -> None:

	output_audio_dtype = OUT_AUDIO_DTYPE.upper()
	model_dir = model_dir.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	for owned_name in (
		DURATION_MODEL_NAME,
		DECODE_MODEL_NAME,
		METADATA_MODEL_NAME,
		SHARED_MODEL_NAME,
		SHARED_DATA_NAME,
	):
		(output_dir / owned_name).unlink(missing_ok=True)

	model, config, symbols, model_family = load_inflect_model(model_dir)
	duration = InflectDuration(model).eval()
	sample_rate = int(config["data"]["sampling_rate"])
	fade_samples = round(sample_rate * FADE_MILLISECONDS / 1000.0)
	decode = InflectDecode(
		model,
		model_sample_rate=sample_rate,
		output_sample_rate=OUT_SAMPLE_RATE,
		output_audio_dtype=output_audio_dtype,
		max_frames=max_frames,
		fade_samples=fade_samples,
	).eval()
	channels = int(model.inter_channels)
	del model

	duration_path = output_dir / DURATION_MODEL_NAME
	decode_path = output_dir / DECODE_MODEL_NAME
	raw_duration_path = raw_dir / "duration.onnx"
	raw_decode_path = raw_dir / "decode.onnx"
	metadata_path = output_dir / METADATA_MODEL_NAME
	dummy_tokens = torch.arange(17, dtype=torch.int32) % len(symbols)
	dummy_speed = torch.tensor(1.0, dtype=torch.float32)
	dummy_priors, dummy_durations = duration(dummy_tokens, dummy_speed)
	dummy_frame_to_token = torch.arange(
		dummy_tokens.numel(),
		dtype=torch.int32,
	).repeat_interleave(
		dummy_durations
	)
	dummy_variation = torch.tensor(0.667, dtype=torch.float32)

	print(f"[Raw export] {raw_duration_path.name}")
	_export_graph(
		duration,
		(dummy_tokens, dummy_speed),
		raw_duration_path,
		["token_ids", "speed"],
		["priors", "durations"],
		{
			"token_ids": {0: "token_count"},
			"priors": {0: "token_count"},
			"durations": {0: "token_count"},
		},
	)
	print(f"[Raw export] {raw_decode_path.name}")
	_export_graph(
		decode,
		(dummy_priors, dummy_frame_to_token, dummy_variation),
		raw_decode_path,
		["priors", "frame_to_token", "variation"],
		["waveform"],
		{
			"priors": {0: "token_count"},
			"frame_to_token": {0: "frame_count"},
			"waveform": {0: "sample_count"},
		},
	)
	raw_hashes = {
		raw_duration_path: _file_sha256(raw_duration_path),
		raw_decode_path: _file_sha256(raw_decode_path),
	}
	for raw_path, final_path in (
		(raw_duration_path, duration_path),
		(raw_decode_path, decode_path),
	):
		shutil.copy2(raw_path, final_path)
	print("[Raw export] staged temporary source-optimized graphs")

	upsample_factor = math.prod(int(rate) for rate in config["model"]["upsample_rates"])
	metadata = {
		"format": "inflect_onnx_runtime_v4",
		"source_model": f"Inflect-{model_family.title()}-v2",
		"graph_layout": "duration_decode",
		"duration_model_file": DURATION_MODEL_NAME,
		"decode_model_file": DECODE_MODEL_NAME,
		"shared_initializer_model_file": SHARED_MODEL_NAME,
		"shared_initializer_data_file": SHARED_DATA_NAME,
		"model_sample_rate": str(sample_rate),
		"out_sample_rate": str(OUT_SAMPLE_RATE),
		"output_audio_dtype": output_audio_dtype,
		"inter_channels": str(channels),
		"upsample_factor": str(upsample_factor),
		"max_frames": str(max_frames),
		"max_tokens": str(MAX_TOKENS),
		"max_audio_samples": str(
			max_frames * upsample_factor * OUT_SAMPLE_RATE // sample_rate
		),
		"semantic_integer_dtype": "int32",
		"fade_samples": str(fade_samples),
		"fade_milliseconds": str(FADE_MILLISECONDS),
		"add_blank": "1" if config["data"]["add_blank"] else "0",
		"symbols_json": json.dumps(symbols, ensure_ascii=False, separators=(",", ":")),
		"opset": str(OPSET_VERSION),
		"optimization": (
			"fused_embedding_scale,pre_shaped_embedding,fused_qkv,"
			"fused_attention_scale,channel_layer_norm,"
			"fused_relative_kv_lookup,sentinel_relative_mask,"
			"precomputed_relative_index_slice,batched_relative_matmul,"
			"mask_free_batch1,token_scale_exp,gather_duration_expansion,"
			"split_channel_partitions,folded_duration_output_affine,"
			"static_statistics_reshape,flip_free_flow,folded_branch_scales,"
			"removed_weight_norm,graph_random_normal_noise,int32_semantic_indices,"
			"graph_edge_fade,graph_output_resample,graph_output_dtype"
		),
	}
	for path in (duration_path, decode_path):
		_set_metadata(path, metadata)
	shared_stats = bundle_shared_initializers(
		output_dir,
		[duration_path, decode_path],
		metadata,
	)
	print(
		f"[Shared weights] {shared_stats['initializer_references']} references -> "
		f"{shared_stats['unique_initializers']} unique tensors, "
		f"{shared_stats['unique_bytes'] / (1024 * 1024):.2f} MiB"
	)
	print(f"[Export] {metadata_path.name}")
	_export_graph(
		MetadataCarrier(),
		(torch.zeros(1, dtype=torch.int32),),
		metadata_path,
		["metadata_marker"],
		["metadata_marker_out"],
		{},
	)
	_set_metadata(metadata_path, metadata)

	print(f"Inflect ONNX package written to {output_dir}")


def export_inflect(model_dir: Path, output_dir: Path, max_frames: int) -> None:
	with tempfile.TemporaryDirectory(prefix="inflect_raw_") as temporary:
		_export_inflect_package(model_dir, output_dir, max_frames, Path(temporary))
	print("[Raw cleanup] temporary source graphs removed")


def main() -> None:
	export_inflect(DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, MAX_FRAMES)


if __name__ == "__main__":
	main()
	print("\nStart running the Inflect demo via Inference_Inflect_ONNX.py ...")
	raise SystemExit(subprocess.call([
		sys.executable,
		str(SCRIPT_DIR / "Inference_Inflect_ONNX.py"),
		"--onnx-folder",
		str(DEFAULT_OUTPUT_DIR),
	]))

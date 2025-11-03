import gc
import site
import shutil
import soundfile as sf
import numpy as np
import onnxruntime
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

shutil.copyfile(r'./modeling_modified/common.py', site.getsitepackages()[-1] + r'/nemo/core/classes/common.py')
from nemo.collections.tts.models import AudioCodecModel


path_kani = r'/home/DakeQQ/Downloads/kani-tts-370m'                                # Set the folder path where the [kani-tts-370m, kani-tts-400m] project downloaded.
path_codec = r'/home/DakeQQ/Downloads/nemo-nano-codec-22khz-0.6kbps-12.5fps/nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo' # The audio codec download path. URL: https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps
onnx_model_A = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Embed.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Main.onnx'            # Assign a path where the exported KaniTTS model stored.
onnx_model_C = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Greedy_Search.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_D = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/First_Beam_Search.onnx'       # Assign a path where the exported KaniTTS model stored.
onnx_model_E = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Second_Beam_Search.onnx'      # Assign a path where the exported KaniTTS model stored.
onnx_model_F = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Reset_Penality.onnx'          # Assign a path where the exported KaniTTS model stored.
onnx_model_G = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Codec.onnx'           # Assign a path where the exported KaniTTS model stored.
generated_audio_path = r"./generated.wav"                                          # The generated audio path.

target_tts = ["大家好，我现在正在大可奇奇体验AI科技。"]                                 # The test query after the export process.
speaker = 'ming'

"""
kani-tts-370m multilingual:
    Speaker List:
        david — David, English (British)
        puck — Puck, English (Gemini)
        kore — Kore, English (Gemini)
        andrew — Andrew, English
        jenny — Jenny, English (Irish)
        simon — Simon, English
        katie — Katie, English
        seulgi — Seulgi, Korean
        bert — Bert, German
        thorsten — Thorsten, German (Hessisch)
        maria — Maria, Spanish
        mei — Mei, Chinese (Cantonese)
        ming — Ming, Chinese (Shanghai OpenAI)
        karim — Karim, Arabic
        nur — Nur, Arabic
"""

STOP_TOKEN = [64402]                                                           # The stop_id in KaniTTS is "64402"
MAX_SEQ_LEN = 1024                                                             # The max decode length.
REPEAT_PENALITY = 0.9                                                          # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                                                            # Penalizes the most recent output. "30" means the last 30 tokens.
USE_BEAM_SEARCH = False                                                        # Use beam search or greedy search.
TOP_K = 5                                                                      # The top k candidate in decoding.
BEAM_SIZE = 5                                                                  # Number of beams in searching.
MAX_BEAM_SIZE = 10                                                             # Max beams for exported model.
SAMPLE_RATE = 22050                                                            # The sample rate of output audio.
MAX_THREADS = 0                                                                # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                                                                  # Default to zero.


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, logits, repeat_penality, penality_value, batch_size):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        batch_indices = self.batch_indices[:batch_size].long()
        repeat_penality[batch_indices, max_logits_idx.squeeze(-1)] *= penality_value
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-8]
        save_id = all_inputs[-7]
        repeat_penality = all_inputs[-6]
        previous_prob = all_inputs[-5]
        batch_indices = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[[0]]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count


class KANITTS_EMBED(torch.nn.Module):
    def __init__(self, kani_tts):
        super(KANITTS_EMBED, self).__init__()
        self.kani_tts = kani_tts

    def forward(self, input_ids):
        return self.kani_tts.model.embed_tokens(input_ids)


class KANITTS_MAIN(torch.nn.Module):
    def __init__(self, kani_tts, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, num_conv_layers, num_attn_layers):
        super(KANITTS_MAIN, self).__init__()
        self.kani_tts = kani_tts
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_conv_layers = num_conv_layers
        self.num_attn_layers = num_attn_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-5)

        scale_factor = float(head_dim ** -0.25)
        for layer in self.kani_tts.model.layers:
            if layer.is_attention_layer:
                layer.self_attn.q_layernorm.weight.data *= scale_factor
                layer.self_attn.k_layernorm.weight.data *= scale_factor

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(0)
        inv_freq_expanded = self.kani_tts.model.pos_emb.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_rotary_pos_emb = (emb.cos() * self.kani_tts.model.pos_emb.attention_scaling).half().unsqueeze(0)
        self.sin_rotary_pos_emb = (emb.sin() * self.kani_tts.model.pos_emb.attention_scaling).half().unsqueeze(0)
        self.num_key_value_layers = self.num_attn_layers + self.num_attn_layers
        self.save_key = [None] * self.num_attn_layers
        self.save_value = [None] * self.num_attn_layers
        self.save_conv = [None] * self.num_conv_layers

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-3]
        history_len = all_inputs[-2]
        ids_len = all_inputs[-1]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        batch_size = hidden_states.shape[0].unsqueeze(0)
        kv_count = 0
        conv_count = 0
        for i, layer in enumerate(self.kani_tts.model.layers):
            hidden_states_norm = layer.operator_norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            if layer.is_attention_layer:
                q = layer.self_attn.q_proj(hidden_states_norm).view(batch_size, -1, self.num_heads, self.head_dim)
                k = layer.self_attn.k_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
                v = layer.self_attn.v_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
                q = (layer.self_attn.q_layernorm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(1, 2)
                k = (layer.self_attn.k_layernorm.weight * (k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(0, 3, 2, 4, 1)
                q = q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
                k = k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
                k = torch.cat((all_inputs[kv_count], k), dim=-1)
                v = torch.cat((all_inputs[kv_count + self.num_attn_layers], v), dim=-2)
                self.save_key[kv_count] = k
                self.save_value[kv_count] = v
                kv_count += 1
                k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                attn = torch.nn.functional.softmax(torch.matmul(q, k), dim=-1, dtype=torch.float32)
                attn_out = layer.self_attn.out_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, layer.self_attn.out_proj.in_features))
            else:
                BCx = layer.conv.in_proj(hidden_states_norm).transpose(-1, -2)
                B, C, x = BCx.chunk(3, dim=-2)
                Bx = B * x
                conv_state = torch.cat([all_inputs[conv_count + self.num_key_value_layers], Bx], dim=-1)
                if i == 0:
                    len_conv_state = conv_state.shape[-1].unsqueeze(0)
                self.save_conv[conv_count] = conv_state[..., -2:]
                conv_count += 1
                conv_out = layer.conv.conv(conv_state)[..., :len_conv_state]
                conv_out = conv_out[..., -ids_len:]
                attn_out = layer.conv.out_proj((C * conv_out).transpose(-1, -2).contiguous())
            hidden_states += attn_out
            hidden_states = hidden_states + layer.feed_forward(layer.ffn_norm(hidden_states))
        hidden_states = hidden_states[:, -1]
        hidden_states = self.kani_tts.model.embedding_norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        logits = self.kani_tts.lm_head(hidden_states)
        return *self.save_key, *self.save_value, *self.save_conv, logits, kv_seq_len


class NEMO_CODEC(torch.nn.Module):
    def __init__(self, nemo_codec, tokeniser_length):
        super(NEMO_CODEC, self).__init__()
        self.nemo_codec = nemo_codec
        self.tokeniser_length = tokeniser_length
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032
        self.codebook = (torch.tensor([self.codebook_size * i for i in range(4)], dtype=torch.int32) + self.audio_tokens_start).view(1, 1, -1)
        self.scale = float(SAMPLE_RATE / 22050.0)

    def forward(self, decode_ids, num_decode):
        audio_codes = decode_ids[[0], 2:num_decode].reshape(1, -1, 4)
        len_ = audio_codes.shape[1].unsqueeze(0)
        audio_codes = audio_codes - self.codebook
        audio_codes = audio_codes.transpose(1, 2)
        reconstructed_audio, _ = self.nemo_codec.decode(tokens=audio_codes, tokens_len=len_)
        reconstructed_audio = reconstructed_audio.view(1, 1, -1)
        if self.scale != 1.0:
            reconstructed_audio = torch.nn.functional.interpolate(
                reconstructed_audio,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False
            )
        audio_out = (reconstructed_audio.clamp(min=-1.0, max=1.0) * 32767.0).to(torch.int16)
        return audio_out


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path_kani, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.model.layers._modules['2'].self_attn.head_dim
    num_layers = model.config.num_hidden_layers
    num_conv_layers = model.config.layer_types.count("conv")
    num_attn_layers = num_layers - num_conv_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    hidden_size = model.model.embed_tokens.embedding_dim
    vocab_size = model.vocab_size

    embed = KANITTS_EMBED(model)
    input_ids = torch.zeros((3, 10), dtype=torch.int32)
    torch.onnx.export(
        embed,
        (input_ids,),
        onnx_model_A,
        input_names=['input_ids'],
        output_names=['hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'hidden_state': {0: 'batch', 1: 'ids_len'},
        },
        do_constant_folding=True,
        opset_version=17
    )
    del embed
    del input_ids

    # Build an optimized model
    kani_tts = KANITTS_MAIN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers, num_conv_layers, num_attn_layers)
    batch_size = 3                                    # "3" is just a dummy value.
    ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
    hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    history_len = torch.tensor([0], dtype=torch.int64)
    past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)
    conv_states = torch.zeros((batch_size, hidden_size, 0), dtype=torch.float32)

    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}
    for i in range(num_attn_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus_ids_len'}
    for i in range(num_attn_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(num_conv_layers):
        name = f'in_conv_{i}'
        input_names.append(name)
        all_inputs.append(conv_states)
        dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
        name = f'out_conv_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    output_names.append('logits')
    output_names.append('kv_seq_len')
    dynamic_axes['logits'] = {0: 'batch'}

    torch.onnx.export(
        kani_tts,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del hidden_states
    del ids_len
    del history_len
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del kani_tts
    del model

    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_value = torch.tensor(REPEAT_PENALITY, dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (logits, repeat_penality, penality_value, beam_size),
        # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_C,
        input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del greedy

    first_beam_search = FIRST_BEAM_SEARCH(num_attn_layers + num_attn_layers + num_conv_layers)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
    past_keys_greedy = past_keys[[0]]
    past_values_greedy = past_values[[0]]
    conv_states_greedy = conv_states[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(num_attn_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys_greedy)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus_ids_len'}
    for i in range(num_attn_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values_greedy)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(num_conv_layers):
        name = f'in_conv_{i}'
        input_names.append(name)
        all_inputs.append(conv_states_greedy)
        dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
        name = f'out_conv_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('top_beam_indices')
    output_names.append('save_id_out')
    output_names.append('repeat_penality_out')
    output_names.append('top_beam_prob')
    output_names.append('batch_indices')
    output_names.append('max_logits_idx')
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['repeat_penality_in'] = {0: 'batch'}
    dynamic_axes['repeat_penality_out'] = {0: 'batch'}
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}
    dynamic_axes['batch_indices'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del first_beam_search

    all_inputs = []
    input_names = []
    for i in range(num_attn_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
    for i in range(num_attn_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
    for i in range(num_conv_layers):
        name = f'in_conv_{i}'
        input_names.append(name)
        all_inputs.append(conv_states)
    input_names.append('logits')
    all_inputs.append(logits)
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('batch_indices')
    all_inputs.append(batch_indices)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}
    output_names.remove("batch_indices")

    second_beam_search = SECOND_BEAM_SEARCH(num_attn_layers + num_attn_layers + num_conv_layers)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
        onnx_model_F,
        input_names=['save_id_in', 'repeat_penality_in', 'penality_reset_count_in', 'batch_indices'],
        output_names=['save_id_out', 'repeat_penality_out', 'penality_reset_count_out'],
        dynamic_axes={
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'penality_reset_count_in': {0: 'batch'},
            'penality_reset_count_out': {0: 'batch'},
            'batch_indices': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )

    decode_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32)  # Dummy values
    num_decode = torch.tensor([decode_ids.shape[-1]], dtype=torch.int64)
    nemo_codec = AudioCodecModel.from_pretrained(path_codec, map_location=torch.device('cpu')).float().eval()
    tokeniser_length = AutoTokenizer.from_pretrained(path_kani).vocab_size
    nemo_codec = NEMO_CODEC(nemo_codec, tokeniser_length)
    torch.onnx.export(
        nemo_codec,
        (decode_ids, num_decode),
        onnx_model_G,
        input_names=['decode_ids', 'num_decode'],
        output_names=['audio_out'],
        dynamic_axes={
            'decode_ids': {0: 'batch_size', 1: 'num_decode'},
            'audio_out': {2: 'audio_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del decode_ids
    del nemo_codec
    del num_decode
    del tokeniser_length
    del batch_indices
    del reset_penality
    del second_beam_search
    del past_keys
    del past_values
    del conv_states
    del past_keys_greedy
    del past_values_greedy
    del conv_states_greedy
    del logits
    del previous_prob
    del save_id
    del repeat_penality
    del penality_reset_count
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()

print('\nExport done!\n\nStart running the KaniTTS by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# Run the exported model by ONNX Runtime
#  settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                  # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ORT_Accelerate_Providers = ['CPUExecutionProvider']
device_type = 'cpu'
provider_options = None

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_B.get_providers()}")
model_dtype = ort_session_B._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]


generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
num_keys_values = 0
for i in ort_session_B._outputs_meta:
    if len(i.shape) == 5:
        num_keys_values += 1
num_keys_values_convs = amount_of_outputs_B - 2
num_conv_layers = num_keys_values_convs - num_keys_values
num_layers = num_keys_values // 2
num_keys_values_convs_plus_1 = num_keys_values_convs + 1
num_keys_values_convs_plus_2 = num_keys_values_convs + 2
num_keys_values_convs_plus_3 = num_keys_values_convs + 3
num_keys_values_convs_plus_4 = num_keys_values_convs + 4
num_keys_values_convs_plus_5 = num_keys_values_convs + 5
num_keys_values_convs_plus_6 = num_keys_values_convs + 6
num_keys_values_convs_plus_7 = num_keys_values_convs + 7
vocab_size = ort_session_B._outputs_meta[num_keys_values_convs].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=model_dtype), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(path_kani)
head_ids = np.array([[64403]], dtype=np.int32)       # For non-Python tokenizer = [64403, 1]
tail_ids = np.array([[2, 64404]], dtype=np.int32)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]
    
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]

    input_feed_D = {
        in_name_D[num_keys_values_convs_plus_3]: penality_value,
        in_name_D[num_keys_values_convs_plus_4]: beam_size
    }

    input_feed_E = {
        in_name_E[num_keys_values_convs_plus_5]: penality_value,
        in_name_E[num_keys_values_convs_plus_6]: beam_size,
        in_name_E[num_keys_values_convs_plus_7]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {in_name_C[2]: penality_value}


ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_G = ort_session_G.get_inputs()
out_name_G = ort_session_G.get_outputs()
in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]


if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False


init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=model_dtype), device_type, DEVICE_ID)
init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=model_dtype), device_type, DEVICE_ID)
init_conv_states_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_keys_values].shape[1], 0), dtype=model_dtype), device_type, DEVICE_ID)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)


# Start to run
for sentence in target_tts:
    sentence = f"{speaker}: {sentence}"
    print(f"\n{sentence}")
    input_ids = tokenizer(sentence, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = np.concatenate([head_ids, input_ids, tail_ids], axis=1)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[1]], dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    ids_len_1 = init_ids_len_1
    history_len = init_history_len
    past_keys_B = init_past_keys_B
    past_values_B = init_past_values_B
    conv_states_B = init_conv_states_B
    save_id = init_save_id
    repeat_penality = init_repeat_penality

    start_time = time.time()

    input_feed_A = {in_name_A: input_ids}
    hidden_states = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

    input_feed_B = {
            in_name_B[num_keys_values_convs]: hidden_states,
            in_name_B[num_keys_values_convs_plus_1]: history_len,
            in_name_B[num_keys_values_convs_plus_2]: ids_len
        }

    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B
    for i in range(num_keys_values, num_keys_values_convs):
        input_feed_B[in_name_B[i]] = conv_states_B

    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
        input_feed_D[in_name_D[num_keys_values_convs_plus_1]] = save_id_beam
        input_feed_D[in_name_D[num_keys_values_convs_plus_2]] = repeat_penality
    else:
        input_feed_C[in_name_C[1]] = repeat_penality
        input_feed_C[in_name_C[3]] = init_batch_size_greedy

    if do_repeat_penality:
        if USE_BEAM_SEARCH:
            input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
        else:
            penality_reset_count_greedy = 0

    num_decode = 0
    start_decode = time.time()
    while num_decode < generate_limit:
        all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
        if USE_BEAM_SEARCH:
            if num_decode < 1:
                input_feed_D.update(zip(in_name_D[:num_keys_values_convs_plus_1], all_outputs_B))
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logits_idx = all_outputs_D[num_keys_values_convs_plus_5].numpy()
                input_feed_E[in_name_E[num_keys_values_convs_plus_4]] = all_outputs_D[num_keys_values_convs_plus_4]
                if do_repeat_penality:
                    input_feed_F[in_name_F[3]] = all_outputs_D[num_keys_values_convs_plus_4]
            else:
                input_feed_E.update(zip(in_name_E[:num_keys_values_convs_plus_1], all_outputs_B))
                all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                max_logits_idx = all_outputs_E[num_keys_values_convs_plus_4].numpy()
            if max_logits_idx in STOP_TOKEN:
                break
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_convs_plus_1]
                input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_convs_plus_2]
                all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                input_feed_F[in_name_F[2]] = all_outputs_F[2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_F[0]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_F[1]
            if num_decode < 1:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_D))
                input_feed_A[in_name_A] = all_outputs_D[num_keys_values_convs]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_D[num_keys_values_convs_plus_1]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_D[num_keys_values_convs_plus_2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_3]] = all_outputs_D[num_keys_values_convs_plus_3]
            else:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_E))
                input_feed_A[in_name_A] = all_outputs_E[num_keys_values_convs]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_E[num_keys_values_convs_plus_1]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_E[num_keys_values_convs_plus_2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_3]] = all_outputs_E[num_keys_values_convs_plus_3]
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
        else:
            input_feed_C[in_name_C[0]] = all_outputs_B[num_keys_values_convs]
            all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
            max_logits_idx = all_outputs_C[0].numpy()[0, 0]
            if max_logits_idx in STOP_TOKEN:
                break
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                reset_ids = save_id_greedy[penality_reset_count_greedy]
                if reset_ids != max_logits_idx:
                    repeat_penality = all_outputs_C[1].numpy()
                    repeat_penality[:, reset_ids] = 1.0
                    input_feed_C[in_name_C[1]].update_inplace(repeat_penality)
                penality_reset_count_greedy += 1
            else:
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
            input_feed_C[in_name_C[0]] = all_outputs_C[0]
            input_feed_B.update(zip(in_name_B[:num_keys_values_convs_plus_1], all_outputs_B))
            input_feed_A[in_name_A] = all_outputs_C[0]
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
            save_id_greedy[num_decode] = max_logits_idx
        input_feed_B[in_name_B[num_keys_values_convs_plus_1]] = all_outputs_B[num_keys_values_convs_plus_1]
        if num_decode < 1:
            input_feed_B[in_name_B[num_keys_values_convs_plus_2]] = ids_len_1
        num_decode += 1
    if num_decode > 0:
        print(f"\n\nDecode: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s")
        if USE_BEAM_SEARCH:
            input_feed_G = {in_name_G[0]: all_outputs_E[num_keys_values_convs_plus_1]}
        else:
            input_feed_G = {in_name_G[0]: onnxruntime.OrtValue.ortvalue_from_numpy(save_id_greedy.reshape(1, -1), device_type, DEVICE_ID)}
        input_feed_G[in_name_G[1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([num_decode], dtype=np.int64), device_type, DEVICE_ID)
        audio_out = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)[0]
        print(f"\nGenerate Complete.\n\nSaving to: {generated_audio_path}.\n\nTime Cost: {time.time() - start_time:.3f} Seconds")
        audio_out = audio_out.numpy().reshape(-1)
        sf.write(generated_audio_path, audio_out, SAMPLE_RATE, format='WAVEX')
    else:
        print("\n Generate Failed")
        

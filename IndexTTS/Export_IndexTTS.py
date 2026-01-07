import os
import gc
import re
import sys
import time
import torch
import shutil
import warnings
import traceback
import torchaudio
import onnxruntime
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
from typing import List, Union, overload
from sentencepiece import SentencePieceProcessor


project_path         = r"/home/DakeQQ/Downloads/index-tts-main"                     # The IndexTTS project path.          URL: https://github.com/index-tts/index-tts
models_path          = r"/home/DakeQQ/Downloads/IndexTTS-1.5"                       # The IndexTTS models download path.  URL: https://modelscope.cn/models/IndexTeam/IndexTTS-1.5/files
onnx_model_A         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_A.onnx"      # The exported onnx model path.
onnx_model_B         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_B.onnx"      # The exported onnx model path.
onnx_model_C         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_C.onnx"      # The exported onnx model path.
onnx_model_D         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_D.onnx"      # The exported onnx model path.
onnx_model_E         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_E.onnx"      # The exported onnx model path.
onnx_model_F         = r"/home/DakeQQ/Downloads/IndexTTS_ONNX/IndexTTS_F.onnx"      # The exported onnx model path.
tokenizer_path       = models_path + "/bpe.model"                                   # The IndexTTS tokenizer path.
generated_audio_path = r"generated.wav"                                             # The generated audio path.
reference_audio      = r"./example/zh.wav"                                          # The reference audio path.
gen_text             = "大家好，我现在正在大可奇奇体验 ai 科技。"                          # The target speech.


# Model Parameters
SAMPLE_RATE = 24000                     # IndexTTS model setting
STOP_TOKEN = [8193]                     # IndexTTS model setting
MAX_GENERATE_LENGTH = 800               # IndexTTS model setting
REPEAT_PENALITY = 0.9                   # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                     # Penalizes the most recent output. "10" means the last 10 mel tokens.

# STFT/ISTFT Settings
AUDIO_LENGTH = 320000                   # Maximum input audio length: the length of the audio input signal (in samples).
MAX_SIGNAL_LENGTH = 4096                # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT. It affects the generated audio length and speech speed.
WINDOW_LENGTH = 1024                    # THe length of windowing
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT


shutil.copy("./modeling_modified/resample.py", project_path + "/indextts/BigVGAN/alias_free_torch/resample.py")
shutil.copy("./modeling_modified/filter.py", project_path + "/indextts/BigVGAN/alias_free_torch/filter.py")
shutil.copy("./modeling_modified/act.py", project_path + "/indextts/BigVGAN/alias_free_torch/act.py")
shutil.copy("./modeling_modified/models.py", project_path + "/indextts/BigVGAN/models.py")


if project_path not in sys.path:
    sys.path.append(project_path)


def _compute_statistics(x, m, dim=2):
    mean = (m * x).sum(dim, keepdim=True)
    std = torch.sqrt((m * (x - mean).pow(2)).sum(dim, keepdim=True).clamp(1e-6))
    return mean, std


def rel_shift(x, x_len, zero_pad, n_head):
    x_padded = torch.cat([zero_pad[:, :x_len].float(), x], dim=-1)
    x_padded = x_padded.view(n_head, -1, x_len)
    x = x_padded[:, 1:].view_as(x)
    return x[:, :, :x_len]


class IndexTTS_A(torch.nn.Module):
    def __init__(self, indexTTS, custom_stft, nfft, n_mels, sample_rate, max_signal_len):
        super(IndexTTS_A, self).__init__()
        self.bigvgan = indexTTS.bigvgan.eval()
        self.indexTTS = indexTTS.gpt.eval()
        self.custom_stft = custom_stft
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        self.indexTTS.conditioning_encoder.embed.pos_enc.pe = self.indexTTS.conditioning_encoder.embed.pos_enc.pe[:, :max_signal_len].half()
        self.indexTTS.conditioning_encoder.embed.out._modules['0'].weight.data *= self.indexTTS.conditioning_encoder.embed.pos_enc.xscale
        self.indexTTS.conditioning_encoder.embed.out._modules['0'].bias.data *= self.indexTTS.conditioning_encoder.embed.pos_enc.xscale
        self.zero_pad = torch.zeros(( self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.h, 2048, 1), dtype=torch.int8)  # 2048 is about 30 seconds audio input.
        self.perceiver_encoder_head = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].heads
        self.perceiver_encoder_head_dim = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.out_features // self.perceiver_encoder_head
        self.latents = self.indexTTS.perceiver_encoder.latents.data.unsqueeze(0)
        self.audio_pad = torch.randn((1, 1, int(sample_rate * 0.1)), dtype=torch.float32)
        num_heads = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.h
        head_dim = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.d_k
        hidden_size = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.linear_q.in_features
        scaling = float(head_dim ** -0.25)
        for layer in self.indexTTS.conditioning_encoder.encoders:
            layer.self_attn.linear_q.weight.data *= scaling
            layer.self_attn.linear_q.bias.data *= scaling
            layer.self_attn.linear_k.weight.data *= scaling
            layer.self_attn.linear_k.bias.data *= scaling
            layer.self_attn.linear_pos.weight.data *= scaling
            layer.self_attn.pos_bias_u.data = layer.self_attn.pos_bias_u.data.unsqueeze(1) * scaling
            layer.self_attn.pos_bias_v.data = layer.self_attn.pos_bias_v.data.unsqueeze(1) * scaling

            layer.self_attn.linear_q.weight.data = layer.self_attn.linear_q.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.self_attn.linear_q.bias.data = layer.self_attn.linear_q.bias.data.view(num_heads, 1, head_dim).contiguous()
            layer.self_attn.linear_k.weight.data = layer.self_attn.linear_k.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.self_attn.linear_k.bias.data = layer.self_attn.linear_k.bias.data.view(num_heads, 1, head_dim).contiguous()
            layer.self_attn.linear_v.weight.data = layer.self_attn.linear_v.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.self_attn.linear_v.bias.data = layer.self_attn.linear_v.bias.data.view(num_heads, 1, head_dim).contiguous()
            layer.self_attn.linear_pos.weight.data = layer.self_attn.linear_pos.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.self_attn.linear_out.weight.data = layer.self_attn.linear_out.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()
            layer.self_attn.linear_out.bias.data = layer.self_attn.linear_out.bias.data.view(1, 1, -1).contiguous()

        num_heads = self.perceiver_encoder_head
        head_dim = self.perceiver_encoder_head_dim
        hidden_size = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.in_features
        scaling = float(head_dim ** -0.25)
        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.weight.data *= scaling
        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_kv.weight.data[:self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.out_features] *= scaling
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.weight.data *= scaling
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_kv.weight.data[:self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.out_features] *= scaling

        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.weight.data = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_k = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_kv.weight.data[:self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.out_features].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_v = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_kv.weight.data[self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.out_features:].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_out.weight.data = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_out.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.weight.data = self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.weight.data.view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_k = self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_kv.weight.data[:self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.out_features].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_v = self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_kv.weight.data[self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_q.out_features:].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
        self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_out.weight.data = self.indexTTS.perceiver_encoder.layers._modules['1']._modules['0'].to_out.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()

    def forward(self, audio: torch.ShortTensor):
        audio = audio.float() * self.inv_int16
        audio = torch.cat([self.audio_pad, audio], dim=-1)
        real_part, imag_part = self.custom_stft(audio, 'constant')
        mel_signal = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).clamp(min=1e-5).log()
        x = self.indexTTS.conditioning_encoder.embed.conv(mel_signal.transpose(1, 2).unsqueeze(1))
        enc_len = x.shape[2].unsqueeze(0)
        x = self.indexTTS.conditioning_encoder.embed.out(x.transpose(1, 2).contiguous().view(1, enc_len, -1))
        pos_emb = self.indexTTS.conditioning_encoder.embed.pos_enc.pe[:, :enc_len].float()
        for encoder_layer in self.indexTTS.conditioning_encoder.encoders:
            x1 = encoder_layer.norm_mha(x)
            q = torch.matmul(x1, encoder_layer.self_attn.linear_q.weight) + encoder_layer.self_attn.linear_q.bias
            k = (torch.matmul(x1, encoder_layer.self_attn.linear_k.weight) + encoder_layer.self_attn.linear_k.bias).transpose(1, 2)
            v = torch.matmul(x1, encoder_layer.self_attn.linear_v.weight) + encoder_layer.self_attn.linear_v.bias
            p = torch.matmul(pos_emb, encoder_layer.self_attn.linear_pos.weight).transpose(1, 2)
            q_with_bias_u = q + encoder_layer.self_attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.self_attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k)
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd, enc_len, self.zero_pad, encoder_layer.self_attn.h)
            attn_out = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            attn_out = torch.matmul(attn_out, encoder_layer.self_attn.linear_out.weight).sum(dim=0, keepdim=True) + encoder_layer.self_attn.linear_out.bias
            x += attn_out
            residual = x
            x = encoder_layer.norm_conv(x).transpose(1, 2)
            x = encoder_layer.conv_module.pointwise_conv1(x)
            x = torch.nn.functional.glu(x, dim=1)
            x = encoder_layer.conv_module.depthwise_conv(x).transpose(1, 2)
            x = encoder_layer.conv_module.activation(encoder_layer.conv_module.norm(x)).transpose(1, 2)
            x = encoder_layer.conv_module.pointwise_conv2(x).transpose(1, 2)
            x += residual
            x = x + encoder_layer.feed_forward(encoder_layer.norm_ff(x))
            x = encoder_layer.norm_final(x)
        x = self.indexTTS.conditioning_encoder.after_norm(x)
        x = self.indexTTS.perceiver_encoder.proj_context(x)
        for attn, ff in self.indexTTS.perceiver_encoder.layers:
            q = torch.matmul(self.latents, attn.to_q.weight)
            cat_latent_x = torch.cat([self.latents, x], dim=1)
            k = torch.matmul(cat_latent_x, attn.to_k).transpose(1, 2)
            v = torch.matmul(cat_latent_x, attn.to_v)
            attn_out = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v)
            attn_out = torch.matmul(attn_out, attn.to_out.weight).sum(dim=0, keepdim=True)
            self.latents = attn_out + self.latents
            self.latents = ff(self.latents) + self.latents
        conds_latent = self.indexTTS.perceiver_encoder.norm(self.latents)

        # bigvgan part
        ref_signal_len = mel_signal.shape[-1].unsqueeze(0)
        speaker_embedding = []
        for i, layer in enumerate(self.bigvgan.speaker_encoder.blocks):
            mel_signal = layer(mel_signal)
            if i > 0:
                speaker_embedding.append(mel_signal)
        speaker_embedding = torch.cat(speaker_embedding, dim=1)
        speaker_embedding = self.bigvgan.speaker_encoder.mfa(speaker_embedding)
        mean, std = _compute_statistics(speaker_embedding, 1.0 / ref_signal_len)
        mean = mean.repeat(1, 1, ref_signal_len)
        std = std.repeat(1, 1, ref_signal_len)
        attn = torch.cat([speaker_embedding, mean, std], dim=1)
        attn = self.bigvgan.speaker_encoder.asp.conv(self.bigvgan.speaker_encoder.asp.tanh(self.bigvgan.speaker_encoder.asp.tdnn(attn)))
        attn = torch.nn.functional.softmax(attn, dim=2)
        mean, std = _compute_statistics(speaker_embedding, attn)
        speaker_embedding = torch.cat((mean, std), dim=1)
        speaker_embedding = self.bigvgan.speaker_encoder.asp_bn(speaker_embedding)
        speaker_embedding = self.bigvgan.speaker_encoder.fc(speaker_embedding)
        bigvgan_cond_layer_speaker_embedding = self.bigvgan.cond_layer(speaker_embedding)
        save_bigvgan_conds = []
        for i in range(self.bigvgan.num_upsamples):
            save_bigvgan_conds.append(self.bigvgan.conds[i](speaker_embedding))
        return *save_bigvgan_conds, bigvgan_cond_layer_speaker_embedding, conds_latent


class IndexTTS_B(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_B, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()
        self.start_ids = torch.tensor([[0]], dtype=torch.int32)
        self.end_ids = torch.tensor([[1]], dtype=torch.int32)

    def forward(self, text_ids):
        text_ids = torch.cat([self.start_ids, text_ids, self.end_ids], dim=-1)
        text_ids_len = text_ids.shape[-1].unsqueeze(0)
        text_emb = self.indexTTS.text_embedding(text_ids) + self.indexTTS.text_pos_embedding.emb.weight[:text_ids_len]
        return text_emb


class IndexTTS_C(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_C, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()

    def forward(self, gpt_ids, gen_len):
        hidden_state = self.indexTTS.inference_model.embeddings(gpt_ids)
        hidden_state += self.indexTTS.inference_model.text_pos_embedding.emb.weight[gen_len]
        return hidden_state, gen_len + 1


class IndexTTS_D(torch.nn.Module):
    def __init__(self):
        super(IndexTTS_D, self).__init__()
        pass

    def forward(self, embed_x, embed_y, embed_z):
        concat_hidden_state = torch.cat([embed_x, embed_y, embed_z], dim=1)
        return concat_hidden_state, concat_hidden_state.shape[1].unsqueeze(0)


class IndexTTS_E(torch.nn.Module):
    def __init__(self, indexTTS, num_layers, max_seq_len):
        super(IndexTTS_E, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()
        self.num_layers = num_layers
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        num_heads = self.indexTTS.inference_model.transformer.h._modules['0'].attn.num_heads
        head_dim = self.indexTTS.inference_model.transformer.h._modules['0'].attn.head_dim
        hidden_size = self.indexTTS.inference_model.transformer.h._modules['0'].attn.embed_dim
        scaling = float(head_dim ** -0.25)
        qk_size = hidden_size + hidden_size
        for layer in self.indexTTS.inference_model.transformer.h:
            layer.attn.c_attn.weight.data = layer.attn.c_attn.weight.data.transpose(0, 1)
            layer.attn.c_proj.weight.data = layer.attn.c_proj.weight.data.transpose(0, 1)
            
            layer.attn.c_attn.weight.data[:qk_size] *= scaling
            layer.attn.c_attn.bias.data[:qk_size] *= scaling
            
            layer.attn.to_q_weight = layer.attn.c_attn.weight.data[:hidden_size].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attn.to_q_bias = layer.attn.c_attn.bias.data[:hidden_size].view(num_heads, 1, head_dim).contiguous()
            layer.attn.to_k_weight = layer.attn.c_attn.weight.data[hidden_size:qk_size].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attn.to_k_bias = layer.attn.c_attn.bias.data[hidden_size:qk_size].view(num_heads, 1, head_dim).contiguous()
            layer.attn.to_v_weight = layer.attn.c_attn.weight.data[qk_size:].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous()
            layer.attn.to_v_bias = layer.attn.c_attn.bias.data[qk_size:].view(num_heads, 1, head_dim).contiguous()
            layer.attn.c_proj.weight.data = layer.attn.c_proj.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()
            layer.attn.c_proj.bias.data = layer.attn.c_proj.bias.data.view(1, 1, -1).contiguous()

    def forward(self, *all_inputs):
        ids_len = all_inputs[-3]
        hidden_state = all_inputs[-2]
        kv_seq_len = all_inputs[-5] + ids_len  # history_len
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()  # attention_mask
        for i, layer in enumerate(self.indexTTS.inference_model.transformer.h):
            hidden_states_norm = layer.ln_1(hidden_state)
            q = torch.matmul(hidden_states_norm, layer.attn.to_q_weight) + layer.attn.to_q_bias
            k = (torch.matmul(hidden_states_norm, layer.attn.to_k_weight) + layer.attn.to_k_bias).transpose(1, 2)
            v = torch.matmul(hidden_states_norm, layer.attn.to_v_weight) + layer.attn.to_v_bias
            k = torch.cat((all_inputs[i], k), dim=2)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=1)
            self.save_key[i] = k
            self.save_value[i] = v
            hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
            hidden_state_attn = torch.matmul(hidden_state_attn, layer.attn.c_proj.weight).sum(dim=0, keepdim=True) + layer.attn.c_proj.bias
            hidden_state += hidden_state_attn
            hidden_state = hidden_state + layer.mlp.c_proj(layer.mlp.act(layer.mlp.c_fc(layer.ln_2(hidden_state))))
        last_hidden_state = self.indexTTS.inference_model.transformer.ln_f(hidden_state[:, -1])
        logits = self.indexTTS.inference_model.lm_head(last_hidden_state) * all_inputs[-4]   # repeat_penality
        max_logit_ids = torch.argmax(logits, dim=-1, keepdim=True).int()                     # Greedy Search
        return *self.save_key, *self.save_value, kv_seq_len, last_hidden_state, max_logit_ids


class IndexTTS_F(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_F, self).__init__()
        self.indexTTS = indexTTS
        self.indexTTS.gpt = self.indexTTS.gpt.eval()
        self.indexTTS.bigvgan = self.indexTTS.bigvgan.eval()
        self.inv_num_kernels = float(1.0 / self.indexTTS.bigvgan.num_kernels)

    def forward(self, *all_inputs):
        latent = self.indexTTS.gpt.final_norm(all_inputs[-1][:-2].unsqueeze(0))
        latent = self.indexTTS.bigvgan.conv_pre(latent.transpose(1, 2)) + all_inputs[-2]
        for i in range(self.indexTTS.bigvgan.num_upsamples):
            for i_up in range(len(self.indexTTS.bigvgan.ups[i])):
                latent = self.indexTTS.bigvgan.ups[i][i_up](latent)
            if self.indexTTS.bigvgan.cond_in_each_up_layer:
                latent = latent + all_inputs[i]
            x = self.indexTTS.bigvgan.resblocks[i * self.indexTTS.bigvgan.num_kernels](latent, i)
            for j in range(1, self.indexTTS.bigvgan.num_kernels):
                x = x + self.indexTTS.bigvgan.resblocks[i * self.indexTTS.bigvgan.num_kernels + j](latent, i)
            latent = x * self.inv_num_kernels
        latent = self.indexTTS.bigvgan.conv_post(self.indexTTS.bigvgan.activation_post(latent, -1))
        generated_wav = torch.tanh(latent)
        return (generated_wav.clamp(min=-1.0, max=1.0) * 32767.0).to(torch.int16)


print("\n\nStart to Export the part_A...\n")
with torch.inference_mode():
    from indextts.infer import IndexTTS
    indexTTS = IndexTTS(model_dir=models_path, cfg_path=models_path + "/config.yaml", is_fp16=False, device='cpu')
    for para in indexTTS.gpt.parameters():
        para.requires_grad = False
    for para in indexTTS.bigvgan.parameters():
        para.requires_grad = False

    NUM_HEADS = indexTTS.gpt.heads
    NUM_LAYERS = indexTTS.gpt.layers
    HIDDEN_SIZE = indexTTS.gpt.model_dim
    HEAD_DIM = indexTTS.gpt.inference_model.transformer.h._modules['0'].attn.head_dim
    SPEAKER_EMBED_SIZE = indexTTS.bigvgan.cond_layer.out_channels
    MEL_CODE_SIZE = indexTTS.gpt.number_mel_codes

    audio = torch.ones((1, 1, AUDIO_LENGTH), dtype=torch.int16)
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    part_A = IndexTTS_A(indexTTS, custom_stft, NFFT, N_MELS, SAMPLE_RATE, MAX_SIGNAL_LENGTH)

    output_names = []
    for i in range(indexTTS.bigvgan.num_upsamples):
        output_names.append(f"save_bigvgan_conds_{i}")
    output_names.append("bigvgan_cond_layer_speaker_embedding")
    output_names.append("conds_latent")

    torch.onnx.export(
        part_A,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'conds_latent': {1: 'ref_signal_len'},
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=17)
    del custom_stft
    del part_A
    del audio
    gc.collect()
    print("\nExport part_A Done.\n\nExport part_B Start...")

    text_ids = torch.ones((1, 10), dtype=torch.int32)
    part_B = IndexTTS_B(indexTTS)
    torch.onnx.export(
        part_B,
        (text_ids,),
        onnx_model_B,
        input_names=['text_ids'],
        output_names=['text_hidden_state'],
        dynamic_axes={
            'text_ids': {1: 'text_ids_len'},
            'text_hidden_state': {1: 'text_ids_len'},
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=17)
    del part_B
    del text_ids
    gc.collect()
    print("\nExport part_B Done.\n\nExport part_C Start...")

    gpt_ids = torch.ones((1, 1), dtype=torch.int32)   # Fixed shape
    kv_seq_len = torch.tensor([1], dtype=torch.long)
    part_C = IndexTTS_C(indexTTS)
    torch.onnx.export(
        part_C,
        (gpt_ids, kv_seq_len),
        onnx_model_C,
        input_names=['gpt_ids', 'kv_seq_len'],
        output_names=['gpt_hidden_state', 'next_kv_seq_len'],
        do_constant_folding=True,
        dynamo=False,
        opset_version=17)
    del part_C
    del gpt_ids
    del kv_seq_len
    gc.collect()
    print("\nExport part_C Done.\n\nExport part_D Start...")

    embed_x = torch.ones((1, 10, HIDDEN_SIZE), dtype=torch.float32)
    embed_y = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    embed_z = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    part_D = IndexTTS_D()
    torch.onnx.export(
        part_D,
        (embed_x, embed_y, embed_z),
        onnx_model_D,
        input_names=['embed_x', 'embed_y', 'embed_z'],
        output_names=['concat_hidden_state', 'concat_len'],
        dynamic_axes={
            'embed_x': {1: 'embed_x_len'},
            'embed_y': {1: 'embed_y_len'},
            'embed_z': {1: 'embed_y_len'},
            'concat_hidden_state': {1: 'concat_len'},
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=17)
    del part_D
    del embed_x
    del embed_y
    del embed_z
    gc.collect()
    print("\nExport part_D Done.\n\nExport part_E Start...")

    # Prepare input and output names
    attention_mask = torch.tensor([0], dtype=torch.int8)
    ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
    history_len = torch.tensor([0], dtype=torch.int64)
    hidden_state = torch.ones((1, ids_len, HIDDEN_SIZE), dtype=torch.float32)
    repeat_penality = torch.ones((1, MEL_CODE_SIZE), dtype=torch.float32)
    past_keys = torch.zeros((NUM_HEADS, HEAD_DIM, 0), dtype=torch.float32)
    past_values = torch.zeros((NUM_HEADS, 0, HEAD_DIM), dtype=torch.float32)
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYERS):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYERS):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'history_len_plus_ids_len'}

    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('repeat_penality')
    all_inputs.append(repeat_penality)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('hidden_state')
    all_inputs.append(hidden_state)
    dynamic_axes['hidden_state'] = {1: 'ids_len'}
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('kv_seq_len')
    output_names.append('last_hidden_state')
    output_names.append('max_logit_id')

    part_E = IndexTTS_E(indexTTS, NUM_LAYERS, MAX_SIGNAL_LENGTH)
    torch.onnx.export(
        part_E,
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
        opset_version=17
    )
    del part_E
    del hidden_state
    del history_len
    del ids_len
    del attention_mask
    del past_keys
    del past_values
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del repeat_penality
    gc.collect()
    print("\nExport part_E Done.\n\nExport part_F Start...")

    all_inputs = []
    input_names = []
    for i in range(indexTTS.bigvgan.num_upsamples):
        input_names.append(f"save_bigvgan_conds_{i}")
        all_inputs.append(torch.ones((1, indexTTS.bigvgan.conds[i].out_channels, 1), dtype=torch.float32))
    input_names.append("bigvgan_cond_layer_speaker_embedding")
    all_inputs.append(torch.ones((1, SPEAKER_EMBED_SIZE, 1), dtype=torch.float32))
    input_names.append("save_hidden_state")
    all_inputs.append(torch.ones((10, HIDDEN_SIZE), dtype=torch.float32))

    part_F = IndexTTS_F(indexTTS)
    torch.onnx.export(
        part_F,
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=['generated_wav'],
        dynamic_axes={
            'save_hidden_state': {0: 'kv_seq_len'},
            'generated_wav': {2: 'generated_len'}
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=17)
    del part_F
    del all_inputs
    del input_names
    print("\nAll Export Done.")


if project_path in sys.path:
    sys.path.remove(project_path)


# From the official code
def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    CJK_RANGE_PATTERN = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


# From the official code
def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    Example:
      input = "你 好 世 界 是 HELLO WORLD 的 中 文"
      output = "你好世界是 hello world 的中文"

    do_lower_case:
      input = "SEE YOU!"
      output = "see you!"
    """
    # replace english words in the line with placeholders
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # restore english sentences
    sent_placeholder_pattern = re.compile(r"^.*?(<sent_(\d+)>)")
    for i in range(len(words)):
        m = sent_placeholder_pattern.match(words[i])
        if m:
            # restore the english word
            placeholder_index = int(m.group(2))
            words[i] = words[i].replace(m.group(1), english_sents[placeholder_index])
            if do_lower_case:
                words[i] = words[i].lower()
    return "".join(words)


# From the official code
class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"
    """
    匹配拼音声调格式：pinyin+数字，声调1-5，5表示轻声
    例如：xuan4, jve2, ying1, zhong4, shang5
    不匹配：beta1, voice2
    """
    NAME_PATTERN = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
    """
    匹配人名，格式：中文·中文，中文·中文-中文
    例如：克里斯托弗·诺兰，约瑟夫·高登-莱维特
    """

    # 匹配常见英语缩写 's，仅用于替换为 is，不匹配所有 's
    ENGLISH_CONTRACTION_PATTERN = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"

    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(re.search(TextNormalizer.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if self.zh_normalizer is not None and self.en_normalizer is not None:
            return
        if platform.system() == "Darwin":
            from wetext import Normalizer

            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            # use new cache dir for build tagger rules with disable remove_interjections and remove_erhua
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tagger_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, ".gitignore"), "w") as f:
                    f.write("*\n")
            self.zh_normalizer = NormalizerZh(
                cache_dir=cache_dir, remove_interjections=False, remove_erhua=False, overwrite_cache=False
            )
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        text = text.replace("嗯", "恩").replace("呣", "母")
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        if self.use_chinese(text):
            text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
            replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())

            replaced_text, original_name_list = self.save_names(replaced_text)
            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())
            # 恢复人名
            result = self.restore_names(result, original_name_list)
            # 恢复拼音声调
            result = self.restore_pinyin_tones(result, pinyin_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                text = re.sub(TextNormalizer.ENGLISH_CONTRACTION_PATTERN, r"\1 is", text, flags=re.IGNORECASE)
                result = self.en_normalizer.normalize(text)
            except Exception:
                result = text
                print(traceback.format_exc())
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def correct_pinyin(self, pinyin: str):
        """
        将 jqx 的韵母为 u/ü 的拼音转换为 v
        如：ju -> jv , que -> qve, xün -> xvn
        """
        if pinyin[0] not in "jqxJQX":
            return pinyin
        # 匹配 jqx 的韵母为 u/ü 的拼音
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        """
        替换人名为占位符 <n_a>、 <n_b>, ...
        例如：克里斯托弗·诺兰 -> <n_a>
        """
        # 人名
        name_pattern = re.compile(TextNormalizer.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        # 替换占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        """
        恢复人名为原来的文字
        例如：<n_a> -> original_name_list[0]
        """
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换为占位符 <n_a>、 <n_b>, ...
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        """
        替换拼音声调为占位符 <pinyin_a>, <pinyin_b>, ...
        例如：xuan4 -> <pinyin_a>
        """
        # 声母韵母+声调数字
        origin_pinyin_pattern = re.compile(TextNormalizer.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        # 替换为占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        # print("original_text: ", original_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """
        恢复拼音中的音调数字（1-5）为原来的拼音
        例如：<pinyin_a> -> original_pinyin_list[0]
        """
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text

        transformed_text = normalized_text
        # 替换占位符 <pinyin_a>, <pinyin_b>, ...
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        # print("normalized_text: ", normalized_text)
        # print("transformed_text: ", transformed_text)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 加载词表
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 预处理器
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # 预处理
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        # 预处理
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_sentences_by_token(
            tokenized_str: List[str], split_tokens: List[str], max_tokens_per_sentence: int
    ) -> List[List[str]]:
        """
        将tokenize后的结果按特定token进一步分割
        """
        # 处理特殊情况
        if len(tokenized_str) == 0:
            return []
        sentences: List[List[str]] = []
        current_sentence = []
        current_sentence_tokens_len = 0
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_sentence.append(token)
            current_sentence_tokens_len += 1
            if current_sentence_tokens_len <= max_tokens_per_sentence:
                if token in split_tokens and current_sentence_tokens_len > 2:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            # 后续token是'，则不切分
                            current_sentence.append(tokenized_str[i + 1])
                            i += 1
                    sentences.append(current_sentence)
                    current_sentence = []
                    current_sentence_tokens_len = 0
                continue
            # 如果当前tokens的长度超过最大限制
            if not ("," in split_tokens or "▁," in split_tokens) and (
                    "," in current_sentence or "▁," in current_sentence):
                # 如果当前tokens中有,，则按,分割
                sub_sentences = TextTokenizer.split_sentences_by_token(
                    current_sentence, [",", "▁,"], max_tokens_per_sentence=max_tokens_per_sentence
                )
            elif "-" not in split_tokens and "-" in current_sentence:
                # 没有,，则按-分割
                sub_sentences = TextTokenizer.split_sentences_by_token(
                    current_sentence, ["-"], max_tokens_per_sentence=max_tokens_per_sentence
                )
            else:
                # 按照长度分割
                sub_sentences = []
                for j in range(0, len(current_sentence), max_tokens_per_sentence):
                    if j + max_tokens_per_sentence < len(current_sentence):
                        sub_sentences.append(current_sentence[j: j + max_tokens_per_sentence])
                    else:
                        sub_sentences.append(current_sentence[j:])
                warnings.warn(
                    f"The tokens length of sentence exceeds limit: {max_tokens_per_sentence}, "
                    f"Tokens in sentence: {current_sentence}."
                    "Maybe unexpected behavior",
                    RuntimeWarning,
                )
            sentences.extend(sub_sentences)
            current_sentence = []
            current_sentence_tokens_len = 0
        if current_sentence_tokens_len > 0:
            assert current_sentence_tokens_len <= max_tokens_per_sentence
            sentences.append(current_sentence)
        # 如果相邻的句子加起来长度小于最大限制，则合并
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            if len(merged_sentences) == 0:
                merged_sentences.append(sentence)
            elif len(merged_sentences[-1]) + len(sentence) <= max_tokens_per_sentence:
                merged_sentences[-1] = merged_sentences[-1] + sentence
            else:
                merged_sentences.append(sentence)
        return merged_sentences

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "▁.",
        # "▁!", # unk
        "▁?",
        "▁...",  # ellipsis
    ]

    def split_sentences(self, tokenized: List[str], max_tokens_per_sentence=120) -> List[List[str]]:
        return TextTokenizer.split_sentences_by_token(
            tokenized, self.punctuation_marks_tokens, max_tokens_per_sentence=max_tokens_per_sentence
        )


normalizer = TextNormalizer()
normalizer.load()
tokenizer = TextTokenizer(tokenizer_path, normalizer)
del normalizer

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
amount_of_outputs_A = len(out_name_A)
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs_A)]
last_output_indices_A = amount_of_outputs_A - 1


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
out_name_B0 = out_name_B[0].name


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name
out_name_C1 = out_name_C[1].name


ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D0 = in_name_D[0].name
in_name_D1 = in_name_D[1].name
in_name_D2 = in_name_D[2].name
out_name_D0 = out_name_D[0].name
out_name_D1 = out_name_D[1].name


ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
model_E_dtype = ort_session_E._inputs_meta[0].type
if 'float16' in model_E_dtype:
    model_E_dtype = np.float16
else:
    model_E_dtype = np.float32
in_names_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
amount_of_inputs_E = len(in_names_E)
amount_of_outputs_E = len(out_name_E)
in_names_E = [in_names_E[i].name for i in range(amount_of_inputs_E)]
out_name_E = [out_name_E[i].name for i in range(amount_of_outputs_E)]
num_layers = (amount_of_outputs_E - 3) // 2
num_layers_2 = num_layers + num_layers
num_layers_2_plus_1 = num_layers_2 + 1
num_layers_2_plus_2 = num_layers_2 + 2
num_layers_2_plus_3 = num_layers_2 + 3
last_input_indices_E = amount_of_inputs_E - 1
last_output_indices_E = amount_of_outputs_E - 1
second_last_output_indices_E = amount_of_outputs_E - 2


ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F0 = out_name_F[0].name


# Run IndexTTS by ONNX Runtime
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio = audio.reshape(1, 1, -1)
init_gpt_ids = np.array([[8192]], dtype=np.int32)
init_gen_len = np.array([0], dtype=np.int64)
init_ids_len_1 = np.array([1], dtype=np.int64)
init_history_len = np.array([0], dtype=np.int64)
init_attention_mask_0 = np.array([0], dtype=np.int8)
init_attention_mask_1 = np.array([1], dtype=np.int8)
init_past_keys_E = np.zeros((ort_session_E._inputs_meta[0].shape[0], ort_session_E._inputs_meta[0].shape[1], 0), dtype=model_E_dtype)
init_past_values_E = np.zeros((ort_session_E._inputs_meta[num_layers].shape[0], 0, ort_session_E._inputs_meta[num_layers].shape[2]), dtype=model_E_dtype)
repeat_penality = np.ones((1, ort_session_E._inputs_meta[num_layers_2_plus_1].shape[1]), dtype=np.float32)
split_pad = np.zeros((1, 1, int(SAMPLE_RATE * 0.2)), dtype=np.int16)  # Default to 200ms split padding.

input_feed_F = {}
input_feed_E = {
    in_names_E[last_input_indices_E]: init_attention_mask_1,
    in_names_E[num_layers_2]: init_history_len,
    in_names_E[num_layers_2_plus_1]: repeat_penality
}
for i in range(num_layers):
    input_feed_E[in_names_E[i]] = init_past_keys_E
for i in range(num_layers, num_layers_2):
    input_feed_E[in_names_E[i]] = init_past_values_E

text_tokens_list = tokenizer.tokenize(gen_text)
sentences = tokenizer.split_sentences(text_tokens_list)
total_sentences = len(sentences)
save_generated_wav = []

# Start to Run IndexTTS
print("\nStart to run the IndexTTS by ONNX Runtime.")
start_time = time.time()

all_outputs_A = ort_session_A.run(
    out_name_A,
    {
        in_name_A0: audio
    })

for i in range(last_output_indices_A):
    input_feed_F[in_name_F[i]] = all_outputs_A[i]

for i in range(total_sentences):
    sent = sentences[i]
    split_text = "".join(sent).replace("▁", " ")
    print(f"\nGenerate the Voice for '{split_text}'")

    text_tokens = tokenizer.convert_tokens_to_ids(sent)
    text_ids = np.array([text_tokens], dtype=np.int32)

    text_hidden_state = ort_session_B.run(
        [out_name_B0],
        {
            in_name_B0: text_ids,
        })[0]

    gpt_hidden_state, gen_len = ort_session_C.run(
        [out_name_C0, out_name_C1],
        {
            in_name_C0: init_gpt_ids,
            in_name_C1: init_gen_len
        })

    gpt_hidden_state, concat_len = ort_session_D.run(
        [out_name_D0, out_name_D1],
        {
            in_name_D0: all_outputs_A[last_output_indices_A],
            in_name_D1: text_hidden_state,
            in_name_D2: gpt_hidden_state
        })

    generate_limit = MAX_GENERATE_LENGTH - concat_len
    input_feed_E[in_names_E[num_layers_2_plus_2]] = concat_len

    save_last_hidden_state = []
    save_max_logits_ids = []
    reset_penality = 0
    num_decode = 0

    decode_time = time.time()
    while num_decode < generate_limit:
        input_feed_E[in_names_E[num_layers_2_plus_3]] = gpt_hidden_state
        all_outputs_E = ort_session_E.run(out_name_E, input_feed_E)
        max_logit_ids = all_outputs_E[last_output_indices_E]
        save_max_logits_ids.append(max_logit_ids)
        save_last_hidden_state.append(all_outputs_E[second_last_output_indices_E])
        num_decode += 1
        if max_logit_ids in STOP_TOKEN:
            break
        if num_decode < 2:
            input_feed_E[in_names_E[last_input_indices_E]] = init_attention_mask_0
            input_feed_E[in_names_E[num_layers_2_plus_2]] = init_ids_len_1
        for i in range(second_last_output_indices_E):
            input_feed_E[in_names_E[i]] = all_outputs_E[i]
        repeat_penality[:, max_logit_ids] = REPEAT_PENALITY
        if (num_decode > PENALITY_RANGE) and (save_max_logits_ids[reset_penality] != max_logit_ids):
            repeat_penality[:, save_max_logits_ids[reset_penality]] = 1.0
            reset_penality += 1
        input_feed_E[in_names_E[num_layers_2_plus_1]] = repeat_penality
        gpt_hidden_state, gen_len = ort_session_C.run(
            [out_name_C0, out_name_C1],
            {
                in_name_C0: max_logit_ids,
                in_name_C1: gen_len
            })
    print(f"\n\nDecode Speed: {num_decode / (time.time() - decode_time):.3f} tokens/s")

    input_feed_F[in_name_F[last_output_indices_A]] = np.concatenate(save_last_hidden_state, axis=0)
    generated_wav = ort_session_F.run(
        [out_name_F0],
        input_feed_F
    )[0]
    generated_wav = np.concatenate([generated_wav, split_pad], axis=-1)

    # Init
    input_feed_E[in_names_E[last_input_indices_E]] = init_attention_mask_1
    input_feed_E[in_names_E[num_layers_2]] = init_history_len
    for i in range(num_layers):
        input_feed_E[in_names_E[i]] = init_past_keys_E
    for i in range(num_layers, num_layers_2):
        input_feed_E[in_names_E[i]] = init_past_values_E

# Save to audio
sf.write(generated_audio_path, generated_wav.reshape(-1), SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{time.time() - start_time:.3f}")

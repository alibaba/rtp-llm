"""Token2Wav model: DiT (codec→mel) + BigVGAN (mel→waveform).

Ported from HuggingFace transformers Qwen2.5-Omni implementation.
"""
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

logger = logging.getLogger(__name__)


# ============================================================
# BigVGAN components
# ============================================================

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)
    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter
    normalized_filter /= normalized_filter.sum()
    return normalized_filter.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filt = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return x[..., self.pad_left:-self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class TorchActivation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                      padding=(kernel_size * d - d) // 2)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=(kernel_size - 1) // 2)
            for _ in dilation
        ])
        num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([
            TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = x
            x = act1(x)
            x = conv1(x)
            x = act2(x)
            x = conv2(x)
            x = residual + x
        return x


class BigVGANModel(nn.Module):
    def __init__(
        self,
        mel_dim: int = 80,
        upsample_initial_channel: int = 1536,
        upsample_rates: List[int] = (5, 3, 2, 2, 2, 2),
        upsample_kernel_sizes: List[int] = (11, 7, 4, 4, 4, 4),
        resblock_kernel_sizes: List[int] = (3, 7, 11),
        resblock_dilation_sizes: List[Tuple[int, ...]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_residual_blocks = len(resblock_kernel_sizes)
        self.num_upsample_layers = len(upsample_rates)
        self.conv_pre = nn.Conv1d(mel_dim, upsample_initial_channel, 7, 1, padding=3)

        ups = []
        for i, (stride, ks) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ups.append(nn.ModuleList([
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    ks, stride, padding=(ks - stride) // 2,
                )
            ]))
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList([
            AMPBlock(upsample_initial_channel // (2 ** (i + 1)), ks, d)
            for i in range(self.num_upsample_layers)
            for ks, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])
        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(upsample_initial_channel // (2 ** self.num_upsample_layers))
        )
        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** self.num_upsample_layers), 1, 7, 1, padding=3, bias=False
        )

    def _normalize_spectrogram(self, spec, max_value=1.0, min_db=-115.0):
        return torch.clamp((2 * max_value) * ((spec - min_db) / (-min_db)) - max_value, -max_value, max_value)

    def _amplitude_to_db(self, amp, min_db_level=-115.0):
        min_level = torch.exp(torch.tensor(min_db_level / 20.0 * np.log(10), device=amp.device, dtype=amp.dtype))
        return 20 * torch.log10(torch.clamp(amp, min=min_level))

    def forward(self, mel_spectrogram):
        amp = torch.exp(mel_spectrogram)
        db = self._amplitude_to_db(amp) - 20
        x = self._normalize_spectrogram(db)
        x = self.conv_pre(x)
        for i in range(self.num_upsample_layers):
            x = self.ups[i][0](x)
            residual = sum(
                self.resblocks[i * self.num_residual_blocks + j](x)
                for j in range(self.num_residual_blocks)
            ) / self.num_residual_blocks
            x = residual
        x = self.activation_post(x)
        x = self.conv_post(x)
        return torch.clamp(x, min=-1.0, max=1.0).squeeze().cpu()


# ============================================================
# DiT components
# ============================================================

class TimeDelayNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                              padding="same", padding_mode="reflect")
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, x):
        outputs = []
        for i, chunk in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                out = chunk
            elif i == 1:
                out = self.blocks[i - 1](chunk)
            else:
                out = self.blocks[i - 1](chunk + out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, 1, padding="same", padding_mode="reflect")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(se_channels, out_channels, 1, padding="same", padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)
        s = self.sigmoid(self.conv2(self.relu(self.conv1(s))))
        return x * s


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(attention_channels, channels, 1, padding="same", padding_mode="reflect")

    def forward(self, x):
        seq_len = x.shape[-1]
        lengths = torch.ones(x.shape[0], device=x.device)
        mask = (torch.arange(seq_len, device=x.device).expand(x.shape[0], -1)
                < (lengths * seq_len).unsqueeze(1)).unsqueeze(1).to(x.dtype)
        total = mask.sum(dim=2, keepdim=True)
        mean = (mask * x).sum(dim=2) / total.squeeze(2)
        std = torch.sqrt(((mask * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2) / total.squeeze(2)).clamp(self.eps))
        mean_rep = mean.unsqueeze(2).repeat(1, 1, seq_len)
        std_rep = std.unsqueeze(2).repeat(1, 1, seq_len)
        attention_input = torch.cat([x, mean_rep, std_rep], dim=1)
        attention = self.conv(self.tanh(self.tdnn(attention_input)))
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=2)
        mean2 = (attention * x).sum(dim=2)
        std2 = torch.sqrt(((attention * (x - mean2.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(self.eps))
        pooled = torch.cat((mean2, std2), dim=1)
        return pooled.unsqueeze(2)


class SERes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128,
                 kernel_size=1, dilation=1):
        super().__init__()
        self.tdnn1 = TimeDelayNetBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, x):
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class ECAPATDNNEncoder(nn.Module):
    def __init__(self, mel_dim=80, channels=(256, 256, 256, 256, 768),
                 kernel_sizes=(5, 3, 3, 3, 1), dilations=(1, 2, 3, 4, 1),
                 res2net_scale=2, se_channels=128, attention_channels=128, enc_dim=192):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(TimeDelayNetBlock(mel_dim, channels[0], kernel_sizes[0], dilations[0]))
        for i in range(1, len(channels) - 1):
            self.blocks.append(SERes2NetBlock(
                channels[i - 1], channels[i], res2net_scale=res2net_scale,
                se_channels=se_channels, kernel_size=kernel_sizes[i], dilation=dilations[i],
            ))
        self.mfa = TimeDelayNetBlock(channels[-1], channels[-1], kernel_sizes[-1], dilations[-1])
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels=attention_channels)
        self.fc = nn.Conv1d(channels[-1] * 2, enc_dim, 1, padding="same", padding_mode="reflect")

    def forward(self, x):
        x = x.transpose(1, 2)
        outs = []
        for layer in self.blocks:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.fc(x)
        return x.squeeze(-1)


class DiTInputEmbedding(nn.Module):
    def __init__(self, mel_dim, enc_dim, enc_emb_dim, emb_dim, hidden_size,
                 spk_encoder_config):
        super().__init__()
        self.proj = nn.Linear(mel_dim + enc_dim + enc_emb_dim + emb_dim, hidden_size)
        self.spk_encoder = ECAPATDNNEncoder(
            mel_dim=spk_encoder_config.get("mel_dim", mel_dim),
            channels=spk_encoder_config.get("channels", (256, 256, 256, 256, 768)),
            kernel_sizes=spk_encoder_config.get("kernel_sizes", (5, 3, 3, 3, 1)),
            dilations=spk_encoder_config.get("dilations", (1, 2, 3, 4, 1)),
            res2net_scale=spk_encoder_config.get("res2net_scale", 2),
            se_channels=spk_encoder_config.get("se_channels", 128),
            attention_channels=spk_encoder_config.get("attention_channels", 128),
            enc_dim=enc_dim,
        )

    def forward(self, hidden_states, speaker_embedding, condition_vector, code_embed,
                drop_audio_cond=False, code_embed_uncond=None, apply_cfg=True):
        if apply_cfg:
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0)
            condition_vector = torch.cat([condition_vector, torch.zeros_like(condition_vector)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        cond = self.spk_encoder(condition_vector).unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        return self.proj(torch.cat((hidden_states, cond, code_embed, speaker_embedding), dim=-1))


class DiTCodecEmbedding(nn.Module):
    def __init__(self, num_embeds, dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(num_embeds + 1, dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = torch.zeros_like(code)
        embed = self.codec_embed(code)
        return torch.repeat_interleave(embed, repeats=self.repeats, dim=1)


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1).to(x.dtype)


class DiTTimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        t = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(t)


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class DiTMLP(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = int(dim * mult)
        self.ff = nn.Sequential(nn.Linear(dim, inner), nn.GELU(approximate="tanh"),
                                nn.Dropout(dropout), nn.Linear(inner, dim))

    def forward(self, x):
        return self.ff(x)


def _rotate_half_codec(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def _apply_dit_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (_rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, dropout=0.0):
        super().__init__()
        self.heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(hidden_size, inner_dim)
        self.to_k = nn.Linear(hidden_size, inner_dim)
        self.to_v = nn.Linear(hidden_size, inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, hidden_size), nn.Dropout(dropout)])

    def forward(self, x, position_embeddings=None, attention_mask=None):
        B = x.shape[0]
        q = self.to_q(x).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q[:, :1], k[:, :1] = _apply_dit_rotary(q[:, :1], k[:, :1], cos, sin)
        if attention_mask is not None:
            mask = attention_mask.float()
            mask = mask.masked_fill(~attention_mask, float("-inf"))
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype))
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, self.heads * self.head_dim).to(q.dtype)
        out = self.to_out[0](attn_out)
        out = self.to_out[1](out)
        return out


class DiTDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, ff_mult, dropout,
                 look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(hidden_size)
        self.attn = DiTAttention(hidden_size, num_heads, head_dim, dropout)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=hidden_size, mult=ff_mult, dropout=dropout)

    def forward(self, x, timestep, position_embeddings=None, block_diff=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=timestep)
        mask = (block_diff >= -float(self.look_backward_block)) & (block_diff <= float(self.look_ahead_block))
        attn_out = self.attn(norm, position_embeddings=position_embeddings, attention_mask=mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        norm2 = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm2)
        return x


class DiTRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids):
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class DiTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config.get("dim", 1024)
        num_layers = config.get("depth", 22)
        num_heads = config.get("heads", 16)
        head_dim = config.get("head_dim", 64)
        ff_mult = config.get("ff_mult", 2)
        self.mel_dim = config.get("mel_dim", 80)
        self.repeats = config.get("repeats", 2)
        num_embeds = config.get("num_embeds", 8193)
        emb_dim = config.get("emb_dim", 512)
        dropout = config.get("dropout", 0.1)
        block_size = config.get("block_size", 24)
        enc_dim = config.get("enc_dim", 192)
        enc_emb_dim = config.get("enc_emb_dim", 0)
        look_ahead_layers = config.get("look_ahead_layers", (10,))
        look_backward_layers = config.get("look_backward_layers", (0, 20))
        max_pos = config.get("max_position_embeddings", 8192)

        self.hidden_size = hidden_size
        self.block_size = block_size
        self.num_heads = num_heads

        self.time_embed = DiTTimestepEmbedding(hidden_size)
        self.text_embed = DiTCodecEmbedding(num_embeds, emb_dim, self.repeats)
        spk_config = {
            "mel_dim": config.get("mel_dim", 80),
            "channels": config.get("enc_channels", (256, 256, 256, 256, 768)),
            "kernel_sizes": config.get("enc_kernel_sizes", (5, 3, 3, 3, 1)),
            "dilations": config.get("enc_dilations", (1, 2, 3, 4, 1)),
            "res2net_scale": config.get("enc_res2net_scale", 2),
            "se_channels": config.get("enc_se_channels", 128),
            "attention_channels": config.get("enc_attention_channels", 128),
        }
        self.input_embed = DiTInputEmbedding(self.mel_dim, enc_dim, enc_emb_dim, emb_dim,
                                             hidden_size, spk_config)
        self.rotary_embed = DiTRotaryEmbedding(head_dim, max_position_embeddings=max_pos)

        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_blocks.append(DiTDecoderLayer(
                hidden_size, num_heads, head_dim, ff_mult, dropout,
                look_ahead_block=1 if i in look_ahead_layers else 0,
                look_backward_block=1 if i in look_backward_layers else 0,
            ))
        self.norm_out = AdaLayerNormZeroFinal(hidden_size)
        self.proj_out = nn.Linear(hidden_size, self.mel_dim)

    def _create_block_diff(self, x):
        seq_len = x.shape[1]
        idx = torch.arange(seq_len, device=x.device) // self.block_size
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)
        return diff.unsqueeze(0).unsqueeze(0)

    def forward(self, hidden_states, condition_vector, speaker_embedding,
                quantized_code, time_step, drop_audio_conditioning=False,
                drop_code=False, apply_cfg=True):
        B = hidden_states.shape[0]
        if time_step.ndim == 0:
            time_step = time_step.repeat(B)
        t_emb = self.time_embed(time_step)
        code_emb = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        code_emb_uncond = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None
        x = self.input_embed(hidden_states, speaker_embedding, condition_vector,
                             code_emb, drop_audio_cond=drop_audio_conditioning,
                             code_embed_uncond=code_emb_uncond, apply_cfg=apply_cfg)
        pos_ids = torch.arange(x.shape[1], device=x.device)[None, :].repeat(B if not apply_cfg else B * 2, 1)
        pos_emb = self.rotary_embed(x, pos_ids)
        block_diff = self._create_block_diff(x)
        for block in self.transformer_blocks:
            x = block(x, t_emb if not apply_cfg else t_emb.repeat(2, 1),
                      position_embeddings=pos_emb, block_diff=block_diff)
        x = self.norm_out(x, t_emb if not apply_cfg else t_emb.repeat(2, 1))
        return self.proj_out(x)

    @torch.no_grad()
    def sample(self, conditioning_vector, reference_mel, quantized_code,
               num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0):
        max_dur = quantized_code.shape[1] * self.repeats
        B = reference_mel.shape[0]
        x0 = torch.randn([B, max_dur, self.mel_dim], dtype=reference_mel.dtype,
                          device=quantized_code.device)
        cond = conditioning_vector.unsqueeze(1).repeat(1, max_dur, 1)

        def ode_fn(t, x):
            if guidance_scale < 1e-5:
                return self(x, speaker_embedding=cond, condition_vector=reference_mel,
                            quantized_code=quantized_code, time_step=t,
                            drop_audio_conditioning=False, drop_code=False, apply_cfg=False)
            out = self(x, quantized_code=quantized_code, speaker_embedding=cond,
                       condition_vector=reference_mel, time_step=t, apply_cfg=True)
            guided, null = torch.chunk(out, 2, dim=0)
            return guided + (guided - null) * guidance_scale

        time_pts = torch.linspace(0, 1, num_steps, device=quantized_code.device,
                                  dtype=conditioning_vector.dtype)
        if sway_coefficient is not None:
            time_pts += sway_coefficient * (torch.cos(torch.pi / 2 * time_pts) - 1 + time_pts)

        x = x0
        for i in range(len(time_pts) - 1):
            t0, t1 = time_pts[i], time_pts[i + 1]
            dt = t1 - t0
            k1 = ode_fn(t0, x)
            k2 = ode_fn(t0 + dt / 3, x + dt * k1 / 3)
            k3 = ode_fn(t0 + 2 * dt / 3, x + dt * (k2 - k1 / 3))
            k4 = ode_fn(t1, x + dt * (k1 - k2 + k3))
            x = x + (k1 + 3 * (k2 + k3) + k4) * dt / 8

        return x.permute(0, 2, 1)


# ============================================================
# Token2Wav: chains DiT → BigVGAN
# ============================================================

class Token2WavModel(nn.Module):
    def __init__(self, dit_config: dict, bigvgan_config: dict):
        super().__init__()
        self.code2wav_dit_model = DiTModel(dit_config)
        self.code2wav_bigvgan_model = BigVGANModel(
            mel_dim=bigvgan_config.get("mel_dim", 80),
            upsample_initial_channel=bigvgan_config.get("upsample_initial_channel", 1536),
            upsample_rates=bigvgan_config.get("upsample_rates", [5, 3, 2, 2, 2, 2]),
            upsample_kernel_sizes=bigvgan_config.get("upsample_kernel_sizes", [11, 7, 4, 4, 4, 4]),
            resblock_kernel_sizes=bigvgan_config.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=bigvgan_config.get("resblock_dilation_sizes",
                                                        [(1, 3, 5), (1, 3, 5), (1, 3, 5)]),
        )

    def forward(self, code, conditioning, reference_mel,
                num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0):
        mel = self.code2wav_dit_model.sample(
            conditioning, reference_mel, code,
            num_steps=num_steps, guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        waveform = self.code2wav_bigvgan_model(mel)
        return waveform

    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str = "cpu"):
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            root_config = json.load(f)

        t2w_config = root_config.get("token2wav_config", {})
        dit_config = t2w_config.get("dit_config", {})
        bigvgan_config = t2w_config.get("bigvgan_config", {})

        model = cls(dit_config, bigvgan_config)

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        t2w_files = set()
        for name, shard in index["weight_map"].items():
            if name.startswith("token2wav."):
                t2w_files.add(shard)

        if not t2w_files:
            raise FileNotFoundError(
                f"No token2wav weight shards found in {ckpt_path}"
            )

        from safetensors.torch import load_file
        state_dict = {}
        for shard_file in t2w_files:
            shard_path = os.path.join(ckpt_path, shard_file)
            shard_weights = load_file(shard_path)
            for k, v in shard_weights.items():
                if k.startswith("token2wav."):
                    new_key = k[len("token2wav."):]
                    state_dict[new_key] = v

        if not state_dict:
            raise ValueError(
                f"No token2wav weights loaded from prefix 'token2wav.' in {ckpt_path}"
            )

        required_components = (
            "code2wav_dit_model.input_embed",
            "code2wav_dit_model.transformer_blocks",
            "code2wav_bigvgan_model",
        )
        loaded_keys = set(state_dict.keys())
        for component in required_components:
            if not any(k.startswith(component) for k in loaded_keys):
                raise RuntimeError(
                    f"Token2Wav missing required component '{component}' in {ckpt_path}"
                )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Token2Wav failed to load weights, missing keys: {missing[:20]}"
            )
        model = model.to(device).float()
        model.eval()
        logger.info(f"Token2WavModel loaded from {ckpt_path} on {device}")
        return model

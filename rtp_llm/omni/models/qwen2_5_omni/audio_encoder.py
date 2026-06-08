"""Qwen2.5-Omni audio encoder: Whisper-variant with chunked attention.

Ported from HuggingFace transformers (Apache 2.0 license).
Key differences from Qwen2AudioEncoder:
- Chunked attention via cu_seqlens (n_window=100)
- Sinusoidal position embeddings (not learned)
- Built-in projection: ln_post + proj(1280→3584)
- audio_bos_eos_token boundary embeddings
- Index-based stride-2 pooling
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AudioEncoderConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    scale_embedding: bool = False
    output_dim: int = 3584
    n_window: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "AudioEncoderConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )


def chunk_and_pad_features(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chunk_size = n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_size).long()
    chunk_lengths = torch.full(
        (chunk_num.sum(),), chunk_size, dtype=torch.long, device=feature_lens.device
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_size
    chunk_lengths = torch.where(chunk_lengths == 0, chunk_size, chunk_lengths)

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(
        chunk_list, batch_first=True
    ).transpose(1, 2)
    return padded_feature, chunk_lengths


def get_valid_indices(chunk_lengths: torch.Tensor) -> torch.Tensor:
    after_conv1 = (chunk_lengths - 1) // 2 + 1
    max_len = after_conv1.max().item()
    mask = torch.arange(max_len, device=chunk_lengths.device) < after_conv1.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_pool_indices(feature_lens: torch.Tensor) -> torch.Tensor:
    after_conv1 = (feature_lens - 1) // 2 + 1
    num_pooled = (after_conv1 - 2) // 2 + 1
    offsets = F.pad(after_conv1[:-1].cumsum(0), (1, 0), value=0)
    pair_offsets = torch.repeat_interleave(offsets, num_pooled)
    local_indices = torch.arange(num_pooled.sum(), device=feature_lens.device)
    local_indices -= torch.repeat_interleave(
        F.pad(num_pooled[:-1].cumsum(0), (1, 0), value=0), num_pooled
    )
    return pair_offsets + local_indices * 2


def get_audio_cu_seqlens(chunk_lengths: torch.Tensor) -> torch.Tensor:
    after_conv1 = (chunk_lengths - 1) // 2 + 1
    return F.pad(after_conv1.cumsum(0), (1, 0), value=0).to(torch.int32)


class Qwen2_5OmniAudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(q, lengths, dim=0)
        k_chunks = torch.split(k, lengths, dim=0)
        v_chunks = torch.split(v, lengths, dim=0)

        outputs = []
        for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
            # [L, H, D] -> [1, H, L, D]
            qc = qc.unsqueeze(0).transpose(1, 2)
            kc = kc.unsqueeze(0).transpose(1, 2)
            vc = vc.unsqueeze(0).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                qc, kc, vc, is_causal=False, scale=1.0
            )
            # [1, H, L, D] -> [L, H, D]
            outputs.append(attn_out.squeeze(0).transpose(0, 1))

        hidden_states = torch.cat(outputs, dim=0)
        hidden_states = hidden_states.reshape(seq_len, self.embed_dim)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioAttention(config, layer_idx=layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class Qwen2_5OmniAudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList(
            [
                Qwen2_5OmniAudioEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, config.output_dim)

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        padded_feature, chunk_lengths = chunk_and_pad_features(
            input_features, feature_lens, self.n_window
        )
        valid_indices = get_valid_indices(chunk_lengths)
        pool_indices = get_pool_indices(feature_lens)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths)

        padded_feature = padded_feature.to(self.conv1.weight.dtype)
        padded_mask = (
            (
                torch.arange(padded_feature.shape[2], device=padded_feature.device)
                < chunk_lengths.unsqueeze(1)
            )
            .unsqueeze(1)
            .long()
        )
        padded_embed = F.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = F.gelu(self.conv2(padded_embed)).transpose(1, 2)

        pos_embed = self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ]
        padded_embed = padded_embed + pos_embed.unsqueeze(0).to(padded_embed.dtype)

        hidden_states = torch.index_select(
            padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)

        # Stride-2 average pooling via precomputed indices
        hidden_states = (
            hidden_states[pool_indices] + hidden_states[pool_indices + 1]
        ) / 2
        hidden_states = self.proj(self.ln_post(hidden_states))
        return hidden_states

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Qwen2_5OmniAudioEncoder":
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            root_config = json.load(f)
        audio_config_dict = root_config["thinker_config"]["audio_config"]
        config = AudioEncoderConfig.from_dict(audio_config_dict)

        model = cls(config)

        from safetensors.torch import load_file

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        prefix = "thinker.audio_tower."
        needed_shards = set()
        for key in index["weight_map"]:
            if key.startswith(prefix):
                needed_shards.add(index["weight_map"][key])

        all_weights = {}
        for shard in needed_shards:
            shard_path = os.path.join(ckpt_path, shard)
            shard_weights = load_file(shard_path)
            for k, v in shard_weights.items():
                if k.startswith(prefix):
                    local_key = k[len(prefix):]
                    all_weights[local_key] = v

        missing, unexpected = model.load_state_dict(all_weights, strict=False)
        # positional_embedding is computed (not learned), so it's expected to be missing
        real_missing = [k for k in missing if "positional_embedding" not in k]
        if real_missing:
            logger.warning(f"Missing keys: {real_missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Audio encoder loaded: {total_params/1e6:.1f}M params")

        model = model.to(device=device, dtype=dtype).eval()
        return model

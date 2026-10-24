"""Vendored DiffusionGemma vision tower + multimodal embedder.

Reference: transformers 5.20 modeling_gemma4.py / modeling_diffusion_gemma.py
(Gemma4VisionModel, Gemma4VisionEncoder, Gemma4MultimodalEmbedder). The
pinned bazel transformers (5.2.0) predates the gemma4 family, so the tower is
implemented here with plain torch. Module and parameter names intentionally
match the HF state dict layout so the checkpoint mapping is the identity on
the vision_tower prefix:

  patch_embedder.input_proj.weight                 [hidden, 3*patch^2]
  patch_embedder.position_embedding_table          [2, pos_emb_size, hidden]
  encoder.layers.{i}.self_attn.{q,k,v,o}_proj.linear.weight
  encoder.layers.{i}.self_attn.{q,k}_norm.weight
  encoder.layers.{i}.mlp.{gate,up,down}_proj.linear.weight
  encoder.layers.{i}.{input,post_attention,pre_feedforward,
      post_feedforward}_layernorm.weight
  std_bias / std_scale                             [hidden]
  embed_vision.embedding_projection.weight         [text_hidden, hidden]
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DiffusionGemmaVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    rope_theta: float = 100.0
    standardize: bool = True

    @classmethod
    def from_hf(cls, vision_config: dict) -> "DiffusionGemmaVisionConfig":
        return cls(
            hidden_size=int(vision_config["hidden_size"]),
            intermediate_size=int(vision_config["intermediate_size"]),
            num_hidden_layers=int(vision_config["num_hidden_layers"]),
            num_attention_heads=int(vision_config["num_attention_heads"]),
            num_key_value_heads=int(
                vision_config.get("num_key_value_heads", vision_config["num_attention_heads"])
            ),
            head_dim=int(vision_config.get("head_dim", vision_config["hidden_size"] // vision_config["num_attention_heads"])),
            rms_norm_eps=float(vision_config.get("rms_norm_eps", 1e-6)),
            patch_size=int(vision_config.get("patch_size", 16)),
            position_embedding_size=int(vision_config.get("position_embedding_size", 10240)),
            pooling_kernel_size=int(vision_config.get("pooling_kernel_size", 3)),
            rope_theta=float(vision_config.get("rope_parameters", {}).get("rope_theta", 100.0)),
            standardize=bool(vision_config.get("standardize", True)),
        )


class _RMSNorm(nn.Module):
    """Gemma4 RMSNorm; with_scale=False reproduces the v_norm / pre-projection
    norm variants that carry no weight."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = hidden_states.float()
        mean_squared = normed.pow(2).mean(-1, keepdim=True) + self.eps
        normed = normed * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(hidden_states)


class _Linear(nn.Module):
    """Gemma4 vision linear (use_clipped_linears is False for this checkpoint)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


class DiffusionGemmaVisionPatchEmbedder(nn.Module):
    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        # Direct nn.Linear: checkpoint key is input_proj.weight (no .linear infix).
        self.input_proj = nn.Linear(3 * config.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
        padding_positions: torch.BoolTensor,
    ) -> torch.Tensor:
        # Reference scales pixels in model code and applies no normalization.
        pixel_values = 2 * (pixel_values - 0.5)
        pixel_values = pixel_values.to(self.input_proj.weight.dtype)
        hidden_states = self.input_proj(pixel_values)
        clamped = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        return hidden_states + position_embeddings


class DiffusionGemmaVisionMLP(nn.Module):
    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.gate_proj = _Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = _Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = _Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        # gelu_pytorch_tanh
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DiffusionGemmaVisionAttention(nn.Module):
    """Bidirectional MHA with per-head qk_norm, scale-free v_norm, axial 2D
    rope over the whole head dim and softmax scale 1.0."""

    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads
        self.scaling = 1.0
        self.q_proj = _Linear(config.hidden_size, self.head_num * self.head_dim)
        self.k_proj = _Linear(config.hidden_size, self.kv_head_num * self.head_dim)
        self.v_proj = _Linear(config.hidden_size, self.kv_head_num * self.head_dim)
        self.o_proj = _Linear(self.head_num * self.head_dim, config.hidden_size)
        self.q_norm = _RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(config.head_dim, config.rms_norm_eps)
        self.v_norm = _RMSNorm(config.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, patches, _ = hidden_states.shape
        shape = (batch, patches, -1, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(shape))
        key = self.k_norm(self.k_proj(hidden_states).view(shape))
        query = _apply_multidimensional_rope(query, cos, sin, pixel_position_ids)
        key = _apply_multidimensional_rope(key, cos, sin, pixel_position_ids)
        value = self.v_norm(self.v_proj(hidden_states).view(shape))

        # [B, H, P, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = self.o_proj(attn_output.reshape(batch, patches, -1))
        return attn_output


class DiffusionGemmaVisionLayer(nn.Module):
    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.self_attn = DiffusionGemmaVisionAttention(config)
        self.mlp = DiffusionGemmaVisionMLP(config)
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, position_ids, attention_mask):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, position_ids, attention_mask
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states


class DiffusionGemmaVisionRotaryEmbedding(nn.Module):
    """Axial 2D rope: half the head dim carries H frequencies duplicated,
    the other half W frequencies duplicated (H-H-W-W interleave)."""

    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, pixel_position_ids: torch.LongTensor
    ):
        freqs = pixel_position_ids[..., None].float() * self.inv_freq.to(
            device=x.device, dtype=torch.float
        )
        cos = self._recompose(freqs.cos())
        sin = self._recompose(freqs.sin())
        return cos.to(x.dtype), sin.to(x.dtype)

    @staticmethod
    def _recompose(freq):
        freq_h, freq_w = freq[..., 0, :], freq[..., 1, :]
        return torch.cat([freq_h, freq_h, freq_w, freq_w], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
) -> torch.Tensor:
    """2-D rope: head dim splits into one 36-channel block per axis; each
    block rotates with its own cos/sin slice (broadcast over heads)."""
    ndim = position_ids.shape[-1]
    channels = x.shape[-1]
    per_dim = 2 * (channels // (2 * ndim))
    cos = cos.unsqueeze(2)  # [B, P, 1, D]
    sin = sin.unsqueeze(2)
    parts = torch.split(x, [per_dim] * ndim, dim=-1)
    cos_parts = torch.split(cos, [per_dim] * ndim, dim=-1)
    sin_parts = torch.split(sin, [per_dim] * ndim, dim=-1)
    out = [
        p * c + _rotate_half(p) * s
        for p, c, s in zip(parts, cos_parts, sin_parts)
    ]
    return torch.cat(out, dim=-1)


class DiffusionGemmaVisionPooler(nn.Module):
    """Spatial pooling + sqrt(hidden) scaling in fp32; standardization is
    applied by the tower (needs std_bias/std_scale)."""

    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
        length: int,
    ):
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {tuple(hidden_states.shape)} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
        padding_positions: torch.BoolTensor,
        output_length: int,
    ):
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens ({output_length}) than patches ({hidden_states.shape[1]})."
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states.float() * self.root_hidden_size
        return hidden_states, padding_positions


class _VisionEncoder(nn.Module):
    """Named `encoder` so state-dict keys match the checkpoint layout
    (encoder.layers.{i}.…), exactly like the HF Gemma4VisionEncoder."""

    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [DiffusionGemmaVisionLayer(config) for _ in range(config.num_hidden_layers)]
        )


class DiffusionGemmaVisionTower(nn.Module):
    """Equivalent of HF Gemma4VisionModel.forward: returns standardized soft
    tokens [num_valid, hidden] with padding rows stripped."""

    def __init__(self, config: DiffusionGemmaVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = DiffusionGemmaVisionPatchEmbedder(config)
        self.encoder = _VisionEncoder(config)
        self.rotary_emb = DiffusionGemmaVisionRotaryEmbedding(config)
        self.pooler = DiffusionGemmaVisionPooler(config)
        if config.standardize:
            self.register_buffer("std_bias", torch.zeros(config.hidden_size))
            self.register_buffer("std_scale", torch.ones(config.hidden_size))

    @property
    def dtype(self):
        return self.patch_embedder.input_proj.weight.dtype

    @property
    def device(self):
        return self.patch_embedder.input_proj.weight.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        pooling_k = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_k * pooling_k)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        hidden_states = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        cos, sin = self.rotary_emb(hidden_states, pixel_position_ids)
        # Bidirectional additive mask over padding keys.
        attention_mask = None
        if padding_positions.any():
            attention_mask = torch.zeros(
                (padding_positions.shape[0], 1, 1, padding_positions.shape[1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask.masked_fill_(
                padding_positions[:, None, None, :], torch.finfo(hidden_states.dtype).min
            )

        for layer in self.encoder.layers:
            hidden_states = layer(
                hidden_states, cos, sin, pixel_position_ids, attention_mask
            )

        hidden_states, pooler_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.standardize:
            hidden_states = (
                hidden_states - self.std_bias.float()
            ) * self.std_scale.float()
        return hidden_states.to(self.dtype)


class DiffusionGemmaMultimodalEmbedder(nn.Module):
    """Scale-free RMSNorm + projection from vision hidden to text hidden
    (checkpoint: model.encoder.embed_vision.embedding_projection.weight)."""

    def __init__(self, vision_hidden: int, text_hidden: int, eps: float):
        super().__init__()
        self.embedding_pre_projection_norm = _RMSNorm(vision_hidden, eps, with_scale=False)
        # Direct nn.Linear: checkpoint key is embedding_projection.weight.
        self.embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds)
        )


__all__ = [
    "DiffusionGemmaVisionConfig",
    "DiffusionGemmaVisionTower",
    "DiffusionGemmaMultimodalEmbedder",
]

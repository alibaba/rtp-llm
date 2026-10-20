"""Whisper audio encoder tower (A1) for openai/whisper-large-v3.

Faithful port of transformers `WhisperEncoder` (checked against v4.46.3 and
v5.20.1 modeling_whisper.py; both are structure-identical):
- conv1: Conv1d(num_mel_bins=128 -> d_model=1280, k=3, s=1, p=1) + GELU
- conv2: Conv1d(d_model -> d_model, k=3, s=2, p=1) + GELU   (3000 -> 1500 frames)
- + learned positional embedding table [1500, 1280]
- 32 pre-LN encoder layers: LN -> self-attn(20 heads, no RoPE, non-causal,
  NO padding mask — the reference implementation ignores attention_mask) ->
  residual -> LN -> GELU MLP -> residual
- final LayerNorm
- fp16 inf/nan clamp after every residual add (modeling_whisper.py:635-639).

Bias layout matches the checkpoint exactly: q/v/out projections carry bias,
k_proj has no bias. The audio runs in the multimodal process engine; its
output (last hidden state [<=1500, 1280]) is delivered to the decoder model
as request-level cross-attention memory (mm_skip_token_expansion).
"""

import math  # noqa: F401  (kept: parity reference formulas live here)
from typing import Optional  # noqa: F401

import torch
from torch import nn

try:
    # Bit-exact Triton replacement for the eager add+nan_to_num pair (the
    # decoder's Phase C kernel); non-CUDA/non-fp16 hosts fall back inside.
    from rtp_llm.models_py.model_desc.whisper_fused_kernels import fused_residual_clamp as _fused_residual_clamp
except Exception:  # pragma: no cover - CPU-only import hosts
    _fused_residual_clamp = None

try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops as _rtp_llm_ops
except Exception:  # pragma: no cover - CPU-only import hosts
    _rtp_llm_ops = None


class _CudaLayerNorm(nn.LayerNorm):
    """nn.LayerNorm parameter contract (checkpoint load by name) with the
    engine's generalLayerNorm kernel as the forward. Falls back to the ATen
    path when the ops binding is unavailable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _rtp_llm_ops is None or not x.is_cuda or x.dim() != 2 or not x.is_contiguous():
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _rtp_llm_ops.layernorm(output, x, self.weight.data, self.bias, self.eps)
        return output


def _clamped_fp32_add(residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if _fused_residual_clamp is not None:
        return _fused_residual_clamp(residual, x)
    out = residual + x
    if out.dtype == torch.float16:
        clamp_value = torch.finfo(out.dtype).max - 1000
        out = torch.nan_to_num(out, nan=clamp_value, posinf=clamp_value, neginf=-clamp_value)
    return out


class WhisperAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, k_bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        # [B, S, H, D] -> [B, H, S, D]; SDPA consumes the strided view, so no
        # explicit contiguous copy (the eager matmul path needed three per
        # layer plus the SxS softmax materialization).
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()
        # The transformers default (attn_implementation="sdpa") applies the
        # head-dim scale inside the kernel; SDPA's default scale equals
        # head_dim**-0.5, so the projection-time pre-scale is dropped.
        # Non-causal, no padding mask: whisper encoder attends to all positions.
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, eps: float):
        super().__init__()
        self.self_attn = WhisperAttention(d_model, num_heads)
        self.self_attn_layer_norm = _CudaLayerNorm(d_model, eps=eps)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = _CudaLayerNorm(d_model, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The engine generalLayerNorm kernel takes [rows, cols]; the tower runs
        # [B, S, D]. Fold the batch/frame axes around the LN calls only.
        residual = hidden_states
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        hidden_states = self.self_attn_layer_norm(rows).view_as(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = _clamped_fp32_add(residual, hidden_states)
        residual = hidden_states
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        hidden_states = self.final_layer_norm(rows).view_as(hidden_states)
        hidden_states = nn.functional.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = _clamped_fp32_add(residual, hidden_states)
        return hidden_states


class WhisperEncoderTower(nn.Module):
    main_input_name = "input_features"

    def __init__(
        self,
        num_mel_bins: int = 128,
        d_model: int = 1280,
        num_layers: int = 32,
        num_heads: int = 20,
        ffn_dim: int = 5120,
        max_source_positions: int = 1500,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.max_source_positions = max_source_positions
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(d_model, num_heads, ffn_dim, layer_norm_eps) for _ in range(num_layers)]
        )
        self.layer_norm = _CudaLayerNorm(d_model, eps=layer_norm_eps)

    @property
    def num_audio_positions(self) -> int:
        return self.max_source_positions

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """input_features: [B, num_mel_bins=128, 3000] -> hidden [B, 1500, d_model]."""
        if input_features.dtype != self.conv1.weight.dtype:
            input_features = input_features.to(self.conv1.weight.dtype)
        expected_seq_length = self.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"whisper expects padded mel frames {expected_seq_length}, got {input_features.shape[-1]}"
            )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        seq_len = inputs_embeds.shape[1]
        positions = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        hidden_states = inputs_embeds + self.embed_positions(positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states.reshape(-1, hidden_states.shape[-1])).view_as(hidden_states)
        return hidden_states

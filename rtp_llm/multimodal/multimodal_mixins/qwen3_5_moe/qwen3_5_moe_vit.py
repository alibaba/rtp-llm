# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (Apache-2.0), commit
# bb233626caa31602728f7ee4625f3d2a4d1a3ad5:
# vllm/model_executor/models/qwen3_vl.py and qwen2_5_vl.py.
# See vision_sources.json for source hashes and runtime adaptations.
"""Qwen3.5 pure vision encoder using vLLM's unquantized, TP=1 compute path.

Video preprocessing, scheduling, and embedding assembly stay in the RTP mixin.
RTP's weight names and HF output envelope are preserved. QKV, row and column
parallel layers use the ported vLLM classes, parameters, and weight loaders with
disable_tp=True for each replicated ViT worker.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .vision_linear import (
    ColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from .vision_linear_runtime import QuantizationConfig


@lru_cache(maxsize=1)
def _flash_attention_backends():
    fa4, fa2 = None, None
    try:
        # Bazel subprocesses do not execute wheel .pth files.
        import os
        import sys

        import nvidia_cutlass_dsl

        for package_dir in nvidia_cutlass_dsl.__path__:
            extra = os.path.join(package_dir, "python_packages")
            if os.path.isdir(extra) and extra not in sys.path:
                sys.path.insert(0, extra)
        from flash_attn.cute import flash_attn_varlen_func

        fa4 = flash_attn_varlen_func
    except ImportError as error:
        logging.info("Qwen3.5 ViT FA4 unavailable: %s", error)
    try:
        from flash_attn import flash_attn_varlen_func

        fa2 = flash_attn_varlen_func
    except ImportError:
        pass
    return fa4, fa2


@lru_cache(maxsize=1)
def _dense_flash_attention_backend():
    # Initialize CuTe's import path in Bazel subprocesses first.
    _flash_attention_backends()
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


def _use_dense_fa4(query, lengths):
    return (
        query.is_cuda
        and query.dtype == torch.bfloat16
        and query.shape[-2] == 16
        and query.shape[-1] == 72
        and lengths[0] >= 1024
        and all(length == lengths[0] for length in lengths)
        and torch.cuda.get_device_capability(query.device) == (10, 3)
    )


def _fa4_vision_attention(query, key, value, cu_seqlens, lengths, scaling):
    # Keep equal-size frame/image segments independent in the dense scheduler.
    if _use_dense_fa4(query, lengths):
        batch_shape = (len(lengths), lengths[0])
        output = _dense_flash_attention_backend()(
            query.view(*batch_shape, *query.shape[-2:]),
            key.view(*batch_shape, *key.shape[-2:]),
            value.view(*batch_shape, *value.shape[-2:]),
            causal=False,
            softmax_scale=scaling,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(value.shape)

    fa4, _ = _flash_attention_backends()
    return fa4(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=False,
        softmax_scale=scaling,
    )


def _select_attention_backend(tensor, requested="auto"):
    if requested not in ("auto", "sdpa", "fa4", "flash_attention_2"):
        raise ValueError(f"unknown Qwen3.5 vision attention backend: {requested}")
    if requested == "sdpa":
        return "sdpa"
    if not tensor.is_cuda or tensor.dtype not in (torch.float16, torch.bfloat16):
        if requested == "auto":
            return "sdpa"
        raise ValueError(f"vision backend {requested} requires CUDA FP16/BF16 input")
    capability = torch.cuda.get_device_capability(tensor.device)
    fa4, fa2 = _flash_attention_backends()
    if (
        requested in ("auto", "fa4")
        and fa4 is not None
        and capability in ((9, 0), (10, 0), (10, 3), (11, 0))
    ):
        return "fa4"
    if (
        requested in ("auto", "flash_attention_2")
        and fa2 is not None
        and capability[0] in (8, 9)
    ):
        return "flash_attention_2"
    if requested != "auto":
        raise RuntimeError(
            f"requested vision backend {requested} is unavailable on {capability}"
        )
    return "sdpa"


class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5MoeVisionLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # CUDA LayerNorm can wrap flattened 32-bit offsets for large video batches.
        # Keep each call below the index limit without splitting a normalized row.
        max_elements = (1 << 31) - 1
        if not hidden_states.is_cuda or hidden_states.numel() <= max_elements:
            return super().forward(hidden_states)

        row_size = math.prod(self.normalized_shape)
        max_rows = max(1, max_elements // row_size)
        rows = hidden_states.reshape(-1, *self.normalized_shape)
        output = torch.empty_like(rows)
        for start in range(0, rows.shape[0], max_rows):
            end = min(start + max_rows, rows.shape[0])
            output[start:end].copy_(super().forward(rows[start:end]))
        return output.reshape(hidden_states.shape)


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

PIN_MEMORY = torch.cuda.is_available()

if HAS_TRITON:

    @triton.jit
    def _bilinear_pos_embed_kernel(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused bilinear pos-embed interpolation with spatial-merge reorder."""
        pid = tl.program_id(0)
        total_spatial = H * W
        spatial_idx = pid % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = tl.math.floor(h_frac).to(tl.int32)
        wf = tl.math.floor(w_frac).to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        # Packed high-resolution videos can exceed 2**31 tensor elements.
        out_off = pid.to(tl.int64) * HIDDEN_DIM

        # Cast weights to output dtype so the multiply-accumulate stays
        # in the same precision as the native PyTorch implementation.
        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
            cols = d + tl.arange(0, BLOCK_D)
            mask = cols < HIDDEN_DIM

            e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
            e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
            e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
            e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

            val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

            tl.store(output_ptr + out_off + cols, val, mask=mask)

    def triton_pos_embed_interpolate(
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Launch the fused Triton kernel for one (t,h,w) grid.

        Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
        bilinearly-interpolated position embeddings in spatial-merge order.
        """
        assert (
            h % m_size == 0 and w % m_size == 0
        ), f"h={h} and w={w} must be divisible by m_size={m_size}"
        hidden_dim = embed_weight.shape[1]
        total_out = t * h * w
        output = torch.empty(
            total_out,
            hidden_dim,
            device=embed_weight.device,
            dtype=dtype,
        )

        h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
        w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0

        BLOCK_D = triton.next_power_of_2(hidden_dim)

        _bilinear_pos_embed_kernel[(total_out,)](
            embed_weight,
            output,
            h,
            w,
            h_scale,
            w_scale,
            num_grid_per_side,
            m_size,
            hidden_dim,
            BLOCK_D,
        )
        return output


def pos_embed_interpolate_native(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch bilinear position-embedding interpolation.

    Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
    bilinearly-interpolated position embeddings in spatial-merge order.
    """
    assert (
        h % m_size == 0 and w % m_size == 0
    ), f"h={h} and w={w} must be divisible by m_size={m_size}"
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        h,
        dtype=torch.float32,
        device=device,
    )
    w_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        w,
        dtype=torch.float32,
        device=device,
    )

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


class Conv3dLayer(nn.Conv3d):
    """vLLM Conv3dLayer's non-overlapping patch GEMM, with torch parameters."""

    @property
    def input_size(self):
        return self.in_channels * int(np.prod(self.kernel_size))

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        B, C, T, H, W = x.shape
        K1, K2, K3 = self.kernel_size
        T, H, W = T // K1, H // K2, W // K3
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)
        x = F.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        x = x.view(B, T, H, W, self.out_channels).permute(0, 4, 1, 2, 3)
        return x

    def forward(self, x):
        return self._forward_mulmat(x)


class ApplyRotaryEmb(nn.Module):
    def forward(self, x, cos, sin):
        if x.is_cuda:
            from .vision_kernels import apply_rotary

            return apply_rotary(x, cos, sin, interleaved=False, inplace=False)
        # CPU reference for the same non-interleaved rotation.
        rotary_dim = 2 * cos.shape[-1]
        x1, x2 = x[..., :rotary_dim].chunk(2, dim=-1)
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        output = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, x[..., rotary_dim:]), -1)
        return output


class VisionAttention(nn.Module):
    """vLLM's unquantized FA wrapper; use RTP's installed backend dependency."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.backend = "auto"

    def forward(self, query, key, value, cu_seqlens, max_seqlen, sequence_lengths=None):
        batch_size = query.shape[0]
        q, k, v = [
            einops.rearrange(t, "b s h d -> (b s) h d") for t in (query, key, value)
        ]
        backend = _select_attention_backend(q, self.backend)
        self.last_backend = backend
        max_len = int(max_seqlen.item())
        if backend == "fa4":
            lengths = sequence_lengths
            if lengths is None:
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
            self.last_layout = "dense" if _use_dense_fa4(q, lengths) else "varlen"
            output = _fa4_vision_attention(q, k, v, cu_seqlens, lengths, self.scale)
            if isinstance(output, tuple):
                output = output[0]
        elif backend == "flash_attention_2":
            _, fa2 = _flash_attention_backends()
            output = fa2(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_len,
                max_seqlen_k=max_len,
                softmax_scale=self.scale,
                causal=False,
            )
            if isinstance(output, tuple):
                output = output[0]
        else:
            lengths = sequence_lengths
            if lengths is None:
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
            output = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        qi.transpose(0, 1)[None],
                        ki.transpose(0, 1)[None],
                        vi.transpose(0, 1)[None],
                        dropout_p=0.0,
                        is_causal=False,
                        scale=self.scale,
                    )[0].transpose(0, 1)
                    for qi, ki, vi in zip(
                        q.split(lengths), k.split(lengths), v.split(lengths)
                    )
                ]
            )
        return einops.rearrange(output, "(b s) h d -> b s h d", b=batch_size)


class Qwen3_5MoeVisionRotaryEmbedding(nn.Module):
    """vLLM RotaryEmbeddingBase's default cosine/sine cache."""

    def __init__(self, dim: int, theta: float = 10000.0, max_position=8192):
        rotary_dim = dim
        base = theta
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer(
            "cos_sin_cache", cache.to(torch.get_default_dtype()), persistent=False
        )

    def get_cos_sin(self, seqlen):
        return self.cos_sin_cache[:seqlen].chunk(2, dim=-1)


class Qwen3_5MoeVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5MoeVisionConfig):
        patch_size = config.patch_size
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.in_channels
        hidden_size = config.hidden_size
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen3_5MoeVisionMLP(nn.Module):
    def __init__(self, config: Qwen3_5MoeVisionConfig, quant_config=None, prefix=""):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        bias = True
        act_fn = nn.GELU(approximate="tanh")
        super().__init__()
        use_data_parallel = True  # RTP runs one full ViT per worker.
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_5MoeVisionAttention(nn.Module):
    """vLLM Q/K packing, fused rotary, varlen attention, and projection."""

    def __init__(self, config: Qwen3_5MoeVisionConfig, quant_config=None, prefix=""):
        embed_dim = projection_size = config.hidden_size
        num_heads = config.num_heads
        super().__init__()
        self.config = config
        # vLLM data-parallel ViT mode: each RTP worker has the complete tower.
        self.tp_size = 1
        self.tp_rank = 0
        self.hidden_size_per_attention_head = projection_size // num_heads
        self.num_attention_heads_per_partition = num_heads // self.tp_size
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=True,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=True,
        )
        self.attn = VisionAttention(self.hidden_size_per_attention_head**-0.5)
        self.apply_rotary_emb = ApplyRotaryEmb()

    @property
    def last_backend(self):
        return self.attn.last_backend

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        sequence_lengths: tuple[int, ...] | None,
    ) -> torch.Tensor:
        self.attn.backend = getattr(self.config, "vit_attention_backend", "auto")
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            # RoPE reads QKV strides directly and preserves the native head size.
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        context_layer = einops.rearrange(
            context_layer, "b s h d -> s b (h d)", b=batch_size
        ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class Qwen3_5MoeVisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeVisionConfig,
        attn_implementation="sdpa",
        *,
        norm_layer=None,
        quant_config=None,
        prefix="",
    ):
        dim = config.hidden_size
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(Qwen3_5MoeVisionLayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen3_5MoeVisionAttention(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_5MoeVisionMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        sequence_lengths: tuple[int, ...] | None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_5MoeVisionPatchMerger(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeVisionConfig,
        use_postshuffle_norm=False,
        *,
        norm_layer=None,
        quant_config=None,
        prefix="",
    ):
        d_model = config.out_hidden_size
        context_dim = config.hidden_size
        spatial_merge_size = config.spatial_merge_size
        super().__init__()
        use_data_parallel = True  # RTP runs one full ViT per worker.
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(Qwen3_5MoeVisionLayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3_5MoeVisionModel(PreTrainedModel):
    config_class = Qwen3_5MoeVisionConfig
    _no_split_modules = ["Qwen3_5MoeVisionBlock"]
    _can_record_outputs = {
        "hidden_states": Qwen3_5MoeVisionBlock,
        "attentions": Qwen3_5MoeVisionAttention,
    }
    _supports_sdpa = True
    _supports_flash_attn = True

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load RTP's visual-relative checkpoint through vLLM parameter loaders.

        RTP has already resolved the checkpoint prefix. Unlike the engine's
        AutoWeightsLoader, this bridge requires a complete fused-QKV checkpoint.
        """
        weights = dict(weights)
        params = dict(self.named_parameters())
        missing = params.keys() - weights.keys()
        unexpected = weights.keys() - params.keys()
        if missing or unexpected:
            raise ValueError(
                f"ViT checkpoint keys mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        # Check the complete checkpoint before modifying any parameter.
        for name, param in params.items():
            if weights[name].shape != param.shape:
                raise ValueError(
                    f"ViT weight {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(weights[name].shape)}"
                )
            if param.is_meta:
                raise ValueError("Materialize ViT parameters before loading weights")
        for name, param in params.items():
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                param.data.copy_(weights[name])
            else:
                weight_loader(param, weights[name])
        for module in self.modules():
            if isinstance(module, LinearBase):
                module.quant_method.process_weights_after_loading(module)
                module.update_param_tp_status()
        return set(params)

    def __init__(
        self,
        config: Qwen3_5MoeVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: object | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        if quant_config is not None:
            raise ValueError("This ViT port supports unquantized vision weights")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_position_embeddings = config.num_position_embeddings
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = config.temporal_patch_size
        self.deepstack_visual_indexes = (
            config.deepstack_visual_indexes
            if hasattr(config, "deepstack_visual_indexes")
            else []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3_5MoeVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(Qwen3_5MoeVisionLayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads

        # FP8 attention: Q/K/V become independent contiguous tensors
        # after quantization, so cu_seqlens uses uniform stride (no 3x V).
        self.fp8_padded_hidden_size = None
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)

        self.merger = Qwen3_5MoeVisionPatchMerger(
            config=config,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_5MoeVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.attn_backend = getattr(config, "vit_attention_backend", "auto")

        self.blocks = nn.ModuleList(
            [
                Qwen3_5MoeVisionBlock(
                    config=config,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(config.depth)
            ]
        )

        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("Qwen3.5 ViT expects gelu_pytorch_tanh")
        for block in self.blocks:
            block.attn.attn.backend = self.attn_backend
        self.post_init()

    @property
    def last_backend(self):
        return self.blocks[0].attn.last_backend

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            (
                self.rot_pos_ids(h, w, self.spatial_merge_size)
                if t == 1
                else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            )
            for t, h, w in grid_thw
        ]
        num_pos = sum(p.shape[0] for p in pos_ids)
        pinned = torch.empty(
            (num_pos, pos_ids[0].shape[1]),
            dtype=pos_ids[0].dtype,
            pin_memory=PIN_MEMORY,
        )
        pos_ids = torch.cat(pos_ids, dim=0, out=pinned).to(
            self.device, non_blocking=True
        )

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        interpolate_fn = (
            triton_pos_embed_interpolate
            if HAS_TRITON and self.pos_embed.weight.is_cuda
            else pos_embed_interpolate_native
        )
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                interpolate_fn(
                    self.pos_embed.weight,
                    t,
                    h,
                    w,
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )
        return torch.cat(outputs, dim=0)

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, ...] | None]:
        """Compute encoder metadata from grid_thw_list.

        Shared by the eager forward path, CUDA graph capture, and
        CUDA graph replay to avoid duplicated implementation.

        Args:
            grid_thw_list: Grid configurations as list of [t, h, w].
            max_batch_size: If set, pad cu_seqlens to this size
                (needed for CUDA graph capture/replay).
            max_frames_per_batch: If set, overrides max_batch_size for
                cu_seqlens padding. For video inputs each item contributes
                T attention sequences (frames); this sizes the buffer to
                the total frame budget so video replays never overflow.
            max_seqlen_override: If set, use this value for max_seqlen
                instead of computing from cu_seqlens (needed for CUDA
                graph capture to cover worst-case replay scenarios).
            device: Device to place tensors on. Defaults to self.device.
        """
        if device is None:
            device = self.device

        metadata: dict[str, torch.Tensor | tuple[int, ...] | None] = {}

        # Positional embeddings
        metadata["pos_embeds"] = self.fast_pos_embed_interpolate(grid_thw_list)
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = rotary_cos
        metadata["rotary_pos_emb_sin"] = rotary_sin

        # cu_seqlens from grid_thw
        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum(
            dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        # Pad cu_seqlens to the required number of sequences.
        # For videos each item contributes T frames = T attention sequences,
        # so the total can exceed max_batch_size. max_frames_per_batch
        # overrides the pad target when set.
        pad_to = (
            max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        )
        if pad_to is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < pad_to:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        # Keep segment lengths on CPU so dense dispatch never copies CUDA
        # cu_seqlens back to the host inside each encoder layer.
        metadata["sequence_lengths"] = tuple(int(n) for n in np.diff(cu_seqlens))
        max_seqlen_val = (
            max_seqlen_override
            if max_seqlen_override is not None
            else int(np.diff(cu_seqlens).max())
        )
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)
        metadata["cu_seqlens"] = (
            torch.from_numpy(cu_seqlens).pin_memory().to(device, non_blocking=True)
            if device.type == "cuda"
            else torch.from_numpy(cu_seqlens).to(device)
        )

        return metadata

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
        _graph_metadata=None,
        return_dict=True,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_metadata is None:
            encoder_metadata = _graph_metadata
        hidden_states = hidden_states.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        hidden_states = self.patch_embed(hidden_states)

        if encoder_metadata is None:
            if isinstance(grid_thw, list):
                grid_thw_list = grid_thw
            else:
                grid_thw_list = grid_thw.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

        pos_embeds = encoder_metadata["pos_embeds"]
        hidden_states = hidden_states + pos_embeds
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        last_hidden_state = hidden_states.squeeze(1)
        hidden_states = self.merger(hidden_states)
        if deepstack_feature_lists:
            hidden_states = torch.cat(
                [hidden_states] + deepstack_feature_lists, dim=1
            )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        output = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state, pooler_output=hidden_states
        )
        return output if return_dict else output.to_tuple()

    def prepare_graph_metadata(self, grid_thw, hidden_states):
        metadata = self.prepare_encoder_metadata(grid_thw.cpu().tolist())
        metadata["attention_backend"] = _select_attention_backend(
            hidden_states, getattr(self.config, "vit_attention_backend", "auto")
        )
        return metadata

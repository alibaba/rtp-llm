"""Qwen4 uniform target Q/K norm, native MRoPE and FP8 cache writer.

The owner validates its uniform target plan before preparing this operation.
Preparation is read-only; run() is the explicit cache mutation boundary.
Positive lengths preserve the native writer's dummy-page writes. Inactive
warmup rows leave the cache and scales untouched. Readers reject page zero;
only this invocation's active K/V entries receive unit scales.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.main_qk_cache import _main_qk_norm_rope_cache

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@dataclass
class MainQkCacheWrite:
    qkv: torch.Tensor
    output: torch.Tensor
    positions: torch.Tensor
    norm: object
    config: object
    cache: torch.Tensor
    scales: torch.Tensor
    lengths: torch.Tensor
    table: torch.Tensor

    def run(self):
        qkv, norm = self.qkv, self.norm
        _main_qk_norm_rope_cache[(qkv.shape[0], norm.head_num + 1)](
            qkv,
            self.output,
            self.positions,
            norm.q_weight,
            norm.k_weight,
            ROWS=qkv.shape[0],
            STRIDE=qkv.stride(0),
            Q_HEADS=norm.head_num,
            KV_HEADS=1,
            EPS=norm.eps,
            BASE=float(self.config.base),
            H_SECTION=11,
            W_SECTION=10,
            LENS=self.lengths,
            TABLE=self.table,
            CACHE=self.cache,
            SF=self.scales,
            QUERY=self.lengths.shape[1],
            TW=self.table.stride(0),
            CP=65536,
            CKV=32768,
            SP=256,
            SKV=128,
            PAGES=self.cache.shape[0],
            num_warps=1,
            enable_fp_fusion=False,
        )
        return self.output


def prepare_main_qk_cache(qkv, positions, norm, config, cache, scales, plan, params):
    """Return a read-only prepared writer for the validated uniform target plan.

    Unsupported layout/math contracts fall back before any tensor mutation.
    Native cache addressing ignores padded page strides, so this candidate
    deliberately admits only dense native pages and scales. Uniform target
    lengths and max_seq_len imply zero padding offsets in the owner producer.
    """
    from rtp_llm.ops import RopeStyle

    rows = qkv.shape[0] if qkv.ndim == 2 else 0
    if not (
        getattr(torch.version, "git_version", None) == _TORCH_GIT
        and qkv.is_cuda
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and not qkv.requires_grad
        and torch.cuda.get_device_capability(qkv.device) == (12, 0)
        and 4 <= rows <= 32
        and norm.head_num in (3, 6, 12)
        and norm.kv_head_num == 1
        and norm.size_per_head == 256
        and qkv.shape[1] == (norm.head_num + 2) * 256
        and math.isfinite(norm.eps)
        and norm.eps > 0
        and all(
            weight.device == qkv.device
            and weight.shape == (256,)
            and weight.is_contiguous()
            and not weight.requires_grad
            and weight.dtype in (torch.float32, torch.bfloat16)
            for weight in (norm.q_weight, norm.k_weight)
        )
        and config.style == RopeStyle.Mrope
        and config.index_factor == 3
        and config.dim == 64
        and config.base == 10000000.0
        and config.scale == 1.0
        and config.offset == 0
        and (config.mrope_dim1, config.mrope_dim2, config.mrope_dim3) == (11, 11, 10)
        and positions is not None
        and positions.device == qkv.device
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and positions.numel() == rows * 3
        and cache is not None
        and cache.device == qkv.device
        and cache.dtype == torch.float8_e4m3fn
        and cache.is_contiguous()
        and (
            (cache.ndim == 2 and cache.shape[1] == 65536)
            or (cache.ndim == 5 and tuple(cache.shape[1:]) == (2, 1, 128, 256))
        )
        and scales is not None
        and scales.device == qkv.device
        and scales.dtype == torch.float32
        and scales.is_contiguous()
        and (
            (scales.ndim == 2 and tuple(scales.shape) == (cache.shape[0], 256))
            or (scales.ndim == 4 and tuple(scales.shape) == (cache.shape[0], 2, 1, 128))
        )
    ):
        return None
    lengths, table = plan.kv_lens, plan.block_table
    if not (
        lengths.device == qkv.device
        and lengths.dtype == torch.int32
        and lengths.ndim == 2
        and lengths.is_contiguous()
        and 1 <= lengths.shape[1] <= 4
        and lengths.numel() == rows
        and params.max_seq_len == lengths.shape[1]
        and params.input_lengths.numel() == lengths.shape[0]
        and table.device == qkv.device
        and table.dtype == torch.int32
        and table.ndim == 2
        and table.shape[0] == lengths.shape[0]
        and table.shape[1] > 0
        and table.stride(1) == 1
    ):
        return None
    output = torch.empty(
        (rows, norm.head_num * 256), device=qkv.device, dtype=qkv.dtype
    )
    return MainQkCacheWrite(
        qkv, output, positions, norm, config, cache, scales, lengths, table
    )

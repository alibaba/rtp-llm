"""Qwen4 uniform target Q/K norm, native MRoPE and FP8 cache writer.

The owner validates its uniform target plan before preparing this operation.
Preparation is read-only; run() is the explicit cache mutation boundary.
Positive lengths preserve the native writer's dummy-page writes. Inactive
warmup rows leave the cache and scales untouched. Readers reject page zero;
only this invocation's active K/V entries receive unit scales.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton

from rtp_llm.models_py.kernels.cuda.qsa.main_qk_cache import _main_qk_norm_rope_cache

from rtp_llm.models_py.utils.device_capability import sm90_pdl_enabled

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@dataclass
class MainQkCacheWrite:
    qkv: torch.Tensor
    output: torch.Tensor
    positions: torch.Tensor
    norm: object
    config: object
    cache: torch.Tensor
    scales: Optional[torch.Tensor]
    lengths: torch.Tensor
    table: torch.Tensor
    page_size: int = 128
    cache_page_stride: int = 65536

    def run(self):
        qkv, norm = self.qkv, self.norm
        page_size = self.page_size
        has_scales = self.scales is not None
        has_pos = self.positions is not None
        pdl = sm90_pdl_enabled(qkv.device)
        _main_qk_norm_rope_cache[(qkv.shape[0], norm.head_num + 1)](
            qkv,
            self.output,
            self.positions if has_pos else self.lengths,
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
            SF=self.scales if has_scales else self.output,
            QUERY=self.lengths.shape[1],
            TW=self.table.stride(0),
            CP=self.cache_page_stride,
            CKV=page_size * 256,
            SP=2 * page_size,
            SKV=page_size,
            PAGES=self.cache.shape[0],
            PAGE_SIZE=page_size,
            HAS_SCALES=has_scales,
            HAS_POS=has_pos,
            PDL=pdl,
            num_warps=1,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
        return self.output


def prepare_main_qk_cache(qkv, positions, norm, config, cache, scales, plan, params):
    """Return a read-only prepared writer for the validated uniform target plan.

    Unsupported layout/math contracts fall back before any tensor mutation.
    """
    from rtp_llm.ops import RopeStyle

    rows = qkv.shape[0] if qkv.ndim == 2 else 0
    capability = torch.cuda.get_device_capability(qkv.device) if qkv.is_cuda else None
    if capability == (12, 0):
        arch_ok = (
            getattr(torch.version, "git_version", None) == _TORCH_GIT
            and 4 <= rows <= 32
        )
    elif capability == (9, 0):
        min_rows = 4 if hasattr(params, "max_seq_len") else 1
        arch_ok = torch.version.hip is None and min_rows <= rows <= 256
    else:
        arch_ok = False
    has_pos = positions is not None and positions.numel() > 0
    if not (
        arch_ok
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and not qkv.requires_grad
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
        and (
            (
                has_pos
                and positions.device == qkv.device
                and positions.dtype == torch.int32
                and positions.is_contiguous()
                and positions.numel() == rows * 3
            )
            or (not has_pos and capability == (9, 0))
        )
        and cache is not None
        and cache.device == qkv.device
        and plan is not None
    ):
        return None

    page_size = 0
    cache_page_stride = 0
    if cache.dtype == torch.float8_e4m3fn:
        if not cache.is_contiguous():
            return None
        if cache.ndim == 2 and cache.shape[1] in (32768, 65536):
            page_size = cache.shape[1] // 512
        elif (
            cache.ndim == 5
            and cache.shape[1:3] == (2, 1)
            and cache.shape[3] in (64, 128)
            and cache.shape[4] == 256
        ):
            page_size = cache.shape[3]
        if page_size == 0 or not (
            scales is not None
            and scales.device == qkv.device
            and scales.dtype == torch.float32
            and scales.is_contiguous()
            and (
                (
                    scales.ndim == 2
                    and tuple(scales.shape) == (cache.shape[0], 2 * page_size)
                )
                or (
                    scales.ndim == 4
                    and tuple(scales.shape) == (cache.shape[0], 2, 1, page_size)
                )
            )
        ):
            return None
        cache_page_stride = 2 * page_size * 256
    elif cache.dtype == torch.bfloat16 and capability == (9, 0) and scales is None:
        if not cache.is_contiguous():
            return None
        if cache.ndim == 2 and cache.shape[1] in (32768, 65536):
            page_size = cache.shape[1] // 512
        elif (
            cache.ndim == 5
            and cache.shape[1:3] == (2, 1)
            and cache.shape[3] in (64, 128)
            and cache.shape[4] == 256
        ):
            page_size = cache.shape[3]
        if page_size == 0:
            return None
        cache_page_stride = 2 * page_size * 256
    else:
        return None

    lengths, table = plan.kv_lens, plan.block_table
    params_input_lengths = getattr(params, "input_lengths", None)
    params_seq_lengths = getattr(params, "sequence_lengths", None)
    params_ok = (
        getattr(params, "max_seq_len", None) == lengths.shape[1]
        and params_input_lengths is not None
        and params_input_lengths.numel() == lengths.shape[0]
    ) or (
        lengths.shape[1] == 1
        and params_seq_lengths is not None
        and params_seq_lengths.numel() == lengths.shape[0]
    )
    if not (
        lengths.device == qkv.device
        and lengths.dtype == torch.int32
        and lengths.ndim == 2
        and lengths.is_contiguous()
        and 1 <= lengths.shape[1] <= 4
        and lengths.numel() == rows
        and params_ok
        and table.device == qkv.device
        and table.dtype == torch.int32
        and table.ndim == 2
        and table.shape[0] == lengths.shape[0]
        and table.shape[1] > 0
        and table.stride(1) == 1
        and int(getattr(plan, "page_size", page_size)) == page_size
    ):
        return None
    output = torch.empty(
        (rows, norm.head_num * 256), device=qkv.device, dtype=qkv.dtype
    )
    return MainQkCacheWrite(
        qkv,
        output,
        positions,
        norm,
        config,
        cache,
        scales,
        lengths,
        table,
        page_size=page_size,
        cache_page_stride=cache_page_stride,
    )

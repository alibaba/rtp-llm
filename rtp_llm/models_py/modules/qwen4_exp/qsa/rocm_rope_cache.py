"""Qwen4 BF16 HND cache writer for ROCm, with live graph metadata.

The rotary-pair axis rule follows bindings/rocm/kernels/
fused_rope_kvcache_kernel.cu:get_mrope_position_dim.  Unlike the generic
ROCm attention writer, this consumer returns packed QKV for cold prefill and
writes the linear HND layout consumed by Qwen4 sparse attention.
"""

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from rtp_llm.ops import RopeStyle
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCacheDecodeOp


@triton.jit
def _write_qkv(
    X,
    OUT,
    CACHE,
    TABLE,
    CU,
    PREFIX,
    LENGTHS,
    POS,
    TOKENS: tl.constexpr,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROTARY: tl.constexpr,
    BASE: tl.constexpr,
    SCALE: tl.constexpr,
    MROPE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SECTION0: tl.constexpr,
    SECTION1: tl.constexpr,
    SECTION2: tl.constexpr,
    POS_FACTOR: tl.constexpr,
    HAS_POS: tl.constexpr,
    PREFILL: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_S0: tl.constexpr,
    TABLE_S1: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    CACHE_S0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    if PREFILL:
        row = tl.full((), 0, tl.int32)
        for batch in range(BATCH):
            end = tl.load(CU + batch + 1)
            row += (token >= end).to(tl.int32)
        start = tl.load(CU + row, row < BATCH, other=0)
        prefix = tl.load(PREFIX + row, row < BATCH, other=0)
        position = prefix + token - start
    else:
        row = token
        position = tl.load(LENGTHS + row)
    page_col = position // PAGE
    physical = tl.load(
        TABLE + row * TABLE_S0 + page_col * TABLE_S1,
        (row < TABLE_ROWS) & (page_col >= 0) & (page_col < TABLE_COLS),
        other=0,
    )
    valid = (position >= 0) & (physical > 0) & (physical < PAGES)
    value = tl.load(X + token * WIDTH + head * DIM + d, d < DIM, other=0).to(tl.float32)
    if head < HEADS + KV_HEADS:
        pair = d % (ROTARY // 2)
        mate = tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2)
        other = tl.load(X + token * WIDTH + head * DIM + mate, d < ROTARY, other=0).to(
            tl.float32
        )
        axis = tl.full((BLOCK,), 0, tl.int32)
        if MROPE:
            if INTERLEAVED:
                axis = tl.where((pair % 3 == 1) & (pair < SECTION1 * 3), 1, axis)
                axis = tl.where((pair % 3 == 2) & (pair < SECTION2 * 3), 2, axis)
            else:
                axis = tl.where(pair >= SECTION0, 1, axis)
                axis = tl.where(pair >= SECTION0 + SECTION1, 2, axis)
        if HAS_POS:
            rotary_position = tl.load(POS + token * POS_FACTOR + axis)
        else:
            rotary_position = position
        angle = (
            rotary_position.to(tl.float32)
            / SCALE
            * tl.exp(-tl.log(BASE) * (2.0 * pair / ROTARY))
        )
        rotated = value * tl.cos(angle) + tl.where(
            d < ROTARY // 2, -other, other
        ) * tl.sin(angle)
        value = tl.where(d < ROTARY, rotated, value)
    if PREFILL:
        tl.store(OUT + token * WIDTH + head * DIM + d, value, d < DIM)
    elif head < HEADS:
        tl.store(
            OUT + token * HEADS * DIM + head * DIM + d,
            tl.where(valid, value, 0),
            d < DIM,
        )
    if head >= HEADS:
        kv = (head - HEADS) // KV_HEADS
        kv_head = (head - HEADS) % KV_HEADS
        offset = (
            physical * CACHE_S0
            + ((kv * KV_HEADS + kv_head) * PAGE + position % PAGE) * DIM
            + d
        )
        tl.store(CACHE + offset, value, valid & (d < DIM))


class RocmQsaRopeCache(FusedRopeKVCacheDecodeOp):
    """BF16-only writer; metadata preparation happens outside graph capture."""

    def __init__(self, config, *, is_prefill):
        super().__init__(config)
        self.is_prefill = is_prefill
        rope = config.rope_config
        if (
            config.kv_head_num <= 0
            or config.head_num <= 0
            or config.head_num % config.kv_head_num
        ):
            raise ValueError("invalid ROCm QSA head ratio")
        if config.kernel_tokens_per_block != 128 or config.size_per_head != 256:
            raise ValueError("ROCm QSA writer requires page128 and head_dim256")
        if config.use_logn_attn or rope.style not in (RopeStyle.Base, RopeStyle.Mrope):
            raise ValueError("ROCm QSA writer supports Base/Mrope without logn")
        if (
            not config.need_rope_kv_cache
            or not rope.is_neox_style
            or rope.offset != 0
            or rope.scale != 1.0
            or rope.extrapolation_factor != 1.0
            or rope.mscale != 1.0
            or rope.factor1 != 1.0
            or rope.factor2 != 1.0
            or rope.index_factor not in (1, 3)
        ):
            raise NotImplementedError(
                "ROCm QSA writer requires canonical unscaled NeoX RoPE"
            )
        if rope.dim <= 0 or rope.dim % 2 or rope.dim > config.size_per_head:
            raise ValueError("invalid ROCm QSA rotary dimension")
        if rope.base <= 0 or rope.scale <= 0:
            raise ValueError("invalid ROCm QSA rotary base/scale")
        if rope.style == RopeStyle.Mrope and (
            rope.index_factor != 3
            or min(rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3) < 0
            or rope.mrope_dim1 + rope.mrope_dim2 + rope.mrope_dim3 != rope.dim // 2
        ):
            raise ValueError("invalid ROCm QSA mRoPE sections")

    def prepare_kv_cache_offset(self, inputs):
        table = inputs.kv_cache_kernel_block_id_device
        if (
            table is None
            or table.ndim != 2
            or table.dtype != torch.int32
            or not table.is_cuda
        ):
            raise ValueError("ROCm QSA requires device int32 block table")
        return table.clone()

    def prepare(self, inputs, *, forbid_reallocation=False):
        if inputs.is_target_verify:
            raise NotImplementedError("ROCm QSA MTP verification is not supported")
        if inputs.context_parallel_info is not None:
            raise NotImplementedError(
                "ROCm QSA context-parallel writer is not supported"
            )
        table = self.prepare_kv_cache_offset(inputs)
        positions = (
            select_prefill_position_ids(inputs)
            if self.is_prefill
            else inputs.combo_position_ids
        )
        if positions is not None:
            if (
                not positions.is_cuda
                or positions.dtype not in (torch.int32, torch.int64)
                or not positions.is_contiguous()
            ):
                raise ValueError(
                    "ROCm QSA positions must be contiguous device integers"
                )
        if self.attn_configs.rope_config.style == RopeStyle.Mrope and positions is None:
            raise ValueError("ROCm QSA mRoPE requires explicit positions")
        if self.is_prefill:
            if (
                inputs.input_lengths.device.type != "cpu"
                or inputs.prefix_lengths.device.type != "cpu"
            ):
                raise ValueError("ROCm QSA prefill requires host length mirrors")
            if (
                inputs.input_lengths.shape != (table.shape[0],)
                or inputs.prefix_lengths.shape != (table.shape[0],)
                or bool(torch.any(inputs.input_lengths <= 0))
                or bool(torch.any(inputs.prefix_lengths < 0))
            ):
                raise ValueError("invalid ROCm QSA prefill lengths")
            prefixes = inputs.prefix_lengths.to(device=table.device, dtype=torch.int32)
            lengths = inputs.sequence_lengths
        else:
            prefixes = inputs.prefix_lengths
            lengths = self.refresh_sequence_lengths(
                inputs, table.device, forbid_reallocation=forbid_reallocation
            )
        return SimpleNamespace(
            kv_cache_offset=table,
            position_ids=positions,
            sequence_lengths=lengths,
            prefix_lengths=prefixes,
            cu_seqlens=inputs.cu_seqlens_device,
            token_count=(
                int(inputs.input_lengths.sum()) if self.is_prefill else table.shape[0]
            ),
            max_prefix_length=(
                int(inputs.prefix_lengths.max()) if self.is_prefill else 0
            ),
        )

    def forward(self, qkv, kv_cache, params):
        config, rope = self.attn_configs, self.attn_configs.rope_config
        cache = kv_cache.kv_cache_base
        heads, kv_heads, dim = config.head_num, config.kv_head_num, config.size_per_head
        width = (heads + 2 * kv_heads) * dim
        if qkv.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                "ROCm QSA writer currently requires BF16 QKV and KV cache"
            )
        if qkv.ndim != 2 or qkv.shape[1] != width or not qkv.is_contiguous():
            raise ValueError("invalid packed ROCm QSA QKV geometry")
        if qkv.shape[0] <= 0 or qkv.shape[0] != params.token_count:
            raise ValueError("ROCm QSA rows do not match prepared token count")
        page_width = 2 * kv_heads * 128 * dim
        if cache.ndim == 2:
            valid_cache = cache.shape[1] >= page_width and cache.stride(1) == 1
        else:
            valid_cache = tuple(cache.shape[1:]) == (2, kv_heads, 128, dim) and tuple(
                cache.stride()[1:]
            ) == (kv_heads * 128 * dim, 128 * dim, dim, 1)
        if not valid_cache or cache.device != qkv.device:
            raise ValueError("ROCm QSA KV cache must use linear HND/HybridPool rows")
        table = params.kv_cache_offset
        if (
            not qkv.is_cuda
            or table.device != qkv.device
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.shape[0] <= 0
            or table.shape[1] <= 0
            or cache.stride(0) < page_width
        ):
            raise ValueError("ROCm QSA cache/table geometry or device mismatch")
        if not self.is_prefill and (
            qkv.shape[0] != table.shape[0]
            or params.sequence_lengths.shape != (qkv.shape[0],)
            or params.sequence_lengths.device != qkv.device
            or params.sequence_lengths.dtype != torch.int32
            or params.sequence_lengths is not self._sequence_lengths_device
        ):
            raise ValueError("ROCm QSA decode rows must match stable sequence lengths")
        if self.is_prefill and (
            params.cu_seqlens.shape != (table.shape[0] + 1,)
            or params.cu_seqlens.device != qkv.device
            or params.prefix_lengths.shape != (table.shape[0],)
            or params.prefix_lengths.device != qkv.device
        ):
            raise ValueError("ROCm QSA prefill metadata geometry mismatch")
        positions = params.position_ids
        if positions is not None and positions.device != qkv.device:
            raise ValueError("ROCm QSA positions and QKV must share a device")
        if (
            positions is not None
            and positions.numel() != qkv.shape[0] * rope.index_factor
        ):
            raise ValueError("ROCm QSA position count does not match packed tokens")
        output = (
            torch.empty_like(qkv)
            if self.is_prefill
            else torch.empty(
                (qkv.shape[0], heads * dim), device=qkv.device, dtype=qkv.dtype
            )
        )
        table = params.kv_cache_offset
        _write_qkv[(qkv.shape[0], heads + 2 * kv_heads)](
            qkv,
            output,
            cache,
            table,
            params.cu_seqlens if self.is_prefill else table,
            params.prefix_lengths if self.is_prefill else table,
            table if self.is_prefill else params.sequence_lengths,
            positions if positions is not None else table,
            TOKENS=qkv.shape[0],
            BATCH=table.shape[0],
            WIDTH=width,
            HEADS=heads,
            KV_HEADS=kv_heads,
            DIM=dim,
            ROTARY=rope.dim,
            BASE=float(rope.base),
            SCALE=float(rope.scale),
            MROPE=rope.style == RopeStyle.Mrope,
            INTERLEAVED=bool(rope.mrope_interleaved),
            SECTION0=rope.mrope_dim1,
            SECTION1=rope.mrope_dim2,
            SECTION2=rope.mrope_dim3,
            POS_FACTOR=rope.index_factor,
            HAS_POS=positions is not None,
            PREFILL=self.is_prefill,
            TABLE_ROWS=table.shape[0],
            TABLE_COLS=table.shape[1],
            TABLE_S0=table.stride(0),
            TABLE_S1=table.stride(1),
            PAGE=128,
            PAGES=cache.shape[0],
            CACHE_S0=cache.stride(0),
            BLOCK=triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )
        return output

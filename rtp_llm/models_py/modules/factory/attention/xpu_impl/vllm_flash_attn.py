"""XPU Flash Attention with RoPE and KV Cache using vllm-xpu-kernels.

Supports batched multi-request decode and prefill for continuous batching.
Uses the framework's LayerKVCache for paged KV storage. Uses flash_attn_varlen
to handle variable-length sequences in a single kernel call.
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig, RopeStyle
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

# RoPE styles unsupported by the XPU Base-frequency cache.
# Yarn/Llama3/Mrope require different freq computations; Glm2/DynamicNTK also
# use non-Base scaling.  All of these produce silently wrong scores if run
# through the plain Base cache, so both XpuVllmPrefillImpl and
# XpuVllmDecodeImpl reject them in support().
_UNSUPPORTED_ROPE_STYLES = {
    RopeStyle.Glm2,
    RopeStyle.DynamicNTK,
    RopeStyle.QwenDynamicNTK,
    RopeStyle.Yarn,
    RopeStyle.Llama3,
    RopeStyle.Mrope,
}

logger = logging.getLogger(__name__)

# Module-level lazy import for flash attention (avoids per-call import overhead)
_flash_attn_varlen = None
def _get_flash_attn_varlen():
    global _flash_attn_varlen
    if _flash_attn_varlen is None:
        from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import flash_attn_varlen
        _flash_attn_varlen = flash_attn_varlen
    return _flash_attn_varlen


def _is_fa2_available() -> bool:
    """Return True if the real FA2 kernel (vllm_fa2_C) is available.

    Must NOT test by importing flash_attn_varlen — that wrapper always
    exists and falls back to SDPA.  Instead delegate to the module-level
    flag set by importing vllm_xpu_kernels._vllm_fa2_C.
    """
    from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import is_fa2_available
    return is_fa2_available()


# ── Cached arange tensors (avoid per-layer recreation) ─────────────────
_arange_cache = {}  # (max_size, device) -> tensor

def _get_arange(size, dtype, device):
    """Return a cached arange [0..size), growing the cache as needed."""
    key = str(device)
    cached = _arange_cache.get(key)
    if cached is not None and cached.numel() >= size:
        return cached[:size].to(dtype=dtype)
    t = torch.arange(max(size, 256), dtype=torch.int32, device=device)
    _arange_cache[key] = t
    return t[:size].to(dtype=dtype)


# ── RoPE ────────────────────────────────────────────────────────────────────

_COS_SIN_CACHE: OrderedDict = OrderedDict()
_COS_SIN_CACHE_MAX_SIZE = 32


def _get_cos_sin_cache(rope_config, head_dim, max_pos, dtype, device):
    rotary_dim = getattr(rope_config, 'dim', 0) or head_dim
    base = getattr(rope_config, 'base', 10000.0) or 10000.0
    scale = getattr(rope_config, 'scale', 1.0) or 1.0
    key = (base, rotary_dim, max_pos, scale, dtype, str(device))
    if key in _COS_SIN_CACHE:
        _COS_SIN_CACHE.move_to_end(key)
        return _COS_SIN_CACHE[key]
    # Evict oldest entries if cache is full
    if len(_COS_SIN_CACHE) >= _COS_SIN_CACHE_MAX_SIZE:
        oldest_key = next(iter(_COS_SIN_CACHE))
        del _COS_SIN_CACHE[oldest_key]
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    if scale != 1.0:
        t = t / scale
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    cache = cache.to(dtype=dtype, device=device)
    _COS_SIN_CACHE[key] = cache
    return cache


def _apply_rotary_emb_neox(x, cos, sin):
    d2 = cos.shape[-1]
    x1, x2 = x[..., :d2], x[..., d2:2*d2]
    x_pass = x[..., 2*d2:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2, x_pass), dim=-1)


def _need_rope(attn_configs):
    if getattr(attn_configs, 'need_rope_kv_cache', False):
        return True
    return getattr(attn_configs.rope_config, 'style', 0) != 0


def _apply_rope(q, k, positions, rope_config, head_dim, num_heads, num_kv_heads, device, dtype, max_pos_hint=None):
    rotary_dim = getattr(rope_config, 'dim', 0) or head_dim
    is_neox = getattr(rope_config, 'is_neox_style', True)
    if max_pos_hint is not None:
        raw_max = max(max_pos_hint + 1, 4096)
    else:
        raw_max = max(int(positions.max().item()) + 1, 4096)
    # Round up to next power-of-2 to reduce unique cache entries
    max_pos = 1 << (raw_max - 1).bit_length()
    cos_sin_cache = _get_cos_sin_cache(rope_config, head_dim, max_pos, dtype, device)
    try:
        from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import rotary_embedding as vllm_rope
        vllm_rope(positions, q, k, head_dim, cos_sin_cache, is_neox)
        return q, k
    except Exception as e:
        if not getattr(_apply_rope, '_rope_fallback_warned', False):
            logger.warning('vllm RoPE kernel failed, using Python fallback (perf degraded): %s', e)
            _apply_rope._rope_fallback_warned = True
    num_tokens = q.shape[0]
    cos_sin = cos_sin_cache[positions.long()]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)
    q_r = q.view(num_tokens, num_heads, head_dim)
    k_r = k.view(num_tokens, num_kv_heads, head_dim)
    q_r = _apply_rotary_emb_neox(q_r, cos, sin)
    k_r = _apply_rotary_emb_neox(k_r, cos, sin)
    return q_r.reshape(num_tokens, -1), k_r.reshape(num_tokens, -1)


def _build_prefill_positions(attn_inputs, total_tokens, device):
    """Build per-token position IDs for a prefill batch.

    For batched prefill, concatenates ``arange(prefix_i, prefix_i + input_i)``
    so each sequence's RoPE positions start from its own prefix offset, not
    from a global ``arange(total_tokens)`` (which would alias positions across
    requests and produce wrong K).  Handles prefix-cache prefill
    (``prefix_lengths > 0``) by offsetting each request's position range.
    """
    input_lengths = getattr(attn_inputs, 'input_lengths', None)
    if input_lengths is None or input_lengths.numel() == 0:
        return torch.arange(total_tokens, dtype=torch.long, device=device)
    input_lengths_cpu = input_lengths if input_lengths.is_cpu else input_lengths.cpu()
    prefix_lengths = getattr(attn_inputs, 'prefix_lengths', None)
    if prefix_lengths is not None and not prefix_lengths.is_cpu:
        prefix_lengths = prefix_lengths.cpu()
    has_prefix = prefix_lengths is not None and prefix_lengths.numel() > 0
    # Fast path: single request, no prefix offset -> simple arange.
    if input_lengths_cpu.numel() == 1 and not (has_prefix and int(prefix_lengths[0]) > 0):
        return torch.arange(total_tokens, dtype=torch.long, device=device)
    parts = []
    for i, slen in enumerate(input_lengths_cpu.tolist()):
        offset = int(prefix_lengths[i]) if has_prefix and prefix_lengths.numel() > i else 0
        parts.append(torch.arange(offset, offset + int(slen), dtype=torch.long))
    return torch.cat(parts).to(device=device, non_blocking=True)


def _split_qkv_and_rope(qkv, attn_inputs, num_heads, num_kv_heads, head_dim, rope_config, need_rope, max_pos_hint=None):
    """Split QKV tensor and apply RoPE. Returns q, k, v as [tokens, heads, dim]."""
    total_tokens = qkv.shape[0]
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    if need_rope:
        positions = attn_inputs.position_ids
        if positions is None:
            positions = _build_prefill_positions(attn_inputs, total_tokens, qkv.device)
            # Cache on attn_inputs so subsequent layers in the same forward
            # pass reuse the same tensor (avoids N CPU->XPU transfers).
            attn_inputs.position_ids = positions
        q_flat = qkv[:, :q_size].contiguous()
        k_flat = qkv[:, q_size:q_size + kv_size].contiguous()
        q_flat, k_flat = _apply_rope(
            q_flat, k_flat, positions, rope_config, head_dim,
            num_heads, num_kv_heads, qkv.device, qkv.dtype,
            max_pos_hint=max_pos_hint,
        )
        q = q_flat.view(total_tokens, num_heads, head_dim)
        k = k_flat.view(total_tokens, num_kv_heads, head_dim)
    else:
        q = qkv[:, :q_size].view(total_tokens, num_heads, head_dim)
        k = qkv[:, q_size:q_size + kv_size].view(total_tokens, num_kv_heads, head_dim)

    v = qkv[:, q_size + kv_size:].view(total_tokens, num_kv_heads, head_dim)
    return q, k, v


# ── Paged KV cache helpers ──────────────────────────────────────────────

# Module-level LRU cache for prefill write indices. The same (bids, start_pos,
# N, tpb) recurs across all layers within one forward pass; precomputing
# device tensors once eliminates (num_layers-1) CPU->XPU transfers per request.
_PREFILL_WRITE_IDX_CACHE: OrderedDict = OrderedDict()
_PREFILL_WRITE_IDX_CACHE_MAX = 64


def _get_prefill_write_indices(bids_cpu, start_pos, N, tpb, device):
    """Return (block_indices_dev, offsets_dev, n_valid). Cached by
    (bids.data_ptr(), start_pos, N, tpb, device).
    """
    # Include content hash to avoid aliasing when tensors are reallocated
    # at the same address across steps.
    key = (bids_cpu.data_ptr(), bids_cpu.numel(), int(bids_cpu.sum()),
           int(bids_cpu[-1]) if bids_cpu.numel() > 0 else 0,
           int(start_pos), int(N), int(tpb), str(device))
    cached = _PREFILL_WRITE_IDX_CACHE.get(key)
    if cached is not None:
        _PREFILL_WRITE_IDX_CACHE.move_to_end(key)
        return cached
    abs_positions = torch.arange(start_pos, start_pos + N, dtype=torch.long)
    blk_slots = abs_positions // tpb
    offsets_cpu = abs_positions % tpb
    valid_mask = blk_slots < bids_cpu.numel()
    if not valid_mask.all():
        blk_slots = blk_slots[valid_mask]
        offsets_cpu = offsets_cpu[valid_mask]
        n_valid = int(valid_mask.sum())
    else:
        n_valid = int(N)
    block_indices_cpu = bids_cpu[blk_slots].long()
    block_indices_dev = block_indices_cpu.to(device, non_blocking=True)
    offsets_dev = offsets_cpu.to(device, non_blocking=True)
    if len(_PREFILL_WRITE_IDX_CACHE) >= _PREFILL_WRITE_IDX_CACHE_MAX:
        _PREFILL_WRITE_IDX_CACHE.popitem(last=False)
    val = (block_indices_dev, offsets_dev, n_valid)
    _PREFILL_WRITE_IDX_CACHE[key] = val
    return val


# Threshold above which scatter_ on a flat view beats index_put_; measured via
# tools/scatter_prototype.py. Tune per device if needed.
_SCATTER_N_THRESHOLD = 256


def _flat_write_kv(cache, block_indices, offsets, k, v, tpb, H, D):
    """Scatter K,V into cache[B, 2, S, H, D] via flat-view index_put_/scatter_.

    Faster than `cache[bi, 0, off, :, :] = k` because it (a) avoids the cost
    of advanced-indexing two leading dims simultaneously and (b) lets the
    XPU runtime emit a single linear scatter kernel. PD-safe: storage layout
    unchanged.
    """
    N = k.shape[0]
    cache_flat = cache.view(-1, H, D)  # [num_blocks*2*tpb, H, D]
    # Linear index: bi * (2*tpb) + role*tpb + off
    base = block_indices * (2 * tpb) + offsets
    flat_idx_k = base                # role=0
    flat_idx_v = base + tpb          # role=1
    if N >= _SCATTER_N_THRESHOLD:
        # scatter_ saturates bandwidth at large N; broadcast indices to [N,H,D]
        idx_k = flat_idx_k.view(-1, 1, 1).expand(-1, H, D)
        idx_v = flat_idx_v.view(-1, 1, 1).expand(-1, H, D)
        cache_flat.scatter_(0, idx_k, k)
        cache_flat.scatter_(0, idx_v, v)
    else:
        cache_flat.index_put_((flat_idx_k,), k)
        cache_flat.index_put_((flat_idx_v,), v)


def _assert_nshd_cache(cache, tpb, num_kv_heads, head_dim):
    """Validate the LayerKVCache tensor is the XPU NSHD layout this module assumes.

    Producer: KVCache::getLayerCache() in rtp_llm/models_py/bindings/OpDefs.h
    (XPU branch) reshapes to [num_blocks, 2, seq_size_per_block, num_kv_heads,
    head_dim]. The read/write helpers below index cache[block, k/v, seq_offset,
    head, dim], so a divergent layout would silently corrupt KV. Fail loud.
    """
    shape = tuple(cache.shape)
    if (cache.dim() != 5 or shape[1] != 2 or shape[2] != tpb
            or shape[3] != num_kv_heads or shape[4] != head_dim):
        raise RuntimeError(
            "XPU KV cache layout mismatch: expected NSHD "
            f"[num_blocks, 2, {tpb}, {num_kv_heads}, {head_dim}] but got {shape}. "
            "The producer (OpDefs.h getLayerCache XPU branch) and this consumer "
            "(vllm_flash_attn paged read/write) have diverged."
        )


def _write_to_paged_cache(k, v, kv_cache, block_ids_cpu, start_pos, num_kv_heads, head_dim):
    """Write k,v [N, kv_heads, dim] to paged LayerKVCache.

    Reuses cached device indices when the same (bids, start_pos, N) recurs
    across layers in one forward pass, then dispatches to an adaptive
    flat-view scatter (index_put_ at small N, scatter_ at large N).
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # XPU flash layout: [num_blocks, 2, tpb, kv_heads, head_dim]
    _assert_nshd_cache(cache, tpb, num_kv_heads, head_dim)
    bids = block_ids_cpu.reshape(-1)
    N = k.shape[0]
    if N == 0:
        return
    block_indices, offsets, n_valid = _get_prefill_write_indices(
        bids, start_pos, N, tpb, cache.device,
    )
    if n_valid != N:
        raise RuntimeError(
            f"_write_to_paged_cache: block table covers only {n_valid} tokens "
            f"but {N} tokens need to be written "
            f"(start_pos={start_pos}, tokens_per_block={tpb}, "
            f"num_blocks={bids.numel()}). "
            "Ensure the block table is allocated for the full sequence length."
        )
    _flat_write_kv(cache, block_indices, offsets, k, v, tpb, num_kv_heads, head_dim)


def _read_from_paged_cache(kv_cache, block_ids_cpu, total_len, num_kv_heads, head_dim):
    """Read K,V [total_len, kv_heads, dim] from paged LayerKVCache.

    Vectorized: computes block/offset mapping for all positions at once
    and gathers via advanced indexing instead of a Python while-loop.
    """
    tpb = kv_cache.seq_size_per_block
    cache = kv_cache.kv_cache_base  # XPU flash layout: [num_blocks, 2, tpb, kv_heads, head_dim]
    _assert_nshd_cache(cache, tpb, num_kv_heads, head_dim)
    bids = block_ids_cpu.reshape(-1)
    if total_len == 0:
        return cache.new_empty(0, num_kv_heads, head_dim), cache.new_empty(0, num_kv_heads, head_dim)
    # Compute block slot and offset for each position
    positions = torch.arange(total_len, dtype=torch.long)
    blk_slots = positions // tpb
    offsets = positions % tpb
    block_indices = bids[blk_slots].long().to(cache.device)
    offsets_dev = offsets.to(cache.device)
    # Gather: cache[block_indices, 0/1, offsets, :, :] -> [N, kv_heads, dim]
    k = cache[block_indices, 0, offsets_dev, :, :].contiguous()
    v = cache[block_indices, 1, offsets_dev, :, :].contiguous()
    return k, v


# ── Attention implementations ───────────────────────────────────────────────

class XpuVllmPrefillImpl(FMHAImplBase):
    """Prefill: full sequence attention, stores K/V to framework\'s LayerKVCache."""

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)
        self.fmha_params = None
        # PD disaggregation: register KV blocks with cache_store after writing
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        logger.debug("[XPU PD] XpuVllmPrefillImpl init is_prefill=%s cache_store_inputs=%s write_op=%s", attn_inputs.is_prefill, bool(attn_inputs.cache_store_inputs), self.write_cache_store_impl is not None)

    @staticmethod
    def support(attn_configs, attn_inputs):
        if not attn_inputs.is_prefill:
            return False
        # XPU attention only supports BASE (unquantized) KV cache.
        kv_dt = getattr(attn_configs, 'kv_cache_dtype', None)
        if kv_dt is not None and kv_dt != KvCacheDataType.BASE:
            return False
        # Prefix-cache (chunked prefill / reuse) is not yet implemented here:
        # the path does not load previously written K/V blocks into attention,
        # so any request with prefix_lengths > 0 would produce wrong results.
        # Decline the impl when prefix tokens are present so the framework can
        # route the request to a backend that supports it.
        pl = getattr(attn_inputs, 'prefix_lengths', None)
        if pl is not None and pl.numel() > 0:
            try:
                if int(pl.sum()) > 0:
                    return False
            except Exception:
                return False
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            return False
        return True

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        flash_attn_varlen = _get_flash_attn_varlen()
        total_tokens = qkv.shape[0]

        input_lengths = self.attn_inputs.input_lengths
        input_lengths_cpu = None
        max_pos_hint = None
        if input_lengths is not None and input_lengths.numel() > 0:
            input_lengths_cpu = input_lengths if input_lengths.is_cpu else input_lengths.cpu()
            max_pos_hint = int(input_lengths_cpu.max())

        # Per-sequence position IDs (with prefix offsets) are derived inside
        # _split_qkv_and_rope via _build_prefill_positions when position_ids
        # is None.  Single source of truth shared with XpuSdpaPrefillImpl.

        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
            max_pos_hint=max_pos_hint,
        )

        # Write K,V to paged LayerKVCache for future decode steps
        if kv_cache is not None:
            # Prefer kernel-granularity block IDs for KV cache writes.
            block_ids_all = self.attn_inputs.kv_cache_kernel_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_kernel_block_id_host
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_host
            if block_ids_all is None or block_ids_all.numel() == 0:
                raise RuntimeError(
                    "XPU prefill: kv_cache is present but no block IDs available. "
                    "Cannot write KV to paged cache without block table.")
            if block_ids_all is not None and block_ids_all.numel() > 0:
                block_ids_cpu = block_ids_all if block_ids_all.is_cpu else block_ids_all.cpu()
                if input_lengths_cpu is not None and input_lengths_cpu.numel() > 1:
                    # Batched prefill: write each request separately
                    num_reqs = input_lengths_cpu.numel()
                    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), input_lengths_cpu.cumsum(0)])
                    # block_ids_all may be [num_reqs, blocks_per_req] or [1, total_blocks]
                    # Reshape to [num_reqs, -1] if needed
                    if block_ids_cpu.dim() == 1:
                        blocks_per_req = block_ids_cpu.numel() // num_reqs
                        bids_2d = block_ids_cpu.reshape(num_reqs, blocks_per_req)
                    elif block_ids_cpu.shape[0] == num_reqs:
                        bids_2d = block_ids_cpu
                    else:
                        blocks_per_req = block_ids_cpu.numel() // num_reqs
                        bids_2d = block_ids_cpu.reshape(num_reqs, blocks_per_req)
                    for req_idx in range(num_reqs):
                        start = int(offsets[req_idx])
                        end = int(offsets[req_idx + 1])
                        bids = bids_2d[req_idx]
                        _write_to_paged_cache(
                            k[start:end], v[start:end], kv_cache, bids, 0,
                            self.num_kv_heads, self.head_dim,
                        )
                else:
                    bids = block_ids_cpu[0]
                    _write_to_paged_cache(k, v, kv_cache, bids, 0,
                                          self.num_kv_heads, self.head_dim)

            # PD disaggregation: notify cache_store the KV blocks for this request
            # are ready so the decode side can fetch them via P2P RPC.
            if self.write_cache_store_impl is not None and layer_idx <= 1:
                ai = self.attn_inputs
                def _shape(t):
                    return None if t is None else (tuple(t.shape) if hasattr(t,"shape") else "?")
                logger.debug(
                    "[XPU PD] write_cache_store layer=%s in_len=%s prefix_len=%s blkid_host=%s",
                    layer_idx,
                    _shape(ai.input_lengths),
                    _shape(ai.prefix_lengths),
                    _shape(ai.kv_cache_block_id_host),
                )
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )

        cu_seqlens_cpu = self.attn_inputs.cu_seqlens
        if cu_seqlens_cpu is None or cu_seqlens_cpu.numel() <= 1:
            cu_seqlens_cpu = torch.tensor([0, total_tokens], dtype=torch.int32)

        if input_lengths_cpu is not None and input_lengths_cpu.numel() > 0:
            max_seqlen = int(input_lengths_cpu.max())
        else:
            max_seqlen = int((cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max())

        cu_seqlens = cu_seqlens_cpu.to(device=qkv.device, dtype=torch.int32)
        output = flash_attn_varlen(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        return output.reshape(total_tokens, -1)


class XpuVllmDecodeImpl(FMHAImplBase):
    """Decode: process new token(s), read K/V from framework\'s LayerKVCache.

    Supports batched decode with multiple requests. Uses flash_attn_varlen
    with per-request cu_seqlens to handle different KV lengths.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)
        self.fmha_params = None

    @staticmethod
    def support(attn_configs, attn_inputs):
        if attn_inputs.is_prefill:
            return False
        # XPU attention only supports BASE (unquantized) KV cache.
        kv_dt = getattr(attn_configs, 'kv_cache_dtype', None)
        if kv_dt is not None and kv_dt != KvCacheDataType.BASE:
            return False
        # Requires FA2 (flash_attn_varlen); fall back to XpuSdpaDecodeImpl
        # when FA2 is not installed, rather than crashing at forward() time.
        if not _is_fa2_available():
            return False
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            return False
        return True

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        if kv_cache is not None:
            return self._paged_decode(qkv, kv_cache, layer_idx)
        # Fallback: no paged cache, self-attend over current tokens
        flash_attn_varlen = _get_flash_attn_varlen()
        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )
        N = qkv.shape[0]
        cu = torch.tensor([0, N], dtype=torch.int32, device=qkv.device)
        output = flash_attn_varlen(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                    max_seqlen_q=N, max_seqlen_k=N, causal=True)
        return output.reshape(N, -1)

    def _paged_decode(self, qkv, kv_cache, layer_idx):
        """Decode using paged LayerKVCache with block_table support.

        Optimized: uses CPU-side metadata to avoid GPU→CPU syncs, passes
        block_table directly to flash_attn_varlen without unique/remap.
        """
        flash_attn_varlen = _get_flash_attn_varlen()
        seq_lengths = self.attn_inputs.sequence_lengths

        # Note on speculative decode (q_len > 1 per request):
        # This path assumes max_seqlen_q=1 per request (normal autoregressive decode).
        # XPU speculative decode is not currently enabled, so this is safe.
        # If speculative decode is added in future, guard with:
        #   num_requests = seq_lengths.numel()
        #   if qkv.shape[0] > num_requests: raise NotImplementedError(...)
        # Do NOT use attn_inputs.input_lengths for this check —
        # input_lengths is the accumulated KV-cache sequence length
        # (tokens seen so far), not the query length for the current step.

        # --- Use CPU-side block IDs to avoid device→host sync ---
        # Prefer kernel-granularity block IDs for attention compute;
        # fall back to physical block IDs (identical when kernel_blocks_per_kv_block == 1).
        block_ids_host = self.attn_inputs.kv_cache_kernel_block_id_host
        if block_ids_host is None:
            block_ids_host = self.attn_inputs.kv_cache_block_id_host
        block_ids_device = self.attn_inputs.kv_cache_kernel_block_id_device
        if block_ids_device is None:
            block_ids_device = self.attn_inputs.kv_cache_block_id_device

        try:
            num_requests = seq_lengths.numel() if seq_lengths is not None else 0
        except (RuntimeError, AttributeError):
            num_requests = 0
        if num_requests == 0:
            num_requests = 1

        # --- Keep seq_lengths on CPU; avoid GPU→CPU sync ---
        if seq_lengths is not None:
            seq_lens_cpu = seq_lengths if seq_lengths.is_cpu else seq_lengths.cpu()
        else:
            seq_lens_cpu = torch.zeros(num_requests, dtype=torch.long)

        # Compute max position on CPU (no GPU sync) for RoPE cache sizing
        max_pos_hint = int(seq_lens_cpu.max()) if seq_lens_cpu.numel() > 0 else 0

        # Class-level step caches (shared across all layers in one decode step).
        cls = type(self)
        # Set position_ids for RoPE — CPU→device transfer (async, no sync).
        # Hoist across layers: identical for all 36 layers in a decode step.
        if self.need_rope:
            _pid_key = (
                seq_lens_cpu.data_ptr(),
                seq_lens_cpu.numel(),
                int(seq_lens_cpu.sum()),
                int(seq_lens_cpu[0]) if seq_lens_cpu.numel() > 0 else 0,
                qkv.device,
            )
            _pid_cache = getattr(cls, "_pos_ids_cache", None)
            if _pid_cache is not None and _pid_cache[0] == _pid_key:
                self.attn_inputs.position_ids = _pid_cache[1]
            else:
                _pid = seq_lens_cpu.to(dtype=torch.long, device=qkv.device, non_blocking=True)
                cls._pos_ids_cache = (_pid_key, _pid)
                self.attn_inputs.position_ids = _pid

        q_new, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
            max_pos_hint=max_pos_hint,
        )

        cache = kv_cache.kv_cache_base
        tpb = kv_cache.seq_size_per_block

        # --- Resolve block IDs on CPU without GPU sync ---
        # TODO(xpu): hybrid KV (GLA/sliding window) uses per-group block
        # tables (kv_cache_block_id varies by cache group).  Currently we use
        # a single block table for all layers.  When hybrid models are needed,
        # select the correct 2D table via kv_cache_layer_to_group[layer_idx].
        if block_ids_host is not None:
            bids_2d_cpu = block_ids_host.reshape(num_requests, -1)
        elif block_ids_device is not None:
            bids_2d_cpu = block_ids_device.reshape(num_requests, -1).cpu()
        else:
            raise RuntimeError("No block IDs available for paged decode")

        # --- Vectorized K,V writes for new decode tokens ---
        # Cache layout: [num_blocks, 2, tpb, kv_heads, head_dim] (XPU flash layout)
        kv_lens = seq_lens_cpu + 1  # CPU tensor
        n_blocks_per_req = (kv_lens + tpb - 1) // tpb  # CPU tensor
        max_blocks_needed = int(n_blocks_per_req.max())
        needed_bids = bids_2d_cpu[:, :max_blocks_needed]

        # Hoist write-indices CPU→GPU across layers (identical for all 36 layers)
        _wk = (
            seq_lens_cpu.data_ptr(),
            seq_lens_cpu.numel(),
            int(seq_lens_cpu.sum()),  # content fingerprint to detect aliasing
            int(seq_lens_cpu[0]) if seq_lens_cpu.numel() > 0 else 0,
            bids_2d_cpu.data_ptr(),
            int(bids_2d_cpu.sum()),   # content fingerprint
            cache.device,
        )
        _wc = getattr(cls, "_write_idx_cache", None)
        if _wc is not None and _wc[0] == _wk:
            bid_dev, off_dev = _wc[1], _wc[2]
        else:
            write_positions = seq_lens_cpu.long()
            blk_slots_cpu = write_positions // tpb
            offsets_cpu = (write_positions % tpb).long()
            bid_indices = torch.tensor(
                [int(bids_2d_cpu[i, int(blk_slots_cpu[i])]) for i in range(num_requests)],
                dtype=torch.long,
            )
            bid_dev = bid_indices.to(cache.device, non_blocking=True)
            off_dev = offsets_cpu.to(cache.device, non_blocking=True)
            cls._write_idx_cache = (_wk, bid_dev, off_dev)

        # Two kernel launches total (vs 2 * num_requests in the per-loop write).
        cache[bid_dev, 0, off_dev, :, :] = k_new
        cache[bid_dev, 1, off_dev, :, :] = v_new

        # --- Gather active blocks into contiguous K/V buffers ---
        # cache[:, 0] / cache[:, 1] are non-contiguous slices (stride 2 along
        # block axis). flash_attn requires contiguous K and V, so we gather
        # the active blocks via index_select. With the flash layout
        # [num_blocks, 2, tpb, H, D] the gather output is already
        # [Nb, tpb, H, D] -- no transpose needed (the legacy MHA layout
        # required an extra transpose copy).
        _fb_key = (
            needed_bids.data_ptr(),
            needed_bids.shape[0],
            needed_bids.shape[1],
            int(needed_bids.sum()),
            cache.device,
        )
        _fb_cache = getattr(cls, "_flat_bids_cache", None)
        if _fb_cache is not None and _fb_cache[0] == _fb_key:
            flat_bids = _fb_cache[1]
        else:
            flat_bids = needed_bids.reshape(-1).long().to(cache.device, non_blocking=True)
            cls._flat_bids_cache = (_fb_key, flat_bids)
        nb = flat_bids.numel()
        H = self.num_kv_heads
        D = self.head_dim
        need_size = nb * tpb * H * D
        # Persistent scratch buffers grow monotonically to avoid per-call
        # XPU allocations across N layers x M steps.
        scratch = getattr(cls, "_kv_scratch", None)
        if scratch is None or scratch[0].device != cache.device or \
                scratch[0].dtype != cache.dtype or \
                scratch[0].numel() < need_size:
            mk = lambda: torch.empty(need_size, dtype=cache.dtype, device=cache.device)
            cls._kv_scratch = (mk(), mk())
            scratch = cls._kv_scratch
        k_cache = scratch[0][:need_size].view(nb, tpb, H, D)
        v_cache = scratch[1][:need_size].view(nb, tpb, H, D)
        torch.index_select(cache[:, 0], 0, flat_bids, out=k_cache)
        torch.index_select(cache[:, 1], 0, flat_bids, out=v_cache)

        # Sequential block_table: blocks gathered in order.
        block_table = _get_arange(
            flat_bids.numel(), torch.int32, qkv.device,
        ).reshape(num_requests, max_blocks_needed)

        max_kv_len = int(kv_lens.max())  # CPU tensor, no GPU sync
        # Hoist seqused_k across layers: kv_lens identical for all layers
        # in one decode step. Saves N-1 CPU->XPU copies per step.
        _sk_key = (
            kv_lens.data_ptr(),
            kv_lens.numel(),
            int(kv_lens.sum()),
            int(kv_lens[0]) if kv_lens.numel() > 0 else 0,
            qkv.device,
        )
        _sk_cache = getattr(cls, "_seqused_k_cache", None)
        if _sk_cache is not None and _sk_cache[0] == _sk_key:
            seqused_k = _sk_cache[1]
        else:
            seqused_k = kv_lens.to(dtype=torch.int32, device=qkv.device, non_blocking=True)
            cls._seqused_k_cache = (_sk_key, seqused_k)
        cu_q = _get_arange(num_requests + 1, torch.int32, qkv.device)

        output = flash_attn_varlen(
            q_new.contiguous(),
            k_cache,
            v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=max_kv_len,
            causal=False,
            block_table=block_table,
            seqused_k=seqused_k,
        )
        return output.reshape(num_requests, -1)

# Aliases for upstream compatibility
XpuVllmFlashAttnPrefillImpl = XpuVllmPrefillImpl
XpuVllmFlashAttnDecodeImpl = XpuVllmDecodeImpl

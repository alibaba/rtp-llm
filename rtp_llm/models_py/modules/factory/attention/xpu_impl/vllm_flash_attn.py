"""XPU Flash Attention with RoPE and KV Cache using vllm-xpu-kernels.

Supports batched multi-request decode and prefill for continuous batching.
Uses the framework's LayerKVCache for paged KV storage. Uses flash_attn_varlen
to handle variable-length sequences in a single kernel call.
"""

import logging
import os
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


def reset_module_caches():
    """Release all module-level GPU tensor caches. Call on model unload."""
    _COS_SIN_CACHE.clear()
    _PREFILL_WRITE_IDX_CACHE.clear()
    _arange_cache.clear()
    # Also drop the per-stream decode KV gather scratch buffers; these are GPU
    # tensors retained on the decode impl and would otherwise leak on unload.
    XpuVllmDecodeImpl.reset_decode_scratch()

_VLLM_ROPE = None
_VLLM_ROPE_CHECKED = False

def _get_vllm_rope():
    global _VLLM_ROPE, _VLLM_ROPE_CHECKED
    if not _VLLM_ROPE_CHECKED:
        _VLLM_ROPE_CHECKED = True
        try:
            from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import rotary_embedding
            _VLLM_ROPE = rotary_embedding
        except ImportError:
            pass
    return _VLLM_ROPE




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


def _apply_rotary_emb_gptj(x, cos, sin):
    # GPT-J / non-neox style: rotate interleaved adjacent pairs
    # (x[..., 0::2], x[..., 1::2]) rather than split halves.
    d2 = cos.shape[-1]
    x_rot = x[..., :2*d2]
    x_pass = x[..., 2*d2:]
    x_rot = x_rot.reshape(*x_rot.shape[:-1], d2, 2)
    x1 = x_rot[..., 0]
    x2 = x_rot[..., 1]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    o = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat((o, x_pass), dim=-1)


def _need_rope(attn_configs):
    if getattr(attn_configs, 'need_rope_kv_cache', False):
        return True
    rope_config = getattr(attn_configs, 'rope_config', None)
    if rope_config is None:
        return False
    style = getattr(rope_config, 'style', RopeStyle.No)
    return style != RopeStyle.No


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
    vllm_rope = _get_vllm_rope()
    if vllm_rope is None:
        # Op genuinely unavailable: fall through to the Python path below.
        if not getattr(_apply_rope, '_rope_fallback_warned', False):
            logger.warning('vllm RoPE op unavailable, using Python fallback (perf degraded)')
            _apply_rope._rope_fallback_warned = True
    else:
        # The kernel writes q/k in place.  If it raises mid-write the tensors are
        # already partially rotated, so re-raise instead of re-applying RoPE on
        # corrupted data via the Python fallback.
        vllm_rope(positions, q, k, head_dim, cos_sin_cache, is_neox)
        return q, k
    num_tokens = q.shape[0]
    cos_sin = cos_sin_cache[positions.long()]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)
    q_r = q.view(num_tokens, num_heads, head_dim)
    k_r = k.view(num_tokens, num_kv_heads, head_dim)
    _rope_fn = _apply_rotary_emb_neox if is_neox else _apply_rotary_emb_gptj
    q_r = _rope_fn(q_r, cos, sin)
    k_r = _rope_fn(k_r, cos, sin)
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
    (numel, content_hash, start_pos, N, tpb, device).
    """
    # Key on a full content digest (not a weak sum+last fingerprint) so a
    # reallocated tensor at the same address, or two block tables that merely
    # share sum/last, can never alias to a stale cached index set.
    bids_contig = bids_cpu.contiguous()
    content_hash = hash(bids_contig.numpy().tobytes())
    key = (bids_cpu.numel(), content_hash,
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
    # Guard against a short block table: reading total_len positions needs
    # ceil(total_len / tpb) block ids. Without this check an undersized table
    # would index out of bounds and silently gather garbage / wrong-request KV.
    blocks_needed = (total_len + tpb - 1) // tpb
    if blocks_needed > bids.numel():
        raise RuntimeError(
            f"paged KV read out of range: need {blocks_needed} blocks for "
            f"total_len={total_len} (tpb={tpb}) but block table has only "
            f"{bids.numel()} entries."
        )
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
        self.is_causal = getattr(attn_configs, "is_causal", True)
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
            raise NotImplementedError(
                f"XPU prefill attention does not support quantized KV cache "
                f"(got {kv_dt}). Use a non-quantized KV cache.")
        # Prefix-cache (chunked prefill / reuse) is not yet implemented:
        # the path does not load previously written K/V blocks into attention,
        # so any request with prefix_lengths > 0 would produce wrong results.
        pl = getattr(attn_inputs, 'prefix_lengths', None)
        if pl is not None and pl.numel() > 0:
            try:
                if int(pl.sum()) > 0:
                    raise NotImplementedError(
                        "XPU prefill attention does not support prefix cache "
                        "(chunked prefill / KV reuse). Disable prefix cache "
                        "for XPU.")
            except NotImplementedError:
                raise
            except Exception:
                raise NotImplementedError(
                    "XPU prefill attention does not support prefix cache "
                    "(chunked prefill / KV reuse). Disable prefix cache "
                    "for XPU.")
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            raise NotImplementedError(
                f"XPU prefill attention does not support RoPE style "
                f"{rope_style}. Supported styles exclude: "
                f"{_UNSUPPORTED_ROPE_STYLES}.")
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
        # is None.  Single source of truth for KV-block id resolution.

        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
            max_pos_hint=max_pos_hint,
        )

        # Write K,V to paged LayerKVCache for future decode steps
        if kv_cache is not None:
            # Prefer host block IDs to avoid device->host sync in the write path.
            block_ids_all = self.attn_inputs.kv_cache_kernel_block_id
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_kernel_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_device
            if block_ids_all is None or block_ids_all.numel() == 0:
                raise RuntimeError(
                    "XPU prefill: kv_cache is present but no block IDs available. "
                    "Cannot write KV to paged cache without block table.")
            block_ids_cpu = block_ids_all if block_ids_all.is_cpu else block_ids_all.cpu()
            if input_lengths_cpu is not None and input_lengths_cpu.numel() > 1:
                # Batched prefill: write each request separately
                num_reqs = input_lengths_cpu.numel()
                offsets = torch.cat([torch.zeros(1, dtype=torch.int32), input_lengths_cpu.cumsum(0)])
                # block_ids_all may be [num_reqs, blocks_per_req] or [1, total_blocks]
                # Reshape to [num_reqs, -1] if needed
                if block_ids_cpu.dim() == 1:
                    blocks_per_req, _rem = divmod(block_ids_cpu.numel(), num_reqs)
                    if _rem != 0:
                        raise RuntimeError(
                            f"XPU batched prefill: block_ids ({block_ids_cpu.numel()}) not evenly "
                            f"divisible by num_reqs ({num_reqs}). Cannot reshape block table.")
                    bids_2d = block_ids_cpu.reshape(num_reqs, blocks_per_req)
                elif block_ids_cpu.shape[0] == num_reqs:
                    bids_2d = block_ids_cpu
                else:
                    blocks_per_req, _rem = divmod(block_ids_cpu.numel(), num_reqs)
                    if _rem != 0:
                        raise RuntimeError(
                            f"XPU batched prefill: block_ids ({block_ids_cpu.numel()}) not evenly "
                            f"divisible by num_reqs ({num_reqs}). Cannot reshape block table.")
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
                    _shape(ai.kv_cache_block_id),
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
            causal=self.is_causal,
        )
        return output.reshape(total_tokens, -1)


class XpuVllmDecodeImpl(FMHAImplBase):
    """Decode: process new token(s), read K/V from framework\'s LayerKVCache.

    Supports batched decode with multiple requests. Uses flash_attn_varlen
    with per-request cu_seqlens to handle different KV lengths.
    """

    # --- Decode KV gather scratch: capacity cap + release policy ---
    # The gather scratch is retained per (device, stream) to avoid per-layer
    # allocations.  To bound steady-state memory: never retain a buffer larger
    # than _SCRATCH_RETAIN_MAX_BYTES (a long-context request above the cap uses
    # a transient buffer freed after the step), and keep at most
    # _SCRATCH_MAX_STREAMS retained entries.  reset_decode_scratch() drops all.
    _SCRATCH_RETAIN_MAX_BYTES = int(
        os.environ.get("XPU_DECODE_SCRATCH_MAX_MB", "512")) * 1024 * 1024
    _SCRATCH_MAX_STREAMS = 8

    @classmethod
    def reset_decode_scratch(cls):
        """Drop all retained decode KV scratch buffers (release policy hook)."""
        cls._kv_scratch_by_stream = {}
        cls._flat_bids_cache = None
        cls._write_idx_cache = None
        cls._pos_ids_cache = None
        cls._seqused_k_cache = None
        cls._last_layer_idx = None

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
            raise NotImplementedError(
                f"XPU decode attention does not support quantized KV cache "
                f"(got {kv_dt}). Use a non-quantized KV cache.")
        # Requires FA2 (flash_attn_varlen).  When FA2 is not installed
        # no XPU decode impl supports this config.
        if not _is_fa2_available():
            raise NotImplementedError(
                "XPU decode attention requires flash_attn (FA2) but it is "
                "not installed. Install vllm-xpu-kernels with FA2 support.")
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            raise NotImplementedError(
                f"XPU decode attention does not support RoPE style "
                f"{rope_style}. Supported styles exclude: "
                f"{_UNSUPPORTED_ROPE_STYLES}.")
        return True

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        if kv_cache is not None:
            return self._paged_decode(qkv, kv_cache, layer_idx)
        # No paged cache: a decode step needs the KV history, and RoPE here would
        # use default arange(N) positions (wrong for decode, where each token's
        # position is its absolute sequence index).  Fail fast rather than emit
        # silently-wrong output; the no-RoPE case can still self-attend.
        if self.need_rope:
            raise NotImplementedError(
                "XpuVllmDecodeImpl requires a paged kv_cache for RoPE decode; "
                "the no-kv-cache fallback would use incorrect RoPE positions.")
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
        if seq_lengths is None:
            raise RuntimeError("XPU paged decode requires sequence_lengths")

        # Guard against speculative/multi-token decode (q_len > 1 per request).
        # This path assumes max_seqlen_q=1 (normal autoregressive decode).
        _num_req = seq_lengths.numel()
        if qkv.shape[0] != _num_req:
            raise NotImplementedError(
                f"XPU paged decode does not support multi-token query "
                f"(got {qkv.shape[0]} tokens for {_num_req} requests). "
                f"Speculative decode is not supported on XPU."
            )

        # --- Use CPU-side block IDs to avoid device→host sync ---
        # Prefer kernel-granularity block IDs for attention compute;
        # fall back to physical block IDs (identical when kernel_blocks_per_kv_block == 1).
        block_ids_host = self.attn_inputs.kv_cache_kernel_block_id
        if block_ids_host is None:
            block_ids_host = self.attn_inputs.kv_cache_block_id
        block_ids_device = self.attn_inputs.kv_cache_kernel_block_id_device
        if block_ids_device is None:
            block_ids_device = self.attn_inputs.kv_cache_block_id_device

        num_requests = seq_lengths.numel()

        # --- Keep seq_lengths on CPU; avoid GPU→CPU sync ---
        seq_lens_cpu = seq_lengths if seq_lengths.is_cpu else seq_lengths.cpu()

        # Compute max position on CPU (no GPU sync) for RoPE cache sizing
        max_pos_hint = int(seq_lens_cpu.max()) if seq_lens_cpu.numel() > 0 else 0

        # Class-level step caches (shared across all layers in one decode step).
        cls = type(self)
        # Step counter: increments exactly once per decode step (not per layer).
        # Derive the step boundary from the authoritative layer_idx sequence
        # (each decode step calls layers 0, 1, ..., num_layers-1 in order)
        # rather than from a seq_lens content fingerprint.  When layer_idx wraps
        # back (current <= last seen) a new decode step has begun, so _step_id is
        # bumped.  This guarantees every step gets a unique, monotonically
        # increasing id, so the per-step device tensors cached below
        # (position_ids / write indices / flat_bids / seqused_k) can never be
        # reused across two different batch shapes that merely share the same
        # seq_lens fingerprint.
        _last_layer = getattr(cls, "_last_layer_idx", None)
        _is_step_start = (_last_layer is None or layer_idx <= _last_layer)
        if _is_step_start:
            cls._step_id = getattr(cls, "_step_id", 0) + 1
        cls._last_layer_idx = layer_idx
        _sid = cls._step_id
        # Content-address every per-step device cache so concurrent decode on
        # different XPU streams (or a _step_id collision under interleaving)
        # cannot produce a false hit: keys carry the current stream plus a full
        # content fingerprint of seq_lens, making each cached value a pure
        # function of its inputs (a stale key is a correct miss, never a wrong
        # reuse).
        try:
            _stream_key = torch.xpu.current_stream(qkv.device)
        except Exception:
            _stream_key = None
        if _is_step_start:
            cls._step_seq_fp = hash(seq_lens_cpu.contiguous().numpy().tobytes())
        _seq_fp = cls._step_seq_fp
        # Set position_ids for RoPE — CPU→device transfer (async, no sync).
        # Hoist across layers: identical for all 36 layers in a decode step.
        if self.need_rope:
            _pid_key = (
                _sid,
                _stream_key,
                _seq_fp,
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
        _assert_nshd_cache(cache, tpb, self.num_kv_heads, self.head_dim)

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

        # Block-table content fingerprint. Within one decode step all layers of
        # a homogeneous model share the same table, so hashing once at step
        # start and reusing it lets the per-step device caches below hit across
        # all layers.
        #
        # TODO(xpu): make this hash per-layer/per-group before enabling hybrid
        # models. The hash is computed ONLY at _is_step_start (layer wrap-around)
        # and reused for every later layer. A hybrid model (GLA/sliding-window)
        # whose layers map to different cache groups has per-group block tables,
        # but later layers would still key off layer-0's STALE hash -> the
        # write_idx_cache / flat_bids_cache would falsely HIT and use the wrong
        # block indices. Safe today only because XPU has no hybrid-model support
        # (single block table for all layers, see the bids_2d_cpu TODO above).
        if _is_step_start:
            cls._step_table_hash = hash(needed_bids.contiguous().numpy().tobytes())
        _table_hash = cls._step_table_hash

        # Hoist write-indices CPU→GPU across layers (identical for all 36 layers)
        _wk = (_sid, _stream_key, _table_hash, _seq_fp, cache.device)
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
        # TODO(xpu perf, tracked): this gather copies the full active KV history
        # every layer x every decode step.  The decode FA2 path already accepts
        # a block_table, so the only thing forcing the copy is the interleaved
        # [num_blocks, 2, tpb, H, D] layout (cache[:, 0]/cache[:, 1] are strided
        # and FA2 needs contiguous paged K/V).  Splitting the cache to
        # [2, num_blocks, tpb, H, D] makes cache[0]/cache[1] contiguous paged
        # tensors, letting us pass them + the real `needed_bids` block_table
        # directly and delete this gather + scratch entirely.  Deferred: it is a
        # KV-layout migration that must move in lockstep with the C++ cache
        # allocator, the prefill write path, and the PD cache-store transfer,
        # and requires full PD accuracy + perf re-validation.
        # Cache key includes the block-table fingerprint so a hybrid model
        # whose layers belong to different cache groups cannot reuse another
        # group's flat_bids within the same step. _sid guards cross-step reuse;
        # for homogeneous models every layer shares one table -> still a hit.
        _fb_key = (_sid, _stream_key, _table_hash, cache.device)
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
        # Scratch is bounded by total KV cache capacity; assert to make bound explicit.
        max_allowed = cache.shape[0] * tpb * H * D
        if need_size > max_allowed:
            raise RuntimeError(
                f"KV scratch {need_size} exceeds cache capacity {max_allowed}")
        # Persistent scratch buffers grow monotonically to avoid per-call
        # XPU allocations across N layers x M steps.  Scratch is keyed by
        # (device, current stream): kernels submitted to one XPU stream are
        # serialized, so a buffer is safe to reuse across the N layers of a
        # single decode step on that stream.  Keying by stream isolates
        # concurrent decode batches running on different streams instead of
        # sharing one class-level buffer.
        scratch_map = getattr(cls, "_kv_scratch_by_stream", None)
        if scratch_map is None:
            scratch_map = {}
            cls._kv_scratch_by_stream = scratch_map
        key = (cache.device, _stream_key)
        mk = lambda: torch.empty(need_size, dtype=cache.dtype, device=cache.device)
        # A retained entry holds TWO buffers (K and V), so the memory actually
        # retained per key is 2 * need_bytes. Compare that total against the cap
        # so XPU_DECODE_SCRATCH_MAX_MB reflects real retained bytes, not half.
        need_bytes = 2 * need_size * cache.element_size()
        if need_bytes > cls._SCRATCH_RETAIN_MAX_BYTES:
            # Above the retention cap: use transient buffers (freed after this
            # step) so a single long-context request cannot permanently inflate
            # the retained scratch.  Drop any retained entry for this key.
            scratch_map.pop(key, None)
            scratch = (mk(), mk())
        else:
            scratch = scratch_map.get(key)
            if scratch is None or scratch[0].dtype != cache.dtype or \
                    scratch[0].numel() < need_size:
                # Bound the number of retained per-stream buffers (evict oldest).
                if key not in scratch_map and len(scratch_map) >= cls._SCRATCH_MAX_STREAMS:
                    scratch_map.pop(next(iter(scratch_map)))
                scratch = (mk(), mk())
                scratch_map[key] = scratch
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
            _sid,
            _stream_key,
            _table_hash,
            kv_lens.numel(),
            _seq_fp,
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

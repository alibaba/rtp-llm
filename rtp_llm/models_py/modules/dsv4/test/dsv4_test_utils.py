"""Shared helpers for DSV4 official-vs-RTP unit tests."""


import math
import os
import sys
import types

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock kernel module before importing official model (needs tilelang which isn't installed)
_kernel = types.ModuleType("kernel")


def _mock_act_quant(x, *args, **kwargs):
    """act_quant: when called with inplace=True (5th arg), modifies x in-place and returns None.
    When called for linear dispatch (3 args), returns (x, fake_scale)."""
    if len(args) >= 4 and args[3] is True:
        return None  # in-place mode
    # Return (x_fp8, scale) for fp8_gemm dispatch - just pass through as bf16
    return x, torch.ones(1)


def _mock_fp4_act_quant(x, *args, **kwargs):
    """fp4_act_quant: in-place quantization simulation, no-op."""
    return None


def _mock_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """sparse_attn: simplified attention using topk indices into kv cache.
    Returns output with same shape as q, contiguous."""
    bsz, seqlen, n_heads, head_dim = q.shape
    # Simple scaled dot-product attention over all kv (ignoring topk for mock)
    k = kv.unsqueeze(2).expand(-1, -1, n_heads, -1)  # [B, S_kv, H, D]
    scores = torch.einsum("bshd,bthd->bsht", q, k) * softmax_scale
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    v = k  # kv is combined, use same as v for mock
    out = torch.einsum("bsht,bthd->bshd", attn, v)
    return out.contiguous()


def _mock_rotate_activation(x):
    """rotate_activation: Hadamard rotation mock - just scale."""
    return x * (x.size(-1) ** -0.5)


def _mock_hc_split_sinkhorn(*args, **kwargs):
    return None


def _mock_fp8_gemm(x, s, weight, weight_scale, scale_dtype):
    return F.linear(x.float(), weight.float()).to(x.dtype)


def _mock_fp4_gemm(x, s, weight, weight_scale, scale_dtype):
    return F.linear(x.float(), weight.float()).to(x.dtype)


_kernel.act_quant = _mock_act_quant
_kernel.fp4_act_quant = _mock_fp4_act_quant
_kernel.sparse_attn = _mock_sparse_attn
_kernel.rotate_activation = _mock_rotate_activation
_kernel.hc_split_sinkhorn = _mock_hc_split_sinkhorn
_kernel.fp8_gemm = _mock_fp8_gemm
_kernel.fp4_gemm = _mock_fp4_gemm
sys.modules["kernel"] = _kernel

# Add official model path.  CI/dev boxes can override this; otherwise keep the
# original author path and allow PYTHONPATH to provide ``model.py``.
OFFICIAL_DIR = os.environ.get(
    "DSV4_OFFICIAL_DIR", "/home/tanboyu.tby/cuda_study/models/DeepSeek/official"
)
if OFFICIAL_DIR:
    sys.path.insert(0, OFFICIAL_DIR)

# Set official model's world_size before importing
import model as _official_model

_official_model.world_size = 1

# Import official implementation
from model import Attention as OfficialAttention
from model import Compressor as OfficialCompressor
from model import Indexer as OfficialIndexer
from model import ModelArgs as OfficialArgs

# Monkey-patch module-level functions that need CUDA (defined in model.py, not kernel)
_official_model.rotate_activation = _mock_rotate_activation

import rtp_llm.models_py.modules.dsv4.attention as _our_attention_module
from rtp_llm.models_py.modules.dsv4.attention import Attention as OurAttention
from rtp_llm.models_py.modules.dsv4.attention import _get_window_topk_idxs, _sparse_attn

# Import our implementation
from rtp_llm.models_py.modules.dsv4.compressor import Compressor as OurCompressor
from rtp_llm.models_py.modules.dsv4.indexer import Indexer as OurIndexer
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis

# ============================================================
# Helpers
# ============================================================

COMPRESSOR_STATE_ATOL = 5e-3


KV_TOKENS_PER_BLOCK = 256
CSA_COMPRESS_RATIO = 4
HCA_COMPRESS_RATIO = 128


def _kv_cache_size(max_seq_len: int, compress_ratio: int) -> int:
    """Logical compressed-KV cache length T = max_seq_len // compress_ratio."""
    return max_seq_len // compress_ratio


def make_small_args():
    """Small model args for fast testing."""
    return dict(
        dim=256,
        head_dim=64,
        rope_head_dim=16,
        n_heads=4,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        window_size=16,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        norm_eps=1e-6,
        max_batch_size=2,
        max_seq_len=256,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=0,
    )


def make_official_args(
    dim=256,
    head_dim=64,
    rope_head_dim=16,
    n_heads=4,
    q_lora_rank=64,
    o_lora_rank=32,
    o_groups=2,
    window_size=16,
    max_batch_size=2,
    max_seq_len=256,
    index_n_heads=4,
    index_head_dim=32,
    index_topk=8,
    compress_ratios=(4, 128),
):
    args = OfficialArgs(vocab_size=1000, dim=dim)
    args.head_dim = head_dim
    args.rope_head_dim = rope_head_dim
    args.n_heads = n_heads
    args.q_lora_rank = q_lora_rank
    args.o_lora_rank = o_lora_rank
    args.o_groups = o_groups
    args.window_size = window_size
    args.max_batch_size = max_batch_size
    args.max_seq_len = max_seq_len
    args.index_n_heads = index_n_heads
    args.index_head_dim = index_head_dim
    args.index_topk = index_topk
    args.compress_ratios = compress_ratios
    args.norm_eps = 1e-6
    args.compress_rope_theta = 160000.0
    args.rope_theta = 10000.0
    args.rope_factor = 1.0
    args.beta_fast = 32
    args.beta_slow = 1
    args.original_seq_len = 0
    return args


def _test_device():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("DSV4 compressor tests require CUDA")
    return torch.device("cuda")


def _init_official_compressor_weights(official: OfficialCompressor):
    """Initialize official compressor weights deterministically for comparison."""
    with torch.no_grad():
        # Initialize all official weights (torch.empty may contain NaN/garbage)
        nn.init.normal_(official.wkv.weight, std=0.02)
        nn.init.normal_(official.wgate.weight, std=0.02)
        nn.init.normal_(official.ape, std=0.02)
        nn.init.normal_(official.norm.weight, std=0.02)


def _our_compressor_weights(official: OfficialCompressor, device: torch.device):
    """Build the production-style weight dict required by OurCompressor."""
    return {
        "ape": official.ape.detach().clone().to(device=device),
        "wkv": official.wkv.weight.detach().clone().to(device=device),
        "wgate": official.wgate.weight.detach().clone().to(device=device),
        # rtp_llm_ops.rmsnorm expects a bf16 weight tensor.
        "norm": official.norm.weight.detach()
        .clone()
        .to(device=device, dtype=torch.bfloat16),
    }


def sync_compressor_weights(our: OurCompressor, official: OfficialCompressor):
    """Copy initialized official compressor weights into current OurCompressor."""
    _init_official_compressor_weights(official)
    device = (
        our.ape.device
        if isinstance(getattr(our, "ape", None), torch.Tensor)
        else official.ape.device
    )
    weights = _our_compressor_weights(official, device)
    with torch.no_grad():
        if hasattr(our.wkv, "weight"):
            our.wkv.weight.copy_(weights["wkv"].to(our.wkv.weight.dtype))
            our.wgate.weight.copy_(weights["wgate"].to(our.wgate.weight.dtype))
        else:
            our.wkv.copy_(weights["wkv"].to(dtype=our.wkv.dtype))
            our.wgate.copy_(weights["wgate"].to(dtype=our.wgate.dtype))
        our.ape.copy_(weights["ape"].to(dtype=our.ape.dtype))
        our.norm.weight.copy_(weights["norm"].to(dtype=our.norm.weight.dtype))


def _fp8_weight(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    return weight.detach().clone().to(device=device).float().to(torch.float8_e4m3fn)


def _fp8_int32_scale_ones(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    n, k = weight.shape
    return torch.full(
        (n, max(1, math.ceil(k / 512))),
        0x7F7F7F7F,
        dtype=torch.int32,
        device=device,
    )


def _ue8m0_scale_ones(shape, device: torch.device) -> torch.Tensor:
    scale = torch.empty(shape, dtype=torch.float8_e8m0fnu, device=device)
    scale.view(torch.uint8).fill_(127)
    return scale


def _bf16_linear_from_weight(weight: torch.Tensor, device: torch.device) -> nn.Linear:
    out_features, in_features = weight.shape
    layer = nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.weight.copy_(weight.detach().to(device=device, dtype=torch.bfloat16))
    return layer


def _init_official_indexer_weights(official: OfficialIndexer):
    with torch.no_grad():
        nn.init.normal_(official.wq_b.weight, std=0.02)
        nn.init.normal_(official.weights_proj.weight, std=0.02)
    _init_official_compressor_weights(official.compressor)


def _indexer_layer_weights(official: OfficialIndexer, device: torch.device):
    from rtp_llm.utils.model_weight import W

    wq_b = official.wq_b.weight.detach()
    return {
        W.v4_indexer_wq_b_w: _fp8_weight(wq_b, device),
        W.v4_indexer_wq_b_s: _fp8_int32_scale_ones(wq_b, device),
        W.v4_indexer_weights_proj_w: official.weights_proj.weight.detach()
        .clone()
        .to(device=device, dtype=torch.bfloat16),
        W.v4_indexer_compressor_ape: official.compressor.ape.detach()
        .clone()
        .to(device=device),
        W.v4_indexer_compressor_wkv: official.compressor.wkv.weight.detach()
        .clone()
        .to(device=device),
        W.v4_indexer_compressor_wgate: official.compressor.wgate.weight.detach()
        .clone()
        .to(device=device),
        W.v4_indexer_compressor_norm: official.compressor.norm.weight.detach()
        .clone()
        .to(device=device, dtype=torch.bfloat16),
    }


def _attention_layer_weights(official: OfficialAttention, device: torch.device):
    from rtp_llm.utils.model_weight import W

    weights = {}
    for w_key, s_key, module in (
        (W.v4_attn_wq_a_w, W.v4_attn_wq_a_s, official.wq_a),
        (W.v4_attn_wq_b_w, W.v4_attn_wq_b_s, official.wq_b),
        (W.v4_attn_wkv_w, W.v4_attn_wkv_s, official.wkv),
        (W.v4_attn_wo_b_w, W.v4_attn_wo_b_s, official.wo_b),
    ):
        weight = module.weight.detach()
        weights[w_key] = _fp8_weight(weight, device)
        weights[s_key] = _fp8_int32_scale_ones(weight, device)

    wo_a = official.wo_a.weight.detach()
    weights[W.v4_attn_wo_a_w] = _fp8_weight(wo_a, device)
    # ``_prepare_wo_a_stacked`` consumes the raw [G * R / 128, K / 128]
    # UE8M0 layout.  The attention unit test uses R=128 so this is one row
    # block per output group.
    g = int(official.n_groups)
    r = int(official.o_lora_rank)
    k = int(wo_a.shape[1])
    weights[W.v4_attn_wo_a_s] = _ue8m0_scale_ones(
        (g * max(1, r // 128), max(1, k // 128)), device
    )

    weights[W.v4_attn_q_norm] = official.q_norm.weight.detach().clone().to(
        device=device, dtype=torch.bfloat16
    )
    weights[W.v4_attn_kv_norm] = official.kv_norm.weight.detach().clone().to(
        device=device, dtype=torch.bfloat16
    )
    weights[W.v4_attn_sink] = official.attn_sink.detach().clone().to(device=device)

    if getattr(official, "compress_ratio", 0):
        weights.update(
            {
                W.v4_compressor_ape: official.compressor.ape.detach()
                .clone()
                .to(device=device),
                W.v4_compressor_wkv: official.compressor.wkv.weight.detach()
                .clone()
                .to(device=device),
                W.v4_compressor_wgate: official.compressor.wgate.weight.detach()
                .clone()
                .to(device=device),
                W.v4_compressor_norm: official.compressor.norm.weight.detach()
                .clone()
                .to(device=device, dtype=torch.bfloat16),
            }
        )
        if getattr(official, "indexer", None) is not None:
            weights.update(_indexer_layer_weights(official.indexer, device))
    return weights


def _make_our_compressor(
    official: OfficialCompressor,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    max_batch_size: int,
    kv_cache_size: int,
    device: torch.device,
):
    _init_official_compressor_weights(official)
    comp = OurCompressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        max_batch_size=max_batch_size,
        norm_eps=1e-6,
        compressor_weights=_our_compressor_weights(official, device),
    )
    comp.configure_kv_cache_shape(kv_cache_size)
    return comp


def _bind_standalone_pools(
    comp: OurCompressor,
    bsz: int,
    kv_cache_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    sequence_lengths=None,
):
    """Attach minimal pools so pool-only Compressor.forward persists state.

    KV cache pool uses a fixed token-based block size: ``KV_TOKENS_PER_BLOCK``
    original tokens per block. This maps to
    ``kv_eb = KV_TOKENS_PER_BLOCK // compress_ratio`` compressed entries per block.
    ``sequence_lengths`` controls the per-request token-block count; state
    pools keep only the last two token blocks, mirroring fixed tail groups.
    """
    if KV_TOKENS_PER_BLOCK % comp.compress_ratio != 0:
        raise ValueError(
            f"KV_TOKENS_PER_BLOCK={KV_TOKENS_PER_BLOCK} must be divisible by "
            f"compress_ratio={comp.compress_ratio}"
        )
    if sequence_lengths is None:
        seq_lens = [int(kv_cache_size) * int(comp.compress_ratio)] * bsz
    elif isinstance(sequence_lengths, torch.Tensor):
        seq_tensor = sequence_lengths.detach().cpu().reshape(-1)
        seq_lens = [int(v) for v in seq_tensor.tolist()]
    elif isinstance(sequence_lengths, int):
        seq_lens = [int(sequence_lengths)] * bsz
    else:
        seq_lens = [int(v) for v in sequence_lengths]
    if len(seq_lens) == 1 and bsz > 1:
        seq_lens = seq_lens * bsz
    if len(seq_lens) != bsz:
        raise ValueError(f"sequence_lengths must have 1 or {bsz} values")

    kv_eb = KV_TOKENS_PER_BLOCK // comp.compress_ratio
    n_kv_blocks_per_batch = [
        max(1, int(math.ceil(max(seq_len, 0) / KV_TOKENS_PER_BLOCK)))
        for seq_len in seq_lens
    ]
    n_kv_blocks = max(n_kv_blocks_per_batch)
    kv_block_table = torch.full((bsz, n_kv_blocks), -1, device=device, dtype=torch.long)
    next_kv_block_id = 1
    for b, block_count in enumerate(n_kv_blocks_per_batch):
        block_ids = torch.arange(
            next_kv_block_id,
            next_kv_block_id + block_count,
            device=device,
            dtype=torch.long,
        )
        kv_block_table[b, :block_count] = block_ids
        next_kv_block_id += block_count
    kv_pool = torch.zeros(
        next_kv_block_id * kv_eb, comp.head_dim, dtype=dtype, device=device
    )

    state_eb = comp._state_rows
    state_block_table = torch.full(
        (bsz, n_kv_blocks), -1, device=device, dtype=torch.long
    )
    next_state_block_id = 1
    for b, block_count in enumerate(n_kv_blocks_per_batch):
        first_tail_block = max(0, block_count - 2)
        tail_count = block_count - first_tail_block
        block_ids = torch.arange(
            next_state_block_id,
            next_state_block_id + tail_count,
            device=device,
            dtype=torch.long,
        )
        state_block_table[b, first_tail_block:block_count] = block_ids
        next_state_block_id += tail_count
    state_pool = torch.zeros(
        next_state_block_id * state_eb,
        2 * comp._state_dim,
        dtype=torch.float32,
        device=device,
    )
    comp.set_pool_context(
        kv_pool, kv_block_table, kv_eb, state_pool, state_block_table, state_eb
    )
    return types.SimpleNamespace(
        kv_pool=kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        n_kv_blocks=n_kv_blocks,
        n_kv_blocks_per_batch=n_kv_blocks_per_batch,
        state_pool=state_pool,
        state_block_table=state_block_table,
        state_eb=state_eb,
        sequence_lengths=seq_lens,
        bsz=bsz,
        device=device,
    )


def _pool_kv_cache(ctx, comp: OurCompressor, batch: int = None):
    if batch is None:
        return torch.stack(
            [_pool_kv_cache(ctx, comp, b) for b in range(ctx.bsz)], dim=0
        )
    block_ids = ctx.kv_block_table[batch]
    blocks = []
    for bid in block_ids.tolist():
        if bid <= 0:
            continue
        start = bid * ctx.kv_eb
        blocks.append(ctx.kv_pool[start : start + ctx.kv_eb])
    if not blocks:
        return torch.zeros(
            comp._kv_cache_t,
            comp.head_dim,
            dtype=ctx.kv_pool.dtype,
            device=ctx.kv_pool.device,
        )
    dense = torch.cat(blocks, dim=0)
    if dense.shape[0] < comp._kv_cache_t:
        pad = torch.zeros(
            comp._kv_cache_t - dense.shape[0],
            comp.head_dim,
            dtype=dense.dtype,
            device=dense.device,
        )
        dense = torch.cat([dense, pad], dim=0)
    return dense[: comp._kv_cache_t]


def _pool_state(ctx, comp: OurCompressor, batch: int = None):
    if batch is None:
        states = [_pool_state(ctx, comp, b) for b in range(ctx.bsz)]
        kv_states = torch.stack([s[0] for s in states], dim=0)
        score_states = torch.stack([s[1] for s in states], dim=0)
        return kv_states, score_states
    valid_block_ids = ctx.state_block_table[batch][ctx.state_block_table[batch] > 0]
    if valid_block_ids.numel() == 0:
        rows = torch.zeros(
            comp._state_rows,
            2 * comp._state_dim,
            dtype=ctx.state_pool.dtype,
            device=ctx.state_pool.device,
        )
        rows[:, comp._state_dim :] = float("-inf")
        return rows[:, : comp._state_dim], rows[:, comp._state_dim :]
    start = int(valid_block_ids[-1].item()) * ctx.state_eb
    rows = ctx.state_pool[start : start + comp._state_rows]
    return rows[:, : comp._state_dim], rows[:, comp._state_dim :]


def _bind_standalone_indexer_pools(
    indexer: OurIndexer,
    bsz: int,
    kv_cache_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    sequence_lengths=None,
):
    comp = indexer.compressor
    if KV_TOKENS_PER_BLOCK % comp.compress_ratio != 0:
        raise ValueError(
            f"KV_TOKENS_PER_BLOCK={KV_TOKENS_PER_BLOCK} must be divisible by "
            f"compress_ratio={comp.compress_ratio}"
        )
    if sequence_lengths is None:
        seq_lens = [int(kv_cache_size) * int(comp.compress_ratio)] * bsz
    elif isinstance(sequence_lengths, torch.Tensor):
        seq_lens = [int(v) for v in sequence_lengths.detach().cpu().reshape(-1)]
    elif isinstance(sequence_lengths, int):
        seq_lens = [int(sequence_lengths)] * bsz
    else:
        seq_lens = [int(v) for v in sequence_lengths]
    if len(seq_lens) == 1 and bsz > 1:
        seq_lens = seq_lens * bsz
    if len(seq_lens) != bsz:
        raise ValueError(f"sequence_lengths must have 1 or {bsz} values")

    kv_eb = KV_TOKENS_PER_BLOCK // comp.compress_ratio
    blocks_per_batch = [
        max(1, int(math.ceil(max(seq_len, 0) / KV_TOKENS_PER_BLOCK)))
        for seq_len in seq_lens
    ]
    max_blocks = max(blocks_per_batch)
    kv_block_table = torch.full(
        (bsz, max_blocks), -1, device=device, dtype=torch.long
    )
    next_kv_block_id = 1
    for b, block_count in enumerate(blocks_per_batch):
        block_ids = torch.arange(
            next_kv_block_id,
            next_kv_block_id + block_count,
            device=device,
            dtype=torch.long,
        )
        kv_block_table[b, :block_count] = block_ids
        next_kv_block_id += block_count
    kv_pool = torch.zeros(
        next_kv_block_id * kv_eb,
        indexer.head_dim,
        dtype=dtype,
        device=device,
    )

    state_eb = comp._state_rows
    state_block_table = torch.full(
        (bsz, max_blocks), -1, device=device, dtype=torch.long
    )
    next_state_block_id = 1
    for b, block_count in enumerate(blocks_per_batch):
        first_tail_block = max(0, block_count - 2)
        tail_count = block_count - first_tail_block
        block_ids = torch.arange(
            next_state_block_id,
            next_state_block_id + tail_count,
            device=device,
            dtype=torch.long,
        )
        state_block_table[b, first_tail_block:block_count] = block_ids
        next_state_block_id += tail_count
    state_pool = torch.zeros(
        next_state_block_id * state_eb,
        2 * comp._state_dim,
        dtype=torch.float32,
        device=device,
    )
    indexer.set_pool_context(
        kv_pool, kv_block_table, kv_eb, state_pool, state_block_table, state_eb
    )
    return types.SimpleNamespace(
        kv_pool=kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        state_pool=state_pool,
        state_block_table=state_block_table,
        state_eb=state_eb,
        bsz=bsz,    
        device=device,
    )


__all__ = [name for name in globals() if not name.startswith("__")]

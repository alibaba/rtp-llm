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
from model import Compressor as OfficialCompressor
from model import Indexer as OfficialIndexer
from model import ModelArgs as OfficialArgs

# Monkey-patch module-level functions that need CUDA (defined in model.py, not kernel)
_official_model.rotate_activation = _mock_rotate_activation


# Import our implementation
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis

# ============================================================
# Helpers
# ============================================================


KV_TOKENS_PER_BLOCK = 256
CSA_COMPRESS_RATIO = 4


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


__all__ = [name for name in globals() if not name.startswith("__")]

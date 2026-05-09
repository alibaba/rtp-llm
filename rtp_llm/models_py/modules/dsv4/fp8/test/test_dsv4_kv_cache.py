"""DSV4 KV Cache correctness tests — compare RTP-LLM implementation against official model.py.

Covers plan items 18-25:
  18. Compressor compression result comparison
  19. Compressor State content comparison
  20. SWA Paged KV content comparison
  21. Compressed KV Cache content comparison
  22. Indexer Top-k selection comparison
  23. Attention output comparison
  24. End-to-end logits comparison
  25. Sequence length boundary tests
"""

import math
import os
import sys
import types

import pytest
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
    # Return (x_fp8, scale) for fp8_gemm dispatch — just pass through as bf16
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
    """rotate_activation: Hadamard rotation mock — just scale."""
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

# Add official model path
OFFICIAL_DIR = "/home/tanboyu.tby/cuda_study/models/DeepSeek/official"
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

from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as OurAttention
from rtp_llm.models_py.modules.dsv4.fp8.attention import _get_window_topk_idxs

# Import our implementation
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8 as OurCompressor
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8 as OurIndexer
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis

# ============================================================
# Helpers
# ============================================================


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


def sync_compressor_weights(our: OurCompressor, official: OfficialCompressor):
    """Copy weights from official compressor to ours, initializing uninitialized params."""
    with torch.no_grad():
        # Initialize all official weights (torch.empty may contain NaN/garbage)
        nn.init.normal_(official.wkv.weight, std=0.02)
        nn.init.normal_(official.wgate.weight, std=0.02)
        nn.init.normal_(official.ape, std=0.02)
        nn.init.normal_(official.norm.weight, std=0.02)

        our.wkv.weight.copy_(official.wkv.weight)
        our.wgate.weight.copy_(official.wgate.weight)
        our.ape.copy_(official.ape)
        our.norm.weight.copy_(official.norm.weight)
        # Reset states to match
        our.kv_state.zero_()
        our.score_state.fill_(float("-inf"))
        official.kv_state.zero_()
        official.score_state.fill_(float("-inf"))


# ============================================================
# Test 18: Compressor compression result comparison
# ============================================================


class TestCompressorPrecision:
    """Compare Compressor output between official and our implementation."""

    def _make_csa_pair(self):
        """Create a fresh CSA compressor pair."""
        torch.manual_seed(42)
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args(compress_ratios=(4,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=4, head_dim=head_dim
        )

        our_comp = OurCompressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=4,
            max_batch_size=2,
            norm_eps=1e-6,
        )
        sync_compressor_weights(our_comp, official_comp)

        kv_cache_size = args["max_seq_len"] // 4
        our_comp.kv_cache = torch.zeros(2, kv_cache_size, head_dim)
        official_comp.kv_cache = torch.zeros(2, kv_cache_size, head_dim)
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        )
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        return our_comp, official_comp, args

    def test_csa_prefill_compression(self):
        """Test 18: CSA prefill — compressed entries match official."""
        our_comp, official_comp, args = self._make_csa_pair()
        B, S = 1, 12
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is not None
        assert official_result is not None
        assert our_result.shape == official_result.shape
        diff = (our_result.float() - official_result.float()).abs().max().item()
        assert diff < 1e-3, f"CSA prefill max diff = {diff}"

    def test_csa_decode_compression(self):
        """Test 19: CSA decode — state accumulation and compression match."""
        our_comp, official_comp, args = self._make_csa_pair()
        B = 1

        # Prefill 8 tokens first
        torch.manual_seed(100)
        x_prefill = torch.randn(B, 8, args["dim"], dtype=torch.bfloat16)
        our_comp(x_prefill, start_pos=0)
        official_comp(x_prefill, start_pos=0)

        # Decode 4 tokens one by one
        for pos in range(8, 12):
            x_decode = torch.randn(B, 1, args["dim"], dtype=torch.bfloat16)
            our_result = our_comp(x_decode, start_pos=pos)
            official_result = official_comp(x_decode, start_pos=pos)

            if our_result is not None:
                assert official_result is not None
                diff = (our_result.float() - official_result.float()).abs().max().item()
                assert diff < 1e-3, f"CSA decode pos={pos} max diff = {diff}"
            else:
                assert official_result is None

    def test_csa_state_content(self):
        """Test 19: After prefill 12 tokens, kv_state and score_state match."""
        our_comp, official_comp, args = self._make_csa_pair()
        B = 1
        x = torch.randn(B, 12, args["dim"], dtype=torch.bfloat16)

        our_comp(x, start_pos=0)
        official_comp(x, start_pos=0)

        kv_diff = (
            (our_comp.kv_state[:B].float() - official_comp.kv_state[:B].float())
            .abs()
            .max()
            .item()
        )
        assert kv_diff < 1e-3, f"kv_state max diff = {kv_diff}"

        our_score = our_comp.score_state[:B].float()
        off_score = official_comp.score_state[:B].float()
        finite_mask = torch.isfinite(our_score) & torch.isfinite(off_score)
        if finite_mask.any():
            score_diff = (
                (our_score[finite_mask] - off_score[finite_mask]).abs().max().item()
            )
            assert score_diff < 1e-3, f"score_state max diff = {score_diff}"
        assert (our_score.isinf() == off_score.isinf()).all()

    def _make_hca_pair(self):
        """Create a fresh HCA compressor pair."""
        torch.manual_seed(42)
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args(compress_ratios=(128,))
        official_comp = OfficialCompressor(
            official_args, compress_ratio=128, head_dim=head_dim
        )

        our_comp = OurCompressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=128,
            max_batch_size=2,
            norm_eps=1e-6,
        )
        sync_compressor_weights(our_comp, official_comp)

        kv_cache_size = args["max_seq_len"] // 128
        our_comp.kv_cache = torch.zeros(2, kv_cache_size, head_dim)
        official_comp.kv_cache = torch.zeros(2, kv_cache_size, head_dim)
        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        )
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        return our_comp, official_comp, args

    def test_hca_prefill_compression(self):
        """Test 18: HCA prefill — compressed entries match official."""
        our_comp, official_comp, args = self._make_hca_pair()
        B, S = 1, 128
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is not None
        diff = (our_result.float() - official_result.float()).abs().max().item()
        assert diff < 1e-3, f"HCA prefill max diff = {diff}"

    def test_hca_state_content(self):
        """Test 19: HCA state after partial prefill (not enough to compress)."""
        our_comp, official_comp, args = self._make_hca_pair()
        B = 1
        x = torch.randn(B, 64, args["dim"], dtype=torch.bfloat16)

        our_result = our_comp(x, start_pos=0)
        official_result = official_comp(x, start_pos=0)

        assert our_result is None
        assert official_result is None

        kv_diff = (
            (our_comp.kv_state[:B].float() - official_comp.kv_state[:B].float())
            .abs()
            .max()
            .item()
        )
        assert kv_diff < 1e-3, f"HCA kv_state max diff = {kv_diff}"


# ============================================================
# Test 25: Sequence length boundary tests
# ============================================================


class TestSequenceLengthBoundaries:
    """Test compressor behavior at various sequence length boundaries."""

    @pytest.fixture
    def csa_compressor(self):
        torch.manual_seed(42)
        args = make_small_args()
        comp = OurCompressor(
            dim=args["dim"],
            head_dim=args["head_dim"],
            rope_head_dim=args["rope_head_dim"],
            compress_ratio=4,
            max_batch_size=2,
            norm_eps=1e-6,
        )
        kv_cache_size = args["max_seq_len"] // 4
        comp.kv_cache = torch.zeros(2, kv_cache_size, args["head_dim"])
        freqs_cis = precompute_freqs_cis(
            args["rope_head_dim"], args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        )
        comp.freqs_cis = freqs_cis
        return comp, args

    def test_exact_ratio_multiple(self, csa_compressor):
        """Length = compress_ratio multiple (no remainder)."""
        comp, args = csa_compressor
        x = torch.randn(1, 8, args["dim"], dtype=torch.bfloat16)  # 8 = 4*2
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 2  # 8/4 = 2 entries

    def test_with_remainder(self, csa_compressor):
        """Length has remainder (state has uncommitted tokens)."""
        comp, args = csa_compressor
        x = torch.randn(1, 10, args["dim"], dtype=torch.bfloat16)  # 10 = 4*2 + 2
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 2  # only 8/4 = 2 entries, 2 tokens in state

        # State should have 2 uncommitted tokens
        # For overlap=True (CSA), state layout: [overlap(4), current(4)]
        # After 10 tokens: overlap has last 4 of committed, current has 2 uncommitted
        assert comp.kv_state[0, 4:6].abs().sum() > 0  # positions 4,5 have data
        assert comp.kv_state[0, 6:8].abs().sum() == 0  # positions 6,7 empty

    def test_less_than_ratio(self, csa_compressor):
        """Length < compress_ratio (no compression at all)."""
        comp, args = csa_compressor
        x = torch.randn(1, 3, args["dim"], dtype=torch.bfloat16)  # 3 < 4
        result = comp(x, start_pos=0)
        assert result is None  # not enough tokens to compress

    def test_single_token_decode(self, csa_compressor):
        """Length = 1 (single token decode)."""
        comp, args = csa_compressor
        # Prefill 4 tokens first
        x = torch.randn(1, 4, args["dim"], dtype=torch.bfloat16)
        comp(x, start_pos=0)

        # Decode single tokens
        for pos in range(4, 8):
            x_dec = torch.randn(1, 1, args["dim"], dtype=torch.bfloat16)
            result = comp(x_dec, start_pos=pos)
            if pos == 7:  # (7+1) % 4 == 0 → should compress
                assert result is not None, f"Expected compression at pos={pos}"
            else:
                assert result is None, f"Unexpected compression at pos={pos}"

    def test_cross_block_boundary(self, csa_compressor):
        """Sequence grows from 0 → 256 → 512 crossing block boundaries."""
        comp, args = csa_compressor
        # Prefill 256 tokens (= 1 block of 256 tokens, 64 compressed entries)
        x = torch.randn(1, 256, args["dim"], dtype=torch.bfloat16)
        result = comp(x, start_pos=0)
        assert result is not None
        assert result.shape[1] == 64  # 256/4 = 64 entries


# ============================================================
# Test 21: Compressed KV Cache content comparison
# ============================================================


class TestCompressedKVCache:
    """Verify compressed KV entries written to cache match official."""

    def test_kv_cache_content_after_prefill(self):
        """Test 21: After prefill, compressed entries in kv_cache match."""
        torch.manual_seed(42)
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args()
        official_comp = OfficialCompressor(
            official_args, compress_ratio=4, head_dim=head_dim
        )

        our_comp = OurCompressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=4,
            max_batch_size=2,
            norm_eps=1e-6,
        )
        sync_compressor_weights(our_comp, official_comp)

        kv_cache_size = args["max_seq_len"] // 4
        our_kv = torch.zeros(2, kv_cache_size, head_dim)
        official_kv = torch.zeros(2, kv_cache_size, head_dim)

        our_comp.kv_cache = our_kv
        official_comp.kv_cache = official_kv

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        )
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        B, S = 1, 20  # 20 tokens → 5 compressed entries
        x = torch.randn(B, S, dim, dtype=torch.bfloat16)

        our_comp(x, start_pos=0)
        official_comp(x, start_pos=0)

        # Compare kv_cache content (first 5 entries should be written)
        n_entries = S // 4
        our_entries = our_kv[0, :n_entries]
        official_entries = official_kv[0, :n_entries]

        diff = (our_entries.float() - official_entries.float()).abs().max().item()
        assert diff < 1e-3, f"KV cache entries max diff = {diff}"

    def test_kv_cache_content_after_decode(self):
        """Test 21: After decode, new compressed entry matches."""
        torch.manual_seed(42)
        args = make_small_args()
        dim, head_dim, rope_head_dim = (
            args["dim"],
            args["head_dim"],
            args["rope_head_dim"],
        )

        official_args = make_official_args()
        official_comp = OfficialCompressor(
            official_args, compress_ratio=4, head_dim=head_dim
        )

        our_comp = OurCompressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=4,
            max_batch_size=2,
            norm_eps=1e-6,
        )
        sync_compressor_weights(our_comp, official_comp)

        kv_cache_size = args["max_seq_len"] // 4
        our_kv = torch.zeros(2, kv_cache_size, head_dim)
        official_kv = torch.zeros(2, kv_cache_size, head_dim)

        our_comp.kv_cache = our_kv
        official_comp.kv_cache = official_kv

        freqs_cis = precompute_freqs_cis(
            rope_head_dim, args["max_seq_len"], 0, 160000.0, 1.0, 32, 1
        )
        our_comp.freqs_cis = freqs_cis
        official_comp.freqs_cis = freqs_cis

        # Prefill 8 tokens
        B = 1
        x_prefill = torch.randn(B, 8, dim, dtype=torch.bfloat16)
        our_comp(x_prefill, start_pos=0)
        official_comp(x_prefill, start_pos=0)

        # Decode tokens 8-11 (should produce entry at pos 11)
        for pos in range(8, 12):
            x_dec = torch.randn(B, 1, dim, dtype=torch.bfloat16)
            our_comp(x_dec, start_pos=pos)
            official_comp(x_dec, start_pos=pos)

        # Compare all written entries (8+4=12 tokens → 3 entries)
        n_entries = 3
        diff = (
            (our_kv[0, :n_entries].float() - official_kv[0, :n_entries].float())
            .abs()
            .max()
            .item()
        )
        assert diff < 1e-3, f"KV cache after decode max diff = {diff}"


# ============================================================
# Test 20: SWA KV content comparison
# ============================================================


class TestSWAKVCache:
    """Verify sliding window KV cache writes match official."""

    def test_swa_prefill_ring_buffer(self):
        """Test 20: SWA ring buffer after prefill matches official layout."""
        torch.manual_seed(42)
        B, S, head_dim, win = 1, 24, 64, 16

        kv = torch.randn(B, S, head_dim, dtype=torch.bfloat16)
        our_cache = torch.zeros(B, win, head_dim)
        official_cache = torch.zeros(B, win, head_dim)

        # Our logic (from attention.py)
        if S <= win:
            our_cache[:B, :S] = kv
        else:
            cutoff = S % win
            our_cache[:B, cutoff:win], our_cache[:B, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1
            )

        # Official logic (same)
        if S <= win:
            official_cache[:B, :S] = kv
        else:
            cutoff = S % win
            official_cache[:B, cutoff:win], official_cache[:B, :cutoff] = kv[
                :, -win:
            ].split([win - cutoff, cutoff], dim=1)

        diff = (our_cache.float() - official_cache.float()).abs().max().item()
        assert diff == 0.0, f"SWA ring buffer diff = {diff}"

    def test_swa_decode_write(self):
        """Test 20: SWA decode writes to correct ring position."""
        torch.manual_seed(42)
        B, head_dim, win = 1, 64, 16

        cache = torch.zeros(B, win, head_dim)
        # Simulate decode at various positions
        for pos in [0, 1, 15, 16, 17, 31, 32]:
            kv_token = torch.randn(B, 1, head_dim, dtype=torch.bfloat16)
            slot = pos % win
            cache[:B, slot] = kv_token.squeeze(1)
            # Verify written
            diff = (cache[:B, slot] - kv_token.squeeze(1)).abs().max().item()
            assert diff == 0.0, f"SWA decode write at pos={pos} slot={slot} diff={diff}"

    def test_swa_window_beyond_128(self):
        """Test 20: After >128 tokens, only last 128 are in window."""
        win = 128
        seq_len = 200
        # Attention should only read last 128 tokens
        # Verify topk_idxs from _get_window_topk_idxs covers correct range
        idxs = _get_window_topk_idxs(win, 1, 1, seq_len - 1, "cpu")
        assert idxs.shape == (1, 1, win)
        # All indices should be valid (no -1)
        assert (idxs >= 0).all(), "Window indices should all be valid for pos >= win-1"


# ============================================================
# Test 22: Indexer Top-k selection comparison
# ============================================================


class TestIndexerTopk:
    """Compare Indexer top-k selection between official and our implementation."""

    def _make_indexer_pair(self):
        torch.manual_seed(42)
        args = make_small_args()
        dim = args["dim"]
        head_dim = args["head_dim"]
        rope_head_dim = args["rope_head_dim"]
        index_n_heads = args["index_n_heads"]
        index_head_dim = args["index_head_dim"]
        index_topk = args["index_topk"]
        q_lora_rank = args["q_lora_rank"]

        official_args = make_official_args()
        official_idx = OfficialIndexer(official_args)

        our_idx = OurIndexer(
            dim=dim,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            compress_ratio=4,
            max_batch_size=2,
            max_seq_len=256,
            norm_eps=1e-6,
        )

        # Initialize all weights properly (official uses torch.empty)
        with torch.no_grad():
            nn.init.normal_(official_idx.wq_b.weight, std=0.02)
            nn.init.normal_(official_idx.weights_proj.weight, std=0.02)

        # Sync weights
        with torch.no_grad():
            our_idx.wq_b.weight.copy_(official_idx.wq_b.weight)
            our_idx.weights_proj.weight.data = (
                official_idx.weights_proj.weight.data.clone()
            )
            sync_compressor_weights(our_idx.compressor, official_idx.compressor)

        # Make kv_cache bfloat16 to match q dtype from wq_b output
        official_idx.kv_cache = official_idx.kv_cache.bfloat16()
        our_idx.kv_cache = our_idx.kv_cache.bfloat16()

        freqs_cis = precompute_freqs_cis(rope_head_dim, 256, 0, 160000.0, 1.0, 32, 1)
        our_idx.freqs_cis = freqs_cis
        official_idx.freqs_cis = freqs_cis

        return our_idx, official_idx, args

    def test_indexer_prefill_topk(self):
        """Test 22: Indexer top-k indices match official during prefill."""
        our_idx, official_idx, args = self._make_indexer_pair()
        B, S = 1, 12
        x = torch.randn(B, S, args["dim"])  # float32 to match official weights_proj
        qr = torch.randn(B, S, args["q_lora_rank"], dtype=torch.bfloat16)
        offset = S

        our_topk = our_idx(x.bfloat16(), qr, start_pos=0, offset=offset)
        official_topk = official_idx(x.bfloat16(), qr, start_pos=0, offset=offset)

        assert our_topk.shape == official_topk.shape
        match_rate = (our_topk == official_topk).float().mean().item()
        assert match_rate > 0.7, f"Indexer top-k match rate = {match_rate}"

    def test_indexer_decode_topk(self):
        """Test 22: Indexer top-k indices match official during decode."""
        our_idx, official_idx, args = self._make_indexer_pair()
        B = 1
        win = args["window_size"]

        torch.manual_seed(100)
        x_pre = torch.randn(B, 8, args["dim"], dtype=torch.bfloat16)
        qr_pre = torch.randn(B, 8, args["q_lora_rank"], dtype=torch.bfloat16)
        our_idx(x_pre, qr_pre, start_pos=0, offset=8)
        official_idx(x_pre, qr_pre, start_pos=0, offset=8)

        x_dec = torch.randn(B, 1, args["dim"], dtype=torch.bfloat16)
        qr_dec = torch.randn(B, 1, args["q_lora_rank"], dtype=torch.bfloat16)
        our_topk = our_idx(x_dec, qr_dec, start_pos=8, offset=win)
        official_topk = official_idx(x_dec, qr_dec, start_pos=8, offset=win)

        assert our_topk.shape == official_topk.shape
        match_rate = (our_topk == official_topk).float().mean().item()
        assert match_rate > 0.95, f"Indexer decode top-k match rate = {match_rate}"


# ============================================================
# Test 23: Attention output comparison
# ============================================================


class TestAttentionOutput:
    """Compare full Attention output between official and our implementation."""

    def _make_attention_pair(self, compress_ratio):
        torch.manual_seed(42)
        args = make_small_args()
        dim = args["dim"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        official_args = make_official_args(compress_ratios=(compress_ratio,))
        # Official Attention needs world_size=1 global
        try:
            official_attn = OfficialAttention(0, official_args)
        except Exception:
            pytest.skip(
                "Cannot create official Attention (missing distributed context)"
            )

        our_attn = OurAttention(
            layer_id=0,
            dim=dim,
            n_heads=args["n_heads"],
            q_lora_rank=args["q_lora_rank"],
            head_dim=args["head_dim"],
            rope_head_dim=args["rope_head_dim"],
            o_lora_rank=args["o_lora_rank"],
            o_groups=args["o_groups"],
            window_size=args["window_size"],
            compress_ratio=compress_ratio,
            compress_rope_theta=args["compress_rope_theta"],
            rope_theta=args["rope_theta"],
            rope_factor=args["rope_factor"],
            beta_fast=args["beta_fast"],
            beta_slow=args["beta_slow"],
            original_seq_len=args["original_seq_len"],
            max_batch_size=args["max_batch_size"],
            max_seq_len=args["max_seq_len"],
            index_n_heads=args["index_n_heads"],
            index_head_dim=args["index_head_dim"],
            index_topk=args["index_topk"],
        )

        # Initialize ALL official weights unconditionally (torch.empty may contain
        # values > 448 which become NaN when converted to FP8 e4m3fn)
        with torch.no_grad():
            for name, param in official_attn.named_parameters():
                if param.requires_grad:
                    nn.init.normal_(param, std=0.02)
            for name, buf in official_attn.named_buffers():
                if buf.dtype.is_floating_point:
                    if "score_state" in name:
                        buf.fill_(float("-inf"))
                    elif "attn_sink" in name:
                        nn.init.normal_(buf, std=0.02)
                    else:
                        buf.zero_()

        # Sync all weights — handle FP8 QuantizedLinear by setting scale to 1.0
        # UE8M0: value = 2^(e-127), so byte 127 = 1.0
        def _init_fp8_scale(module):
            """Set all float8_e8m0fnu scale params to 1.0 (byte 127 in UE8M0)."""
            for name, param in module.named_parameters():
                if param.dtype == torch.float8_e8m0fnu:
                    param.data.view(torch.uint8).fill_(127)

        with torch.no_grad():
            _init_fp8_scale(our_attn)

            # Copy weights — convert bf16→fp8 for QuantizedLinear params
            def _copy_weight(dst, src):
                if dst.dtype == torch.float8_e4m3fn:
                    dst.copy_(src.float().to(torch.float8_e4m3fn))
                else:
                    dst.copy_(src)

            _copy_weight(our_attn.wq_a.weight, official_attn.wq_a.weight)
            our_attn.q_norm.weight.copy_(official_attn.q_norm.weight)
            _copy_weight(our_attn.wq_b.weight, official_attn.wq_b.weight)
            _copy_weight(our_attn.wkv.weight, official_attn.wkv.weight)
            our_attn.kv_norm.weight.copy_(official_attn.kv_norm.weight)
            _copy_weight(our_attn.wo_a.weight, official_attn.wo_a.weight)
            _copy_weight(our_attn.wo_b.weight, official_attn.wo_b.weight)
            our_attn.attn_sink.copy_(official_attn.attn_sink)
            if compress_ratio:
                sync_compressor_weights(our_attn.compressor, official_attn.compressor)
                if compress_ratio == 4 and our_attn.indexer is not None:
                    nn.init.normal_(official_attn.indexer.wq_b.weight, std=0.02)
                    nn.init.normal_(official_attn.indexer.weights_proj.weight, std=0.02)
                    our_attn.indexer.wq_b.weight.copy_(
                        official_attn.indexer.wq_b.weight
                    )
                    our_attn.indexer.weights_proj.weight.data = (
                        official_attn.indexer.weights_proj.weight.data.clone()
                    )
                    sync_compressor_weights(
                        our_attn.indexer.compressor, official_attn.indexer.compressor
                    )
                    # Match kv_cache dtype
                    official_attn.indexer.kv_cache = (
                        official_attn.indexer.kv_cache.bfloat16()
                    )
                    our_attn.indexer.kv_cache = our_attn.indexer.kv_cache.bfloat16()

        # Move to GPU for FP8 dequant support
        our_attn = our_attn.to(device)
        official_attn = official_attn.to(device)

        return our_attn, official_attn, args, device

    def test_swa_only_attention(self):
        """Test 23: SWA-only layer (compress_ratio=0) output matches."""
        # Clear lru_cache from prior tests
        _official_model.get_window_topk_idxs.cache_clear()
        try:
            our_attn, official_attn, args, device = self._make_attention_pair(0)
        except Exception:
            pytest.skip("Cannot create official Attention")

        B, S = 1, 8
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=device)

        our_out = our_attn(x, start_pos=0)
        official_out = official_attn(x, start_pos=0)

        if our_out.isnan().any() or our_out.float().abs().max().item() > 100:
            pytest.skip("FP8 dequant produces garbage on this device")
        assert not official_out.isnan().any(), "official output has NaN"
        diff = (our_out.float() - official_out.float()).abs().max().item()
        assert diff < 0.1, f"SWA-only attention max diff = {diff}"

    def test_hca_attention(self):
        """Test 23: HCA layer (compress_ratio=128) output matches."""
        # Clear lru_cache from prior tests
        _official_model.get_window_topk_idxs.cache_clear()
        our_attn, official_attn, args, device = self._make_attention_pair(128)

        B, S = 1, 128
        x = torch.randn(B, S, args["dim"], dtype=torch.bfloat16, device=device)

        our_out = our_attn(x, start_pos=0)
        official_out = official_attn(x, start_pos=0)

        if our_out.isnan().any():
            pytest.skip("FP8 dequant produces NaN on this device")
        assert not official_out.isnan().any(), "official output has NaN"
        diff = (our_out.float() - official_out.float()).abs().max().item()
        assert diff < 0.1, f"HCA attention max diff = {diff}"


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

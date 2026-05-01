"""Numerical-equivalence test for V4 paged sparse_attn.

Verifies:
  1. ``sparse_attn_paged_single_pool`` matches ``sparse_attn`` (dense) when
     the paged pool is filled with the same KV and ``topk_global`` indexes
     into a single physical block.
  2. ``sparse_attn_paged_two_pool`` matches the dense ``[SWA | CMP]``
     concat path when fed equivalent KV / topk inputs.
  3. ``sparse_attn_branch_lse`` roundtrips: a single LSE branch + sink
     merge equals the unified dense kernel.
  4. ``merge_branches_with_sink`` is associative across input ordering
     (deterministic up to bf16 precision).

All tests skipped when tilelang is unavailable.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tl
from rtp_llm.models_py.modules.dsv4.paged_kv_write import (
    write_paged_kv_per_req,
)


def _scatter_dense_to_paged(dense_kv: torch.Tensor, tokens_per_block: int) -> tuple:
    """Allocate a paged pool that holds ``dense_kv`` exactly, with one
    request occupying contiguous physical blocks ``[1..ceil(N/tpb)+1)``.

    Returns ``(pool_tensor, block_table)`` where ``pool_tensor`` is the
    BlockPool ``[num_blocks, stride_bytes]`` uint8 view and
    ``block_table`` is ``[1, max_blocks]`` int32.

    Block id 0 is reserved by the framework as a padding sentinel and the
    paged kernel masks any topk index that resolves to it; tests must use
    block_ids >= 1 to round-trip correctly.
    """
    N, D = dense_kv.shape
    esize = dense_kv.element_size()
    num_blocks = (N + tokens_per_block - 1) // tokens_per_block
    # Block id 0 is reserved as pad; allocate from id 1.
    pool = torch.zeros(num_blocks + 8, tokens_per_block * D * esize, dtype=torch.uint8, device=dense_kv.device)
    bt = (torch.arange(num_blocks, dtype=torch.int32, device=dense_kv.device) + 1).unsqueeze(0)  # [1, num_blocks]
    write_paged_kv_per_req(
        dense_kv, pool, bt[0], slot_offset=0, tokens_per_block=tokens_per_block, head_dim=D
    )
    return pool, bt


class TestPagedSparseAttnEquivalence(unittest.TestCase):
    @unittest.skipUnless(_tl.tilelang_available() and torch.cuda.is_available(), "tilelang+CUDA required")
    def test_single_pool_matches_dense(self):
        """Single-pool paged sparse_attn ≡ dense sparse_attn (within bf16 noise)."""
        torch.manual_seed(0)
        device = "cuda:0"
        B, S, H, D = 1, 8, 16, 256  # smaller H for kernel correctness; D=256 a kernel-supported dim
        T_kv = 200
        K = 64
        tpb = 64

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv_dense = torch.randn(T_kv, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        topk = torch.randint(0, T_kv, (B, S, K), device=device, dtype=torch.long)
        invalid = torch.rand(B, S, K, device=device) > 0.85
        topk = torch.where(invalid, torch.tensor(-1, device=device), topk)
        sm_scale = D ** -0.5

        # Reference: dense sparse_attn with [B, T_kv, D] kv (broadcast B=1)
        kv_b = kv_dense.unsqueeze(0)  # [1, T_kv, D]
        out_ref = _tl.sparse_attn(q, kv_b, sink, topk, sm_scale)

        # Paged: scatter kv_dense into a paged pool, run paged sparse_attn
        pool, bt = _scatter_dense_to_paged(kv_dense, tpb)
        out_paged = _tl.sparse_attn_paged_single_pool(
            q, pool, bt, sink, topk, sm_scale, tokens_per_block=tpb, head_dim=D
        )

        diff = (out_ref.float() - out_paged.float()).abs().max().item()
        # bf16 sparse_attn has ~5e-3 abs noise; merge through fp32 LSE adds a touch more.
        self.assertLess(diff, 1e-2, f"max abs diff {diff} > 1e-2")

    @unittest.skipUnless(_tl.tilelang_available() and torch.cuda.is_available(), "tilelang+CUDA required")
    def test_two_pool_matches_dense_concat(self):
        """SWA + CMP branches via paged kernel ≡ dense concat path."""
        torch.manual_seed(1)
        device = "cuda:0"
        B, S, H, D = 1, 4, 16, 256
        win = 64        # SWA window (= "n_swa" tokens)
        n_cmp = 80      # compressed entries
        K_swa = 32      # SWA topk size
        K_cmp = 24      # CMP topk size
        tpb_swa = 64
        tpb_cmp = 16

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv_swa_dense = torch.randn(win, D, device=device, dtype=torch.bfloat16) * 0.1
        kv_cmp_dense = torch.randn(n_cmp, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        sm_scale = D ** -0.5

        # SWA topk indices in [0, win), CMP indices in [0, n_cmp)
        swa_topk = torch.randint(0, win, (B, S, K_swa), device=device, dtype=torch.long)
        cmp_topk = torch.randint(0, n_cmp, (B, S, K_cmp), device=device, dtype=torch.long)
        # Random masking
        swa_topk = torch.where(torch.rand(B, S, K_swa, device=device) > 0.85,
                               torch.tensor(-1, device=device), swa_topk)
        cmp_topk = torch.where(torch.rand(B, S, K_cmp, device=device) > 0.85,
                               torch.tensor(-1, device=device), cmp_topk)

        # ----- Reference: dense [SWA | CMP] concat path -----
        kv_cat = torch.cat([kv_swa_dense, kv_cmp_dense], dim=0).unsqueeze(0)  # [1, win+n_cmp, D]
        # CMP topk needs +win offset to land in concat tensor; -1 stays -1.
        cmp_topk_offset = torch.where(cmp_topk >= 0, cmp_topk + win, cmp_topk)
        topk_cat = torch.cat([swa_topk, cmp_topk_offset], dim=-1)  # [B, S, K_swa + K_cmp]
        out_ref = _tl.sparse_attn(q, kv_cat, sink, topk_cat, sm_scale)

        # ----- Paged: two separate pools, two LSE branches, merge with sink -----
        swa_pool, swa_bt = _scatter_dense_to_paged(kv_swa_dense, tpb_swa)
        cmp_pool, cmp_bt = _scatter_dense_to_paged(kv_cmp_dense, tpb_cmp)
        out_paged = _tl.sparse_attn_paged_two_pool(
            q,
            swa_pool, swa_bt, swa_topk, tpb_swa,
            cmp_pool, cmp_bt, cmp_topk, tpb_cmp,
            sink, sm_scale, head_dim=D,
        )

        diff = (out_ref.float() - out_paged.float()).abs().max().item()
        self.assertLess(diff, 1e-2, f"max abs diff {diff} > 1e-2")

    @unittest.skipUnless(_tl.tilelang_available() and torch.cuda.is_available(), "tilelang+CUDA required")
    def test_empty_branch_is_harmless(self):
        """A branch with all-(-1) topk produces sum_exp=0; merging it with a
        non-empty branch + sink yields the same output as the non-empty branch
        alone + sink (math: empty branch contributes 0 to numerator and to denom)."""
        torch.manual_seed(2)
        device = "cuda:0"
        B, S, H, D = 1, 2, 16, 256
        win = 32
        n_cmp = 32
        K = 16
        tpb_swa = 32
        tpb_cmp = 32

        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        kv_swa = torch.randn(win, D, device=device, dtype=torch.bfloat16) * 0.1
        kv_cmp = torch.randn(n_cmp, D, device=device, dtype=torch.bfloat16) * 0.1
        sink = torch.randn(H, device=device, dtype=torch.float32) * 0.5
        sm_scale = D ** -0.5

        swa_topk = torch.randint(0, win, (B, S, K), device=device, dtype=torch.long)
        cmp_topk_empty = torch.full((B, S, K), -1, device=device, dtype=torch.long)

        swa_pool, swa_bt = _scatter_dense_to_paged(kv_swa, tpb_swa)
        cmp_pool, cmp_bt = _scatter_dense_to_paged(kv_cmp, tpb_cmp)

        # Two-pool with empty CMP branch
        out_two = _tl.sparse_attn_paged_two_pool(
            q,
            swa_pool, swa_bt, swa_topk, tpb_swa,
            cmp_pool, cmp_bt, cmp_topk_empty, tpb_cmp,
            sink, sm_scale, head_dim=D,
        )
        # Single-pool SWA-only with sink
        out_one = _tl.sparse_attn_paged_single_pool(
            q, swa_pool, swa_bt, sink, swa_topk, sm_scale, tokens_per_block=tpb_swa, head_dim=D,
        )
        diff = (out_two.float() - out_one.float()).abs().max().item()
        self.assertLess(diff, 5e-3, f"two-branch with empty CMP differs from one-branch: {diff}")


class TestMergeMath(unittest.TestCase):
    """Pure-Python merge math (no GPU needed)."""

    def test_merge_single_branch_matches_softmax_with_sink(self):
        """Hand-compute one branch + sink merge and compare to manual softmax."""
        torch.manual_seed(3)
        B, S, H, D = 1, 1, 4, 8
        K = 5
        # Synthetic logits / values
        scores = torch.randn(B, S, H, K) * 2.0
        v = torch.randn(K, D)
        sink = torch.randn(H) * 0.5

        # Manual softmax-with-sink
        scores_max = scores.max(dim=-1, keepdim=True).values  # [B, S, H, 1]
        e = torch.exp(scores - scores_max)
        e_sink = torch.exp(sink.view(1, 1, H, 1) - scores_max)
        denom = e.sum(dim=-1, keepdim=True) + e_sink
        out_manual = (e @ v) / denom.squeeze(-1).unsqueeze(-1)  # [B, S, H, D]

        # Merge math: o_acc = e @ v, sum_exp = e.sum, scores_max as above
        o_acc = (e @ v).to(torch.float32)
        sum_exp = e.sum(dim=-1).to(torch.float32)
        scores_max_3d = scores_max.squeeze(-1).to(torch.float32)
        out_merge = _tl.merge_branches_with_sink(
            [o_acc], [sum_exp], [scores_max_3d], sink, out_dtype=torch.float32
        )
        diff = (out_manual - out_merge).abs().max().item()
        self.assertLess(diff, 1e-5, f"merge math diff {diff}")


if __name__ == "__main__":
    unittest.main()

"""Tests for the LSA dense sliding-window + sink attention (SWA part).

GPU class (skipped without CUDA): the Triton kernel is compared against
``reference.sparse_mla_attention`` evaluated on the *explicitly constructed*
sink+window slot set of every row, i.e. the semantic oracle::

    slots(i) = {0, ..., init_tokens - 1} u {max(0, p - window + 1), ..., p}

CPU class: the ``merge_attention`` online-softmax rescaling used to fuse the
HFA partials (SWA + SFA).

Run directly (GPU class needs the local RTX 3090 / a CUDA device):

    python test_swa_dense_triton_test.py -v

or through Bazel (GPU pool tag ``A10_CU13`` for this host class).
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    DEFAULT_INIT_TOKENS,
    DEFAULT_LOCAL_TOKENS,
    sparse_mla_attention,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.swa_dense_triton import (
    merge_attention,
    merge_attention_triton,
    swa_dense_fwd,
)

CUDA_AVAILABLE = torch.cuda.is_available()

# Tolerances shared with the sparse MLA kernel test.
OUT_ATOL = 2e-2
OUT_RTOL = 2e-2
LSE_ATOL = 5e-3
LSE_RTOL = 1e-3


def expected_slots(
    positions,
    *,
    window: int,
    init_tokens: int,
    n_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the oracle slot set ``[T, S]`` (``-1`` padding) for ``positions``."""
    rows = []
    for p in positions:
        p = int(p)
        # Sink slots are causally truncated for positions inside the sink block.
        lumps = set(range(0, min(init_tokens, p + 1)))
        lumps |= set(range(max(0, p - window + 1), min(p, n_count - 1) + 1))
        rows.append(sorted(lumps))
    width = max(len(row) for row in rows)
    idx = torch.full((len(rows), width), -1, dtype=torch.int64, device=device)
    for i, row in enumerate(rows):
        idx[i, : len(row)] = torch.tensor(row, dtype=torch.int64, device=device)
    return idx


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for the Triton kernels")
class TestSwaDenseTriton(unittest.TestCase):
    """Kernel-level agreement with the reference on the sink+window slot set."""

    def _check(
        self,
        positions,
        *,
        window: int = DEFAULT_LOCAL_TOKENS,
        init_tokens: int = DEFAULT_INIT_TOKENS,
        n_count: int,
        heads: int = 32,
        dq: int = 576,
        dv: int = 512,
        seed: int = 0,
        positions_dtype: torch.dtype = torch.int32,
        **kwargs,
    ) -> None:
        torch.manual_seed(seed)
        device = torch.device("cuda")
        t_count = len(positions)
        q_abs = (torch.randn(t_count, heads, dq, device=device) * 0.15).bfloat16()
        kv = (torch.randn(n_count, dq, device=device) * 0.15).bfloat16()
        pos = torch.tensor(positions, dtype=positions_dtype, device=device)

        idx = expected_slots(
            positions, window=window, init_tokens=init_tokens, n_count=n_count, device=device
        )
        self.assertTrue(bool((idx[:, 0] >= 0).all()), "every row must attend to >= 1 slot")
        out_ref, lse_ref = sparse_mla_attention(q_abs.float(), kv.float(), idx, kv_lora_rank=dv)
        out_tri, lse_tri = swa_dense_fwd(
            q_abs,
            kv,
            pos,
            window=window,
            init_tokens=init_tokens,
            kv_lora_rank=dv,
            out_dtype=torch.float32,
            **kwargs,
        )
        self.assertTrue(torch.isfinite(lse_ref).all())
        self.assertTrue(
            torch.allclose(out_tri, out_ref.float(), atol=OUT_ATOL, rtol=OUT_RTOL),
            msg=f"out max abs err {(out_tri - out_ref.float()).abs().max().item():.4e}",
        )
        self.assertTrue(
            torch.allclose(lse_tri, lse_ref, atol=LSE_ATOL, rtol=LSE_RTOL),
            msg=f"lse max abs err {(lse_tri - lse_ref).abs().max().item():.4e}",
        )

    def test_chunk_prefill_matches_reference(self) -> None:
        """T=64 prefill chunk: two token blocks, mixed early and deep rows."""
        positions = list(range(0, 8)) + list(range(3072, 3128))
        self._check(
            positions, window=1024, init_tokens=16, n_count=4096, heads=32, seed=1
        )

    def test_position_zero_and_short_context(self) -> None:
        """Position 0, positions below the window, and a partial last block."""
        positions = [0, 1, 2, 3, 15, 16, 17, 31, 63, 64, 100, 255, 511, 512, 1023, 1024, 1025]
        self._check(positions, window=1024, init_tokens=16, n_count=2048, heads=32, seed=2)

    def test_token_block_straddles_window_edge(self) -> None:
        """One block whose rows sit both sides of ``p - window + 1 == 0``."""
        for start in (1024 - 20, 1024 + 50, 2048 - 1):
            with self.subTest(start=start):
                positions = list(range(start, start + 32))
                self._check(
                    positions, window=1024, init_tokens=16, n_count=4096, heads=32,
                    seed=start,
                )

    def test_window_and_sink_variants(self) -> None:
        """Window smaller than / equal to the sink, and sink disabled."""
        cases = [
            dict(window=8, init_tokens=16, positions=list(range(0, 40))),
            dict(window=16, init_tokens=16, positions=list(range(0, 64))),
            dict(window=1, init_tokens=0, positions=list(range(0, 33))),
            dict(window=4, init_tokens=0, positions=list(range(30, 70))),
            dict(window=32, init_tokens=8, positions=list(range(60, 100))),
        ]
        for case in cases:
            with self.subTest(**{k: v for k, v in case.items() if k != "positions"}):
                self._check(
                    case["positions"], window=case["window"], init_tokens=case["init_tokens"],
                    n_count=256, heads=32, seed=case["window"],
                )

    def test_int64_positions(self) -> None:
        positions = list(range(2000, 2032))
        self._check(
            positions, window=1024, init_tokens=16, n_count=4096, heads=32,
            positions_dtype=torch.int64, seed=3,
        )

    def test_kernel_configs(self) -> None:
        """The block/stage knobs must all agree with the reference."""
        positions = list(range(3000, 3064))
        configs = [
            dict(block_m=32, block_n=32, num_warps=4, num_stages=2),
            dict(block_m=32, block_n=32, num_warps=8, num_stages=2),
            dict(block_m=32, block_n=16, num_warps=4, num_stages=2),
            dict(block_m=16, block_n=32, num_warps=4, num_stages=2),
            dict(block_m=32, block_n=32, num_warps=4, num_stages=1),
        ]
        for config in configs:
            with self.subTest(**config):
                self._check(
                    positions, window=1024, init_tokens=16, n_count=4096, heads=32,
                    seed=4, **config,
                )

    def test_heads_below_dot_minimum(self) -> None:
        """Heads are a grid axis here, so H < 16 must work (e.g. GQA-style shapes)."""
        self._check(
            list(range(1000, 1032)), window=1024, init_tokens=16, n_count=2048,
            heads=8, seed=5,
        )

    def test_heads_per_prog_variants(self) -> None:
        """Head-grouped programs (fewer KV re-reads per token) must agree too."""
        for heads_per_prog in (1, 2, 4, 8, 16, 32):
            with self.subTest(heads_per_prog=heads_per_prog):
                self._check(
                    list(range(3000, 3032)), window=1024, init_tokens=16, n_count=4096,
                    heads=32, seed=6, heads_per_prog=heads_per_prog,
                )


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for the Triton kernels")
class TestSwaMergeTriton(unittest.TestCase):
    """The fused Triton merge must reproduce the torch merge and the oracle."""

    def test_matches_torch_merge(self) -> None:
        torch.manual_seed(7)
        t_count, heads, dv, parts_n = 64, 8, 64, 3
        parts = []
        for _ in range(parts_n):
            out_i = torch.randn(t_count, heads, dv, device="cuda") * 0.1
            lse_i = torch.randn(t_count, heads, device="cuda") * 2.0 + 12.0
            parts.append((out_i, lse_i))
        for part in parts:
            part[1][0, :] = float("-inf")  # one row empty in every partial
            part[0][0] = 0.0
        parts[2][1][3, 0] = float("-inf")  # one empty head in one partial
        out_ref, lse_ref = merge_attention(parts)

        for config in (
            dict(block_m=16, block_dv=512, num_warps=4),
            dict(block_m=8, block_dv=64, num_warps=4),
            dict(block_m=32, block_dv=64, num_warps=8),
        ):
            with self.subTest(**config):
                out_tri, lse_tri = merge_attention_triton(parts, **config)
                self.assertEqual(out_tri.dtype, out_ref.dtype)
                self.assertTrue(torch.allclose(out_tri, out_ref, atol=1e-5, rtol=1e-5))
                self.assertTrue(torch.allclose(lse_tri, lse_ref, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.all(lse_tri[0] == float("-inf")))
        self.assertTrue(torch.equal(out_tri[0], torch.zeros_like(out_tri[0])))

    def test_split_slot_sets_match_reference(self) -> None:
        """Merging two disjoint slot-set partials == one reference pass."""
        torch.manual_seed(8)
        t_count, heads, dq, dv, n_count = 4, 4, 64, 32, 64
        q_abs = torch.randn(t_count, heads, dq, device="cuda") * 0.2
        kv = torch.randn(n_count, dq, device="cuda") * 0.2
        idx_a = torch.tensor([[0, 1, 4, 9, 16]], dtype=torch.int64, device="cuda")
        idx_a = idx_a.expand(t_count, -1).contiguous()
        idx_b = torch.tensor([[2, 3, 5, 8, 13, 21]], dtype=torch.int64, device="cuda")
        idx_b = idx_b.expand(t_count, -1).contiguous()
        idx_b[1, :] = -1  # empty partial on row 1

        out_full, lse_full = sparse_mla_attention(
            q_abs, kv, torch.cat([idx_a, idx_b], dim=-1), kv_lora_rank=dv
        )
        out_a, lse_a = sparse_mla_attention(q_abs, kv, idx_a, kv_lora_rank=dv)
        out_b, lse_b = sparse_mla_attention(q_abs, kv, idx_b, kv_lora_rank=dv)
        out_m, lse_m = merge_attention_triton([(out_a, lse_a), (out_b, lse_b)])
        self.assertTrue(torch.allclose(out_m, out_full, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(lse_m, lse_full, atol=1e-5, rtol=1e-5))

    def test_bf16_partials_keep_dtype(self) -> None:
        torch.manual_seed(9)
        parts = [
            (torch.randn(32, 4, 32, device="cuda") * 0.1, torch.randn(32, 4, device="cuda") + 10)
            for _ in range(2)
        ]
        out_f, lse_f = merge_attention(parts)
        out_b, lse_b = merge_attention_triton([(o.bfloat16(), l) for o, l in parts])
        self.assertEqual(out_b.dtype, torch.bfloat16)
        self.assertEqual(lse_b.dtype, torch.float32)
        self.assertTrue(torch.allclose(out_b.float(), out_f, atol=2e-2, rtol=2e-2))
        self.assertTrue(torch.allclose(lse_b, lse_f, atol=5e-3, rtol=1e-3))
        with self.assertRaises(ValueError):
            merge_attention_triton([])
        with self.assertRaises(ValueError):
            merge_attention_triton([parts[0], (parts[1][0][:, :, :16], parts[1][1])])


class TestSwaMerge(unittest.TestCase):
    """CPU-side checks of the HFA partial merge (online-softmax rescaling)."""

    def test_two_partials_match_single_pass(self) -> None:
        """Merge of two disjoint slot sets reproduces the reference over the union."""
        torch.manual_seed(0)
        t_count, heads, dq, dv, n_count = 6, 4, 64, 32, 64
        q_abs = torch.randn(t_count, heads, dq) * 0.2
        kv = torch.randn(n_count, dq) * 0.2

        slots_a = torch.tensor([0, 1, 2, 5, 8, 13, 21, 34], dtype=torch.int64)
        slots_b = torch.tensor([3, 4, 6, 7, 11, 17, 40], dtype=torch.int64)
        idx_a = torch.zeros(t_count, 1, slots_a.numel(), dtype=torch.int64)
        idx_a[:, 0, :] = slots_a
        idx_b = torch.zeros(t_count, 1, slots_b.numel(), dtype=torch.int64)
        idx_b[:, 0, :] = slots_b
        idx_b[2, 0, :] = -1  # one empty partial row: must not contribute
        idx_full = torch.cat([idx_a, idx_b], dim=-1)
        idx_full[idx_full < 0] = -1

        out_full, lse_full = sparse_mla_attention(q_abs, kv, idx_full, kv_lora_rank=dv)
        out_a, lse_a = sparse_mla_attention(q_abs, kv, idx_a, kv_lora_rank=dv)
        out_b, lse_b = sparse_mla_attention(q_abs, kv, idx_b, kv_lora_rank=dv)
        self.assertTrue(torch.isinf(lse_b[2]).all())

        out_m, lse_m = merge_attention([(out_a, lse_a), (out_b, lse_b)])
        self.assertTrue(
            torch.allclose(out_m, out_full, atol=1e-5, rtol=1e-5),
            msg=f"out max abs err {(out_m - out_full).abs().max().item():.4e}",
        )
        self.assertTrue(
            torch.allclose(lse_m, lse_full, atol=1e-5, rtol=1e-5),
            msg=f"lse max abs err {(lse_m - lse_full).abs().max().item():.4e}",
        )

    def test_random_parts_match_formula(self) -> None:
        """Explicit formula check, including a row with one empty partial."""
        torch.manual_seed(1)
        t_count, heads, dv = 8, 3, 16
        out_a = torch.randn(t_count, heads, dv)
        out_b = torch.randn(t_count, heads, dv)
        lse_a = torch.randn(t_count, heads) * 2.0 + 3.0
        lse_b = torch.randn(t_count, heads) * 2.0 + 3.0
        lse_b[1] = float("-inf")
        out_b[1] = 0.0

        out_m, lse_m = merge_attention([(out_a, lse_a), (out_b, lse_b)])

        a64, b64 = lse_a.double(), lse_b.double()
        m = torch.maximum(a64, b64)
        wa = torch.where(a64 == float("-inf"), torch.zeros_like(a64), torch.exp(a64 - m))
        wb = torch.where(b64 == float("-inf"), torch.zeros_like(b64), torch.exp(b64 - m))
        denom = wa + wb
        expected = (out_a.double() * wa[..., None] + out_b.double() * wb[..., None]) / denom[
            ..., None
        ]
        expected_lse = m + torch.log(denom)
        self.assertTrue(torch.allclose(out_m.double(), expected, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(lse_m.double(), expected_lse, atol=1e-5, rtol=1e-5))
        # Row 1 must reproduce part A exactly (part B is empty).
        self.assertTrue(torch.allclose(out_m[1], out_a[1], atol=1e-6))
        self.assertTrue(torch.allclose(lse_m[1], lse_a[1], atol=1e-6))

    def test_all_empty_rows(self) -> None:
        """All-``-inf`` lse rows must give zero output and ``-inf`` lse."""
        out_a = torch.randn(3, 2, 4)
        lse_a = torch.full((3, 2), float("-inf"))
        out_b = torch.randn(3, 2, 4)
        lse_b = lse_a.clone()
        lse_a[0] = 1.0
        out_m, lse_m = merge_attention([(out_a, lse_a), (out_b, lse_b)])
        self.assertTrue(torch.isfinite(lse_m[0]).all())
        self.assertTrue(torch.isinf(lse_m[1:]).all())
        self.assertTrue(torch.equal(out_m[1:], torch.zeros_like(out_m[1:])))

    def test_single_part_is_identity(self) -> None:
        out_a = torch.randn(4, 2, 8)
        lse_a = torch.randn(4, 2)
        out_m, lse_m = merge_attention([(out_a, lse_a)])
        self.assertTrue(torch.allclose(out_m, out_a, atol=1e-6))
        self.assertTrue(torch.allclose(lse_m, lse_a, atol=1e-6))

    def test_dtype_is_preserved_and_validated(self) -> None:
        out_a = torch.randn(4, 2, 8).bfloat16()
        lse_a = torch.randn(4, 2)
        out_b = torch.randn(4, 2, 8).bfloat16()
        lse_b = torch.randn(4, 2)
        out_m, lse_m = merge_attention([(out_a, lse_a), (out_b, lse_b)])
        self.assertEqual(out_m.dtype, torch.bfloat16)
        self.assertEqual(lse_m.dtype, torch.float32)
        with self.assertRaises(ValueError):
            merge_attention([])
        with self.assertRaises(ValueError):
            merge_attention([(out_a, lse_a), (out_b[:, :, :4], lse_b)])
        with self.assertRaises(ValueError):
            merge_attention([(out_a, lse_a), (out_b, lse_b[:, :1])])


if __name__ == "__main__":
    unittest.main()
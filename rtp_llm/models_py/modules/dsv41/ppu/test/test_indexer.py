"""Indexer scoring tests: bf16 rounding chain, causal and candidate masking.

The oracle is an independent re-implementation of rtp's reference rounding
(float64 dot -> relu -> BF16 -> * BF16 weight -> 4-lane BF16 accumulation ->
pairwise BF16 lane sums), written here from the documented algorithm.
"""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor
from rtp_llm.models_py.modules.dsv41.ppu import _quant, indexer


def _oracle_logits(q_values, k_values, weights):
    """Independent 4-lane reference (float64 math, BF16 rounding points)."""
    dots = torch.einsum("mhd,kd->mhk", q_values.double(), k_values.double())
    values = dots.relu().bfloat16().double()
    w = weights.bfloat16().double()
    lanes = [
        torch.zeros(values.shape[0], values.shape[-1], dtype=torch.float64)
        for _ in range(4)
    ]
    for head in range(q_values.shape[1]):
        lane = head % 4
        lanes[lane] = (
            (values[:, head] * w[:, head, None] + lanes[lane]).bfloat16().double()
        )
    even = (lanes[0] + lanes[2]).bfloat16().double()
    odd = (lanes[1] + lanes[3]).bfloat16().double()
    return (even + odd).bfloat16().float()


def _mask_causal(reference, ratio=1, clamp=None):
    """Apply the visible = min((pos+1)//ratio, T) mask to an oracle."""
    rows, total = reference.shape
    if clamp is None:
        clamp = total
    visible = ((torch.arange(rows) + 1) // ratio).clamp_max(clamp)
    mask = torch.arange(total)[None, :] < visible[:, None]
    return reference.masked_fill(~mask, float("-inf"))


def _make_qk(M, H, T, seed, quantize=True):
    torch.manual_seed(seed)
    q = torch.randn(M, H, 128, dtype=torch.bfloat16)
    k = torch.randn(T, 128, dtype=torch.bfloat16)
    if not quantize:
        return (
            V41QuantizedTensor(q, None, "bf16"),
            V41QuantizedTensor(k, None, "bf16"),
            q.float(),
            k.float(),
        )
    qp, qs = _quant.quantize_indexer_q(q)
    kp, ks = _quant.quantize_indexer_k_rows(k)
    return (
        V41QuantizedTensor(qp, qs, "fp4_ue8m0"),
        V41QuantizedTensor(kp, ks, "fp4_ue8m0"),
        _quant.dequantize_mx_fp4_rows(qp, qs),
        _quant.dequantize_mx_fp4_rows(kp, ks),
    )


class FullLogitsTest(unittest.TestCase):
    def test_matches_lane_algorithm_exactly(self):
        q_t, k_t, q_hat, k_hat = _make_qk(9, 32, 48, 41)
        weights = torch.randn(9, 32) / (32 * 128) ** 0.5
        ctx = SimpleNamespace(positions=torch.arange(9))
        scores = indexer.index_score(
            q_t, k_t, weights, layer_id=2, candidate_blocks=None, context=ctx
        )
        reference = _mask_causal(_oracle_logits(q_hat, k_hat, weights))
        torch.testing.assert_close(scores, reference, rtol=0, atol=0)

    def test_bf16_rounding_is_not_float64_upper_bound(self):
        # The output must carry the BF16 rounding points: it stays within a
        # tight tolerance of the float64 (unrounded) upper bound but is not
        # bit-equal to it.
        q_t, k_t, q_hat, k_hat = _make_qk(8, 32, 40, 43)
        weights = torch.randn(8, 32) / (32 * 128) ** 0.5
        ctx = SimpleNamespace(positions=torch.arange(8))
        scores = indexer.index_score(
            q_t, k_t, weights, layer_id=2, candidate_blocks=None, context=ctx
        )
        upper = (
            torch.einsum("mhd,kd->mhk", q_hat.double(), k_hat.double())
            .relu()
            .double()
            .mul(weights.double()[:, :, None])
            .sum(1)
            .float()
        )
        upper = _mask_causal(upper)
        visible = torch.isfinite(scores)
        diff = (scores - upper).abs()[visible].max()
        self.assertGreater(diff.item(), 0.0)  # rounding points are visible
        torch.testing.assert_close(
            scores[visible], upper[visible], rtol=2e-2, atol=2e-2
        )

    def test_causal_ratio_1(self):
        M, T = 12, 30
        q_t, k_t, q_hat, k_hat = _make_qk(M, 4, T, 45)
        weights = torch.randn(M, 4)
        ctx = SimpleNamespace(positions=torch.arange(M))
        scores = indexer.index_score(
            q_t, k_t, weights, layer_id=2, candidate_blocks=None, context=ctx
        )
        reference = _oracle_logits(q_hat, k_hat, weights)
        for m in range(M):
            # visible = (pos + 1) // 1 clamp_max(T)
            visible = min(m + 1, T)
            if visible < T:
                self.assertTrue(
                    torch.isinf(scores[m, visible:]).all().item()
                    and (scores[m, visible:] < 0).all().item()
                )
                scores_m = scores[m, :visible]
                ref_m = reference[m, :visible]
            else:
                scores_m, ref_m = scores[m], reference[m]
            torch.testing.assert_close(scores_m, ref_m, rtol=0, atol=0)

    def test_causal_ratio_2(self):
        M, T = 10, 25
        q_t, k_t, q_hat, k_hat = _make_qk(M, 4, T, 47)
        weights = torch.randn(M, 4)
        ctx = SimpleNamespace(positions=torch.arange(M))
        scores = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=None,
            context=ctx,
            ratio=2,
        )
        reference = _oracle_logits(q_hat, k_hat, weights)
        for m in range(M):
            visible = min((m + 1) // 2, T)
            self.assertTrue(
                torch.isinf(scores[m, visible:]).all().item()
                and (scores[m, visible:] < 0).all().item()
            )
            torch.testing.assert_close(
                scores[m, :visible], reference[m, :visible], rtol=0, atol=0
            )

    def test_plain_bf16_tensors_pass_through(self):
        q_t, k_t, q_hat, k_hat = _make_qk(5, 4, 12, 49, quantize=False)
        weights = torch.randn(5, 4)
        ctx = SimpleNamespace(positions=torch.arange(5))
        scores = indexer.index_score(
            q_t, k_t, weights, layer_id=2, candidate_blocks=None, context=ctx
        )
        reference = _oracle_logits(q_hat, k_hat, weights)
        torch.testing.assert_close(scores, _mask_causal(reference), rtol=0, atol=0)

    def test_weights_applied_as_given(self):
        # Doubling the weights must double every visible score (relu clips
        # negatives, so use positive dots by construction).
        q_t, k_t, q_hat, k_hat = _make_qk(3, 4, 6, 51)
        q_hat = q_hat.abs()
        k_hat = k_hat.abs()
        q_t = V41QuantizedTensor(q_hat.to(torch.bfloat16), None, "bf16")
        k_t = V41QuantizedTensor(k_hat.to(torch.bfloat16), None, "bf16")
        w = torch.ones(3, 4) * 0.25
        ctx = SimpleNamespace(positions=torch.arange(3))
        s1 = indexer.index_score(
            q_t, k_t, w, layer_id=2, candidate_blocks=None, context=ctx
        )
        s2 = indexer.index_score(
            q_t, k_t, w * 2, layer_id=2, candidate_blocks=None, context=ctx
        )
        self.assertTrue(torch.allclose(s1[:, :6] * 2, s2[:, :6], rtol=1e-2, atol=1e-3))

    def test_key_positions_shuffle_visibility(self):
        # Keys stored out of order: visibility follows key_positions, not the
        # row index.
        M, T = 4, 8
        q_t, k_t, q_hat, k_hat = _make_qk(M, 4, T, 53)
        perm = torch.tensor([7, 3, 0, 6, 1, 5, 2, 4])
        k_shuffled = V41QuantizedTensor(
            k_t.values[perm],
            k_t.scales[perm] if k_t.scales is not None else None,
            k_t.format,
        )
        key_positions = torch.arange(T)[perm]  # raw positions in shuffled order
        weights = torch.randn(M, 4)
        ctx = SimpleNamespace(positions=torch.arange(M))
        scores = indexer.index_score(
            q_t,
            k_shuffled,
            weights,
            layer_id=2,
            candidate_blocks=None,
            key_positions=key_positions,
            context=ctx,
        )
        reference = _oracle_logits(q_hat, k_hat[perm], weights)
        for m in range(M):
            visible_mask = (key_positions // 1) < (m + 1)
            visible = reference[m, visible_mask]
            torch.testing.assert_close(scores[m, visible_mask], visible, rtol=0, atol=0)
            self.assertTrue(
                torch.isinf(scores[m, ~visible_mask]).all().item()
                and (scores[m, ~visible_mask] < 0).all().item()
            )


class CandidateBlocksTest(unittest.TestCase):
    def _candidate_setup(self, seed, M=20, T=40, block_size=8, ratio=1):
        q_t, k_t, q_hat, k_hat = _make_qk(M, 8, T, seed)
        weights = torch.randn(M, 8) / (8 * 128) ** 0.5
        ctx = SimpleNamespace(positions=torch.arange(M))
        full = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=None,
            context=ctx,
            ratio=ratio,
        )
        return full, ctx, q_t, k_t, weights

    def test_candidate_pooled_matches_full_logits(self):
        full, ctx, q_t, k_t, weights = self._candidate_setup(55)
        M, T, bs = full.shape[0], full.shape[1], 8
        torch.manual_seed(555)
        candidates = torch.randint(-1, 5, (M, 6), dtype=torch.int32)
        candidates[:, 2] = 100  # out-of-range block id
        cand = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=candidates,
            context=ctx,
            block_size=bs,
        )
        self.assertEqual(tuple(cand.shape), (M, 6))
        for m in range(M):
            visible = min(m + 1, T)
            newest = (visible - 1) // bs if visible > 0 else -1
            for j in range(6):
                b = int(candidates[m, j])
                if b < 0 or b * bs >= T:
                    self.assertEqual(cand[m, j].item(), float("-inf"))
                elif b == newest:
                    self.assertEqual(cand[m, j].item(), float("inf"))
                else:
                    lo, hi = b * bs, min((b + 1) * bs, T)
                    entries = [i for i in range(lo, hi) if i < visible]
                    expected = (
                        full[m, entries].max().item() if entries else float("-inf")
                    )
                    self.assertEqual(cand[m, j].item(), expected, (m, j, b))

    def test_newest_visible_block_is_pinned(self):
        full, ctx, q_t, k_t, weights = self._candidate_setup(57)
        M, T, bs = full.shape[0], full.shape[1], 8
        candidates = torch.zeros((M, 1), dtype=torch.int32)
        cand = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=candidates,
            context=ctx,
            block_size=bs,
        )
        for m in range(M):
            visible = min(m + 1, T)
            newest = (visible - 1) // bs
            if newest == 0:
                self.assertEqual(cand[m, 0].item(), float("inf"))
        # block 0 is pinned for every row whose newest visible block is 0.

    def test_ratio_2_candidate_masking(self):
        full, ctx, q_t, k_t, weights = self._candidate_setup(59, ratio=2)
        M, T, bs = full.shape[0], full.shape[1], 8
        candidates = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32).repeat(M // 2, 1)
        cand = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=candidates,
            context=ctx,
            block_size=bs,
            ratio=2,
        )
        for m in range(M):
            visible = min((m + 1) // 2, T)
            newest = (visible - 1) // bs if visible > 0 else -1
            for j, b in enumerate((0, 1)):
                if b == newest:
                    self.assertEqual(cand[m, j].item(), float("inf"))
                else:
                    lo, hi = b * bs, min((b + 1) * bs, T)
                    entries = [i for i in range(lo, hi) if i < visible]
                    expected = (
                        full[m, entries].max().item() if entries else float("-inf")
                    )
                    self.assertEqual(cand[m, j].item(), expected, (m, j))

    def test_invalid_blocks_score_neg_inf(self):
        full, ctx, q_t, k_t, weights = self._candidate_setup(61)
        candidates = torch.full((full.shape[0], 3), -1, dtype=torch.int32)
        candidates[0, 0] = 1000  # beyond nblocks
        cand = indexer.index_score(
            q_t,
            k_t,
            weights,
            layer_id=2,
            candidate_blocks=candidates,
            context=ctx,
            block_size=8,
        )
        self.assertTrue(torch.isinf(cand).all().item())
        self.assertTrue((cand < 0).all().item())


class PpuKernelTest(unittest.TestCase):
    def setUp(self):
        if not _quant.is_ppu():
            self.skipTest("requires a PPU device")

    def test_triton_tile_matches_reference(self):
        torch.manual_seed(63)
        from rtp_llm.models_py.modules.dsv41.ppu import _quant as q

        k = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
        payload, sf = q.quantize_indexer_k_rows(k)
        entries = 16
        pool = torch.zeros((4, entries * 68), dtype=torch.uint8, device="cuda")
        slots = torch.arange(64, device="cuda")
        q.store_indexer_k_planar(pool, slots, payload, sf)
        qbf = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(2, 8, device="cuda") / 8
        lens = torch.tensor([64, 32], device="cuda", dtype=torch.int64)
        out = indexer.fp4_index_logits_decode_ppu(
            qbf, weights, slots.view(2, 32), lens[:1].repeat(2), pool, entries
        )
        k_hat = q.dequantize_mx_fp4_rows(payload, sf).double()
        q_hat = qbf.double()
        reference = _oracle_logits(q_hat.cpu(), k_hat[:32].cpu(), weights.cpu())
        torch.testing.assert_close(
            out[1, :32].cpu(), reference[1], rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for the PPU V4.1 MoE module (CPU-runnable; GPU tests skip off-PPU).

Covers:
  * ``sqrtsoftplus_gate`` oracle properties: hand-computed small case, bias
    used for ranking only, non-finite row fallback, sum-of-weights ≈ 1.5,
    empty batch, 8-slot padding invariance.
  * ``pad_topk_for_dispatch`` shape / -1 / 0.0 fill.
  * FP4 e2m1 dequant reference (LUT, nibble order, scale expansion).
  * ``PPUMoE`` LOCAL path vs a per-expert manual loop (random small config),
    empty-T, -1 slots produce no contribution, SwiGLU clamp, per-layer
    weight binding, shared expert + combine.
  * GPU-only: fused Triton gate vs the torch reference, and the copied
    sglang ep_scatter/ep_gather kernels round-trip.
"""

import math
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    COMM_SLOTS,
    ROUTED_SCALING_FACTOR,
    SWIGLU_LIMIT,
    PPUMoE,
    dequant_fp4_weight,
    expert_forward_reference,
    pad_topk_for_dispatch,
    preprocess_mxfp4_w4a16_mma_scales,
    shared_expert_forward,
    sqrtsoftplus_gate,
)

_FP4_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _is_ppu() -> bool:
    try:
        from rtp_llm.device.device_type import is_ppu

        return is_ppu()
    except Exception:
        return False


def manual_dequant_fp4(packed, scale_bytes):
    """Independent per-element FP4 dequant oracle.

    ``packed``: [N, K/2] uint8; ``scale_bytes``: [N, K/32] uint8.
    Low nibble -> even k, high nibble -> odd k (k // 32 picks the group).
    """
    n, packed_k = packed.shape
    k = packed_k * 2
    out = torch.zeros(n, k, dtype=torch.float32)
    for row in range(n):
        for j in range(packed_k):
            byte = int(packed[row, j])
            scale = 2.0 ** (int(scale_bytes[row, j // 16]) - 127)
            out[row, 2 * j] = _FP4_LUT[byte & 0xF] * scale
            out[row, 2 * j + 1] = _FP4_LUT[(byte >> 4) & 0xF] * scale
    return out.to(torch.bfloat16)


def manual_expert(x, w13_w, w13_s, w2_w, w2_s, limit=SWIGLU_LIMIT):
    """Independent single-expert SwiGLU forward (mirrors module semantics)."""
    w13 = manual_dequant_fp4(w13_w, w13_s)
    w2 = manual_dequant_fp4(w2_w, w2_s)
    gate_up = F.linear(x.to(torch.bfloat16), w13)
    gate, up = gate_up.chunk(2, dim=-1)
    up = up.clamp(min=-limit, max=limit)
    gate = gate.clamp(max=limit)
    hidden = F.silu(gate.float()) * up.float()
    return F.linear(hidden.to(torch.bfloat16), w2)


def manual_moe(x, ids, weights, w13_w, w13_s, w2_w, w2_s, limit=SWIGLU_LIMIT):
    """Independent per-slot manual MoE loop (the LOCAL-path oracle)."""
    t, k = x.shape
    y = torch.zeros(t, k, dtype=torch.float32)
    for row in range(t):
        for slot in range(ids.shape[1]):
            e = int(ids[row, slot])
            if e < 0:
                continue
            weight = float(weights[row, slot])
            if weight == 0.0:
                continue
            out = manual_expert(
                x[row : row + 1],
                w13_w[e],
                w13_s[e],
                w2_w[e],
                w2_s[e],
                limit=limit,
            )
            y[row] += weight * out[0].float()
    return y.to(torch.bfloat16)


def make_expert_weights(e, i, k, seed, scale_lo=124, scale_hi=132):
    g = torch.Generator().manual_seed(seed)
    w13_w = torch.randint(0, 256, (e, 2 * i, k // 2), generator=g).to(torch.uint8)
    w13_s = torch.randint(scale_lo, scale_hi, (e, 2 * i, k // 32), generator=g).to(
        torch.uint8
    )
    w2_w = torch.randint(0, 256, (e, k, i // 2), generator=g).to(torch.uint8)
    w2_s = torch.randint(scale_lo, scale_hi, (e, k, i // 32), generator=g).to(
        torch.uint8
    )
    return (w13_w, w13_s), (w2_w, w2_s)


def make_routing(t, num_experts, top_k, seed, slots=COMM_SLOTS):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, num_experts, (t, top_k), generator=g)
    weights = torch.rand(t, top_k, generator=g) + 0.1
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-12) * ROUTED_SCALING_FACTOR
    ids_p, weights_p = pad_topk_for_dispatch(
        ids.to(torch.int64), weights.to(torch.float32), slots=slots
    )
    return ids_p, weights_p


class SqrtSoftplusGateTest(unittest.TestCase):
    E = 4
    K = 8

    def _gate(self, hidden, weight, bias, top_k=2, scale=1.5, **kwargs):
        return sqrtsoftplus_gate(
            hidden, weight, bias, top_k=top_k, routed_scaling_factor=scale, **kwargs
        )

    def test_hand_computed_small_case(self):
        # logits = [0, 1, 2, 3]; bias = [0.1, -5.0, 0.0, 0.0].
        hidden = torch.ones(1, self.K, dtype=torch.bfloat16)
        weight = torch.zeros(self.E, self.K, dtype=torch.bfloat16)
        weight[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.bfloat16)
        bias = torch.tensor([0.1, -5.0, 0.0, 0.0], dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias)
        # unbiased sqrt(softplus(logits))
        expected_scores = torch.tensor(
            [math.sqrt(math.log(1.0 + math.exp(v))) for v in (0.0, 1.0, 2.0, 3.0)]
        )
        # biased ranking: expert 3 (1.7457) > expert 2 (1.4584) > e0 (0.9326)
        # > e1 (-3.8540) -> top-2 = {2, 3}.
        self.assertEqual(sorted(ids[0].tolist()), [2, 3])
        top = expected_scores[ids[0].long()]
        expected_weights = top / (top.sum() + 1e-12) * 1.5
        torch.testing.assert_close(weights[0], expected_weights, rtol=1e-4, atol=1e-5)
        self.assertAlmostEqual(weights[0].sum().item(), 1.5, delta=1e-4)

    def test_sum_of_weights_is_routed_scaling_factor(self):
        torch.manual_seed(301)
        t, e, k = 16, 32, 64
        hidden = torch.randn(t, k, dtype=torch.bfloat16)
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        bias = torch.randn(e, dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias, top_k=6, scale=1.5)
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int64)
        torch.testing.assert_close(
            weights.sum(-1), torch.full((t,), 1.5), rtol=1e-4, atol=1e-4
        )

    def test_weights_come_from_unbiased_scores(self):
        torch.manual_seed(302)
        t, e, k = 8, 16, 32
        hidden = torch.randn(t, k, dtype=torch.bfloat16)
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        logits = F.linear(hidden, weight).float()
        original = F.softplus(logits).sqrt()
        # Strong bias that completely reorders ranking vs. no bias.
        bias = torch.linspace(5.0, -5.0, e, dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias, top_k=6)
        unbiased_scores = original.gather(1, ids)
        expected = (
            unbiased_scores / (unbiased_scores.sum(-1, keepdim=True) + 1e-12) * 1.5
        )
        torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-6)
        # And a different bias must actually change the ranking.
        ids_nobias, _ = self._gate(hidden, weight, None, top_k=6)
        self.assertFalse(torch.equal(ids, ids_nobias))

    def test_expert_axis_permutation_keeps_weights(self):
        # Relabeling experts (router weight rows AND bias entries together)
        # relabels ids but must not change the weight attached to each
        # underlying expert: weights come from un-biased scores only.
        torch.manual_seed(303)
        t, e, k = 12, 24, 32
        hidden = torch.randn(t, k, dtype=torch.bfloat16)
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        bias = torch.randn(e, dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias, top_k=6)
        perm = torch.randperm(e)
        ids2, weights2 = self._gate(hidden, weight[perm], bias[perm], top_k=6)
        inverse = torch.empty(e, dtype=torch.long)
        inverse[perm] = torch.arange(e)
        for row in range(t):
            self.assertEqual(set(inverse[ids[row]].tolist()), set(ids2[row].tolist()))
            w1 = {
                int(i): float(weights[row, s]) for s, i in enumerate(ids[row].tolist())
            }
            # New expert i2 corresponds to old expert perm[i2].
            w2 = {
                int(perm[i]): float(weights2[row, s])
                for s, i in enumerate(ids2[row].tolist())
            }
            self.assertEqual(set(w1), set(w2))
            for expert, value in w1.items():
                self.assertAlmostEqual(value, w2[expert], places=5)

    def test_nonfinite_row_gets_uniform_fallback(self):
        torch.manual_seed(304)
        t, e, k = 4, 16, 32
        hidden = torch.randn(t, k, dtype=torch.bfloat16)
        hidden[1] = float("nan")
        hidden[2] = float("inf")
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        bias = torch.randn(e, dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias, top_k=6)
        for good in (0, 3):
            self.assertTrue(
                torch.isfinite(weights[good]).all() and weights[good].sum() != 0
            )
        for bad in (1, 2):
            torch.testing.assert_close(
                weights[bad],
                torch.full((6,), 1.5 / 6),
                rtol=0,
                atol=0,
            )
            # ids stay valid integers for dispatch.
            self.assertTrue((ids[bad] >= 0).all() and (ids[bad] < e).all())

    def test_nonfinite_bias_row_falls_back(self):
        e, k = 16, 32
        hidden = torch.randn(2, k, dtype=torch.bfloat16)
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        bias = torch.full((e,), -float("inf"), dtype=torch.float32)
        _, weights = self._gate(hidden, weight, bias, top_k=6)
        torch.testing.assert_close(
            weights, torch.full_like(weights, 1.5 / 6), rtol=0, atol=0
        )

    def test_empty_batch_returns_empty(self):
        hidden = torch.zeros(0, 8, dtype=torch.bfloat16)
        weight = torch.randn(16, 8, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.float32)
        ids, weights = self._gate(hidden, weight, bias, top_k=6)
        self.assertEqual(tuple(ids.shape), (0, 6))
        self.assertEqual(ids.dtype, torch.int64)
        self.assertEqual(tuple(weights.shape), (0, 6))
        self.assertEqual(weights.dtype, torch.float32)

    def test_image_rows_have_no_effect_without_vl_bias(self):
        # rtp gate.py only diverges for image tokens when bias_vl exists.
        torch.manual_seed(305)
        t, e, k = 6, 16, 32
        hidden = torch.randn(t, k, dtype=torch.bfloat16)
        weight = torch.randn(e, k, dtype=torch.bfloat16)
        bias = torch.randn(e, dtype=torch.float32)
        image_rows = torch.tensor([True, False] * 3)
        ids_a, weights_a = self._gate(hidden, weight, bias, top_k=6)
        ids_b, weights_b = self._gate(
            hidden, weight, bias, top_k=6, image_rows=image_rows
        )
        torch.testing.assert_close(weights_a, weights_b, rtol=0, atol=0)
        self.assertTrue(torch.equal(ids_a, ids_b))


class PadTopkForDispatchTest(unittest.TestCase):
    def test_pads_six_to_eight_with_sentinels(self):
        ids = torch.arange(12).reshape(2, 6).to(torch.int64)
        weights = torch.ones(2, 6, dtype=torch.float32)
        ids_p, weights_p = pad_topk_for_dispatch(ids, weights)
        self.assertEqual(tuple(ids_p.shape), (2, 8))
        self.assertEqual(tuple(weights_p.shape), (2, 8))
        torch.testing.assert_close(ids_p[:, :6], ids, rtol=0, atol=0)
        torch.testing.assert_close(weights_p[:, :6], weights, rtol=0, atol=0)
        self.assertTrue((ids_p[:, 6:] == -1).all())
        self.assertTrue((weights_p[:, 6:] == 0.0).all())

    def test_passthrough_when_already_at_slot_width(self):
        ids = torch.zeros(3, 8, dtype=torch.int64)
        weights = torch.ones(3, 8, dtype=torch.float32)
        ids_p, weights_p = pad_topk_for_dispatch(ids, weights)
        self.assertIs(ids_p, ids)
        self.assertIs(weights_p, weights)

    def test_custom_slot_width(self):
        ids = torch.zeros(5, 2, dtype=torch.int64)
        weights = torch.ones(5, 2, dtype=torch.float32)
        ids_p, _ = pad_topk_for_dispatch(ids, weights, slots=4)
        self.assertEqual(tuple(ids_p.shape), (5, 4))
        self.assertTrue((ids_p[:, 2:] == -1).all())


class FP4DequantTest(unittest.TestCase):
    def test_nibble_order_low_is_even_k(self):
        # byte 0x21: low nibble 1 -> k=0 value 0.5; high nibble 2 -> k=1 = 1.0
        packed = torch.tensor([[0x21]], dtype=torch.uint8)
        scale = torch.full((1, 1), 127, dtype=torch.uint8)
        out = dequant_fp4_weight(packed, scale).float()
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(out[0, 0].item(), 0.5)
        self.assertEqual(out[0, 1].item(), 1.0)

    def test_negative_nibbles(self):
        # byte 0x98: low nibble 8 -> -0.0; high nibble 9 -> -0.5
        packed = torch.tensor([[0x98]], dtype=torch.uint8)
        scale = torch.full((1, 1), 127, dtype=torch.uint8)
        out = dequant_fp4_weight(packed, scale).float()
        self.assertEqual(out[0, 0].item(), 0.0)
        self.assertEqual(out[0, 1].item(), -0.5)

    def test_full_lut(self):
        bytes_ = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]
        packed = torch.tensor([bytes_], dtype=torch.uint8)
        scale = torch.full((1, 1), 127, dtype=torch.uint8)
        out = dequant_fp4_weight(packed, scale).float().flatten().tolist()
        expected = []
        for byte in bytes_:
            expected.extend([_FP4_LUT[byte & 0xF], _FP4_LUT[(byte >> 4) & 0xF]])
        self.assertEqual(out, expected)
        # All 16 codes covered.
        self.assertEqual(sorted(out), sorted(_FP4_LUT))

    def test_scale_group_expansion(self):
        # K=64 -> 2 groups; group 0 scale 2^(130-127)=8, group 1 scale 1.
        packed = torch.full((1, 32), 0x11, dtype=torch.uint8)
        scale = torch.tensor([[130, 127]], dtype=torch.uint8)
        out = dequant_fp4_weight(packed, scale).float()
        self.assertTrue((out[:, :32] == 0.5 * 8).all())
        self.assertTrue((out[:, 32:] == 0.5).all())

    def test_int8_packed_bytes(self):
        packed = torch.tensor([[0x21]], dtype=torch.uint8).to(torch.int8)
        scale = torch.full((1, 1), 127, dtype=torch.uint8)
        out = dequant_fp4_weight(packed, scale).float()
        self.assertEqual(out[0, 0].item(), 0.5)
        self.assertEqual(out[0, 1].item(), 1.0)

    def test_e8m0fnu_scale_dtype(self):
        packed = torch.tensor([[0x11]], dtype=torch.uint8)
        scale = torch.tensor([[130]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        out = dequant_fp4_weight(packed, scale).float()
        self.assertEqual(out[0, 0].item(), 0.5 * 8)

    def test_matches_manual_on_random_tensors(self):
        torch.manual_seed(401)
        packed = torch.randint(0, 256, (9, 48), dtype=torch.int64).to(torch.uint8)
        scale = torch.randint(118, 138, (9, 3), dtype=torch.int64).to(torch.uint8)
        actual = dequant_fp4_weight(packed, scale)
        expected = manual_dequant_fp4(packed, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class PreprocessW4A16ScalesTest(unittest.TestCase):
    def test_is_a_byte_permutation(self):
        torch.manual_seed(402)
        scales = torch.randint(0, 256, (2, 64, 2), dtype=torch.int64).to(torch.uint8)
        out = preprocess_mxfp4_w4a16_mma_scales(scales)
        self.assertEqual(tuple(out.shape), (2, 1, 128))
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(
            sorted(out.flatten().tolist()), sorted(scales.flatten().tolist())
        )

    def test_rejects_non_multiple_of_64(self):
        scales = torch.zeros(1, 32, 1, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            preprocess_mxfp4_w4a16_mma_scales(scales)


class ExpertReferenceTest(unittest.TestCase):
    E, I, K = 4, 64, 32

    def _weights(self, seed, scale_lo=124, scale_hi=132):
        return make_expert_weights(self.E, self.I, self.K, seed, scale_lo, scale_hi)

    def test_matches_manual(self):
        (w13_w, w13_s), (w2_w, w2_s) = self._weights(501)
        x = torch.randn(5, self.K, dtype=torch.bfloat16)
        actual = expert_forward_reference(x, w13_w[0], w13_s[0], w2_w[0], w2_s[0])
        expected = manual_expert(x, w13_w[0], w13_s[0], w2_w[0], w2_s[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_swiglu_clamp_changes_out_of_range_output(self):
        # Large UE8M0 exponents push gate/up far past the limit of 10.
        (w13_w, w13_s), (w2_w, w2_s) = self._weights(502, scale_lo=138, scale_hi=142)
        x = torch.randn(3, self.K, dtype=torch.bfloat16)
        clamped = expert_forward_reference(
            x, w13_w[0], w13_s[0], w2_w[0], w2_s[0], swiglu_limit=SWIGLU_LIMIT
        )
        # Manual unclamped forward.
        w13 = manual_dequant_fp4(w13_w[0], w13_s[0])
        w2 = manual_dequant_fp4(w2_w[0], w2_s[0])
        gate_up = F.linear(x, w13)
        gate, up = gate_up.chunk(2, dim=-1)
        # Sanity: the raw activations really exceed the clamp.
        self.assertGreater(gate_up.float().abs().max().item(), 10.0)
        unclamped = F.linear((F.silu(gate.float()) * up.float()).to(torch.bfloat16), w2)
        self.assertFalse(
            torch.allclose(clamped.float(), unclamped.float(), rtol=1e-2, atol=1e-2)
        )
        # And the clamped result matches the manual clamped forward.
        expected = manual_expert(x, w13_w[0], w13_s[0], w2_w[0], w2_s[0])
        torch.testing.assert_close(clamped, expected, rtol=0, atol=0)


class PPUMoELocalTest(unittest.TestCase):
    E, I, K = 8, 64, 32

    def _make_moe(self, seed, scale_lo=124, scale_hi=132, swiglu_limit=SWIGLU_LIMIT):
        moe = PPUMoE(
            num_routed_experts=self.E,
            num_local_experts=self.E,
            ep_rank=0,
            ep_size=1,
            hidden_size=self.K,
            inter_dim=self.I,
            swiglu_limit=swiglu_limit,
        )
        (w13_w, w13_s), (w2_w, w2_s) = make_expert_weights(
            self.E, self.I, self.K, seed, scale_lo, scale_hi
        )
        moe.bind_expert_weights(0, (w13_w, w13_s), (w2_w, w2_s))
        return moe, (w13_w, w13_s), (w2_w, w2_s)

    def test_local_path_matches_manual_loop(self):
        moe, (w13_w, w13_s), (w2_w, w2_s) = self._make_moe(601)
        t = 7
        x = torch.randn(t, self.K, dtype=torch.bfloat16)
        ids, weights = make_routing(t, self.E, 6, seed=602)
        actual = moe(x, ids, weights)
        expected = manual_moe(x, ids, weights, w13_w, w13_s, w2_w, w2_s)
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=1e-2
        )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(tuple(actual.shape), (t, self.K))

    def test_empty_tokens_returns_empty(self):
        moe, _, _ = self._make_moe(603)
        x = torch.empty(0, self.K, dtype=torch.bfloat16)
        ids = torch.zeros(0, COMM_SLOTS, dtype=torch.int64)
        weights = torch.zeros(0, COMM_SLOTS, dtype=torch.float32)
        out = moe(x, ids, weights)
        self.assertEqual(tuple(out.shape), (0, self.K))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_minus_one_slots_contribute_nothing(self):
        moe, (w13_w, w13_s), (w2_w, w2_s) = self._make_moe(604)
        t = 5
        x = torch.randn(t, self.K, dtype=torch.bfloat16)
        ids, weights = make_routing(t, self.E, 6, seed=605)
        # Same routing, but the padding slots carry a nonzero weight: the -1
        # sentinel must still produce no contribution.
        weights_dirty = weights.clone()
        weights_dirty[:, 6:] = 0.7
        out_clean = moe(x, ids, weights)
        out_dirty = moe(x, ids, weights_dirty)
        torch.testing.assert_close(out_clean.float(), out_dirty.float(), rtol=0, atol=0)
        # And zero-weight real slots contribute nothing either.
        weights_zeroed = weights.clone()
        weights_zeroed[:, 0] = 0.0
        ids_zeroed = ids.clone()
        out_zero = moe(x, ids_zeroed, weights_zeroed)
        expected = manual_moe(x, ids_zeroed, weights_zeroed, w13_w, w13_s, w2_w, w2_s)
        torch.testing.assert_close(
            out_zero.float(), expected.float(), rtol=1e-2, atol=1e-2
        )

    def test_output_is_linear_in_router_weights(self):
        moe, _, _ = self._make_moe(606)
        t = 4
        x = torch.randn(t, self.K, dtype=torch.bfloat16)
        ids, weights = make_routing(t, self.E, 6, seed=607)
        y1 = moe(x, ids, weights).float()
        y2 = moe(x, ids, weights * 2.0).float()
        torch.testing.assert_close(y2, y1 * 2.0, rtol=1e-2, atol=1e-2)
        y0 = moe(x, ids, weights * 0.0).float()
        torch.testing.assert_close(y0, torch.zeros_like(y0), rtol=0, atol=0)

    def test_bind_validation(self):
        moe, (w13_w, w13_s), (w2_w, w2_s) = self._make_moe(608)
        with self.assertRaises(ValueError):
            moe.bind_expert_weights(1, (w13_w[:-1], w13_s[:-1]), (w2_w, w2_s))
        with self.assertRaises(TypeError):
            moe.bind_expert_weights(1, (w13_w.float(), w13_s), (w2_w, w2_s))
        fresh = PPUMoE(
            num_routed_experts=self.E,
            num_local_experts=self.E,
            ep_size=1,
            hidden_size=self.K,
            inter_dim=self.I,
        )
        with self.assertRaises(RuntimeError):
            fresh(
                torch.randn(2, self.K, dtype=torch.bfloat16),
                torch.zeros(2, 8, dtype=torch.int64),
                torch.ones(2, 8),
            )

    def test_per_layer_weight_binding(self):
        moe, (w13_a, w13_s_a), (w2_a, w2_s_a) = self._make_moe(609)
        (w13_b, w13_s_b), (w2_b, w2_s_b) = make_expert_weights(
            self.E, self.I, self.K, 610
        )
        moe.bind_expert_weights(1, (w13_b, w13_s_b), (w2_b, w2_s_b))
        t = 3
        x = torch.randn(t, self.K, dtype=torch.bfloat16)
        ids, weights = make_routing(t, self.E, 6, seed=611)
        y0 = moe(x, ids, weights, layer_id=0).float()
        y1 = moe(x, ids, weights, layer_id=1).float()
        expected0 = manual_moe(x, ids, weights, w13_a, w13_s_a, w2_a, w2_s_a).float()
        expected1 = manual_moe(x, ids, weights, w13_b, w13_s_b, w2_b, w2_s_b).float()
        torch.testing.assert_close(y0, expected0, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y1, expected1, rtol=1e-2, atol=1e-2)
        self.assertFalse(torch.equal(y0, y1))


class SharedExpertTest(unittest.TestCase):
    def test_shared_expert_forward_matches_manual(self):
        torch.manual_seed(701)
        i, k = 64, 32
        w13 = PPUResidentBF16Linear(torch.randn(2 * i, k, dtype=torch.bfloat16))
        w2 = PPUResidentBF16Linear(torch.randn(k, i, dtype=torch.bfloat16))
        x = torch.randn(6, k, dtype=torch.bfloat16)
        actual = shared_expert_forward(x, w13, w2)
        gate_up = w13(x)
        gate, up = gate_up.chunk(2, dim=-1)
        up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        gate = gate.clamp(max=SWIGLU_LIMIT)
        hidden = F.silu(gate.float()) * up.float()
        expected = w2(hidden.to(torch.bfloat16))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_expert_clamp_active_on_large_activations(self):
        torch.manual_seed(702)
        i, k = 64, 32
        w13 = PPUResidentBF16Linear(torch.randn(2 * i, k, dtype=torch.bfloat16) * 50)
        w2 = PPUResidentBF16Linear(torch.randn(k, i, dtype=torch.bfloat16))
        x = torch.randn(2, k, dtype=torch.bfloat16)
        gate_up = w13(x)
        self.assertGreater(gate_up.float().abs().max().item(), 10.0)
        clamped = shared_expert_forward(x, w13, w2)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate.float()) * up.float()
        unclamped = w2(hidden.to(torch.bfloat16))
        self.assertFalse(
            torch.allclose(clamped.float(), unclamped.float(), rtol=1e-2, atol=1e-2)
        )

    def test_combine_routed_and_shared_is_plain_add(self):
        from rtp_llm.models_py.modules.dsv41.ppu.moe import (
            combine_routed_and_shared,
        )

        torch.manual_seed(703)
        routed = torch.randn(4, 32, dtype=torch.bfloat16)
        shared = torch.randn(4, 32, dtype=torch.bfloat16)
        out = combine_routed_and_shared(routed, shared)
        torch.testing.assert_close(
            out, (routed.float() + shared.float()).to(torch.bfloat16), rtol=0, atol=0
        )


@unittest.skipUnless(_is_ppu(), "requires PPU device")
class FusedGateTritonTest(unittest.TestCase):
    def test_fused_gate_matches_torch_reference(self):
        from rtp_llm.models_py.modules.dsv41.ppu.moe import (
            _fused_sqrtsoftplus_gate_triton,
        )

        torch.manual_seed(801)
        t, e, k = 64, 384, 512
        hidden = torch.randn(t, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(e, k, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(e, dtype=torch.float32, device="cuda")
        logits = F.linear(hidden, weight).float().contiguous()
        fused_w, fused_i = _fused_sqrtsoftplus_gate_triton(
            logits, bias.contiguous(), topk=6, routed_scaling_factor=1.5
        )
        torch_w, torch_i = sqrtsoftplus_gate(
            hidden, weight, bias, top_k=6, routed_scaling_factor=1.5
        )
        self.assertTrue(
            all(
                set(a.tolist()) == set(b.tolist())
                for a, b in zip(fused_i.long().cpu(), torch_i.cpu())
            )
        )
        torch.testing.assert_close(fused_w, torch_w, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            fused_w.sum(-1), torch.full((t,), 1.5, device="cuda"), rtol=1e-4, atol=1e-4
        )


@unittest.skipUnless(_is_ppu(), "requires PPU device")
class EpScatterGatherTritonTest(unittest.TestCase):
    def test_scatter_gather_identity_roundtrip(self):
        from rtp_llm.models_py.modules.dsv41.ppu.moe import (
            ep_gather,
            ep_scatter_sail,
        )

        torch.manual_seed(802)
        num_experts, hidden, t, slots = 4, 128, 33, 8
        g = torch.Generator(device="cuda").manual_seed(802)
        x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, num_experts, (t, slots), generator=g, device="cuda").to(
            torch.int64
        )
        # Punch -1 holes (padding slots).
        ids[:, 6:] = -1
        weights = torch.rand(t, slots, generator=g, device="cuda", dtype=torch.float32)
        weights[:, 6:] = 0.0
        counts = torch.bincount(ids.flatten().clamp(min=0), minlength=num_experts).to(
            torch.int32
        )
        all_tokens = int(counts.sum().item())
        input_tensor = torch.empty(
            (all_tokens, hidden), dtype=torch.bfloat16, device="cuda"
        )
        m_indices = torch.empty((all_tokens,), dtype=torch.int32, device="cuda")
        output_index = torch.empty((t, slots), dtype=torch.int32, device="cuda")
        expert_start_loc = torch.empty_like(counts)
        ep_scatter_sail(
            x,
            ids.to(torch.int32),
            counts,
            expert_start_loc,
            input_tensor,
            m_indices,
            output_index,
            block_e=1,
        )
        # m_indices groups rows by expert in count order.
        expected_m = torch.cat(
            [
                torch.full((int(c),), e, dtype=torch.int32, device="cuda")
                for e, c in enumerate(counts.tolist())
            ]
        )
        torch.testing.assert_close(m_indices, expected_m, rtol=0, atol=0)
        # Identity "expert": gather straight back and compare with the
        # per-slot weighted sum of the input rows.
        gather_out = torch.empty((t, hidden), dtype=torch.bfloat16, device="cuda")
        ep_gather(input_tensor, ids.to(torch.int32), weights, output_index, gather_out)
        expected = ((x.float().unsqueeze(1) * weights.unsqueeze(-1)).sum(1)).to(
            torch.bfloat16
        )
        torch.testing.assert_close(
            gather_out.float(), expected.float(), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()

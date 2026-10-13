"""gpt-oss math-contract unit tests (independent oracles).

Each test re-derives the reference math from the HF modeling contract
(transformers models/gpt_oss/modeling_gpt_oss.py, verified against
openai/gpt-oss-20b @ 6cee5e81) and checks the RTP wiring pieces against it
with independent encode/decode helpers — the oracles deliberately do NOT
reuse the production decode paths.

Covers:
  1. MXFP4 layout through ``deinterleave_stack_moe_w1`` /
     ``flatten_stack_moe``: E2M1 nibbles (low=even K, high=odd K), UE8M0
     group-32 scales, gate/up row interleave -> [gate; up]. Exact
     byte-level checks (pure rearrangement, no numeric change).
  2. ``silu_mul_fp8_quant_packed`` gpt-oss activation variant
     (alpha=1.702, up_shift=1.0, clamp 7) vs the fp32 reference, and the
     default (SiLU) bit-compat.
  3. ``Gate`` ``topk_softmax`` routing (raw biased logits -> top-4 ->
     softmax over the selected) vs eager.
  4. Attention-sink equivalence: softmax with a denominator-only extra
     column equals O * sigmoid(LSE - c) for per-head c.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed,
)
from rtp_llm.utils.model_weight import (
    W,
    deinterleave_stack_moe_w1,
    flatten_stack_moe,
)

# E2M1 magnitudes by 3-bit code; sign bit is the top nibble bit.
E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _random_mxfp4(e: int, n: int, k: int, seed: int):
    """Ground-truth checkpoint tensors: codes (per element, 0..15) + per
    (row, group) UE8M0 exponent bytes. Returns the 4D blocks, the 2D
    scales, and the decoded fp32 weights [e, n, k]."""
    torch.manual_seed(seed)
    assert k % 32 == 0
    g = k // 32
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8)
    exps = torch.randint(120, 135, (e, n, g), dtype=torch.uint8)
    # pack pairs: byte j = (code_{2j+1} << 4) | code_{2j}
    packed = (codes[..., 1::2].to(torch.int16) << 4) | codes[..., 0::2].to(
        torch.int16
    )
    packed = packed.to(torch.uint8)
    blocks = packed.reshape(e, n, g, 16)
    mags = E2M1_MAGS
    sign = torch.where((codes & 8) > 0, -1.0, 1.0)
    decoded = mags[(codes & 7).long()] * sign
    scales = torch.pow(2.0, exps.to(torch.float32) - 127.0)
    w = decoded.reshape(e, n, g, 32) * scales.unsqueeze(-1)
    return blocks, exps, w.reshape(e, n, k)


def _decode_packed(packed: torch.Tensor, exps: torch.Tensor):
    """Independent decoder for [.., K/2] packed + [.., K/32] exponent bytes."""
    low = (packed & 0xF).long()
    high = (packed >> 4).long()
    mags = E2M1_MAGS
    lo = mags[low & 7] * torch.where((low & 8) > 0, -1.0, 1.0)
    hi = mags[high & 7] * torch.where((high & 8) > 0, -1.0, 1.0)
    out = torch.stack([lo, hi], dim=-1).flatten(-2)  # [.., K]
    scales = torch.pow(2.0, exps.to(torch.float32) - 127.0)
    k = out.shape[-1]
    g = k // 32
    out = out.reshape(*out.shape[:-1], g, 32) * scales.unsqueeze(-1)
    return out.reshape(*out.shape[:-2], k)


class GptOssMxfp4InterleaveTest(unittest.TestCase):
    def test_deinterleave_exact_rearrangement(self):
        e, n, k = 3, 8, 64
        blocks, exps, w = _random_mxfp4(e, n, k, seed=7)
        # per-expert slices, as the stacked loader hands to process_fun
        w_loaded = deinterleave_stack_moe_w1([blocks[i] for i in range(e)])
        s_loaded = deinterleave_stack_moe_w1([exps[i] for i in range(e)])

        self.assertEqual(w_loaded.shape, (e, n, k // 2))
        self.assertEqual(s_loaded.shape, (e, n, k // 32))

        decoded = _decode_packed(w_loaded, s_loaded)
        # reference: even rows are gate, odd rows are up; loader packs
        # [gate; up] so row r < n/2 of the loaded tensor is original row 2r
        expected = torch.cat([w[:, 0::2, :], w[:, 1::2, :]], dim=1)
        self.assertTrue(torch.equal(decoded, expected))
        expected_scales = torch.cat([exps[:, 0::2, :], exps[:, 1::2, :]], dim=1)
        self.assertTrue(torch.equal(s_loaded, expected_scales))

    def test_flatten_stack_moe_exact(self):
        e, n, k = 2, 6, 64
        blocks, exps, w = _random_mxfp4(e, n, k, seed=9)
        out = flatten_stack_moe([blocks[i] for i in range(e)])
        self.assertEqual(out.shape, (e, n, k // 2))
        decoded = _decode_packed(out, exps)
        self.assertTrue(torch.equal(decoded, w))

    def test_bias_deinterleave(self):
        bias = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
        # per-expert 1D slices, as the stacked loader provides them
        out = deinterleave_stack_moe_w1([bias[i] for i in range(4)])
        self.assertEqual(out.shape, (4, 6))
        expected = torch.cat([bias[:, 0::2], bias[:, 1::2]], dim=1)
        self.assertTrue(torch.equal(out, expected))


def _group_quant_fp8(x: torch.Tensor, group: int = 128):
    """Reference per-group E4M3 dequantized output: quantize with power-of-2
    scales (amax/448, ceil-log2 — same derivation as the kernel) then
    multiply the scale back, so the comparison measures only the activation
    math plus rounding, not the scale layout. Rounds through bf16 first,
    mirroring the kernel's documented legacy-precision behavior."""
    x = x.to(torch.bfloat16).to(torch.float32)
    out = torch.empty_like(x)
    for i in range(0, x.shape[1], group):
        seg = x[:, i : i + group]
        absmax = seg.abs().amax(dim=1, keepdim=True).clamp_min(1e-10)
        expo = torch.ceil(torch.log2(absmax / 448.0))
        scale = torch.pow(2.0, expo)
        q = (seg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
        out[:, i : i + group] = q * scale
    return out


def _unpack_packed_scale(s_packed: torch.Tensor, num_groups: int):
    """[M, num_packed_groups] int32 -> per-group fp32 scale [M, num_groups]."""
    m, packs = s_packed.shape
    scales = []
    for p in range(packs):
        for k in range(4):
            gid = p * 4 + k
            if gid < num_groups:
                exponent = ((s_packed[:, p] >> (8 * k)) & 0xFF).float() - 127.0
                scales.append(torch.pow(2.0, exponent))
    return torch.stack(scales, dim=1)


class GptOssSwigluKernelTest(unittest.TestCase):
    def _run_variant(self, gate, up, limit, alpha, shift):
        gate_up = torch.cat([gate, up], dim=1).contiguous()
        h_fp8, h_scale = silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=limit,
            group_size=128,
            act_alpha=alpha,
            up_shift=shift,
        )
        inter = gate.shape[1]
        scales = _unpack_packed_scale(h_scale, inter // 128)  # [M, groups]
        # dequantized kernel output
        decoded = h_fp8.float()
        decoded = decoded.reshape(decoded.shape[0], -1, 128) * scales.unsqueeze(-1)
        return decoded.reshape(decoded.shape[0], inter)

    def _reference(self, gate, up, limit, alpha, shift):
        g = gate.float().clamp(max=limit) if limit > 0 else gate.float()
        u = up.float()
        if limit > 0:
            u = u.clamp(min=-limit, max=limit)
        return (g * torch.sigmoid(alpha * g)) * (u + shift)

    def test_quick_gelu_variant_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel")
        torch.manual_seed(11)
        m, inter, limit, alpha, shift = 256, 256, 7.0, 1.702, 1.0
        gate = (torch.randn(m, inter, device="cuda") * 6).bfloat16()
        up = (torch.randn(m, inter, device="cuda") * 6).bfloat16()
        got = self._run_variant(gate, up, limit, alpha, shift)
        ref = self._reference(gate, up, limit, alpha, shift)
        ref_q = _group_quant_fp8(ref)
        diff = (got - ref_q).abs()
        denom = ref_q.abs().clamp_min(1e-3)
        self.assertTrue(
            torch.allclose(got, ref_q, atol=0.05, rtol=0.05),
            f"max rel diff {(diff / denom).max().item():.4f}",
        )

    def test_defaults_are_silu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for triton kernel")
        torch.manual_seed(13)
        gate = (torch.randn(128, 128, device="cuda") * 3).bfloat16()
        up = (torch.randn(128, 128, device="cuda") * 3).bfloat16()
        got = self._run_variant(gate, up, 0.0, 1.0, 0.0)
        ref = torch.nn.functional.silu(gate.float()) * up.float()
        ref_q = _group_quant_fp8(ref)
        self.assertTrue(torch.allclose(got, ref_q, atol=0.05, rtol=0.05))


class GptOssGateTest(unittest.TestCase):
    def _gate(self, weight, bias):
        layer_weights = {
            W.moe_gate: weight,
            W.moe_gate_bias: bias,
        }
        return Gate(
            layer_id=0,
            dim=4,
            n_routed_experts=8,
            n_activated_experts=4,
            score_func="topk_softmax",
            route_scale=1.0,
            n_hash_layers=0,
            vocab_size=0,
            layer_weights=layer_weights,
        )

    def test_topk_softmax_matches_eager(self):
        torch.manual_seed(3)
        weight = torch.randn(8, 4)
        bias = torch.randn(8)
        x = torch.randn(64, 4)
        gate = self._gate(weight, bias)
        weights, indices = gate(x)
        # mirror the gate's own precision path (bf16 GEMM, fp32 epilogue):
        # this test checks the topk_softmax SEMANTICS; bf16-vs-fp32 GEMM
        # noise can flip near-tied expert ORDER, which is not under test.
        logits = torch.nn.functional.linear(
            x.to(torch.bfloat16), weight.to(torch.bfloat16)
        ).float()
        biased = logits + bias
        ref_idx = biased.topk(4, dim=-1)[1]
        ref_w = torch.softmax(biased.gather(1, ref_idx), dim=-1)
        self.assertTrue(torch.equal(indices, ref_idx))
        self.assertTrue(
            torch.allclose(weights, ref_w, atol=1e-6),
            f"max diff {(weights - ref_w).abs().max().item()}",
        )
        self.assertTrue(
            torch.allclose(weights.sum(-1), torch.ones(64), atol=1e-6)
        )

    def test_topk_softmax_nonfinite_row_uniform(self):
        weight = torch.eye(8, 4)
        bias = torch.zeros(8)
        gate = self._gate(weight, bias)
        x = torch.zeros(2, 4)
        x[0, 0] = float("nan")
        weights, indices = gate(x)
        self.assertTrue(torch.allclose(weights[0], torch.full((4,), 0.25)))


class GptOssSinkEquivalenceTest(unittest.TestCase):
    def test_sigmoid_lse_correction_equals_denominator_column(self):
        torch.manual_seed(5)
        heads, kv = 3, 64
        logits = torch.randn(heads, kv) * 3.0
        sinks = torch.randn(heads)
        v = torch.randn(heads, kv, 8)
        # reference: softmax over [logits | sink] with the sink column dropped
        combined = torch.cat([logits, sinks[:, None]], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = torch.softmax(combined, dim=-1)
        out_ref = torch.einsum("hk,hkd->hd", probs[:, :-1], v)
        # LSE form: O_std * sigmoid(LSE - c)
        lse = torch.logsumexp(logits, dim=-1)
        probs_std = torch.softmax(logits, dim=-1)
        out_std = torch.einsum("hk,hkd->hd", probs_std, v)
        out_corr = out_std * torch.sigmoid(lse - sinks)[:, None]
        self.assertTrue(
            torch.allclose(out_ref, out_corr, atol=1e-5),
            f"max diff {(out_ref - out_corr).abs().max().item():.2e}",
        )


if __name__ == "__main__":
    unittest.main()

"""Diagnostic streaming FP4 -> FP8 weights with SM90 DeepGEMM FP8 MMA.

Resident weights remain Humming packed MXFP4. A uses the original RTP g128
payload unchanged. A/B scale formats match SM90 DeepGEMM's (1,128,128)
recipe; B is a LOSSLESS representation change for the guarded checkpoint range,
not a new weight quantizer. FP8 MMA numerical acceptance is separately tested.
No production switches, full-model caches, or relaxed correctness thresholds.
"""

import json

import pytest
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_deepgemm_candidates import (
    HUMMING_LAYOUT_VERSION,
    _StreamingDeepGemm,
    _metadata,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input,
    _independent_weight,
    _metrics,
    _quantize,
)


@triton.jit
def _humming_to_fp8(W, S, SF, B, N: tl.constexpr, K: tl.constexpr,
                    NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < NUMEL
    k = index % K
    n = (index // K) % N
    expert = index // (N * K)
    word_index = (
        (expert * (K // 16) + k // 16) * (2 * N)
        + (n // 64) * 128 + (n % 8) * 16
        + ((k % 8) // 2) * 4 + (n % 64) // 16
    )
    slot = (k % 2) * 4 + ((k // 8) % 2) * 2 + ((n // 8) % 2)
    word = tl.load(W + word_index, valid, other=0).to(tl.uint32)
    code = (word >> (4 * slot)) & 15
    magnitude = code & 7
    value = tl.where(
        magnitude < 4, magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude < 6, magnitude - 2, (magnitude - 4) * 2).to(tl.float32),
    )
    scale_index = (
        (expert * (K // 32) + k // 32) * N
        + (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
    )
    exponent = tl.load(S + scale_index, valid, other=127).to(tl.uint32)
    block_scale = tl.load(
        SF + (expert * (N // 128) + n // 128) * (K // 128) + k // 128,
        valid, other=1.0,
    )
    residual_exponent = (block_scale.to(tl.uint32, bitcast=True) >> 23) & 255
    # B_q = E2M1 * 2**(original_exponent - block_max_exponent + 6).
    # For exponent span <=11: nonzero values stay in [2**-6, 384],
    # normal E4M3 with enough mantissa bits for every E2M1 value.
    shifted_scale = ((exponent - residual_exponent + 127) << 23).to(tl.float32, bitcast=True)
    positive = (value * shifted_scale).to(tl.float8e4nv)
    # Preserve code8 negative zero after conversion, not via float algebra.
    bits = positive.to(tl.uint8, bitcast=True) | ((code & 8).to(tl.uint8) << 4)
    tl.store(B + index, bits.to(tl.float8e4nv, bitcast=True), valid)


class _StreamingFP8(_StreamingDeepGemm):
    """Reuse only the BF16 diagnostic's checked layout and padded route metadata."""

    def __init__(self, linear, counts):
        super().__init__(linear, counts)
        assert self.n % 128 == 0 and self.k % 128 == 0
        e = len(counts)
        # N permutation is contained within64 rows, so block128 extrema are
        # unchanged by Humming's scale permutation. This small immutable table
        # is computed once; no resident raw or expanded weight copy is retained.
        blocks = self.scales.view(torch.uint8).reshape(e, self.k // 128, 4, self.n // 128, 128)
        maximum = blocks.amax(dim=(2, 4)).to(torch.int16)
        minimum = blocks.amin(dim=(2, 4)).to(torch.int16)
        assert bool(((maximum - minimum) <= 11).all())
        # Audited real checkpoint range. Refuse unvalidated exponent extremes.
        assert bool((minimum >= 117).all() and (maximum <= 127).all())
        self.weight_scales = torch.pow(2.0, maximum.float() - 133).transpose(1, 2).contiguous()

    def decode_weight(self):
        b = torch.empty(
            (len(self.counts), self.n, self.k), device=self.weight.device,
            dtype=torch.float8_e4m3fn,
        )
        _humming_to_fp8[(triton.cdiv(b.numel(), 1024),)](
            self.weight, self.scales.view(torch.uint8), self.weight_scales, b,
            self.n, self.k, b.numel(), 1024, enable_fp_fusion=False,
        )
        return b

    def prepare_a(self, x):
        q, packed = _quantize(x)
        scales = unpack_activation_scales(packed, self.k // 128)
        # Byte copies avoid unsupported FP8 scatter kernels and preserve payload.
        padded_bytes = torch.zeros(
            (self.padded_m, self.k), device=x.device, dtype=torch.uint8
        )
        padded_bytes.index_copy_(0, self.positions, q.view(torch.uint8))
        padded_scales = torch.zeros(
            (self.padded_m, self.k // 128), device=x.device, dtype=torch.float32
        )
        padded_scales.index_copy_(0, self.positions, scales)
        return padded_bytes.view(torch.float8_e4m3fn), padded_scales

    def __call__(self, x):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            m_grouped_fp8_fp4_gemm_nt_contiguous,
        )

        a = self.prepare_a(x)
        b = self.decode_weight()
        output = x.new_empty((self.padded_m, self.n))
        # Generic API accepts FP8/FP8 on SM90; no packed FP4 is passed to GEMM.
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, (b, self.weight_scales), output, self.indices,
            recipe=(1, 128, 128), disable_ue8m0_cast=True,
        )
        return output.index_select(0, self.positions)


def _run(n, k, m, layout, *, experts=48, benchmark=True):
    import triton.testing

    torch.manual_seed(20261016)
    weight = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
    scales = torch.randint(117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    counts = [0] * (experts - 1) + [m] if layout == "hot" else [
        m // experts + int(e < m % experts) for e in range(experts)
    ]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    baseline = SM90HummingGroupedLinear(weight, scales)
    candidate = _StreamingFP8(baseline, counts)
    q_ref, s_ref = _independent_quantized_input(x)
    padded_q, padded_s = candidate.prepare_a(x)
    torch.testing.assert_close(
        padded_q.view(torch.uint8).index_select(0, candidate.positions),
        q_ref.view(torch.uint8), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        padded_s.index_select(0, candidate.positions).view(torch.int32),
        s_ref.view(torch.int32), rtol=0, atol=0,
    )
    del padded_q, padded_s
    reference = x.new_empty((m, n))
    b = candidate.decode_weight()
    for expert, count in enumerate(counts):
        independent_w = _independent_weight(weight[expert], scales[expert])
        # Check every expert, not only experts receiving rows. Reconstruction
        # happens one expert at a time solely in this untimed independent oracle.
        sf = candidate.weight_scales[expert].repeat_interleave(128, 0).repeat_interleave(128, 1)
        reconstructed = b[expert].float() * sf
        torch.testing.assert_close(reconstructed.view(torch.int32), independent_w.view(torch.int32), rtol=0, atol=0)
        expected_payload = (independent_w / sf).to(torch.float8_e4m3fn)
        torch.testing.assert_close(b[expert].view(torch.uint8), expected_payload.view(torch.uint8), rtol=0, atol=0)
        if count:
            begin, end = host_offsets[expert:expert + 2]
            independent_x = (
                q_ref[begin:end].float().reshape(count, k // 128, 128)
                * s_ref[begin:end].unsqueeze(-1)
            ).reshape(count, k)
            reference[begin:end] = (independent_x @ independent_w.T).bfloat16()
            del independent_x
        del independent_w, sf, reconstructed, expected_payload
    del b, weight, scales, q_ref, s_ref
    baseline_output = baseline(x, offsets)
    result = candidate(x)
    error = _metrics(result, reference)
    baseline_error = _metrics(baseline_output, reference)
    repeatable = all(torch.equal(candidate(x), result) for _ in range(3))
    timings = {"humming_full_ms": [], "fp8_streaming_full_ms": []}
    if benchmark:
        for repeat in range(3):
            jobs = [("humming_full_ms", lambda: baseline(x, offsets)),
                    ("fp8_streaming_full_ms", lambda: candidate(x))]
            if repeat % 2:
                jobs.reverse()
            for name, fn in jobs:
                timings[name].append(triton.testing.do_bench(fn, warmup=20, rep=100, return_mode="median"))
    record = {
        "candidate": "streaming_humming_mxfp4_to_fp8_deepgemm_sm90",
        "E": experts, "M": m, "N": n, "K": k, "layout": layout,
        "M_padded": candidate.padded_m, "expert_counts": counts,
        "weight_layout_version": HUMMING_LAYOUT_VERSION,
        "recipe": [1, 128, 128], "weight_decode_bitwise": True,
        "activation_payload_and_scale_bitwise": True,
        "resident_extra_weight_scale_bytes": candidate.weight_scales.numel() * 4,
        "temporary_fp8_weight_bytes": experts * n * k,
        "timed_scope": "RTPg128 quant + FP32 scale decode + pad + E48 packed-to-FP8 + DG FP8 MMA + unpad",
        "untimed_scope": "fixed route metadata and tiny immutable B scale table",
        "error_vs_independent_reference": error,
        "baseline_error_vs_independent_reference": baseline_error,
        "error_vs_humming": _metrics(result, baseline_output),
        "repeatable": repeatable,
        "decision": "BOUNDED_SUPPORT" if error["outside_existing_tolerance"] == 0 and repeatable else "REJECT",
        "scope": "projection diagnostic only; routed pre-down weighting/full model remain untested",
        **timings,
    }
    print("SM90_MOE_DEEPGEMM_FP8 " + json.dumps(record), flush=True)
    print("SM90_MOE_DEEPGEMM_FP8_SOURCE " + json.dumps(_metadata()), flush=True)
    assert baseline_error["outside_existing_tolerance"] == 0, baseline_error
    # This is a hard numerical gate, including for the initial small fixture.
    assert error["outside_existing_tolerance"] == 0 and repeatable, record


def _with_fp32_oracle(n, k, m, layout, *, experts=48, benchmark=True):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run(n, k, m, layout, experts=experts, benchmark=benchmark)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90",
)


def test_sm90_fp8_small_fixture():
    _with_fp32_oracle(128, 128, 4, "balanced", experts=8, benchmark=False)


@pytest.mark.parametrize("projection,n,k", [("w13", 4608, 5120), ("down", 5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,layout", [(1, "hot"), (4096, "balanced"), (32768, "hot"), (32768, "balanced")], ids=["decode", "prefill4096", "hot32768", "balanced32768"])
def test_sm90_fp8_realshape(projection, n, k, m, layout):
    _with_fp32_oracle(n, k, m, layout)

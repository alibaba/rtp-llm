"""Test-only streaming MXFP4 decode + SM90 DeepGEMM BF16 grouped candidate.

No full-model BF16 weight cache and no production backend selection. Each timed
candidate call includes weight decoding, original RTP g128 activation quantize /
decode, expert padding, GEMM, and unpadding. Passing this diagnostic is NOT MoE
or model acceptance. The original .02/.05 numerical gate is never relaxed.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    decode_quantized_activation,
    unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input,
    _independent_weight,
    _metrics,
    _quantize,
)


@triton.jit
def _decode_fp4_weight_kernel(W, S, D, NUMEL: tl.constexpr, BLOCK: tl.constexpr):
    """Logical [E,N,K] decode, bounded E8M0 scale range checked by test."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < NUMEL
    packed = tl.load(W + index // 2, valid, other=0).to(tl.uint8)
    code = (packed >> ((index % 2) * 4)) & 15
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude < 6, magnitude - 2, (magnitude - 4) * 2).to(tl.float32),
    )
    exponent = tl.load(S + index // 32, valid, other=127).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    # Set sign after ALL floating-point arithmetic/conversion. A float select
    # between +/-0 can be folded by compiler optimizations; E2M1 code 8 must
    # retain the exact BF16 0x8000 representation for the unchanged bitwise gate.
    magnitude_bits = (value * scale).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits = magnitude_bits | ((code & 8).to(tl.uint16) << 12)
    tl.store(D + index, bits.to(tl.bfloat16, bitcast=True), valid)


def _decode_weight(weight, scales):
    experts, n, half_k = weight.shape
    decoded = torch.empty(
        (experts, n, half_k * 2), dtype=torch.bfloat16, device=weight.device
    )
    _decode_fp4_weight_kernel[(triton.cdiv(decoded.numel(), 1024),)](
        weight, scales.view(torch.uint8), decoded, decoded.numel(), 1024,
        enable_fp_fusion=False,
    )
    return decoded


HUMMING_LAYOUT_VERSION = "sm90_wgmma_a16_w4_interleave3_transpose_miniblock_v1"


@triton.jit
def _decode_humming_weight_kernel(
    W, S, D, N: tl.constexpr, K: tl.constexpr, NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inverse of Humming process.cuh weight_repack_nk<4,16,...,true,...>
    # and humming_pack_weight interleave_mode=3. No raw-packed intermediate.
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
    code = (word >> (slot * 4)) & 15
    magnitude = code & 7
    value = tl.where(
        magnitude < 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(magnitude < 6, magnitude - 2, (magnitude - 4) * 2).to(tl.float32),
    )
    # transform_humming_weight_scale: transpose NK -> KN, then permute
    # each 64-wide N block [0,8,...,56,1,9,...,57,...,7,15,...,63].
    scale_index = (
        (expert * (K // 32) + k // 32) * N
        + (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
    )
    exponent = tl.load(S + scale_index, valid, other=127).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    # Integer sign insertion avoids losing E2M1 negative zero in float algebra.
    magnitude_bits = (value * scale).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    bits = magnitude_bits | ((code & 8).to(tl.uint16) << 12)
    tl.store(D + index, bits.to(tl.bfloat16, bitcast=True), valid)


def _check_humming_layout(linear):
    from humming import dtypes
    from humming.config import MmaType

    config = linear.config
    assert config.mma_type == MmaType.WGMMA
    assert config.a_dtype == dtypes.bfloat16 and config.b_dtype == dtypes.float4e2m1
    assert config.bs_dtype == dtypes.float8e8m0
    assert config.is_group_weight_scale and config.weight_scale_group_size == 32
    assert not config.should_apply_bs_on_c
    assert not config.use_fused_e8m0_scale and not config.use_int_weight_scale
    assert not config.use_packed_k_layout and not config.use_native_dequant
    assert not config.has_zero_point and not config.has_bias
    assert not config.pad_shape_n and not config.pad_shape_k
    assert getattr(linear, "weight_scale_2", None) is None
    e, n, k = config.num_experts, config.shape_n, config.shape_k
    assert n % 64 == 0 and k % 128 == 0
    assert linear.weight.dtype == torch.int32
    assert linear.weight_scale.dtype == torch.float8_e8m0fnu
    assert linear.weight.is_contiguous() and linear.weight_scale.is_contiguous()
    assert linear.weight.shape == (e, k // 16, n * 2)
    assert linear.weight_scale.shape == (e, k // 32, n)
    return e, n, k


def _decode_humming_weight(weight, scales, n, k):
    decoded = torch.empty(
        (weight.shape[0], n, k), dtype=torch.bfloat16, device=weight.device
    )
    _decode_humming_weight_kernel[(triton.cdiv(decoded.numel(), 1024),)](
        weight, scales.view(torch.uint8), decoded, n, k, decoded.numel(), 1024,
        enable_fp_fusion=False,
    )
    return decoded


class _StreamingDeepGemm:
    def __init__(self, linear, counts):
        import deep_gemm

        experts, self.n, self.k = _check_humming_layout(linear)
        assert len(counts) == experts
        # Share precisely the baseline's resident buffers, never a raw copy.
        self.weight, self.scales = linear.weight, linear.weight_scale
        weight = self.weight
        self.counts = counts
        self.alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
        assert self.alignment > 0
        # No block may straddle experts: metadata alone cannot fix unpadded A.
        aligned = [triton.cdiv(count, self.alignment) * self.alignment for count in counts]
        self.padded_m = sum(aligned)
        self.indices = torch.full(
            (self.padded_m,), -1, dtype=torch.int32, device=weight.device
        )
        positions, begin = [], 0
        for expert, (count, padded) in enumerate(zip(counts, aligned)):
            self.indices[begin:begin + count] = expert
            positions.extend(range(begin, begin + count))
            begin += padded
        self.positions = torch.tensor(positions, dtype=torch.int64, device=weight.device)

    def __call__(self, x):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            m_grouped_bf16_gemm_nt_contiguous,
        )

        payload, packed = _quantize(x)
        decoded_x = decode_quantized_activation(payload, packed)
        padded_x = x.new_zeros((self.padded_m, x.shape[1]))
        padded_x.index_copy_(0, self.positions, decoded_x)
        # This allocation is intentionally inside every timed invocation.
        decoded_weight = _decode_humming_weight(self.weight, self.scales, self.n, self.k)
        padded_output = x.new_empty((self.padded_m, self.n))
        m_grouped_bf16_gemm_nt_contiguous(
            padded_x, decoded_weight, padded_output, self.indices
        )
        # Padding outputs may be unspecified. Never feed them into reduction.
        return padded_output.index_select(0, self.positions)


def _metadata():
    import deep_gemm
    import humming

    package = Path(deep_gemm.__file__).parent
    humming_package = Path(humming.__file__).parent
    paths = [
        package / "include/deep_gemm/impls/sm90_bf16_gemm.cuh",
        package / "csrc/apis/gemm.hpp",
        package / "csrc/jit_kernels/impls/sm90_bf16_gemm.hpp",
        humming_package / "transform.py",
        humming_package / "include/humming/kernel/process.cuh",
        humming_package / "kernel/repack_weight.py",
    ]
    return {
        "deep_gemm_version": importlib.metadata.version("deep-gemm"),
        "deep_gemm_package": str(package),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()
        },
        "humming_version": importlib.metadata.version("humming-kernels"),
        "torch": torch.__version__,
    }


def _run(n, k, m, layout):
    import triton.testing

    experts = 48
    torch.manual_seed(20261015)
    weight = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
    scales = torch.randint(
        117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    counts = ([0] * (experts - 1) + [m]) if layout == "hot" else [
        m // experts + int(expert < m % experts) for expert in range(experts)
    ]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)

    # Independent g128 payload and scale oracle, before testing GEMM arithmetic.
    q_ref, s_ref = _independent_quantized_input(x)
    q, packed = _quantize(x)
    torch.testing.assert_close(q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(unpack_activation_scales(packed, k // 128), s_ref, rtol=0, atol=0)
    decoded_x = decode_quantized_activation(q, packed)
    reference = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    baseline = SM90HummingGroupedLinear(weight, scales)
    candidate = _StreamingDeepGemm(baseline, counts)
    # Validate every decoded weight, including empty experts, without keeping
    # a second E48 FP32/INT64-expanded weight tensor resident.
    decoded_weight = _decode_humming_weight(baseline.weight, baseline.weight_scale, n, k)
    for expert, count in enumerate(counts):
        independent_w = _independent_weight(weight[expert], scales[expert])
        torch.testing.assert_close(decoded_weight[expert].float(), independent_w, rtol=0, atol=0)
        # Compare BF16 bits as well, including E2M1's negative zero. The raw
        # decoder is an untimed independent control, at most one expert wide.
        expected_bits = independent_w.bfloat16().view(torch.int16)
        torch.testing.assert_close(decoded_weight[expert].view(torch.int16), expected_bits, rtol=0, atol=0)
        raw_control = _decode_weight(weight[expert:expert + 1], scales[expert:expert + 1])
        torch.testing.assert_close(raw_control[0].view(torch.int16), expected_bits, rtol=0, atol=0)
        del raw_control, expected_bits
        if count:
            begin, end = host_offsets[expert:expert + 2]
            independent_x = (
                q_ref[begin:end].float().reshape(count, k // 128, 128)
                * s_ref[begin:end].unsqueeze(-1)
            ).reshape(count, k)
            torch.testing.assert_close(decoded_x[begin:end].float(), independent_x, rtol=0, atol=0)
            # TF32 disabled by caller; independent FP32 matmul then BF16 RN.
            reference[begin:end] = (independent_x @ independent_w.T).bfloat16()
            del independent_x
        del independent_w
    del decoded_weight, decoded_x, q_ref, s_ref, q, packed
    # Raw checkpoint tensors are oracle fixtures only; drop even that test-only
    # copy before timing so the candidate's residency matches the proposal.
    del weight, scales
    assert candidate.weight.data_ptr() == baseline.weight.data_ptr()
    assert candidate.scales.data_ptr() == baseline.weight_scale.data_ptr()
    baseline_output = baseline(x, offsets)
    result = candidate(x)
    baseline_error = _metrics(baseline_output, reference)
    error = _metrics(result, reference)
    versus_baseline = _metrics(result, baseline_output)
    repeatable = all(torch.equal(candidate(x), result) for _ in range(3))
    timings = {"humming_quant_decode_gemm_ms": [], "deepgemm_full_streaming_ms": []}
    for repeat in range(3):
        work = [
            ("humming_quant_decode_gemm_ms", lambda: baseline(x, offsets)),
            ("deepgemm_full_streaming_ms", lambda: candidate(x)),
        ]
        if repeat % 2:
            work.reverse()
        for name, fn in work:
            timings[name].append(triton.testing.do_bench(fn, warmup=20, rep=100, return_mode="median"))

    # Report allocator peak separately; no empty_cache or cross-process changes.
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    peak_output = candidate(x)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - allocated_before
    del peak_output
    eligible = error["outside_existing_tolerance"] == 0 and repeatable
    record = {
        "candidate": "deepgemm_bf16_streaming_humming_layout_decode", "E": experts,
        "M": m, "N": n, "K": k, "layout": layout, "expert_counts": counts,
        "M_padded": candidate.padded_m, "M_alignment": candidate.alignment,
        "temporary_bf16_weight_bytes": experts * n * k * 2,
        "weight_layout_version": HUMMING_LAYOUT_VERSION,
        "weight_layout_guards": "WGMMA/A16/W4/g32E8M0/interleave3; no fused,int,native,packedK,ZP,bias,padding,scale2",
        "resident_raw_weight_copy": False,
        "shared_humming_buffers": True,
        "weight_decode_check": "all E48 elements bitwise BF16 versus independent raw oracle and raw decoder",
        "peak_extra_allocated_bytes": peak_extra,
        "timed_scope": "g128 quant/decode + padding + ALL E48 Humming-layout weight decode + GEMM + unpadding",
        "untimed_scope": "static expert-id/position metadata and Humming initialization",
        "error_vs_independent_reference": error,
        "baseline_error_vs_independent_reference": baseline_error,
        "error_vs_humming": versus_baseline,
        "bitwise_equal_to_humming": bool(torch.equal(result.view(torch.int16), baseline_output.view(torch.int16))),
        "repeatable": repeatable,
        "decision": "BOUNDED_SUPPORT" if eligible else "REJECT",
        "scope": "synthetic checkpoint-range projection only; no routed MoE or model acceptance",
        **timings,
    }
    print("SM90_MOE_DEEPGEMM_CANDIDATE " + json.dumps(record), flush=True)
    print("SM90_MOE_DEEPGEMM_SOURCE " + json.dumps(_metadata()), flush=True)
    assert baseline_error["outside_existing_tolerance"] == 0, baseline_error
    # Experimental rejection is explicit diagnostic output, not a changed gate.
    return record


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90",
)
@pytest.mark.parametrize("projection,n,k", [("w13", 4608, 5120), ("down", 5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,layout", [(4096, "balanced"), (32768, "hot")], ids=["prefill4096", "hot32768"])
def test_sm90_moe_deepgemm_candidates(projection, n, k, m, layout):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run(n, k, m, layout)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

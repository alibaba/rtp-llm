"""SM90 FP8 arithmetic localization and an unfused-weight-scale candidate.

Unlike the rejected fused g32-promotion experiment, the new path leaves the
E2M1 values unscaled inside FP8 MMA and applies each original g32 weight scale
to its partial result in FP32. Activations STILL use RTP g128/eps=1e-4; g32
metadata variants repeat scales without requantizing any activation value.
Only dependency APIs are used. No production dispatch or precision gate changes.
"""

import dataclasses
import bisect
import json
import math

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _Candidate,
    _independent_quantized_input,
    _independent_weight,
    _metrics,
    _quantize,
    _source_metadata,
)


_MODES = (
    ("fused_g128", True, 128),
    ("fused_g32_metadata_only", True, 32),
    ("generic_g128", False, 128),
    ("generic_g32_metadata_only", False, 32),
)

_MMA_MODES = (
    ("wgmma_generic_g128_control", False, 128, "wgmma"),
    ("mma_generic_g128", False, 128, "mma"),
    ("mma_generic_g32_metadata_only", False, 32, "mma"),
)


class _FP8Adapter(_Candidate):
    def __init__(self, weight, scales, *, fused, metadata_group, mma_type="wgmma"):
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors

        experts, n, packed_k = weight.shape
        config = prepare_layer_config(
            shape_n=n, shape_k=2 * packed_k, num_experts=experts,
            weight_schema=HummingWeightSchema(
                b_dtype="float4e2m1", bs_dtype="float8e8m0",
                weight_scale_group_size=32, weight_scale_type="group",
            ),
            input_schema=HummingInputSchema(
                a_dtype="float8e4m3", input_scale_dtype="float32",
                input_scale_group_size=metadata_group,
            ),
            torch_dtype=torch.bfloat16, device=weight.device,
        )
        if not fused:
            # Rebuild rather than mutating a cached config string. Undo the
            # secondary tensor-scale contract installed by fused normalization.
            config = dataclasses.replace(
                config, use_fused_e8m0_scale=False,
                weight_scale_2_type="none", use_packed_k_layout=False,
            )
        if mma_type != "wgmma":
            # The MMA path needs a distinct transformed weight/scale layout.
            # Rebuild BEFORE transform; never reuse WGMMA-packed tensors.
            config = dataclasses.replace(config, mma_type=mma_type)
        raw = scales.view(torch.uint8).flatten(1).to(torch.int16)
        assert bool(((raw.amax(1) - raw.amin(1)) <= 11).all())
        linear = torch.nn.Module()
        linear.config = config
        tensors = transform_humming_tensors(config, {
            "weight": weight.contiguous().clone().view(torch.int32),
            "weight_scale": scales.contiguous().clone(),
        })
        for name, tensor in tensors.items():
            linear.register_buffer(name, tensor)
        linear.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=weight.device))
        tuning = None
        if mma_type == "mma":
            # H20's default tiny-M heuristic chooses warpM=8, which would emit
            # an unsupported m8n8k16 FP8 MMA. Use the existing native m16n8k32
            # family with a multiple-of-16 warpM and unchanged K64 promotion.
            tuning = dict(
                block_shape=(32, 128, 64), warp_shape=(32, 32, 64),
                num_stages=3, num_ctas_per_sm=2,
                use_stream_k=False, use_tma=False, use_warp_spec=False,
                use_mbarrier=False,
                num_sms=torch.cuda.get_device_properties(weight.device).multi_processor_count,
            )
        super().__init__(linear, activation_bits=8, batch_invariant=True, tuning=tuning)
        self.metadata_group = metadata_group

    def prepare_input(self, payload, packed):
        scale = unpack_activation_scales(packed, payload.shape[1] // 128)
        if self.metadata_group == 32:
            scale = scale.repeat_interleave(4, dim=-1).contiguous()
        return payload, scale


def _weights(experts, n, k):
    weight = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
    raw = torch.randint(117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
    # Every expert really exercises the endpoints, including the short K128
    # case. No reliance on a stochastic narrow-span draw.
    raw[:, :, 0] = 117
    raw[:, :, 1] = 127
    return weight, raw.view(torch.float8_e8m0fnu)


def _check_quantization(x):
    q, packed = _quantize(x)
    q_ref, scale_ref = _independent_quantized_input(x)
    torch.testing.assert_close(q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        unpack_activation_scales(packed, x.shape[1] // 128), scale_ref, rtol=0, atol=0,
    )
    qx = (q_ref.float().reshape(x.shape[0], -1, 128) * scale_ref.unsqueeze(-1)).reshape_as(x)
    return qx


@pytest.fixture(autouse=True)
def _strict_fp32_oracle():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; run through GPU-locked Bazel target",
)


@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_actual_span10_weight_basis(n, k):
    """Every transformed coefficient is read by exactly one nonzero product."""
    torch.manual_seed(20261015)
    weight, scales = _weights(1, n, k)
    x = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    _check_quantization(x)
    expected = _independent_weight(weight[0], scales[0]).T.bfloat16()
    offsets = torch.tensor([0, k], device="cuda", dtype=torch.int32)
    failures = []
    for name, fused, metadata_group in _MODES:
        candidate = _FP8Adapter(weight, scales, fused=fused, metadata_group=metadata_group)
        actual = candidate(x, offsets)
        error = _metrics(actual, expected)
        # Signed-zero normalization in Humming is not a changed nonzero weight.
        exact = bool(torch.equal(actual, expected))
        print("SM90_FP8_WEIGHT_BASIS " + json.dumps({
            "mode": name, "N": n, "K": k, "E": 1,
            "scale_range": [117, 127], "exact_coefficients": exact,
            "error": error, "layer_config": candidate.linear.config.to_str(),
        }), flush=True)
        if not exact:
            failures.append(name)
        del candidate
    assert not failures, f"Weight conversion/repacking not isolated: {failures}"


def _mismatched_dot_anchors(actual, reference, qx, weight, scales, host_offsets, active_k):
    """Compare the actual mismatching coordinates, not just a fixed sample."""
    anchors = []
    for row, col in (actual != reference).nonzero()[:4].tolist():
        expert = bisect.bisect_right(host_offsets, row) - 1
        w = _independent_weight(weight[expert, col:col + 1], scales[expert, col:col + 1]).flatten()
        a_cpu, b_cpu = qx[row].double().cpu(), w.double().cpu()
        dot_fsum = math.fsum((a_cpu * b_cpu).tolist())
        rounded = torch.tensor(dot_fsum, dtype=torch.float64).bfloat16().item()
        record = {
            "row": row, "column": col, "expert": expert,
            "actual_bf16": actual[row, col].item(),
            "fp32_oracle_bf16": reference[row, col].item(),
            "cpu_fsum_products": dot_fsum, "cpu_fsum_bf16": rounded,
            "actual_equals_fsum_bf16": actual[row, col].item() == rounded,
        }
        if active_k and active_k <= 128:
            # Exact, portable minimal-dot operands for reproducing MMA errors.
            record["decoded_quantized_a"] = a_cpu[:active_k].tolist()
            record["decoded_weight_b"] = b_cpu[:active_k].tolist()
        anchors.append(record)
    return anchors


def _random_diagnosis(n, k, m, experts, active_k, *, mma_compare=False):
    import triton.testing

    torch.manual_seed(20261015)
    weight, scales = _weights(experts, n, k)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    if active_k:
        x[:, active_k:] = 0
    counts = [m // experts + int(i < m % experts) for i in range(experts)]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    qx = _check_quantization(x)
    reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    # FP64 checks full-K dot products on fixed rows/columns, not an approximate
    # reduced-K oracle. All-output FP32 remains the original acceptance gate.
    anchor_rows, anchor_expected, anchor_fp32 = [], [], []
    for expert, count in enumerate(counts):
        if not count:
            continue
        begin, end = host_offsets[expert:expert + 2]
        w = _independent_weight(weight[expert], scales[expert])
        fp32 = qx[begin:end] @ w.T
        reference[begin:end] = fp32.bfloat16()
        if expert < 2:
            rows = list(range(begin, min(end, begin + 4)))
            anchor_rows.extend(rows)
            anchor_expected.append((qx[rows].double() @ w[:64].double().T).bfloat16())
            anchor_fp32.append(fp32[:len(rows), :64].bfloat16())
    anchor_expected = torch.cat(anchor_expected)
    anchor_fp32 = torch.cat(anchor_fp32)
    print("SM90_FP8_ORACLE_ANCHOR " + json.dumps({
        "M": m, "N": n, "K": k, "active_k": active_k,
        "rows": anchor_rows, "columns": 64,
        "fp32_vs_fp64": _metrics(anchor_fp32, anchor_expected),
    }), flush=True)
    torch.testing.assert_close(anchor_fp32, anchor_expected, rtol=.02, atol=.05)
    baseline = SM90HummingGroupedLinear(weight, scales)
    baseline_output = baseline(x, offsets)
    torch.testing.assert_close(baseline_output, reference, rtol=.02, atol=.05)
    promoted = []
    modes = _MMA_MODES if mma_compare else tuple((name, fused, group, "wgmma") for name, fused, group in _MODES)
    for name, fused, metadata_group, mma_type in modes:
        candidate = _FP8Adapter(weight, scales, fused=fused, metadata_group=metadata_group, mma_type=mma_type)
        actual = candidate(x, offsets)
        error = _metrics(actual, reference)
        anchor_error = _metrics(actual[anchor_rows, :64], anchor_expected)
        repeatable = all(torch.equal(candidate(x, offsets), actual) for _ in range(3))
        latencies = {"baseline_chain_ms": [], "candidate_chain_ms": []}
        for repeat in range(3 if m >= 4096 else 1):
            cases = [("baseline_chain_ms", lambda: baseline(x, offsets)),
                     ("candidate_chain_ms", lambda: candidate(x, offsets))]
            if repeat % 2:
                cases.reverse()
            for label, fn in cases:
                latencies[label].append(triton.testing.do_bench(
                    fn, warmup=20, rep=100, return_mode="median",
                ))
        accepted = error["outside_existing_tolerance"] == 0 and anchor_error["outside_existing_tolerance"] == 0 and repeatable
        if not fused and accepted and (not mma_compare or mma_type == "mma"):
            promoted.append(name)
        print("SM90_FP8_ARITHMETIC_DIAGNOSIS " + json.dumps({
            "mode": name, "M": m, "N": n, "K": k, "E": experts,
            "active_k": active_k, "quantizer": "unchanged g128 eps=1e-4",
            "error_vs_fp32_bf16": error, "anchors_vs_fp64_bf16": anchor_error,
            "error_vs_w4a16": _metrics(actual, baseline_output),
            "bitwise_equal_vs_w4a16": bool(torch.equal(actual.view(torch.int16), baseline_output.view(torch.int16))),
            "mismatched_dot_anchors": _mismatched_dot_anchors(actual, reference, qx, weight, scales, host_offsets, active_k),
            "repeatable": repeatable,
            "decision": "BOUNDED_SUPPORT" if accepted else "REJECT",
            "scope": "component tolerance only; original MoE intermediate bitwise gates/full smoke NOT_RUN",
            "layer_config": candidate.linear.config.to_str(), **latencies,
        }), flush=True)
        del candidate
    print("SM90_FP8_SOURCE " + json.dumps(_source_metadata()), flush=True)
    # This is deliberately a hard unchanged numerical gate for the NEW path,
    # not a test that passes merely because it successfully measured failures.
    assert promoted, "No tested candidate passed the unchanged gate (MMA required for explicit MMA comparison)"


@pytest.mark.parametrize("active_k", [32, 64, 128], ids=["one_mma32", "two_mma64", "four_mma128"])
def test_isolate_fp8_partial_accumulation(active_k):
    _random_diagnosis(128, 128, 17, 2, active_k)


@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,experts", [(1, 1), (17, 2), (4096, 48)], ids=["decode1", "small17", "prefill4096"])
def test_generic_w4a8_actual_shape_acceptance(n, k, m, experts):
    _random_diagnosis(n, k, m, experts, active_k=0)


@pytest.mark.parametrize("active_k", [32, 64, 128], ids=["one_mma32", "two_mma64", "four_mma128"])
def test_mma_sync_small_fp8_comparison(active_k):
    _random_diagnosis(128, 128, 17, 2, active_k, mma_compare=True)


@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m,experts", [(1, 1), (17, 2), (4096, 48)], ids=["decode1", "small17", "prefill4096"])
def test_mma_sync_actual_shape_acceptance(n, k, m, experts):
    _random_diagnosis(n, k, m, experts, active_k=0, mma_compare=True)

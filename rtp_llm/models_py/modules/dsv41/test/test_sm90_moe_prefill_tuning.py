"""Bounded large-M SM90 FP8 MMA.sync scheduling sweep, test-only.

The current W4A16 production chain is the baseline. All FP8 candidates keep
original g128 quantized activation values and g32 MXFP4 weights; only existing
Humming tile/pipeline options vary. Component tolerance is not full MoE/smoke
acceptance. Known rejected WGMMA FP8 and no-gain W4A16 TMA sweeps are not repeated.
"""

import collections
import bisect
import json
import math
import os
import statistics
import traceback
from pathlib import Path

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import SM90HummingGroupedLinear
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_fp8_diagnosis import (
    _FP8Adapter,
    _check_quantization,
    _weights,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _Candidate,
    _independent_weight,
    _metrics,
    _source_metadata,
)


# (M,N,K), (warpM,warpN,warpK), stages, CTAs/SM, TMA+warp-specialization.
# All native MMA warpM are multiples of 16; no unsupported FP8 m8n8k16.
_TILES = (
    ("m32n128_s3c2_control", (32, 128, 64), (32, 32, 64), 3, 2, False),
    ("m16n128_s4c3", (16, 128, 64), (16, 32, 64), 4, 3, False),
    ("m32n256_s4c2", (32, 256, 64), (32, 64, 64), 4, 2, False),
    ("m64n128_s4c2", (64, 128, 64), (32, 32, 64), 4, 2, False),
    ("m64n256_s4c1", (64, 256, 64), (32, 64, 64), 4, 1, False),
    ("m32n128_wm16_s3c2", (32, 128, 64), (16, 32, 64), 3, 2, False),
    ("m64n128_tma_ws_s4c1", (64, 128, 64), (32, 32, 64), 4, 1, True),
    ("m64n256_tma_ws_s4c1", (64, 256, 64), (32, 64, 64), 4, 1, True),
)


def _make_tuning(tile, device):
    _, block, warp, stages, ctas, tma_ws = tile
    return dict(
        block_shape=block, warp_shape=warp, num_stages=stages,
        num_ctas_per_sm=ctas, use_stream_k=False, use_cp_async=True,
        use_tma=tma_ws, use_tma_a=tma_ws, use_tma_b=tma_ws,
        use_tma_c=False, use_tma_as=False, use_tma_as2=False,
        use_tma_bs=False, use_tma_bs2=False, use_tma_bzp=False, use_tma_bias=False,
        use_warp_spec=tma_ws, use_mbarrier=tma_ws,
        num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
    )


def _union_ms(intervals):
    total, end = 0., float("-inf")
    for begin, stop in sorted(intervals):
        total += max(0., stop - max(begin, end))
        end = max(end, stop)
    return total / 1000.


def _profile_chain(fn, output_dir, label):
    # Profile separately from benchmark: profiler overhead is never silently
    # included in or compared against the unprofiled latency distribution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        start.record()
        with torch.profiler.record_function(label + "/quantize_prepare_gemm"):
            fn()
        end.record()
        torch.cuda.synchronize()
    event_ms = start.elapsed_time(end)
    path = output_dir / (label + ".trace.json")
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    all_gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    histogram = collections.defaultdict(lambda: {"count": 0, "sum_ms": 0.})
    for event in kernels:
        histogram[event["name"]]["count"] += 1
        histogram[event["name"]]["sum_ms"] += event["dur"] / 1000.
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in all_gpu]
    assert kernels, "Profiler did not capture CUDA kernels; not a valid kernel audit"
    span = (max(stop for _, stop in intervals) - min(begin for begin, _ in intervals)) / 1000.
    union = _union_ms(intervals)
    record = {
        "trace_path": str(path), "scope": "one profiled full quantize+prepare+GEMM call",
        "profiled_cuda_event_ms": event_ms,
        "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000.,
        "kernel_count": len(kernels), "gpu_union_ms": union,
        "gpu_span_ms": span, "gpu_gap_ms": span - union,
        "kernels_by_full_name": dict(histogram),
    }
    assert any("humming" in name.lower() for name in histogram), "Expected Humming GEMM was not observed"
    (output_dir / (label + ".profile_summary.json")).write_text(json.dumps(record, indent=2))
    return record


def _time_ab(baseline_fn, candidate_fn):
    import triton.testing

    result = {"baseline_event_ms": [], "candidate_event_ms": []}
    for repeat in range(3):
        cases = [("baseline_event_ms", baseline_fn), ("candidate_event_ms", candidate_fn)]
        if repeat % 2:
            cases.reverse()
        for name, fn in cases:
            result[name].append(triton.testing.do_bench(fn, warmup=20, rep=100, return_mode="median"))
    result["speedup"] = statistics.median(result["baseline_event_ms"]) / statistics.median(result["candidate_event_ms"])
    return result


def _bounded_metrics(actual, reference):
    """Same fixed elementwise gate, bounded diagnostic scratch at routed M>128K."""
    records = [_metrics(actual[begin:begin + 2048], reference[begin:begin + 2048])
               for begin in range(0, actual.shape[0], 2048)]
    total = sum(record["elements"] for record in records)
    return {
        "elements": total,
        "unequal": sum(record["unequal"] for record in records),
        "outside_existing_tolerance": sum(record["outside_existing_tolerance"] for record in records),
        "max_abs": max(record["max_abs"] for record in records),
        "rms_abs": (sum(record["rms_abs"] ** 2 * record["elements"] for record in records) / total) ** .5,
        "finite": all(record["finite"] for record in records),
    }


def _large_candidates(baseline, fp8, device):
    # Actual chunk4096 trace routed M median31650/max129176; these schedules
    # target hundreds/thousands of rows per expert, not M4096/48~85. No new
    # arithmetic, dependency code, or weight layout is introduced.
    result = []
    for index in (2, 4):
        tile = _TILES[index]
        adapter = _Candidate(fp8.linear, activation_bits=8, tuning=_make_tuning(tile, device))
        result.append(("fp8_" + tile[0], adapter, False))
    for bm, bn, stages, ctas in ((64, 128, 4, 2), (64, 256, 4, 1), (128, 128, 4, 1)):
        # Existing WGMMA native warpgroup with sequential K32 and FP32 accum.
        # Unlike round2 TMA trials, test cp.async occupancy/reuse at balanced
        # actual routed M. Shared packed BF16-activation weight layout is safe.
        tile = (f"w4a16_m{bm}n{bn}_s{stages}c{ctas}", (bm, bn, 32),
                (bm, bn // 4, 32), stages, ctas, False)
        result.append((tile[0], _Candidate(baseline, tuning=_make_tuning(tile, device)), True))
    return result


def _w4a16_no_streamk_candidates(baseline, m):
    from humming.tune import get_heuristics_config

    # Keep baseline M/N tile, loader, residency, and packed weights. Only vary
    # K pipeline/CTA partitioning, explicitly separating it from Stream-K.
    base = get_heuristics_config(
        baseline.config, shape_m=m, use_f16_accum=False,
        use_batch_invariant=True, gemm_type="grouped_contiguous",
        device=baseline.weight.device,
    )
    result = []
    for block_k, warp_k in ((64, 64), (64, 32), (128, 64)):
        tuning = dict(base, use_stream_k=False)
        tuning["block_shape"] = (*base["block_shape"][:2], block_k)
        tuning["warp_shape"] = (*base["warp_shape"][:2], warp_k)
        # WGMMA BF16 swizzle supports warpK<=64. blockK>warpK creates an
        # intra-CTA FP32 reduction, not Stream-K or a BF16 partial buffer.
        name = f"w4a16_no_streamk_bk{block_k}_wk{warp_k}"
        result.append((name, _Candidate(baseline, batch_invariant=False,
                                        tuning=tuning), True))
    return result


def _bf16_bits(value):
    return int(torch.tensor(value, dtype=torch.bfloat16).view(torch.int16)) & 65535


def _bf16_ordered(bits):
    return 32768 - (bits & 32767) if bits & 32768 else 32768 + bits


def _striped_dot_anchors(x, weight, scales, offsets, actual, baseline):
    coordinates = []
    for start in range(0, actual.shape[0], 2048):
        local = (actual[start:start + 2048] != baseline[start:start + 2048]).nonzero()
        coordinates.extend((start + row, col) for row, col in local[:8 - len(coordinates)].tolist())
        if len(coordinates) == 8:
            break
    result = []
    for row, column in coordinates:
        expert = bisect.bisect_right(offsets, row) - 1
        qx = _check_quantization(x[row:row + 1]).flatten().cpu().double()
        w = _independent_weight(weight[expert, column:column + 1],
                                scales[expert, column:column + 1]).flatten().cpu().double()
        products = qx * w
        exact = math.fsum(products.tolist())
        fp64_dot = float(torch.dot(qx, w))
        rounded = torch.tensor(exact, dtype=torch.bfloat16)
        below = torch.nextafter(rounded, torch.tensor(float("-inf"), dtype=torch.bfloat16))
        above = torch.nextafter(rounded, torch.tensor(float("inf"), dtype=torch.bfloat16))
        lower, upper = (float(rounded), float(above)) if float(rounded) <= exact else (float(below), float(rounded))
        midpoint = (lower + upper) / 2
        a, b = float(actual[row, column]), float(baseline[row, column])
        ab, bb, eb = _bf16_bits(a), _bf16_bits(b), _bf16_bits(exact)
        result.append({
            "row": row, "column": column, "expert": expert,
            "cpu_fsum": exact, "cpu_fp64_dot": fp64_dot,
            "fp64_dot_equal_fsum": fp64_dot == exact,
            "baseline_bf16": b, "candidate_bf16": a, "fsum_rounded_bf16": float(rounded),
            "baseline_bits": hex(bb), "candidate_bits": hex(ab), "fsum_bf16_bits": hex(eb),
            "baseline_to_candidate_bf16_ulps": abs(_bf16_ordered(ab) - _bf16_ordered(bb)),
            "baseline_to_fsum_bf16_ulps": abs(_bf16_ordered(bb) - _bf16_ordered(eb)),
            "candidate_to_fsum_bf16_ulps": abs(_bf16_ordered(ab) - _bf16_ordered(eb)),
            "neighbor_bf16_lower": lower, "neighbor_bf16_upper": upper,
            "bf16_midpoint": midpoint, "fsum_minus_midpoint": exact - midpoint,
            "quantized_input_values": qx.tolist(), "decoded_weight_values": w.tolist(),
            "proof_boundary": "native K16/partition FP32 partials not observed; not a proof of native rounding order",
        })
    return result


def _basis_for_pipeline(weight, scales, tuning):
    # One actual expert, every coefficient, with this exact new K schedule.
    n, k = weight.shape[1], weight.shape[2] * 2
    linear = SM90HummingGroupedLinear(weight[:1], scales[:1])
    candidate = _Candidate(linear, batch_invariant=False, tuning=tuning)
    x = torch.eye(k, dtype=torch.bfloat16, device=weight.device)
    offsets = torch.tensor([0, k], dtype=torch.int32, device=weight.device)
    decoded = _independent_weight(weight[0], scales[0])
    raw_reference = decoded.T.bfloat16()
    # Identity GEMM starts a +0 accumulator, unlike directly copying an E2M1
    # coefficient (code8 decodes as -0). Keep the independent GEMM's bitwise
    # output contract, without changing original weight/payload byte checks.
    assert not torch.backends.cuda.matmul.allow_tf32
    reference = (x.float() @ decoded.T).bfloat16()
    actual = candidate(x, offsets)
    metrics = _bounded_metrics(actual, reference)
    raw_bit_mismatch = actual.view(torch.int16) != raw_reference.view(torch.int16)
    both_zero = (actual == 0) & (raw_reference == 0)
    return {"E": 1, "M": k, "N": n, "K": k,
            "bitwise": torch.equal(actual.view(torch.int16), reference.view(torch.int16)),
            "reference": "independent FP32 identity @ decoded_weight.T, TF32 disabled, then BF16",
            "raw_decoded_negative_zero_count": int((raw_reference.view(torch.int16) == -32768).sum()),
            "raw_bit_mismatch_zero_sign_only_count": int((raw_bit_mismatch & both_zero).sum()),
            "raw_bit_mismatch_nonzero_count": int((raw_bit_mismatch & ~both_zero).sum()),
            "error": metrics, "scope": "all coefficients of expert0 only; remaining experts not basis-tested"}


def _screen(m, n, k, *, actual_routed_m=False, no_streamk_pipeline=False,
            striped_diagnostic=False):
    experts = 48
    torch.manual_seed(20261015)
    weight, scales = _weights(experts, n, k)
    x = (torch.randn((m, k), device="cuda") * 12).bfloat16()
    counts = [m // experts + int(e < m % experts) for e in range(experts)]
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    qx = None if actual_routed_m else _check_quantization(x)
    reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for expert, count in enumerate(counts):
        if count:
            begin, end = host_offsets[expert:expert + 2]
            w = _independent_weight(weight[expert], scales[expert])
            # Independent FP32 oracle keeps only one expert's decoded weight
            # and <=2048 quantized rows live for large-M fixtures.
            if actual_routed_m:
                for first in range(begin, end, 2048):
                    last = min(first + 2048, end)
                    rows = _check_quantization(x[first:last])
                    reference[first:last] = (rows @ w.T).bfloat16()
                del rows
            else:
                reference[begin:end] = (qx[begin:end] @ w.T).bfloat16()
            del w
    del qx
    baseline = SM90HummingGroupedLinear(weight, scales)
    baseline_fn = lambda: baseline(x, offsets)
    baseline_output = baseline_fn()
    metrics = _bounded_metrics if actual_routed_m else _metrics
    baseline_error = metrics(baseline_output, reference)
    assert baseline_error["outside_existing_tolerance"] == 0, baseline_error
    if no_streamk_pipeline:
        candidates = _w4a16_no_streamk_candidates(baseline, m)
        if striped_diagnostic:
            candidates = [case for case in candidates
                          if case[1].tuning["block_shape"][2] != case[1].tuning["warp_shape"][2]]
    else:
        candidate = _FP8Adapter(weight, scales, fused=False, metadata_group=128, mma_type="mma")
        if actual_routed_m:
            candidates = _large_candidates(baseline, candidate, x.device)
        else:
            candidates = [(tile[0], _Candidate(candidate.linear, activation_bits=8,
                                             tuning=_make_tuning(tile, x.device)), False) for tile in _TILES]
    suite = "prefill_striped_diagnostic" if striped_diagnostic else "prefill_no_streamk_pipeline" if no_streamk_pipeline else "prefill_actual_routed" if actual_routed_m else "prefill"
    output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / f"{suite}_M{m}_N{n}_K{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_profile = _profile_chain(baseline_fn, output_dir, "w4a16_baseline")
    common = {
        "M_total": m, "N": n, "K": k, "E": experts,
        "expert_counts": counts, "max_expert_M": max(counts), "min_expert_M": min(counts),
        "quantizer": "original RTP g128/eps=1e-4; payload and scales bitwise verified",
        "baseline_profile": baseline_profile, "sources": _source_metadata(),
        "distribution_scope": "synthetic balanced routed rows; not measured per-expert skew",
        "baseline_error": baseline_error,
    }
    (output_dir / "metadata.json").write_text(json.dumps(common, indent=2))
    results, correctness_failures = [], []
    for name, candidate, require_bitwise in candidates:
        candidate_fn = lambda: candidate(x, offsets)
        record = {"candidate": name, "M_total": m, "N": n, "K": k,
                  "expert_counts": counts, "compute_config": candidate.compute,
                  "tuning_config": candidate.tuning, "layer_config": candidate.linear.config.to_str()}
        try:
            # Compiles and checks the new configuration BEFORE any timing.
            actual = candidate_fn()
            torch.cuda.synchronize()
        except (AssertionError, ValueError) as exc:
            # Dependency configuration rejection is reported, never counted as
            # a numerical PASS. CUDA/runtime failures are not swallowed.
            record.update(decision="INVALID_CONFIG", error=repr(exc))
            results.append(record)
            print("SM90_MOE_PREFILL_TUNING " + json.dumps(record), flush=True)
            continue
        except RuntimeError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            compile_only = str(exc).endswith("run failed") and any(
                frame.filename.endswith("humming/jit/compiler.py") for frame in frames
            )
            record.update(
                decision="INVALID_COMPILE_CONFIG" if compile_only else "RUNTIME_ERROR",
                error=repr(exc), traceback=traceback.format_exc(),
            )
            results.append(record)
            (output_dir / (name + ".result.json")).write_text(json.dumps(record, indent=2))
            print("SM90_MOE_PREFILL_TUNING " + json.dumps(record), flush=True)
            if not compile_only:
                raise
            continue
        error = metrics(actual, reference)
        relative = metrics(actual, baseline_output)
        repeatable = all(torch.equal(candidate_fn(), actual) for _ in range(3))
        record.update(
            error_vs_independent_oracle=error, error_vs_w4a16=relative,
            bitwise_equal_to_w4a16=bool(torch.equal(actual.view(torch.int16), baseline_output.view(torch.int16))),
            repeatable=repeatable,
            requires_bitwise_w4a16=require_bitwise,
        )
        valid = error["outside_existing_tolerance"] == 0 and relative["outside_existing_tolerance"] == 0 and repeatable
        if striped_diagnostic:
            record["basis"] = _basis_for_pipeline(weight, scales, candidate.tuning)
            anchors = _striped_dot_anchors(x, weight, scales, host_offsets, actual, baseline_output)
            anchor_path = output_dir / (name + ".dot_anchors.json")
            anchor_path.write_text(json.dumps(anchors, indent=2))
            record["dot_anchor_path"] = str(anchor_path)
            record["dot_anchor_summary"] = [{key: value for key, value in anchor.items()
                                             if key not in ("quantized_input_values", "decoded_weight_values")}
                                            for anchor in anchors]
            valid = valid and record["basis"]["bitwise"]
        else:
            valid = valid and (not require_bitwise or record["bitwise_equal_to_w4a16"])
        if not valid:
            record["decision"] = "REJECT_NUMERICAL"
            correctness_failures.append(name)
        else:
            record.update(_time_ab(baseline_fn, candidate_fn))
            profile = _profile_chain(candidate_fn, output_dir, name)
            record["profile"] = profile
            record["decision"] = "COMPONENT_FASTER" if record["speedup"] > 1 else "REJECT_PERFORMANCE"
            record["acceptance_scope"] = "component tolerance only; original MoE exact-stage/EP/model smoke still required"
            if striped_diagnostic:
                record["decision"] = "REJECT_BITWISE_DIAGNOSTIC_ONLY" if not record["bitwise_equal_to_w4a16"] else "DIAGNOSTIC_ONLY_NOT_PROMOTED"
                record["acceptance_scope"] = "diagnostic timing only; original bitwise gate remains unchanged and failed candidates stay rejected"
        results.append(record)
        (output_dir / (name + ".result.json")).write_text(json.dumps(record, indent=2))
        # Full template kernel names remain in the adjacent JSON artifact.
        concise = dict(record)
        if "profile" in concise:
            concise["profile"] = {key: value for key, value in concise["profile"].items() if key != "kernels_by_full_name"}
        print("SM90_MOE_PREFILL_TUNING " + json.dumps(concise), flush=True)
    (output_dir / "all_results.json").write_text(json.dumps(results, indent=2))
    assert any("error_vs_independent_oracle" in result for result in results), "No legal candidate ran"
    assert not correctness_failures, f"Unchanged correctness gate failed: {correctness_failures}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; root owns GPU locking and Bazel execution",
)
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m", [4096, 8192, 4099], ids=["m4096", "m8192", "m4099_prime"])
def test_sm90_moe_prefill_screen(m, n, k):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _screen(m, n, k)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; root owns GPU locking and Bazel execution",
)
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m", [32768, 65536, 131071], ids=["m32768", "m65536", "m131071_tail"])
def test_sm90_moe_prefill_actual_routed_m(m, n, k):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _screen(m, n, k, actual_routed_m=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; root owns GPU locking and Bazel execution",
)
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
@pytest.mark.parametrize("m", [32768], ids=["m32768"])
def test_sm90_moe_prefill_w4a16_no_streamk_pipeline(m, n, k):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _screen(m, n, k, actual_routed_m=True, no_streamk_pipeline=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires SM90; root owns GPU locking and Bazel execution",
)
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_sm90_moe_prefill_striped_diagnostic(n, k):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _screen(32768, n, k, actual_routed_m=True, no_streamk_pipeline=True,
                striped_diagnostic=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

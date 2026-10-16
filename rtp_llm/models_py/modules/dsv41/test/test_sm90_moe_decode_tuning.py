"""Bounded, test-only decode tuning using existing Humming GEMMs.

M counts already-routed rows, not original requests or global EP tokens.
The full chain here is one projection: RTP g128 quantize -> input preparation
-> grouped GEMM. Router/SiLU clamp/pre-down route weighting/combine are NOT
reimplemented or claimed covered. No resident unpacked model weights.
"""

import json
import os
import statistics
import time
from collections import Counter

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
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_fp8_diagnosis import (
    _FP8Adapter,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 only; parent owns GPU lock and test execution",
)


@pytest.fixture(autouse=True)
def _strict_oracle():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _counts(m, distribution, experts=48):
    counts = [0] * experts
    if distribution == "balanced":
        counts = [m // experts + int(e < m % experts) for e in range(experts)]
    elif distribution == "sparse":
        # Internal, leading and trailing empty experts, not just a compact prefix.
        for row in range(m):
            counts[(1, 17, 31)[row % 3]] += 1
    elif distribution == "hotspot":
        counts[-1] = m
    else:
        raise ValueError(distribution)
    assert sum(counts) == m
    return counts


def _candidates(linear, weight, scales, m, allowed_names=None):
    """Lazy construction keeps only one optional MMA-packed copy at a time."""
    from humming.tune import get_heuristics_config

    base = get_heuristics_config(
        linear.config, shape_m=m, use_f16_accum=False,
        use_batch_invariant=True, gemm_type="grouped_contiguous",
        device=weight.device,
    )
    assert base["block_shape"][2] == base["warp_shape"][2] == 32
    # Test-only selector bounds first-pass JIT; never changes production dispatch.
    selected = set(filter(None, os.environ.get("DSV41_DECODE_CANDIDATES", "").split(",")))
    if allowed_names is not None:
        selected = selected.intersection(allowed_names) if selected else set(allowed_names)
        assert selected, "No requested candidate applies to the controlled profile"
    for bm, bn in ((8, 128), (16, 128), (8, 256)):
        name = f"w4a16_m{bm}n{bn}k32"
        if selected and name not in selected:
            continue
        tuning = dict(
            base, block_shape=(bm, bn, 32), warp_shape=(bm, bn // 4, 32),
            num_stages=3, num_ctas_per_sm=2, use_stream_k=False,
            use_tma=False, use_warp_spec=False, use_mbarrier=False,
        )
        yield name, _Candidate(linear, tuning=tuning), True
    if linear.in_features > 1024 and (not selected or "w4a16_streamk_heuristic" in selected):
        # This API offers Stream-K, not a supported arbitrary split_k knob.
        # BF16 partial reductions can change values/order: never call it exact.
        tuning = get_heuristics_config(
            linear.config, shape_m=m, use_f16_accum=False,
            use_batch_invariant=False, gemm_type="grouped_contiguous",
            device=weight.device,
        )
        assert tuning["use_stream_k"]
        yield "w4a16_streamk_heuristic", _Candidate(
            linear, batch_invariant=False, tuning=tuning,
        ), False
    for group in (128, 32):
        name = f"w4a8_mma_g{group}"
        if selected and name not in selected:
            continue
        # Existing, round3-tested legal MMA tile (32,128,64); g32 only repeats
        # g128 metadata. Never requantize activation values to a different group.
        yield name, _FP8Adapter(
            weight, scales, fused=False, metadata_group=group, mma_type="mma",
        ), False


def _bits_equal(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def _profile(fn, flush=None):
    """Exactly one warmed invocation; sums only actual device kernel events."""
    fn()
    torch.cuda.synchronize()
    if flush is not None:
        # Finish eviction before profiler entry. Queuing fn behind a pending
        # flush would hide host-launch gaps and invalidate hot/cold comparison.
        flush.zero_()
        torch.cuda.synchronize()
    enter_begin = time.perf_counter()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        enter_end = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        invocation_end = time.perf_counter()
    exit_end = time.perf_counter()
    counts, times = Counter(), Counter()
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        # Memcpy/memset are reported separately, not mislabeled GEMM kernels.
        counts[event.name] += 1
        times[event.name] += event.time_range.elapsed_us()
    return {
        "device_events": [
            {"name": name, "count": counts[name], "total_us": times[name]}
            for name in sorted(counts)
        ],
        "pure_kernel_sum_us": sum(
            value for name, value in times.items()
            if not any(tag in name.lower() for tag in ("memcpy", "memset"))
        ),
        "invocations": 1, "warmup_in_capture": False,
        "cache_condition": "evicted_and_synchronized" if flush is not None else "warm_and_synchronized",
        "profile_enter_wall_us": (enter_end - enter_begin) * 1e6,
        "invocation_and_sync_wall_us": (invocation_end - enter_end) * 1e6,
        "profile_exit_wall_us": (exit_end - invocation_end) * 1e6,
    }


def _event_times(fn, flush, repeats=13):
    """Eager device span, no CUDA graph; L2 eviction is outside event interval."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        if flush is not None:
            flush.zero_()
            torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(values), "samples_ms": values}


def _measurement(candidate, baseline, x, offsets, flush, controlled=False):
    q, packed = _quantize(x)
    prepared, scale = candidate.prepare_input(q, packed)
    candidate_chain = lambda: candidate(x, offsets)
    baseline_chain = lambda: baseline(x, offsets)
    candidate_gemm = lambda: candidate.prepared_gemm(prepared, scale, offsets)
    baseline_prepared, baseline_scale = baseline.prepare_input(q, packed)
    baseline_gemm = lambda: baseline.prepared_gemm(baseline_prepared, baseline_scale, offsets)
    # Alternate paired baseline/candidate order across independent rounds.
    times = {key: [] for key in ("baseline_hot", "candidate_hot", "baseline_evicted", "candidate_evicted")}
    for repeat in range(3):
        pair = [("baseline", baseline_chain), ("candidate", candidate_chain)]
        if repeat % 2:
            pair.reverse()
        for name, fn in pair:
            times[name + "_hot"].append(_event_times(fn, None))
            times[name + "_evicted"].append(_event_times(fn, flush))
    result = {
        "timing": times,
        "baseline_full_chain_profile": _profile(baseline_chain),
        "candidate_full_chain_profile": _profile(candidate_chain),
        "candidate_prepared_gemm_profile": _profile(candidate_gemm),
        "candidate_prepared_gemm_evicted": _event_times(candidate_gemm, flush),
        "timing_scope": "one projection quantize+prepare+GEMM, eager device span",
        "l2_evict_bytes": flush.numel(),
        "l2_caveat": "eviction buffer is >=256MiB and >=4x reported L2; not proof of perfectly cold HBM",
        "measurement_revision": "v2: eviction synchronized before events/profiler; tuning JSON pre-serialized",
    }
    if controlled:
        profiles = {name: {"hot": [], "evicted": []} for name in (
            "baseline_full_chain", "candidate_full_chain", "baseline_prepared_gemm", "candidate_prepared_gemm",
        )}
        for repeat in range(3):
            pairs = [
                ("baseline_full_chain", baseline_chain),
                ("candidate_full_chain", candidate_chain),
                ("baseline_prepared_gemm", baseline_gemm),
                ("candidate_prepared_gemm", candidate_gemm),
            ]
            if repeat % 2:
                pairs.reverse()
            cache_modes = [("hot", None), ("evicted", flush)]
            if repeat % 2:
                cache_modes.reverse()
            for name, fn in pairs:
                for cache_name, eviction in cache_modes:
                    profiles[name][cache_name].append(_profile(fn, eviction))
        result["controlled_profiles"] = profiles
        result["profile_interpretation"] = (
            "Compare like-for-like pure_kernel_sum_us with 3 samples and AB/BA order; "
            "profiler CPU wall overhead and eager event launch gaps are not kernel cost. "
            "No CUDA graph; synthetic distribution, not a model-speed claim."
        )
    return result


def _run(m, n, k, distribution, controlled=False):
    torch.manual_seed(20261016 + m)
    experts = 48
    weight = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
    exponents = torch.randint(117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
    exponents[:, :, 0], exponents[:, :, 1] = 117, 127
    scales = exponents.view(torch.float8_e8m0fnu)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    counts = _counts(m, distribution)
    host_offsets = [0]
    for count in counts:
        host_offsets.append(host_offsets[-1] + count)
    offsets = torch.tensor(host_offsets, device="cuda", dtype=torch.int32)
    q, packed = _quantize(x)
    q_ref, s_ref = _independent_quantized_input(x)
    torch.testing.assert_close(q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(unpack_activation_scales(packed, k // 128), s_ref, rtol=0, atol=0)
    qx = (q_ref.float().reshape(m, -1, 128) * s_ref.unsqueeze(-1)).reshape(m, k)
    reference = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    for expert, count in enumerate(counts):
        if count:
            begin, end = host_offsets[expert:expert + 2]
            # Decode at most one expert in the independent oracle, outside timing.
            decoded = _independent_weight(weight[expert], scales[expert])
            reference[begin:end] = (qx[begin:end] @ decoded.T).bfloat16()
            del decoded
    linear = SM90HummingGroupedLinear(weight, scales)
    baseline = _Candidate(linear)
    baseline_output = baseline(x, offsets)
    torch.testing.assert_close(baseline_output, reference, rtol=.02, atol=.05)
    assert all(_bits_equal(baseline(x, offsets), baseline_output) for _ in range(5))
    properties = torch.cuda.get_device_properties(weight.device)
    l2_bytes = getattr(properties, "L2_cache_size", getattr(properties, "l2_cache_size", 0))
    flush = torch.empty(max(256 * 1024**2, 4 * l2_bytes), dtype=torch.uint8, device="cuda")
    common = {
        "M_routed_rows": m, "N": n, "K": k, "E": experts,
        "distribution": distribution, "counts": counts,
        "empty_experts": counts.count(0), "source": _source_metadata(),
        "device": properties.name, "reported_l2_bytes": l2_bytes,
        "scope": "component only; full MoE/EP/model smoke NOT_RUN",
        "input_contract": "unchanged g128 eps1e-4 E4M3 payload and UE8M0 scales",
        "weight_contract": "same raw MXFP4 g32 E8M0 [117,127], one projection packed",
        "baseline_error": _metrics(baseline_output, reference),
    }
    tested = 0
    allowed_names = {"w4a16_m8n128k32", "w4a16_m16n128k32"} if controlled else None
    for name, candidate, require_bitwise in _candidates(linear, weight, scales, m, allowed_names):
        tested += 1
        # The installed forward API accepts JSON str directly. Serialize once,
        # outside timing, avoiding a candidate-only json.dumps on every call.
        # Keep baseline tuning=None: its default config/path is not replaced.
        if isinstance(candidate.tuning, (dict, list)):
            candidate.tuning = json.dumps(candidate.tuning)
        # No exception-swallowing: invalid launch/config is a genuine failed test.
        actual = candidate(x, offsets)
        error = _metrics(actual, reference)
        baseline_error = _metrics(actual, baseline_output)
        bitwise = _bits_equal(actual, baseline_output)
        repeatable = all(_bits_equal(candidate(x, offsets), actual) for _ in range(5))
        accepted = (
            error["outside_existing_tolerance"] == 0
            and baseline_error["outside_existing_tolerance"] == 0
            and repeatable and (bitwise or not require_bitwise)
        )
        record = dict(
            common, candidate=name, decision="BOUNDED_SUPPORT" if accepted else "REJECT",
            original_gate={"rtol": .02, "atol": .05}, error_vs_fp32=error,
            error_vs_baseline=baseline_error, baseline_bitwise=bitwise,
            bitwise_required=require_bitwise, deterministic_5_repeats=repeatable,
            compute=candidate.compute, tuning=candidate.tuning,
            layer_config=candidate.linear.config.to_str(),
        )
        if accepted:
            record.update(_measurement(candidate, baseline, x, offsets, flush, controlled))
        print("SM90_DECODE_TUNING " + json.dumps(record), flush=True)
        del candidate
    assert tested, "DSV41_DECODE_CANDIDATES selected no known candidate"
    # This is a diagnostic sweep: REJECT candidates are never benchmarked or
    # promoted. A successful test means the bounded experiment completed, NOT
    # that all candidates passed or that model-level correctness is established.


def test_decode_fixture():
    _run(17, 256, 256, "sparse")


@pytest.mark.parametrize("m", [1, 17], ids=["m1", "m17"])
@pytest.mark.parametrize("distribution", ["balanced", "hotspot"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_decode_priority(m, distribution, n, k):
    _run(m, n, k, distribution)


@pytest.mark.parametrize("m", [8, 64, 128], ids=["m8", "m64", "m128"])
@pytest.mark.parametrize("distribution", ["balanced", "sparse", "hotspot"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_decode_extended(m, distribution, n, k):
    _run(m, n, k, distribution)


@pytest.mark.parametrize("m", [1, 17], ids=["m1", "m17"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_decode_sparse_priority(m, n, k):
    _run(m, n, k, "sparse")


@pytest.mark.parametrize("m", [8, 16, 24, 40], ids=["m8", "m16", "m24", "m40"])
@pytest.mark.parametrize("distribution", ["balanced", "sparse", "hotspot"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_decode_trace_buckets(m, distribution, n, k):
    # M buckets are observed in bs1 TP8/EP8 decode trace; expert counts are NOT
    # available there. All three distributions remain synthetic diagnostics.
    _run(m, n, k, distribution)


@pytest.mark.parametrize("m", [8, 16, 24, 40], ids=["m8", "m16", "m24", "m40"])
@pytest.mark.parametrize("distribution", ["hotspot", "sparse"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_decode_controlled_profile(m, distribution, n, k):
    _run(m, n, k, distribution, controlled=True)

"""Original gates for the opt-in indexed local MoE; parent owns GPU execution."""

import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_indexed import try_forward_indexed
from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import SM90HummingGroupedLinear
from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import _GroupedLocalExperts
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input,
    _independent_weight,
    _metrics,
)


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 under parent GPU lock",
)


def _quantized_values(x):
    q, scales = _independent_quantized_input(x)
    return (q.float().reshape(x.shape[0], -1, 128) * scales.unsqueeze(-1)).reshape_as(x)


def _reference(x, weights, ids, start, w13, s13, w2, s2, *, stages=False,
               precision=torch.float32):
    rows, topk = ids.shape
    slot_result = torch.zeros((rows * topk, x.shape[1]), device=x.device, dtype=torch.float32)
    qx = _quantized_values(x)
    intermediates = {}
    if stages:
        for name, width in (("gate_up", w13.shape[1]), ("hidden", w2.shape[2] * 2),
                            ("down_input", w2.shape[2] * 2), ("down", x.shape[1])):
            intermediates[name] = torch.zeros((rows * topk, width), device=x.device, dtype=torch.bfloat16)
    flat = ids.flatten()
    for expert in range(w13.shape[0]):
        slots = torch.where(flat == start + expert)[0]
        if not slots.numel():
            continue
        up_weight = _independent_weight(w13[expert], s13[expert])
        gu = (qx.index_select(0, slots // topk).to(precision)
              @ up_weight.T.to(precision)).bfloat16().to(precision)
        del up_weight
        gate, up = gu.chunk(2, -1)
        hidden = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                  * weights.flatten().index_select(0, slots)[:, None].to(precision)).bfloat16()
        down_weight = _independent_weight(w2[expert], s2[expert])
        down_input = _quantized_values(hidden)
        down = (down_input.to(precision) @ down_weight.T.to(precision)).bfloat16().float()
        del down_weight
        slot_result.index_copy_(0, slots, down)
        if stages:
            for name, value in (("gate_up", gu), ("hidden", hidden),
                                ("down_input", down_input), ("down", down)):
                intermediates[name].index_copy_(0, slots, value.bfloat16())
    output = slot_result.reshape(rows, topk, x.shape[1]).sum(1)
    return (output, intermediates) if stages else output


def _comparison(actual, reference):
    record = _metrics(actual, reference)
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    record["bitwise_unequal"] = int((actual.contiguous().view(integer)
                                    != reference.contiguous().view(integer)).sum())
    return record


def _stage_diagnostics(local, x, weights, ids, start, end, raw, monkeypatch):
    """Diagnostic-only reruns. FP64 controls DO NOT replace the original gate."""
    from rtp_llm.models_py.modules.dsv41.moe import _sm90_humming_indexed as indexed
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import group_local_routes

    slots, _ = group_local_routes(ids, start, end)
    captured = {}
    for mode in ("grouped", "indexed"):
        stages = {}
        def save(name, value):
            value = value.detach().clone()
            if mode == "grouped":
                ordered = torch.zeros((ids.numel(), value.shape[1]), device=x.device, dtype=value.dtype)
                ordered.index_copy_(0, slots, value)
                value = ordered
            stages[name] = value
        with monkeypatch.context() as patch:
            patch.setenv("DSV41_SM90_INDEXED_MOE", "1" if mode == "indexed" else "0")
            handles = []
            if mode == "grouped":
                def down_hook(module, args, result):
                    save("hidden", args[0])
                    save("down", result)
                handles.append(local.w13.register_forward_hook(lambda module, args, result: save("gate_up", result)))
                handles.append(local.w2.register_forward_hook(down_hook))
            else:
                original_gemm = indexed._gemm
                original_swiglu = indexed.routed_swiglu
                def gemm(linear, inputs, output, metadata, topk, tuning):
                    result = original_gemm(linear, inputs, output, metadata, topk, tuning)
                    save("gate_up" if linear is local.w13 else "down", output)
                    if linear is local.w2:
                        save("down_input", inputs)
                    return result
                def swiglu(*args, **kwargs):
                    result = original_swiglu(*args, **kwargs)
                    save("hidden", result)
                    return result
                patch.setattr(indexed, "_gemm", gemm)
                patch.setattr(indexed, "routed_swiglu", swiglu)
            try:
                local._forward_into_buf(x, weights, ids, start, end)
            finally:
                for handle in handles:
                    handle.remove()
        if mode == "grouped":
            # Exact production quantizer/decode, in original slot order.
            stages["down_input"] = indexed._decode(stages["hidden"])
        captured[mode] = stages

    reference32, stages32 = _reference(x, weights, ids, start, *raw, stages=True)
    reference64, stages64 = _reference(x, weights, ids, start, *raw, stages=True, precision=torch.float64)
    record = {"note": "FP64 diagnostic changes accumulation/nonlinearity precision, not acceptance gate",
              "fp32_vs_fp64_final": _comparison(reference32, reference64), "stages": {}}
    for name in stages32:
        record["stages"][name] = {
            "indexed_vs_grouped": _comparison(captured["indexed"][name], captured["grouped"][name]),
            "grouped_vs_fp32": _comparison(captured["grouped"][name], stages32[name]),
            "grouped_vs_fp64": _comparison(captured["grouped"][name], stages64[name]),
            "fp32_vs_fp64": _comparison(stages32[name], stages64[name]),
        }
    # Isolate SiLU arithmetic with the SAME rounded gate/up, then quantization
    # with the SAME BF16 hidden. This distinguishes upstream GEMM differences
    # from sigmoid approximation and a subsequent FP8 threshold crossing.
    gu = captured["grouped"]["gate_up"]
    gate, up = gu.float().chunk(2, -1)
    control32 = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                 * weights.reshape(-1, 1)).bfloat16()
    control64 = (torch.nn.functional.silu(gate.double().clamp(max=10)) * up.double().clamp(-10, 10)
                 * weights.reshape(-1, 1).double()).bfloat16()
    valid = ((ids >= start) & (ids < end)).flatten()
    hidden = captured["grouped"]["hidden"]
    for label, control in (("fp32_silu_same_gate", control32), ("fp64_silu_same_gate", control64)):
        record[label] = _comparison(hidden[valid], control[valid])
    record["g128_same_hidden"] = _comparison(captured["grouped"]["down_input"][valid],
                                              _quantized_values(hidden).bfloat16()[valid])
    return record


def _persist(record):
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if directory:
        path = Path(directory) / (f"indexed_routes_r{record['rows']}_k{record['topk']}_"
                                  f"{record['pattern']}_h{record['hidden']}_"
                                  f"{record.get('acceptance', 'strict')}_"
                                  f"{record.get('fixture_source', 'random')}.json")
        path.write_text(json.dumps(record, indent=2))


def _make_local(rows, hidden, intermediate, experts):
    torch.manual_seed(20261018 + rows)
    def projection(n, k):
        w = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
        s = torch.randint(117, 128, (experts, n, k // 32), device="cuda", dtype=torch.uint8)
        s[:, :, 0], s[:, :, 1] = 117, 127
        return w, s.view(torch.float8_e8m0fnu)
    w13, s13 = projection(2 * intermediate, hidden)
    w2, s2 = projection(hidden, intermediate)
    local = _GroupedLocalExperts(SimpleNamespace(swiglu_limit=10.0))
    local.w13, local.w2 = SM90HummingGroupedLinear(w13, s13), SM90HummingGroupedLinear(w2, s2)
    x = (torch.randn(rows, hidden, device="cuda") * 3).bfloat16()
    return local, x, (w13, s13, w2, s2)


def _make_checkpoint_local(rows, checkpoint_root):
    """Select layer6 E48 packed tensors only; no full model/expanded weights."""
    from safetensors import safe_open

    root = Path(checkpoint_root)
    index_path = root / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    weight_map = json.loads(index_bytes)["weight_map"]
    keys = [f"layers.6.ffn.experts.{expert}.{part}.{kind}"
            for expert in range(48) for part in ("w1", "w3", "w2") for kind in ("weight", "scale")]
    missing = [key for key in keys if key not in weight_map]
    assert not missing, f"Checkpoint fixture requires exact V4.1 keys: {missing}"
    # Preallocate only the final packed CPU layout, copying one expert at a
    # time. uint8 scale storage preserves E8M0 bytes without arithmetic casts.
    cpu = {
        "w13": torch.empty((48, 4608, 2560), dtype=torch.int8),
        "s13": torch.empty((48, 4608, 160), dtype=torch.uint8),
        "w2": torch.empty((48, 5120, 1152), dtype=torch.int8),
        "s2": torch.empty((48, 5120, 72), dtype=torch.uint8),
    }
    selected_shards = sorted({weight_map[key] for key in keys})
    key_records = []
    for shard in selected_shards:
        with safe_open(str(root / shard), framework="pt", device="cpu") as reader:
            for key in keys:
                if weight_map[key] != shard:
                    continue
                expert, part, kind = int(key.split(".")[4]), key.split(".")[5], key.split(".")[6]
                tensor = reader.get_tensor(key)
                expected_shape = ((5120, 1152 if kind == "weight" else 72) if part == "w2"
                                  else (2304, 2560 if kind == "weight" else 160))
                expected_dtype = torch.int8 if kind == "weight" else torch.float8_e8m0fnu
                assert tuple(tensor.shape) == expected_shape and tensor.dtype == expected_dtype, (key, tensor.shape, tensor.dtype)
                name = ("w" if kind == "weight" else "s") + ("2" if part == "w2" else "13")
                target = cpu[name][expert]
                if part != "w2":
                    offset = 0 if part == "w1" else 2304
                    target = target[offset:offset + 2304]
                target.copy_(tensor if kind == "weight" else tensor.view(torch.uint8))
                key_records.append(dict(key=key, shard=shard, shape=list(tensor.shape), dtype=str(tensor.dtype)))
    metadata = dict(checkpoint=str(root), layer=6, expert_start=0, expert_end=48,
                    index_sha256=hashlib.sha256(index_bytes).hexdigest(), tensors=key_records,
                    activation="SYNTHETIC BF16 random*3; NOT captured model activation",
                    scale_byte_ranges={name: [int(cpu[name].min()), int(cpu[name].max())]
                                       for name in ("s13", "s2")})
    raw = (cpu["w13"].cuda(), cpu["s13"].view(torch.float8_e8m0fnu).cuda(),
           cpu["w2"].cuda(), cpu["s2"].view(torch.float8_e8m0fnu).cuda())
    del cpu
    local = _GroupedLocalExperts(SimpleNamespace(swiglu_limit=10.0))
    local.w13, local.w2 = SM90HummingGroupedLinear(raw[0], raw[1]), SM90HummingGroupedLinear(raw[2], raw[3])
    torch.manual_seed(20261018 + rows)
    x = (torch.randn(rows, 5120, device="cuda") * 3).bfloat16()
    return local, x, raw, metadata


def _ids(rows, topk, experts, start, pattern):
    values = []
    for row in range(rows):
        current = []
        for slot in range(topk):
            value = start + (row * 3 + slot) % experts
            if pattern == "mixed":
                value = (-1, start - 1, start + experts, value)[slot % 4]
            elif pattern == "all_invalid":
                value = (-1, start - 1, start + experts, 2**40)[slot % 4]
            elif pattern == "duplicate":
                value = start + row % experts
            current.append(value)
        values.append(current)
    return torch.tensor(values, dtype=torch.int64, device="cuda").reshape(rows, topk)


def _baseline(local, x, weights, ids, start, end, monkeypatch):
    # Robust when parent subsequently hooks indexed into the production method.
    with monkeypatch.context() as patch:
        patch.setenv("DSV41_SM90_INDEXED_MOE", "0")
        return local._forward_into_buf(x, weights, ids, start, end)


def _run(rows, topk, pattern, hidden, intermediate, experts, monkeypatch, *,
         equivalence_only=False, benchmark=False, checkpoint_root=None):
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    monkeypatch.setenv("DSV41_SM90_INDEXED_MOE", "1")
    try:
        if checkpoint_root is None:
            local, x, raw = _make_local(rows, hidden, intermediate, experts)
            start, source = 96, None
        else:
            assert (hidden, intermediate, experts) == (5120, 2304, 48)
            assert not equivalence_only and not benchmark
            local, x, raw, source = _make_checkpoint_local(rows, checkpoint_root)
            start = 0
        end = start + experts
        ids = _ids(rows, topk, experts, start, pattern)
        weights = torch.linspace(.03, .97, rows * topk, device="cuda").reshape(rows, topk)
        before = [(tensor.data_ptr(), tensor._version) for module in (local.w13, local.w2)
                  for tensor in (module.weight, module.weight_scale)]
        records = []
        record = dict(rows=rows, topk=topk, pattern=pattern, hidden=hidden, intermediate=intermediate,
                      experts=experts, start=start, route_checks=records,
                      acceptance="equivalence_only" if equivalence_only else "strict",
                      fixture_source="checkpoint_layer6" if source is not None else "random",
                      checkpoint_metadata=source,
                      promotion="OPT_IN_ONLY; independent stress failures remain unresolved",
                      scope="local MoE only; EP dispatch/combine/model NOT_RUN")
        for route_number, route in enumerate((ids, ids.flip(0).contiguous(), _ids(rows, topk, experts, start, "duplicate"))):
            reference = _reference(x, weights, route, start, *raw)
            baseline = _baseline(local, x, weights, route, start, end, monkeypatch)
            actual = try_forward_indexed(local, x, weights, route, start, end)
            assert actual is not None
            metrics = dict(route_number=route_number,
                           baseline_vs_reference=_comparison(baseline, reference),
                           indexed_vs_reference=_comparison(actual, reference),
                           indexed_vs_baseline=_comparison(actual, baseline))
            records.append(metrics)
            # Persist all three comparisons BEFORE the original first assertion.
            # A reference failure must not hide indexed-vs-baseline evidence.
            _persist(record)
            print("INDEXED_ROUTES_METRICS " + json.dumps(metrics), flush=True)
            failed = any(metrics[key]["outside_existing_tolerance"] for key in
                         ("baseline_vs_reference", "indexed_vs_reference"))
            failed = failed or metrics["indexed_vs_baseline"]["bitwise_unequal"] != 0
            if failed and not equivalence_only and os.environ.get("DSV41_INDEXED_STAGE_DIAGNOSTICS", "0") == "1":
                metrics["stage_diagnostics"] = _stage_diagnostics(
                    local, x, weights, route, start, end, raw, monkeypatch)
                _persist(record)
                print("INDEXED_ROUTES_STAGES " + json.dumps(metrics["stage_diagnostics"]), flush=True)
            metrics["independent_gate_passed"] = not any(
                metrics[key]["outside_existing_tolerance"] for key in
                ("baseline_vs_reference", "indexed_vs_reference"))
            _persist(record)
            if not equivalence_only:
                # Original pressure fixtures retain BOTH assertions unchanged.
                torch.testing.assert_close(baseline, reference, rtol=.02, atol=.05)
                torch.testing.assert_close(actual, reference, rtol=.02, atol=.05)
            torch.testing.assert_close(actual.view(torch.int32), baseline.view(torch.int32), rtol=0, atol=0)
            for _ in range(3):
                repeated = try_forward_indexed(local, x, weights, route, start, end)
                torch.testing.assert_close(repeated.view(torch.int32), actual.view(torch.int32), rtol=0, atol=0)
            metrics["baseline_bitwise"] = True
            metrics["repeatability_bitwise"] = True
        after = [(tensor.data_ptr(), tensor._version) for module in (local.w13, local.w2)
                 for tensor in (module.weight, module.weight_scale)]
        assert before == after, "Existing packed weights must not be replaced or mutated"
        record["reused_weight_buffers"] = True
        record["all_independent_gates_passed"] = all(r["independent_gate_passed"] for r in records)
        if benchmark:
            assert not equivalence_only and record["all_independent_gates_passed"]
            record["measurements"] = _benchmark(local, x, weights, ids, start, end,
                                                  pattern, monkeypatch)
        _persist(record)
        print("INDEXED_ROUTES " + json.dumps({key: record[key] for key in ("rows", "topk", "pattern", "hidden")}), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_opt_in_and_unsupported_cpu(monkeypatch):
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    weights = torch.ones(1, 8)
    ids = torch.zeros(1, 8, dtype=torch.int64)
    monkeypatch.delenv("DSV41_SM90_INDEXED_MOE", raising=False)
    assert try_forward_indexed(None, x, weights, ids, 0, 48) is None
    monkeypatch.setenv("DSV41_SM90_INDEXED_MOE", "1")
    assert try_forward_indexed(None, x, weights, ids, 0, 48) is None


@_GPU
@pytest.mark.parametrize("pattern", ["valid", "mixed", "all_invalid", "duplicate"])
def test_indexed_routes_fixture(pattern, monkeypatch):
    _run(3, 8, pattern, 256, 128, 8, monkeypatch)


@_GPU
@pytest.mark.parametrize("topk", [1, 2, 3, 4, 5, 6, 7, 8])
def test_indexed_topk_fixture(topk, monkeypatch):
    _run(1, topk, "valid", 256, 128, 8, monkeypatch)


@_GPU
@pytest.mark.parametrize("rows", [1, 3, 8, 16, 40], ids=["r1", "r3", "r8", "r16", "r40"])
@pytest.mark.parametrize("pattern", ["mixed", "duplicate"])
def test_indexed_routes_actual(rows, pattern, monkeypatch):
    _run(rows, 8, pattern, 5120, 2304, 48, monkeypatch)


@_GPU
@pytest.mark.parametrize("rows", [1, 3, 8, 16, 40], ids=["r1", "r3", "r8", "r16", "r40"])
@pytest.mark.parametrize("pattern", ["mixed", "duplicate"])
def test_actual_equivalence(rows, pattern, monkeypatch):
    """Legacy-semantic equivalence, explicitly NOT independent accuracy PASS."""
    _run(rows, 8, pattern, 5120, 2304, 48, monkeypatch, equivalence_only=True)


@_GPU
def test_indexed_edges_empty(monkeypatch):
    monkeypatch.setenv("DSV41_SM90_INDEXED_MOE", "1")
    local, x, _ = _make_local(0, 256, 128, 8)
    ids = torch.empty((0, 8), device="cuda", dtype=torch.int64)
    weights = torch.empty((0, 8), device="cuda")
    result = try_forward_indexed(local, x, weights, ids, 96, 104)
    assert result is not None and result.shape == x.shape and result.dtype == torch.float32
    assert result.device == x.device


@_GPU
def test_indexed_edges_unsupported(monkeypatch):
    monkeypatch.setenv("DSV41_SM90_INDEXED_MOE", "1")
    x = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros((1, 8), device="cuda", dtype=torch.int64)
    weights = torch.ones((1, 8), device="cuda")
    unsupported = [
        (x.float(), weights, ids), (x.half(), weights, ids),
        (torch.zeros((1, 512), device="cuda", dtype=x.dtype)[:, ::2], weights, ids),
        (x, torch.ones((1, 16), device="cuda")[:, ::2], ids),
        (x, weights, torch.zeros((1, 16), device="cuda", dtype=ids.dtype)[:, ::2]),
        (x, weights.half(), ids), (x, weights, ids.float()),
        (x, torch.ones((1, 9), device="cuda"), torch.zeros((1, 9), device="cuda", dtype=ids.dtype)),
        (x.expand(128, -1).contiguous(), weights.expand(128, -1).contiguous(), ids.expand(128, -1).contiguous()),
    ]
    for bad_x, bad_weights, bad_ids in unsupported:
        # None local proves rejection occurs before dereferencing model weights.
        assert try_forward_indexed(None, bad_x, bad_weights, bad_ids, 96, 104) is None


@_GPU
def test_indexed_edges_nondefault_stream(monkeypatch):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(3, 8, "mixed", 256, 128, 8, monkeypatch)
        finished = torch.cuda.Event()
        finished.record()
    torch.cuda.current_stream().wait_event(finished)
    finished.synchronize()


@_GPU
@pytest.mark.parametrize("pattern", ["mixed", "all_invalid"])
def test_indexed_edges_invalid_nan_weights(pattern, monkeypatch):
    monkeypatch.setenv("DSV41_SM90_INDEXED_MOE", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    local, x, raw = _make_local(3, 256, 128, 8)
    ids = _ids(3, 8, 8, 96, pattern)
    weights = torch.full((3, 8), .3, device="cuda")
    weights[(ids < 96) | (ids >= 104)] = float("nan")
    reference = _reference(x, weights, ids, 96, *raw)
    baseline = _baseline(local, x, weights, ids, 96, 104, monkeypatch)
    actual = try_forward_indexed(local, x, weights, ids, 96, 104)
    assert actual is not None
    torch.testing.assert_close(baseline, reference, rtol=.02, atol=.05)
    torch.testing.assert_close(actual, reference, rtol=.02, atol=.05)
    torch.testing.assert_close(actual.view(torch.int32), baseline.view(torch.int32), rtol=0, atol=0)


def _measure_full_chain(fn, flush, trace_path):
    """No graph: stream spans and pure kernel sums are DIFFERENT measurements."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize timing events outside measurements as well.
    begin.record()
    end.record()
    end.synchronize()
    samples = []
    for _ in range(9):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    profile_wall_ms = (time.perf_counter() - wall_start) * 1000
    profiler.export_chrome_trace(str(trace_path))
    counts, sums = Counter(), Counter()
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[event.name] += 1
            sums[event.name] += event.time_range.elapsed_us()
    kernels = [name for name in counts if "memcpy" not in name.lower() and "memset" not in name.lower()]
    humming = sum(counts[name] for name in kernels if re.search(r"(?:^|::|\s)humming(?:[<(]|$)", name))
    result = dict(stream_span_ms=samples, median_stream_span_ms=statistics.median(samples),
                  profile=dict(iterations=1, profiler_context_wall_ms=profile_wall_ms,
                               kernel_sum_us_per_iter=sum(sums[name] for name in kernels),
                               pure_kernel_launch_count=sum(counts[name] for name in kernels),
                               humming_launch_count=humming,
                               gpu_events=[dict(name=name, count=counts[name], us=sums[name]) for name in sorted(counts)],
                               trace=str(trace_path)))
    assert humming == 2, f"Expected complete two-projection chain: {result}"
    return result


def _benchmark(local, x, weights, ids, start, end, pattern, monkeypatch):
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not directory:
        pytest.fail("Benchmark requires TEST_UNDECLARED_OUTPUTS_DIR for persistent traces")
    root = Path(directory) / f"full_chain_r{x.shape[0]}_{pattern}"
    root.mkdir(parents=True, exist_ok=False)
    flush = torch.empty(256 * 1024**2, device=x.device, dtype=torch.uint8)
    results = {name: {mode: [] for mode in ("hot", "cold")} for name in ("baseline", "indexed")}
    for repeat in range(3):
        order = ("baseline", "indexed") if repeat % 2 == 0 else ("indexed", "baseline")
        for name in order:
            with monkeypatch.context() as patch:
                patch.setenv("DSV41_SM90_INDEXED_MOE", "1" if name == "indexed" else "0")
                # Same production entrypoint, including its existing env check.
                fn = lambda: local._forward_into_buf(x, weights, ids, start, end)
                for mode in ("hot", "cold"):
                    results[name][mode].append(_measure_full_chain(
                        fn, flush if mode == "cold" else None,
                        root / f"{name}_{mode}_{repeat}.trace.json"))
    record = dict(results=results, l2_flush_bytes=flush.numel(),
                  scope="Complete local MoE only; EP/model NOT_RUN; default promotion BLOCKED by independent stress failures",
                  cache_note="256MiB eviction plus synchronization outside both measurements; not an isolated cache-speed claim",
                  timing_note="Events include launch gaps/allocator; profiler kernel sums reported separately; warmup and profiler setup excluded from event spans")
    (root / "result.json").write_text(json.dumps(record, indent=2))
    for name, modes in results.items():
        for mode, samples in modes.items():
            print(f"INDEXED_NATIVE_PERF {name} {mode}: span_ms="
                  f"{statistics.median(s['median_stream_span_ms'] for s in samples):.6f}, "
                  f"kernel_us={statistics.median(s['profile']['kernel_sum_us_per_iter'] for s in samples):.3f}", flush=True)
    return record


@_GPU
@pytest.mark.parametrize("pattern", ["mixed", "duplicate"])
def test_indexed_full_chain_performance(pattern, monkeypatch):
    # Only this previously passing R1 geometry is eligible; rerun all independent
    # gates before timing. Do not benchmark failed pressure fixtures as accurate.
    _run(1, 8, pattern, 5120, 2304, 48, monkeypatch, benchmark=True)


@_GPU
@pytest.mark.parametrize("rows,pattern", [(3, "mixed"), (8, "valid")], ids=["r3-mixed", "r8-valid"])
def test_indexed_checkpoint_routes(rows, pattern, monkeypatch):
    checkpoint_root = os.environ.get("DSV41_INDEXED_CHECKPOINT")
    if not checkpoint_root:
        pytest.skip("Opt-in real packed weights: set DSV41_INDEXED_CHECKPOINT; activation remains synthetic")
    _run(rows, 8, pattern, 5120, 2304, 48, monkeypatch, checkpoint_root=checkpoint_root)

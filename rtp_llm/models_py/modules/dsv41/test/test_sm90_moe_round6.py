"""Bounded persistent-grid experiment: unchanged installed W4A16 kernel.

Synthetic skew is a diagnostic, not measured per-expert production routing.
Root owns GPU/Bazel. No dependency, production, quantization or tolerance edits.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear, decode_quantized_activation, unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input, _independent_weight, _metrics, _quantize, _source_metadata,
)


def _counts(m):
    # Four hot experts receive 60%; 36 Zipf tails share 40%; eight are empty.
    fractions = [.15] * 4
    tail = [1 / (i + 1) ** 1.2 for i in range(36)]
    fractions += [.4 * value / sum(tail) for value in tail] + [0.] * 8
    counts = [math.floor(m * value) for value in fractions]
    residual = m - sum(counts)
    order = sorted(range(40), key=lambda i: m * fractions[i] - counts[i], reverse=True)
    for i in order[:residual]:
        counts[i] += 1
    assert len(counts) == 48 and sum(counts) == m
    return counts


def _offsets(counts, device):
    result = [0]
    for count in counts:
        result.append(result[-1] + count)
    return torch.tensor(result, dtype=torch.int32, device=device)


def _bounded_metrics(actual, reference):
    parts = [_metrics(actual[i:i + 512], reference[i:i + 512])
             for i in range(0, actual.shape[0], 512)]
    return {"elements": sum(p["elements"] for p in parts),
            "unequal": sum(p["unequal"] for p in parts),
            "outside_existing_tolerance": sum(p["outside_existing_tolerance"] for p in parts),
            "max_abs": max(p["max_abs"] for p in parts),
            "finite": all(p["finite"] for p in parts)}


def _oracle(x, weight, scales, counts):
    reference = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=torch.bfloat16)
    begin = 0
    for expert, count in enumerate(counts):
        if count:
            decoded_w = _independent_weight(weight[expert], scales[expert])
            for first in range(begin, begin + count, 512):
                last = min(first + 512, begin + count)
                q, sf = _independent_quantized_input(x[first:last])
                actual_q, actual_sf = _quantize(x[first:last])
                assert torch.equal(q.view(torch.uint8), actual_q.view(torch.uint8))
                assert torch.equal(sf, unpack_activation_scales(actual_sf, x.shape[1] // 128))
                values = (q.float().reshape(last - first, -1, 128) * sf[..., None]).reshape_as(q)
                reference[first:last] = (values @ decoded_w.T).bfloat16()
            del decoded_w
        begin += count
    return reference


def _selected_tuning(linear, m):
    from humming.tune import get_heuristics_config

    # Production passes tuning=None, selecting from this interval table. Do not
    # substitute get_config(m), whose exact-M estimate can differ at boundaries.
    table = get_heuristics_config(linear.config, device=linear.weight.device, **linear.compute_config)
    matches = [(lo, hi, config) for lo, hi, config in table if lo < m <= (hi or 1 << 30)]
    assert len(matches) == 1
    lo, hi, config = matches[0]
    assert config["block_shape"][2] == config["warp_shape"][2] == 32
    assert not config["use_stream_k"] and not config["use_f16_accum"]
    return dict(config), [lo, hi]


def _forward(linear, x, offsets, tuning):
    from humming.forward import humming_forward

    q, sf = _quantize(x)
    return humming_forward(
        linear.config, inputs=decode_quantized_activation(q, sf),
        weight=linear.weight, weight_scale=linear.weight_scale,
        weight_scale_2=getattr(linear, "weight_scale_2", None),
        expert_layout=offsets, valid_shape_m=x.shape[0], locks=linear.locks,
        compute_config=linear.compute_config, tuning_config=tuning,
    )


def _profile(graph, directory, name, expected_grid):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            with torch.profiler.record_function(name + "/full_projection_graph_replay"):
                graph.replay()
        torch.cuda.synchronize()
    path = directory / (name + ".trace.json")
    prof.export_chrome_trace(str(path))
    kernels = [event for event in json.loads(path.read_text())["traceEvents"]
               if event.get("cat") == "kernel" and event.get("ph") == "X"]
    humming = [event for event in kernels if "void humming<" in event["name"]]
    assert len(humming) == 3, [event["name"] for event in kernels]
    assert sum("per_token_group_quant" in event["name"] for event in kernels) == 3
    assert sum("_decode_g128_fp8_to_bf16" in event["name"] for event in kernels) == 3
    grids = [event["args"].get("grid") for event in humming]
    assert all(grid == [expected_grid, 1, 1] for grid in grids), (expected_grid, grids)
    names = sorted(set(event["name"] for event in humming))
    blocks = [event["args"].get("block") for event in humming]
    return {"iterations": 3, "kernel_sum_us_per_iter": sum(e["dur"] for e in kernels) / 3,
            "humming_us_per_iter": sum(e["dur"] for e in humming) / 3,
            "launch_count_per_iter": len(kernels) / 3, "humming_names": names,
            "humming_grid": grids, "humming_block": blocks, "trace_path": str(path)}


def _benchmark(functions, tunings, x, offsets, counts, directory):
    graphs, held = {}, {}
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            held[name] = fn()

    # Change all live activations and route metadata before trusting replay.
    # Immutable expert weights are parameters, not capture-time route inputs.
    original_x, original_offsets = x.clone(), offsets.clone()
    changed_x = (x.float().flip(0) * .5 + .125).bfloat16()
    changed_offsets = _offsets(list(reversed(counts)), x.device)
    assert not torch.equal(original_offsets, changed_offsets)
    replay_checks = {}
    try:
        x.copy_(changed_x)
        offsets.copy_(changed_offsets)
        for name, fn in functions.items():
            eager = fn().clone()
            graphs[name].replay()
            torch.cuda.synchronize()
            assert torch.equal(eager.view(torch.int16), held[name].view(torch.int16)), name
            replay_checks[name] = True
    finally:
        x.copy_(original_x)
        offsets.copy_(original_offsets)
        torch.cuda.synchronize()
    for graph in graphs.values():
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    # All graph pools, events, outputs and JIT work are established before timing.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = {name: [] for name in functions}
    names = list(functions)
    for repeat in range(21):
        for name in names[repeat % 3:] + names[:repeat % 3]:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    profiles = {}
    for name in names:
        tuning = tunings[name]
        grid = tuning["num_sms"] * tuning["num_ctas_per_sm"]
        profiles[name] = _profile(graphs[name], directory, name, grid)
    # Demonstrate the same named CUDA arithmetic kernel/block, different grids.
    assert all(profiles[name]["humming_names"] == profiles[names[0]]["humming_names"] for name in names)
    assert all(profiles[name]["humming_block"] == profiles[names[0]]["humming_block"] for name in names)
    medians = {name: statistics.median(value) for name, value in samples.items()}
    return {"graph_stream_span_ms": samples, "median_graph_stream_span_ms": medians,
            "speedup_vs_production": {name: medians["production"] / medians[name] for name in names},
            "changed_input_metadata_replay_bitwise": replay_checks, "profile": profiles,
            "scope": "full quantize+exact BF16 decode+GEMM captured replay; allocation/warmup/capture excluded; not eager/model latency"}


@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_round6_persistent_grid_skew(n, k):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0), "SM90 GPU lock required"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _run(n, k)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _run(n, k):
    m, e = 32768, 48
    torch.manual_seed(20261016)
    source = _source_metadata()
    assert source["humming_version"] == "0.1.16", source
    import humming
    humming_root = Path(humming.__file__).resolve().parent
    for relative in ("include/humming/scheduler.cuh", "csrc/launcher/launcher.cpp"):
        path = humming_root / relative
        source["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    weight = torch.randint(-128, 128, (e, n, k // 2), dtype=torch.int8, device="cuda")
    raw_scale = torch.randint(117, 128, (e, n, k // 32), dtype=torch.uint8, device="cuda")
    raw_scale[..., 0], raw_scale[..., 1] = 117, 127
    scales = raw_scale.view(torch.float8_e8m0fnu)
    linear = SM90HummingGroupedLinear(weight, scales)
    counts = _counts(m)
    offsets = _offsets(counts, weight.device)
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    base, interval = _selected_tuning(linear, m)
    physical_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    base["num_sms"] = base.get("num_sms", 0) or physical_sms
    tunings = {"production": base,
               "half_sm_grid": dict(base, num_sms=max(1, physical_sms // 2)),
               "double_sm_grid": dict(base, num_sms=2 * physical_sms)}
    assert len({t["num_sms"] for t in tunings.values()}) == 3, tunings
    for tuning in tunings.values():
        assert {key: value for key, value in tuning.items() if key != "num_sms"} == {
            key: value for key, value in base.items() if key != "num_sms"}
    functions = {"production": lambda: linear(x, offsets)}
    for name in ("half_sm_grid", "double_sm_grid"):
        functions[name] = lambda name=name: _forward(linear, x, offsets, tunings[name])
    reference = _oracle(x, weight, scales, counts)
    outputs = {name: fn() for name, fn in functions.items()}
    checks = {name: {"oracle": _bounded_metrics(output, reference),
                     "bitwise_vs_production": torch.equal(output.view(torch.int16), outputs["production"].view(torch.int16)),
                     "repeat_bitwise": all(torch.equal(fn().view(torch.int16), output.view(torch.int16)) for _ in range(3))}
              for (name, fn), output in zip(functions.items(), outputs.values())}
    directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / f"round6_grid_M{m}_N{n}_K{k}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"source": source, "M_routed": m, "E": e, "N": n, "K": k,
              "distribution": "constructed four-hot60%/Zipf36-tail40%/empty8; NOT measured expert routing",
              "expert_counts": counts, "physical_sms": physical_sms, "baseline_dispatch_interval": interval,
              "tunings": tunings, "checks": checks,
              "scope": "projection scheduling only; route-weight/pre-down boundary untouched; full MoE NOT_RUN"}
    valid = all(check["oracle"]["outside_existing_tolerance"] == 0 and check["bitwise_vs_production"]
                and check["repeat_bitwise"] for check in checks.values())
    record["decision"] = "COMPONENT_PASS_NOT_PROMOTED" if valid else "REJECT_NUMERICAL_NO_BENCHMARK"
    path = directory / "result.json"
    path.write_text(json.dumps(record, indent=2))
    print("ROUND6_GRID_CORRECTNESS " + json.dumps(record), flush=True)
    assert valid, record
    del reference, outputs
    record["benchmark"] = _benchmark(functions, tunings, x, offsets, counts, directory)
    path.write_text(json.dumps(record, indent=2))
    print("ROUND6_GRID_PERF " + json.dumps(record["benchmark"]), flush=True)

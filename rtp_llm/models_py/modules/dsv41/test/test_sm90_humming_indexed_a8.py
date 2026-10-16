"""Native A8 indexed candidate gates; root owns serialized GPU execution.

Local Graph replay is not evidence of DP8 scheduling/collective correctness.
No tolerance is relaxed: indexed-vs-fixed is bitwise, dynamic eager .02/.05.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_graph import forward_local_static
from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_indexed_a8 import forward_local_indexed_a8


@pytest.fixture
def sm90(monkeypatch):
    assert torch.cuda.is_available(), "Requires a parent-locked GPU"
    assert torch.cuda.get_device_capability() == (9, 0), "Requires SM90"
    monkeypatch.setenv("DSV41_SM90_NATIVE_A8", "1")
    monkeypatch.setenv("DSV41_SM90_A8_PROMOTION_GROUP", "32")
    torch.manual_seed(180018)


def _local(experts=4, dim=128, inter=128):
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import _GroupedLocalExperts
    from rtp_llm.utils.model_weight import W

    local = _GroupedLocalExperts(SimpleNamespace(swiglu_limit=10.0))
    tensors = {}
    for wk, sk, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, inter, dim),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, inter, dim),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, dim, inter),
    ):
        tensors[wk] = torch.randint(-128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8)
        tensors[sk] = torch.randint(121, 125, (experts, n, k // 32), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
    local.setup_weights(tensors)
    return local


def _bits(actual, expected):
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)


def _check(local, x, weights, ids, start, end):
    actual = forward_local_indexed_a8(local, x, weights, ids, start, end)
    fixed = forward_local_static(local, x, weights, ids, start, end)
    dynamic = local._forward_into_buf(x, weights, ids, start, end)
    _bits(actual, fixed)
    torch.testing.assert_close(actual, dynamic, rtol=.02, atol=.05)
    return actual


@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_indexed_a8_routes_invalid_duplicate_nan(sm90, topk, id_dtype):
    local = _local()
    start, end = 12, 16
    ids = torch.tensor([[12, 12, 15, 14, -1, 17, 13, 12], [-1] * 8],
                       device="cuda", dtype=id_dtype)[:, :topk].contiguous()
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    x[1] = float("nan")
    weights = torch.rand((2, topk), device="cuda")
    weights.masked_fill_((ids < start) | (ids >= end), float("nan"))
    actual = _check(local, x, weights, ids, start, end)
    assert torch.count_nonzero(actual[1]) == 0
    ids.fill_(-1)
    x.fill_(float("nan"))
    weights.fill_(float("nan"))
    assert torch.count_nonzero(_check(local, x, weights, ids, start, end)) == 0


@pytest.mark.parametrize("rows", [1, 8])
def test_indexed_a8_graph_replay_all_live_inputs(sm90, rows):
    local = _local()
    x = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((rows, 8), device="cuda")
    ids = torch.arange(rows * 8, device="cuda").reshape(rows, 8) % 4
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward_local_indexed_a8(local, x, weights, ids, 0, 4)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_local_indexed_a8(local, x, weights, ids, 0, 4)
    for step in range(5):
        # Reuse addresses, not values. Exercise all empty -> live again too.
        ids.copy_(torch.randint(-1, 5, ids.shape, device="cuda"))
        if step == 1:
            ids.fill_(2)  # repeated expert with distinct original route slots
        if step == 2:
            ids.fill_(-1)
        if step == 3:
            ids[0].fill_(2**40)  # int64 invalid IDs must not narrow/wrap
        x.copy_(torch.randn_like(x) * (step + 1))
        weights.copy_(torch.rand_like(weights))
        valid = (ids >= 0) & (ids < 4)
        weights.masked_fill_(~valid, float("nan"))
        x.masked_fill_(~valid.any(1)[:, None], float("nan"))
        expected = _check(local, x, weights, ids, 0, 4)
        graph.replay()
        torch.cuda.synchronize()
        _bits(captured, expected)
        if step == 2:
            assert torch.count_nonzero(captured) == 0


def test_indexed_a8_empty_rows_and_guards(sm90):
    local = _local()
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    ids = torch.empty((0, 8), device="cuda", dtype=torch.int64)
    weights = torch.empty((0, 8), device="cuda")
    out = forward_local_indexed_a8(local, x, weights, ids, 0, 4)
    assert out.shape == x.shape and out.dtype == torch.float32
    local.native_a8 = False
    with pytest.raises(ValueError, match="native A8"):
        forward_local_indexed_a8(local, x, weights, ids, 0, 4)
    local.native_a8 = True
    local.w13.promotion_group_size = 128
    with pytest.raises(ValueError, match="K32"):
        forward_local_indexed_a8(local, x, weights, ids, 0, 4)


def test_indexed_a8_actual_shape(sm90):
    """Actual E48/D5120/I2304, fixed recv8/topk8; still not a model/DP gate."""
    local = _local(experts=48, dim=5120, inter=2304)
    x = torch.randn((8, 5120), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((8, 8), device="cuda")
    ids = torch.full((8, 8), -1, device="cuda", dtype=torch.int64)
    ids[0, :6] = torch.tensor([0, 1, 2, 3, 4, 5], device="cuda")
    ids[1, :6] = torch.tensor([47, 47, 0, 1, 2, 3], device="cuda")
    weights.masked_fill_(ids < 0, float("nan"))
    x[2:] = float("nan")
    _check(local, x, weights, ids, 0, 48)


@pytest.mark.skipif(
    os.environ.get("DSV41_SM90_INDEXED_A8_BENCH", "0") != "1",
    reason="Opt-in microbenchmark; not part of default arithmetic acceptance",
)
@pytest.mark.parametrize("pattern", ["uniform", "sparse", "empty"])
def test_indexed_a8_graph_microbenchmark(sm90, tmp_path, pattern):
    """Whole local chain, hot weights, Graph event span; NOT model/EP TPOT.

    Set DSV41_SM90_INDEXED_A8_BENCH=1 to enable. Optional separate traces use
    DSV41_SM90_INDEXED_A8_TRACE=1. No profiler is active during clean timing.
    """
    local = _local(experts=48, dim=5120, inter=2304)
    rows, topk = 8, 8
    x = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((rows, topk), device="cuda")
    ids = (torch.arange(rows * topk, device="cuda").reshape(rows, topk) % 48).contiguous()
    if pattern != "uniform":
        ids.fill_(-1)
        if pattern == "sparse":
            ids[0, :6] = torch.tensor([0, 1, 2, 3, 4, 5], device="cuda")
    valid = ids >= 0
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(1)[:, None], float("nan"))
    # Acceptance first. A timing result cannot waive numerical differences.
    _check(local, x, weights, ids, 0, 48)
    functions = {"grouped": forward_local_static, "indexed": forward_local_indexed_a8}
    graphs, outputs = {}, {}
    for name, function in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(local, x, weights, ids, 0, 48)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = function(local, x, weights, ids, 0, 48)
        graphs[name] = graph
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
    _bits(outputs["indexed"], outputs["grouped"])
    iterations = int(os.environ.get("DSV41_SM90_INDEXED_A8_BENCH_ITERS", "100"))
    repeats = int(os.environ.get("DSV41_SM90_INDEXED_A8_BENCH_REPEATS", "5"))
    assert iterations > 0 and repeats >= 3, "Need positive iterations and >=3 paired repetitions"
    samples = {name: [] for name in functions}
    for repeat in range(repeats):
        order = ("grouped", "indexed") if repeat % 2 == 0 else ("indexed", "grouped")
        for name in order:
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    _bits(outputs["indexed"], outputs["grouped"])
    directory = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    traces = {}
    if os.environ.get("DSV41_SM90_INDEXED_A8_TRACE", "0") == "1":
        for name, graph in graphs.items():
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]) as profiler:
                with torch.profiler.record_function(f"indexed_a8_{pattern}_{name}_graph"):
                    graph.replay()
                torch.cuda.synchronize()
            trace = directory / f"indexed_a8_{pattern}_{name}.trace.json"
            profiler.export_chrome_trace(str(trace))
            traces[name] = str(trace)
    record = {
        "scope": "LOCAL_ONLY_HOT_WEIGHT_GRAPH_EVENT_SPAN_NOT_KERNEL_SUM_OR_TPOT",
        "shape": {"experts": 48, "dim": 5120, "inter": 2304, "recv_rows": rows, "topk": topk},
        "pattern": pattern, "iterations": iterations, "repeats": repeats,
        "route_valid_slots": int(valid.sum().item()),
        "activation_quant": "original_g128_payload_K32_promotion",
        "clean_ms_per_graph": samples, "separate_profile_traces": traces,
        "numerical_gate": "indexed_vs_fixed_bitwise_and_dynamic_rtol.02_atol.05",
    }
    (directory / f"indexed_a8_{pattern}_bench.json").write_text(json.dumps(record, indent=2))
    print("SM90_INDEXED_A8_BENCH " + json.dumps(record), flush=True)

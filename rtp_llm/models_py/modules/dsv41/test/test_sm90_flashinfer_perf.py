"""Diagnostic local-MoE Graph benchmark; not a full-model TPOT estimate.

Both arms consume identical RANDOM packed MXFP4 weights, NOT checkpoint
weights. Existing BF16 mathematical .02/.05 gates remain mandatory before
timing. FI row-A8/finalize weighting and RTP G128/pre-down weighting differ;
no bitwise equivalence between those quantization schemes is asserted.
Run explicitly with DSV41_FLASHINFER_PERF=1. Parent owns GPU execution.
"""

import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.test.test_sm90_flashinfer_local import (
    _bits,
    _reference,
    _report_and_check,
    _sha,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("DSV41_FLASHINFER_PERF") != "1",
    reason="opt-in real-shape local MoE microbenchmark",
)


@pytest.fixture(scope="module")
def pair():
    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_moe import (
        Sm90FlashInferLocalExperts,
    )
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        _GroupedLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    patch = pytest.MonkeyPatch()
    patch.setenv("DSV41_SM90_NATIVE_A8", "1")
    patch.setenv("DSV41_SM90_A8_PROMOTION_GROUP", "32")
    cfg = SimpleNamespace(
        n_routed_experts=384,
        ep_size=8,
        dim=5120,
        moe_inter_dim=2304,
        swiglu_limit=10.0,
    )
    torch.manual_seed(26092)  # Same fixed real-shape random stress as local gate.
    raw = {}
    for wk, sk, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, 2304, 5120),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, 2304, 5120),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, 5120, 2304),
    ):
        raw[wk] = torch.randint(
            0, 256, (48, n, k // 2), device="cuda", dtype=torch.uint8
        ).view(torch.int8)
        raw[sk] = torch.randint(
            121, 125, (48, n, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    hashes = {str(k): _sha(v) for k, v in raw.items()}
    fi = Sm90FlashInferLocalExperts(cfg)
    old = _GroupedLocalExperts(cfg)
    fi.setup_weights(dict(raw))
    old.setup_weights(dict(raw))
    assert old.native_a8
    for linear in (old.w13, old.w2):
        assert linear.promotion_group_size == 32
        assert str(linear.config.a_dtype) == "float8e4m3"
        assert linear.compute_config["use_batch_invariant"]
        assert not linear.compute_config["use_f16_accum"]
    fi.prepare(8, 8, tune=True)
    try:
        yield SimpleNamespace(model=fi, old=old, raw=raw, cfg=cfg)
        assert hashes == {str(k): _sha(v) for k, v in raw.items()}
    finally:
        patch.undo()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _inputs(valid_total):
    torch.manual_seed(26149)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16) * 0.125
    weights = torch.rand(8, 8, device="cuda", dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    indices = torch.full((8, 8), -1, device="cuda", dtype=torch.int64)
    # Column-major placement spreads six/eight TOTAL local routes over rows.
    # It does not confuse six routes per row with the EP8 mean of six/rank.
    for j in range(valid_total):
        row, slot = j % 8, j // 8
        indices[row, slot] = (0, 47, 7, 39, 15, 31, 23, 1)[j % 8]
    valid = indices >= 0
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(-1)[:, None], float("nan"))
    return x, weights, indices


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def _measure(graph, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


@pytest.mark.parametrize("valid_total", [6, 8, 48, 64])
@torch.no_grad()
def test_same_routes_local_graph_benchmark(pair, valid_total):
    from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_graph import (
        forward_local_static,
    )

    inputs = _inputs(valid_total)
    saved = tuple(_sha(t) for t in inputs)
    arms = {
        "flashinfer_opensource": lambda: pair.model.forward_local(
            *inputs, output_dtype=torch.bfloat16
        ),
        "standalone_humming": lambda: forward_local_static(
            pair.old, *inputs, 0, 48, output_dtype=torch.bfloat16
        ),
    }
    expected = _reference(pair, *inputs)
    captured = {}
    # Numerical or Graph failure stops timing; it is never marked a perf win.
    for name, fn in arms.items():
        eager = fn()
        _report_and_check(
            eager.float(), expected, label=f"perf_gate_{name}_valid{valid_total}"
        )
        graph, output = _capture(fn)
        graph.replay()
        torch.cuda.synchronize()
        _bits(output, eager)
        captured[name] = graph, output

    # Reuse each graph with changed x, IDs and weights, including poisoned
    # empty rows. The same live allocations feed both arms throughout.
    for count in (0, 8, valid_total):
        updated = _inputs(count)
        updated[0].neg_()
        updated[1].mul_(0.75)
        updated[2].copy_(torch.where(updated[2] >= 0, 47 - updated[2], updated[2]))
        for dst, src in zip(inputs, updated):
            dst.copy_(src)
        for name, fn in arms.items():
            graph, output = captured[name]
            eager = fn()
            graph.replay()
            torch.cuda.synchronize()
            _bits(output, eager)
            if count == 0:
                _bits(output, torch.zeros_like(output))
            else:
                _report_and_check(
                    output.float(),
                    _reference(pair, *inputs),
                    label=f"perf_live_gate_{name}_valid{count}",
                )
    for dst, src in zip(inputs, _inputs(valid_total)):
        dst.copy_(src)
    assert saved == tuple(_sha(t) for t in inputs)
    for graph, _ in captured.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    iterations = int(os.environ.get("DSV41_FLASHINFER_PERF_ITERS", "100"))
    assert iterations >= 1
    samples = {name: [] for name in arms}
    # Alternate ordering to avoid always assigning one arm hot/cold clocks.
    for repeat in range(7):
        names = list(arms)
        if repeat % 2:
            names.reverse()
        for name in names:
            samples[name].append(_measure(captured[name][0], iterations))
    report = dict(
        evidence="RTP_MEASURED",
        scope="local graph only; no DeepEP communication; not TPOT",
        weights="random I8 MXFP4/E8M0 bytes121..124; NOT checkpoint",
        shape=dict(rows=8, experts=48, dim=5120, inter=2304, topk=8),
        total_valid_routes=valid_total,
        timing="CUDA events; complete local Graph; no oracle/copy/profile in timing",
        output_boundary="BF16 normal-combine boundary for both arms",
        numerical_gate="BF16 math rtol=.02 atol=.05 PASS; Graph replay bitexact PASS",
        quantization_equivalence="NOT_EQUIVALENT: row-A8/finalize vs G128/hidden weighting",
        samples_us=samples,
        median_us={name: statistics.median(values) for name, values in samples.items()},
    )
    print("FLASHINFER_LOCAL_PERF " + json.dumps(report), flush=True)
    output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"flashinfer_local_perf_valid{valid_total}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

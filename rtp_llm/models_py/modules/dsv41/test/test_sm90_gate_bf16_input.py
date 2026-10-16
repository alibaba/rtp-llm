"""SM90 exact gates for eliminating the materialized router-score cast.

Calls the actual dsv41 fused epilogue, not the generic/SM100 gate module.
Production Gate.forward is unchanged. Both arms retain FP32 epilogue math and
FP32 weights/INT64 routes; only the input's BF16->FP32 conversion location
differs. Parent owns GPU/Bazel. Timing is cast+gate only, not E2E TPOT.
"""

import json
import os
import statistics

import pytest
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41._gate_fused_triton import (
    fused_sqrtsoftplus_gate,
)


@pytest.fixture(autouse=True)
def _sm90_only():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    ), "Requires a parent-locked SM90 GPU; missing hardware is not PASS"


def _run(scores, bias, topk=6, *, materialize=False):
    return fused_sqrtsoftplus_gate(
        (scores.float() if materialize else scores).contiguous(),
        bias.contiguous(),
        topk=topk,
        route_scale=2.5,
        norm_eps=1e-12,
    )


def _bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
    b = torch.empty(expected.numel(), device=expected.device, dtype=expected.dtype)
    a.copy_(actual.reshape(-1))
    b.copy_(expected.reshape(-1))
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


def _outputs_exact(actual, expected):
    assert actual[0].dtype == torch.float32 and actual[1].dtype == torch.int64
    for a, e in zip(actual, expected):
        _bits(a, e)


def _random_case(rows, experts=384, *, seed=24091):
    torch.manual_seed(seed)
    scores = torch.randn(rows, experts, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(experts, device="cuda", dtype=torch.float32) * 0.1
    return scores, bias


def _independent_finite_reference(scores, bias, topk):
    original = F.softplus(scores.float()).sqrt()
    indices = (original + bias).topk(topk, dim=-1).indices
    weights = original.gather(1, indices)
    return weights / (weights.sum(-1, keepdim=True) + 1e-12) * 2.5, indices


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 8, 32, 33])
@pytest.mark.parametrize("experts,topk", [(384, 6), (256, 8), (385, 6)])
@torch.no_grad()
def test_finite_scores_exact_old_chain_and_independent_reference(rows, experts, topk):
    scores, bias = _random_case(rows, experts)
    saved = scores.clone(), bias.clone()
    actual = _run(scores, bias, topk)
    _outputs_exact(actual, _run(scores, bias, topk, materialize=True))
    expected_weights, expected_indices = _independent_finite_reference(
        scores, bias, topk
    )
    torch.testing.assert_close(actual[1], expected_indices, rtol=0, atol=0)
    if rows:
        # Existing generic gate acceptance: max absolute error / mean reference
        # magnitude < 1e-4. New-vs-old fused outputs above must still be bit exact.
        error = (actual[0] - expected_weights).abs().max()
        magnitude = expected_weights.abs().mean() + 1e-9
        assert float(error / magnitude) < 1e-4
    _bits(scores, saved[0])
    _bits(bias, saved[1])


@torch.no_grad()
def test_extreme_scores_bias_nonfinite_threshold_ties_and_signed_zero():
    scores, bias = _random_case(16)
    special = torch.tensor(
        [
            0.0,
            -0.0,
            2**-133,
            -(2**-133),
            2**-126,
            -(2**-126),
            -100,
            -20,
            19.875,
            20,
            20.125,
            100,
            torch.finfo(torch.bfloat16).max,
            -torch.finfo(torch.bfloat16).max,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    scores[3:7] = special[torch.arange(384, device="cuda") % special.numel()]
    scores[0, 3] = float("nan")
    scores[1, 7] = float("inf")
    scores[2, 11] = -float("inf")
    scores[7].zero_()
    scores[8].fill_(-0.0)
    scores[9].fill_(20)
    scores[10].fill_(-100)
    for poison in (None, float("nan"), float("inf"), -float("inf")):
        active_bias = bias.clone() if poison is not None else torch.zeros_like(bias)
        if poison is not None:
            active_bias[17] = poison
        saved_scores, saved_bias = scores.clone(), active_bias.clone()
        actual = _run(scores, active_bias)
        _outputs_exact(actual, _run(scores, active_bias, materialize=True))
        invalid_rows = range(16) if poison is not None else range(3)
        # The existing fused contract has a deterministic safe fallback, not
        # PyTorch topk's unspecified tie order on sanitized nonfinite scores.
        for row in invalid_rows:
            _bits(actual[1][row], torch.arange(6, device="cuda", dtype=torch.int64))
            _bits(actual[0][row], torch.full((6,), 2.5 / 6, device="cuda"))
        if poison is None:
            for row in (7, 8, 9, 10):
                _bits(actual[1][row], torch.arange(6, device="cuda", dtype=torch.int64))
        assert bool(torch.isfinite(actual[0]).all())
        _bits(scores, saved_scores)
        _bits(active_bias, saved_bias)


@torch.no_grad()
def test_all_bf16_bit_patterns_preserve_old_fused_outputs():
    # Every BF16 payload, including signaling/quiet NaN variants, appears as
    # an actual score. Avoid assuming Torch and Triton preserve NaN payloads
    # on widening: the observable fallback and route outputs must agree.
    scores = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    scores = scores.view(torch.bfloat16).reshape(256, 256)
    bias = torch.zeros(256, device="cuda")
    _outputs_exact(_run(scores, bias), _run(scores, bias, materialize=True))


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


@pytest.mark.parametrize("rows", [1, 4, 32])
@torch.no_grad()
def test_graph_changes_scores_and_bias_exact_old_chain(rows):
    scores, bias = _random_case(rows)
    graph, output = _capture(lambda: _run(scores, bias))
    old_graph, old_output = _capture(lambda: _run(scores, bias, materialize=True))
    for phase in range(5):
        replacement_scores, replacement_bias = _random_case(rows, seed=109 + phase)
        scores.copy_(replacement_scores)
        bias.copy_(replacement_bias)
        if phase == 1:
            scores.zero_()
            bias.zero_()
        elif phase == 2:
            scores[0, 1] = float("nan")
        elif phase == 3:
            bias[17] = float("inf")
        snapshot = scores.clone(), bias.clone()
        graph.replay()
        old_graph.replay()
        _outputs_exact(output, old_output)
        _outputs_exact(output, _run(scores, bias, materialize=True))
        _bits(scores, snapshot[0])
        _bits(bias, snapshot[1])


@pytest.mark.skipif(
    os.getenv("DSV41_SM90_GATE_BF16_BENCH") != "1",
    reason="opt-in isolated component timing",
)
@torch.no_grad()
def test_clean_graph_cast_and_gate_benchmark():
    for rows in (1, 2, 4, 8, 32):
        scores, bias = _random_case(rows)
        old_graph, old_output = _capture(lambda: _run(scores, bias, materialize=True))
        new_graph, new_output = _capture(lambda: _run(scores, bias))
        _outputs_exact(new_output, old_output)
        graphs, times = (old_graph, new_graph), [[], []]
        for repeat in range(6):
            for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(200):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                times[arm].append(start.elapsed_time(end) * 1000 / 200)
        _outputs_exact(new_output, old_output)
        old_us, new_us = map(statistics.median, times)
        print(
            "SM90_GATE_BF16_BENCH "
            + json.dumps(
                dict(
                    rows=rows,
                    experts=384,
                    topk=6,
                    old_us=old_us,
                    candidate_us=new_us,
                    speedup=old_us / new_us,
                    scope="cast+gate only, no GEMM/dispatch or E2E TPOT",
                )
            ),
            flush=True,
        )

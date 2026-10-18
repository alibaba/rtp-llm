"""Exact copy-only transport padding gates, without changing production hooks.

No expert selection, precision, normal DeepEP protocol, or GEMM is replaced.
Benchmarks cover the padding component only, not communication or E2E TPOT.
Parent owns GPU/Bazel execution.
"""

import json
import os
import statistics

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_dispatch_pad import pad_deepep_routes


@pytest.fixture(autouse=True)
def _sm90_only():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    ), "Requires a parent-locked SM90 GPU; missing hardware is not PASS"


def _reference(indices, weights, pad_to=8):
    # The old strategy's exact two-fill/two-cat expression for unsupported
    # widths, also defining the standalone fresh-copy contract for equal K.
    rows, topk = indices.shape
    return (
        torch.cat((indices, indices.new_full((rows, pad_to - topk), -1)), -1),
        torch.cat((weights, weights.new_zeros((rows, pad_to - topk))), -1),
    )


def _bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a = torch.empty(actual.numel(), dtype=actual.dtype, device=actual.device)
    b = torch.empty(expected.numel(), dtype=expected.dtype, device=expected.device)
    a.copy_(actual.reshape(-1))
    b.copy_(expected.reshape(-1))
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


def _case(rows, topk, layout="contiguous", phase=0):
    count = rows * topk
    # IDs exceed INT32 to catch accidental truncation; negative original slots
    # are copied verbatim rather than sanitized or treated as new padding.
    ids = torch.tensor([-1, 0, 47, 383, 2**40 + 17, -(2**40), 7], device="cuda")
    # Includes distinct NaN payloads, signed zero, infinities, subnormal, 1/-1.
    patterns = torch.tensor(
        [
            0,
            -2147483648,
            0x7FC00001,
            -4194302,
            0x7F800000,
            -8388608,
            1,
            0x3F800000,
            -1082130432,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    offset = torch.arange(count, device="cuda") + phase
    indices = ids[offset % ids.numel()].reshape(rows, topk)
    weights = (
        patterns[offset % patterns.numel()].view(torch.float32).reshape(rows, topk)
    )
    if layout == "strided":
        ib = torch.full(
            (rows * 2 + 3, topk * 3 + 5), 999, device="cuda", dtype=torch.int64
        )
        wb = torch.full((rows * 3 + 4, topk * 2 + 7), 123.0, device="cuda")
        iv = ib[1 : 1 + rows * 2 : 2, 2 : 2 + topk * 3 : 3]
        wv = wb[2 : 2 + rows * 3 : 3, 3 : 3 + topk * 2 : 2]
        iv.copy_(indices)
        wv.copy_(weights)
        return iv, wv
    if layout == "transpose":
        return indices.T.contiguous().T, weights.T.contiguous().T
    if layout == "broadcast" and rows:
        return indices[:1].expand(rows, -1), weights[:1].expand(rows, -1)
    return indices, weights


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 4, 8, 31, 32, 33, 257])
@pytest.mark.parametrize(
    "topk,pad_to",
    [(1, 2), (2, 2), (3, 4), (4, 8), (6, 8), (7, 8), (8, 8), (15, 16), (16, 16)],
)
def test_copy_sentinel_bits_and_fresh_storage(rows, topk, pad_to):
    indices, weights = _case(rows, topk)
    snapshots = indices.clone(), weights.clone()
    actual = pad_deepep_routes(indices, weights, pad_to)
    expected = _reference(indices, weights, pad_to)
    for a, e, source in zip(actual, expected, (indices, weights)):
        _bits(a, e)
        assert a.is_contiguous()
        if rows:
            assert a.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    for source, snapshot in zip((indices, weights), snapshots):
        _bits(source, snapshot)
    if rows:
        actual[0].fill_(5)
        actual[1].fill_(6)
        for source, snapshot in zip((indices, weights), snapshots):
            _bits(source, snapshot)


@pytest.mark.parametrize("rows", [0, 1, 2, 8, 32])
@pytest.mark.parametrize("layout", ["strided", "transpose", "broadcast"])
def test_strided_inputs_and_untouched_guard_storage(rows, layout):
    indices, weights = _case(rows, 6, layout)
    bases = tuple(x._base if x._base is not None else x for x in (indices, weights))
    snapshots = tuple(x.clone() for x in bases)
    for actual, expected in zip(
        pad_deepep_routes(indices, weights), _reference(indices, weights)
    ):
        _bits(actual, expected)
    for base, snapshot in zip(bases, snapshots):
        _bits(base, snapshot)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_graph_changes_every_live_input(rows, layout):
    indices, weights = _case(rows, 6, layout)
    for _ in range(3):
        pad_deepep_routes(indices, weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = pad_deepep_routes(indices, weights)
    for phase in (1, 5, 0, 11):
        replacement = _case(rows, 6, phase=phase)
        indices.copy_(replacement[0])
        weights.copy_(replacement[1])
        if phase == 5:
            indices.fill_(-1)
        graph.replay()
        for a, e in zip(actual, _reference(indices, weights)):
            _bits(a, e)
        _bits(indices, torch.full_like(indices, -1) if phase == 5 else replacement[0])
        _bits(weights, replacement[1])


def test_invalid_metadata_rejected():
    indices, weights = _case(2, 6)
    cases = [
        (indices.int(), weights, 8),
        (indices, weights.bfloat16(), 8),
        (indices, weights[:, :5], 8),
        (indices.flatten(), weights.flatten(), 8),
        (indices, weights.cpu(), 8),
        (indices.cpu(), weights.cpu(), 8),
        (indices[:, :0], weights[:, :0], 8),
        (indices, weights, 4),
        (indices, weights, 7),
        (indices, weights, True),
        (indices, weights, 8.0),
    ]
    for i, w, width in cases:
        with pytest.raises(ValueError, match="matching CUDA INT64/FP32"):
            pad_deepep_routes(i, w, width)


@pytest.mark.skipif(
    os.getenv("DSV41_SM90_DISPATCH_PAD_BENCH") != "1",
    reason="opt-in isolated component timing",
)
def test_clean_graph_padding_benchmark():
    for rows in (1, 2, 4, 8, 32):
        indices, weights = _case(rows, 6)
        graphs, outputs = [], []
        for fn in (_reference, pad_deepep_routes):
            for _ in range(5):
                fn(indices, weights)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn(indices, weights)
            graphs.append(graph)
            outputs.append(output)
        for a, e in zip(outputs[1], outputs[0]):
            _bits(a, e)
        times = [[], []]
        for round_id in range(6):
            for arm in (0, 1) if round_id % 2 == 0 else (1, 0):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(200):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                times[arm].append(start.elapsed_time(end) * 1000 / 200)
        for a, e in zip(outputs[1], outputs[0]):
            _bits(a, e)
        old_us, new_us = map(statistics.median, times)
        print(
            "SM90_DISPATCH_PAD_BENCH "
            + json.dumps(
                dict(
                    rows=rows,
                    topk=6,
                    transport_topk=8,
                    old_us=old_us,
                    candidate_us=new_us,
                    speedup=old_us / new_us,
                    scope="padding only; no communication or E2E",
                )
            ),
            flush=True,
        )

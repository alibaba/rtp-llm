"""Independent old-expression gates for bounded SM90 graph routing glue.

The quantizers, W4A8/K32 GEMMs, normal DeepEP protocol and BF16 combine
boundary are not substituted. Parent owns GPU/Bazel execution. Component
timings explicitly exclude communication; no E2E TPOT is inferred from them.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_graph import (
    forward_local_static,
    static_group_routes,
)
from rtp_llm.models_py.modules.dsv41.moe._sm90_route_kernels import (
    prepare_static_routes,
    reduce_routes,
    reduce_static_routes,
    routed_swiglu,
    static_route_metadata,
)


@pytest.fixture(autouse=True)
def _gpu_contract(monkeypatch):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    ), "Requires a parent-locked SM90 GPU; missing hardware is not PASS"
    monkeypatch.setenv("DSV41_SM90_NATIVE_A8", "1")
    monkeypatch.setenv("DSV41_SM90_A8_PROMOTION_GROUP", "32")
    torch.manual_seed(24901)


def _metadata_reference(indices, start, end, *, build_inverse=True):
    flat = indices.reshape(-1)
    valid = (flat >= start) & (flat < end)
    expert = flat - start
    keys = torch.where(valid, expert, end - start)
    slots = torch.argsort(keys, stable=True)
    counts = (
        (expert[:, None] == torch.arange(end - start, device=flat.device)[None])
        & valid[:, None]
    ).sum(0, dtype=torch.int32)
    counts[-1] += (~valid).sum(dtype=torch.int32)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    inverse = None
    if build_inverse:
        inverse = torch.empty_like(slots, dtype=torch.int32)
        inverse[slots] = torch.arange(
            slots.numel(), device=slots.device, dtype=torch.int32
        )
    return slots, offsets, valid, inverse


def _prepare_reference(x, weights, indices, start, end, *, timing=False):
    slots, offsets, valid, inverse = _metadata_reference(
        indices, start, end, build_inverse=not timing
    )
    grouped_valid = valid.index_select(0, slots)
    grouped_x = torch.where(
        grouped_valid[:, None],
        x.index_select(0, slots // indices.shape[1]),
        0,
    ).contiguous()
    clean_weights = torch.where(valid.reshape_as(weights), weights, 0).contiguous()
    return slots, offsets, valid, inverse, grouped_x, clean_weights


def _sum_reference(down, slots, valid, rows, topk, dtype):
    # Independent old ATen expression, including select rather than NaN * 0.
    clean_down = torch.where(valid[slots, None], down, 0)
    scratch = torch.zeros(
        (rows * topk, down.shape[1]), device=down.device, dtype=torch.float32
    )
    scratch.index_copy_(0, slots, clean_down.float())
    return scratch.reshape(rows, topk, down.shape[1]).sum(1).to(dtype)


def _bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    # contiguous() can legally retain a non-unit stride for singleton/empty
    # dimensions. Reinterpret only fresh canonical 1-D storage, preserving
    # the original dtype and all value bits (including NaN payloads).
    actual_flat = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
    expected_flat = torch.empty(
        expected.numel(), device=expected.device, dtype=expected.dtype
    )
    actual_flat.copy_(actual.reshape(-1))
    expected_flat.copy_(expected.reshape(-1))
    torch.testing.assert_close(
        actual_flat.view(torch.uint8),
        expected_flat.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _inputs(rows, topk=8, dim=5120, *, start=192, strided=False, all_invalid=False):
    x = torch.randn(
        rows, dim * (2 if strided else 1), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.rand(rows, topk * (2 if strided else 1), device="cuda")
    indices = torch.randint(
        start, start + 48, weights.shape, device="cuda", dtype=torch.int64
    )
    if strided:
        x, weights, indices = x[:, ::2], weights[:, ::2], indices[:, ::2]
    indices[:, 0] = start
    if topk > 1:
        indices[:, 1] = start  # stable duplicate expert order
    indices[:, min(6, topk) :] = -1
    if rows > 1:
        indices[1].fill_(-1)
    if rows > 2:
        indices[2, -1] = start + 48  # nonlocal positive ID
    if all_invalid:
        indices.fill_(-1)
    valid = (indices >= start) & (indices < start + 48)
    weights.masked_fill_(~valid, float("nan"))
    x.masked_fill_(~valid.any(1)[:, None], float("nan"))
    return x, weights, indices


@pytest.mark.parametrize(
    "rows,topk",
    [(0, 8), (1, 1), (1, 8), (2, 8), (3, 6), (4, 8), (7, 8), (8, 8), (31, 8), (32, 8)],
)
@pytest.mark.parametrize("strided", [False, True])
def test_metadata_gather_original_bits_and_no_input_mutation(rows, topk, strided):
    for all_invalid in (False, True):
        x, weights, indices = _inputs(
            rows, topk, dim=513, strided=strided, all_invalid=all_invalid
        )
        snapshots = [tensor.clone() for tensor in (x, weights, indices)]
        expected = _prepare_reference(x, weights, indices, 192, 240)
        actual = prepare_static_routes(x, weights, indices, 192, 240)
        for got, want in zip(actual, expected):
            _bits(got, want)
            assert got.is_contiguous()
            for source in (x, weights, indices):
                assert not torch._C._overlaps(got, source)
        for tensor, saved in zip((x, weights, indices), snapshots):
            _bits(tensor, saved)
        # Normal DeepEP's local recv_idx no longer needs a global-ID tensor.
        local_idx = torch.where(indices >= 0, indices - 192, -1).to(torch.int32)
        local = prepare_static_routes(x, weights, local_idx, 0, 48)
        for got, want in zip(local, actual):
            _bits(got, want)


@pytest.mark.parametrize(
    "rows,topk,dim",
    [(1, 8, 5120), (4, 8, 513), (8, 8, 5120), (32, 8, 129), (3, 6, 255)],
)
@pytest.mark.parametrize("down_dtype", [torch.bfloat16, torch.float32])
def test_reduction_same_bits_cancellation_invalid_nan_and_bf16_boundary(
    rows, topk, dim, down_dtype
):
    _, _, indices = _inputs(rows, topk, dim=dim)
    slots, _, valid, inverse = static_route_metadata(indices, 192, 240)
    storage = torch.randn(rows * topk, dim + 17, device="cuda", dtype=down_dtype)
    down = storage[:, :dim]
    # A reduction-tree change loses these small terms. Values are per original
    # slot and reordered exactly as the expert output would be.
    pattern = torch.tensor([2.0**100, 1, -(2.0**100), 2, 3, -4, 5, -6], device="cuda")
    down[:, 0] = pattern[slots % topk].to(down_dtype)
    down.masked_fill_(~valid[slots, None], float("nan"))
    for dtype in (torch.float32, torch.bfloat16):
        expected = _sum_reference(down, slots, valid, rows, topk, dtype)
        actual = reduce_static_routes(
            down, inverse, valid, rows, topk, output_dtype=dtype
        )
        _bits(actual, expected)
        assert torch.isfinite(actual).all()
        assert not torch._C._overlaps(actual, down)
    # A valid NaN must remain a NaN; cleanup is restricted to dummy slots.
    first_valid_route = torch.nonzero(valid[slots], as_tuple=False)[0, 0]
    down[first_valid_route, 1] = float("nan")
    actual = reduce_static_routes(down, inverse, valid, rows, topk)
    expected = _sum_reference(down, slots, valid, rows, topk, torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


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


def test_graph_refreshes_all_live_inputs_metadata_padding_and_empty_experts():
    for rows in (1, 4, 8, 32):
        x, weights, indices = _inputs(rows, strided=True)
        down = torch.randn(rows * 8, 5120, device="cuda", dtype=torch.bfloat16)

        def chain():
            prepared = prepare_static_routes(x, weights, indices, 192, 240)
            slots, offsets, valid, inverse, grouped_x, clean_weights = prepared
            reduced = reduce_static_routes(
                down, inverse, valid, rows, 8, output_dtype=torch.bfloat16
            )
            return (*prepared, reduced)

        graph, captured = _capture(chain)
        for step in range(4):
            new_x, new_weights, new_indices = _inputs(rows, all_invalid=step == 2)
            if step == 1:
                new_indices[:, :6] = 239  # only last expert active
                new_x.normal_()
                new_weights.normal_()
            x.copy_(new_x)
            weights.copy_(new_weights)
            indices.copy_(new_indices)
            down.normal_()
            expected = _prepare_reference(x, weights, indices, 192, 240)
            slots, _, valid, _ = expected[:4]
            down.masked_fill_(~valid[slots, None], float("nan"))
            reference_sum = _sum_reference(down, slots, valid, rows, 8, torch.bfloat16)
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(captured, (*expected, reference_sum)):
                _bits(got, want)
            assert torch.isfinite(captured[-1]).all()


def _make_local(experts=48, dim=128, inter=128):
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        _GroupedLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    local = _GroupedLocalExperts(SimpleNamespace(swiglu_limit=10.0))
    tensors = {}
    for wk, sk, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, inter, dim),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, inter, dim),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, dim, inter),
    ):
        tensors[wk] = torch.randint(
            -128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8
        )
        tensors[sk] = torch.randint(
            121, 125, (experts, n, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    local.setup_weights(tensors)
    assert local.native_a8
    assert local.w13.promotion_group_size == local.w2.promotion_group_size == 32
    return local


def _old_local(local, x, weights, indices, start, end, *, output_dtype, timing=False):
    if timing:
        slots, offsets, valid, _ = _metadata_reference(
            indices, start, end, build_inverse=False
        )
        grouped_valid = valid.index_select(0, slots)
        grouped_x = torch.where(
            grouped_valid[:, None],
            x.index_select(0, slots // indices.shape[1]),
            0,
        ).contiguous()
        clean_weights = torch.where(valid.reshape_as(weights), weights, 0).contiguous()
    else:
        slots, offsets, valid, _, grouped_x, clean_weights = _prepare_reference(
            x, weights, indices, start, end
        )
    gate_up = local.w13(grouped_x, offsets)
    hidden = routed_swiglu(
        gate_up,
        clean_weights,
        slots,
        clamp_limit=local.cfg.swiglu_limit,
        output_dtype=x.dtype,
    )
    down = local.w2(hidden, offsets)
    if timing:
        # Exact previous production chain, not the independent scratch oracle.
        down = torch.where(grouped_valid[:, None], down, 0).contiguous()
        return reduce_routes(down, slots, x.shape[0], indices.shape[1]).to(output_dtype)
    return _sum_reference(
        down, slots, valid, x.shape[0], indices.shape[1], output_dtype
    )


def test_real_w4a8_k32_local_chain_capture_and_capacity_fallback():
    local = _make_local()
    # 257 slots is the first unsupported metadata capacity; topk remains 1.
    # 33x8 additionally covers the real topk8 fallback above the bound.
    for rows, topk in ((1, 8), (4, 8), (8, 8), (32, 8), (257, 1), (33, 8)):
        x, weights, indices = _inputs(rows, topk, dim=128)
        if indices.numel() > 256:
            with pytest.raises(ValueError, match="bounded"):
                static_route_metadata(indices, 192, 240)
        for got, expected in zip(
            static_group_routes(indices, 192, 240),
            _metadata_reference(indices, 192, 240)[:3],
        ):
            _bits(got, expected)
        for output_dtype in (torch.float32, torch.bfloat16):
            fn = lambda: forward_local_static(
                local, x, weights, indices, 192, 240, output_dtype=output_dtype
            )
            graph, captured = _capture(fn)
            for empty in (False, True, False):
                new_x, new_weights, new_indices = _inputs(
                    rows, topk, dim=128, all_invalid=empty
                )
                x.copy_(new_x)
                weights.copy_(new_weights)
                indices.copy_(new_indices)
                expected = _old_local(
                    local, x, weights, indices, 192, 240, output_dtype=output_dtype
                )
                actual = fn()
                graph.replay()
                torch.cuda.synchronize()
                _bits(actual, expected)
                _bits(captured, expected)
                assert torch.isfinite(actual).all()


def test_reject_unsupported_layouts_without_mutation():
    x, weights, indices = _inputs(4)
    with pytest.raises(ValueError, match="metadata"):
        static_route_metadata(indices.float(), 192, 240)
    with pytest.raises(ValueError, match="metadata"):
        static_route_metadata(indices, 192, 192)
    with pytest.raises(ValueError, match="gather"):
        prepare_static_routes(x, weights[:, :4], indices, 192, 240)
    slots, _, valid, inverse = static_route_metadata(indices, 192, 240)
    down = torch.randn(slots.numel(), 5120, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="reduction"):
        reduce_static_routes(down[:, ::2], inverse, valid, 4, 8)
    with pytest.raises(ValueError, match="reduction"):
        reduce_static_routes(down, inverse, valid, 4, 8, output_dtype=torch.float16)


@pytest.mark.skipif(
    os.environ.get("DSV41_SM90_STATIC_ROUTE_BENCH") != "1",
    reason="explicit benchmark opt-in",
)
def test_clean_graph_real_shape_local_chain_benchmark():
    # Includes metadata/gather/masking, both actual W4A8/K32 GEMMs, weighted
    # activation, reduction and BF16 combine-input boundary. Communication is
    # deliberately absent: these are local-chain timings, not E2E TPOT.
    local = _make_local(dim=5120, inter=2304)
    for rows in (1, 4, 8, 32):
        x, weights, indices = _inputs(rows)
        old = lambda: _old_local(
            local,
            x,
            weights,
            indices,
            192,
            240,
            output_dtype=torch.bfloat16,
            timing=True,
        )
        new = lambda: forward_local_static(
            local, x, weights, indices, 192, 240, output_dtype=torch.bfloat16
        )
        reference = _old_local(
            local, x, weights, indices, 192, 240, output_dtype=torch.bfloat16
        )
        _bits(old(), reference)
        _bits(new(), reference)
        old_graph, old_output = _capture(old)
        new_graph, new_output = _capture(new)

        def measure(graph):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(100):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / 100

        old_ms, new_ms = [], []
        for repeat in range(5):
            for graph, samples in ((old_graph, old_ms), (new_graph, new_ms))[
                :: 1 if repeat % 2 == 0 else -1
            ]:
                samples.append(measure(graph))
        _bits(old_output, reference)
        _bits(new_output, reference)
        print(
            "SM90_STATIC_ROUTE_LOCAL_CHAIN_BENCH "
            + json.dumps(
                dict(
                    recv_rows=rows,
                    topk=8,
                    active_topk_max=6,
                    experts=48,
                    dim=5120,
                    inter=2304,
                    production_EP8_max_rows1_recv_capacity=rows == 8,
                    old_ms=sorted(old_ms)[2],
                    fused_ms=sorted(new_ms)[2],
                    speedup=sorted(old_ms)[2] / sorted(new_ms)[2],
                    precision="unchanged native W4A8/K32; weighted activation before down quant; BF16 combine input",
                    scope="complete local chain; excludes DeepEP dispatch/combine and E2E TPOT",
                    correctness="bitwise equal original expression before timing",
                    profiling=False,
                )
            ),
            flush=True,
        )

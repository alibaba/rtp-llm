"""Bounded native W4A8/K32 graph replay gates; parent owns GPU execution.

Requires two NVLink-connected SM90 GPUs. Capture/replay is bitwise checked
against the same fixed-shape eager path. The original dynamic eager path
retains its historical rtol=.02/atol=.05 gate, without tolerance relaxation.
"""

import multiprocessing as mp
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_graph import (
    forward_local_static,
    static_group_routes,
)
from rtp_llm.test.utils.port_util import PortManager


def test_static_metadata_preserves_routes_and_covers_capacity():
    for ids in (
        [[3, -1, 1, 1, 6, 2, -1, 0]],
        [[-1] * 8, [-1] * 8],
        [[0, 3, 3, 2, 1, 0, -1, 9], [3, 2, 1, 0, -1, -1, -1, -1]],
    ):
        indices = torch.tensor(ids, dtype=torch.int64)
        slots, offsets, valid = static_group_routes(indices, 0, 4)
        flat = indices.flatten()
        expected = sorted(
            (i for i in range(flat.numel()) if 0 <= flat[i] < 4),
            key=lambda i: int(flat[i]),
        )
        actual = slots[valid[slots]].tolist()
        assert actual == expected
        assert sorted(slots.tolist()) == list(range(flat.numel()))
        assert offsets.dtype == torch.int32
        assert offsets[0] == 0 and offsets[-1] == flat.numel()
        assert bool((offsets[1:] >= offsets[:-1]).all())
        for expert in range(4):
            part = slots[offsets[expert] : offsets[expert + 1]]
            assert bool(((flat[part] == expert) | ~valid[part]).all())
            if expert != 3:
                assert bool(valid[part].all())


def _make_local(experts=4, start=0, dim=128):
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        _GroupedLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    cfg = SimpleNamespace(swiglu_limit=10.0)
    local = _GroupedLocalExperts(cfg)
    tensors = {}
    for wk, sk in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s),
        (W.v4_routed_w3_w, W.v4_routed_w3_s),
        (W.v4_routed_w2_w, W.v4_routed_w2_s),
    ):
        tensors[wk] = torch.randint(
            -128, 128, (experts, dim, dim // 2), device="cuda", dtype=torch.int8
        )
        tensors[sk] = torch.randint(
            121, 125, (experts, dim, dim // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    local.setup_weights(tensors)
    return local


def _assert_bits(actual, expected):
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    torch.testing.assert_close(
        actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
    )


def test_local_capture_replays_changed_routes_weights_tokens_and_empty(monkeypatch):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    monkeypatch.setenv("DSV41_SM90_NATIVE_A8", "1")
    monkeypatch.setenv("DSV41_SM90_A8_PROMOTION_GROUP", "32")
    torch.manual_seed(21415)
    local = _make_local()
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((2, 8), device="cuda")
    indices = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]] * 2, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward_local_static(local, x, weights, indices, 0, 4)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_local_static(local, x, weights, indices, 0, 4)
    with torch.inference_mode():
        for step, route in enumerate(
            (
                [[0, 0, 3, 1, -1, -1, -1, -1], [2, 1, 3, 0, -1, -1, -1, -1]],
                [[2, 3, 0, 1, 3, 2, -1, -1], [-1] * 8],
                [[-1] * 8, [-1] * 8],
                [[1, 3, 0, 2, -1, -1, -1, -1], [3, 2, 1, 0, -1, -1, -1, -1]],
            )
        ):
            indices.copy_(torch.tensor(route, device="cuda"))
            x.copy_(torch.randn_like(x) * (step + 1))
            weights.copy_(torch.rand_like(weights))
            # Simulate undefined receive padding, including NaN payload/weights.
            weights.masked_fill_(indices < 0, float("nan"))
            x.masked_fill_((indices < 0).all(1)[:, None], float("nan"))
            expected = forward_local_static(local, x, weights, indices, 0, 4)
            old = local._forward_into_buf(x, weights, indices, 0, 4)
            graph.replay()
            torch.cuda.synchronize()
            _assert_bits(captured, expected)
            torch.testing.assert_close(expected, old, rtol=0.02, atol=0.05)
            if step == 2:
                assert torch.count_nonzero(captured) == 0


def _ep_worker(rank, port, with_shared=False):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        DSV41_SM90_NATIVE_A8="1",
        DSV41_SM90_A8_PROMOTION_GROUP="32",
        DSV41_SM90_DECODE_GRAPH="1",
        DSV41_SM90_DECODE_GRAPH_MAX_ROWS="2",
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(21415 + rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    from rtp_llm.models_py.distributed.deepep_wrapper import (
        DeepEPBuffer,
        DeepEPMode,
        DeepEPWrapper,
    )
    from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        SM90HummingStrategy,
    )

    dim = 512 if with_shared else 128
    size = max(
        c.get_nvl_buffer_size_hint(dim * 2, 2)
        for c in (
            DeepEPBuffer.get_dispatch_config(2),
            DeepEPBuffer.get_combine_config(2),
        )
    )
    buffer = DeepEPBuffer(
        dist.group.WORLD,
        num_nvl_bytes=size,
        num_rdma_bytes=0,
        low_latency_mode=False,
        explicitly_destroy=True,
    )
    previous = DeepEPWrapper._instance
    try:
        DeepEPWrapper._instance = SimpleNamespace(
            buffer=buffer,
            mode=DeepEPMode.NORMAL,
            config=SimpleNamespace(ep_size=2, use_deepep_internode=False),
        )
        cfg = MoeCfg(
            layer_id=0,
            dim=dim,
            moe_inter_dim=dim,
            n_routed_experts=8,
            n_activated_experts=6,
            swiglu_limit=10.0,
            ep_size=2,
            ep_rank=rank,
            n_local_experts=4,
            local_expert_start=rank * 4,
            local_expert_end=(rank + 1) * 4,
            max_tokens_per_rank=2,
        )
        strategy = SM90HummingStrategy(cfg)
        local = _make_local(dim=dim)
        strategy._local = local
        strategy._dispatch._local = local
        strategy.setup_runtime()
        x = torch.randn((2, dim), device="cuda", dtype=torch.bfloat16)
        weights = torch.rand((2, 6), device="cuda")
        indices = torch.tensor([[0, 1, 2, 4, 5, 6], [7, 6, 5, 3, 2, 1]], device="cuda")
        if with_shared:
            import json

            from rtp_llm.models_py.modules.dsv41.moe.shared_expert import (
                MXFP8SharedExpertExecutor,
                W13SharedExpert,
                combine_routed_and_shared,
            )

            os.environ.update(
                DSV4_SHARED_EXPERT_MODE="overlap",
                MOEDBG="0",
                DSV4_MOE_STRICT_FUSED="1",
                DSV4_SHARED_EXPERT_BF16_ADD="0",
            )
            shared_weights = {}
            for name, n, k in (("w13", 512, dim), ("w2", dim, 256)):
                shared_weights[name + "_w"] = torch.randn(n, k, device="cuda").to(
                    torch.float8_e4m3fn
                )
                shared_weights[name + "_s"] = torch.full(
                    (n // 32, k // 32),
                    2**-6,
                    device="cuda",
                ).to(torch.float8_e8m0fnu)
            shared = W13SharedExpert(dim, 256, shared_weights, swiglu_limit=10.0)
            executor = MXFP8SharedExpertExecutor()
            executor.prepare(shared)
            assert executor._stream is not None
            shared_linears = (shared.w13._sm90_linear, shared.w2._sm90_linear)
            for linear in shared_linears:
                config = json.loads(linear.config.to_str())
                assert config["mma_type"] == "wgmma"
                assert config["a_dtype"] == config["b_dtype"] == "float8e4m3"
            # Do not infer isolation from class names: check the actual live
            # shared/routed linear lock allocations used in this capture.
            lock_tensors = [
                linear.locks for linear in (*shared_linears, local.w13, local.w2)
            ]
            assert len({value.data_ptr() for value in lock_tensors}) == 4

        def run():
            if not with_shared:
                return strategy(x, weights, indices)
            executor.start(shared, x)
            assert executor._active_stream is executor._stream
            try:
                routed_output = strategy(x, weights, indices)
            finally:
                shared_output = executor.finish()
            assert routed_output.dtype == torch.float32
            assert shared_output.dtype == torch.bfloat16
            combined = combine_routed_and_shared(routed_output, shared_output, x.dtype)
            return routed_output, shared_output, combined

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        # Same graph: change destination ranks, tokens, weights and emptiness.
        cases = (
            [[0, 1, 2, 4, 5, 6], [7, 6, 5, 3, 2, 1]],
            [[0, 1, 2, 3, -1, -1], [3, 2, 1, 0, -1, -1]],
            [[4, 5, 6, 7, -1, -1], [7, 6, 5, 4, -1, -1]],
            [[-1] * 6, [-1] * 6],
            [[7, 5, 3, 1, 0, 2], [0, 2, 4, 6, 7, 5]],
        )
        failures = []
        for step, route in enumerate(cases):
            indices.copy_(torch.tensor(route, device="cuda"))
            x.copy_(torch.randn_like(x) * (step + 1))
            weights.copy_(torch.rand_like(weights))
            expected = strategy(x, weights, indices)
            original = strategy._dispatch(x, weights, indices)
            if with_shared:
                sequential_shared = shared(x)
                # Reference explicitly expands both operands to FP32 before
                # invoking the real final fused consumer. No mocked comm or
                # replacement GEMMs participate in either execution arm.
                sequential_combined = combine_routed_and_shared(
                    expected.float(),
                    sequential_shared.float(),
                    x.dtype,
                )
                eager = run()
            graph.replay()
            torch.cuda.synchronize()
            try:
                if with_shared:
                    _assert_bits(captured[0], expected)
                    _assert_bits(captured[1], sequential_shared)
                    _assert_bits(captured[2], sequential_combined)
                    for got, want in zip(
                        eager, (expected, sequential_shared, sequential_combined)
                    ):
                        _assert_bits(got, want)
                    assert executor._out is None and executor._input is None
                    if step == 3:
                        assert torch.count_nonzero(captured[0]) == 0
                        # Shared still contributes when routed is empty;
                        # compare against the FP32-add consumer above, which
                        # also preserves the original signed-zero semantics.
                        assert torch.count_nonzero(captured[2]) > 0
                else:
                    _assert_bits(captured, expected)
                torch.testing.assert_close(expected, original, rtol=0.02, atol=0.05)
                if step == 3 and not with_shared:
                    assert torch.count_nonzero(captured) == 0
            except AssertionError as exc:
                failures.append(f"rank={rank} step={step}: {exc}")
            label = "SM90_GRAPH_EP2_SHARED_OVERLAP" if with_shared else "SM90_GRAPH_EP2"
            print(
                f"{label} rank={rank} step={step} max_diff={(expected-original).abs().max().item()}",
                flush=True,
            )
            dist.barrier()
        if failures:
            raise AssertionError("\n".join(failures))
    finally:
        DeepEPWrapper._instance = previous
        buffer.destroy()
        dist.destroy_process_group()


def _run_ep2_workers(*, with_shared=False):
    assert torch.cuda.device_count() >= 2, "Requires two parent-locked NVLink SM90 GPUs"
    assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(2))
    ports, locks = PortManager().get_consecutive_ports(1)
    processes = []
    try:
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=_ep_worker, args=(rank, ports[0], with_shared))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 600
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for lock in locks:
            lock.__exit__(None, None, None)


def test_ep2_normal_graph_replay_changed_routes_and_empty_ranks():
    _run_ep2_workers()


def test_ep2_normal_graph_native_shared_overlap_changed_live_inputs():
    _run_ep2_workers(with_shared=True)

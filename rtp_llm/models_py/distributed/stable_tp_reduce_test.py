"""Real TP4 transport and captured rank-order sums across message sizes."""

import json
import os
import socket
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port):
    os.environ["GLM53_STABLE_TP_REDUCE"] = "1"
    from rtp_llm.models_py.distributed import collective_torch as collective
    from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    collective._group_map[collective.Group.DP_AND_TP] = dist.group.WORLD
    collective._parallelism_config = SimpleNamespace(tp_size=4, dp_size=1, world_size=4)
    collective._initialized = True
    collective._refresh_glm53_collective_options()
    collective._reset_tp_collective_context()
    pattern = (1.0, 256.0, -256.0, 1.0)
    baseline = None
    diagnostic = []
    for rows in (1, 2, 4, 5, 17, 65):
        original = torch.full(
            (rows, 9216), pattern[rank], device="cuda", dtype=torch.bfloat16
        )
        native = original.clone()
        dist.all_reduce(native)
        if baseline is None:
            baseline = native[:1].clone()
        diagnostic.append(
            {
                "rows": rows,
                "native_values": native.unique().float().cpu().tolist(),
                "native_first_row_changed": not torch.equal(native[:1], baseline),
                "native_rows_identical": bool(torch.all(native == native[:1]).item()),
            }
        )
        eager = original.clone()
        collective.all_reduce(eager, collective.Group.TP)
        torch.testing.assert_close(eager, torch.full_like(eager, 2.0), rtol=0, atol=0)
        os.environ["GLM53_PREFILL_STABLE_TP_FUSION"] = "1"
        collective._refresh_glm53_collective_options()
        token = _FORWARD_METADATA.set({})
        try:
            fused = original.clone()
            collective.all_reduce(fused, collective.Group.TP)
            torch.testing.assert_close(fused, eager, rtol=0, atol=0)
        finally:
            _FORWARD_METADATA.reset(token)
            os.environ["GLM53_PREFILL_STABLE_TP_FUSION"] = "0"
            collective._refresh_glm53_collective_options()
        static = original.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                static.copy_(original)
                collective.all_reduce(static, collective.Group.TP)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            collective.all_reduce(static, collective.Group.TP)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            static.copy_(original)
            graph.replay()
            torch.testing.assert_close(static, eager, rtol=0, atol=0)
        # Release captured NCCL work while the process group is still alive.
        # Its shutdown must not race a graph/event destructor holding the GIL.
        torch.cuda.synchronize()
        graph.reset()
        del graph
    if rank == 0:
        os.environ["GLM53_PREFILL_STABLE_TP_REDUCE"] = "1"
        os.environ["GLM53_PREFILL_STABLE_TP_FUSION"] = "1"
        os.environ["GLM53_STABLE_TP_REDUCE"] = "0"
        collective._refresh_glm53_collective_options()
        collective._reset_tp_collective_context()
        token = _FORWARD_METADATA.set({})
        try:
            expected_group = collective._get_group(collective.Group.TP)
            actual_group, actual_world = collective._get_tp_collective_context()
            assert actual_group is expected_group
            assert actual_world == 4
            with mock.patch.object(
                collective, "_get_group", side_effect=AssertionError("hot group lookup")
            ), mock.patch.object(
                dist, "get_world_size", side_effect=AssertionError("hot world lookup")
            ):
                for _ in range(1000):
                    cached_group, cached_world = collective._get_tp_collective_context()
                    assert cached_group is expected_group
                    assert cached_world == 4

            def reference_metadata():
                stable = os.environ.get("GLM53_STABLE_TP_REDUCE", "0") == "1"
                if (
                    not stable
                    and os.environ.get("GLM53_PREFILL_STABLE_TP_REDUCE", "0") == "1"
                ):
                    stable = _FORWARD_METADATA.get() is not None
                group = collective._get_group(collective.Group.TP)
                fusion = os.environ.get("GLM53_PREFILL_STABLE_TP_FUSION", "0") == "1"
                world = dist.get_world_size(group)
                return stable, fusion, group, world

            def cached_metadata():
                stable = collective._GLM53_STABLE_TP_REDUCE
                if not stable and collective._GLM53_PREFILL_STABLE_TP_REDUCE:
                    stable = _FORWARD_METADATA.get() is not None
                group, world = collective._get_tp_collective_context()
                return stable, collective._GLM53_PREFILL_STABLE_TP_FUSION, group, world

            samples = {"reference": [], "cached": []}
            for repeat in range(20):
                order = (
                    ("reference", "cached")
                    if repeat % 2 == 0
                    else ("cached", "reference")
                )
                for name in order:
                    function = (
                        reference_metadata if name == "reference" else cached_metadata
                    )
                    begin = time.thread_time_ns()
                    for _ in range(10000):
                        value = function()
                    samples[name].append((time.thread_time_ns() - begin) / 10000)
                    assert value[:2] == (True, True)
                    assert value[2] is expected_group
                    assert value[3] == 4
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            assert medians["cached"] < medians["reference"] * 0.9, medians
            (
                Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                / "tp_lookup_benchmark.json"
            ).write_text(
                json.dumps({"ns_per_call": medians, "samples": samples}, indent=2)
            )
            print("TP_LOOKUP_BENCHMARK", medians, flush=True)
        finally:
            _FORWARD_METADATA.reset(token)
    os.environ["GLM53_TP_REDUCE_WORKSPACE_REUSE"] = "1"
    os.environ["GLM53_PREFILL_STABLE_TP_FUSION"] = "1"
    collective._refresh_glm53_collective_options()
    collective._reset_tp_collective_context()
    token = _FORWARD_METADATA.set({})
    try:
        original = torch.full(
            (17, 9216), pattern[rank], device="cuda", dtype=torch.bfloat16
        )
        first = original.clone()
        collective.all_reduce(first, collective.Group.TP)
        torch.testing.assert_close(first, torch.full_like(first, 2.0), rtol=0, atol=0)
        assert len(collective._tp_reduce_workspaces) == 1
        first_workspace = next(iter(collective._tp_reduce_workspaces.values()))

        second = original.clone()
        collective.all_reduce(second, collective.Group.TP)
        torch.testing.assert_close(
            second, torch.full_like(second, 2.0), rtol=0, atol=0
        )
        assert next(iter(collective._tp_reduce_workspaces.values())) is first_workspace

        alternate = torch.cuda.Stream()
        alternate.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(alternate):
            third = original.clone()
            collective.all_reduce(third, collective.Group.TP)
        torch.cuda.current_stream().wait_stream(alternate)
        torch.testing.assert_close(third, torch.full_like(third, 2.0), rtol=0, atol=0)
        assert len(collective._tp_reduce_workspaces) == 2

        if rank == 0:
            samples = {"allocate": [], "reuse": []}
            for repeat in range(20):
                order = (
                    ("allocate", "reuse")
                    if repeat % 2 == 0
                    else ("reuse", "allocate")
                )
                for name in order:
                    collective._GLM53_TP_REDUCE_WORKSPACE_REUSE = name == "reuse"
                    begin = time.thread_time_ns()
                    for _ in range(1000):
                        workspace = collective._get_tp_reduce_workspace(original, 4)
                    samples[name].append((time.thread_time_ns() - begin) / 1000)
                    assert workspace.shape == (4, *original.shape)
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            print("TP_WORKSPACE_BENCHMARK", medians, flush=True)
    finally:
        _FORWARD_METADATA.reset(token)
        collective._GLM53_TP_REDUCE_WORKSPACE_REUSE = False
        os.environ["GLM53_TP_REDUCE_WORKSPACE_REUSE"] = "0"
        os.environ["GLM53_PREFILL_STABLE_TP_FUSION"] = "0"
        collective._reset_tp_collective_context()
    if rank == 0:
        print("NCCL_BF16_MESSAGE_SHAPE", diagnostic, flush=True)
        path = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "tp_message_shapes.json"
        )
        path.write_text(json.dumps(diagnostic, indent=2))
    print("TP4_FINAL_BARRIER", rank, flush=True)
    dist.barrier()
    print("TP4_DESTROY_GROUP", rank, flush=True)
    dist.destroy_process_group()
    print("TP4_DONE", rank, flush=True)


if __name__ == "__main__":
    assert torch.cuda.device_count() == 4, "real TP4 test must not skip"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=4, join=True)

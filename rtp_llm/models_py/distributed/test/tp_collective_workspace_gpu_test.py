"""Exercise generic TP kernels and graph replay without a model-name gate."""

import datetime
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import tp_collective_workspace as comm
from rtp_llm.models_py.distributed.fp8_collective_projection import Fp8Activation


def _check_native_bindings():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    for name in (
        "push_reduce_scatter",
        "is_cuda_fabric_allocation",
        "custom_all_gather_staging",
        "custom_all_gather_direct",
        "custom_all_gather_fp8_staging",
        "custom_all_gather_fp8_direct",
    ):
        assert callable(getattr(rtp_llm_ops, name)), f"missing native binding: {name}"
    return rtp_llm_ops


def _check_direct_all_gather(group, device, rank, world_size):
    from rtp_llm.models_py.distributed.custom_all_gather import create_custom_all_gather

    for fp8, width in ((False, 520), (False, 640), (True, 640)):
        state = create_custom_all_gather(
            group,
            device,
            max_m=513 * world_size,
            k=width,
            fp8=fp8,
            staging_only=False,
        )
        assert state is not None, "direct AllGather must be initialized"
        # Exercise direct semaphores repeatedly, changing local rows and grid.
        for local_rows in (1, 3, 5, 128, 513, 17, 1):
            dtype = torch.uint8 if fp8 else torch.int16
            bits = torch.arange(local_rows * width, device=device)
            bits = (bits + rank * 127).to(dtype).reshape(local_rows, width)
            expected = [
                torch.empty_like(bits.view(torch.uint8)) for _ in range(world_size)
            ]
            dist.all_gather(expected, bits.view(torch.uint8), group=group)
            if fp8:
                groups = (width + 511) // 512
                wire = torch.full(
                    (groups, (local_rows + 3) // 4 * 4),
                    0x7F7F7F7F + rank,
                    dtype=torch.int32,
                    device=device,
                )
                actual, scales = state.all_gather_fp8(
                    bits.view(torch.float8_e4m3fn), wire, staging=False
                )
                torch.testing.assert_close(
                    actual.view(torch.uint8), torch.cat(expected), rtol=0, atol=0
                )
                expected_scales = torch.cat(
                    [
                        torch.full(
                            (local_rows, groups),
                            0x7F7F7F7F + r,
                            dtype=torch.int32,
                            device=device,
                        )
                        for r in range(world_size)
                    ]
                )
                torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
            else:
                actual = state.all_gather(bits.view(torch.bfloat16), staging=False)
                torch.testing.assert_close(
                    actual.view(torch.uint8), torch.cat(expected), rtol=0, atol=0
                )


def _worker(rank, world_size, port):
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        init_method="env://" if "RANK" in os.environ else f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=180),
    )
    group = dist.group.WORLD
    try:
        from unittest.mock import patch

        from rtp_llm.models_py.distributed import collective_torch as ct
        from rtp_llm.models_py.distributed.push_reduce_scatter import (
            _is_fabric_allocation,
        )

        _check_native_bindings()
        with patch.object(ct, "_get_group", return_value=group):
            for width in (640, 1024, 4096, 7168):
                state = comm.configure_tp_collectives(
                    group,
                    device,
                    max_m=max(256, 64 * world_size),
                    width=width,
                    fp8=True,
                )
                assert state.push_rs is not None, "Push RS path must execute"
                assert state.bf16_ag is not None, "BF16 staging path must execute"
                assert state.fp8_ag is not None, "FP8 staging path must execute"
                # Call the real C++ allocation check even in single-host CI.
                # Mocking this boundary hid the missing export on cross-host TP16.
                for allocation in (
                    state.push_rs.storage,
                    *state.bf16_ag.storage.values(),
                    *state.fp8_ag.storage.values(),
                ):
                    assert isinstance(_is_fabric_allocation(allocation), bool)
                for rows in (world_size, 6 * world_size, 64 * world_size):
                    partial = torch.arange(rows * width, device=device).reshape(
                        rows, width
                    )
                    partial = ((partial % 31 - 15) * (rank + 1)).to(torch.bfloat16)
                    reference = sum(
                        (partial.float() / (rank + 1) * (r + 1))
                        for r in range(world_size)
                    )
                    reference = reference.to(torch.bfloat16).chunk(world_size)[rank]
                    # Require dispatch through Push RS; the reference follows
                    # its FP32 rank-order sum rounded once to BF16.
                    with patch.object(
                        dist,
                        "reduce_scatter_tensor",
                        side_effect=AssertionError("unexpected NCCL RS"),
                    ):
                        output = ct.reduce_scatter(partial, ct.Group.TP)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    local = partial[: rows // world_size]
                    expected = [torch.empty_like(local) for _ in range(world_size)]
                    dist.all_gather(expected, local, group=group)
                    with patch.object(
                        dist,
                        "all_gather_into_tensor",
                        side_effect=AssertionError("unexpected NCCL AG"),
                    ):
                        first = ct.all_gather(local, ct.Group.TP)
                        ct.all_gather(local * 2, ct.Group.TP)
                    torch.testing.assert_close(
                        first, torch.cat(expected), rtol=0, atol=0
                    )

                # Odd local row counts force packed UE8M0 tails to be repacked
                # across source-rank boundaries, including signed FP8 values.
                for local_rows in (1, 3, 5, 17):
                    values = torch.arange(local_rows * width, device=device).reshape(
                        local_rows, width
                    )
                    values = ((values % 17 - 8) * (rank + 1)).to(torch.float8_e4m3fn)
                    groups = (width + 511) // 512
                    wire = torch.full(
                        (groups, (local_rows + 3) // 4 * 4),
                        0x7F7F7F7F + rank,
                        dtype=torch.int32,
                        device=device,
                    )
                    activation = Fp8Activation(values, wire)

                    class Projection:
                        K = width
                        scale_ue8m0 = True

                        def forward_quantized(self, v, s):
                            self.scales = s.clone()
                            return v.float()

                    projection = Projection()
                    result = comm.try_all_gather_projection(
                        activation, projection, group
                    )
                    expected_values = [
                        torch.empty_like(values.view(torch.uint8))
                        for _ in range(world_size)
                    ]
                    dist.all_gather(
                        expected_values, values.view(torch.uint8), group=group
                    )
                    expected_scales = torch.cat(
                        [
                            torch.full(
                                (local_rows, groups),
                                0x7F7F7F7F + r,
                                dtype=torch.int32,
                                device=device,
                            )
                            for r in range(world_size)
                        ]
                    )
                    torch.testing.assert_close(
                        result,
                        torch.cat(expected_values).view(torch.float8_e4m3fn).float(),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        projection.scales, expected_scales, rtol=0, atol=0
                    )

                from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
                    CudaFp8DeepGEMMLinear,
                )

                linear = CudaFp8DeepGEMMLinear(
                    torch.ones((128, width), device=device).to(torch.float8_e4m3fn),
                    torch.full(
                        ((width + 511) // 512, 128),
                        0x7F7F7F7F,
                        dtype=torch.int32,
                        device=device,
                    ).T,
                )
                local = (
                    torch.arange(3 * width, device=device).reshape(3, width) % 127 - 63
                )
                local = (local * (rank + 1) / 256).to(torch.bfloat16)
                full = [torch.empty_like(local) for _ in range(world_size)]
                dist.all_gather(full, local, group=group)
                reference = linear(torch.cat(full))
                with patch.object(
                    dist,
                    "all_gather_into_tensor",
                    side_effect=AssertionError("unexpected BF16 NCCL AG"),
                ):
                    actual = linear.forward_all_gather(local)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

                from types import SimpleNamespace

                from rtp_llm.models_py.model_desc.module_base import GptModelBase
                from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
                    CudaFp8GEMMLinear,
                )

                wrapper = CudaFp8GEMMLinear(linear.weight, linear.weight_scales)
                model = GptModelBase.__new__(GptModelBase)
                torch.nn.Module.__init__(model)
                model.config = SimpleNamespace(hidden_size=width, gen_num_per_cycle=1)
                model.parallelism_config = SimpleNamespace(
                    tp_size=world_size, enable_sp=True
                )
                model.weight = SimpleNamespace(get_global_weight=lambda _: None)
                model._tp_collective_batch_capacity = 128
                model._tp_collective_workspace = None
                model.projection = wrapper
                self_initialized = model.initialize(SimpleNamespace(kv_cache=None))
                assert self_initialized and model._tp_collective_workspace is state
                with patch.object(
                    dist,
                    "all_gather_into_tensor",
                    side_effect=AssertionError("unexpected BF16 NCCL AG"),
                ):
                    actual = wrapper.forward_all_gather(local)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

                partial = torch.full(
                    (16 * world_size, width),
                    rank + 1.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                output = torch.empty((16, width), dtype=torch.bfloat16, device=device)
                state.push_rs.reduce_scatter(partial, output)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    state.push_rs.reduce_scatter(partial, output)
                    state.fp8_ag.all_gather_fp8(values, wire, staging=True)
                for step in range(4):
                    partial.fill_(rank + step + 1.0)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = world_size * (world_size - 1) / 2 + world_size * (
                        step + 1
                    )
                    torch.testing.assert_close(
                        output, torch.full_like(output, expected), rtol=0, atol=0
                    )
        _check_direct_all_gather(group, device, rank, world_size)
        dist.barrier()
    finally:
        comm.clear_tp_collective_workspaces()
        dist.destroy_process_group()


class TpCollectiveWorkspaceGpuTest(unittest.TestCase):
    def test_native_binding_contract(self):
        ops = _check_native_bindings()
        with self.assertRaisesRegex(RuntimeError, "nonempty CUDA tensor"):
            ops.is_cuda_fabric_allocation(torch.empty(1))

    def test_generic_widths_and_graph_replay(self):
        # The same assertions support a real multi-host torchrun launch.
        if "RANK" in os.environ:
            _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), None)
            return
        world_size = int(os.environ.get("GPU_COUNT", "2"))
        self.assertGreaterEqual(torch.cuda.device_count(), world_size)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    unittest.main()

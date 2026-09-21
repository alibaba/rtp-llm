"""TP8 native AG payloads, mixed sizes, graph replay and AG/GEMM dispatch.

Run on eight SM100/SM103 NVLink GPUs with the compiled RTP CUDA bindings.
"""

import socket
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import custom_all_gather as custom_ag
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3 import all_gather_gemm as ag


def _reference(values, wire, group):
    rows, k = values.shape
    out = torch.empty((rows * 8, k), dtype=values.dtype, device=values.device)
    dist.all_gather_into_tensor(
        out.view(torch.uint8), values.view(torch.uint8), group=group
    )
    if wire is None:
        return out
    parts = [torch.empty_like(wire) for _ in range(8)]
    dist.all_gather(parts, wire, group=group)
    return out, torch.cat([part[:, :rows] for part in parts], dim=1).T


def _payload(rows, rank, fp8, *, raw=False):
    # Rank-local offsets exercise copying without rank-local collective fallback.
    offset = int(rank == 0)
    dtype = torch.uint8 if fp8 else torch.bfloat16
    storage = torch.empty(rows * 7168 + offset, dtype=dtype, device="cuda")
    values = storage[offset:].view(rows, 7168)
    if not fp8:
        values.normal_()
        values[:, :8].zero_()
        return values, None
    if raw:
        values.random_(0, 256)
    else:
        values.copy_(
            torch.randint(-3, 4, values.shape, device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    padded = (rows + 3) // 4 * 4
    offset = int(rank == 3)
    storage = torch.empty(14 * padded + offset, dtype=torch.int32, device="cuda")
    wire = storage[offset:].view(14, padded)
    if raw:
        wire.random_(-(1 << 31), (1 << 31) - 1)
    else:
        r = torch.arange(padded, device="cuda")[None]
        g = torch.arange(14, device="cuda")[:, None]
        wire.copy_((124 + (r + g + rank) % 4) * 0x01010101)
    wire[:, rows:].fill_(0x5A5A5A5A)
    return values.view(torch.float8_e4m3fn), wire


def _check_payload(workspace, values, wire, group):
    staging = values.shape[0] < 128
    if wire is None:
        actual = workspace.all_gather(values, staging=staging)
        torch.testing.assert_close(
            actual, _reference(values, None, group), rtol=0, atol=0
        )
    else:
        actual, scales = workspace.all_gather_fp8(values, wire, staging=staging)
        expected, expected_scales = _reference(values, wire, group)
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        assert scales.stride() == (1, values.shape[0] * 8)


def _graph_payloads(workspace, rank, fp8, group):
    inputs = [
        _payload(rows, rank, fp8, raw=True) for rows in (3, 128, 1, 256, 127, 512)
    ]
    order = (0, 1, 2, 3, 4, 5, 0, 1)

    def run():
        outputs = []
        for i in order:
            values, wire = inputs[i]
            staging = values.shape[0] < 128
            if fp8:
                gathered, scales = workspace.all_gather_fp8(
                    values, wire, staging=staging
                )
                outputs.extend((gathered.view(torch.uint8).clone(), scales.clone()))
            else:
                outputs.append(workspace.all_gather(values, staging=staging).clone())
        return outputs

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        outputs = run()
    for step in range(4):
        for values, wire in inputs:
            if fp8:
                values.view(torch.uint8).random_(0, 256)
                wire.random_(-(1 << 31), (1 << 31) - 1)
            else:
                values.normal_().mul_(rank + step + 1)
        # Eager and graph calls interleave on the same initialized state.
        run()
        graph.replay()
        expected = []
        for i in order:
            values, wire = inputs[i]
            ref = _reference(values, wire, group)
            if fp8:
                gathered, scales = ref
                expected.extend((gathered.view(torch.uint8), scales))
            else:
                expected.append(ref)
        for actual, ref in zip(outputs, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    torch.cuda.synchronize()
    graph.reset()


def _projection(fp8):
    if not fp8:
        return torch.randn((7168, 64), dtype=torch.bfloat16, device="cuda")
    # The merged MLA projection crosses the original non-128-aligned N=2112 boundary.
    weight = torch.randint(-3, 4, (3648, 7168), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((14, 3648), 0x7C7C7C7C, dtype=torch.int32, device="cuda").T
    return CudaFp8DeepGEMMLinear(weight, scales)


def _gemm_dispatch(group, rank, fp8, prefill):
    device = torch.device("cuda", rank)
    # Boundary and adjacent per-rank inputs, including decode above the prefill
    # staging limit. Decode must still fit and replay its staging workspace.
    local_rows = (
        (1, 127, 128, 129, 4095, 4096, 4097) if prefill else (1, 127, 128, 129, 512)
    )
    ag.configure_all_gather_gemm(
        group,
        device,
        max_m=max(local_rows) * 8,
        k=7168,
        dtype=torch.bfloat16,
        fp8=fp8,
        use_fused=prefill,
    )
    state = ag._STATES[(group, rank, fp8)]
    assert state.custom is not None
    assert state.custom.storage["values"].shape[0] == (32760 if prefill else 4096)
    projection = _projection(fp8)
    for local_m in local_rows:
        m = local_m * 8
        values, wire = _payload(local_m, rank, fp8)
        if prefill and local_m >= 4096:
            # The original overlap consumers use the rank-local tensor directly.
            # Match their normal aligned GEMM operands: misaligned BF16 cuBLAS
            # chooses a different reduction algorithm, and FP8 TMA needs alignment.
            # Rank-local offsets are tested separately on both custom AG paths.
            values = values.clone()
            if wire is not None:
                wire = wire.clone()
        local = QuantizedActivation(values, wire) if fp8 else values
        ref = _reference(values, wire, group)
        expected = projection.forward_quantized(*ref) if fp8 else ref @ projection
        with patch.object(ag, "get_process_group", return_value=group):
            actual = ag.all_gather_gemm(local, [projection], logical_m=m - 3)[0]
            overlap = prefill and local_m >= 4096
            torch.testing.assert_close(
                actual,
                expected[: m - 3],
                rtol=0.02 if overlap else 0,
                atol=0.0625 if overlap else 0,
            )
            if not prefill:
                for _ in range(2):
                    ag.all_gather_gemm(local, [projection], logical_m=m - 3)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                    captured = ag.all_gather_gemm(local, [projection], logical_m=m - 3)[
                        0
                    ]
                for step in range(2):
                    if fp8:
                        values.view(torch.uint8).copy_(
                            torch.randint(-3, 4, values.shape, device="cuda")
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                        )
                        wire[:, : values.shape[0]].fill_(
                            (124 + step + rank % 2) * 0x01010101
                        )
                    else:
                        values.normal_()
                    graph.replay()
                    ref = _reference(values, wire, group)
                    expected = (
                        projection.forward_quantized(*ref) if fp8 else ref @ projection
                    )
                    torch.testing.assert_close(
                        captured, expected[: m - 3], rtol=0, atol=0
                    )
                torch.cuda.synchronize()
                graph.reset()
        if rank == 0:
            print(
                f"PASS AG/GEMM fp8={fp8} prefill={prefill} local_M={local_m} global_M={m}",
                flush=True,
            )


def worker(rank, port):
    torch.cuda.set_device(rank)
    torch.manual_seed(991 + rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(minutes=5),
    )
    group = dist.group.WORLD
    device = torch.device("cuda", rank)
    for fp8 in (False, True):
        # Exercise both native kernels independently of decode's staging policy.
        workspace = custom_ag.create_custom_all_gather(
            group, device, max_m=32768, k=7168, fp8=fp8
        )
        assert workspace is not None, "custom AG must be enabled on the supported setup"
        for m in (8, 1016, 1024, 24, 8192, 512, 32760, 8, 32768):
            values, wire = _payload(m // 8, rank, fp8, raw=True)
            _check_payload(workspace, values, wire, group)
            values.view(torch.uint8).zero_()
            if wire is not None:
                wire.zero_()
            _check_payload(workspace, values, wire, group)
        _graph_payloads(workspace, rank, fp8, group)
        if rank == 0:
            print(
                f"PASS native AG fp8={fp8}: payloads, padding, offsets, grids and graph replay",
                flush=True,
            )
        _gemm_dispatch(group, rank, fp8, False)
    prefill_group = dist.new_group(list(range(8)))
    for fp8 in (False, True):
        _gemm_dispatch(prefill_group, rank, fp8, True)

    # A single peer's rejection must select fallback for the complete group.
    fallback_group = dist.new_group(list(range(8)))
    with patch.object(custom_ag, "_nvlink_peers", return_value=rank != 0):
        ag.configure_all_gather_gemm(
            fallback_group,
            device,
            max_m=8,
            k=7168,
            dtype=torch.bfloat16,
            use_fused=False,
        )
    assert ag._STATES[(fallback_group, rank, False)].custom is None
    values, _ = _payload(1, rank, False)
    weight = _projection(False)
    with patch.object(ag, "get_process_group", return_value=fallback_group):
        actual = ag.all_gather_gemm(values, [weight], logical_m=8)[0]
    expected = _reference(values, None, fallback_group) @ weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rank == 0:
        print(
            "PASS single-peer rejection selects working NCCL fallback on all ranks",
            flush=True,
        )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group(prefill_group)
    dist.destroy_process_group(fallback_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=8, join=True)

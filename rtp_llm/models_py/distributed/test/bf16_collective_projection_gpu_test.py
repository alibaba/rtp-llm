"""Actual TP8 BF16 projection backends and strided KDA norm contracts."""

import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=180)
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
        Bf16CollectiveProjection,
        prepare_bf16_projection,
    )
    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated

    device = torch.device("cuda", rank)
    torch.manual_seed(3101 + rank)
    output_projection = SimpleNamespace(
        weight=(torch.randn((7168, 1536), device=device) * 0.01).bfloat16(),
        bias=None,
    )
    prepare_bf16_projection(output_projection)
    collective = Bf16CollectiveProjection(
        dist.group.WORLD, device, max_m=65536, hidden_size=7168,
        output_projections=[output_projection],
    )
    x = (torch.randn((8192, 7168), device=device) * 0.05).bfloat16()
    projection = SimpleNamespace(
        weight=(torch.randn((6284, 7168), device=device) * 0.01).bfloat16(),
        bias=None,
    )
    prepare_bf16_projection(projection, alignment=128)
    # Initialize/JIT, then one complete representative warmup. Reference
    # verification happens outside the measured production backend.
    collective.all_gather_gemm(x, projection)
    torch.cuda.synchronize()
    collective.all_gather_gemm(x, projection)
    torch.cuda.synchronize()
    ag_ms = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        result = collective.all_gather_gemm(x, projection)
        end.record()
        end.synchronize()
        ag_ms.append(start.elapsed_time(end))
    gathered = x.new_empty((65536, 7168))
    dist.all_gather_into_tensor(gathered, x)
    ag_error = 0.0
    for start in range(0, 65536, 1024):
        # Reference batching bounds only the test oracle's memory, not the
        # model or measured fused AG/GEMM call.
        expected = (
            gathered[start : start + 1024].float() @ projection.weight[:6284].float().T
        ).bfloat16()
        actual = result[start : start + 1024, :6284]
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.0005)
        ag_error = max(ag_error, (actual.float() - expected.float()).abs().max().item())
    assert result[:, 6284:].count_nonzero().item() == 0
    del gathered, result, projection, x

    a = (torch.randn((4096, 1536), device=device) * 0.05).bfloat16()
    collective.gemm_reduce_scatter(a, output_projection)
    torch.cuda.synchronize()
    collective.gemm_reduce_scatter(a, output_projection)
    torch.cuda.synchronize()
    result = collective.gemm_reduce_scatter(a, output_projection)
    # DeepGEMM publishes BF16 GEMM partials over NVLink, then accumulates
    # their SUM in FP32. Use that same rounding boundary in the oracle.
    partial = (a.float() @ output_projection.weight.float().T).bfloat16().float()
    dist.all_reduce(partial)
    expected = partial.chunk(world_size)[rank].bfloat16()
    torch.testing.assert_close(result, expected, rtol=0.03, atol=0.0005)
    rs_error = (result.float() - expected.float()).abs().max().item()

    packed = torch.randn((257, 6400), device=device, dtype=torch.bfloat16)
    gate = packed[:, 4608:6144].view(257, 12, 128)
    norm_input = torch.randn_like(gate)
    norm = FusedRMSNormGated(torch.ones(128, device=device, dtype=torch.bfloat16), 1e-5)
    actual = norm(norm_input, gate)
    contiguous_reference = norm(
        norm_input.reshape(-1, 128), gate.contiguous().reshape(-1, 128)
    ).view_as(actual)
    torch.testing.assert_close(actual, contiguous_reference, rtol=0, atol=0)
    torch.cuda.synchronize()
    print(
        json.dumps(
            dict(
                rank=rank,
                passed=True,
                ag_max_abs=ag_error,
                rs_max_abs=rs_error,
                ag_ms=ag_ms,
                ag_gemm_dtype="bf16",
                ag_communication_dtype="bf16",
                rs_gemm_dtype="bf16",
                rs_communication_dtype="bf16",
                gemm_accumulator_dtype="fp32",
                rs_accumulator_dtype="fp32",
                rs_workspace_bytes=collective.rs_workspace.num_bytes,
                peak_allocated=torch.cuda.max_memory_allocated(),
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")
    mp.spawn(worker, args=(8,), nprocs=8, join=True)

"""Manual real-NCCL Decode AG regression with K3 FP8 projection geometry.

TP8: K=7168, Q/KV latent projection N=1536+512+64 and local output-gate
projection N=96/8*128. Tests correctness, not distributed performance.
"""

import argparse
import socket
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3 import all_gather_gemm as ag


def quantize(x, projection):
    values, scales = projection.quantize_input(x)
    rows, k = x.shape
    aligned = (rows + 3) // 4 * 4
    return QuantizedActivation(
        values, scales.as_strided(((k + 511) // 512, aligned), (aligned, 1))
    )


def reference(payload, projections, pg, logical_m):
    """The previous NCCL + Torch scale repacking + same-shape GEMM path."""
    ranks, rows = pg.size(), payload.shape[0]
    groups, local_pad = payload.scale_wire.shape
    values = payload.values.new_empty((ranks * rows, payload.shape[1]))
    wire = payload.scale_wire.new_empty((ranks * groups, local_pad))
    dist.all_gather_into_tensor(
        values.view(torch.uint8), payload.values.view(torch.uint8), group=pg
    )
    dist.all_gather_into_tensor(wire, payload.scale_wire, group=pg)
    scales = wire.new_zeros((groups, (ranks * rows + 3) // 4 * 4))
    scales[:, : ranks * rows].copy_(
        wire.view(ranks, groups, local_pad)[:, :, :rows]
        .permute(1, 0, 2)
        .reshape(groups, ranks * rows)
    )
    return [
        p.forward_quantized(values, scales.T[: ranks * rows])[:logical_m]
        for p in projections
    ]


def worker(rank, ranks, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=ranks,
        timeout=timedelta(minutes=5),
    )
    pg = dist.group.WORLD
    try:
        torch.manual_seed(31415 + rank)
        k = 7168
        projections = [
            CudaFp8DeepGEMMLinear(
                weight=torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
                weight_scales=torch.full(
                    (k // 512, n), 0x7B7C7D7E, device=device, dtype=torch.int32
                ).T,
                quant_config=init_quant_config("FP8_PER_BLOCK"),
            )
            for n in (1536 + 512 + 64, 96 // ranks * 128)
        ]
        ag.configure_all_gather_gemm(
            pg,
            device,
            max_m=32 * ranks,
            k=k,
            dtype=torch.bfloat16,
            fp8=True,
            use_fused=False,
        )
        with patch.object(ag, "get_process_group", return_value=pg):
            for rows in (1, 2, 3, 4, 8, 16, 32):
                # Different exponent patterns across ranks/rows/K-groups expose
                # misplaced scale words even when all FP8 payloads look alike.
                amplitudes = torch.exp2(
                    (
                        torch.arange(rows, device=device)[:, None]
                        + torch.arange(k // 128, device=device)[None, :]
                        + rank
                    )
                    % 7
                    - 3
                ).repeat_interleave(128, dim=1)
                x = (torch.randn(rows, k, device=device) * amplitudes).to(
                    torch.bfloat16
                )
                logical = rows * ranks - int(rows == 3)

                def run(prequantized):
                    source = quantize(x, projections[0]) if prequantized else x
                    return ag.all_gather_gemm(source, projections, logical_m=logical)

                for prequantized in (False, True):
                    expected = reference(
                        quantize(x, projections[0]), projections, pg, logical
                    )
                    actual = run(prequantized)
                    for result, want in zip(actual, expected):
                        torch.testing.assert_close(result, want, rtol=0, atol=0)
                    if rows in (1, 2, 32):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                run(prequantized)
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            captured = run(prequantized)
                        for step in range(3):
                            x.normal_().mul_(2.0 ** (rank % 4 + step - 2))
                            graph.replay()
                            expected = reference(
                                quantize(x, projections[0]), projections, pg, logical
                            )
                            for result, want in zip(captured, expected):
                                torch.testing.assert_close(result, want, rtol=0, atol=0)
                        torch.cuda.synchronize()
                        graph.reset()
                    if rank == 0:
                        print(
                            f"PASS TP{ranks} local_M={rows} global_M={rows*ranks} K={k} prequantized={prequantized}",
                            flush=True,
                        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8, choices=(2, 4, 8))
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"requires {args.world_size} visible CUDA devices")
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    mp.spawn(worker, args=(args.world_size, port), nprocs=args.world_size, join=True)

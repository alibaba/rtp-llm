"""Real NCCL/symmetric-memory FP8 pair AG and peer-output RS regression.

Run the Bazel test executable with --world-size 1, 2, 4, or 8.
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
from rtp_llm.models_py.modules.hybrid.test import collective_gemm_reference as ref
from rtp_llm.models_py.modules.kimi_k3 import all_gather_gemm as ag
from rtp_llm.models_py.modules.kimi_k3 import gemm_reduce_scatter as rs
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import rmsnorm_fp8


def worker(rank, size, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        timeout=timedelta(minutes=5),
    )
    pg = dist.group.WORLD
    torch.manual_seed(100 + rank)
    k, n, capacity = 512, 512, 65536
    projection = CudaFp8DeepGEMMLinear(
        weight=torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
        weight_scales=torch.full(
            (k // 512, n), 0x7F7F7F7F, device=device, dtype=torch.int32
        ).T,
        input_scales=None,
        bias=None,
        quant_config=init_quant_config("FP8_PER_BLOCK"),
    )
    weight = torch.ones(k, device=device, dtype=torch.bfloat16)
    ag.configure_all_gather_gemm(
        pg,
        device,
        max_m=capacity,
        k=k,
        dtype=torch.bfloat16,
        fp8=True,
    )
    if size > 1:
        rs.configure_gemm_reduce_scatter(pg, device, max_m=capacity, n=n, fp8=True)

    def gather(x, out, group):
        dist.all_gather_into_tensor(out, x, group=pg)
        return out

    with patch.object(ag, "get_process_group", return_value=pg), patch.object(
        ref, "get_process_group", return_value=pg
    ), patch.object(ref, "all_gather_into", side_effect=gather):
        for logical in sorted(
            {0, 1, 2, 3, 4, size - 1, size + 1, 32767, 32768, 32769, 65536}
        ):
            rows = (logical + size - 1) // size
            x = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
            # Rank-dependent scales expose payload/scale pairing and padding errors.
            x.mul_(2.0 ** (rank - 3))
            payload = rmsnorm_fp8(x, weight, 1.0e-5)
            reference = ref.all_gather_gemm_reference(
                payload, [projection], logical_m=logical
            )[0]
            actual = ag.all_gather_gemm(payload, [projection], logical_m=logical)[0]
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            if logical in (1, 3, 32769):
                for _ in range(3):
                    ag.all_gather_gemm(payload, [projection], logical_m=logical)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_payload = rmsnorm_fp8(x, weight, 1.0e-5)
                    graph_output = ag.all_gather_gemm(
                        graph_payload, [projection], logical_m=logical
                    )[0]
                for step in range(3):
                    x.normal_().mul_(step + 1)
                    graph.replay()
                    expected = ref.all_gather_gemm_reference(
                        rmsnorm_fp8(x, weight, 1.0e-5), [projection], logical_m=logical
                    )[0]
                    torch.testing.assert_close(graph_output, expected, atol=0, rtol=0)
            if rank == 0:
                print(f"PASS TP{size} AG M={logical}", flush=True)
        if size > 1:
            for m in (0, 1, 2, 3, 4, size - 1, size + 1, 32767, 32768, 32769, 65536):
                x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
                payload = rmsnorm_fp8(x, weight, 1.0e-5)
                actual = rs.gemm_reduce_scatter(payload, projection, pg, pad_rows=True)
                expected = ref.gemm_reduce_scatter_reference(
                    payload, projection, pg, pad_rows=True, ordered_sum=True
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                if rank == 0:
                    print(f"PASS TP{size} RS M={m}", flush=True)
    bf16_weight = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    ag.configure_all_gather_gemm(pg, device, max_m=capacity, k=k, dtype=torch.bfloat16)
    with patch.object(ag, "get_process_group", return_value=pg), patch.object(
        ref, "get_process_group", return_value=pg
    ), patch.object(ref, "all_gather_into", side_effect=gather):
        for logical in sorted(
            {0, 1, 2, 3, size - 1, size + 1, 32767, 32768, 32769, 65536}
        ):
            rows = (logical + size - 1) // size
            x = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
            actual = ag.all_gather_gemm(x, [bf16_weight], logical_m=logical)[0]
            expected = ref.all_gather_gemm_reference(
                x, [bf16_weight], logical_m=logical
            )[0]
            torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.02)
            if size > 1:
                rs_x = torch.randn(logical, k, device=device, dtype=torch.bfloat16)
                actual = rs.gemm_reduce_scatter(rs_x, bf16_weight, pg, pad_rows=True)
                # Match the fused kernel's FP32 accumulation of BF16 partials.
                # NCCL BF16 reduction rounds intermediate sums differently.
                expected = ref.gemm_reduce_scatter_reference(
                    rs_x, bf16_weight, pg, pad_rows=True, ordered_sum=True
                )
                torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.01)
                # Independently check the mathematical GEMM + sum in FP32.
                oracle = ref.gemm_reduce_scatter_reference(
                    rs_x.float(), bf16_weight.float(), pg, pad_rows=True
                )
                error = (actual.float() - oracle).norm()
                magnitude = oracle.norm().clamp_min(1.0e-12)
                relative_error = (error / magnitude).item()
                assert relative_error < 0.005, (size, logical, rank, relative_error)
                if rank == 0:
                    print(
                        f"BF16 RS FP32-reference relative L2 M={logical}: {relative_error:.6g}",
                        flush=True,
                    )
            if rank == 0:
                print(f"PASS TP{size} BF16 AG/RS M={logical}", flush=True)
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args.world_size, port), nprocs=args.world_size, join=True)

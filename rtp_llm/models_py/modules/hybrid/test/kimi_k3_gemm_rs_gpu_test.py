"""Real BF16/FP8 GEMM-RS with K3 MLA/KDA o_proj shapes.

Both projections use 96 heads * 128 dimensions -> hidden size 7168.
Run on SM100/SM103 with --world-size 2, 4 or 8 (default: 8).
No model weights are needed: inputs, weights and scales vary by rank.
"""

import argparse
import socket
from datetime import timedelta
from unittest import TestCase
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
from rtp_llm.models_py.modules.kimi_k3 import gemm_reduce_scatter as rs


def quantized_input(x, projection):
    m, k = x.shape
    wire = torch.full(
        ((k + 511) // 512, (m + 3) // 4 * 4),
        0x7F7F7F7F,
        dtype=torch.int32,
        device=x.device,
    )
    if m == 0:
        return QuantizedActivation(x.to(torch.float8_e4m3fn), wire)
    values, scales = projection.quantize_input(x)
    wire[:, :m].copy_(scales.T)
    return QuantizedActivation(values, wire)


def reference(x, weight, pg, *, fused):
    """Separate GEMM + communication, matching each backend's reduction dtype."""
    size, rank = dist.get_world_size(pg), dist.get_rank(pg)
    m, k = x.shape
    physical_m = (m + size - 1) // size * size
    n = weight.shape[1] if isinstance(weight, torch.Tensor) else weight.N
    if m == 0:
        return torch.empty((0, n), dtype=torch.bfloat16, device=x.device)
    if isinstance(x, QuantizedActivation):
        padded = x.pad_rows(physical_m)
        partial = weight.forward_quantized(padded.values, padded.scales)
    else:
        padded = torch.nn.functional.pad(x, (0, 0, 0, physical_m - m))
        partial = (
            padded @ weight if isinstance(weight, torch.Tensor) else weight(padded)
        )
    out = torch.empty((physical_m // size, n), dtype=torch.bfloat16, device=x.device)
    if fused:
        # The fused API rounds each partial to BF16, then accumulates FP32 in
        # source-rank order. NCCL's BF16 tree can round differently.
        gathered = [torch.empty_like(partial) for _ in range(size)]
        dist.all_gather(gathered, partial, group=pg)
        total = torch.zeros_like(out, dtype=torch.float32)
        for source in gathered:
            total.add_(source.chunk(size)[rank].float())
        out.copy_(total)
    else:
        dist.reduce_scatter_tensor(out, partial, group=pg)
    return out


def assert_numerics(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all().item()
    if actual.numel():
        # Allow BF16 GEMM accumulation-order differences; mismatched payload,
        # scale packing, rank order or padding are far outside these bounds.
        diff = (actual.float() - expected.float()).abs()
        relative_l2 = diff.norm() / expected.float().norm().clamp_min(1e-6)
        assert relative_l2.item() < 0.006, relative_l2.item()
        assert diff.max().item() <= expected.float().abs().max().item() * 0.03 + 1e-3


def worker(rank, size, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    assert torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        timeout=timedelta(minutes=5),
    )
    pg = dist.group.WORLD
    torch.manual_seed(1729 + rank)
    k, n, capacity = 96 * 128 // size, 7168, 2048
    bf16_weight = (torch.randn(k, n, device=device) / k**0.5).to(torch.bfloat16)
    packed_scales = torch.full(
        (k // 512, n), 0x7A7B7C7D, device=device, dtype=torch.int32
    ).T
    projection = CudaFp8DeepGEMMLinear(
        weight=torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
        weight_scales=packed_scales,
        quant_config=init_quant_config("FP8_PER_BLOCK"),
    )
    key = rs.collective_gemm_state_key(pg, device)
    for prefill in (True, False):
        # BF16-first configuration followed by FP8 must share one workspace.
        rs.configure_gemm_reduce_scatter(
            pg, device, max_m=capacity, n=n, use_fused=prefill
        )
        workspace = rs._STATES[key].workspace
        rs.configure_gemm_reduce_scatter(
            pg, device, max_m=capacity, n=n, fp8=True, use_fused=prefill
        )
        assert rs._STATES[key].workspace is workspace
        for m in (0, 1, 8, 256, 504, 511, 512, 513, 1024):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16).mul_(
                2.0 ** (rank % 4 - 2)
            )
            for precision in ("bf16", "fp8", "prequantized"):
                weight = bf16_weight if precision == "bf16" else projection

                def run():
                    payload = (
                        quantized_input(x, projection)
                        if precision == "prequantized"
                        else x
                    )
                    return rs.gemm_reduce_scatter(payload, weight, pg, pad_rows=True)

                fused = prefill and m >= 512
                backend = rs._STATES[key].deep_gemm
                # Spy on real kernels/collectives to prove selection, not just
                # numerical equivalence between the two possible paths.
                with patch.object(
                    rs.dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor
                ) as nccl:
                    if prefill:
                        with patch.object(
                            backend, "bf16_gemm_rs_nn", wraps=backend.bf16_gemm_rs_nn
                        ) as bf16, patch.object(
                            backend, "fp8_gemm_rs_nt", wraps=backend.fp8_gemm_rs_nt
                        ) as fp8:
                            actual = run()
                            assert bf16.call_count == int(fused and precision == "bf16")
                            assert fp8.call_count == int(fused and precision != "bf16")
                    else:
                        actual = run()
                    assert nccl.call_count == int(m > 0 and not fused)
                payload = (
                    quantized_input(x, projection) if precision == "prequantized" else x
                )
                expected = reference(payload, weight, pg, fused=fused)
                assert_numerics(actual, expected)
                if m in (511, 512, 513):
                    # Move the shared workspace onto the capture stream during
                    # warmup. A cross-stream wait first introduced inside capture
                    # would create a dependency on uncaptured work.
                    capture_stream = torch.cuda.Stream()
                    capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(capture_stream):
                        for _ in range(3):
                            run()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        graph_output = run()
                    for step in range(3):
                        x.normal_().mul_((step + 1) * 2.0 ** (rank % 4 - 2))
                        graph.replay()
                        payload = (
                            quantized_input(x, projection)
                            if precision == "prequantized"
                            else x
                        )
                        assert_numerics(
                            graph_output, reference(payload, weight, pg, fused=fused)
                        )
                    torch.cuda.synchronize()
                    graph.reset()
                    del graph, graph_output
                if rank == 0:
                    print(
                        f"PASS TP{size} prefill={prefill} M={m} K={k} N={n} {precision} fused={fused}",
                        flush=True,
                    )
        if prefill:
            # Non-power-of-two FP32 scales and a bias must keep the original
            # projection's numerical contract, even at the fusion threshold.
            x = torch.randn(512, k, device=device, dtype=torch.bfloat16)
            for kind in ("fp32_scales", "bias"):
                fallback = CudaFp8DeepGEMMLinear(
                    weight=projection.weight,
                    weight_scales=(
                        torch.rand(n // 128, k // 128, device=device) * 0.09 + 0.03
                        if kind == "fp32_scales"
                        else packed_scales
                    ),
                    bias=(
                        torch.randn(n, device=device, dtype=torch.bfloat16)
                        if kind == "bias"
                        else None
                    ),
                    quant_config=init_quant_config("FP8_PER_BLOCK"),
                )
                try:
                    expected = reference(x, fallback, pg, fused=False)
                except RuntimeError as exc:
                    # SM103's ordinary GEMM currently rejects FP32 scales when
                    # disable_ue8m0_cast=True. Preserve that explicit rejection,
                    # never silently round scales by entering the fused API.
                    message = "Unsupported architecture or scaling factor types"
                    if kind != "fp32_scales" or message not in str(exc):
                        raise
                    with patch.object(backend, "fp8_gemm_rs_nt") as fused_kernel:
                        with TestCase().assertRaisesRegex(RuntimeError, message):
                            rs.gemm_reduce_scatter(x, fallback, pg, pad_rows=False)
                        fused_kernel.assert_not_called()
                    if rank == 0:
                        print(
                            f"PASS TP{size} FP32-scale backend rejection preserved (numerics unsupported)",
                            flush=True,
                        )
                    continue
                with patch.object(
                    rs.dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor
                ) as nccl, patch.object(
                    backend, "fp8_gemm_rs_nt", wraps=backend.fp8_gemm_rs_nt
                ) as fused_kernel:
                    actual = rs.gemm_reduce_scatter(x, fallback, pg, pad_rows=False)
                    assert nccl.call_count == 1
                    assert fused_kernel.call_count == 0
                assert_numerics(actual, expected)
                if rank == 0:
                    print(f"PASS TP{size} {kind} M=512 NCCL fallback", flush=True)
        torch.cuda.synchronize()
        dist.barrier()
        rs._STATES.pop(key)
        if workspace is not None:
            workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8), default=8)
    args = parser.parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"requires {args.world_size} visible GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args.world_size, port), nprocs=args.world_size, join=True)

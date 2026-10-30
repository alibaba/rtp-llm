"""Check and benchmark K3.1 TP4 Target Verify communication and its consumers."""

import argparse
import tempfile
from datetime import timedelta
from itertools import cycle
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import create_custom_all_gather
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
    DecodeNvfp4AllGather,
    decode_packet_capacity,
)
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual

TP = 4
VERIFY_ROWS = 64
HIDDEN = 7168
GATHERS_PER_GRAPH = 32
GRAPH_WARMUPS = 4
SAMPLES = 25


def _capture(operation, stream, benchmark):
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUPS):
            operation()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(GATHERS_PER_GRAPH if benchmark else 2):
            output = operation()
    return graph, output


def _time_graph_pair(captures, device):
    names = list(captures)
    timings = {name: [] for name in names}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for sample in range(GRAPH_WARMUPS + SAMPLES):
        order = names if sample % 2 == 0 else names[::-1]
        for name in order:
            dist.barrier()
            start.record()
            captures[name].replay()
            end.record()
            end.synchronize()
            if sample >= GRAPH_WARMUPS:
                timings[name].append(start.elapsed_time(end) * 1000 / GATHERS_PER_GRAPH)
    results = {}
    for name, times in timings.items():
        samples = torch.tensor(times, device=device)
        dist.all_reduce(samples, op=dist.ReduceOp.MAX)
        ordered = sorted(samples.tolist())
        results[name] = (median(ordered), ordered[(95 * SAMPLES + 99) // 100 - 1])
    return results


def _verify_packed_projection(rank, device, group, benchmark):
    from rtp_llm.config.quant_config import ModelOptFp4Config
    from rtp_llm.models.kimi_k3.nvfp4_tensor import quantize_nvfp4_weight
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
        CudaFp4GEMMLinear,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
        CudaFp8DeepGEMMLinear,
    )

    rows = VERIFY_ROWS // TP
    local_heads = 96 // TP
    head_width = local_heads * 128
    kda_logical_width = 4 * head_width + 128 + local_heads
    kda_width = (kda_logical_width + 31) // 32 * 32
    dsa_width = 1536 + 512 + 64 + head_width
    index_width = 128
    gate_columns = 3 * 32
    # Rotate projection weights across graph nodes to reflect distinct layers.
    # A single repeatedly used projection can be served from GPU L2.
    weight_count = 16 if benchmark else 1
    options = {"dtype": torch.bfloat16, "device": device}
    prefix = torch.randn(rows, HIDDEN, **options)
    norm = torch.ones(HIDDEN, **options)
    anchor = torch.empty((rows, 0, HIDDEN), **options)
    residual = KimiK3AttentionResidual(norm, norm, 1e-5)
    quant_config = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=False, activation_scheme="dynamic"
    )

    def nvfp4_projection(width):
        weight = torch.randn(width, HIDDEN, **options) * 0.02
        codes, scales, scale = quantize_nvfp4_weight(weight)
        return CudaFp4GEMMLinear(
            codes, scales, quant_config=quant_config, weight_scale_2=scale
        )

    def index_projection():
        weight = torch.randn(index_width, HIDDEN, **options) * 0.02
        return CudaFp8DeepGEMMLinear(*quant_weight_ue8m0_packed(weight))

    direct = create_custom_all_gather(
        group,
        device,
        max_m=VERIFY_ROWS,
        k=HIDDEN,
        fp8=False,
        staging_only=False,
        packed_bytes=decode_packet_capacity(rows, HIDDEN, gate_columns, rowwise=True),
    )
    exchange = DecodeNvfp4AllGather(group, direct)
    index_stream = torch.cuda.Stream(device=device)
    capture_stream = torch.cuda.Stream(device=device)

    def project(packet, gather, projection, index):
        fp4, fp8, gate = gather(packet)
        if index is None:
            return projection.forward_quantized(fp4)
        # Verify issues input and key projections on separate streams.
        compute = torch.cuda.current_stream(device)
        index_stream.wait_stream(compute)
        with torch.cuda.stream(index_stream):
            key = index.forward_quantized(*fp8)
        value = projection.forward_quantized(fp4)
        compute.wait_stream(index_stream)
        return value, key, gate

    mixed_cases = {}
    for case, width, columns in (
        ("kda", kda_width, 0),
        ("dsa", dsa_width, gate_columns),
    ):
        packet = residual(
            prefix,
            anchor,
            output_norm_weight=norm,
            output_norm_eps=1e-5,
            quantize_nvfp4=True,
            rowwise_nvfp4=True,
            decode_head_gate_columns=columns,
        )
        if columns:
            head_gate = packet.views()[2]
            head_gate.copy_(
                torch.arange(head_gate.numel(), device=device).view_as(head_gate)
            )
            head_gate.add_(rank * head_gate.numel() + 0.25)

        reference, reference_fp8, reference_gate = exchange.gather_and_unpack(packet)
        layout = exchange.gather_gemm_layout(packet)
        if layout is None:
            raise RuntimeError(f"packed {case} GEMM-layout AG was unavailable")
        fp4, fp8, head_gate = layout
        torch.testing.assert_close(fp4.values, reference.values, rtol=0, atol=0)
        torch.testing.assert_close(fp4.scale, reference.scale, rtol=0, atol=0)
        # A CUTLASS 128x4 scale tile has four row banks. M=64 only
        # consumes the first two banks of four bytes per row.
        torch.testing.assert_close(
            fp4.block_scales.view(-1, 16)[:, :8],
            reference.block_scales.view(-1, 16)[:, :8],
            rtol=0,
            atol=0,
        )
        if columns:
            torch.testing.assert_close(
                (fp8[0].view(torch.uint8), fp8[1], head_gate),
                (reference_fp8[0].view(torch.uint8), reference_fp8[1], reference_gate),
                rtol=0,
                atol=0,
            )

        projections = [
            (nvfp4_projection(width), index_projection() if columns else None)
            for _ in range(weight_count)
        ]

        last_projection = projections[
            (GRAPH_WARMUPS + GATHERS_PER_GRAPH - 1) % weight_count
        ]
        expected = project(packet, exchange.gather_and_unpack, *last_projection)
        mixed_cases[case] = packet, last_projection, expected
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        captures = {}
        for name, gather in (
            ("direct_unpack", exchange.gather_and_unpack),
            ("gemm_layout", exchange),
        ):
            weights = cycle(projections)
            graph, output = _capture(
                lambda: project(packet, gather, *next(weights)),
                capture_stream,
                benchmark,
            )
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            captures[name] = graph

        if not benchmark:
            continue

        for name, (p50, p95) in _time_graph_pair(captures, device).items():
            if rank == 0:
                print(
                    f"K31_PACKED_PROJECTION {case} {name} "
                    f"local_m={rows} m={VERIFY_ROWS} n={width} k={HIDDEN} "
                    f"index_n={index_width if columns else 0} "
                    f"head_gate_columns={columns} packet_bytes={packet.wire.numel()} "
                    f"weights_per_graph={weight_count} "
                    f"p50_us={p50:.3f} p95_us={p95:.3f}",
                    flush=True,
                )

    kda_packet, kda_projection, kda_expected = mixed_cases["kda"]
    dsa_packet, dsa_projection, dsa_expected = mixed_cases["dsa"]
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    dist.barrier()
    mixed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(mixed_graph, stream=capture_stream):
        dsa_value, dsa_key, dsa_gate = project(dsa_packet, exchange, *dsa_projection)
        # The next layer may overwrite the gather workspace after its consumers.
        consumed_gate = dsa_gate.clone()
        kda_value = project(kda_packet, exchange, *kda_projection)
    for _ in range(2):
        mixed_graph.replay()
        torch.testing.assert_close(kda_value, kda_expected, rtol=0, atol=0)
        torch.testing.assert_close(
            (dsa_value, dsa_key, consumed_gate), dsa_expected, rtol=0, atol=0
        )


def _worker(rank, rendezvous, benchmark):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("expected SM103 for this target-verify comparison")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=TP,
        timeout=timedelta(seconds=90),
    )
    try:
        group = dist.group.WORLD
        _verify_packed_projection(rank, device, group, benchmark)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Skip timing samples")
    args = parser.parse_args()
    if torch.cuda.device_count() != TP:
        raise RuntimeError(
            f"expected {TP} CUDA devices, found {torch.cuda.device_count()}"
        )
    with tempfile.TemporaryDirectory(prefix="k31-ag-") as rendezvous_dir:
        rendezvous = (Path(rendezvous_dir) / "store").as_uri()
        mp.spawn(_worker, args=(rendezvous, not args.check_only), nprocs=TP)

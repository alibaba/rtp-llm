"""Manual two-GPU GenericMoE scheduling benchmark.

It constructs the actual RTP GenericMoeLayer, FusedMoe and DenseMLP paths with
deterministic synthetic BF16 or FP8 weights. Shape: E256/topk8/H2048/local-I256/
shared-I256. It measures CUDA-event whole-layer latency only, never accuracy,
TTFT, or pure kernel time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.ops import ActivationType, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

E, K, I, TOPK = 256, 2048, 256, 8


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _quantize_expert_weights(weight):
    """Make FP8 weights/scales in DeepGEMM's [E, N, K] layout."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        per_block_cast_to_fp8,
        requant_weight_ue8m0,
    )

    quantized, scales = [], []
    for expert_weight in weight:
        fp8_weight, float_scale = per_block_cast_to_fp8(expert_weight, use_ue8m0=False)
        quantized.append(fp8_weight)
        scales.append(float_scale)
    # Pack the stacked batch once: stacking already-packed scales would destroy
    # the MN-major TMA strides required by grouped DeepGEMM.
    return requant_weight_ue8m0(torch.stack(quantized), torch.stack(scales))


def _quantize_linear_weight(weight_kn):
    """Make an E8M0 DeepGEMM linear weight in native [N, K] layout."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        per_block_cast_to_fp8,
        requant_weight_ue8m0,
    )

    fp8_weight, float_scale = per_block_cast_to_fp8(
        weight_kn.T.contiguous(), use_ue8m0=False
    )
    return requant_weight_ue8m0(fp8_weight, float_scale)


def _layer(rank, chunks, mode, tokens, weight_dtype):
    os.environ.update(
        MOE_TP_CHUNKS=str(chunks),
        MOE_TP_CHUNK_MODE=mode,
        MOE_TP_CHUNK_MIN_TOKENS="1",
    )
    model = ModelConfig()
    model.hidden_size, model.inter_size, model.moe_inter_size = K, 2 * I, 2 * I
    model.activation_type = ActivationType.Swiglu
    model.has_moe_norm = True
    model.expert_num = E
    model.moe_k = TOPK
    model.moe_style = 2
    model.n_shared_experts = 1
    model.max_seq_len, model.data_type = tokens, "bf16"
    if weight_dtype == "fp8":
        model.quant_config = init_quant_config("FP8_PER_BLOCK")
    parallel = ParallelismConfig()
    parallel.world_size = parallel.local_world_size = parallel.tp_size = 2
    parallel.local_rank = parallel.world_rank = parallel.tp_rank = rank
    parallel.ffn_tp_size = 2
    parallel.ffn_tp_rank = rank
    parallel.dp_size = parallel.ep_size = 1
    parallel.dp_rank = parallel.ep_rank = 0
    moe = MoeConfig()
    moe.moe_strategy = (
        "fp8_per_block_no_dp" if weight_dtype == "fp8" else "no_quant_cpp"
    )
    moe.use_all_gather = True
    rank_local = torch.Generator(device="cuda")
    rank_local.manual_seed(20260928 + rank)
    replicated = torch.Generator(device="cuda")
    replicated.manual_seed(20260928)

    def rnd(*shape):
        return torch.randn(
            *shape, generator=rank_local, device="cuda", dtype=torch.bfloat16
        ).mul_(0.02)

    def replicated_rnd(*shape):
        return torch.randn(
            *shape, generator=replicated, device="cuda", dtype=torch.bfloat16
        ).mul_(0.02)

    # The input to each TP rank is replicated, so routing and the scalar shared
    # gate must be replicated too. Expert and shared-expert projections model
    # rank-local TP shards.
    weights = {
        W.moe_gate: replicated_rnd(K, E),
        W.moe_w1: rnd(E, 2 * I, K),
        W.moe_w2: rnd(E, K, I),
        W.ffn_w13: rnd(K, 2 * I),
        W.ffn_w2: rnd(I, K),
        W.shared_expert_gate: replicated_rnd(K, 1),
    }
    if weight_dtype == "fp8":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )

        if not is_deep_gemm_e8m0_used():
            raise RuntimeError(
                "--weight-dtype fp8 requires the E8M0 DeepGEMM weight-scale path"
            )
        weights[W.moe_w1], weights[W.moe_s1] = _quantize_expert_weights(
            weights[W.moe_w1]
        )
        weights[W.moe_w2], weights[W.moe_s2] = _quantize_expert_weights(
            weights[W.moe_w2]
        )
        weights[W.ffn_w13], weights[W.ffn_s13] = _quantize_linear_weight(
            weights[W.ffn_w13]
        )
        weights[W.ffn_w2], weights[W.ffn_s2] = _quantize_linear_weight(
            weights[W.ffn_w2]
        )
        assert weights[W.ffn_s13].dtype == torch.int32
        assert weights[W.ffn_s2].dtype == torch.int32
        assert weights[W.moe_s1].dtype == torch.int32
        assert weights[W.moe_s2].dtype == torch.int32
    return GenericMoeLayer(
        model,
        parallel,
        weights,
        moe,
        enable_cuda_graph=weight_dtype == "fp8",
    )


def _nccl(tensor, **_):
    dist.all_reduce(tensor, group=dist.group.WORLD)
    return tensor


def _measure(layer, x, warmup, iterations):
    for _ in range(warmup):
        layer(x, allow_tp_chunking=True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        layer(x, allow_tp_chunking=True)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {**_summarize_samples(samples), "samples_us": samples}


def _summarize_samples(samples):
    ordered = sorted(samples)
    return {
        "median_us": median(samples),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _error_stats(candidate, baseline):
    difference = (candidate.float() - baseline.float()).flatten()
    baseline_norm = torch.linalg.vector_norm(baseline.float()).clamp_min(1e-12)
    return {
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item() / baseline_norm.item()
        ),
    }


def _profile(layer, x, profile_dir, variant, rank, weight_dtype):
    """Export three post-warmup forwards; never include them in latency samples."""
    trace_path = (
        Path(profile_dir) / f"moe_tp_chunking_{weight_dtype}_{variant}_rank{rank}.json"
    )
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        for _ in range(3):
            with torch.profiler.record_function(f"moe_tp_chunking/{variant}/forward"):
                layer(x, allow_tp_chunking=True)
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    return str(trace_path)


def _collective_require(condition, description):
    conditions = [None, None]
    dist.all_gather_object(conditions, bool(condition))
    if not all(conditions):
        raise AssertionError(
            f"all-rank requirement failed: {description}: {conditions}"
        )


def _runtime_details(layer, x, chunks, weight_dtype):
    fused_moe = layer.fused_moe
    shared = layer.shared_expert
    executor = fused_moe.fused_experts
    can_chunk = layer._can_chunk_tp_prefill(x, allow_tp_chunking=True)
    details = {
        "strategy": fused_moe.strategy_name,
        "router": type(fused_moe.router).__name__,
        "executor": type(executor).__name__,
        "shared_expert": type(shared).__name__ if shared else None,
        "shared_up_linear": type(shared.up_proj).__name__ if shared else None,
        "shared_down_linear": type(shared.down_proj).__name__ if shared else None,
        "unified_tp_allreduce": layer.use_unified_tp_allreduce,
        "chunk_eligible": can_chunk,
        "ffn_tp_size": layer.ffn_tp_size,
        "executor_cuda_graph_capable": getattr(executor, "enable_cuda_graph", False),
        "forward_is_eager": not torch.cuda.is_current_stream_capturing(),
    }
    if weight_dtype == "fp8":
        details["fp8_weight_scale_dtype"] = str(executor.w13_weight_scale_inv.dtype)
    return details


def _worker(rank, port, tokens, warmup, iterations, profile_dir, weight_dtype):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        FT_DISABLE_CUSTOM_AR="1",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.model_desc import generic_moe

        collective_torch._get_group = lambda _group: dist.group.WORLD
        generator = torch.Generator(device="cuda")
        generator.manual_seed(7)
        x = torch.randn(
            tokens, K, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        variants = (
            ("A_nccl_full", 0, "overlap"),
            ("B_nccl_serial_2", 2, "serial"),
            ("C_nccl_overlap_2", 2, "overlap"),
            ("D_nccl_serial_4", 4, "serial"),
            ("E_nccl_overlap_4", 4, "overlap"),
        )
        original = generic_moe.all_reduce
        out = {
            "synthetic_weights": True,
            "weight_dtype": weight_dtype,
            "nccl_version": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(),
            "shape": {
                "experts": E,
                "topk": TOPK,
                "hidden": K,
                "local_intermediate": I,
                "shared_intermediate": I,
                "tokens": tokens,
            },
            "timing": "cuda_event_end_to_end_layer_us",
            "collective_backend": "torch.distributed NCCL only",
            "numerical_check": {},
            "runtime": {},
        }
        if profile_dir:
            out["profile_traces"] = {}
        baseline = None
        serial_outputs = {}
        for name, chunks, mode in variants:
            layer = _layer(rank, chunks, mode, tokens, weight_dtype)
            generic_moe.all_reduce = _nccl
            try:
                runtime = _runtime_details(layer, x, chunks, weight_dtype)
                out["runtime"][name] = runtime
                _collective_require(
                    runtime["unified_tp_allreduce"],
                    f"{name} must use one unified TP reduction",
                )
                assert runtime["unified_tp_allreduce"]
                if chunks:
                    _collective_require(
                        runtime["chunk_eligible"],
                        f"{name} must enter the TP chunk path",
                    )
                    assert runtime["chunk_eligible"]
                if weight_dtype == "fp8":
                    _collective_require(
                        runtime["executor"] == "DeepGemmHybridExecutor"
                        and runtime["fp8_weight_scale_dtype"] == "torch.int32"
                        and runtime["shared_up_linear"] == "CudaFp8GEMMLinear"
                        and runtime["shared_down_linear"] == "CudaFp8GEMMLinear"
                        and runtime["executor_cuda_graph_capable"],
                        f"{name} must use E8M0 DeepGemmHybridExecutor",
                    )
                candidate = layer(x, allow_tp_chunking=True).detach().clone()
                if baseline is None:
                    baseline = candidate
                    out["numerical_check"][name] = {
                        "max_abs": 0.0,
                        "relative_l2": 0.0,
                    }
                    rank_zero_baseline = baseline.clone()
                    dist.broadcast(rank_zero_baseline, src=0)
                    _collective_require(
                        torch.allclose(
                            baseline, rank_zero_baseline, rtol=1e-2, atol=2e-2
                        ),
                        f"{name} baseline must agree across TP ranks",
                    )
                else:
                    out["numerical_check"][name] = _error_stats(candidate, baseline)
                    print(
                        f"[NUMERICAL] rank={rank} variant={name} stats={out['numerical_check'][name]}",
                        flush=True,
                    )
                    _collective_require(
                        torch.allclose(candidate, baseline, rtol=1e-2, atol=2e-2),
                        f"{name} must match the full NCCL baseline",
                    )
                if mode == "serial":
                    serial_outputs[chunks] = candidate
                elif chunks:
                    # Same chunk sizes and kernels: scheduling alone must not
                    # change the result. Full-vs-chunked GEMMs use BF16 tolerance.
                    _collective_require(
                        torch.equal(candidate, serial_outputs[chunks]),
                        f"{name} must bitwise match serial chunks={chunks}",
                    )
                print(
                    f"[BENCHMARK] rank={rank} dtype={weight_dtype} variant={name} numerical=passed measure=start",
                    flush=True,
                )
                out[name] = _measure(layer, x, warmup, iterations)
                statistics = {
                    key: value
                    for key, value in out[name].items()
                    if key != "samples_us"
                }
                print(
                    f"[BENCHMARK] rank={rank} variant={name} result={statistics}",
                    flush=True,
                )
                if profile_dir:
                    out["profile_traces"][name] = _profile(
                        layer, x, profile_dir, name, rank, weight_dtype
                    )
            finally:
                generic_moe.all_reduce = original
            del layer
            torch.cuda.empty_cache()
            dist.barrier()
        gathered = [None, None]
        dist.all_gather_object(gathered, out)
        if rank == 0:
            worst_rank = {
                name: _summarize_samples(
                    [
                        max(paired)
                        for paired in zip(
                            *(item[name]["samples_us"] for item in gathered)
                        )
                    ]
                )
                for name, _, _ in variants
            }
            for item in gathered:
                for name, _, _ in variants:
                    del item[name]["samples_us"]
            summary = {
                "key_statistic": "percentiles_of_per_iteration_worst_rank",
                "per_rank": gathered,
                "worst_rank": worst_rank,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument(
        "--weight-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="BF16 weights or FP8_PER_BLOCK E8M0 weights with DeepGEMM",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="export three CPU+CUDA forward traces per rank and variant",
    )
    parser.add_argument(
        "--profile-dir",
        help="trace directory; defaults to TEST_UNDECLARED_OUTPUTS_DIR",
    )
    args = parser.parse_args()
    if args.tokens < 4 or args.warmup < 0 or args.iterations < 1:
        parser.error("tokens must be >= 4, warmup >= 0 and iterations >= 1")
    profile_dir = args.profile_dir or os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if args.profile and not profile_dir:
        raise RuntimeError(
            "--profile requires --profile-dir or TEST_UNDECLARED_OUTPUTS_DIR"
        )
    if args.profile:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("requires two CUDA GPUs and NCCL")
    mp.spawn(
        _worker,
        args=(
            _port(),
            args.tokens,
            args.warmup,
            args.iterations,
            profile_dir if args.profile else None,
            args.weight_dtype,
        ),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    main()

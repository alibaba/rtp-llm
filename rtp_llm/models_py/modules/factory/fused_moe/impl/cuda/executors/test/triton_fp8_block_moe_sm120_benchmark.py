"""SM120 FP8 MoE microbenchmark for the Qwen3-30B-A3B decode shape.

This is a manual performance test, not a timing assertion in regular CI.  It
compares CUDA Graph replay of the complete Triton and DeepGEMM executor paths
and can search Triton GEMM launch configurations without changing production
code between runs.
"""

import argparse
import json
import statistics
from collections.abc import Callable
from typing import Any

import torch
import triton.language as tl

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel,
    moe_align_block_size_compiled,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# Qwen3-30B-A3B, TP=1.  The two expert GEMMs are [1536, 2048] and
# [2048, 768], respectively.
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
GATE_UP_SIZE = 2 * INTERMEDIATE_SIZE


def make_config() -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.quant_config = Fp8BlockWiseQuantConfig()
    model_config.data_type = "bf16"
    model_config.expert_num = NUM_EXPERTS
    model_config.moe_k = TOP_K
    model_config.hidden_size = HIDDEN_SIZE
    model_config.moe_inter_size = INTERMEDIATE_SIZE
    model_config.activation_type = "SiGLU"

    parallelism_config = ParallelismConfig()
    parallelism_config.world_size = 1
    parallelism_config.local_world_size = 1
    parallelism_config.tp_size = 1
    parallelism_config.ep_size = 1

    moe_config = MoeConfig()
    moe_config.use_all_gather = True
    moe_config.use_deepep_moe = False
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        enable_cuda_graph=True,
    )


def make_packed_weight(
    experts: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm.utils.layout

    # Directly construct representable FP8 values and non-uniform scales.  This
    # keeps setup cheap while ensuring the benchmark can detect numerically
    # invalid tile configurations instead of timing an all-zero no-op result.
    weight = torch.full(
        (experts, n, k), 0.015625, device="cuda", dtype=torch.float8_e4m3fn
    )
    # UE8M0 encodes powers of two.  Vary both dimensions while preserving that
    # format contract so scale-indexing mistakes remain observable.
    n_groups = torch.arange(n // 128, device="cuda", dtype=torch.float32)
    k_groups = torch.arange(k // 128, device="cuda", dtype=torch.float32)
    expert_groups = torch.arange(experts, device="cuda", dtype=torch.float32)
    n_scale = torch.pow(2.0, n_groups.remainder(4) - 2)
    k_scale = torch.pow(2.0, k_groups.remainder(3) - 2)
    expert_scale = torch.pow(2.0, expert_groups.remainder(4) - 2)
    block_scale = (
        expert_scale[:, None, None] * n_scale[None, :, None] * k_scale[None, None, :]
    )
    expanded_scale = block_scale.index_select(-2, torch.arange(n, device="cuda") // 128)
    scale = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
        expanded_scale
    )
    return weight, scale


def make_executor() -> DeepGemmHybridExecutor:
    if not is_deep_gemm_e8m0_used():
        raise RuntimeError("benchmark requires DeepGEMM packed UE8M0 scales")
    w13, s13 = make_packed_weight(NUM_EXPERTS, GATE_UP_SIZE, HIDDEN_SIZE)
    w2, s2 = make_packed_weight(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return DeepGemmHybridExecutor(
        config=make_config(),
        quant_config=FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
        ),
        weights={
            W.moe_w1: w13,
            W.moe_s1: s13,
            W.moe_w2: w2,
            W.moe_s2: s2,
        },
    )


def make_topk_ids(num_tokens: int, routing: str) -> torch.Tensor:
    route_ids = torch.arange(num_tokens * TOP_K, device="cuda")
    if routing == "balanced":
        # Maximizes active experts at small M and is the conservative case for
        # a routed kernel whose work scales with active-expert padding.
        topk_ids = route_ids % NUM_EXPERTS
    elif routing == "random":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260831 + num_tokens)
        scores = torch.rand(
            (num_tokens, NUM_EXPERTS), device="cuda", generator=generator
        )
        return scores.topk(TOP_K, dim=1).indices.to(torch.int64)
    elif routing == "skewed":
        topk_ids = route_ids % 16
    else:
        raise ValueError(f"unsupported routing distribution: {routing}")
    return topk_ids.view(num_tokens, TOP_K).to(torch.int64)


def make_payload(num_tokens: int, routing: str) -> ExpertForwardPayload:
    token_pattern = torch.arange(
        num_tokens * HIDDEN_SIZE, device="cuda", dtype=torch.float32
    ).view(num_tokens, HIDDEN_SIZE)
    hidden = ((token_pattern.remainder(17) - 8) * 0.00390625).to(torch.bfloat16)
    hidden_fp8, hidden_scale = sgl_per_token_group_quant_fp8(
        hidden,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    topk_ids = make_topk_ids(num_tokens, routing)
    topk_weights = torch.full(
        (num_tokens, TOP_K),
        1.0 / TOP_K,
        device="cuda",
        dtype=torch.float32,
    )
    counts = torch.bincount(topk_ids.view(-1).long(), minlength=NUM_EXPERTS).to(
        torch.int32
    )
    return ExpertForwardPayload(
        expert_x=hidden_fp8,
        expert_x_scale=hidden_scale,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=counts, expert_num_tokens_cpu=None
        ),
    )


def benchmark_graph(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
    samples: int,
) -> float:
    # Compile/JIT before capture.  Two calls also populate allocator caches.
    fn()
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / repetitions)

    del graph_output
    graph.reset()
    del graph
    torch.cuda.empty_cache()
    return statistics.median(timings)


def triton_call(
    executor: DeepGemmHybridExecutor, payload: ExpertForwardPayload
) -> torch.Tensor:
    return executor.execute_triton_fp8(
        payload, activation="SiGLU", apply_router_weight_on_input=False
    ).fused_expert_output


def deepgemm_call(
    executor: DeepGemmHybridExecutor, payload: ExpertForwardPayload
) -> torch.Tensor:
    return executor.execute_contiguous(
        payload,
        activation="SiGLU",
        expert_map=None,
        a2_scale=None,
        apply_router_weight_on_input=False,
        extra_expert_args=None,
    ).fused_expert_output


def benchmark_thresholds(
    executor: DeepGemmHybridExecutor,
    tokens: list[int],
    routings: list[str],
    warmup: int,
    repetitions: int,
    samples: int,
) -> None:
    for routing in routings:
        for num_tokens in tokens:
            payload = make_payload(num_tokens, routing)
            torch.testing.assert_close(
                triton_call(executor, payload),
                deepgemm_call(executor, payload),
                rtol=2e-2,
                atol=2e-2,
            )
            triton_ms = benchmark_graph(
                lambda: triton_call(executor, payload),
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            deepgemm_ms = benchmark_graph(
                lambda: deepgemm_call(executor, payload),
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            print(
                json.dumps(
                    {
                        "type": "threshold",
                        "routing": routing,
                        "tokens": num_tokens,
                        "triton_ms": triton_ms,
                        "deepgemm_ms": deepgemm_ms,
                        "speedup": deepgemm_ms / triton_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def candidate_configs() -> list[dict[str, int]]:
    return [
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for block_m in (8, 16, 32)
        for block_n in (64, 128, 256)
        for block_k in (64, 128)
        for num_warps in (4, 8)
        for num_stages in (2, 3, 4, 5)
    ]


def make_kernel_inputs(
    executor: DeepGemmHybridExecutor,
    num_tokens: int,
    routing: str,
    block_m: int,
    gemm: str,
) -> tuple[Callable[[dict[str, int]], None], torch.Tensor, list[torch.Tensor]]:
    payload = make_payload(num_tokens, routing)
    assert payload.expert_topk_ids is not None
    assert payload.expert_topk_weights is not None
    assert payload.expert_x is not None
    assert payload.expert_x_scale is not None
    topk_ids = payload.expert_topk_ids
    sorted_ids, expert_ids, padded = moe_align_block_size_compiled(
        topk_ids, block_m, NUM_EXPERTS
    )
    route_num = num_tokens * TOP_K

    if gemm == "gate_up":
        a = payload.expert_x
        a_scale = payload.expert_x_scale
        b = executor.w13_weight
        b_scale = executor.w13_weight_scale_inv
        output = torch.empty(
            (route_num, GATE_UP_SIZE), device="cuda", dtype=torch.bfloat16
        )
        mul_routed_weight = False
        kernel_topk = TOP_K
    elif gemm == "down":
        down_pattern = torch.arange(
            route_num * INTERMEDIATE_SIZE, device="cuda", dtype=torch.float32
        ).view(route_num, INTERMEDIATE_SIZE)
        down_input = ((down_pattern.remainder(13) - 6) * 0.00390625).to(torch.bfloat16)
        a, a_scale = sgl_per_token_group_quant_fp8(
            down_input,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        b = executor.w2_weight
        b_scale = executor.w2_weight_scale_inv
        output = torch.empty(
            (route_num, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
        )
        mul_routed_weight = True
        kernel_topk = 1
    else:
        raise ValueError(gemm)

    def invoke(config: dict[str, int]) -> None:
        invoke_fused_moe_kernel(
            a,
            b,
            output,
            payload.expert_topk_weights.view(-1),
            topk_ids.view(-1),
            sorted_ids,
            expert_ids,
            padded,
            mul_routed_weight,
            kernel_topk,
            config,
            tl.bfloat16,
            A_scale=a_scale,
            B_scale=b_scale,
            block_shape=[128, 128],
            scale_ue8m0=True,
        )

    keepalive = [
        a,
        a_scale,
        b,
        b_scale,
        output,
        topk_ids,
        payload.expert_topk_weights,
        sorted_ids,
        expert_ids,
        padded,
    ]
    return invoke, output, keepalive


def benchmark_kernel_config(
    executor: DeepGemmHybridExecutor,
    config: dict[str, int],
    gemm: str,
    tokens: list[int],
    routing: str,
    warmup: int,
    repetitions: int,
    samples: int,
) -> tuple[float, list[float]]:
    timings = []
    for num_tokens in tokens:
        invoke, output, keepalive = make_kernel_inputs(
            executor, num_tokens, routing, config["BLOCK_SIZE_M"], gemm
        )
        # Build the known-good result with independently aligned routing
        # metadata. Reusing the candidate's BLOCK_SIZE_M layout would allow an
        # alignment/config interaction to validate itself.
        reference_invoke, reference_output, reference_keepalive = make_kernel_inputs(
            executor, num_tokens, routing, 16, gemm
        )
        reference_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3 if gemm == "gate_up" else 2,
        }
        reference_output.fill_(torch.nan)
        reference_invoke(reference_config)
        if not torch.isfinite(reference_output).all():
            raise AssertionError(f"non-finite {gemm} reference output")
        reference = reference_output.clone()
        output.fill_(torch.nan)
        invoke(config)
        if not torch.isfinite(output).all():
            raise AssertionError(f"non-finite {gemm} candidate output: {config}")
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        timing = benchmark_graph(
            lambda: invoke(config),
            warmup=warmup,
            repetitions=repetitions,
            samples=samples,
        )
        timings.append(timing)
        del keepalive
        del reference_keepalive
    return statistics.geometric_mean(timings), timings


def benchmark_tuning(
    executor: DeepGemmHybridExecutor,
    tokens: list[int],
    routing: str,
    warmup: int,
    repetitions: int,
    samples: int,
    verbose: bool,
) -> None:
    # Establish an independent DeepGEMM oracle before ranking any candidate.
    # The per-expert scales make routing mistakes visible, while the per-K
    # scales cover all packed UE8M0 words of both Qwen GEMMs.
    for num_tokens in tokens:
        payload = make_payload(num_tokens, routing)
        torch.testing.assert_close(
            triton_call(executor, payload),
            deepgemm_call(executor, payload),
            rtol=2e-2,
            atol=2e-2,
        )

    for gemm in ("gate_up", "down"):
        results = []
        errors = []
        configs = candidate_configs()
        for index, config in enumerate(configs, start=1):
            try:
                score, timings = benchmark_kernel_config(
                    executor,
                    config,
                    gemm,
                    tokens,
                    routing,
                    warmup,
                    repetitions,
                    samples,
                )
            except Exception as error:
                torch.cuda.synchronize()
                errors.append({"config": config, "error": str(error)})
                continue
            result = {
                "type": "tune",
                "gemm": gemm,
                "config": config,
                "tokens": tokens,
                "timings_ms": timings,
                "score_ms": score,
            }
            results.append(result)
            if verbose:
                print(json.dumps(result, sort_keys=True), flush=True)
            elif index % 24 == 0:
                print(
                    json.dumps(
                        {
                            "type": "tune_progress",
                            "gemm": gemm,
                            "completed": index,
                            "total": len(configs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        results.sort(key=lambda result: result["score_ms"])
        if not results:
            raise RuntimeError(
                f"all {len(configs)} {gemm} tuning candidates failed: {errors[:3]}"
            )
        print(
            json.dumps(
                {
                    "type": "tune_best",
                    "gemm": gemm,
                    "error_count": len(errors),
                    "error_examples": errors[:10],
                    "results": results[:10],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("threshold", "tune"), default="threshold")
    parser.add_argument("--tokens", default="1,2,4,8,12,16,20,24,28,32,40,48,64")
    parser.add_argument("--routing", default="balanced,random,skewed")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("SM120 benchmark must run on RTX PRO 5000 Blackwell")
    # The production server captures a short, configured list of decode batch
    # sizes.  This benchmark intentionally sweeps more than Dynamo's default
    # eight static-shape specializations in one process.
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, len(parse_int_list(args.tokens)) + 4
    )
    print(
        json.dumps(
            {
                "type": "environment",
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "tokens": parse_int_list(args.tokens),
                "mode": args.mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    executor = make_executor()
    if args.mode == "threshold":
        benchmark_thresholds(
            executor,
            parse_int_list(args.tokens),
            args.routing.split(","),
            args.warmup,
            args.repetitions,
            args.samples,
        )
    else:
        routing = args.routing.split(",")[0]
        benchmark_tuning(
            executor,
            parse_int_list(args.tokens),
            routing,
            args.warmup,
            args.repetitions,
            args.samples,
            args.verbose,
        )


if __name__ == "__main__":
    main()

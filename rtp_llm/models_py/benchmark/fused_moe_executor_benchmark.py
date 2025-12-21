import itertools
import math
import random
from unittest import SkipTest, TestCase, main

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.moe.executors.deepep_normal_executor import (
    DeepGemmContinousExecutor,
)
from rtp_llm.models_py.modules.quantization.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
)
from rtp_llm.utils.model_weight import W


class FusedMoEExecutorBenchmark(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_deepgemm_continuous_benchmark(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        expert_num: int,
        dp_size: int,
    ):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)

        config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            layer_num=1,
            max_seq_len=4096,
            vocab_size=32000,
        )
        config.expert_num = expert_num
        config.ep_size = dp_size
        config.ep_rank = 0
        config.moe_k = 2
        config.moe_inter_padding_size = inter_size
        config.activation_type = "silu"

        # In DP (EP), we need to split expert numbers
        num_local_experts = expert_num // dp_size
        local_expert_num = num_local_experts

        # Generate weights
        torch_dtype = torch.float8_e4m3fn
        w1_scale = torch.randn(
            (local_expert_num, inter_size * 2 // 128, hidden_size // 128),
            device="cuda",
            dtype=torch.float32,
        )
        w2_scale = torch.randn(
            (local_expert_num, hidden_size // 128, inter_size // 128),
            device="cuda",
            dtype=torch.float32,
        )
        weights = {
            W.moe_w1: torch.randn(
                (local_expert_num, inter_size * 2, hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            ).to(dtype=torch_dtype),
            W.moe_w2: torch.randn(
                (local_expert_num, hidden_size, inter_size),
                device="cuda",
                dtype=torch.bfloat16,
            ).to(dtype=torch_dtype),
            W.moe_s1: w1_scale,
            W.moe_s2: w2_scale,
        }

        executor = DeepGemmContinousExecutor(config, weights)

        # Pre-generate payloads
        payloads = []
        warmup_iter = 10
        bench_iter = 20
        profile_iter = 5
        total_iter = warmup_iter + bench_iter + profile_iter

        for _ in range(total_iter):
            # Generate payload
            expert_x = torch.randn(
                (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
            ).to(dtype=torch_dtype)

            if is_deep_gemm_e8m0_used():
                # Match shape logic in DeepGemmContinousExecutor.execute
                scale_dim = (hidden_size // 128 + 3) // 4
                expert_x_scale = torch.zeros(
                    (num_tokens, scale_dim), device="cuda", dtype=torch.int
                )
            else:
                expert_x_scale = torch.randn(
                    (num_tokens, hidden_size // 128), device="cuda", dtype=torch.float32
                )

            expert_num_tokens = torch.zeros(
                (local_expert_num,), device="cuda", dtype=torch.int32
            )
            # Distribute tokens
            base_tokens = num_tokens // local_expert_num
            remainder = num_tokens % local_expert_num
            for i in range(local_expert_num):
                expert_num_tokens[i] = base_tokens + (1 if i < remainder else 0)

            topk_ids = torch.randint(
                0,
                local_expert_num,
                (num_tokens, config.moe_k),
                device="cuda",
                dtype=torch.int32,
            )
            topk_weights = torch.rand(
                (num_tokens, config.moe_k), device="cuda", dtype=torch.float32
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            payload = ExpertForwardPayload(
                expert_x=expert_x,
                expert_x_origin_dtype=torch.bfloat16,
                expert_x_scale=expert_x_scale,
                expert_tokens_meta=ExpertTokensMetadata(
                    expert_num_tokens=expert_num_tokens,
                    expert_num_tokens_cpu=None,
                ),
                expert_topk_ids=topk_ids,
                expert_topk_weights=topk_weights,
            )
            payloads.append(payload)

        # Warmup
        for i in range(warmup_iter):
            executor.execute(
                payload=payloads[i],
                activation="silu",
                global_num_experts=expert_num,
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            )

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        print(
            f"\nBenchmarking DeepGemmContinousExecutor: num_tokens={num_tokens}, hidden_size={hidden_size}, expert_num={expert_num}, dp_size={dp_size}"
        )

        # Measure Total Time
        start_event.record()
        for i in range(warmup_iter, warmup_iter + bench_iter):
            executor.execute(
                payload=payloads[i],
                activation="silu",
                global_num_experts=expert_num,
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / bench_iter
        print(f"Average execution time: {avg_time_ms:.4f} ms")

        # Profile Kernel Times
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            # Run a few iterations for profiling
            for i in range(warmup_iter + bench_iter, total_iter):
                executor.execute(
                    payload=payloads[i],
                    activation="silu",
                    global_num_experts=expert_num,
                    expert_map=None,
                    a2_scale=None,
                    apply_router_weight_on_input=False,
                    extra_expert_args=None,
                )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    def test_deepgemm_continuous_benchmark(self):
        NUM_TOKENS_LIST = [4, 8, 16, 32, 64]
        HIDDEN_SIZE_LIST = [6144]
        INTER_SIZE_LIST = [5120]
        EXPERT_NUM_LIST = [160]
        DP_SIZE_LIST = [4]

        for params in itertools.product(
            NUM_TOKENS_LIST,
            HIDDEN_SIZE_LIST,
            INTER_SIZE_LIST,
            EXPERT_NUM_LIST,
            DP_SIZE_LIST,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                inter_size=params[2],
                expert_num=params[3],
                dp_size=params[4],
            ):
                self._run_deepgemm_continuous_benchmark(*params)

    def _run_fp8_per_block_prefill_moe(
        self,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        expert_num: int,
        dp_size: int,
    ):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)

        config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            layer_num=1,
            max_seq_len=4096,
            vocab_size=32000,
        )
        config.expert_num = expert_num
        config.ep_size = dp_size
        config.ep_rank = 0
        config.moe_k = 2
        config.moe_inter_padding_size = inter_size
        config.activation_type = "silu"

        num_local_experts = expert_num // dp_size
        local_expert_num = num_local_experts

        # Generate weights
        torch_dtype = torch.float8_e4m3fn
        w1_scale = torch.randn(
            (local_expert_num, inter_size * 2 // 128, hidden_size // 128),
            device="cuda",
            dtype=torch.float32,
        )
        w2_scale = torch.randn(
            (local_expert_num, hidden_size // 128, inter_size // 128),
            device="cuda",
            dtype=torch.float32,
        )
        weights = {
            W.moe_w1: torch.randn(
                (local_expert_num, inter_size * 2, hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            ).to(dtype=torch_dtype),
            W.moe_w2: torch.randn(
                (local_expert_num, hidden_size, inter_size),
                device="cuda",
                dtype=torch.bfloat16,
            ).to(dtype=torch_dtype),
            W.moe_s1: w1_scale,
            W.moe_s2: w2_scale,
        }

        executor = DeepGemmContinousExecutor(config, weights)

        # Pre-generate payloads for all iterations
        payloads = []
        total_iter = 5 + 10

        for _ in range(total_iter):
            # Generate payload
            expert_x = torch.randn(
                (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
            ).to(dtype=torch_dtype)

            if is_deep_gemm_e8m0_used():
                # Match shape logic in DeepGemmContinousExecutor.execute
                # K // 128 // 4 (packed)
                scale_dim = (hidden_size // 128 + 3) // 4
                expert_x_scale = torch.zeros(
                    (num_tokens, scale_dim), device="cuda", dtype=torch.int
                )
            else:
                expert_x_scale = torch.randn(
                    (num_tokens, hidden_size // 128), device="cuda", dtype=torch.float32
                )

            # Distribute tokens to experts
            expert_num_tokens = torch.zeros(
                (local_expert_num,), device="cuda", dtype=torch.int32
            )
            # Randomly assign counts that sum to num_tokens
            # Simplification: distribute evenly + remainder
            base_tokens = num_tokens // local_expert_num
            remainder = num_tokens % local_expert_num
            for i in range(local_expert_num):
                expert_num_tokens[i] = base_tokens + (1 if i < remainder else 0)

            topk_ids = torch.randint(
                0,
                local_expert_num,
                (num_tokens, config.moe_k),
                device="cuda",
                dtype=torch.int32,
            )
            topk_weights = torch.rand(
                (num_tokens, config.moe_k), device="cuda", dtype=torch.float32
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            payload = ExpertForwardPayload(
                expert_x=expert_x,
                expert_x_origin_dtype=torch.bfloat16,
                expert_x_scale=expert_x_scale,
                expert_tokens_meta=ExpertTokensMetadata(
                    expert_num_tokens=expert_num_tokens,
                    expert_num_tokens_cpu=None,
                ),
                expert_topk_ids=topk_ids,
                expert_topk_weights=topk_weights,
            )
            payloads.append(payload)

        # Warmup
        for i in range(5):
            executor.execute(
                payload=payloads[i],
                activation="silu",
                global_num_experts=expert_num,
                expert_map=None,
                a2_scale=None,
                apply_router_weight_on_input=False,
                extra_expert_args=None,
            )

        # Profile
        print(
            f"\nBenchmarking with num_tokens={num_tokens}, hidden_size={hidden_size}, expert_num={expert_num}, dp_size={dp_size}"
        )
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for i in range(5, 15):
                executor.execute(
                    payload=payloads[i],
                    activation="silu",
                    global_num_experts=expert_num,
                    expert_map=None,
                    a2_scale=None,
                    apply_router_weight_on_input=False,
                    extra_expert_args=None,
                )

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    def test_fp8_per_block_prefill_moe(self):
        NUM_TOKENS_LIST = [4, 8, 16, 32, 64]
        HIDDEN_SIZE_LIST = [6144]
        INTER_SIZE_LIST = [5120]
        EXPERT_NUM_LIST = [160]
        DP_SIZE_LIST = [4]

        for params in itertools.product(
            NUM_TOKENS_LIST,
            HIDDEN_SIZE_LIST,
            INTER_SIZE_LIST,
            EXPERT_NUM_LIST,
            DP_SIZE_LIST,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                inter_size=params[2],
                expert_num=params[3],
                dp_size=params[4],
            ):
                self._run_fp8_per_block_prefill_moe(*params)


if __name__ == "__main__":
    main()

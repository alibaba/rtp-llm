"""Unit tests for DeepGemmNormalEpBf16Executor.

Tests verify:
1. Executor produces numerically correct BF16 output.
2. CudaNoQuantEpNormalMaskedStrategy is selected over CudaNoQuantDpNormalStrategy
   when deep_gemm and DeepEP are available on SM9+.
3. Performance comparison vs TritonFusedMoeExecutor (GLM-5 shapes, ep_size=4, batch=32).
"""

import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
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
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


def _make_config(
    num_experts: int = 64,
    hidden_size: int = 512,
    moe_inter_size: int = 256,
    ep_size: int = 4,
    ep_rank: int = 0,
    max_tokens: int = 128,
    top_k: int = 2,
) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 512
    model_config.vocab_size = 1000
    model_config.expert_num = num_experts
    model_config.hidden_size = hidden_size
    model_config.moe_inter_size = moe_inter_size
    model_config.moe_k = top_k

    parallelism_config = ParallelismConfig()
    parallelism_config.world_size = ep_size
    parallelism_config.dp_size = ep_size
    parallelism_config.tp_size = 1
    parallelism_config.ep_size = ep_size
    parallelism_config.dp_rank = ep_rank
    parallelism_config.tp_rank = 0
    parallelism_config.ep_rank = ep_rank
    parallelism_config.world_rank = ep_rank
    parallelism_config.local_world_size = 1

    moe_config = MoeConfig()
    moe_config.ll_num_max_token = max_tokens
    moe_config.use_deepep_moe = True
    moe_config.use_deepep_low_latency = False

    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


def _ref_forward(
    expert_x: torch.Tensor,  # (M, K) BF16 contiguous
    local_topk_ids: torch.Tensor,  # (M, topk) int32, local IDs
    topk_weights: torch.Tensor,  # (M, topk)
    w1: torch.Tensor,  # (E_local, N, K) BF16
    w2: torch.Tensor,  # (E_local, K, N//2) BF16
    num_local_experts: int,
) -> torch.Tensor:
    """Reference BF16 MoE forward in plain PyTorch."""
    M, K = expert_x.shape
    N = w1.shape[1]
    output = torch.zeros(M, K, device=expert_x.device, dtype=torch.float32)
    for m in range(M):
        for k_idx in range(local_topk_ids.shape[1]):
            eid = local_topk_ids[m, k_idx].item()
            w = topk_weights[m, k_idx].item()
            if eid < 0 or eid >= num_local_experts:
                continue
            x = expert_x[m].float()
            gate_up = x @ w1[eid].float().T  # (N,)
            gate = gate_up[N // 2 :]
            up = gate_up[: N // 2]
            hidden = up * torch.sigmoid(gate) * gate  # SiLU gate
            out = hidden @ w2[eid].float().T  # (K,)
            output[m] += out * w
    return output.to(torch.bfloat16)


class TestDeepGemmNormalEpBf16Executor(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)

    def _skip_if_unavailable(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        sm = torch.cuda.get_device_capability()
        if sm[0] < 9:
            self.skipTest(f"SM{sm[0]} < 9, deep_gemm masked BF16 not supported")
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                has_deep_gemm,
                is_deep_gemm_e8m0_used,
            )

            if not has_deep_gemm():
                self.skipTest("deep_gemm not available")
            if is_deep_gemm_e8m0_used():
                self.skipTest(
                    f"SM{sm[0]} uses e8m0 (Blackwell+), BF16 masked GEMM not supported"
                )
        except ImportError:
            self.skipTest("deepgemm_wrapper not importable")

    def test_executor_correctness(self):
        """Verify DeepGemmNormalEpBf16Executor output matches reference."""
        self._skip_if_unavailable()

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_normal_ep_bf16_executor import (
            DeepGemmNormalEpBf16Executor,
        )

        num_experts = 64
        ep_size = 4
        ep_rank = 0
        top_k = 2
        hidden_size = 512  # must be divisible by 512 for ep_gather BLOCK_D=512
        moe_inter_size = 256
        max_tokens = 64

        config = _make_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_inter_size=moe_inter_size,
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_tokens=max_tokens,
            top_k=top_k,
        )

        E_local = num_experts // ep_size
        rank_expert_offset = ep_rank * E_local
        K = hidden_size
        N = moe_inter_size * 2
        M = 48  # dispatched token count

        # BF16 weights
        w1 = (torch.randn(E_local, N, K, device="cuda") * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(E_local, K, N // 2, device="cuda") * 0.02).to(torch.bfloat16)
        weights = {W.moe_w1: w1, W.moe_w2: w2, W.moe_s1: None, W.moe_s2: None}

        # Contiguous input tokens (M, K)
        expert_x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)

        # Assign tokens to local experts (some invalid with -1)
        local_topk_ids = torch.full((M, top_k), -1, device="cuda", dtype=torch.int32)
        tokens_per_expert = [0] * E_local
        for i in range(M):
            eid = i % E_local
            local_topk_ids[i, 0] = eid
            tokens_per_expert[eid] += 1

        topk_weights = torch.ones(M, top_k, device="cuda", dtype=torch.float32)
        topk_weights[:, 1] = 0.0  # second slot invalid (weight=0, id=-1)

        # Global IDs as the router produces them (add rank_expert_offset, replace -1)
        global_topk_ids = torch.where(
            local_topk_ids == -1,
            torch.tensor(
                num_experts - 1 if rank_expert_offset == 0 else 0,
                device="cuda",
                dtype=torch.int64,
            ),
            local_topk_ids.to(torch.int64) + rank_expert_offset,
        )

        num_recv_tokens_per_expert = torch.tensor(
            tokens_per_expert, device="cuda", dtype=torch.int32
        )

        payload = ExpertForwardPayload(
            expert_x=expert_x.clone(),
            expert_x_scale=None,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=global_topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=num_recv_tokens_per_expert,
            ),
        )

        executor = DeepGemmNormalEpBf16Executor(
            config,
            FusedMoEQuantConfig(quant_dtype=None),
            weights,
        )

        result = executor.execute(payload, "silu", None, None, False, None)
        got = result.fused_expert_output  # (M, K)

        # Reference
        ref = _ref_forward(expert_x, local_topk_ids, topk_weights, w1, w2, E_local)

        diff = calc_diff(got, ref)
        self.assertLess(
            diff,
            0.03,
            f"Numerical difference too large: {diff:.4f} (expected < 0.03 for BF16)",
        )

    def test_strategy_selection_prefers_masked_executor(self):
        """Verify CudaNoQuantEpNormalMaskedStrategy wins over CudaNoQuantDpNormalStrategy."""
        self._skip_if_unavailable()

        from rtp_llm.models_py.modules.factory.fused_moe import (
            FusedMoeFactory,
            StrategyRegistry,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_normal_ep_bf16_executor import (
            DeepGemmNormalEpBf16Executor,
        )

        config = _make_config(
            num_experts=64,
            hidden_size=512,
            moe_inter_size=256,
            ep_size=4,
            ep_rank=0,
            max_tokens=128,
            top_k=2,
        )

        registry = FusedMoeFactory._registry
        try:
            strategy = registry.get_strategy(config)
        except Exception as e:
            self.skipTest(f"Strategy selection failed (likely DeepEP unavailable): {e}")

        attrs = strategy.get_attributes()
        executor_cls = attrs.get_executor_class()
        self.assertIs(
            executor_cls,
            DeepGemmNormalEpBf16Executor,
            f"Expected DeepGemmNormalEpBf16Executor but got {executor_cls.__name__}. "
            f"Selected strategy: {strategy.__class__.__name__}",
        )


class TestGlm5MoEPerf(unittest.TestCase):
    """Performance benchmark: DeepGemmNormalEpBf16Executor vs TritonFusedMoeExecutor.

    Mimics GLM-5 MoE with ep_size=4 (256 experts / 4 ranks = 64 per rank),
    batch=32 decode mode: each token dispatches top_k=8 across 4 ranks,
    so ~64 tokens/rank (32 * 8 / 4 = 64).
    """

    # GLM-5 model config
    NUM_EXPERTS = 256
    EP_SIZE = 4
    EP_RANK = 0
    E_LOCAL = NUM_EXPERTS // EP_SIZE  # 64
    K = 6144  # hidden_size
    INTER_SIZE = 2048  # moe_intermediate_size
    N = INTER_SIZE * 2  # 4096 (gate + up fused)
    TOP_K = 8

    # batch=32 decode: 32 * top_k / ep_size = 64 dispatched tokens per rank
    M = 64

    WARMUP = 20
    REPEAT = 200

    def _skip_if_unavailable(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        sm = torch.cuda.get_device_capability()
        if sm[0] < 9:
            self.skipTest(f"SM{sm[0]} < 9, deep_gemm BF16 masked not supported")
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                has_deep_gemm,
                is_deep_gemm_e8m0_used,
            )

            if not has_deep_gemm():
                self.skipTest("deep_gemm not available")
            if is_deep_gemm_e8m0_used():
                self.skipTest(
                    f"SM{sm[0]} uses e8m0 (Blackwell+), BF16 masked GEMM not supported"
                )
        except ImportError:
            self.skipTest("deepgemm_wrapper not importable")

    def _make_weights(self):
        torch.manual_seed(42)
        w1 = (
            (torch.randn(self.E_LOCAL, self.N, self.K) * 0.01).to(torch.bfloat16).cuda()
        )
        w2 = (
            (torch.randn(self.E_LOCAL, self.K, self.N // 2) * 0.01)
            .to(torch.bfloat16)
            .cuda()
        )
        return w1, w2

    def _make_payload(self):
        """Build ExpertForwardPayload matching the DeepEP normal dispatch format.

        All top_k slots point to local experts (round-robin across E_LOCAL)
        so both executors do identical useful work with no zero-weight masking.
        """
        M = self.M
        device = "cuda"
        torch.manual_seed(7)
        expert_x = (torch.randn(M, self.K, device=device) * 0.1).to(torch.bfloat16)

        local_topk_ids = torch.zeros(M, self.TOP_K, device=device, dtype=torch.int32)
        tokens_per_expert = [0] * self.E_LOCAL
        for i in range(M):
            for k in range(self.TOP_K):
                eid = (i * self.TOP_K + k) % self.E_LOCAL
                local_topk_ids[i, k] = eid
                tokens_per_expert[eid] += 1

        rank_offset = self.EP_RANK * self.E_LOCAL
        global_topk_ids = local_topk_ids.to(torch.int64) + rank_offset
        topk_weights = torch.full(
            (M, self.TOP_K), 1.0 / self.TOP_K, device=device, dtype=torch.float32
        )
        num_recv_per_expert = torch.tensor(
            tokens_per_expert, device=device, dtype=torch.int32
        )

        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=None,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=global_topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=num_recv_per_expert
            ),
        )

    def _bench(self, fn):
        for _ in range(self.WARMUP):
            fn()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        latencies = []
        for _ in range(self.REPEAT):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            latencies.append(start_ev.elapsed_time(end_ev))

        latencies.sort()
        n = len(latencies)
        return {
            "median_ms": latencies[n // 2],
            "p90_ms": latencies[int(n * 0.9)],
            "min_ms": latencies[0],
        }

    def test_perf_comparison(self):
        """GLM-5 ep_size=4 batch=32 decode: mega kernel vs Triton fused MoE."""
        self._skip_if_unavailable()

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_normal_ep_bf16_executor import (
            DeepGemmNormalEpBf16Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )

        w1, w2 = self._make_weights()
        weights = {W.moe_w1: w1, W.moe_w2: w2, W.moe_s1: None, W.moe_s2: None}
        config = _make_config(
            num_experts=self.NUM_EXPERTS,
            hidden_size=self.K,
            moe_inter_size=self.INTER_SIZE,
            ep_size=self.EP_SIZE,
            ep_rank=self.EP_RANK,
            max_tokens=self.M * 4,
            top_k=self.TOP_K,
        )
        quant_cfg = FusedMoEQuantConfig(quant_dtype=None)
        payload = self._make_payload()

        mega_exec = DeepGemmNormalEpBf16Executor(config, quant_cfg, weights)
        triton_exec = TritonFusedMoeExecutor(config, quant_cfg, weights)

        mega_stats = self._bench(
            lambda: mega_exec.execute(payload, "silu", None, None, False, None)
        )
        triton_stats = self._bench(
            lambda: triton_exec.execute(payload, "silu", None, None, False, None)
        )

        # Effective FLOPS: GEMM1 (M*top_k dispatches, each K→N) + GEMM2 (N//2→K)
        total_dispatches = self.M * self.TOP_K  # 512 token-expert pairs
        flops = 2 * total_dispatches * (self.N * self.K + self.K * (self.N // 2))

        def tflops(ms):
            return flops / (ms * 1e-3) / 1e12

        alignment = align(self.M, 128)
        expected_m = min(alignment, ceil_div(self.M * self.TOP_K, self.E_LOCAL))

        sep = "=" * 68
        print(f"\n{sep}")
        print("  GLM-5 MoE Perf: DeepGemmNormalEpBf16 vs TritonFusedMoe")
        print(
            f"  Model : GLM-5  E={self.NUM_EXPERTS}  hidden={self.K}  inter={self.INTER_SIZE}  top_k={self.TOP_K}"
        )
        print(
            f"  EP    : ep_size={self.EP_SIZE}  E_local={self.E_LOCAL}  ep_rank={self.EP_RANK}"
        )
        print(f"  Batch : batch=32 decode  M={self.M} dispatched tokens/rank")
        print(
            f"  GEMM  : alignment={alignment}  expected_m={expected_m}  eff_GFLOP={flops/1e9:.1f}"
        )
        print(f"  Iters : warmup={self.WARMUP}  repeat={self.REPEAT}")
        print(f"{sep}")
        print(f"  {'Executor':<28} {'median':>9} {'p90':>9} {'min':>9} {'TFLOPS':>9}")
        print(f"  {'-'*66}")
        print(
            f"  {'DeepGemmNormalEpBf16':<28}"
            f" {mega_stats['median_ms']:>8.3f}ms"
            f" {mega_stats['p90_ms']:>8.3f}ms"
            f" {mega_stats['min_ms']:>8.3f}ms"
            f" {tflops(mega_stats['median_ms']):>8.3f}"
        )
        print(
            f"  {'TritonFusedMoe':<28}"
            f" {triton_stats['median_ms']:>8.3f}ms"
            f" {triton_stats['p90_ms']:>8.3f}ms"
            f" {triton_stats['min_ms']:>8.3f}ms"
            f" {tflops(triton_stats['median_ms']):>8.3f}"
        )
        speedup = triton_stats["median_ms"] / mega_stats["median_ms"]
        print(f"{sep}")
        print(
            f"  Speedup (mega vs triton): {speedup:.2f}x  ({'faster' if speedup > 1 else 'slower'})"
        )
        print(f"{sep}\n")


if __name__ == "__main__":
    unittest.main()

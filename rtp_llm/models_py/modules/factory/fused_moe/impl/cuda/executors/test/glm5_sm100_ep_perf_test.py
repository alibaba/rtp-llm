"""Performance comparison: DeepGemm FP8 mega kernel vs TritonFusedMoeExecutor on SM100.

GLM-5 MoE shape, ep_size=4, batch=32 decode (64 dispatched tokens/rank).
Mega kernel path: BF16 weights -> on-the-fly FP8 quantize -> m_grouped_fp8_gemm_nt_masked x2.
Runs on SM100 (Blackwell / L20A). Skipped on SM < 9 or non-CUDA.
"""

import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    m_grouped_fp8_gemm_nt_masked,
)
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
    silu_and_mul_masked_post_quant_packed_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather, ep_scatter_bf16
from rtp_llm.models_py.utils.arch import get_num_device_sms
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# GLM-5 MoE shape
NUM_EXPERTS = 256
EP_SIZE = 4
EP_RANK = 0
E_LOCAL = NUM_EXPERTS // EP_SIZE  # 64
K = 6144  # hidden_size
INTER_SIZE = 2048
N = INTER_SIZE * 2  # 4096  (gate + up fused)
TOP_K = 8
M = 64  # batch=32 decode: 32 * 8 / 4 = 64 tokens/rank
EXPERT_ALIGNMENT = 128
BLOCK = 128  # FP8 quantisation block size

WARMUP = 20
REPEAT = 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config() -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 512
    model_config.vocab_size = 1000
    model_config.expert_num = NUM_EXPERTS
    model_config.hidden_size = K
    model_config.moe_inter_size = INTER_SIZE
    model_config.moe_k = TOP_K

    pc = ParallelismConfig()
    pc.world_size = EP_SIZE
    pc.dp_size = EP_SIZE
    pc.tp_size = 1
    pc.ep_size = EP_SIZE
    pc.dp_rank = EP_RANK
    pc.tp_rank = 0
    pc.ep_rank = EP_RANK
    pc.world_rank = EP_RANK
    pc.local_world_size = 1

    mc = MoeConfig()
    mc.ll_num_max_token = M * 4
    mc.use_deepep_moe = True
    mc.use_deepep_low_latency = False

    return MoEConfigAdapter(
        model_config=model_config, parallelism_config=pc, moe_config=mc
    )


def _make_weights():
    torch.manual_seed(42)
    w1 = (torch.randn(E_LOCAL, N, K) * 0.01).to(torch.bfloat16).cuda()
    w2 = (torch.randn(E_LOCAL, K, N // 2) * 0.01).to(torch.bfloat16).cuda()
    return w1, w2


def _make_payload(w1, w2):
    """Build ExpertForwardPayload with global IDs (all valid, round-robin experts)."""
    torch.manual_seed(7)
    expert_x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
    local_topk_ids = torch.zeros(M, TOP_K, device="cuda", dtype=torch.int32)
    tokens_per_expert = [0] * E_LOCAL
    for i in range(M):
        for k in range(TOP_K):
            eid = (i * TOP_K + k) % E_LOCAL
            local_topk_ids[i, k] = eid
            tokens_per_expert[eid] += 1
    global_topk_ids = local_topk_ids.to(torch.int64) + EP_RANK * E_LOCAL
    topk_weights = torch.full(
        (M, TOP_K), 1.0 / TOP_K, device="cuda", dtype=torch.float32
    )
    num_recv = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int32)
    return ExpertForwardPayload(
        expert_x=expert_x,
        expert_x_scale=None,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=global_topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=num_recv),
    )


def _quantize_bf16_to_fp8_per_block(x: torch.Tensor, block_k: int = BLOCK):
    """Per-token × per-K-block FP8 quantisation.  Returns (fp8, scale).

    x    : (E, T, K)  bf16
    fp8  : (E, T, K)  float8_e4m3fn
    scale: (E, T, K // block_k)  float32  — gran_mn=1 → packed by m_grouped_fp8 wrapper
    """
    E, T, K_ = x.shape
    xf = x.float().reshape(E, T, K_ // block_k, block_k)
    scale = xf.abs().amax(dim=-1).clamp(min=1e-12) / 448.0  # (E, T, K//block_k)
    xq = (
        (xf / scale.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .reshape(E, T, K_)
        .to(torch.float8_e4m3fn)
    )
    return xq, scale


def _quantize_weight_fp8_per_block(
    w: torch.Tensor, block_n: int = BLOCK, block_k: int = BLOCK
):
    """Per-(block_n × block_k) FP8 quantisation of MoE weight.

    w    : (E, N_out, K_in)  bf16
    fp8  : (E, N_out, K_in)  float8_e4m3fn
    scale: (E, N_out // block_n, K_in // block_k)  float32
    """
    E, N_, K_ = w.shape
    assert N_ % block_n == 0 and K_ % block_k == 0
    wf = w.float().reshape(E, N_ // block_n, block_n, K_ // block_k, block_k)
    scale = wf.abs().amax(dim=(-1, -3)).clamp(min=1e-12) / 448.0  # (E, N//bn, K//bk)
    wq = (wf / scale[:, :, None, :, None]).clamp(-448.0, 448.0)
    wq = wq.reshape(E, N_, K_).to(torch.float8_e4m3fn)
    return wq, scale


def _bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(REPEAT):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))
    lats.sort()
    n = len(lats)
    return {"median_ms": lats[n // 2], "p90_ms": lats[int(n * 0.9)], "min_ms": lats[0]}


class TestGlm5Sm100EpPerfComparison(unittest.TestCase):
    """GLM-5 ep_size=4 batch=32 decode on SM100: FP8 mega kernel vs Triton BF16."""

    def setUp(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)

    def _skip_if_unavailable(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        sm = torch.cuda.get_device_capability()
        if sm[0] < 9:
            self.skipTest(f"SM{sm[0]} < 9, deep_gemm not supported")
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

            if not has_deep_gemm():
                self.skipTest("deep_gemm not available")
        except ImportError:
            self.skipTest("deepgemm_wrapper not importable")

    def test_perf_comparison(self):
        """Compare DeepGemm FP8 mega kernel vs TritonFusedMoeExecutor on SM100."""
        self._skip_if_unavailable()

        config = _make_config()
        w1_bf16, w2_bf16 = _make_weights()
        weights_bf16 = {
            W.moe_w1: w1_bf16,
            W.moe_w2: w2_bf16,
            W.moe_s1: None,
            W.moe_s2: None,
        }
        payload = _make_payload(w1_bf16, w2_bf16)

        # Pre-quantise weights (offline, excluded from timing)
        w1_fp8, w1_scale = _quantize_weight_fp8_per_block(w1_bf16)
        w2_fp8, w2_scale = _quantize_weight_fp8_per_block(
            w2_bf16, block_n=BLOCK, block_k=BLOCK
        )
        torch.cuda.synchronize()

        alignment = align(M, EXPERT_ALIGNMENT)
        expected_m = min(alignment, ceil_div(M * TOP_K, E_LOCAL))
        num_sms = get_num_device_sms()
        local_topk_ids = (payload.expert_topk_ids - EP_RANK * E_LOCAL).to(torch.int32)
        num_recv = payload.expert_tokens_meta.expert_num_tokens
        expert_x_orig = payload.expert_x

        # ------------------------------------------------------------------
        # FP8 mega kernel: scatter → quant → GEMM1 → SiLU FP8 → GEMM2 → gather
        # ------------------------------------------------------------------
        def run_fp8_mega():
            output_tensor = torch.zeros(
                E_LOCAL * alignment, K, device="cuda", dtype=torch.bfloat16
            )
            output_index = torch.full(
                local_topk_ids.shape, -1, device="cuda", dtype=torch.int32
            )
            expert_start_loc = torch.empty(E_LOCAL, device="cuda", dtype=torch.int32)

            ep_scatter_bf16(
                recv_x=expert_x_orig,
                recv_topk=local_topk_ids,
                alignment=alignment,
                expert_start_loc=expert_start_loc,
                output_tensor=output_tensor,
                output_index=output_index,
            )

            masked_input = output_tensor.view(E_LOCAL, alignment, K)

            # BF16 → FP8 (per-token × per-K-block)
            x_fp8, x_scale = _quantize_bf16_to_fp8_per_block(masked_input)

            with configure_deep_gemm_num_sms(num_sms):
                upgate = torch.empty(
                    E_LOCAL, alignment, N, device="cuda", dtype=torch.bfloat16
                )
                m_grouped_fp8_gemm_nt_masked(
                    (x_fp8, x_scale),
                    (w1_fp8, w1_scale),
                    upgate,
                    num_recv,
                    expected_m,
                )

                down_input = torch.empty(
                    E_LOCAL, alignment, N // 2, device="cuda", dtype=torch.float8_e4m3fn
                )
                down_scale = create_packed_scale_tensor(
                    E_LOCAL, alignment, N, BLOCK, "cuda"
                )
                silu_and_mul_masked_post_quant_packed_fwd(
                    upgate, down_input, down_scale, BLOCK, num_recv
                )

                down_out = torch.empty(
                    E_LOCAL, alignment, K, device="cuda", dtype=torch.bfloat16
                )
                m_grouped_fp8_gemm_nt_masked(
                    (down_input, down_scale),
                    (w2_fp8, w2_scale),
                    down_out,
                    num_recv,
                    expected_m,
                )

            gather_out = torch.empty(
                expert_x_orig.shape, device="cuda", dtype=torch.bfloat16
            )
            ep_gather(
                down_out.view(E_LOCAL * alignment, K),
                local_topk_ids,
                payload.expert_topk_weights,
                output_index,
                gather_out,
            )
            return gather_out

        # ------------------------------------------------------------------
        # Triton BF16 baseline
        # ------------------------------------------------------------------
        triton_exec = TritonFusedMoeExecutor(
            config, FusedMoEQuantConfig(quant_dtype=None), weights_bf16
        )

        def run_triton():
            return triton_exec.execute(payload, "silu", None, None, False, None)

        # Verify FP8 mega kernel actually runs
        _ = run_fp8_mega()
        torch.cuda.synchronize()

        fp8_stats = _bench(run_fp8_mega)
        triton_stats = _bench(run_triton)

        # FLOP accounting (same for both)
        total_dispatches = M * TOP_K  # 512 token-expert pairs
        flops = 2 * total_dispatches * (N * K + K * (N // 2))

        def tflops(ms):
            return flops / (ms * 1e-3) / 1e12

        speedup = triton_stats["median_ms"] / fp8_stats["median_ms"]
        sm = torch.cuda.get_device_capability()
        gpu = torch.cuda.get_device_name(0)

        sep = "=" * 72
        print(f"\n{sep}")
        print("  GLM-5 MoE Perf: DeepGemmFP8MegaKernel vs TritonFusedMoe")
        print(f"  GPU   : {gpu}  SM{sm[0]}.{sm[1]}")
        print(
            f"  Model : E={NUM_EXPERTS}  hidden={K}  inter={INTER_SIZE}  top_k={TOP_K}"
        )
        print(f"  EP    : ep_size={EP_SIZE}  E_local={E_LOCAL}  ep_rank={EP_RANK}")
        print(f"  Batch : decode batch=32  M={M} tokens/rank  alignment={alignment}")
        print(f"  GEMM  : expected_m={expected_m}  eff_GFLOP={flops / 1e9:.1f}")
        print(f"  FP8   : block={BLOCK}x{BLOCK}  weight pre-quantised offline")
        print(f"  Iters : warmup={WARMUP}  repeat={REPEAT}")
        print(f"{sep}")
        print(f"  {'Executor':<32} {'median':>9} {'p90':>9} {'min':>9} {'TFLOPS':>9}")
        print(f"  {'-' * 70}")
        print(
            f"  {'DeepGemmFP8 (scatter+quant+GEMM)':<32}"
            f" {fp8_stats['median_ms']:>8.3f}ms"
            f" {fp8_stats['p90_ms']:>8.3f}ms"
            f" {fp8_stats['min_ms']:>8.3f}ms"
            f" {tflops(fp8_stats['median_ms']):>8.3f}"
        )
        print(
            f"  {'TritonFusedMoe (BF16)':<32}"
            f" {triton_stats['median_ms']:>8.3f}ms"
            f" {triton_stats['p90_ms']:>8.3f}ms"
            f" {triton_stats['min_ms']:>8.3f}ms"
            f" {tflops(triton_stats['median_ms']):>8.3f}"
        )
        print(f"{sep}")
        verdict = "faster" if speedup > 1 else "slower"
        print(f"  Speedup DeepGemmFP8 vs Triton: {speedup:.2f}x  ({verdict})")
        print(f"  Note: FP8 timing includes on-the-fly activation quantisation.")
        print(f"        Weight quantisation is offline (excluded).")
        print(f"{sep}\n")

        # Sanity: output shape matches
        self.assertEqual(run_fp8_mega().shape, (M, K))


if __name__ == "__main__":
    unittest.main()

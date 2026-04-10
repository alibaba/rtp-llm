# -*- coding: utf-8 -*-
"""
Correctness and performance tests for the GDN (Gated Delta Net) prefill pipeline.

Tests the full _fla computation:
  fused_gdn_gating -> chunk_gated_delta_rule (with optional l2norm)

Correctness is validated against a token-by-token recurrent reference impl.
Performance is benchmarked with CUDA events across Qwen3.5 model configs.
"""

import logging
import os
import time
import unittest
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.utils import assert_close

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Reference implementations (fp32, token-by-token)
# ---------------------------------------------------------------------------


def gdn_gating_ref(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)."""
    x = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(x)
    beta = b.float().sigmoid()
    # reshape to match fused kernel output: (1, seq_len, num_heads)
    g = g.unsqueeze(0)
    beta = beta.unsqueeze(0)
    return g, beta


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Token-by-token recurrent reference (fp32) for GDN with GVA support.

    Handles Grouped Value Attention where H_v can be a multiple of H_k.
    q, k: (B, T, H_k, K), v: (B, T, H_v, V), g, beta: (B, T, H_v).
    State h: (B, H_v, K, V).
    """
    B, T, H_k, K = q.shape
    H_v = v.shape[2]
    V = v.shape[-1]
    GVA_ratio = H_v // H_k

    q = q.transpose(1, 2).contiguous().to(torch.float32)  # (B, H_k, T, K)
    k = k.transpose(1, 2).contiguous().to(torch.float32)  # (B, H_k, T, K)
    v = v.transpose(1, 2).contiguous().to(torch.float32)  # (B, H_v, T, V)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)  # (B, H_v, T)
    g = g.transpose(1, 2).contiguous().to(torch.float32)  # (B, H_v, T)

    o = torch.zeros(B, H_v, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, H_v, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.float()
    if scale is None:
        scale = K**-0.5

    for i in range(T):
        for j in range(H_v):
            k_head = j // GVA_ratio
            b_q = q[:, k_head, i] * scale  # (B, K)
            b_k = k[:, k_head, i]  # (B, K)
            b_v = v[:, j, i].clone()  # (B, V)
            h[:, j] = h[:, j].clone() * g[:, j, i].exp().unsqueeze(-1).unsqueeze(-1)
            b_beta = beta[:, j, i]  # (B,)
            b_v = b_v - (h[:, j].clone() * b_k.unsqueeze(-1)).sum(-2)
            b_v = b_v * b_beta.unsqueeze(-1)
            h[:, j] = h[:, j].clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
            o[:, j, i] = (b_q.unsqueeze(-1) * h[:, j]).sum(-2)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def full_gdn_prefill_ref(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Full GDN prefill reference: gating + l2norm(q,k) + recurrent with GVA."""
    g_ref, beta_ref = gdn_gating_ref(A_log, a, b, dt_bias)

    query, key, value = torch.split(
        mixed_qkv.float(),
        [num_k_heads * head_k_dim, num_k_heads * head_k_dim, num_v_heads * head_v_dim],
        dim=-1,
    )
    # (1, T, H_k, K) and (1, T, H_v, V)
    query = F.normalize(
        query.view(1, query.shape[0], num_k_heads, head_k_dim), p=2, dim=-1
    )
    key = F.normalize(key.view(1, key.shape[0], num_k_heads, head_k_dim), p=2, dim=-1)
    value = value.view(1, value.shape[0], num_v_heads, head_v_dim)
    # g_ref, beta_ref: (1, T, H_v)

    N = len(cu_seqlens) - 1
    refs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        h0_i = initial_state[i : i + 1] if initial_state is not None else None
        ref_o, ref_h = recurrent_gated_delta_rule_ref(
            q=query[:, s:e],
            k=key[:, s:e],
            v=value[:, s:e],
            beta=beta_ref[:, s:e],
            g=g_ref[:, s:e],
            initial_state=h0_i,
            output_final_state=True,
        )
        refs.append(ref_o)
        ref_states.append(ref_h)
    ref_out = torch.cat(refs, dim=1).squeeze(0)
    ref_final_state = torch.cat(ref_states, dim=0) if ref_states else None
    return ref_out, ref_final_state


# ---------------------------------------------------------------------------
# The actual GDN prefill under test (mirrors _fla logic)
# ---------------------------------------------------------------------------


def gdn_prefill_under_test(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Runs the actual fused gating + chunk_gated_delta_rule pipeline."""
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)

    query, key, value = torch.split(
        mixed_qkv,
        [num_k_heads * head_k_dim, num_k_heads * head_k_dim, num_v_heads * head_v_dim],
        dim=-1,
    )
    query = query.view(1, query.shape[0], num_k_heads, head_k_dim)
    key = key.view(1, key.shape[0], num_k_heads, head_k_dim)
    value = value.view(1, value.shape[0], num_v_heads, head_v_dim)

    attn_out, h, final_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    return attn_out.squeeze(0), h, final_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_inputs(
    total_seq_len: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    cu_seqlens: List[int],
    dtype: torch.dtype = torch.bfloat16,
    with_initial_state: bool = False,
):
    """Create random inputs for GDN prefill testing."""
    torch.manual_seed(42)
    qkv_dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    mixed_qkv = torch.randn(total_seq_len, qkv_dim, dtype=dtype, device=DEVICE)
    A_log = torch.randn(num_v_heads, dtype=dtype, device=DEVICE)
    a = torch.randn(total_seq_len, num_v_heads, dtype=dtype, device=DEVICE)
    b = torch.randn(total_seq_len, num_v_heads, dtype=dtype, device=DEVICE)
    dt_bias = torch.randn(num_v_heads, dtype=dtype, device=DEVICE)
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=DEVICE)

    N = len(cu_seqlens) - 1
    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(
            N,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.float32,
            device=DEVICE,
        )

    return mixed_qkv, A_log, a, b, dt_bias, cu_seqlens_t, initial_state


# ---------------------------------------------------------------------------
# Qwen3.5-35B-A3B model configs (per-TP shard)
# ---------------------------------------------------------------------------

# Qwen3.5-35B-A3B: h_k=16, h_v=64, d=128, GVA ratio=4
QWEN35_35B_CONFIGS = {
    "tp1": {"num_k_heads": 16, "num_v_heads": 64, "head_k_dim": 128, "head_v_dim": 128},
    "tp2": {"num_k_heads": 8, "num_v_heads": 32, "head_k_dim": 128, "head_v_dim": 128},
    "tp4": {"num_k_heads": 4, "num_v_heads": 16, "head_k_dim": 128, "head_v_dim": 128},
}


class TestGDNPrefillCorrectness(unittest.TestCase):
    """Correctness tests comparing fused pipeline vs token-by-token reference."""

    def _run_correctness(
        self,
        cu_seqlens: List[int],
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: torch.dtype,
        with_initial_state: bool,
        output_rtol: float = 0.005,
        state_rtol: float = 0.005,
    ):
        total_seq_len = cu_seqlens[-1]
        mixed_qkv, A_log, a, b, dt_bias, cu_seqlens_t, initial_state = make_test_inputs(
            total_seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            cu_seqlens,
            dtype,
            with_initial_state,
        )

        # Run the actual pipeline
        out, h, final_state = gdn_prefill_under_test(
            mixed_qkv,
            A_log,
            a,
            b,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            cu_seqlens_t,
            initial_state,
        )

        # Run reference
        ref_out, ref_final_state = full_gdn_prefill_ref(
            mixed_qkv,
            A_log,
            a,
            b,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            cu_seqlens_t,
            initial_state,
        )

        tag = (
            f"h_k={num_k_heads} h_v={num_v_heads} d={head_k_dim} "
            f"seqlens={cu_seqlens} init_state={with_initial_state}"
        )
        assert_close(f"output {tag}", ref_out.float(), out.float(), output_rtol)
        if final_state is not None and ref_final_state is not None:
            assert_close(
                f"final_state {tag}",
                ref_final_state.float(),
                final_state.float(),
                state_rtol,
            )

    # --- Single sequence tests ---

    def test_single_short_seq_bf16(self):
        """Single short sequence, bf16, no initial state."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 128], **cfg, dtype=torch.bfloat16, with_initial_state=False
        )

    def test_single_medium_seq_bf16(self):
        """Single medium sequence, bf16, no initial state."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 512], **cfg, dtype=torch.bfloat16, with_initial_state=False
        )

    def test_single_seq_with_initial_state(self):
        """Single sequence with initial state (simulates incremental prefill)."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 256], **cfg, dtype=torch.bfloat16, with_initial_state=True
        )

    def test_single_seq_non_chunk_aligned(self):
        """Sequence length not aligned to chunk_size=64."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 100], **cfg, dtype=torch.bfloat16, with_initial_state=False
        )

    def test_single_seq_fp16(self):
        """fp16 precision."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 256], **cfg, dtype=torch.float16, with_initial_state=False
        )

    # --- Variable length (batched) tests ---

    def test_varlen_two_seqs(self):
        """Two sequences with different lengths."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 128, 384], **cfg, dtype=torch.bfloat16, with_initial_state=True
        )

    def test_varlen_four_seqs(self):
        """Four sequences with varying lengths."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 64, 192, 320, 512],
            **cfg,
            dtype=torch.bfloat16,
            with_initial_state=True,
        )

    def test_varlen_uneven_seqs(self):
        """Uneven sequence lengths (not chunk-aligned)."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 37, 100, 213],
            **cfg,
            dtype=torch.bfloat16,
            with_initial_state=False,
        )

    # --- Different TP configs ---

    def test_tp1_config(self):
        """TP1 config (full model heads)."""
        cfg = QWEN35_35B_CONFIGS["tp1"]
        self._run_correctness(
            [0, 128], **cfg, dtype=torch.bfloat16, with_initial_state=False
        )

    def test_tp4_config(self):
        """TP4 config (fewer heads)."""
        cfg = QWEN35_35B_CONFIGS["tp4"]
        self._run_correctness(
            [0, 256], **cfg, dtype=torch.bfloat16, with_initial_state=True
        )

    # --- Edge cases ---

    def test_minimal_seq(self):
        """Minimal sequence (1 token)."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 1], **cfg, dtype=torch.bfloat16, with_initial_state=False
        )

    def test_exactly_one_chunk(self):
        """Exactly one chunk (64 tokens)."""
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._run_correctness(
            [0, 64], **cfg, dtype=torch.bfloat16, with_initial_state=True
        )


class TestGDNPrefillBenchmark(unittest.TestCase):
    """Performance benchmarks for GDN prefill pipeline."""

    def _benchmark(
        self,
        label: str,
        total_seq_len: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        cu_seqlens: List[int],
        dtype: torch.dtype = torch.bfloat16,
        warmup: int = 10,
        repeat: int = 50,
    ):
        mixed_qkv, A_log, a, b, dt_bias, cu_seqlens_t, initial_state = make_test_inputs(
            total_seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            cu_seqlens,
            dtype,
            with_initial_state=True,
        )

        def run():
            return gdn_prefill_under_test(
                mixed_qkv,
                A_log,
                a,
                b,
                dt_bias,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                cu_seqlens_t,
                initial_state,
            )

        # Warmup
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        # Benchmark with CUDA events
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for i in range(repeat):
            start_events[i].record()
            run()
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times.sort()
        # Use median of middle 80% to remove outliers
        trim = len(times) // 10
        trimmed = times[trim:-trim] if trim > 0 else times
        avg_ms = sum(trimmed) / len(trimmed)
        min_ms = min(times)
        max_ms = max(times)
        p50 = times[len(times) // 2]
        p90 = times[int(len(times) * 0.9)]

        logger.info(
            f"[BENCHMARK] {label}: avg={avg_ms:.3f}ms min={min_ms:.3f}ms "
            f"max={max_ms:.3f}ms p50={p50:.3f}ms p90={p90:.3f}ms "
            f"(seq_len={total_seq_len}, h_k={num_k_heads}, h_v={num_v_heads}, d={head_k_dim})"
        )
        return avg_ms

    def test_bench_tp2_seq256(self):
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._benchmark("TP2 seq=256", 256, **cfg, cu_seqlens=[0, 256])

    def test_bench_tp2_seq1024(self):
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._benchmark("TP2 seq=1024", 1024, **cfg, cu_seqlens=[0, 1024])

    def test_bench_tp2_seq4096(self):
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._benchmark("TP2 seq=4096", 4096, **cfg, cu_seqlens=[0, 4096])

    def test_bench_tp2_seq8192(self):
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._benchmark("TP2 seq=8192", 8192, **cfg, cu_seqlens=[0, 8192])

    def test_bench_tp1_seq4096(self):
        cfg = QWEN35_35B_CONFIGS["tp1"]
        self._benchmark("TP1 seq=4096", 4096, **cfg, cu_seqlens=[0, 4096])

    def test_bench_tp4_seq4096(self):
        cfg = QWEN35_35B_CONFIGS["tp4"]
        self._benchmark("TP4 seq=4096", 4096, **cfg, cu_seqlens=[0, 4096])

    def test_bench_tp2_batch4_varlen(self):
        cfg = QWEN35_35B_CONFIGS["tp2"]
        self._benchmark(
            "TP2 batch4 varlen",
            4096,
            **cfg,
            cu_seqlens=[0, 1024, 2048, 3072, 4096],
        )


if __name__ == "__main__":
    unittest.main()

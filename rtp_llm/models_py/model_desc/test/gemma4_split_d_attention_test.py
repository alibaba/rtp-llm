"""Focused oracle tests for Gemma4 two-phase split-D attention.

Phase 1 computes one causal FP32 online-softmax ``(m,l)`` trajectory over the
four D=128 QK chunks.  Phase 2 recomputes QK against those fixed statistics,
forms BF16 P, applies both Dv=256 V halves, and normalizes by the shared l.

The stats oracle remains pure fp32 eager math with its original 2e-3 gate.
The output oracle follows the model contract (FP32 scores/softmax, BF16 P,
FP32 PV accumulation, BF16 output) with the established BF16 5e-2 gate.
"""

import os
from unittest import TestCase, main

import torch
from rtp_llm.models_py.kernels.cuda.gemma4_full_attention import (
    Gemma4SplitDPrefillWrapper,
    gemma4_split_d_attention,
    gemma4_split_d_attention_support,
    gemma4_split_d_stats,
    gemma4_split_d_support,
)

H_Q = 16
H_KV = 2
HEAD_DIM = 512
# 64: single partial tile (sub-tile TMA fill); 128: exact one tile; 129:
# partial second tile; 256/1152/2048: multi-tile causality at increasing
# trip counts.
SEQ_LENS = [64, 128, 129, 256, 1152, 2048]
OUTPUT_SEQ_LENS = [64, 128, 129, 256]
OUTPUT_RTOL = 5e-2
OUTPUT_ATOL = 5e-2


def reference_stats(q: torch.Tensor, k: torch.Tensor, sm_scale: float):
    """fp32 eager oracle.

    Returns (m, l), each ``[T, Hq]``: ``m`` = row max of the raw (unscaled)
    masked scores, ``l`` = sum over unmasked keys of exp((S - m) * sm_scale)
    — the exact statistics the kernel stores.
    """
    seq_len = q.shape[0]
    h_r = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(h_r, dim=1)
    scores = torch.matmul(
        q.transpose(0, 1).unsqueeze(0),
        k_expanded.transpose(0, 1).unsqueeze(0).transpose(2, 3),
    ).squeeze(0)
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).triu(
        diagonal=1
    )
    scores = scores.masked_fill(causal, float("-inf")).float()
    m = scores.amax(dim=-1)  # [Hq, T]
    l = torch.exp((scores - m.unsqueeze(-1)) * sm_scale).sum(dim=-1)  # [Hq, T]
    return m.t(), l.t()


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Independent eager oracle with BF16 probabilities and FP32 PV."""
    seq_len = q.shape[0]
    h_r = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(h_r, dim=1)
    v_expanded = v.repeat_interleave(h_r, dim=1)
    scores = torch.matmul(
        q.transpose(0, 1).unsqueeze(0),
        k_expanded.transpose(0, 1).unsqueeze(0).transpose(2, 3),
    )
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).triu(
        diagonal=1
    )
    scores.masked_fill_(causal[None, None, :, :], torch.finfo(q.dtype).min)
    probs = torch.softmax(scores * sm_scale, dim=-1, dtype=torch.float32).to(
        torch.bfloat16
    )
    return (
        torch.matmul(probs, v_expanded.transpose(0, 1).unsqueeze(0))
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )


def chunked_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    seq_len = q.shape[0]
    h_r = q.shape[1] // k.shape[1]
    k_expanded = k.repeat_interleave(h_r, dim=1)
    v_expanded = v.repeat_interleave(h_r, dim=1)
    key_states = k_expanded.transpose(0, 1).unsqueeze(0)
    value_states = v_expanded.transpose(0, 1).unsqueeze(0)
    key_positions = torch.arange(seq_len, device=q.device)
    outputs = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        query_states = q[start:end].transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(query_states, key_states.transpose(2, 3))
        allowed = key_positions[None, :] <= key_positions[start:end, None]
        scores.masked_fill_(~allowed[None, None, :, :], torch.finfo(q.dtype).min)
        probs = torch.softmax(scores * sm_scale, dim=-1, dtype=torch.float32).to(
            torch.bfloat16
        )
        outputs.append(
            torch.matmul(probs, value_states).squeeze(0).transpose(0, 1).contiguous()
        )
    return torch.cat(outputs, dim=0)


class Gemma4SplitDStatsTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("CUDA is required by this dedicated SM100 target")
        if torch.cuda.get_device_capability()[0] != 10:
            self.fail(
                "SM100/SM103 (capability 10.x) is required by this dedicated SM100 target"
            )
        # The reference must be true fp32, not TF32.
        torch.backends.cuda.matmul.allow_tf32 = False

    def _rand_qk(self, seq_len: int, dtype: torch.dtype):
        q = (torch.randn(seq_len, H_Q, HEAD_DIM, device="cuda") * 0.25).to(dtype)
        k = (torch.randn(seq_len, H_KV, HEAD_DIM, device="cuda") * 0.25).to(dtype)
        return q, k

    def _rand_qkv(self, seq_len: int):
        q, k = self._rand_qk(seq_len, torch.bfloat16)
        v = (torch.randn(seq_len, H_KV, HEAD_DIM, device="cuda") * 0.25).to(
            torch.bfloat16
        )
        return q, k, v

    def _check_stats_case(
        self, seq_len: int, dtype: torch.dtype, sm_scale: float = 1.0
    ):
        q, k = self._rand_qk(seq_len, dtype)

        stats = gemma4_split_d_stats(q, k, sm_scale)
        self.assertEqual(tuple(stats.shape), (seq_len, H_Q, 2))
        self.assertEqual(stats.dtype, torch.float32)

        m_ref, l_ref = reference_stats(q, k, sm_scale)
        # m is a raw score (values ~O(10) here): tight atol.  l shifts
        # multiplicatively by exp(delta_m) plus its own sum-order error.
        torch.testing.assert_close(stats[..., 0], m_ref, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(stats[..., 1], l_ref, rtol=2e-3, atol=2e-3)

        # Determinism: fixed tiling and accumulation order, no atomics —
        # repeated runs must agree bitwise.
        stats_repeat = gemma4_split_d_stats(q, k, sm_scale)
        self.assertTrue(torch.equal(stats, stats_repeat))
        return stats

    def _check_output_case(self, seq_len: int, sm_scale: float = 1.0):
        q, k, v = self._rand_qkv(seq_len)
        output = gemma4_split_d_attention(q, k, v, sm_scale)
        self.assertEqual(tuple(output.shape), (seq_len, H_Q, HEAD_DIM))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(output.is_contiguous())

        reference = reference_attention(q, k, v, sm_scale)
        torch.testing.assert_close(
            output,
            reference,
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )

        output_repeat = gemma4_split_d_attention(q, k, v, sm_scale)
        self.assertTrue(torch.equal(output, output_repeat))

    def test_stats_match_fp32_reference_bf16(self):
        # Gemma4 full-attention shape/scale: [T, 16/2, 512], scale=1.0.
        for seq_len in SEQ_LENS:
            with self.subTest(seq_len=seq_len):
                self._check_stats_case(seq_len, torch.bfloat16)

    def test_stats_match_fp32_reference_fp16(self):
        for seq_len in (128, 129):
            with self.subTest(seq_len=seq_len):
                self._check_stats_case(seq_len, torch.float16)

    def test_stats_respect_sm_scale(self):
        # Non-unit scale exercises the exp2((S - m) * scale * log2e) path;
        # m stays the max of the UNSCALED scores in both kernel and oracle.
        self._check_stats_case(256, torch.bfloat16, sm_scale=0.5)

    def test_output_matches_bf16_reference_and_is_deterministic(self):
        for seq_len in OUTPUT_SEQ_LENS:
            with self.subTest(seq_len=seq_len):
                self._check_output_case(seq_len)

    def test_output_halves_use_the_same_stats(self):
        q, k, v = self._rand_qkv(129)
        v_swapped = torch.cat((v[..., 256:], v[..., :256]), dim=-1)

        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        output = wrapper.run_attention(q, k, v)
        output_swapped = wrapper.run_attention(q, k, v_swapped)

        # Swapping only V halves must swap output halves bitwise.  This guards
        # against either half deriving an independent softmax trajectory.
        self.assertTrue(torch.equal(output_swapped[..., :256], output[..., 256:]))
        self.assertTrue(torch.equal(output_swapped[..., 256:], output[..., :256]))

    def test_full_wrapper_preallocated_output_and_stats(self):
        q, k, v = self._rand_qkv(128)
        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        output_buf = torch.full_like(q, torch.nan)
        stats_buf = torch.full(
            (128, H_Q, 2), float("nan"), dtype=torch.float32, device="cuda"
        )
        output = wrapper.run_attention(q, k, v, out=output_buf, stats_out=stats_buf)
        self.assertIs(output, output_buf)
        self.assertFalse(torch.isnan(output.float()).any())
        self.assertFalse(torch.isnan(stats_buf).any())
        reference = reference_attention(q, k, v, 1.0)
        torch.testing.assert_close(
            output,
            reference,
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )

    def test_wrapper_preallocated_out(self):
        q, k = self._rand_qk(128, torch.bfloat16)
        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        out = torch.full(
            (128, H_Q, 2), float("nan"), dtype=torch.float32, device="cuda"
        )
        stats = wrapper.run(q, k, out=out)
        self.assertIs(stats, out)
        self.assertFalse(torch.isnan(stats).any())
        m_ref, l_ref = reference_stats(q, k, 1.0)
        torch.testing.assert_close(stats[..., 0], m_ref, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(stats[..., 1], l_ref, rtol=2e-3, atol=2e-3)

    def test_output_matches_chunked_reference_8k(self):
        if os.environ.get("GEMMA4_RUN_SPLIT_D_8K") != "1":
            self.skipTest("set GEMMA4_RUN_SPLIT_D_8K=1")
        q, k, v = self._rand_qkv(8192)
        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        wrapper.run_attention(q, k, v)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = wrapper.run_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        peak_bytes = torch.cuda.max_memory_allocated()
        reference = chunked_reference_attention(q, k, v, 1.0)
        torch.testing.assert_close(
            output,
            reference,
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )
        print(
            f"split-D T8192: elapsed_ms={elapsed_ms:.3f} " f"peak_bytes={peak_bytes}",
            flush=True,
        )

    def test_output_and_perf_64k(self):
        """Gate-shape scaling probe: T=65536 numerics (sampled rows) + perf.

        The fp32 chunked reference at 64K is memory-heavy but tractable
        (~64K x 16 x 512 fp32 = 2GB per buffer); verify a ROW SAMPLE
        against the reference plus full-output sanity (no NaN, finite
        sums), and time the kernel with CUDA events.
        """
        if os.environ.get("GEMMA4_RUN_SPLIT_D_64K") != "1":
            self.skipTest("set GEMMA4_RUN_SPLIT_D_64K=1")
        T = 65536
        torch.manual_seed(20261007)
        q, k, v = self._rand_qkv(T)
        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        wrapper.run_attention(q, k, v)  # warmup/compile
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = wrapper.run_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[len(times) // 2]
        flops = 2.0 * H_Q * HEAD_DIM * T * T
        tflops = flops / (median_ms / 1e3) / 1e12
        # sampled-row numerics vs the chunked fp32 reference
        reference = chunked_reference_attention(q, k, v, 1.0)
        idx = torch.randperm(T, device=q.device)[:64]
        torch.testing.assert_close(
            output[idx],
            reference[idx],
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
            msg=lambda m: f"64K sampled-row mismatch: {m}",
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertTrue(torch.isfinite(output).all())
        del reference
        torch.cuda.empty_cache()
        print(
            f"split-D T65536: median_ms={median_ms:.3f} tflops={tflops:.1f} "
            f"({tflops / 1725.3 * 100:.1f}% of proxy)",
            flush=True,
        )

    def test_output_and_perf_128k(self):
        """128K kernel-level numerics + perf (20261007: the ENGINE A/B
        diverged from triton at 128K after 2 tokens while 64K is
        token-identical; this test isolates kernel-intrinsic vs
        engine-integration by comparing sampled rows vs the chunked fp32
        reference AT THE KERNEL LEVEL)."""
        if os.environ.get("GEMMA4_RUN_SPLIT_D_128K") != "1":
            self.skipTest("set GEMMA4_RUN_SPLIT_D_128K=1")
        T = 131072
        torch.manual_seed(20261008)
        q, k, v = self._rand_qkv(T)
        wrapper = Gemma4SplitDPrefillWrapper()
        wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
        output = wrapper.run_attention(q, k, v)
        torch.cuda.synchronize()
        reference = chunked_reference_attention(q, k, v, 1.0)
        idx = torch.randperm(T, device=q.device)[:128]
        diff = (output[idx].float() - reference[idx]).abs()
        max_diff = float(diff.max())
        torch.testing.assert_close(
            output[idx],
            reference[idx],
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
            msg=lambda m: f"128K sampled-row mismatch (max_abs={max_diff}): {m}",
        )
        self.assertFalse(torch.isnan(output).any())
        del reference
        torch.cuda.empty_cache()
        times = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wrapper.run_attention(q, k, v)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[1]
        flops = 2.0 * H_Q * HEAD_DIM * T * T
        tflops = flops / (median_ms / 1e3) / 1e12
        print(
            f"split-D T131072: median_ms={median_ms:.3f} tflops={tflops:.1f} "
            f"({tflops / 1725.3 * 100:.1f}% of proxy) sampled_max_diff={max_diff:.5f}",
            flush=True,
        )

    def test_phase_timing_64k_128k(self):
        """Per-phase CUDA-event timing: stats kernel vs apply kernel.

        Evidence base for the single-pass redesign decision. The apply
        kernel's grid Z dimension runs both 256-wide output halves, each
        recomputing QK, so the two-phase total executes QK three times:

            phase1 FLOPs (causal) = Hq * D * T^2          (QK only)
            phase2 FLOPs (causal) = 2 * Hq * (D + 256) * T^2
            useful  FLOPs (causal) = 2 * Hq * D * T^2     (kernel = 2.0x useful)

        A single-pass kernel (QK once + online softmax + both PV halves)
        would make kernel FLOPs = useful FLOPs; the projection below uses
        the measured per-phase raw rates.
        """
        if os.environ.get("GEMMA4_RUN_SPLIT_D_PHASE_TIMING") != "1":
            self.skipTest("set GEMMA4_RUN_SPLIT_D_PHASE_TIMING=1")
        for T in (65536, 131072):
            torch.manual_seed(20261008)
            q, k, v = self._rand_qkv(T)
            wrapper = Gemma4SplitDPrefillWrapper()
            wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
            stats = torch.empty((T, H_Q, 2), dtype=torch.float32, device=q.device)
            output = torch.empty(
                (T, H_Q, HEAD_DIM), dtype=torch.bfloat16, device=q.device
            )
            problem_size = (T, H_Q, H_KV, HEAD_DIM)
            # warmup both phases
            wrapper.run(q, k, out=stats)
            wrapper._compiled_apply(q, k, v, stats, output, problem_size, 1.0)
            torch.cuda.synchronize()
            self.assertFalse(torch.isnan(output).any())

            def _median(fn, n=5):
                times = []
                for _ in range(n):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                times.sort()
                return times[len(times) // 2]

            t1 = _median(lambda: wrapper.run(q, k, out=stats))
            t2 = _median(
                lambda: wrapper._compiled_apply(
                    q, k, v, stats, output, problem_size, 1.0
                )
            )
            f1 = H_Q * HEAD_DIM * T * T  # QK only, causal
            f2 = 2.0 * H_Q * (HEAD_DIM + 256) * T * T  # 2 halves x (QK + PV)
            fu = 2.0 * H_Q * HEAD_DIM * T * T  # useful causal FMHA
            r1 = f1 / (t1 / 1e3) / 1e12
            r2 = f2 / (t2 / 1e3) / 1e12
            ru = fu / ((t1 + t2) / 1e3) / 1e12
            raw_total = (f1 + f2) / ((t1 + t2) / 1e3) / 1e12
            # Single-pass projections. Apply does 2048 MMA units (2 halves x
            # (QK 512 + PV 256)); stats does 512 (QK only, plus softmax
            # handshake); a single-pass kernel does 1024 units (QK 512 + PV
            # 512) with the softmax handshake fused into the loop.
            #   softmax_overhead ~= t1 - t2*(512/2048) = t1 - t2/4
            #   full-rate-MMA projection (M=64 full rate or bf16-O tile_m=128):
            #       t1 - t2/4 + t2/2 = t1 + t2/4
            #   half-rate-MMA projection (M=64 at 50% tensor-core rate):
            #       t1 - t2/4 + t2 = t1 + 3*t2/4
            proj_full_ms = t1 + t2 / 4.0
            proj_half_ms = t1 + 3.0 * t2 / 4.0
            proj_full = fu / (proj_full_ms / 1e3) / 1e12
            proj_half = fu / (proj_half_ms / 1e3) / 1e12
            print(
                f"split-D phases T={T}: stats={t1:.1f}ms apply={t2:.1f}ms | "
                f"raw stats={r1:.0f} raw apply={r2:.0f} "
                f"raw_total={raw_total:.0f} TFLOP/s | "
                f"useful={ru:.0f} ({ru / 1725.3 * 100:.1f}% of proxy) | "
                f"single-pass proj full-rate={proj_full:.0f} "
                f"({proj_full / 1725.3 * 100:.1f}%), "
                f"half-rate={proj_half:.0f} "
                f"({proj_half / 1725.3 * 100:.1f}% of proxy)",
                flush=True,
            )
            del q, k, v, stats, output
            torch.cuda.empty_cache()

    def test_stats_mma_rate_m64_vs_m128(self):
        """Decisive M64 tcgen05 rate probe: stats kernel at tile_m=128 vs 64.

        Total QK work is identical for both configurations (same T, same
        causal structure); tile_m only changes the MMA instruction shape and
        the grid.  The wall-time ratio therefore measures the aggregate M64
        vs M128 tensor-core rate under the production softmax handshake.

        tile_m=64 stores WRONG stats (the Ld32x32b partition maps 2 threads
        per row); this test times only and asserts no NaN/hang.
        """
        if os.environ.get("GEMMA4_RUN_SPLIT_D_M64_PROBE") != "1":
            self.skipTest("set GEMMA4_RUN_SPLIT_D_M64_PROBE=1")
        from rtp_llm.models_py.kernels.cuda.gemma4_full_attention import (
            gemma4_split_d_prefill as _sdk,
        )

        T = 65536
        torch.manual_seed(20261008)
        q, k = self._rand_qk(T, torch.bfloat16)

        def _measure():
            wrapper = Gemma4SplitDPrefillWrapper()
            wrapper.plan(H_Q, H_KV, sm_scale=1.0, q_data_type=torch.bfloat16)
            wrapper.run(q, k)  # warmup/compile
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                wrapper.run(q, k)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            times.sort()
            return times[len(times) // 2]

        os.environ.pop("GEMMA4_SPLIT_D_TILE_M", None)
        _sdk._get_compiled_stats_kernel.cache_clear()
        t128 = _measure()

        os.environ["GEMMA4_SPLIT_D_TILE_M"] = "64"
        _sdk._get_compiled_stats_kernel.cache_clear()
        t64 = _measure()
        os.environ.pop("GEMMA4_SPLIT_D_TILE_M", None)
        _sdk._get_compiled_stats_kernel.cache_clear()

        # causal QK FLOPs of the stats pass: Hq * D * T^2 (GQA rows counted
        # on the Hq side; the h_r broadcast does not add QK work)
        f = H_Q * HEAD_DIM * T * T
        r128 = f / (t128 / 1e3) / 1e12
        r64 = f / (t64 / 1e3) / 1e12
        ratio = r64 / r128
        verdict = (
            "FULL-RATE" if ratio > 0.85 else ("HALF-RATE" if ratio < 0.6 else "PARTIAL")
        )
        print(
            f"M64 rate probe T={T}: tile_m=128 {t128:.1f}ms ({r128:.0f} raw "
            f"TFLOP/s) | tile_m=64 {t64:.1f}ms ({r64:.0f}) | "
            f"ratio={ratio:.2f} => {verdict}",
            flush=True,
        )

    def test_reject_unsupported_inputs(self):
        # All rejection paths are pure-Python validation: no kernel launch.
        q_ok = torch.zeros(4, H_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_ok = torch.zeros(4, H_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v_ok = torch.zeros_like(k_ok)
        self.assertTrue(gemma4_split_d_support(q_ok, k_ok))
        self.assertTrue(gemma4_split_d_attention_support(q_ok, k_ok, v_ok))

        v_fp16 = v_ok.to(torch.float16)
        self.assertFalse(gemma4_split_d_attention_support(q_ok, k_ok, v_fp16))
        with self.assertRaises(ValueError):
            gemma4_split_d_attention(q_ok, k_ok, v_fp16)

        v_bad_len = torch.zeros(8, H_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(gemma4_split_d_attention_support(q_ok, k_ok, v_bad_len))
        with self.assertRaises(ValueError):
            gemma4_split_d_attention(q_ok, k_ok, v_bad_len)

        # head_dim != 512.
        q = torch.zeros(4, H_Q, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(4, H_KV, 128, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(gemma4_split_d_support(q, k))
        with self.assertRaises(ValueError):
            gemma4_split_d_stats(q, k)

        # Hq % Hkv != 0.
        q = torch.zeros(4, 5, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(4, 2, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(gemma4_split_d_support(q, k))
        with self.assertRaises(ValueError):
            gemma4_split_d_stats(q, k)

        # Unsupported dtype (fp32).
        q = torch.zeros(4, H_Q, HEAD_DIM, dtype=torch.float32, device="cuda")
        k = torch.zeros(4, H_KV, HEAD_DIM, dtype=torch.float32, device="cuda")
        self.assertFalse(gemma4_split_d_support(q, k))
        with self.assertRaises(ValueError):
            gemma4_split_d_stats(q, k)

        # Sequence-length mismatch (single-sequence prototype).
        q = torch.zeros(4, H_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(8, H_KV, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(gemma4_split_d_support(q, k))
        with self.assertRaises(ValueError):
            gemma4_split_d_stats(q, k)

    def test_plan_rejects_head_dim(self):
        wrapper = Gemma4SplitDPrefillWrapper()
        with self.assertRaises(ValueError):
            wrapper.plan(H_Q, H_KV, q_data_type=torch.bfloat16, head_dim=256)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Consistency tests: FlashInfer GDN kernels vs original Triton/CuTe-DSL kernels.

Verifies that the new FlashInfer-based decode and prefill paths produce
numerically identical results to the existing implementations.
"""

import logging
import math
import random
import unittest
from typing import List

import torch
import torch.nn.functional as F

logging.basicConfig(
    level="INFO",
    format="[%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    return (
        F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    )


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float() - b.detach().float()).abs().max().item()


class TestFlashInferGDNDecodeConsistency(unittest.TestCase):
    """Compare FlashInfer gdn_decode_bf16_state vs Triton fused_recurrent_gated_delta_rule."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                gated_delta_rule as flashinfer_gdn_decode,
            )
            cls.flashinfer_gdn_decode = staticmethod(flashinfer_gdn_decode)
        except ImportError:
            raise unittest.SkipTest("flashinfer.gdn_kernels.gdn_decode_bf16_state not available")

        from rtp_llm.models_py.triton_kernels.fla import fused_recurrent_gated_delta_rule
        from rtp_llm.models_py.triton_kernels.fla.block import compute_state_indices_from_block_map
        from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

        cls.fused_recurrent = staticmethod(fused_recurrent_gated_delta_rule)
        cls.compute_indices = staticmethod(compute_state_indices_from_block_map)
        cls.fused_gdn_gating = staticmethod(fused_gdn_gating)

    def _run_decode_comparison(self, B: int, H: int, HV: int, D: int, seq_size_per_block: int = 128):
        device = "cuda"
        torch.manual_seed(42)

        # Model weights (shared)
        A_log = -torch.rand(HV, device=device, dtype=torch.float32).abs()
        dt_bias = torch.randn(HV, device=device, dtype=torch.float32) * 0.1

        # Inputs
        q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, 1, HV, D, dtype=torch.bfloat16, device=device)
        a = torch.randn(B, HV, dtype=torch.bfloat16, device=device)
        b = torch.randn(B, HV, dtype=torch.bfloat16, device=device)

        # Block map setup (mimics continuous batching)
        sequence_lengths = [random.randint(10, 1024) for _ in range(B)]
        block_num = [math.ceil(sl / seq_size_per_block) + 1 for sl in sequence_lengths]
        total_block_num = sum(block_num) + 1  # +1 for null block 0
        block_map = torch.zeros(B, max(block_num) + 1, dtype=torch.int32, device=device)
        offset = 1
        for i in range(B):
            block_map[i, :block_num[i]] = torch.arange(
                offset, offset + block_num[i], dtype=torch.int32
            )
            offset += block_num[i]

        # SSM states pool [pool_size, HV, V, K] bf16
        ssm_states_fi = torch.randn(
            total_block_num, HV, D, D, dtype=torch.bfloat16, device=device
        ) * 0.01
        ssm_states_triton = ssm_states_fi.clone()

        # sequence_lengths_plus_1_d (the "+1" convention used in rtp-llm)
        seq_lens_plus1 = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=device
        )

        # --- FlashInfer path ---
        read_indices, write_indices = self.compute_indices(
            seq_lens_plus1, block_map, seq_size_per_block
        )
        a_reshaped = a.view(B, 1, HV)
        b_reshaped = b.view(B, 1, HV)
        fi_out = self.flashinfer_gdn_decode(
            A_log=A_log,
            a=a_reshaped,
            dt_bias=dt_bias,
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            b=b_reshaped,
            initial_state_source=ssm_states_fi,
            initial_state_indices=read_indices,
            output_state_indices=write_indices,
            use_qk_l2norm_in_kernel=True,
        )
        fi_out = fi_out.squeeze(1)  # [B, HV, V]

        # --- Triton path ---
        # RTP-LLM Triton kernel expects state as [pool, HV, K, V] (K-major),
        # while FlashInfer uses [pool, HV, V, K] (V-major). Transpose for Triton.
        ssm_states_triton_t = ssm_states_triton.transpose(-1, -2).contiguous()
        g, beta = self.fused_gdn_gating(A_log, a, b, dt_bias)
        g = g.view(B, 1, HV)
        beta = beta.view(B, 1, HV)
        triton_out, _ = self.fused_recurrent(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states_triton_t,
            inplace_final_state=True,
            block_map=block_map,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=seq_lens_plus1,
            use_qk_l2norm_in_kernel=True,
        )
        triton_out = triton_out.reshape(B, HV, D)

        # --- Compare output ---
        sim = cos_sim(fi_out, triton_out)
        diff = max_abs_diff(fi_out, triton_out)
        logger.info(
            f"Decode B={B} H={H} HV={HV} D={D}: cos_sim={sim:.6f} max_abs_diff={diff:.6f}"
        )
        self.assertGreater(sim, 0.99, f"cos_sim too low: {sim}")

        # --- Compare updated states ---
        # Triton writes in [K, V] layout, FlashInfer in [V, K]. Transpose for comparison.
        for i in range(B):
            wi = write_indices[i].item()
            if wi > 0:
                fi_state = ssm_states_fi[wi]  # [HV, V, K]
                triton_state = ssm_states_triton_t[wi].transpose(-1, -2)  # [HV, K, V] -> [HV, V, K]
                state_sim = cos_sim(fi_state, triton_state)
                self.assertGreater(
                    state_sim, 0.99,
                    f"State cos_sim too low at batch {i} block {wi}: {state_sim}"
                )

    def test_B1(self):
        self._run_decode_comparison(B=1, H=16, HV=32, D=128)

    def test_B4(self):
        self._run_decode_comparison(B=4, H=16, HV=32, D=128)

    def test_B16(self):
        self._run_decode_comparison(B=16, H=16, HV=32, D=128)

    def test_B32(self):
        self._run_decode_comparison(B=32, H=16, HV=32, D=128)

    def test_B1_GQA(self):
        """H_qk != H_v (grouped query attention)."""
        self._run_decode_comparison(B=1, H=8, HV=32, D=128)

    def test_B8_small_block(self):
        """Smaller seq_size_per_block to test block boundary crossings."""
        self._run_decode_comparison(B=8, H=16, HV=32, D=128, seq_size_per_block=16)


class TestFlashInferGDNPrefillConsistency(unittest.TestCase):
    """Compare FlashInfer chunk_gated_delta_rule_sm100 vs Triton chunk_gated_delta_rule for prefill."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            raise unittest.SkipTest("Blackwell (SM100+) GPU required for prefill test")

        try:
            from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
                chunk_gated_delta_rule_sm100,
            )
            cls.flashinfer_prefill = staticmethod(chunk_gated_delta_rule_sm100)
        except ImportError as e:
            raise unittest.SkipTest(f"FlashInfer SM100 prefill not available: {e}")

        try:
            from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
            cls.old_prefill = staticmethod(chunk_gated_delta_rule)
        except ImportError as e:
            raise unittest.SkipTest(f"Triton chunk_gated_delta_rule not available: {e}")

        from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd

        cls.l2norm_fwd = staticmethod(l2norm_fwd)

    def _call_flashinfer(self, q, k, v, g, beta, initial_state, output_final_state, cu_seqlens):
        """Call FlashInfer SM100 kernel with the same shape convention as model code."""
        import math

        D = q.shape[-1]
        H_v = v.shape[2]
        total_tokens = q.shape[1]
        scale = 1.0 / math.sqrt(D)

        q_normed = self.l2norm_fwd(q)
        k_normed = self.l2norm_fwd(k)

        q_3d = q_normed.squeeze(0).contiguous()
        k_3d = k_normed.squeeze(0).contiguous()
        v_3d = v.squeeze(0).contiguous()
        gate = g.view(total_tokens, H_v).exp()
        beta_2d = beta.view(total_tokens, H_v).float()

        output = torch.empty_like(v_3d)
        output_state = None
        if output_final_state and initial_state is not None:
            output_state = torch.empty_like(initial_state)
        elif output_final_state:
            N = cu_seqlens.shape[0] - 1
            output_state = torch.empty(N, H_v, D, D, dtype=torch.float32, device=q.device)

        self.flashinfer_prefill(
            q=q_3d, k=k_3d, v=v_3d, gate=gate, beta=beta_2d,
            output=output, cu_seqlens=cu_seqlens,
            initial_state=initial_state, output_state=output_state, scale=scale,
        )

        output_4d = output.unsqueeze(0)
        return (output_4d, output_state) if output_final_state else output_4d

    def _run_prefill_comparison(
        self, H_qk: int, H_v: int, D: int, seq_lengths: List[int]
    ):
        device = "cuda"
        torch.manual_seed(42)

        total_tokens = sum(seq_lengths)
        N = len(seq_lengths)
        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=device)
        for i, sl in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        q = torch.randn(1, total_tokens, H_qk, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, total_tokens, H_qk, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, total_tokens, H_v, D, dtype=torch.bfloat16, device=device)
        g = F.logsigmoid(torch.randn(1, total_tokens, H_v, dtype=torch.float32, device=device))
        beta = torch.rand(1, total_tokens, H_v, dtype=torch.float32, device=device)
        initial_state = torch.randn(N, H_v, D, D, dtype=torch.float32, device=device) * 0.01

        # --- Triton chunked kernel ---
        old_out, _, old_state = self.old_prefill(
            q=q.clone(), k=k.clone(), v=v.clone(),
            g=g.clone(), beta=beta.clone(),
            initial_state=initial_state.clone(), output_final_state=True,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        )

        # --- FlashInfer SM100 ---
        new_out, new_state = self._call_flashinfer(
            q=q.clone(), k=k.clone(), v=v.clone(),
            g=g.clone(), beta=beta.clone(),
            initial_state=initial_state.clone(), output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        # --- Compare output ---
        out_sim = cos_sim(old_out, new_out)
        out_diff = max_abs_diff(old_out, new_out)
        logger.info(
            f"Prefill H_qk={H_qk} H_v={H_v} D={D} seqs={seq_lengths}: "
            f"output cos_sim={out_sim:.6f} max_abs_diff={out_diff:.6f}"
        )
        self.assertGreater(out_sim, 0.999, f"Output cos_sim too low: {out_sim}")

        # --- Compare final state ---
        # Triton state layout: [N, HV, K, V], FlashInfer: [N, HV, V, K]
        new_state_t = new_state.transpose(-1, -2).contiguous()
        state_sim = cos_sim(old_state, new_state_t)
        state_diff = max_abs_diff(old_state, new_state_t)
        logger.info(
            f"  state cos_sim={state_sim:.6f} max_abs_diff={state_diff:.6f}"
        )
        self.assertGreater(state_sim, 0.999, f"State cos_sim too low: {state_sim}")

    def test_single_short_seq(self):
        self._run_prefill_comparison(H_qk=16, H_v=32, D=128, seq_lengths=[256])

    def test_single_medium_seq(self):
        self._run_prefill_comparison(H_qk=16, H_v=32, D=128, seq_lengths=[1024])

    def test_single_long_seq(self):
        self._run_prefill_comparison(H_qk=16, H_v=32, D=128, seq_lengths=[4096])

    def test_multi_seq_varlen(self):
        self._run_prefill_comparison(H_qk=16, H_v=32, D=128, seq_lengths=[128, 256, 512])

    def test_no_initial_state(self):
        """Prefill without initial state (first request, no KV cache)."""
        device = "cuda"
        torch.manual_seed(42)
        H_qk, H_v, D = 16, 32, 128
        seq_lengths = [512]
        total_tokens = sum(seq_lengths)
        cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)

        q = torch.randn(1, total_tokens, H_qk, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, total_tokens, H_qk, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, total_tokens, H_v, D, dtype=torch.bfloat16, device=device)
        g = F.logsigmoid(torch.randn(1, total_tokens, H_v, dtype=torch.float32, device=device))
        beta = torch.rand(1, total_tokens, H_v, dtype=torch.float32, device=device)

        old_out, _, _ = self.old_prefill(
            q=q.clone(), k=k.clone(), v=v.clone(),
            g=g.clone(), beta=beta.clone(),
            initial_state=None, output_final_state=False,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        )
        new_out = self._call_flashinfer(
            q=q.clone(), k=k.clone(), v=v.clone(),
            g=g.clone(), beta=beta.clone(),
            initial_state=None, output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        if isinstance(new_out, tuple):
            new_out = new_out[0]

        sim = cos_sim(old_out, new_out)
        logger.info(f"Prefill no_initial_state: cos_sim={sim:.6f}")
        self.assertGreater(sim, 0.999, f"cos_sim too low: {sim}")


if __name__ == "__main__":
    unittest.main()

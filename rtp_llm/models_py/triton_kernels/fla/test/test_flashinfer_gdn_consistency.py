"""Production-path consistency test for Qwen3-Next Gated Delta Net.

Mimics the inference paths in Qwen3NextGatedDeltaNetPrefill / Decode and verifies
that FlashInfer (primary) and Triton (fallback) paths produce the same outputs
and the same ssm_states pool content after running the full load/kernel/store
pipeline that production uses.

Pool layout convention (post-Option-A fix): (pool_size, HV, head_v_dim, head_k_dim)
i.e. true (V, K) physical layout, matching FlashInfer's native expectation.
"""

import logging
import math
import random
import unittest
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.block import (
    compute_state_indices_from_block_map,
    load_initial_state_from_block_map,
    store_final_state_only_to_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd

try:
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
        chunk_gated_delta_rule_sm100 as _flashinfer_gdn_prefill,
    )
except ImportError:
    _flashinfer_gdn_prefill = None

try:
    from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as _flashinfer_gdn_decode,
    )
except ImportError:
    _flashinfer_gdn_decode = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@dataclass
class FakeAttnInputs:
    """Minimal stand-in for PyAttentionInputs (only the fields production reads)."""

    prefix_lengths_d: torch.Tensor
    cu_seqlens: torch.Tensor
    kv_cache_kernel_block_id_device: torch.Tensor
    sequence_lengths_plus_1_d: torch.Tensor


def make_pool(pool_size, hv, head_v_dim, head_k_dim, dtype, device="cuda", scale=0.01):
    """Allocate ssm_states pool in production (V, K) physical layout."""
    pool = (
        torch.randn(pool_size, hv, head_v_dim, head_k_dim, dtype=dtype, device=device)
        * scale
    )
    pool[0].zero_()  # null block
    return pool


def make_block_map(batch_size, block_counts, device="cuda"):
    max_blocks = max(block_counts)
    block_map = torch.zeros(
        batch_size, max_blocks + 1, dtype=torch.int32, device=device
    )
    offset = 1
    for i in range(batch_size):
        block_map[i, : block_counts[i]] = torch.arange(
            offset, offset + block_counts[i], dtype=torch.int32, device=device
        )
        offset += block_counts[i]
    return block_map


# ---------------------------------------------------------------------------
# Production paths (mimic qwen3_next.py)
# ---------------------------------------------------------------------------


def prefill_path_flashinfer(
    q,
    k,
    v,
    g,
    beta,
    pool,
    attn_inputs,
    seq_size_per_block,
    head_k_dim,
    head_v_dim,
    local_num_v_heads,
):
    """Mimic Qwen3NextGatedDeltaNetPrefill._fla_blackwell FlashInfer branch (post-fix)."""
    context_batch_size = attn_inputs.cu_seqlens.shape[0] - 1
    initial_states = torch.empty(
        context_batch_size,
        local_num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=torch.float32,
        device=q.device,
    )
    load_initial_state_from_block_map(
        attn_inputs.prefix_lengths_d,
        attn_inputs.kv_cache_kernel_block_id_device,
        pool,
        initial_states,
        seq_size_per_block,
    )

    total_tokens = q.shape[1]
    q_3d = l2norm_fwd(q.squeeze(0).contiguous())
    k_3d = l2norm_fwd(k.squeeze(0).contiguous())
    v_3d = v.squeeze(0).contiguous()
    gate = g.view(total_tokens, local_num_v_heads).exp()
    beta_2d = beta.view(total_tokens, local_num_v_heads).float()
    output = torch.empty_like(v_3d)
    output_state = torch.empty(
        context_batch_size,
        local_num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=torch.float32,
        device=q.device,
    )
    scale = 1.0 / math.sqrt(head_k_dim)
    _flashinfer_gdn_prefill(
        q=q_3d,
        k=k_3d,
        v=v_3d,
        gate=gate,
        beta=beta_2d,
        output=output,
        cu_seqlens=attn_inputs.cu_seqlens,
        initial_state=initial_states,
        output_state=output_state,
        scale=scale,
    )
    # Post-fix: write output_state directly without spurious transpose.
    store_final_state_only_to_block_map(
        output_state,
        attn_inputs.prefix_lengths_d,
        attn_inputs.cu_seqlens,
        attn_inputs.kv_cache_kernel_block_id_device,
        pool,
        seq_size_per_block,
    )
    return output.unsqueeze(0).squeeze_(0)


def prefill_path_triton(
    q,
    k,
    v,
    g,
    beta,
    pool,
    attn_inputs,
    seq_size_per_block,
    head_k_dim,
    head_v_dim,
    local_num_v_heads,
):
    """Mimic Qwen3NextGatedDeltaNetPrefill._fla Triton path (post-fix)."""
    context_batch_size = attn_inputs.cu_seqlens.shape[0] - 1
    initial_states = torch.empty(
        context_batch_size,
        local_num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=torch.float32,
        device=q.device,
    )
    load_initial_state_from_block_map(
        attn_inputs.prefix_lengths_d,
        attn_inputs.kv_cache_kernel_block_id_device,
        pool,
        initial_states,
        seq_size_per_block,
    )
    # Post-fix: pool stores (V, K); triton needs (K, V).
    tri_initial = initial_states.transpose(-1, -2).contiguous()
    attn_out, h, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=tri_initial,
        output_final_state=True,
        cu_seqlens=attn_inputs.cu_seqlens.long(),
        use_qk_l2norm_in_kernel=True,
    )
    # Post-fix: triton returns (K, V); transpose back to pool (V, K) before store.
    h = h.transpose(-1, -2).contiguous()
    final_state = final_state.transpose(-1, -2).contiguous()
    store_ssm_state_to_block_map(
        h,
        final_state,
        attn_inputs.prefix_lengths_d,
        attn_inputs.cu_seqlens,
        attn_inputs.kv_cache_kernel_block_id_device,
        pool,
        seq_size_per_block,
        chunk_size=64,
    )
    return attn_out.squeeze_(0)


def decode_path_flashinfer(
    query,
    key,
    value,
    a,
    b,
    pool,
    attn_inputs,
    seq_size_per_block,
    local_num_v_heads,
    alog,
    dt_bias,
):
    """Mimic Qwen3NextGatedDeltaNetDecode._fla_flashinfer_decode."""
    batch = query.shape[0]
    read_indices, write_indices = compute_state_indices_from_block_map(
        attn_inputs.sequence_lengths_plus_1_d,
        attn_inputs.kv_cache_kernel_block_id_device,
        seq_size_per_block,
    )
    a_r = a.view(batch, 1, local_num_v_heads)
    b_r = b.view(batch, 1, local_num_v_heads)
    core_attn_out = _flashinfer_gdn_decode(
        A_log=alog,
        a=a_r,
        dt_bias=dt_bias,
        q=query,
        k=key,
        v=value,
        b=b_r,
        initial_state_source=pool,
        initial_state_indices=read_indices,
        output_state_indices=write_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return core_attn_out.squeeze(1)


def decode_path_triton(
    query,
    key,
    value,
    a,
    b,
    pool,
    attn_inputs,
    seq_size_per_block,
    local_num_v_heads,
    alog,
    dt_bias,
):
    """Mimic Qwen3NextGatedDeltaNetDecode._fla Triton fallback (post-fix)."""
    batch, seq = query.shape[0], query.shape[1]
    g, beta = fused_gdn_gating(alog, a, b, dt_bias)
    g = g.view(batch, seq, local_num_v_heads)
    beta = beta.view(batch, seq, local_num_v_heads)
    # Post-fix: pool stores (V, K); triton expects (K, V).
    pool_kv = pool.transpose(-1, -2).contiguous()
    core_attn_out, _ = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        scale=None,
        initial_state=pool_kv,
        inplace_final_state=True,
        block_map=attn_inputs.kv_cache_kernel_block_id_device,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
        use_qk_l2norm_in_kernel=True,
    )
    pool.copy_(pool_kv.transpose(-1, -2).contiguous())
    return core_attn_out.reshape(-1, core_attn_out.shape[2], core_attn_out.shape[3])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrefillPathConsistency(unittest.TestCase):
    """FlashInfer prefill vs Triton prefill via production helpers."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        major, _ = torch.cuda.get_device_capability()
        if major < 10:
            raise unittest.SkipTest("Blackwell SM100+ required")
        if _flashinfer_gdn_prefill is None:
            raise unittest.SkipTest("flashinfer prefill SM100 not available")

    def _run(self, batch_size, seq_lengths):
        head_k_dim = head_v_dim = 128
        local_num_k_heads, local_num_v_heads = 16, 32
        seq_size_per_block = 64
        device = "cuda"

        torch.manual_seed(2026)
        random.seed(2026)
        total_tokens = sum(seq_lengths)

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for i, sl in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        q = torch.randn(
            1,
            total_tokens,
            local_num_k_heads,
            head_k_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k = torch.randn(
            1,
            total_tokens,
            local_num_k_heads,
            head_k_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.randn(
            1,
            total_tokens,
            local_num_v_heads,
            head_v_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        g = F.logsigmoid(
            torch.randn(
                1, total_tokens, local_num_v_heads, dtype=torch.float32, device=device
            )
        )
        beta = torch.rand(
            1, total_tokens, local_num_v_heads, dtype=torch.float32, device=device
        )

        prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
        block_counts = [math.ceil(sl / seq_size_per_block) + 1 for sl in seq_lengths]
        block_map = make_block_map(batch_size, block_counts, device=device)
        seq_lens_plus1 = torch.tensor(
            [sl + 1 for sl in seq_lengths], dtype=torch.int32, device=device
        )
        attn_inputs = FakeAttnInputs(
            prefix_lengths, cu_seqlens, block_map, seq_lens_plus1
        )

        pool_size = sum(block_counts) + 1
        pool_orig = make_pool(
            pool_size,
            local_num_v_heads,
            head_v_dim,
            head_k_dim,
            torch.bfloat16,
            device=device,
        )

        pool_fi = pool_orig.clone()
        out_fi = prefill_path_flashinfer(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            pool_fi,
            attn_inputs,
            seq_size_per_block,
            head_k_dim,
            head_v_dim,
            local_num_v_heads,
        )
        pool_tri = pool_orig.clone()
        out_tri = prefill_path_triton(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            pool_tri,
            attn_inputs,
            seq_size_per_block,
            head_k_dim,
            head_v_dim,
            local_num_v_heads,
        )

        out_sim = cos_sim(out_fi, out_tri)
        out_diff = max_abs_diff(out_fi, out_tri)
        logger.info(
            "Prefill seqs=%s: attn_out cos_sim=%.6f max_abs_diff=%.6f",
            seq_lengths,
            out_sim,
            out_diff,
        )
        self.assertGreater(out_sim, 0.999, f"attn_out cos_sim too low: {out_sim}")

        for i in range(batch_size):
            block_idx = (seq_lengths[i] - 1) // seq_size_per_block
            wi = block_map[i, block_idx].item()
            if wi <= 0:
                continue
            sim = cos_sim(pool_fi[wi], pool_tri[wi])
            diff = max_abs_diff(pool_fi[wi], pool_tri[wi])
            logger.info(
                "  batch %d wi=%d: state cos_sim=%.6f max_abs_diff=%.6f",
                i,
                wi,
                sim,
                diff,
            )
            self.assertGreater(sim, 0.999, f"state cos_sim too low (batch {i}): {sim}")

    def test_single_short(self):
        self._run(batch_size=1, seq_lengths=[256])

    def test_single_medium(self):
        self._run(batch_size=1, seq_lengths=[1024])

    def test_single_long(self):
        self._run(batch_size=1, seq_lengths=[4096])

    def test_multi_seq_varlen(self):
        self._run(batch_size=3, seq_lengths=[128, 256, 512])


class TestDecodePathConsistency(unittest.TestCase):
    """FlashInfer decode vs Triton decode via production helpers."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if _flashinfer_gdn_decode is None:
            raise unittest.SkipTest("flashinfer decode not available")

    def _run(
        self,
        B,
        past_seq_lengths,
        head_k=128,
        head_v=128,
        h_qk=16,
        h_v=32,
        seq_size_per_block=64,
    ):
        device = "cuda"
        torch.manual_seed(2026)
        random.seed(2026)

        block_counts = [
            math.ceil(sl / seq_size_per_block) + 1 for sl in past_seq_lengths
        ]
        pool_size = sum(block_counts) + 1
        block_map = make_block_map(B, block_counts, device=device)
        # +1 convention: sequence_lengths_plus_1 = past_seq_len + 1 (the current decode token position).
        seq_lens_plus1 = torch.tensor(
            [sl + 1 for sl in past_seq_lengths], dtype=torch.int32, device=device
        )
        prefix_lengths = torch.tensor(
            past_seq_lengths, dtype=torch.int32, device=device
        )
        cu_seqlens = torch.arange(
            0, B + 1, dtype=torch.int32, device=device
        )  # 1 token per batch
        attn_inputs = FakeAttnInputs(
            prefix_lengths, cu_seqlens, block_map, seq_lens_plus1
        )

        pool_orig = make_pool(
            pool_size, h_v, head_v, head_k, torch.bfloat16, device=device
        )

        query = torch.randn(B, 1, h_qk, head_k, dtype=torch.bfloat16, device=device)
        key = torch.randn(B, 1, h_qk, head_k, dtype=torch.bfloat16, device=device)
        value = torch.randn(B, 1, h_v, head_v, dtype=torch.bfloat16, device=device)
        a = torch.randn(B, h_v, dtype=torch.bfloat16, device=device)
        b = torch.randn(B, h_v, dtype=torch.bfloat16, device=device)
        alog = -torch.rand(h_v, device=device, dtype=torch.float32).abs()
        dt_bias = torch.randn(h_v, device=device, dtype=torch.float32) * 0.1

        pool_fi = pool_orig.clone()
        out_fi = decode_path_flashinfer(
            query.clone(),
            key.clone(),
            value.clone(),
            a.clone(),
            b.clone(),
            pool_fi,
            attn_inputs,
            seq_size_per_block,
            h_v,
            alog,
            dt_bias,
        )

        pool_tri = pool_orig.clone()
        out_tri = decode_path_triton(
            query.clone(),
            key.clone(),
            value.clone(),
            a.clone(),
            b.clone(),
            pool_tri,
            attn_inputs,
            seq_size_per_block,
            h_v,
            alog,
            dt_bias,
        )

        out_sim = cos_sim(out_fi, out_tri)
        out_diff = max_abs_diff(out_fi, out_tri)
        logger.info(
            "Decode B=%d past=%s: attn_out cos_sim=%.6f max_abs_diff=%.6f",
            B,
            past_seq_lengths,
            out_sim,
            out_diff,
        )
        self.assertGreater(out_sim, 0.999, f"attn_out cos_sim too low: {out_sim}")

        for i in range(B):
            new_seq_len = past_seq_lengths[i] + 1
            block_idx = (new_seq_len - 1) // seq_size_per_block
            wi = block_map[i, block_idx].item()
            if wi <= 0:
                continue
            sim = cos_sim(pool_fi[wi], pool_tri[wi])
            diff = max_abs_diff(pool_fi[wi], pool_tri[wi])
            logger.info(
                "  batch %d wi=%d: state cos_sim=%.6f max_abs_diff=%.6f",
                i,
                wi,
                sim,
                diff,
            )
            self.assertGreater(sim, 0.999, f"state cos_sim too low (batch {i}): {sim}")

    def test_B1(self):
        self._run(B=1, past_seq_lengths=[200])

    def test_B4(self):
        self._run(B=4, past_seq_lengths=[100, 300, 600, 1000])

    def test_B8_block_boundary(self):
        # past_seq_len at block boundary tests boundary handling
        self._run(
            B=8,
            past_seq_lengths=[64, 128, 192, 256, 320, 384, 448, 512],
            seq_size_per_block=64,
        )


if __name__ == "__main__":
    unittest.main()

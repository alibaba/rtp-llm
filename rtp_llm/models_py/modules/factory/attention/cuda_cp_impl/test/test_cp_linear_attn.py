"""
Unit tests for CP linear attention (GatedDeltaNet) per-layer all-gather path.

Tests:
  1. Index math: cp_local_extract_indices correctly maps zigzag positions
  2. Full forward: single-rank mock verifies CP output matches non-CP reference
"""

import contextlib
import logging
import math
import unittest
from typing import List
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    build_padding_mask,
    build_restore_indices,
    compute_rank_positions,
    zigzag_positions_for_rank,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class _AttnInputsWrapper:
    """Thin wrapper to override readonly pybind11 attributes for testing."""

    def __init__(self, wrapped, overrides: dict):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        try:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)
        except AttributeError:
            object.__getattribute__(self, "_overrides")[name] = value


def _add_device_tensors(inputs, device: torch.device):
    """Wrap PyAttentionInputs with device tensors that C++ normally creates."""
    return _AttnInputsWrapper(
        inputs,
        {
            "prefix_lengths_d": inputs.prefix_lengths.to(device),
            "input_lengths_d": inputs.input_lengths.to(device),
        },
    )


class TestCPLinearAttnIndexMath(unittest.TestCase):
    """Verify that _build_cp_linear_attn_metadata produces correct extract indices."""

    def _build_indices(
        self,
        sequence_lengths: List[int],
        cp_size: int,
        cp_rank: int,
        device: torch.device,
    ):
        """Reproduce the index construction from Qwen3NextModel._build_cp_linear_attn_metadata."""
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        restore_indices = build_restore_indices(cp_chunk_lengths, cp_size).to(device)
        padding_mask = build_padding_mask(cp_chunk_lengths, cp_size).to(device)
        unpad_restore = restore_indices[padding_mask == 1]

        total_ag = padding_mask.shape[0]
        local_chunk_total = total_ag // cp_size
        local_start = cp_rank * local_chunk_total
        local_end = local_start + local_chunk_total

        inv_restore = torch.empty(total_ag, dtype=torch.long, device=device)
        inv_restore.fill_(-1)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=device
        )

        local_mask = padding_mask[local_start:local_end]
        local_ag_positions = torch.arange(local_start, local_end, device=device)[
            local_mask == 1
        ]
        return inv_restore[local_ag_positions]

    def test_single_seq_cp2_rank0(self):
        device = torch.device("cpu")
        seq_lengths = [16]
        cp_size, cp_rank = 2, 0
        idx = self._build_indices(seq_lengths, cp_size, cp_rank, device)
        expected = zigzag_positions_for_rank(16, cp_size, cp_rank)
        self.assertEqual(idx.tolist(), expected)

    def test_single_seq_cp2_rank1(self):
        device = torch.device("cpu")
        seq_lengths = [16]
        cp_size, cp_rank = 2, 1
        idx = self._build_indices(seq_lengths, cp_size, cp_rank, device)
        expected = zigzag_positions_for_rank(16, cp_size, cp_rank)
        self.assertEqual(idx.tolist(), expected)

    def test_single_seq_cp4(self):
        device = torch.device("cpu")
        for rank in range(4):
            idx = self._build_indices([32], 4, rank, device)
            expected = zigzag_positions_for_rank(32, 4, rank)
            self.assertEqual(idx.tolist(), expected, f"rank={rank}")

    def test_multi_batch_cp2(self):
        device = torch.device("cpu")
        seq_lengths = [8, 16]
        cp_size, cp_rank = 2, 0
        idx = self._build_indices(seq_lengths, cp_size, cp_rank, device)
        expected = []
        offset = 0
        for sl in seq_lengths:
            positions = zigzag_positions_for_rank(sl, cp_size, cp_rank)
            expected.extend([p + offset for p in positions])
            offset += sl
        self.assertEqual(idx.tolist(), expected)

    def test_roundtrip_all_ranks_cover_all_tokens(self):
        """All ranks together should cover every token exactly once."""
        device = torch.device("cpu")
        seq_lengths = [16, 32]
        cp_size = 2
        all_indices = []
        for rank in range(cp_size):
            idx = self._build_indices(seq_lengths, cp_size, rank, device)
            all_indices.extend(idx.tolist())
        total = sum(seq_lengths)
        self.assertEqual(sorted(all_indices), list(range(total)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCPLinearAttnForward(unittest.TestCase):
    """Verify that CP GatedDeltaNet forward matches non-CP reference on a single GPU."""

    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _run_cp_vs_nocp(
        self,
        sequence_lengths: List[int],
        cp_size: int = 2,
        cp_rank: int = 0,
        num_k_heads: int = 4,
        num_v_heads: int = 4,
        head_k_dim: int = 64,
        head_v_dim: int = 64,
        hidden_size: int = 256,
        conv_kernel_dim: int = 4,
    ):
        """Test that CP linear attn forward matches non-CP on the same data."""
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextMetadata,
        )
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
        from rtp_llm.ops.compute_ops import PyAttentionInputs, PyContextParallelParams

        assert all(sl % (cp_size * 2) == 0 for sl in sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        total_tokens = sum(sequence_lengths)
        batch_size = len(sequence_lengths)

        linear_cfg = LinearAttentionConfig()
        linear_cfg.linear_num_key_heads = num_k_heads
        linear_cfg.linear_num_value_heads = num_v_heads
        linear_cfg.linear_key_head_dim = head_k_dim
        linear_cfg.linear_value_head_dim = head_v_dim
        linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
        linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
        linear_cfg.conv_state_dtype = DataType.TYPE_BF16

        par_cfg = ParallelismConfig()
        par_cfg.tp_size = 1
        par_cfg.tp_rank = 0

        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        z_dim = head_v_dim * num_v_heads
        qkvz_dim = qkv_dim + z_dim
        ba_dim = num_v_heads * 2

        torch.manual_seed(123)
        conv_w = torch.randn(
            qkv_dim, 1, conv_kernel_dim, device=self.device, dtype=torch.bfloat16
        )
        dt_b = torch.randn(num_v_heads, device=self.device, dtype=torch.bfloat16)
        alog = torch.randn(num_v_heads, device=self.device, dtype=torch.bfloat16)
        norm_w = torch.randn(head_v_dim, device=self.device, dtype=torch.bfloat16)

        from rtp_llm.utils.model_weight import W

        qkvz_w = torch.randn(
            hidden_size, qkvz_dim, device=self.device, dtype=torch.bfloat16
        )
        ba_w = torch.randn(
            hidden_size, ba_dim, device=self.device, dtype=torch.bfloat16
        )
        out_w = torch.randn(
            num_v_heads * head_v_dim,
            hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )

        weights = {
            W.linear_attn_conv1d_w: conv_w,
            W.linear_attn_dt_b: dt_b,
            W.linear_attn_alog: alog,
            W.linear_attn_norm_w: norm_w,
            W.linear_attn_qkvz_w: qkvz_w,
            W.linear_attn_qkvz_s: None,
            W.linear_attn_ba_w: ba_w,
            W.linear_attn_out_w: out_w,
            W.linear_attn_out_s: None,
        }

        module = Qwen3NextGatedDeltaNet(
            linear_cfg, par_cfg, weights, layernorm_eps=1e-6
        ).to(self.device)

        full_hidden = torch.randn(
            total_tokens, hidden_size, device=self.device, dtype=torch.bfloat16
        )

        # --- Non-CP reference ---
        full_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i, sl in enumerate(sequence_lengths):
            full_cu[i + 1] = full_cu[i] + sl

        nocp_inputs = PyAttentionInputs()
        nocp_inputs.is_prefill = True
        nocp_inputs.cu_seqlens = full_cu
        nocp_inputs.input_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        )
        nocp_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        )
        nocp_inputs.context_parallel_info = None
        nocp_inputs = _add_device_tensors(nocp_inputs, self.device)

        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu, device=self.device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        with torch.no_grad():
            ref_output = module(full_hidden, None, None, nocp_inputs, nocp_meta)

        # --- CP path (mocked all_gather) ---
        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        rank_positions = all_rank_pos[cp_rank]
        rank_idx = torch.tensor(rank_positions, device=self.device)
        local_hidden = full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block=16,
            device=self.device,
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, self.device)

        all_rank_packed: List[torch.Tensor] = []
        with torch.no_grad():
            for r in range(cp_size):
                r_pos = torch.tensor(all_rank_pos[r], device=self.device)
                r_hidden = full_hidden[r_pos]
                r_qkvz = module.in_proj_qkvz(r_hidden)
                r_ba = module.in_proj_ba(r_hidden)
                r_mixed_qkv, r_z, r_b, r_a = module.fix_query_key_value_ordering(
                    r_qkvz, r_ba
                )
                all_rank_packed.append(torch.cat([r_mixed_qkv, r_b, r_a], dim=-1))

        cp_info = cp_attn_inputs.context_parallel_info
        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_restore = restore_indices[padding_mask == 1]

        total_ag = padding_mask.shape[0]
        local_chunk_total = total_ag // cp_size
        local_start = cp_rank * local_chunk_total
        local_end = local_start + local_chunk_total

        inv_restore = torch.empty(total_ag, dtype=torch.long, device=self.device)
        inv_restore.fill_(-1)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=self.device
        )
        local_inv = inv_restore[local_start:local_end]
        cp_local_valid_mask = local_inv >= 0
        cp_local_extract_idx = local_inv[cp_local_valid_mask]

        actual_lengths = torch.tensor(sequence_lengths, dtype=torch.int32)
        full_cu_from_actual = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        full_cu_from_actual[1:] = torch.tensor(
            sequence_lengths, device=self.device
        ).cumsum(0)

        full_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu_from_actual, device=self.device
        )

        cp_meta = Qwen3NextMetadata(
            full_prefill_conv1d_meta=full_conv_meta,
            full_prefill_cu_seqlens=full_cu_from_actual,
            cp_restore_indices=restore_indices,
            cp_local_extract_indices=cp_local_extract_idx,
            cp_local_valid_mask=cp_local_valid_mask,
        )

        def mock_ag(tensor, group=None):
            return torch.cat(all_rank_packed, dim=0)

        AG_MODULE = "rtp_llm.models_py.model_desc.qwen3_next"
        with patch(f"{AG_MODULE}.all_gather", side_effect=mock_ag):
            with torch.no_grad():
                cp_output = module(local_hidden, None, None, cp_attn_inputs, cp_meta)

        ref_local = ref_output[rank_idx]
        diff = (cp_output.float() - ref_local.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        logging.info(f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        self.assertTrue(
            torch.allclose(cp_output.float(), ref_local.float(), rtol=1e-2, atol=1e-2),
            f"CP vs non-CP mismatch: max_diff={max_diff}, mean_diff={mean_diff}",
        )

    def test_single_seq_cp2(self):
        self._run_cp_vs_nocp(sequence_lengths=[32], cp_size=2, cp_rank=0)

    def test_single_seq_cp2_rank1(self):
        self._run_cp_vs_nocp(sequence_lengths=[32], cp_size=2, cp_rank=1)

    def test_multi_batch_cp2(self):
        self._run_cp_vs_nocp(sequence_lengths=[16, 32], cp_size=2, cp_rank=0)


if __name__ == "__main__":
    unittest.main()

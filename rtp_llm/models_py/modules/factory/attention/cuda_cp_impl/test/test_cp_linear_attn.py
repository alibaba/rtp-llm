"""
Unit tests for CP linear attention (GatedDeltaNet) fallback and halo paths.

Tests:
  1. Index math: cp_local_extract_indices correctly maps zigzag positions
  2. Full forward: single-rank mock verifies CP output matches non-CP reference
  3. Cache: zigzag halo conv and every GDN block boundary match the full sequence
"""

import contextlib
import logging
import math
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.linear_attn_utils import (
    ZigzagCPPlan,
    get_segment_valid_lengths,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    build_padding_mask,
    build_restore_indices,
    compute_rank_positions,
    zigzag_positions_for_rank,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
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
        padded_sequence_lengths: List[int] | None = None,
    ):
        """Reproduce the index construction from Qwen3NextModel._build_cp_linear_attn_metadata."""
        if padded_sequence_lengths is None:
            padded_sequence_lengths = sequence_lengths
        cp_chunk_lengths = [sl // cp_size for sl in padded_sequence_lengths]
        padding_lengths = [
            padded - actual
            for padded, actual in zip(padded_sequence_lengths, sequence_lengths)
        ]
        restore_indices = build_restore_indices(cp_chunk_lengths, cp_size).to(device)
        padding_mask = build_padding_mask(
            cp_chunk_lengths, cp_size, padding_lengths
        ).to(device)
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

        local_inv = inv_restore[local_start:local_end]
        return local_inv[local_inv >= 0]

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

    def test_non_aligned_sequence_uses_only_real_tokens(self):
        actual_length = 257
        padded_length = 512
        all_indices = []
        for rank in range(2):
            idx = self._build_indices(
                [actual_length],
                cp_size=2,
                cp_rank=rank,
                device=torch.device("cpu"),
                padded_sequence_lengths=[padded_length],
            )
            expected = [
                position
                for position in zigzag_positions_for_rank(padded_length, 2, rank)
                if position < actual_length
            ]
            valid_indices = idx.tolist()
            self.assertEqual(valid_indices, expected)
            all_indices.extend(valid_indices)
        self.assertEqual(sorted(all_indices), list(range(actual_length)))

    def test_cp2_relay_eligibility_reports_invalid_internal_prefix(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        attention_inputs = SimpleNamespace(
            input_lengths=torch.tensor([256], dtype=torch.int32),
            is_cuda_graph=False,
            prefix_lengths=torch.tensor([64], dtype=torch.int32),
            kv_cache_kernel_block_id_host=torch.tensor([[1]], dtype=torch.int32),
        )

        reason = Qwen3NextGatedDeltaNet._get_linear_cp_relay_fallback_reason(
            attention_inputs,
            kv_cache_tensor=torch.empty(1, 1),
            seq_size_per_block=64,
        )
        self.assertIsNone(reason)

        attention_inputs.prefix_lengths = torch.tensor([65], dtype=torch.int32)
        reason = Qwen3NextGatedDeltaNet._get_linear_cp_relay_fallback_reason(
            attention_inputs,
            kv_cache_tensor=torch.empty(1, 1),
            seq_size_per_block=64,
        )
        self.assertEqual(reason, "unaligned_internal_prefix")
        with self.assertRaisesRegex(RuntimeError, "cannot safely reconstruct"):
            Qwen3NextGatedDeltaNet._raise_for_invalid_cp_state(reason, 65)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCPLinearAttnForward(unittest.TestCase):
    """Verify that CP GatedDeltaNet forward matches non-CP reference on a single GPU."""

    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def test_cp2_gdn_relay_and_block_states_match_full_sequence(self):
        """Aligned CP2 segments and block states must match a full GDN call."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule

        total_tokens = 256
        num_k_heads = 2
        num_v_heads = 4
        head_dim = 64

        query = torch.randn(
            1,
            total_tokens,
            num_k_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn(
            1,
            total_tokens,
            num_v_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        g = -torch.rand(
            1,
            total_tokens,
            num_v_heads,
            device=self.device,
            dtype=torch.float32,
        )
        beta = torch.rand(
            1,
            total_tokens,
            num_v_heads,
            device=self.device,
            dtype=torch.bfloat16,
        )
        initial_state = torch.randn(
            1,
            num_v_heads,
            head_dim,
            head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        full_cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=self.device)

        with torch.no_grad():
            expected, expected_chunks, expected_final = chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=full_cu,
                use_qk_l2norm_in_kernel=True,
            )

            state = initial_state.clone()
            block_ends = torch.arange(
                64,
                total_tokens + 1,
                64,
                dtype=torch.long,
                device=self.device,
            )
            rank_block_states = [
                torch.zeros(
                    block_ends.shape[0],
                    num_v_heads,
                    head_dim,
                    head_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for _ in range(2)
            ]
            relayed_outputs = []
            # CP2 ownership order is rank0-front, rank1-middle, rank0-back.
            for owner, start, end in [(0, 0, 64), (1, 64, 192), (0, 192, 256)]:
                segment_cu = torch.tensor(
                    [0, end - start], dtype=torch.int32, device=self.device
                )
                segment_out, segment_chunks, state = chunk_gated_delta_rule(
                    query[:, start:end],
                    key[:, start:end],
                    value[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=segment_cu,
                    use_qk_l2norm_in_kernel=True,
                )
                relayed_outputs.append(segment_out)
                Qwen3NextGatedDeltaNet._record_linear_cp_segment_ssm_states(
                    rank_block_states[owner],
                    block_ends,
                    start,
                    end,
                    segment_chunks,
                    state,
                )

        actual = torch.cat(relayed_outputs, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(state, expected_final, rtol=1e-3, atol=1e-3)

        expected_block_states = torch.cat([expected_chunks[0, 1:], expected_final]).to(
            torch.bfloat16
        )
        actual_block_states = rank_block_states[0] + rank_block_states[1]
        torch.testing.assert_close(
            actual_block_states, expected_block_states, rtol=1e-2, atol=1e-2
        )

    def test_cp2_padded_gdn_relay_matches_unpadded_sequence(self):
        """Padding must not affect outputs, final state, or cache block states."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule

        actual_tokens = 257
        segment_tokens = 128
        num_k_heads = 2
        num_v_heads = 4
        head_dim = 64

        query = torch.randn(
            1,
            actual_tokens,
            num_k_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn(
            1,
            actual_tokens,
            num_v_heads,
            head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        g = -torch.rand(
            1,
            actual_tokens,
            num_v_heads,
            device=self.device,
            dtype=torch.float32,
        )
        beta = torch.rand(
            1,
            actual_tokens,
            num_v_heads,
            device=self.device,
            dtype=torch.bfloat16,
        )
        initial_state = torch.randn(
            1,
            num_v_heads,
            head_dim,
            head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        full_cu = torch.tensor(
            [0, actual_tokens], dtype=torch.int32, device=self.device
        )

        with torch.no_grad():
            expected, expected_chunks, expected_final = chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=full_cu,
                use_qk_l2norm_in_kernel=True,
            )

            state = initial_state.clone()
            block_ends = torch.cat(
                [
                    torch.arange(
                        64,
                        actual_tokens,
                        64,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    torch.tensor([actual_tokens], dtype=torch.long, device=self.device),
                ]
            )
            rank_block_states = [
                torch.zeros(
                    block_ends.shape[0],
                    num_v_heads,
                    head_dim,
                    head_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for _ in range(2)
            ]
            relayed_outputs = []
            valid_lengths = get_segment_valid_lengths(
                actual_tokens, segment_tokens, cp_size=2
            )
            schedule = [
                (0, 0, valid_lengths[0]),
                (1, segment_tokens, valid_lengths[1] + valid_lengths[2]),
                (0, 3 * segment_tokens, valid_lengths[3]),
            ]
            for owner, start, length in schedule:
                if length == 0:
                    continue
                end = start + length
                segment_cu = torch.tensor(
                    [0, length], dtype=torch.int32, device=self.device
                )
                segment_out, segment_chunks, state = chunk_gated_delta_rule(
                    query[:, start:end],
                    key[:, start:end],
                    value[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=segment_cu,
                    use_qk_l2norm_in_kernel=True,
                )
                relayed_outputs.append(segment_out)
                Qwen3NextGatedDeltaNet._record_linear_cp_segment_ssm_states(
                    rank_block_states[owner],
                    block_ends,
                    start,
                    end,
                    segment_chunks,
                    state,
                )

        actual = torch.cat(relayed_outputs, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(state, expected_final, rtol=1e-3, atol=1e-3)

        expected_block_states = torch.cat([expected_chunks[0, 1:], expected_final]).to(
            torch.bfloat16
        )
        actual_block_states = rank_block_states[0] + rank_block_states[1]
        torch.testing.assert_close(
            actual_block_states, expected_block_states, rtol=1e-2, atol=1e-2
        )

    def test_cp2_prefix_cache_blocks_use_absolute_positions(self):
        """New states must use cache slots after the reused prefix blocks."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        attention_inputs = build_cp_attn_inputs(
            sequence_lengths=[384],
            cp_chunk_lengths=[128],
            cp_size=2,
            tokens_per_block=64,
            prefix_lengths=[128],
            device=self.device,
        )
        block_ids = torch.tensor(
            [[11, 12, 13, 14, 15, 16]],
            dtype=torch.int32,
            device=self.device,
        )
        attention_inputs = _AttnInputsWrapper(
            attention_inputs,
            {"kv_cache_kernel_block_id_device": block_ids},
        )

        block_ends, selected_ids = Qwen3NextGatedDeltaNet._get_linear_cp_cache_blocks(
            attention_inputs, seq_size_per_block=64
        )

        self.assertEqual(block_ends.tolist(), [64, 128, 192, 256])
        self.assertEqual(selected_ids.tolist(), [13, 14, 15, 16])

    def test_cp2_short_prefill_cache_uses_partial_final_block(self):
        """A sub-block CP prefill must only write its partial final state."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        attention_inputs = build_cp_attn_inputs(
            sequence_lengths=[1],
            cp_chunk_lengths=[64],
            cp_size=2,
            tokens_per_block=64,
            device=self.device,
        )
        block_ids = torch.tensor([[11]], dtype=torch.int32, device=self.device)
        attention_inputs = _AttnInputsWrapper(
            attention_inputs,
            {"kv_cache_kernel_block_id_device": block_ids},
        )

        block_ends, selected_ids = Qwen3NextGatedDeltaNet._get_linear_cp_cache_blocks(
            attention_inputs, seq_size_per_block=64
        )

        self.assertEqual(block_ends.tolist(), [1])
        self.assertEqual(selected_ids.tolist(), [11])

    def test_cp2_cache_state_sync_uses_collective_result(self):
        """Cache writes must use collectives that return a new output tensor."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        local_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.bfloat16,
            device=self.device,
        )
        original_local_states = local_states.clone()
        reduced_states = torch.tensor(
            [[11.0, 12.0], [13.0, 14.0]],
            dtype=torch.bfloat16,
            device=self.device,
        )
        cache_states = torch.zeros(4, 2, dtype=torch.bfloat16, device=self.device)
        block_ids = torch.tensor([1, 3], dtype=torch.long, device=self.device)

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.all_reduce",
            return_value=reduced_states,
        ):
            Qwen3NextGatedDeltaNet._sync_linear_cp_cache_states(
                local_states, cache_states, block_ids
            )

        torch.testing.assert_close(cache_states[block_ids], reduced_states)
        torch.testing.assert_close(local_states, original_local_states)

    def test_fp32_prefix_ssm_state_uses_detached_relay_buffer(self):
        """In-place relay communication must not overwrite a cached prefix state."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        cache_storage = torch.randn(
            3 * 2 * 4 * 4, dtype=torch.float32, device=self.device
        )
        cache_states = cache_storage.view(3, 2, 4, 4)
        expected_cache = cache_states.clone()

        relay_state = Qwen3NextGatedDeltaNet._copy_linear_cp_prefix_ssm_state(
            cache_states, prefix_block_id=1
        )
        self.assertEqual(relay_state.dtype, torch.float32)
        self.assertNotEqual(
            relay_state.untyped_storage().data_ptr(),
            cache_states.untyped_storage().data_ptr(),
        )

        relay_state.fill_(0)
        torch.testing.assert_close(cache_states, expected_cache)

    def test_padded_conv_cache_uses_predecessor_halo(self):
        """A short real segment must not read its cache tail from padding."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        local_block_states = torch.zeros(
            2, 3, 1, dtype=torch.bfloat16, device=self.device
        )
        block_ends = torch.tensor([1, 2], dtype=torch.long, device=self.device)
        predecessor_halo = torch.tensor(
            [[10.0], [11.0], [12.0]],
            dtype=torch.bfloat16,
            device=self.device,
        )
        padded_segment = torch.tensor(
            [[20.0], [21.0], [99.0], [99.0]],
            dtype=torch.bfloat16,
            device=self.device,
        )

        Qwen3NextGatedDeltaNet._record_linear_cp_segment_conv_states(
            local_block_states,
            block_ends,
            segment_start=0,
            segment_valid_length=2,
            segment_with_halo=torch.cat([predecessor_halo, padded_segment]),
        )

        expected = torch.tensor(
            [[[11.0], [12.0], [20.0]], [[12.0], [20.0], [21.0]]],
            dtype=torch.bfloat16,
            device=self.device,
        )
        torch.testing.assert_close(local_block_states, expected)

    def test_relay_follows_generic_zigzag_schedule(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        module_path = "rtp_llm.models_py.model_desc.qwen3_next"
        local_qkv = torch.arange(4, dtype=torch.bfloat16, device=self.device).unsqueeze(
            1
        )
        z = torch.zeros_like(local_qkv)
        g = torch.zeros(1, 4, 1, dtype=torch.float32, device=self.device)
        beta = torch.zeros(1, 4, 1, dtype=torch.bfloat16, device=self.device)

        for cp_size in (2, 3, 4, 5, 8):
            relay_steps = ZigzagCPPlan(cp_size, 0).relay_steps
            for rank in range(cp_size):
                segment_calls = []
                broadcast_sources = []

                class _RelayHarness:
                    _linear_cp_relay_logged = True

                harness = _RelayHarness()
                harness.parallelism_config = SimpleNamespace(tp_rank=rank)
                harness.prefill_gdn = SimpleNamespace(
                    alog=torch.zeros(1, dtype=torch.bfloat16, device=self.device),
                    dt_bias=torch.zeros(1, dtype=torch.bfloat16, device=self.device),
                    local_num_v_heads=1,
                    head_k_dim=1,
                    head_v_dim=1,
                )
                harness.head_v_dim = 1
                harness.local_num_v_heads = 1
                harness.norm = lambda attn_out, _: attn_out
                harness.out_proj = lambda attn_out: attn_out

                def run_segment(mixed_qkv, segment_g, segment_beta, initial_state):
                    segment_calls.append(
                        (mixed_qkv.clone(), float(initial_state.item()))
                    )
                    segment_out = torch.zeros(
                        mixed_qkv.shape[0],
                        1,
                        1,
                        dtype=mixed_qkv.dtype,
                        device=mixed_qkv.device,
                    )
                    chunk_states = torch.empty(
                        1, 1, 1, 1, 1, dtype=torch.float32, device=mixed_qkv.device
                    )
                    return segment_out, chunk_states, initial_state + 1

                def relay_broadcast(state, src, group):
                    broadcast_sources.append(src)
                    if rank != src:
                        state.fill_(len(broadcast_sources))

                harness._run_linear_cp_gdn_segment = run_segment
                with patch(
                    f"{module_path}.fused_gdn_gating", return_value=(g, beta)
                ), patch(f"{module_path}.broadcast", side_effect=relay_broadcast):
                    output = Qwen3NextGatedDeltaNet._forward_linear_cp_relay(
                        harness,
                        local_qkv,
                        z,
                        torch.empty(4, 1, device=self.device),
                        torch.empty(4, 1, device=self.device),
                        prefix_ssm_state=None,
                        kv_cache_tensor=None,
                        cache_block_ends=None,
                        cache_block_ids=None,
                        cp_plan=ZigzagCPPlan(cp_size, rank),
                        segment_valid_lengths=(2,) * (2 * cp_size),
                    )

                self.assertEqual(
                    broadcast_sources,
                    [step.owner_rank for step in relay_steps[:-1]],
                )
                self.assertEqual(output.shape, (4, 1))
                owned_steps = [
                    (step_index, step)
                    for step_index, step in enumerate(relay_steps)
                    if step.owner_rank == rank
                ]
                self.assertEqual(
                    [call[1] for call in segment_calls],
                    [float(step_index) for step_index, _ in owned_steps],
                )
                for call, (_, step) in zip(segment_calls, owned_steps):
                    local_start = step.first_local_segment * 2
                    local_end = local_start + step.segment_count * 2
                    torch.testing.assert_close(
                        call[0], local_qkv[local_start:local_end]
                    )

    def test_cp2_relay_trims_padded_segments(self):
        """The production relay must never advance state through padding tokens."""
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet

        module_path = "rtp_llm.models_py.model_desc.qwen3_next"
        segment_valid_lengths = (128, 128, 1, 0)

        for rank, expected_call_length in ((0, 128), (1, 129)):

            class _RelayHarness:
                _linear_cp_relay_logged = True

            harness = _RelayHarness()
            harness.parallelism_config = SimpleNamespace(tp_rank=rank)
            harness.prefill_gdn = SimpleNamespace(
                alog=torch.zeros(1, dtype=torch.bfloat16, device=self.device),
                dt_bias=torch.zeros(1, dtype=torch.bfloat16, device=self.device),
                local_num_v_heads=1,
                head_k_dim=1,
                head_v_dim=1,
            )
            harness.head_v_dim = 1
            harness.local_num_v_heads = 1
            harness.norm = lambda attn_out, _: attn_out
            harness.out_proj = lambda attn_out: attn_out

            segment_call_lengths = []

            def run_segment(mixed_qkv, segment_g, segment_beta, initial_state):
                segment_call_lengths.append(mixed_qkv.shape[0])
                segment_out = torch.ones(
                    mixed_qkv.shape[0],
                    1,
                    1,
                    dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device,
                )
                chunk_states = torch.empty(
                    1, 1, 1, 1, 1, dtype=torch.float32, device=mixed_qkv.device
                )
                return segment_out, chunk_states, initial_state

            harness._run_linear_cp_gdn_segment = run_segment
            local_qkv = torch.randn(256, 1, dtype=torch.bfloat16, device=self.device)
            z = torch.zeros_like(local_qkv)
            g = torch.zeros(1, 256, 1, dtype=torch.float32, device=self.device)
            beta = torch.zeros(1, 256, 1, dtype=torch.bfloat16, device=self.device)

            with patch(
                f"{module_path}.fused_gdn_gating", return_value=(g, beta)
            ), patch(f"{module_path}.broadcast"):
                output = Qwen3NextGatedDeltaNet._forward_linear_cp_relay(
                    harness,
                    local_qkv,
                    z,
                    torch.empty(256, 1, device=self.device),
                    torch.empty(256, 1, device=self.device),
                    prefix_ssm_state=None,
                    kv_cache_tensor=None,
                    cache_block_ends=None,
                    cache_block_ids=None,
                    cp_plan=ZigzagCPPlan(2, rank),
                    segment_valid_lengths=segment_valid_lengths,
                )

            self.assertEqual(segment_call_lengths, [expected_call_length])
            local_valid_tokens = 128 if rank == 0 else 129
            self.assertEqual(
                torch.count_nonzero(output[:local_valid_tokens]).item(),
                local_valid_tokens,
            )
            self.assertEqual(torch.count_nonzero(output[local_valid_tokens:]).item(), 0)

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
        halo_conv_only: bool = False,
        prefix_tokens: int = 0,
    ):
        """Test that CP linear attn forward matches non-CP on the same data."""
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNet,
            Qwen3NextMetadata,
            fused_gdn_gating,
        )
        from rtp_llm.models_py.triton_kernels.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )
        from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
        from rtp_llm.ops.compute_ops import PyAttentionInputs, PyContextParallelParams

        if halo_conv_only:
            sequence_alignment = cp_size * 2 * 64
            padded_sequence_lengths = [
                math.ceil(sl / sequence_alignment) * sequence_alignment
                for sl in sequence_lengths
            ]
        else:
            assert all(sl % (cp_size * 2) == 0 for sl in sequence_lengths)
            padded_sequence_lengths = sequence_lengths
        cp_chunk_lengths = [sl // cp_size for sl in padded_sequence_lengths]
        total_tokens = sum(sequence_lengths)
        batch_size = len(sequence_lengths)
        assert prefix_tokens == 0 or batch_size == 1

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
        if padded_sequence_lengths != sequence_lengths:
            assert batch_size == 1
            padding_hidden = torch.randn(
                padded_sequence_lengths[0] - sequence_lengths[0],
                hidden_size,
                device=self.device,
                dtype=torch.bfloat16,
            )
            cp_full_hidden = torch.cat([full_hidden, padding_hidden])
        else:
            cp_full_hidden = full_hidden

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
        all_rank_pos = compute_rank_positions(padded_sequence_lengths, cp_size)
        rank_positions = all_rank_pos[cp_rank]
        rank_idx = torch.tensor(rank_positions, device=self.device)
        local_hidden = cp_full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            [sl + prefix_tokens for sl in sequence_lengths],
            cp_chunk_lengths,
            cp_size,
            tokens_per_block=16,
            prefix_lengths=[prefix_tokens] * batch_size,
            device=self.device,
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, self.device)

        all_rank_packed: List[torch.Tensor] = []
        all_rank_tails: List[torch.Tensor] = []
        with torch.no_grad():
            for r in range(cp_size):
                r_pos = torch.tensor(all_rank_pos[r], device=self.device)
                r_hidden = cp_full_hidden[r_pos]
                r_qkvz = module.in_proj_qkvz(r_hidden)
                r_ba = module.in_proj_ba(r_hidden)
                r_mixed_qkv, r_z, r_b, r_a = module.fix_query_key_value_ordering(
                    r_qkvz, r_ba
                )
                all_rank_packed.append(torch.cat([r_mixed_qkv, r_b, r_a], dim=-1))
                local_segment_tokens = r_mixed_qkv.shape[0] // 2
                all_rank_tails.append(
                    r_mixed_qkv.reshape(2, local_segment_tokens, -1)[
                        :, -(conv_kernel_dim - 1) :, :
                    ].contiguous()
                )

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

        local_conv_meta = None
        local_conv_cu = None
        local_conv_prefix_lengths = None
        if batch_size == 1:
            state_len = conv_kernel_dim - 1
            local_segment_tokens = cp_chunk_lengths[0] // 2
            haloed_segment_tokens = local_segment_tokens + state_len
            local_conv_cu = torch.tensor(
                [0, haloed_segment_tokens, 2 * haloed_segment_tokens],
                dtype=torch.int32,
                device=self.device,
            )
            local_conv_prefix_lengths = torch.zeros(
                2, dtype=torch.int32, device=self.device
            )
            local_conv_meta = prepare_causal_conv1d_metadata(
                query_start_loc=local_conv_cu, device=self.device
            )

        cp_meta = Qwen3NextMetadata(
            full_prefill_conv1d_meta=full_conv_meta,
            full_prefill_cu_seqlens=full_cu_from_actual,
            cp_local_conv1d_meta=local_conv_meta,
            cp_local_conv_cu_seqlens=local_conv_cu,
            cp_local_conv_prefix_lengths=local_conv_prefix_lengths,
            is_cp_linear_attn=True,
            cp_local_extract_indices=cp_local_extract_idx,
            cp_local_valid_mask=cp_local_valid_mask,
        )

        def mock_ag(tensor, group=None):
            if tensor.shape[-1] == qkv_dim:
                return torch.cat([tail.flatten(0, 1) for tail in all_rank_tails], dim=0)
            return torch.cat(all_rank_packed, dim=0)

        module.parallelism_config.tp_size = cp_size
        module.parallelism_config.tp_rank = cp_rank
        module.parallelism_config.dp_size = 1

        AG_MODULE = "rtp_llm.models_py.model_desc.qwen3_next"
        if halo_conv_only:
            full_packed = torch.cat(all_rank_packed, dim=0)[unpad_restore]
            full_mixed_qkv = full_packed[:, :qkv_dim].contiguous()
            with torch.no_grad():
                prefix_conv_state = None
                expected_conv_input = full_mixed_qkv
                expected_conv_cu = full_cu_from_actual
                if prefix_tokens > 0:
                    prefix_hidden = torch.randn(
                        prefix_tokens,
                        hidden_size,
                        device=self.device,
                        dtype=torch.bfloat16,
                    )
                    prefix_qkvz = module.in_proj_qkvz(prefix_hidden)
                    prefix_ba = module.in_proj_ba(prefix_hidden)
                    prefix_mixed_qkv, _, _, _ = module.fix_query_key_value_ordering(
                        prefix_qkvz, prefix_ba
                    )
                    prefix_conv_state = prefix_mixed_qkv[
                        -(conv_kernel_dim - 1) :
                    ].contiguous()
                    expected_conv_input = torch.cat(
                        [prefix_mixed_qkv, full_mixed_qkv], dim=0
                    )
                    expected_conv_cu = torch.tensor(
                        [0, prefix_tokens + total_tokens],
                        dtype=torch.int32,
                        device=self.device,
                    )
                expected_conv_meta = prepare_causal_conv1d_metadata(
                    query_start_loc=expected_conv_cu, device=self.device
                )
                expected_full_conv = causal_conv1d_fn(
                    x=expected_conv_input.transpose(0, 1),
                    weight=module.prefill_gdn.conv_weights,
                    bias=None,
                    conv_states=None,
                    query_start_loc=expected_conv_cu,
                    block_map=None,
                    prefix_lengths=torch.zeros(
                        batch_size, dtype=torch.int32, device=self.device
                    ),
                    seq_size_per_block=1,
                    metadata=expected_conv_meta,
                ).transpose(0, 1)
                expected_full_conv = expected_full_conv[prefix_tokens:]

                local_qkvz = module.in_proj_qkvz(local_hidden)
                local_ba = module.in_proj_ba(local_hidden)
                local_mixed_qkv, _, _, _ = module.fix_query_key_value_ordering(
                    local_qkvz, local_ba
                )
                state_len = conv_kernel_dim - 1
                cache_block_ends = torch.cat(
                    [
                        torch.arange(
                            64,
                            total_tokens,
                            64,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.tensor(
                            [total_tokens], dtype=torch.long, device=self.device
                        ),
                    ]
                )
                cache_block_ids = torch.arange(
                    1,
                    cache_block_ends.shape[0] + 1,
                    dtype=torch.long,
                    device=self.device,
                )
                tail_offsets = torch.arange(
                    -state_len, 0, dtype=torch.long, device=self.device
                )
                expected_block_states = full_mixed_qkv[
                    cache_block_ends[:, None] + tail_offsets[None, :]
                ]
                conv_states = torch.zeros(
                    cache_block_ends.shape[0] + 1,
                    state_len,
                    qkv_dim,
                    dtype=local_mixed_qkv.dtype,
                    device=self.device,
                )

                global_segment_ids = (cache_block_ends - 1) // local_segment_tokens
                owners = torch.minimum(
                    global_segment_ids, 2 * cp_size - 1 - global_segment_ids
                )

                def mock_all_reduce(tensor, group=None):
                    owned = owners == cp_rank
                    torch.testing.assert_close(
                        tensor[owned], expected_block_states[owned]
                    )
                    self.assertEqual(torch.count_nonzero(tensor[~owned]).item(), 0)
                    tensor.copy_(expected_block_states)
                    return tensor

                with patch(f"{AG_MODULE}.all_gather", side_effect=mock_ag), patch(
                    f"{AG_MODULE}.all_reduce", side_effect=mock_all_reduce
                ), patch.object(
                    type(module.prefill_gdn),
                    "_get_conv_states",
                    return_value=conv_states,
                ):
                    actual_local_conv = module._forward_linear_cp_conv(
                        local_mixed_qkv,
                        prefix_conv_state,
                        kv_cache_tensor=torch.empty(0, device=self.device),
                        cache_block_ends=cache_block_ends,
                        cache_block_ids=cache_block_ids,
                        attn_meta=cp_meta,
                        cp_plan=ZigzagCPPlan(cp_size, cp_rank),
                        segment_valid_lengths=get_segment_valid_lengths(
                            total_tokens, local_segment_tokens, cp_size
                        ),
                    )

            expected_local_conv = expected_full_conv[cp_local_extract_idx]
            torch.testing.assert_close(
                actual_local_conv[cp_local_valid_mask],
                expected_local_conv,
                rtol=1e-2,
                atol=1e-2,
            )
            torch.testing.assert_close(
                conv_states[cache_block_ids], expected_block_states
            )
            self.assertEqual(torch.count_nonzero(conv_states[0]).item(), 0)
            return

        cp_plan = None
        expected_broadcast_states = []
        if batch_size == 1:
            full_packed = torch.cat(all_rank_packed, dim=0)[unpad_restore]
            full_mixed_qkv, full_b, full_a = torch.split(
                full_packed, [qkv_dim, num_v_heads, num_v_heads], dim=-1
            )
            with torch.no_grad():
                full_conv = causal_conv1d_fn(
                    x=full_mixed_qkv.transpose(0, 1),
                    weight=module.prefill_gdn.conv_weights,
                    bias=None,
                    conv_states=None,
                    query_start_loc=full_cu_from_actual,
                    block_map=None,
                    prefix_lengths=torch.zeros(
                        batch_size, dtype=torch.int32, device=self.device
                    ),
                    seq_size_per_block=1,
                    metadata=full_conv_meta,
                ).transpose(0, 1)
                full_g, full_beta = fused_gdn_gating(
                    module.prefill_gdn.alog,
                    full_a.contiguous(),
                    full_b.contiguous(),
                    module.prefill_gdn.dt_bias,
                )

                cp_plan = ZigzagCPPlan(cp_size, cp_rank)
                segment_tokens = local_hidden.shape[0] // 2
                segment_valid_lengths = get_segment_valid_lengths(
                    total_tokens, segment_tokens, cp_size
                )
                relay_state = torch.zeros(
                    1,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
                for step_index, step in enumerate(cp_plan.relay_steps):
                    valid_tokens = step.valid_token_count(segment_valid_lengths)
                    if valid_tokens > 0:
                        global_start = step.first_global_segment * segment_tokens
                        global_end = global_start + valid_tokens
                        _, _, relay_state = module._run_linear_cp_gdn_segment(
                            full_conv[global_start:global_end],
                            full_g[:, global_start:global_end].contiguous(),
                            full_beta[:, global_start:global_end].contiguous(),
                            relay_state,
                        )
                    if step_index + 1 < len(cp_plan.relay_steps):
                        expected_broadcast_states.append(relay_state.clone())

        broadcast_index = 0

        def mock_broadcast(state, src, group=None):
            nonlocal broadcast_index
            self.assertIsNotNone(cp_plan)
            step = cp_plan.relay_steps[broadcast_index]
            self.assertEqual(src, step.owner_rank)
            expected_state = expected_broadcast_states[broadcast_index]
            if cp_rank == src:
                torch.testing.assert_close(
                    state, expected_state, rtol=1e-3, atol=1e-3
                )
            else:
                state.copy_(expected_state)
            broadcast_index += 1

        with patch(f"{AG_MODULE}.all_gather", side_effect=mock_ag), patch(
            f"{AG_MODULE}.broadcast", side_effect=mock_broadcast
        ):
            with torch.no_grad():
                cp_output = module(local_hidden, None, None, cp_attn_inputs, cp_meta)

        self.assertEqual(broadcast_index, len(expected_broadcast_states))

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
        self._run_cp_vs_nocp(sequence_lengths=[256], cp_size=2, cp_rank=0)

    def test_single_seq_cp2_rank1(self):
        self._run_cp_vs_nocp(sequence_lengths=[256], cp_size=2, cp_rank=1)

    def test_aligned_single_seq_cp2(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[256], cp_size=2, cp_rank=0, halo_conv_only=True
        )

    def test_aligned_single_seq_cp2_rank1(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[256], cp_size=2, cp_rank=1, halo_conv_only=True
        )

    def test_aligned_single_seq_cp2_with_prefix(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[256],
            cp_size=2,
            cp_rank=0,
            halo_conv_only=True,
            prefix_tokens=64,
        )

    def test_padded_single_seq_cp2(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[257], cp_size=2, cp_rank=0, halo_conv_only=True
        )

    def test_padded_single_seq_cp2_rank1(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[257], cp_size=2, cp_rank=1, halo_conv_only=True
        )

    def test_padded_single_seq_cp2_with_prefix(self):
        self._run_cp_vs_nocp(
            sequence_lengths=[257],
            cp_size=2,
            cp_rank=0,
            halo_conv_only=True,
            prefix_tokens=64,
        )

    def test_aligned_single_seq_cp4_all_ranks(self):
        for rank in range(4):
            self._run_cp_vs_nocp(
                sequence_lengths=[512],
                cp_size=4,
                cp_rank=rank,
                halo_conv_only=True,
            )

    def test_aligned_single_seq_cp8_outer_and_middle_ranks(self):
        for rank in (0, 7):
            self._run_cp_vs_nocp(
                sequence_lengths=[1024],
                cp_size=8,
                cp_rank=rank,
                halo_conv_only=True,
            )

    def test_multi_batch_cp2(self):
        self._run_cp_vs_nocp(sequence_lengths=[16, 32], cp_size=2, cp_rank=0)


if __name__ == "__main__":
    unittest.main()

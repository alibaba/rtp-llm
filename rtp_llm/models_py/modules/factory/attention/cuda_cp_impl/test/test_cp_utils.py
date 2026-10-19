"""
Unit tests for cp_utils.py — zigzag index generation functions.

Covers:
  - generate_q_indices
  - generate_full_causal_kv_indices
  - generate_nonlocal_causal_kv_indices
  - generate_half_q_indices
  - generate_half_kv_indices
  - select_prefill_position_ids

Correctness is verified against an independent reference implementation
that computes zigzag positions from first principles.
"""

import math
import unittest
from types import SimpleNamespace
from typing import List, Set, Tuple
from unittest.mock import patch

import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.modules.factory.attention import common as attention_common
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_full_causal_kv_indices,
    generate_half_kv_indices,
    generate_half_q_indices,
    generate_nonlocal_causal_kv_indices,
    generate_q_indices,
    plan_cp_attention,
    plan_prefix_paged_attention,
)
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyContextParallelParams,
    fill_mla_params,
)


class QwenCPRaggedWorkspaceTest(unittest.TestCase):
    def test_qwen_full_layer_suffix_geometry(self):
        torch.manual_seed(109)
        device = torch.device("cuda")
        q_len, heads, kv_heads, dim = 175, 16, 2, 256
        for kv_len in (175, 350, 525, 700):
            with self.subTest(kv_len=kv_len):
                workspace = torch.empty(1, dtype=torch.uint8, device=device)
                wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    workspace, kv_layout="NHD", backend="fa2"
                )
                plan_cp_attention(
                    wrapper,
                    qo_indptr=torch.tensor(
                        [0, q_len], dtype=torch.int32, device=device
                    ),
                    kv_indptr=torch.tensor(
                        [0, kv_len], dtype=torch.int32, device=device
                    ),
                    num_qo_heads=heads,
                    num_kv_heads=kv_heads,
                    head_dim_qk=dim,
                    causal=True,
                    q_data_type=torch.bfloat16,
                )
                q = torch.randn(q_len, heads, dim, device=device, dtype=torch.bfloat16)
                k = torch.randn(
                    kv_len, kv_heads, dim, device=device, dtype=torch.bfloat16
                )
                v = torch.randn_like(k)
                out = wrapper.run(q, k, v)
                keys = (
                    k.float()
                    .repeat_interleave(heads // kv_heads, dim=1)
                    .transpose(0, 1)
                )
                values = (
                    v.float()
                    .repeat_interleave(heads // kv_heads, dim=1)
                    .transpose(0, 1)
                )
                scores = q.float().transpose(0, 1) @ keys.transpose(-1, -2) / dim**0.5
                counts = torch.arange(q_len, device=device) + kv_len - q_len + 1
                invalid = (
                    torch.arange(kv_len, device=device)[None, :] >= counts[:, None]
                )
                scores.masked_fill_(invalid, float("-inf"))
                expected = (scores.softmax(-1) @ values).transpose(0, 1)
                torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
                _, lse = wrapper.run(torch.zeros_like(q), k, v, return_lse=True)
                expected_lse = counts.float().log2()[:, None].expand(-1, heads)
                torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)


class PrefixPagedAttentionTailTest(unittest.TestCase):
    def test_short_prefix_excludes_appended_suffix_and_unused_tail(self):
        # Exercise actual planning, append and attention kernels. A permuted
        # block table ensures batch offsets cannot be mistaken for page IDs.
        torch.manual_seed(108)
        device = torch.device("cuda")
        page_size, heads, dim = 16, 2, 64
        prefixes = [0, 1, 16, 17, 1300]
        suffixes = [8, 8, 8, 8, 8]
        lengths = [p + n for p, n in zip(prefixes, suffixes)]
        pages = [(n + page_size - 1) // page_size for n in lengths]
        block_ids = torch.full((len(prefixes), max(pages)), -1, dtype=torch.int32)
        permutation = torch.randperm(sum(pages), dtype=torch.int32)
        offset = 0
        for i, count in enumerate(pages):
            block_ids[i, :count] = permutation[offset : offset + count]
            offset += count
        cache = torch.full(
            (sum(pages), 2, heads, page_size, dim),
            64.0,
            device=device,
            dtype=torch.bfloat16,
        )
        prefix_kv = []
        for i, count in enumerate(prefixes):
            kv = torch.randn(count, 2, heads, dim, device=device, dtype=torch.bfloat16)
            prefix_kv.append(kv)
            for pos in range(count):
                cache[block_ids[i, pos // page_size].item(), :, :, pos % page_size] = (
                    kv[pos]
                )
        params = fill_mla_params(
            torch.tensor(prefixes, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            torch.tensor(suffixes, dtype=torch.int32),
            block_ids,
            page_size,
        )
        suffix_kv = torch.randn(
            sum(suffixes), 2, heads, dim, device=device, dtype=torch.bfloat16
        )
        append_paged_kv_cache(
            suffix_kv[:, 0].contiguous(),
            suffix_kv[:, 1].contiguous(),
            params.batch_indice_d,
            params.positions_d,
            cache,
            params.page_indice_d,
            params.decode_page_indptr_d,
            params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )
        offset = 0
        for i, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
            expected = torch.cat((prefix_kv[i], suffix_kv[offset : offset + suffix]))
            actual = torch.stack(
                [
                    cache[block_ids[i, pos // page_size].item(), :, :, pos % page_size]
                    for pos in range(prefix + suffix)
                ]
            )
            self.assertTrue(torch.equal(actual, expected))
            offset += suffix
        qo_indptr = (
            torch.arange(len(prefixes) + 1, dtype=torch.int32, device=device) * 2
        )
        q = torch.randn(
            len(prefixes) * 2, heads, dim, device=device, dtype=torch.bfloat16
        )
        shared_workspace = torch.empty(1, dtype=torch.uint8, device=device)
        other_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            shared_workspace, kv_layout="HND", backend="fa2"
        )
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            shared_workspace,
            kv_layout="HND",
            backend="fa2",
        )
        integer_workspace = wrapper._int_workspace_buffer
        plan_prefix_paged_attention(
            wrapper,
            qo_indptr,
            torch.tensor(prefixes, dtype=torch.int32),
            params,
            num_qo_heads=heads,
            num_kv_heads=heads,
            head_dim=dim,
            page_size=page_size,
            device=device,
        )
        self.assertIs(wrapper._int_workspace_buffer, integer_workspace)
        self.assertIs(other_wrapper._float_workspace_buffer, shared_workspace)
        self.assertEqual(shared_workspace.numel(), 1)
        self.assertGreater(wrapper._float_workspace_buffer.numel(), 1)
        output, lse = wrapper.run(q, cache, return_lse=True)
        self.assertTrue(torch.equal(output[:2], torch.zeros_like(output[:2])))
        # FlashInfer represents empty attention with its finite -math::inf
        # sentinel and base-2 LSE. Its contract is a neutral merge state.
        other = torch.randn_like(output[:2])
        other_lse = torch.zeros_like(lse[:2])
        merged, merged_lse = merge_state(output[:2], lse[:2], other, other_lse)
        self.assertTrue(torch.equal(merged, other))
        self.assertTrue(torch.equal(merged_lse, other_lse))
        for i, prefix in enumerate(prefixes):
            if not prefix:
                continue
            queries = q[i * 2 : (i + 1) * 2].float().transpose(0, 1)
            keys = prefix_kv[i][:, 0].float().transpose(0, 1)
            values = prefix_kv[i][:, 1].float().transpose(0, 1)
            scores = queries @ keys.transpose(-1, -2) / dim**0.5
            expected = (scores.softmax(-1) @ values).transpose(0, 1)
            torch.testing.assert_close(
                output[i * 2 : (i + 1) * 2].float(), expected, rtol=1e-2, atol=1e-2
            )
        # FA2 accumulates its denominator from BF16-rounded exponentials
        # (compute_sfm_v), so random FP32 logsumexp is not a 1e-5 oracle.
        # Zero Q makes every valid exponential exactly one: the independent
        # analytical LSE now measures the precise number of admitted tokens.
        uniform_output, uniform_lse = wrapper.run(
            torch.zeros_like(q), cache, return_lse=True
        )
        for i, prefix in enumerate(prefixes):
            if not prefix:
                continue
            expected_lse = torch.full_like(
                uniform_lse[i * 2 : (i + 1) * 2], math.log2(prefix)
            )
            torch.testing.assert_close(
                uniform_lse[i * 2 : (i + 1) * 2],
                expected_lse,
                rtol=1e-5,
                atol=1e-5,
            )
            expected_mean = prefix_kv[i][:, 1].float().mean(0).expand(2, -1, -1)
            torch.testing.assert_close(
                uniform_output[i * 2 : (i + 1) * 2].float(),
                expected_mean,
                rtol=1e-2,
                atol=1e-2,
            )


class CacheStoreInputLengthTest(unittest.TestCase):
    # Cache-store planning moved into the C++ CacheStoreWriter (main #1250);
    # python only forwards the writer + inputs pair. The CP full-length
    # override now happens in PyWrappedModel::forward before the writer runs.
    def test_writer_receives_writer_and_inputs_pair(self):
        cache_store_inputs = object()
        writer = object()
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            cache_store_writer=writer,
        )

        op = object()
        with patch.object(
            attention_common, "WriteCacheStoreOp", return_value=op
        ) as constructor:
            self.assertIs(
                attention_common.create_write_cache_store_impl(attn_inputs), op
            )

        args = constructor.call_args.args
        self.assertIs(args[0], writer)
        self.assertIs(args[1], cache_store_inputs)

    def test_no_writer_returns_none(self):
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=object(),
            cache_store_writer=None,
        )
        self.assertIsNone(attention_common.create_write_cache_store_impl(attn_inputs))


# ---------------------------------------------------------------------------
# Reference helpers (independent of the code under test)
# ---------------------------------------------------------------------------


def _zigzag_positions(full_len: int, cp_size: int, rank: int) -> Set[int]:
    """Original-sequence positions assigned to *rank* under zigzag."""
    h = full_len // cp_size // 2
    first = set(range(rank * h, (rank + 1) * h))
    second = set(range(full_len - (rank + 1) * h, full_len - rank * h))
    return first | second


def _ref_full_causal(
    cp_chunk_lengths: List[int], cp_rank: int, cp_size: int
) -> Tuple[List[int], List[int]]:
    """Reference: full causal KV range for each Q-half."""
    p0, p1 = [], []
    offset = 0
    for cl in cp_chunk_lengths:
        h = cl // 2
        p0.extend(range(offset, offset + h * (cp_rank + 1)))
        p1.extend(range(offset, offset + h * (2 * cp_size - cp_rank)))
        offset += cl * cp_size
    return p0, p1


def _ref_nonlocal_causal(
    cp_chunk_lengths: List[int], cp_rank: int, cp_size: int
) -> Tuple[List[int], List[int]]:
    """Reference: non-local causal KV range (full minus local positions)."""
    p0, p1 = [], []
    offset = 0
    for cl in cp_chunk_lengths:
        h = cl // 2
        full_len = cl * cp_size
        local = _zigzag_positions(full_len, cp_size, cp_rank)
        fp0 = set(range(0, h * (cp_rank + 1)))
        fp1 = set(range(0, h * (2 * cp_size - cp_rank)))
        p0.extend([x + offset for x in sorted(fp0 - local)])
        p1.extend([x + offset for x in sorted(fp1 - local)])
        offset += full_len
    return p0, p1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateQIndices(unittest.TestCase):

    def test_single_chunk(self):
        idx0, idx1 = generate_q_indices([8])
        self.assertEqual(idx0, [0, 1, 2, 3])
        self.assertEqual(idx1, [4, 5, 6, 7])

    def test_multi_chunk(self):
        idx0, idx1 = generate_q_indices([8, 4, 4])
        self.assertEqual(idx0, [0, 1, 2, 3, 8, 9, 12, 13])
        self.assertEqual(idx1, [4, 5, 6, 7, 10, 11, 14, 15])

    def test_no_overlap(self):
        """First-half and second-half indices must be disjoint and exhaustive."""
        for chunks in [[4], [6, 10], [2, 4, 8, 6]]:
            idx0, idx1 = generate_q_indices(chunks)
            total = sum(chunks)
            self.assertEqual(sorted(idx0 + idx1), list(range(total)))
            self.assertEqual(len(set(idx0) & set(idx1)), 0)

    def test_odd_chunk(self):
        idx0, idx1 = generate_q_indices([5])
        self.assertEqual(idx0, [0, 1, 2])
        self.assertEqual(idx1, [3, 4])


class TestGenerateFullCausalKvIndices(unittest.TestCase):
    """Tests for generate_full_causal_kv_indices (all-gather impl)."""

    def _check(self, cp_chunk_lengths, cp_rank, cp_size):
        actual = generate_full_causal_kv_indices(cp_chunk_lengths, cp_rank, cp_size)
        expected = _ref_full_causal(cp_chunk_lengths, cp_rank, cp_size)
        self.assertEqual(
            actual,
            expected,
            f"Mismatch: chunks={cp_chunk_lengths} rank={cp_rank} size={cp_size}",
        )

    def test_cp2_single_batch(self):
        for rank in range(2):
            self._check([4], rank, 2)

    def test_cp4_single_batch(self):
        for rank in range(4):
            self._check([4], rank, 4)

    def test_cp8_single_batch(self):
        for rank in range(8):
            self._check([4], rank, 8)

    def test_cp16_single_batch(self):
        for rank in range(16):
            self._check([4], rank, 16)

    def test_cp4_multi_batch(self):
        for rank in range(4):
            self._check([4, 8], rank, 4)

    def test_cp8_multi_batch(self):
        for rank in range(8):
            self._check([4, 6, 8], rank, 8)

    def test_rank0_part0_empty(self):
        """rank=0, part0 should have exactly h items (not 0)."""
        p0, _ = generate_full_causal_kv_indices([4], 0, 4)
        self.assertEqual(len(p0), 2)

    def test_counts(self):
        """Verify KV count formulas: |p0| = h*(rank+1), |p1| = h*(2S-rank)."""
        for cp_size in [2, 4, 8]:
            for cl in [4, 8]:
                h = cl // 2
                for rank in range(cp_size):
                    p0, p1 = generate_full_causal_kv_indices([cl], rank, cp_size)
                    self.assertEqual(len(p0), h * (rank + 1))
                    self.assertEqual(len(p1), h * (2 * cp_size - rank))


class TestGenerateNonlocalCausalKvIndices(unittest.TestCase):
    """Tests for generate_nonlocal_causal_kv_indices (overlap impl)."""

    def _check(self, cp_chunk_lengths, cp_rank, cp_size):
        actual = generate_nonlocal_causal_kv_indices(cp_chunk_lengths, cp_rank, cp_size)
        expected = _ref_nonlocal_causal(cp_chunk_lengths, cp_rank, cp_size)
        self.assertEqual(
            actual,
            expected,
            f"Mismatch: chunks={cp_chunk_lengths} rank={cp_rank} size={cp_size}",
        )

    def test_cp2_single_batch(self):
        for rank in range(2):
            self._check([4], rank, 2)

    def test_cp4_single_batch(self):
        for rank in range(4):
            self._check([4], rank, 4)

    def test_cp8_single_batch(self):
        for rank in range(8):
            self._check([4], rank, 8)

    def test_cp16_single_batch(self):
        for rank in range(16):
            self._check([4], rank, 16)

    def test_cp4_multi_batch(self):
        for rank in range(4):
            self._check([4, 8], rank, 4)

    def test_cp8_multi_batch(self):
        for rank in range(8):
            self._check([4, 6, 8], rank, 8)

    def test_counts(self):
        """Verify KV count formulas: |p0| = h*rank, |p1| = h*(2S-rank-2)."""
        for cp_size in [2, 4, 8, 16]:
            for cl in [4, 8]:
                h = cl // 2
                for rank in range(cp_size):
                    p0, p1 = generate_nonlocal_causal_kv_indices([cl], rank, cp_size)
                    self.assertEqual(len(p0), h * rank)
                    self.assertEqual(len(p1), h * (2 * cp_size - rank - 2))

    def test_no_local_positions(self):
        """Non-local indices must never contain any local zigzag position."""
        for cp_size in [2, 4, 8]:
            cl = 4
            full_len = cl * cp_size
            for rank in range(cp_size):
                local = _zigzag_positions(full_len, cp_size, rank)
                p0, p1 = generate_nonlocal_causal_kv_indices([cl], rank, cp_size)
                self.assertEqual(
                    len(set(p0) & local),
                    0,
                    f"p0 contains local pos: rank={rank} size={cp_size}",
                )
                self.assertEqual(
                    len(set(p1) & local),
                    0,
                    f"p1 contains local pos: rank={rank} size={cp_size}",
                )

    def test_causal_completeness(self):
        """non-local + local = full causal range for each Q-half."""
        for cp_size in [2, 4, 8]:
            cl = 4
            h = cl // 2
            full_len = cl * cp_size
            for rank in range(cp_size):
                local = sorted(_zigzag_positions(full_len, cp_size, rank))
                nl_p0, nl_p1 = generate_nonlocal_causal_kv_indices([cl], rank, cp_size)
                full_p0, full_p1 = generate_full_causal_kv_indices([cl], rank, cp_size)
                self.assertEqual(
                    sorted(nl_p0 + local[:h]),
                    full_p0,
                    f"part0 incomplete: rank={rank} size={cp_size}",
                )
                self.assertEqual(
                    sorted(nl_p1 + local),
                    full_p1,
                    f"part1 incomplete: rank={rank} size={cp_size}",
                )


class TestGenerateHalfQIndices(unittest.TestCase):

    def test_single_chunk(self):
        result = generate_half_q_indices([8])
        self.assertEqual(result, [4, 5, 6, 7])

    def test_multi_chunk(self):
        result = generate_half_q_indices([4, 6])
        self.assertEqual(result, [2, 3, 7, 8, 9])

    def test_matches_q_indices_second_half(self):
        """Should match the second-half output of generate_q_indices for even chunks."""
        for chunks in [[4], [8, 4], [6, 10, 2]]:
            _, q1 = generate_q_indices(chunks)
            hq = generate_half_q_indices(chunks)
            self.assertEqual(hq, q1)


class TestGenerateHalfKvIndices(unittest.TestCase):

    def test_single_chunk(self):
        result = generate_half_kv_indices([8])
        self.assertEqual(result, [0, 1, 2, 3])

    def test_multi_chunk(self):
        result = generate_half_kv_indices([4, 6])
        self.assertEqual(result, [0, 1, 4, 5, 6])

    def test_matches_q_indices_first_half(self):
        """Should match the first-half output of generate_q_indices for even chunks."""
        for chunks in [[4], [8, 4], [6, 10, 2]]:
            q0, _ = generate_q_indices(chunks)
            hk = generate_half_kv_indices(chunks)
            self.assertEqual(hk, q0)

    def test_disjoint_with_half_q(self):
        """half_kv (first halves) and half_q (second halves) should be disjoint."""
        for chunks in [[4, 8], [6, 10]]:
            hk = generate_half_kv_indices(chunks)
            hq = generate_half_q_indices(chunks)
            self.assertEqual(len(set(hk) & set(hq)), 0)
            self.assertEqual(sorted(hk + hq), list(range(sum(chunks))))


class TestPrefillPositionIds(unittest.TestCase):

    def _make_cp_inputs(self) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        cp_info = PyContextParallelParams()
        cp_info.prefill_shuffle_indices = torch.arange(8, dtype=torch.int32)
        attn_inputs.context_parallel_info = cp_info
        attn_inputs.is_prefill = True
        return attn_inputs

    def test_cp_keeps_explicit_three_axis_mrope_position_ids(self):
        position_ids = torch.stack(
            (
                torch.arange(8, dtype=torch.int32),
                torch.arange(100, 108, dtype=torch.int32),
                torch.arange(200, 208, dtype=torch.int32),
            ),
            dim=1,
        ).flatten()
        attn_inputs = self._make_cp_inputs()
        attn_inputs.combo_position_ids = position_ids

        selected = select_prefill_position_ids(attn_inputs)

        self.assertEqual(selected.shape, (24,))
        torch.testing.assert_close(selected, position_ids)

    def test_cp_text_path_falls_back_when_positions_are_unassigned(self):
        attn_inputs = self._make_cp_inputs()
        shuffle_indices = torch.arange(8, dtype=torch.int32)

        self.assertIsNone(attn_inputs.combo_position_ids)
        torch.testing.assert_close(
            select_prefill_position_ids(attn_inputs), shuffle_indices
        )

    def test_cp_text_path_falls_back_for_empty_explicit_positions(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.combo_position_ids = torch.empty(0, dtype=torch.int32)
        shuffle_indices = torch.arange(8, dtype=torch.int32)

        torch.testing.assert_close(
            select_prefill_position_ids(attn_inputs), shuffle_indices
        )

    def test_cp_text_path_rejects_prefix_reuse_without_explicit_positions(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.prefix_lengths = torch.tensor([4], dtype=torch.int32)

        with self.assertRaisesRegex(
            ValueError, "prefix reuse requires explicit position ids"
        ):
            select_prefill_position_ids(attn_inputs)

    def test_cp_rejects_position_ids_with_unsupported_axis_count(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.combo_position_ids = torch.arange(16, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "one text axis or three mRoPE axes"):
            select_prefill_position_ids(attn_inputs)

    def test_cp_rejects_positions_when_local_token_count_is_zero(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.context_parallel_info.prefill_shuffle_indices = torch.empty(
            0, dtype=torch.int32
        )
        attn_inputs.combo_position_ids = torch.arange(3, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "align with the local token count"):
            select_prefill_position_ids(attn_inputs)

    def test_cp_rejects_misaligned_position_ids(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.context_parallel_info.prefill_shuffle_indices = torch.arange(
            8, dtype=torch.int32
        )
        attn_inputs.combo_position_ids = torch.arange(9, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "align with the local token count"):
            select_prefill_position_ids(attn_inputs)

    def test_non_cp_path_keeps_explicit_positions(self):
        attn_inputs = PyAttentionInputs()
        position_ids = torch.arange(8, dtype=torch.int32)
        attn_inputs.combo_position_ids = position_ids
        attn_inputs.context_parallel_info = None

        torch.testing.assert_close(
            select_prefill_position_ids(attn_inputs), position_ids
        )

    def test_non_cp_path_without_positions_returns_none(self):
        attn_inputs = PyAttentionInputs()
        attn_inputs.context_parallel_info = None

        self.assertIsNone(select_prefill_position_ids(attn_inputs))

    def test_cp_rejects_non_prefill_batch(self):
        attn_inputs = self._make_cp_inputs()
        attn_inputs.is_prefill = False

        with self.assertRaisesRegex(
            ValueError, "context-parallel position ids require a pure-prefill batch"
        ):
            select_prefill_position_ids(attn_inputs)


if __name__ == "__main__":
    unittest.main()

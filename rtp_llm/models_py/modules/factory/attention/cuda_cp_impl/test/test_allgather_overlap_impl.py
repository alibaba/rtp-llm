"""
Unit tests for PCPAllGatherOverlapAttnOp (all-gather with compute-communication overlap).

Tests two scenarios:
  1. Normal context-parallel attention (no prefix cache)
  2. Context-parallel attention with prefix cache
"""

import contextlib
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_overlap_impl import (
    PCPAllGatherOverlapAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    CPAttnTestBase,
)

_AG_MODULE = (
    "rtp_llm.models_py.modules.factory.attention."
    "cuda_cp_impl.prefill_mha.allgather_overlap_impl"
)


class TestPCPAllGatherOverlapAttnOp(CPAttnTestBase):
    OP_CLASS = PCPAllGatherOverlapAttnOp
    AG_MODULE = _AG_MODULE

    def _extra_patches(self, stack: contextlib.ExitStack):
        """Disable user-buffers communicator so the op falls back to
        the standard ``all_gather`` that we mock."""
        stack.enter_context(
            patch(
                f"{_AG_MODULE}.get_user_buffers_communicator",
                return_value=None,
            )
        )

    # ==================================================================
    # Case 1: Normal CP attention (no prefix cache)
    # ==================================================================

    def test_no_prefix_single_seq_rank0(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=0)

    def test_no_prefix_single_seq_rank1(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=1)

    def test_no_prefix_multi_batch(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[32, 64], cp_size=2, cp_rank=0
        )

    def test_no_prefix_larger(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_no_prefix_cp4(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_size=4, cp_rank=2)

    def test_no_prefix_gqa4(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[32],
            cp_size=2,
            cp_rank=0,
            head_num=16,
            kv_head_num=4,
            head_dim=64,
        )

    def test_no_prefix_multi_batch_cp4(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[64, 64], cp_size=4, cp_rank=1
        )

    # ==================================================================
    # Case 2: CP attention with prefix cache
    # ==================================================================

    def test_prefix_single_seq_rank0(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_single_seq_rank1(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=1,
            tokens_per_block=16,
        )

    def test_prefix_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[32, 64],
            prefix_lengths=[64, 128],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=32,
        )

    def test_prefix_larger(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_prefix_cp4(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[64],
            cp_size=4,
            cp_rank=2,
            tokens_per_block=16,
        )

    # ==================================================================
    # Case 3: Irregular seq_len (non-power-of-2, partial pages)
    # ==================================================================

    def test_no_prefix_irregular_seqlen(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[20],
            cp_size=2,
            cp_rank=0,
        )

    def test_no_prefix_irregular_multi_batch(self):
        self.run_no_prefix(
            batch_size=2,
            sequence_lengths=[20, 36],
            cp_size=2,
            cp_rank=1,
        )

    def test_prefix_irregular_seqlen(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[20],
            prefix_lengths=[48],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_irregular_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[20, 36],
            prefix_lengths=[48, 32],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for PCPAllGatherAttnOp (all-gather without overlap).

Tests two scenarios:
  1. Normal context-parallel attention (no prefix cache)
  2. Context-parallel attention with prefix cache
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl import (
    prefill_cp_flashinfer,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_cp_flashinfer import (
    CPFlashInferImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    CPAttnTestBase,
)

_AG_MODULE = (
    "rtp_llm.models_py.modules.factory.attention."
    "cuda_cp_impl.prefill_mha.allgather_cp_impl"
)


class TestCPFlashInferForwardContract(unittest.TestCase):
    @staticmethod
    def _make_impl(need_rope_kv_cache):
        method = object()
        attn_configs = SimpleNamespace(need_rope_kv_cache=need_rope_kv_cache)
        attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
        parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(method=method)
        )
        fmha_impl = Mock()
        rope_impl = Mock()

        with patch.dict(
            prefill_cp_flashinfer.impl_map,
            {method: Mock(return_value=fmha_impl)},
            clear=True,
        ), patch.object(
            prefill_cp_flashinfer,
            "FusedRopeKVCachePrefillOpQKVOut",
            return_value=rope_impl,
        ):
            impl = CPFlashInferImpl(attn_configs, attn_inputs, parallelism_config)

        return impl, fmha_impl, rope_impl

    def test_layer_zero_does_not_disable_rope(self):
        impl, fmha_impl, rope_impl = self._make_impl(True)

        qkv = torch.randn(2, 8)
        rope_output = torch.randn(2, 8)
        attention_output = torch.randn(2, 4)
        rope_impl.forward.return_value = rope_output
        fmha_impl.forward.return_value = attention_output

        result = impl.forward(qkv, kv_cache=None, layer_idx=0)

        rope_impl.forward.assert_called_once_with(qkv, None, impl.rope_params)
        fmha_impl.forward.assert_called_once_with(rope_output, None, impl.fmha_params)
        self.assertIs(result, attention_output)

    def test_forward_skips_rope_when_disabled(self):
        impl, fmha_impl, rope_impl = self._make_impl(False)

        qkv = torch.randn(2, 8)
        attention_output = torch.randn(2, 4)
        fmha_impl.forward.return_value = attention_output

        result = impl.forward(qkv, kv_cache=None, layer_idx=0)

        rope_impl.forward.assert_not_called()
        fmha_impl.forward.assert_called_once_with(qkv, None, impl.fmha_params)
        self.assertIs(result, attention_output)


class TestPCPAllGatherAttnOp(CPAttnTestBase):
    OP_CLASS = PCPAllGatherAttnOp
    AG_MODULE = _AG_MODULE

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

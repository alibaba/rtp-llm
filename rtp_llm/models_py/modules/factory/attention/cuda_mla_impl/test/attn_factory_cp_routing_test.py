import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.hybrid.indexer import Indexer


class _BaseFakeImpl:
    @staticmethod
    def support(attn_configs, attn_inputs):
        return True

    @staticmethod
    def is_sparse():
        return False

    def __init__(self, *args, **kwargs):
        pass

    def support_cuda_graph(self):
        return True


class _NonCpImpl(_BaseFakeImpl):
    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return False


class _CpImpl(_BaseFakeImpl):
    @staticmethod
    def is_sparse():
        return True

    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return True


class AttentionFactoryCpRoutingTest(unittest.TestCase):
    def setUp(self):
        self.config = types.SimpleNamespace(indexer_topk=8, is_sparse=False)
        self.weights = types.SimpleNamespace(
            weights=[],
            get_global_weight_or_none=lambda name: None,
        )
        self.parallelism = types.SimpleNamespace(
            prefill_cp_config=types.SimpleNamespace(is_enabled=lambda: True)
        )

    @staticmethod
    def _attention_inputs(is_prefill, cp_info, kv_len=16):
        return types.SimpleNamespace(
            is_prefill=is_prefill,
            context_parallel_info=cp_info,
            cu_kv_seqlens=torch.tensor([0, kv_len], dtype=torch.int32),
        )

    def _select(self, attention_inputs):
        with patch.object(attn_factory, "PREFILL_MLA_IMPS", [_NonCpImpl, _CpImpl]), patch.object(
            attn_factory, "DECODE_MLA_IMPS", [_NonCpImpl, _CpImpl]
        ):
            return attn_factory.get_mla_impl(
                self.config,
                self.weights,
                attention_inputs,
                parallelism_config=self.parallelism,
            )

    def test_decode_does_not_inherit_prefill_cp_requirement(self):
        selected = self._select(self._attention_inputs(False, None))
        self.assertIsInstance(selected, _NonCpImpl)

    def test_prefill_without_cp_metadata_uses_non_cp_implementation(self):
        selected = self._select(self._attention_inputs(True, None))
        self.assertIsInstance(selected, _NonCpImpl)

    def test_prefill_with_cp_metadata_requires_cp_implementation(self):
        selected = self._select(self._attention_inputs(True, types.SimpleNamespace()))
        self.assertIsInstance(selected, _CpImpl)

    def test_short_prefill_without_cp_metadata_keeps_dense_fast_path(self):
        selected = self._select(self._attention_inputs(True, None, kv_len=4))
        self.assertIsInstance(selected, _NonCpImpl)

    def test_short_prefill_with_cp_metadata_does_not_use_dense_fast_path(self):
        selected = self._select(
            self._attention_inputs(True, types.SimpleNamespace(), kv_len=4)
        )
        self.assertIsInstance(selected, _CpImpl)

    def test_indexer_uses_the_same_request_level_cp_signal(self):
        self.assertFalse(
            Indexer._is_sparse_prefill_cp(
                None, self._attention_inputs(False, types.SimpleNamespace())
            )
        )
        self.assertFalse(
            Indexer._is_sparse_prefill_cp(None, self._attention_inputs(True, None))
        )
        self.assertTrue(
            Indexer._is_sparse_prefill_cp(
                None, self._attention_inputs(True, types.SimpleNamespace())
            )
        )


if __name__ == "__main__":
    unittest.main()

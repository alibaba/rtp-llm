import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class _ImplWithoutPrefillCP:
    accepts_fmha_config = False

    @staticmethod
    def support(attn_configs, attn_inputs):
        return True

    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return False

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.fmha_params = object()

    def support_cuda_graph(self):
        return False


class _MlaImpl:
    sparse = False

    @staticmethod
    def support(attn_configs, attn_inputs):
        return True

    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return True

    @classmethod
    def is_sparse(cls):
        return cls.sparse

    def __init__(self, *args, **kwargs):
        pass

    def support_cuda_graph(self):
        return False


class _SparseMlaImpl(_MlaImpl):
    sparse = True


class AttentionFactoryCPSelectionTest(unittest.TestCase):
    def _get_impl(self, context_parallel_info):
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=context_parallel_info,
        )
        attn_configs = SimpleNamespace(
            rope_config=SimpleNamespace(style=None, mrope_interleaved=True)
        )
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", [_ImplWithoutPrefillCP]):
            return attn_factory.get_fmha_impl(
                attn_configs,
                None,
                attn_inputs,
                parallelism_config=object(),
            )

    def _get_mla_impl(self, context_parallel_info):
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=context_parallel_info,
            cu_kv_seqlens_device=SimpleNamespace(
                max=lambda: SimpleNamespace(item=lambda: 4)
            ),
        )
        attn_configs = SimpleNamespace(indexer_topk=8, is_sparse=True)
        weights = SimpleNamespace(weights={}, get_global_weight_or_none=lambda _: None)
        with patch.object(attn_factory, "PREFILL_MLA_IMPS", [_SparseMlaImpl, _MlaImpl]):
            return attn_factory.get_mla_impl(
                attn_configs,
                weights,
                attn_inputs,
                parallelism_config=object(),
            )

    def test_process_cp_config_does_not_filter_a_regular_forward(self):
        self.assertIsInstance(self._get_impl(None), _ImplWithoutPrefillCP)
        self.assertIsInstance(self._get_mla_impl(None), _MlaImpl)

    def test_cp_metadata_filters_an_incompatible_prefill_implementation(self):
        with self.assertRaisesRegex(Exception, "can not find mha type"):
            self._get_impl(object())
        self.assertIsInstance(self._get_mla_impl(object()), _SparseMlaImpl)


if __name__ == "__main__":
    unittest.main()

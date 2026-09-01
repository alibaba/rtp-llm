import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention import attn_factory


class _ImplWithoutPrefillCP:
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


class TestAttentionFactoryCPSelection(unittest.TestCase):
    def test_cp_config_does_not_filter_non_cp_forward(self):
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            context_parallel_info=None,
        )
        with patch.object(
            attn_factory, "PREFILL_MHA_IMPS", [_ImplWithoutPrefillCP]
        ):
            impl = attn_factory.get_fmha_impl(
                None,
                None,
                attn_inputs,
                parallelism_config=object(),
            )

        self.assertIsInstance(impl, _ImplWithoutPrefillCP)

    def test_cp_metadata_filters_non_cp_prefill_implementation(self):
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            context_parallel_info=object(),
        )
        with patch.object(
            attn_factory, "PREFILL_MHA_IMPS", [_ImplWithoutPrefillCP]
        ):
            with self.assertRaisesRegex(Exception, "can not find mha type"):
                attn_factory.get_fmha_impl(
                    None,
                    None,
                    attn_inputs,
                    parallelism_config=object(),
                )


if __name__ == "__main__":
    unittest.main()

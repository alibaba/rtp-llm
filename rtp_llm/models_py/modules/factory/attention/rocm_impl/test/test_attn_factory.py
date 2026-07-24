import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _is_fmha_impl_disabled,
    get_fmha_impl,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


class AiterPrefillImplPaged:
    last_v1_kv_layout = None

    def __init__(
        self,
        attn_configs,
        attn_inputs,
        parallelism_config=None,
        v1_kv_layout=False,
    ):
        type(self).last_v1_kv_layout = v1_kv_layout

    @classmethod
    def support(cls, attn_configs, attn_inputs):
        return True

    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return True

    def support_cuda_graph(self):
        return True


class AiterPrefillImplAsm:
    @classmethod
    def support(cls, attn_configs, attn_inputs):
        return True


class AiterPrefillFactoryTest(unittest.TestCase):
    @staticmethod
    def _config(*, use_aiter_pa: bool, use_asm_pa: bool, use_triton_pa: bool):
        return SimpleNamespace(
            use_aiter_pa=use_aiter_pa,
            use_asm_pa=use_asm_pa,
            use_triton_pa=use_triton_pa,
        )

    def test_paged_prefill_is_enabled_by_aiter(self):
        config = self._config(use_aiter_pa=True, use_asm_pa=False, use_triton_pa=False)

        self.assertFalse(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_paged_prefill_is_not_enabled_by_asm_alone(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=True, use_triton_pa=False)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_paged_prefill_is_not_enabled_by_triton_alone(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=False, use_triton_pa=True)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplPaged", config))

    def test_triton_does_not_enable_pure_asm_prefill(self):
        config = self._config(use_aiter_pa=False, use_asm_pa=False, use_triton_pa=True)

        self.assertTrue(_is_fmha_impl_disabled("AiterPrefillImplAsm", config))

    def _select_paged_prefill(self, *, use_asm_pa: bool):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        config = self._config(
            use_aiter_pa=True,
            use_asm_pa=use_asm_pa,
            use_triton_pa=False,
        )
        AiterPrefillImplPaged.last_v1_kv_layout = None
        with patch.object(
            attn_factory,
            "PREFILL_MHA_IMPS",
            [AiterPrefillImplPaged, AiterPrefillImplAsm],
        ):
            get_fmha_impl(
                SimpleNamespace(),
                None,
                inputs,
                fmha_config=config,
            )
        return AiterPrefillImplPaged.last_v1_kv_layout

    def test_nonasm_prefill_propagates_v1_cache_layout(self):
        self.assertTrue(self._select_paged_prefill(use_asm_pa=False))

    def test_asm_prefill_propagates_vectorized_cache_layout(self):
        self.assertFalse(self._select_paged_prefill(use_asm_pa=True))


if __name__ == "__main__":
    unittest.main()

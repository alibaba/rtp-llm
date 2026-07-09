import unittest
from types import SimpleNamespace

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _is_fmha_impl_disabled,
)


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


if __name__ == "__main__":
    unittest.main()

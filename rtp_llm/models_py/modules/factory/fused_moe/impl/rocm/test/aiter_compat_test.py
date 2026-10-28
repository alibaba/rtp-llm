import importlib.util
import os
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch


_IMPL_DIR = Path(__file__).resolve().parents[1]
_MODULE_PATH = _IMPL_DIR / "aiter_compat.py"
_SPEC = importlib.util.spec_from_file_location(
    "rtp_aiter_compat_test_module", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class AiterCompatTest(TestCase):
    def test_affected_gfx942_build_uses_ck_sorting(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942:sramecc+:xnack-",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertTrue(changed)
            self.assertEqual(os.environ["AITER_USE_CK_MOE_SORTING"], "1")

    def test_gfx950_keeps_opus_default(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx950",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertFalse(changed)
            self.assertNotIn("AITER_USE_CK_MOE_SORTING", os.environ)

    def test_newer_aiter_build_keeps_its_default(self):
        with patch.dict(os.environ, {}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942",
                aiter_version="0.1.18.dev1+gabcdef.d20260801",
            )

            self.assertFalse(changed)
            self.assertNotIn("AITER_USE_CK_MOE_SORTING", os.environ)

    def test_explicit_user_choice_is_preserved(self):
        with patch.dict(os.environ, {"AITER_USE_CK_MOE_SORTING": "0"}, clear=True):
            changed = _MODULE.configure_aiter_moe_sorting(
                arch="gfx942",
                aiter_version="0.1.17.dev79+g2570b35f9.d20260623",
            )

            self.assertFalse(changed)
            self.assertEqual(os.environ["AITER_USE_CK_MOE_SORTING"], "0")

    def test_executor_configures_sorting_before_importing_aiter(self):
        source = (_IMPL_DIR / "executors" / "rocm_moe.py").read_text()

        self.assertLess(
            source.index("configure_aiter_moe_sorting()"),
            source.index('importlib.import_module("aiter")'),
        )


if __name__ == "__main__":
    main()

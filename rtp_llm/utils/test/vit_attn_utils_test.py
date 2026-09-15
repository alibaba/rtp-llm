import os
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from rtp_llm.utils.vit_attn_utils import get_vit_attn_implementation

MODULE = "rtp_llm.utils.vit_attn_utils"


class VitAttnImplementationTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {}, clear=False))
        os.environ.pop("VIT_ATTN_IMPLEMENTATION", None)
        self.device = stack.enter_context(
            patch(f"{MODULE}.can_use_flash_attn", return_value=True)
        )
        self.available = stack.enter_context(
            patch(f"{MODULE}.is_flash_attn_2_available", return_value=False)
        )
        self.import_module = stack.enter_context(
            patch(f"{MODULE}.importlib.import_module", return_value=Mock())
        )

    def test_mi308_without_flash_attn_falls_back_and_logs(self):
        with patch(
            "rtp_llm.utils.flash_attn_utils.torch.cuda.get_device_capability",
            return_value=(9, 4),
        ), patch(
            "rtp_llm.utils.flash_attn_utils.torch.cuda.get_device_name",
            return_value="AMD Instinct MI308X",
        ):
            from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

            self.device.side_effect = can_use_flash_attn
            with self.assertLogs(level="INFO") as logs:
                self.assertEqual(get_vit_attn_implementation(), "sdpa")
        self.assertIn(
            "requested=auto selected=sdpa reason=flash_attn_unavailable",
            logs.output[-1],
        )
        self.import_module.assert_not_called()

    def test_available_flash_attn_is_selected(self):
        self.available.return_value = True
        self.assertEqual(get_vit_attn_implementation(), "flash_attention_2")
        self.import_module.assert_called_once_with("flash_attn")

    def test_unsupported_device_falls_back(self):
        self.device.return_value = False
        self.assertEqual(get_vit_attn_implementation(), "sdpa")
        self.available.assert_not_called()

    def test_device_check_error_falls_back(self):
        self.device.side_effect = RuntimeError("No GPU")
        self.assertEqual(get_vit_attn_implementation(), "sdpa")

    def test_broken_flash_attn_extension_falls_back(self):
        self.available.return_value = True
        self.import_module.side_effect = ImportError("undefined symbol")
        self.assertEqual(get_vit_attn_implementation(), "sdpa")

    def test_explicit_sdpa_and_eager_skip_flash_attn_checks(self):
        for backend in ("sdpa", "eager"):
            with self.subTest(backend=backend):
                os.environ["VIT_ATTN_IMPLEMENTATION"] = backend
                self.assertEqual(get_vit_attn_implementation(), backend)
        self.device.assert_not_called()
        self.available.assert_not_called()
        self.import_module.assert_not_called()

    def test_explicit_unavailable_flash_attn_fails_clearly(self):
        os.environ["VIT_ATTN_IMPLEMENTATION"] = "flash_attention_2"
        with self.assertRaisesRegex(RuntimeError, "flash_attn_unavailable"):
            get_vit_attn_implementation()

    def test_explicit_available_flash_attn_is_selected(self):
        os.environ["VIT_ATTN_IMPLEMENTATION"] = "flash_attention_2"
        self.available.return_value = True
        self.assertEqual(get_vit_attn_implementation(), "flash_attention_2")

    def test_invalid_backend_rejected(self):
        os.environ["VIT_ATTN_IMPLEMENTATION"] = "typo"
        with self.assertRaisesRegex(ValueError, "Invalid VIT_ATTN_IMPLEMENTATION"):
            get_vit_attn_implementation()
        self.device.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import builtins
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def load_flash_attn_utils(cuda_capability=(8, 0), device_name="NVIDIA A10"):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            get_device_capability=mock.Mock(return_value=cuda_capability),
            get_device_name=mock.Mock(return_value=device_name),
        )
    )
    module_path = Path(__file__).resolve().parents[1] / "flash_attn_utils.py"
    module_name = "flash_attn_utils_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    return module


class FlashAttnUtilsTest(unittest.TestCase):
    def test_env_override_sdpa_skips_cuda_and_import_checks(self):
        flash_attn_utils = load_flash_attn_utils()
        with mock.patch.dict(os.environ, {"RTP_LLM_VISION_ATTN_IMPL": "sdpa"}):
            with mock.patch.object(
                flash_attn_utils.torch.cuda,
                "get_device_capability",
                side_effect=AssertionError("cuda should not be queried"),
            ):
                self.assertEqual(
                    flash_attn_utils.get_default_vision_attention_impl(), "sdpa"
                )

    def test_auto_falls_back_to_sdpa_when_flash_attn_is_missing(self):
        flash_attn_utils = load_flash_attn_utils()
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("flash_attn missing")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(builtins, "__import__", fake_import):
                self.assertEqual(
                    flash_attn_utils.get_default_vision_attention_impl(), "sdpa"
                )

    def test_auto_uses_flash_attention_2_when_supported_and_installed(self):
        flash_attn_utils = load_flash_attn_utils()
        real_import = builtins.__import__
        fake_flash_attn = mock.Mock()

        def fake_import(name, *args, **kwargs):
            if name == "flash_attn":
                return fake_flash_attn
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(builtins, "__import__", fake_import):
                self.assertEqual(
                    flash_attn_utils.get_default_vision_attention_impl(),
                    "flash_attention_2",
                )

    def test_invalid_env_override_raises_value_error(self):
        flash_attn_utils = load_flash_attn_utils()
        with mock.patch.dict(os.environ, {"RTP_LLM_VISION_ATTN_IMPL": "invalid"}):
            with self.assertRaisesRegex(ValueError, "RTP_LLM_VISION_ATTN_IMPL"):
                flash_attn_utils.get_default_vision_attention_impl()


if __name__ == "__main__":
    unittest.main()

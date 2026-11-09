import builtins
import itertools
import os
import unittest
from unittest.mock import patch

from rtp_llm.models_py.sampling.adaptive_runtime import create_controller


class AdaptiveRuntimeTest(unittest.TestCase):
    def test_disabled_and_default_do_not_import_gpu_dependencies(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.endswith("adaptive_controller"):
                self.fail(f"disabled adaptive verification imported {name}")
            return original_import(name, *args, **kwargs)

        for adaptive, gumbel, rejection in itertools.product(
            (None, "0"), (None, "0", "1"), (None, "0", "1")
        ):
            env = {
                key: value
                for key, value in (
                    ("RTP_LLM_ADAPTIVE_VERIFICATION", adaptive),
                    ("RTP_LLM_USE_GUMBEL_SAMPLING", gumbel),
                    ("RTP_LLM_USE_VLLM_REJECTION_SAMPLING", rejection),
                )
                if value is not None
            }
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with patch("builtins.__import__", side_effect=guarded_import):
                    self.assertIsNone(create_controller(8, 3, "cuda"))
                self.assertEqual(dict(os.environ), env)

    def test_invalid_values_fail_instead_of_silently_enabling(self) -> None:
        for value in ("", "true", "false", "2", "-1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"RTP_LLM_ADAPTIVE_VERIFICATION": value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    create_controller(8, 3, "cuda")


if __name__ == "__main__":
    unittest.main()

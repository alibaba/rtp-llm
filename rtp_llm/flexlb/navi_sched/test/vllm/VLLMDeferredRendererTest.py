import builtins
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

source = Path(__file__).resolve().parents[2] / "adapters/vllm/VLLMDeferredRenderer.py"
spec = importlib.util.spec_from_file_location("deferred_renderer_test_module", source)
deferred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deferred)


class VLLMDeferredRendererTest(unittest.TestCase):
    def test_text_renderer_does_not_import_multimodal_models(self):
        renderer = SimpleNamespace(mm_processor=None)
        original_import = builtins.__import__

        def ppu_import(name, *args, **kwargs):
            if name.startswith("vllm.model_executor.models."):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}),
            mock.patch("builtins.__import__", side_effect=ppu_import),
        ):
            self.assertIs(renderer, deferred.create_deferred_renderer(renderer))

    def test_disabled_mode_preserves_renderer_with_media(self):
        renderer = SimpleNamespace(mm_processor=object())
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "0"}):
            self.assertIs(renderer, deferred.create_deferred_renderer(renderer))


if __name__ == "__main__":
    unittest.main()

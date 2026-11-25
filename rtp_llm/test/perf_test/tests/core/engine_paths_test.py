"""Hub references remain usable without probe environment overrides."""

import os
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from rtp_llm.test.perf_test import engine_paths


class EnginePathsTest(unittest.TestCase):
    def test_hub_checkpoint_resolves_with_default_probe_urls(self):
        for fallback in (False, True):
            with self.subTest(huggingface_fallback=fallback):
                modelscope = types.ModuleType("modelscope")
                huggingface = types.ModuleType("huggingface_hub")

                def download_modelscope(repo_id):
                    if fallback:
                        raise RuntimeError("ModelScope download unavailable")
                    return "/downloaded/" + repo_id

                modelscope.snapshot_download = download_modelscope
                huggingface.snapshot_download = lambda repo_id: "/fallback/" + repo_id
                with patch.dict(os.environ, {}, clear=True), patch.dict(
                    sys.modules,
                    {"modelscope": modelscope, "huggingface_hub": huggingface},
                ), patch.object(
                    engine_paths.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, stdout="200"),
                ):
                    try:
                        result = engine_paths.resolve_engine_paths(
                            ["--checkpoint_path=Qwen/test-model"]
                        )
                    except NameError as error:
                        self.fail(f"Hub resolution cannot use default probes: {error}")
                prefix = "/fallback/" if fallback else "/downloaded/"
                self.assertEqual(
                    ["--checkpoint_path=" + prefix + "Qwen/test-model"], result
                )


if __name__ == "__main__":
    unittest.main()

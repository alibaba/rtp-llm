import subprocess
import sys
import types
import unittest

from rtp_llm.models_py.quantization_exclusion import (
    collect_quantization_exclusions,
    is_module_ignored,
    normalize_module_patterns,
)


class QuantizationExclusionTest(unittest.TestCase):
    def test_collects_aliases_and_matches_moe_prefixes(self):
        config = types.SimpleNamespace(
            ignored_layers=["model.layers.1.mlp.experts"],
            exclude_modules=["layers.{i}.self_attn"],
            modules_to_not_convert=["re:^layers\\.3\\.mlp\\.experts"],
        )

        patterns = collect_quantization_exclusions(config)

        self.assertTrue(is_module_ignored("layers.1.mlp.experts", patterns))
        self.assertTrue(is_module_ignored("layers.2.self_attn.q_proj", patterns))
        self.assertTrue(is_module_ignored("layers.3.mlp.experts", patterns))
        self.assertFalse(is_module_ignored("layers.0.mlp.experts", patterns))

    def test_set_normalization_is_deterministic(self):
        self.assertEqual(normalize_module_patterns({"b", "a"}), ["a", "b"])

    def test_helper_import_does_not_load_deepep_or_quant_runtime(self):
        # Other suites legitimately import quant_methods. Check the helper's
        # first-import boundary in isolation rather than global test order.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "import rtp_llm.models_py.quantization_exclusion; "
                "forbidden = ('rtp_llm.models_py.distributed.deepep_wrapper', "
                "'rtp_llm.models_py.quant_methods.base'); "
                "loaded = [name for name in forbidden if name in sys.modules]; "
                "assert not loaded, loaded",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

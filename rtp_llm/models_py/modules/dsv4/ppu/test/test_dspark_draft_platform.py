"""Static contract: the PPU DSpARK draft must not reach CUDA-only operators.

Importing the real modules needs the native runtime, so these assertions read
the sources instead.  They exist to catch the two regressions that silently
put PPU back on CUDA kernels: dropping a ``_dspark_*`` hook from the shared
attention body, and dropping the PPU branch from model construction.
"""

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[6]
PPU_DSPARK = ROOT / "rtp_llm/models_py/model_desc/deepseek_v4_ppu/dspark_model.py"
SHARED_DSPARK = ROOT / "rtp_llm/models_py/model_desc/deepseek_v4_dspark_model.py"
MODEL_ENTRY = ROOT / "rtp_llm/models/deepseek_v4.py"


def _parse(path):
    return ast.parse(Path(path).read_text())


def _class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls, name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found on {cls.name}")


def _imports_from(tree, module_suffix):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith(
            module_suffix
        ):
            names.update(alias.name for alias in node.names)
    return names


class PpuDsparkDraftContractTest(unittest.TestCase):
    def test_ppu_draft_mixes_shared_dspark_with_ppu_buffers(self):
        cls = _class(_parse(PPU_DSPARK), "PpuDeepSeekV4DSparkModel")
        bases = [ast.unparse(base) for base in cls.bases]
        self.assertEqual(bases, ["DeepSeekV4DSparkModel", "PpuDeepSeekV4Model"])

    def test_ppu_draft_binds_every_decode_operator_to_ppu(self):
        tree = _parse(PPU_DSPARK)
        self.assertIn(
            "decode_compute_qkv",
            _imports_from(tree, "ppu.attention.fp8.decode.compute_qkv"),
        )
        self.assertIn(
            "decode_output_proj",
            _imports_from(tree, "ppu.attention.fp8.decode.output_proj"),
        )
        self.assertIn(
            "attn_fp8_swa_paged",
            _imports_from(tree, "ppu.attention.fp8.decode.attention_kernels"),
        )
        self.assertIn(
            "get_or_build_sched_meta",
            _imports_from(tree, "ppu.attention.fp8.decode.decode_attn_metadata"),
        )
        load = _method(
            _class(tree, "PpuDeepSeekV4DSparkModel"), "_load_extra_weights"
        )
        self.assertIn("platform_provider=self._platform_provider", ast.unparse(load))

    def test_shared_attention_body_only_talks_to_hooks(self):
        cls = _class(_parse(SHARED_DSPARK), "DeepSeekV4DSparkModel")
        body = ast.unparse(_method(cls, "_forward_dspark_attention"))
        for hook in (
            "self._dspark_compute_qkv",
            "self._dspark_sched_meta",
            "self._dspark_attend",
            "self._dspark_output_proj",
        ):
            self.assertIn(hook, body)
        # Both platforms share one KV writer; it stays inline instead of
        # becoming an unused seam.
        self.assertIn("decode_write_swa_fp8(", body)
        for cuda_only in (
            "decode_compute_qkv(",
            "decode_output_proj(",
            "_get_fp8_decode_op",
        ):
            self.assertNotIn(cuda_only, body)

    def test_model_entry_selects_ppu_draft_on_ppu_flash(self):
        cls = _class(_parse(MODEL_ENTRY), "DeepSeekV4DSpark")
        body = ast.unparse(_method(cls, "_create_python_model"))
        self.assertIn("_is_ppu_flash(self.model_config)", body)
        self.assertIn("deepseek_v4_ppu.dspark_model", body)
        self.assertIn("PpuDeepSeekV4DSparkModel", body)
        self.assertIn("deepseek_v4_dspark_model", body)


if __name__ == "__main__":
    unittest.main()

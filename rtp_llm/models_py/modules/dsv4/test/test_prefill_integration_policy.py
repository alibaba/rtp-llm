"""CPU checks for SM120 backend selection and indexer representation."""

import ast
import os
import runpy
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/models_py/modules/dsv4/fp8/attention.py").is_file()
)


class IntegrationPolicyTest(unittest.TestCase):
    def test_output_projection_and_warmup_select_the_same_backend(self):
        def dispatch(path, method, sm120):
            tree = ast.parse((ROOT / path).read_text())
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == method
            )
            condition = next(
                node.test
                for node in ast.walk(function)
                if isinstance(node, ast.If) and "is_sm120" in ast.unparse(node.test)
            )
            return eval(
                compile(ast.Expression(condition), path, "eval"),
                {
                    "is_sm120": lambda device: sm120,
                    "device": object(),
                    "o_fp8": types.SimpleNamespace(device=object()),
                },
            )

        for sm120 in (False, True):
            for path, method in (
                (
                    "rtp_llm/models_py/modules/dsv4/fp8/attention.py",
                    "_wo_a_einsum_from_fp8",
                ),
                (
                    "rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py",
                    "warmup_batched_fp8_einsum_jit",
                ),
            ):
                with self.subTest(sm120=sm120, method=method):
                    self.assertEqual(dispatch(path, method, sm120), sm120)

    def test_indexer_layout_rejects_unsupported_modes(self):
        layout = runpy.run_path(str(ROOT / "rtp_llm/models/dsv4_indexer_cache.py"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(layout["indexer_entry_bytes"](), 132)
        for mode in ("fp4", "invalid", ""):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"DSV4_INDEXER_CACHE_MODE": mode}
            ):
                with self.assertRaisesRegex(ValueError, "supports only fp8"):
                    layout["indexer_entry_bytes"]()


if __name__ == "__main__":
    unittest.main()

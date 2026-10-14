"""Decode HC Graph construction gate no longer whitelists TP1 (source-level).

The fused kernels consume replicated full-width inputs; the deterministic
prenorm GEMM reduces one complete K=hc*dim plane per rank, so identical
replicated activations produce bit-identical HC outputs at any TP size.  The
construction-time TP1 whitelist was the only blocker.
"""

import ast
import unittest
from pathlib import Path

_HC_PATH = Path(__file__).resolve().parents[1] / "models/dsv4/ppu_hc.py"
_DECODE_PROVIDER_PATH = (
    Path(__file__).resolve().parents[1] / "models/dsv4/ppu_decode_provider.py"
)
_FACTORY_PATH = (
    Path(__file__).resolve().parents[3] / "models_py/modules/dsv4/hc/factory.py"
)


def _init_tests(tree):
    unit = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PpuHCUnit"
    )
    init = next(
        node for node in unit.body if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
    )
    return [
        node.test
        for node in ast.walk(init)
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
    ]


class HCGraphTopologyGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _HC_PATH.read_text()
        cls.tree = ast.parse(cls.source)

    def test_graph_gate_requires_only_the_deterministic_backend(self):
        tests = _init_tests(self.tree)
        graph_gates = [
            test for test in tests if "allow_graph" in ast.unparse(test)
        ]
        self.assertTrue(graph_gates)
        for test in graph_gates:
            self.assertNotIn("tp_size", ast.unparse(test))
        self.assertIn(
            'raise ValueError("PPU HC Graph requires deterministic prenorm")',
            self.source,
        )
        self.assertNotIn("requires TP1", self.source)
        self.assertIn("self._backend != \"deepgemm_deterministic\"", self.source)

    def test_fused_prenorm_gate_requires_capture_and_determinism_only(self):
        tests = _init_tests(self.tree)
        prenorm_gates = [
            test for test in tests if "fuse_prenorm" in ast.unparse(test)
        ] or [
            test
            for test in tests
            if "not allow_graph" in ast.unparse(test)
        ]
        self.assertTrue(prenorm_gates)
        for test in prenorm_gates:
            self.assertNotIn("tp_size", ast.unparse(test))
        self.assertIn(
            'raise ValueError("Fused HC reduction requires deterministic Decode")',
            self.source,
        )
        self.assertNotIn("TP1 deterministic Decode", self.source)

    def test_replicated_input_guard_and_deterministic_partials_are_intact(self):
        self.assertIn(
            "Fused HC norm requires replicated contiguous BF16 inputs", self.source
        )
        self.assertIn("tf32_hc_prenorm_partials", self.source)
        self.assertIn("prenorm_output=\"reduced\"", self.source)
        self.assertIn(
            "This PPU HC instance does not allow Graph capture", self.source
        )
        self.assertIn("tp_size <= 0", self.source)

    def test_decode_provider_still_opts_into_the_fused_boundary(self):
        source = _DECODE_PROVIDER_PATH.read_text()
        for fragment in ("allow_graph=True", "fuse_prenorm=True", "fuse_norm=True"):
            self.assertIn(fragment, source)
        tree = ast.parse(source)
        builder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "build_hc_unit"
        )
        call = next(
            node
            for node in ast.walk(builder)
            if isinstance(node, ast.Call) and "PpuHCUnit" in ast.unparse(node.func)
        )
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        self.assertEqual(keywords.get("options"), "self.execution_options")
        self.assertEqual(keywords.get("allow_graph"), "True")
        self.assertEqual(keywords.get("fuse_prenorm"), "True")
        self.assertEqual(keywords.get("fuse_norm"), "True")
        # The factory forwards tp_size/tp_rank through **kwargs into the unit.
        self.assertIn("tp_size=tp_size", self.source.replace(" ", ""))
        self.assertIn(
            "tp_size=tp_size", _FACTORY_PATH.read_text().replace(" ", "")
        )


if __name__ == "__main__":
    unittest.main()

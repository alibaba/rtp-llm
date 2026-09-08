"""Source-only gates for the final DSV4 MoE strategy marker."""

from __future__ import annotations

import ast
import pathlib
import unittest


_MOE_LAYER = pathlib.Path(__file__).parents[1] / "moe" / "moe_layer.py"
_MARKER = "DSV4_MOE_FINAL_STRATEGY"
_REQUIRED_FIELDS = (
    "env_strategy=",
    "env_use_mega=",
    "env_use_mega_se=",
    "env_use_grouped_fp4=",
    "ctor=",
    "resolved=",
    "strict=",
    "selected_name=",
    "selected_class=",
    "layer=",
    "rank=",
    "tp=",
    "ep=",
    "ep_rank=",
)
_FORBIDDEN_CALLS = {"item", "cpu", "tolist", "synchronize"}


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


class MoeFinalStrategyObservabilitySourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = _MOE_LAYER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        moe = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MoE"
        )
        cls.init = next(
            node
            for node in moe.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        )
        cls.forward = next(
            node
            for node in moe.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "forward"
        )
        cls.marker_calls = [
            node
            for node in ast.walk(cls.init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.func.attr == "info"
            and _MARKER in ast.get_source_segment(cls.source, node)
        ]

    def test_marker_is_unique_and_after_strategy_instantiation(self) -> None:
        self.assertEqual(len(self.marker_calls), 1)
        marker = self.marker_calls[0]
        assignments = [
            node
            for node in ast.walk(self.init)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_strategy"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertLess(assignments[0].lineno, marker.lineno)
        self.assertNotIn(_MARKER, ast.get_source_segment(self.source, self.forward))

    def test_marker_exposes_final_selection_inputs_and_topology(self) -> None:
        marker_source = ast.get_source_segment(self.source, self.marker_calls[0])
        for field in _REQUIRED_FIELDS:
            self.assertIn(field, marker_source)

    def test_marker_has_no_device_sync_or_tensor_materialization(self) -> None:
        calls = {
            _call_name(node)
            for node in ast.walk(self.marker_calls[0])
            if isinstance(node, ast.Call)
        }
        self.assertTrue(_FORBIDDEN_CALLS.isdisjoint(calls), calls)


if __name__ == "__main__":
    unittest.main()

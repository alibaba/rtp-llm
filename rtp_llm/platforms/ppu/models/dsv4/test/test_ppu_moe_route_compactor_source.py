"""Source contracts for the fused PPU MXFP4 route compactor."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

_KERNEL_PATH = Path(__file__).resolve().parents[3] / "kernels/ppu_moe_route_compactor.py"
_WRAPPER_PATH = Path(__file__).resolve().parents[3] / "kernels/ppu_moe_nopad.py"


def _load_enable_helper():
    tree = ast.parse(_WRAPPER_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "route_compactor_fused_enabled"
    )
    namespace = {"os": os, "_COMPACTOR_ENV": "DSV4_MOE_ROUTE_COMPACTOR"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(_WRAPPER_PATH), "exec"),
        namespace,
    )
    return namespace["route_compactor_fused_enabled"]


class PpuMoeRouteCompactorSourceTest(unittest.TestCase):
    def test_fused_default_and_torch_rollback(self):
        enabled = _load_enable_helper()
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(enabled())
        with patch.dict(
            os.environ, {"DSV4_MOE_ROUTE_COMPACTOR": "fused"}, clear=True
        ):
            self.assertTrue(enabled())
        with patch.dict(
            os.environ, {"DSV4_MOE_ROUTE_COMPACTOR": "torch"}, clear=True
        ):
            self.assertFalse(enabled())
        with patch.dict(
            os.environ, {"DSV4_MOE_ROUTE_COMPACTOR": "invalid"}, clear=True
        ):
            with self.assertRaises(ValueError):
                enabled()

    def test_hot_path_has_no_global_sort_or_index_select(self):
        source = _KERNEL_PATH.read_text()
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("argsort", called_attributes)
        self.assertIn("_count_routes_kernel", source)
        self.assertIn("_prefix_routes_kernel", source)
        self.assertIn("_assign_routes_kernel", source)
        self.assertIn("tl.atomic_add", source)

    def test_wrapper_keeps_eager_rollback(self):
        source = _WRAPPER_PATH.read_text()
        fused_call = source.index("compact_mxfp4_routes_nopad_triton")
        eager_sort = source.index("torch.argsort")
        self.assertLess(fused_call, eager_sort)
        self.assertIn("DSV4_MOE_ROUTE_COMPACTOR", source)


if __name__ == "__main__":
    unittest.main()

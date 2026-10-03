"""CPU regression gate for the Prefill router's widened pointer offsets."""

import ast
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "models_py/triton_kernels/minimax_m31_prefill_router.py"
)


class PrefillRouterAddressTest(unittest.TestCase):
    def test_indices_are_widened_before_pointer_arithmetic(self):
        tree = ast.parse(SOURCE.read_text())
        for function, indices in (
            ("_prefill_router_partials", ("rows", "split")),
            ("_prefill_router_reduce", ("elements", "splits")),
        ):
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == function
            )
            assignments = {
                n.targets[0].id: n.value
                for n in node.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            }
            for index in indices:
                with self.subTest(function=function, index=index):
                    value = assignments[index]
                    self.assertIsInstance(value, ast.Call)
                    self.assertEqual(ast.unparse(value.func).split(".")[-1], "to")
                    self.assertEqual(
                        [ast.unparse(arg) for arg in value.args], ["tl.int64"]
                    )

    def test_hidden_stride_overflow_boundary(self):
        maximum = (1 << 31) - 1
        self.assertLessEqual(349524 * 6144 + 6143, maximum)
        self.assertEqual(349525 * 6144 + 2047, maximum)
        self.assertEqual(349525 * 6144 + 2048, maximum + 1)
        # At the supported 1M-token limit both input and reduction offsets
        # remain representable in INT64, including the eighth partial plane.
        for rows in (349525, 349526, 1048576):
            for offset in ((rows - 1) * 6144 + 6143, 8 * rows * 128 - 1):
                self.assertGreaterEqual(offset, 0)
                self.assertLess(offset, 1 << 63)


if __name__ == "__main__":
    unittest.main()

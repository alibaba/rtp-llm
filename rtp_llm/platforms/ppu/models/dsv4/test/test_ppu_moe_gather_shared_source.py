import ast
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[5]
GATHER = PACKAGE_ROOT / "platforms/ppu/kernels/ppu_moe_exact_gather.py"
TP_MOE = PACKAGE_ROOT / "platforms/ppu/models/dsv4/ppu_tp_moe.py"


class GatherSharedFusionSourceTest(unittest.TestCase):
    def test_kernel_keeps_bf16_rounding_before_shared_add(self):
        source = GATHER.read_text()
        self.assertIn("(accumulated * ROUTE_SCALE).to(tl.bfloat16)", source)
        self.assertIn("routed + shared", source)
        self.assertIn("ADD_SHARED=True", source)

    def test_tp_moe_has_default_on_rollback_switch(self):
        tree = ast.parse(TP_MOE.read_text())
        constants = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn("DSV4_MOE_GATHER_SHARED_FUSED", constants)
        source = TP_MOE.read_text()
        self.assertIn("if self._fused_gather_shared:", source)
        self.assertIn("combine_tp_partials(routed, shared, self.route_scale)", source)


if __name__ == "__main__":
    unittest.main()

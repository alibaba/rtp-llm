import ast
import importlib.util
import unittest
from pathlib import Path

COMMON_PATH = Path(__file__).resolve().parents[1] / "common.py"


def _source_tree():
    return ast.parse(COMMON_PATH.read_text(encoding="utf-8"), filename=str(COMMON_PATH))


class TestCommonWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        compile(_source_tree(), str(COMMON_PATH), "exec")

    def test_module_is_ascend_only(self):
        source = COMMON_PATH.read_text(encoding="utf-8")
        self.assertNotIn("triton", source)
        self.assertNotIn("_is_npu_tensor", source)


try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestCommonTorchReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ascendc_common_test", COMMON_PATH
        )
        cls.common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.common)

    def test_fused_gdn_gating_matches_torch_formula(self):
        A_log = torch.randn(4, dtype=torch.bfloat16)
        a = torch.randn(3, 4, dtype=torch.bfloat16)
        b = torch.randn(3, 4, dtype=torch.bfloat16)
        dt_bias = torch.randn(4, dtype=torch.bfloat16)
        beta = 0.75
        threshold = 12.0

        actual_g, actual_beta = self.common.fused_gdn_gating(
            A_log, a, b, dt_bias, beta=beta, threshold=threshold
        )

        expected_g = -torch.exp(A_log.float()) * F.softplus(
            a.float() + dt_bias.float(), beta=beta, threshold=threshold
        )
        expected_beta = torch.sigmoid(b.float()).to(b.dtype)
        torch.testing.assert_close(actual_g, expected_g.unsqueeze(0))
        torch.testing.assert_close(actual_beta, expected_beta.unsqueeze(0))
        self.assertEqual(actual_g.dtype, torch.float32)
        self.assertEqual(actual_beta.dtype, b.dtype)

    def test_grouped_rms_norm_then_silu_gate(self):
        x = torch.randn(2, 8, dtype=torch.float32)
        gate = torch.randn_like(x)
        weight = torch.randn(8, dtype=torch.float32)
        bias = torch.randn(8, dtype=torch.float32)
        module = self.common.RmsNormGated(
            weight, bias=bias, group_size=4, eps=1e-5, activation="silu"
        )

        actual = module(x, gate)

        grouped_x = x.float().reshape(2, 2, 4)
        variance = grouped_x.pow(2).mean(dim=-1, keepdim=True)
        expected = grouped_x * torch.rsqrt(variance + 1e-5)
        expected = expected.reshape_as(x) * weight.float() + bias.float()
        expected = (expected * F.silu(gate.float())).to(x.dtype)
        torch.testing.assert_close(actual, expected)

    def test_grouped_rms_norm_supports_sigmoid_gate(self):
        x = torch.randn(2, 6, dtype=torch.float32)
        gate = torch.randn_like(x)
        weight = torch.randn(6, dtype=torch.float32)
        module = self.common.RmsNormGated(weight, group_size=3, activation="sigmoid")

        actual = module(x, gate)

        grouped_x = x.float().reshape(2, 2, 3)
        variance = grouped_x.pow(2).mean(dim=-1, keepdim=True)
        expected = grouped_x * torch.rsqrt(variance + module.eps)
        expected = expected.reshape_as(x) * weight.float()
        expected = (expected * torch.sigmoid(gate.float())).to(x.dtype)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()

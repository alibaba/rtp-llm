import ast
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "linear_attention.py"


class TestLinearAttentionWithoutRuntimeDependencies(unittest.TestCase):
    def test_module_is_valid_python(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        compile(source, str(MODULE_PATH), "exec")

    def test_accelerator_dependencies_are_lazy(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        eager_imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for node in eager_imports:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            else:
                names.append(node.module or "")
        self.assertFalse(any("triton" in name or "fla_npu" in name for name in names))

    def test_head_first_uses_not_implemented_error(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        chunk_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "chunk_gated_delta_rule"
        )
        raised_errors = [
            node.exc.func.id
            for node in ast.walk(chunk_function)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
        ]
        self.assertIn("NotImplementedError", raised_errors)
        self.assertNotIn("DeprecationWarning", raised_errors)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
class TestLinearAttentionTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ascend_linear_attention", MODULE_PATH
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_torch_l2norm_fallback_matches_rtp_formula(self):
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        with mock.patch.object(self.module, "_get_npu_l2norm", return_value=None):
            actual = self.module.l2norm_fwd(x, eps=1e-6)
        expected = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(actual, expected)

    def test_kkt_adapts_layout_and_expands_gqa_heads(self):
        class FakeOps:
            calls = None

            def __init__(self):
                self.calls = []

            def npu_chunk_scaled_dot_kkt(self, **kwargs):
                self.calls.append(kwargs)
                batch, heads, seqlen, _ = kwargs["k"].shape
                offset = kwargs["beta"][0, 0, 0]
                values = (torch.arange(heads, dtype=torch.float32) * 10 + offset).view(
                    1, heads, 1, 1
                )
                return values.expand(batch, heads, seqlen, kwargs["chunk_size"]).clone()

        fake = FakeOps()
        k = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
        beta = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        g = beta + 100
        cu = torch.tensor([0, 4], dtype=torch.int32)
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            out = self.module.chunk_scaled_dot_kkt_fwd(
                k, beta, g_cumsum=g, cu_seqlens=cu
            )

        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(
            all(tuple(call["k"].shape) == (1, 2, 4, 8) for call in fake.calls)
        )
        self.assertTrue(
            all(tuple(call["beta"].shape) == (1, 2, 4) for call in fake.calls)
        )
        torch.testing.assert_close(fake.calls[0]["beta"], beta.transpose(1, 2)[:, 0::2])
        torch.testing.assert_close(fake.calls[1]["beta"], beta.transpose(1, 2)[:, 1::2])
        self.assertEqual(tuple(out.shape), (1, 4, 4, 64))
        self.assertFalse(torch.equal(out[:, :, 0], out[:, :, 1]))
        self.assertFalse(torch.equal(out[:, :, 2], out[:, :, 3]))

    def test_solve_tri_uses_fp16_npu_input(self):
        class FakeOps:
            dtype = None

            def npu_solve_tri(self, **kwargs):
                self.dtype = kwargs["x"].dtype
                return kwargs["x"]

        fake = FakeOps()
        A = torch.randn(1, 4, 2, 16, dtype=torch.float32)
        with mock.patch.object(self.module, "_get_ascendc_ops", return_value=fake):
            out = self.module.solve_tril(A, output_dtype=torch.float32)
        self.assertEqual(fake.dtype, torch.float16)
        self.assertEqual(out.dtype, torch.float32)

    def test_chunk_interface_rejects_head_first_layout(self):
        q = torch.zeros((1, 1, 1, 1), dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "head_first=True"):
            self.module.chunk_gated_delta_rule(q, q, q, q, q, head_first=True)


if __name__ == "__main__":
    unittest.main()

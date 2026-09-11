"""Q-only CMP RoPE byte parity; GPU cases require an explicitly reserved GPU."""

import ast
import importlib.util
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_kernel(name):
    for root in Path(__file__).resolve().parents:
        path = root / "rtp_llm/models_py/triton_kernels/sparse_mla" / f"{name}.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(name)


class SplitQContractCpuTest(unittest.TestCase):
    def test_requires_caller_outputs_and_has_no_allocation_or_host_reads(self):
        module = load_kernel("glm5_cmp_q_rope")
        signature = inspect.signature(module.split_q_rope)
        for name in ("q_nope_out", "q_out", "is_neox_style"):
            self.assertEqual(
                signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
            )
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)
        tree = ast.parse(inspect.getsource(module.split_q_rope))
        forbidden = {
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "clone",
            "item",
            "tolist",
            "cpu",
            "synchronize",
        }
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden
                for node in ast.walk(tree)
            )
        )


class SplitQRoPEGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("An explicitly reserved CUDA GPU is required")
        cls.module = load_kernel("glm5_cmp_q_rope")
        cls.baseline = load_kernel("fused_qk_rope_cat_cache_mla")

    def make(self, rows, heads, dtype=torch.int32):
        generator = torch.Generator(device="cuda").manual_seed(739 + rows + heads)
        projected = torch.randn(
            (rows, heads * 256), device="cuda", generator=generator
        ).bfloat16()
        projected.view(rows, heads, 256)[..., 0] = -0.0
        angles = torch.randn((257, 32), device="cuda", generator=generator)
        cosine = torch.cat((angles.cos(), angles.sin()), dim=1)
        positions = torch.randint(
            0, 257, (rows,), device="cuda", dtype=dtype, generator=generator
        )
        nope = torch.full((rows, heads, 192), 37.0, device="cuda", dtype=torch.bfloat16)
        query = torch.full(
            (rows, heads, 576), -19.0, device="cuda", dtype=torch.bfloat16
        )
        return projected, cosine, positions, nope, query

    def run_split(self, inputs, neox):
        projected, cosine, positions, nope, query = inputs
        outputs = self.module.split_q_rope(
            projected,
            cosine,
            positions,
            q_nope_out=nope,
            q_out=query,
            is_neox_style=neox,
        )
        self.assertIs(outputs[0], nope)
        self.assertIs(outputs[1], query)

    def assert_bytes(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def assert_baseline(self, inputs, neox):
        projected, cosine, positions, nope, query = inputs
        rows, heads = nope.shape[:2]
        reference = projected.view(rows, heads, 256).clone()
        self.baseline.fused_qk_rope_cat_cache_mla(
            reference,
            torch.zeros((rows, 512), device="cuda", dtype=torch.bfloat16),
            torch.zeros((rows, 64), device="cuda", dtype=torch.bfloat16),
            torch.empty((1, 64, 576), device="cuda", dtype=torch.bfloat16),
            torch.full((rows,), -1, device="cuda", dtype=torch.int64),
            positions,
            cosine,
            kv_lora_rank=512,
            rope_head_dim=64,
            is_neox_style=neox,
            kv_cache_type="auto",
        )
        self.assert_bytes(nope, reference[..., :192])
        self.assert_bytes(query[..., 512:], reference[..., 192:])
        self.assertTrue(bool((query[..., :512] == -19).all()))

    def test_exact_eager_reference_both_styles_and_shapes(self):
        for rows in (1, 4, 6, 64, 256):
            for heads in (8, 64):
                for neox in (False, True):
                    with self.subTest(rows=rows, heads=heads, neox=neox):
                        inputs = self.make(rows, heads)
                        saved = [tensor.clone() for tensor in inputs[:3]]
                        self.run_split(inputs, neox)
                        self.assert_baseline(inputs, neox)
                        for actual, expected in zip(inputs[:3], saved):
                            self.assert_bytes(actual, expected)

    def test_graph_replay_reads_changed_projection_and_positions(self):
        for heads in (8, 64):
            for neox in (False, True):
                with self.subTest(heads=heads, neox=neox):
                    inputs = self.make(6, heads, torch.int64)
                    for _ in range(3):
                        self.run_split(inputs, neox)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self.run_split(inputs, neox)
                    for step in range(3):
                        inputs[0].mul_(-0.75)
                        inputs[2].copy_((inputs[2] + 31 + step).remainder(257))
                        inputs[3].fill_(37)
                        inputs[4].fill_(-19)
                        graph.replay()
                        self.assert_baseline(inputs, neox)

    def test_empty_rows_skip_launch_and_invalid_metadata_rejected(self):
        inputs = self.make(0, 8)
        with patch.object(self.module, "_split_q_rope_kernel") as kernel:
            self.run_split(inputs, False)
            kernel.__getitem__.assert_not_called()
        inputs = list(self.make(4, 8))
        for index, replacement in (
            (0, inputs[0].float()),
            (1, inputs[1].double()),
            (1, torch.empty((), device="cuda", dtype=torch.float32)),
            (1, inputs[1][:, ::2]),
            (2, inputs[2].float()),
            (2, inputs[2].cpu()),
            (2, torch.empty(8, device="cuda", dtype=torch.int32)[::2]),
            (3, inputs[3][:, :, :128]),
            (4, inputs[4].transpose(0, 1)),
        ):
            invalid = inputs.copy()
            invalid[index] = replacement
            with self.subTest(index=index, shape=replacement.shape):
                with self.assertRaises(ValueError):
                    self.run_split(invalid, False)


if __name__ == "__main__":
    unittest.main()

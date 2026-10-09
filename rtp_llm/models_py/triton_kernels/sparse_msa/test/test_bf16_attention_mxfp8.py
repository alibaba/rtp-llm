"""Bit comparison against the frozen pre-extraction MXFP8 epilogue.

The independent pre-change oracle lives in the adjacent fixtures directory.
No reference package or device is needed for the AST gate. GPU tests use existing RTP BF16
combine to supply the intermediate, including BF16 rounding of nonfinite data.
"""

import ast
import importlib.util
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode import nvfp4_q8_combine_mxfp8 as current
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    LOG2E_F32, _q8kv4_combine,
)


class Bf16AttentionMxfp8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_path = Path(__file__).parent / "fixtures" / "original_combine_mxfp8.py"

    def test_epilogue_ast_preserved(self):
        old_tree = ast.parse(self.reference_path.read_text())
        new_tree = ast.parse(Path(current.__file__).read_text())
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_combine_mxfp8")
        helper = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_store_mxfp8")
        start = next(i for i, n in enumerate(old.body)
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == "pair_bits")
        self.assertEqual(
            [ast.dump(n) for n in old.body[start:]],
            [ast.dump(n) for n in helper.body[2:]],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_original_fused_and_bf16_bridge_bytes(self):
        name = "rtp_llm.models_py.triton_kernels.sparse_msa.decode._mxfp8_original_reference"
        spec = importlib.util.spec_from_file_location(name, self.reference_path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for rows in (1, 3, 5, 20, 40, 100):
            for kind in ("zeros", "random", "ties", "nonfinite"):
                with self.subTest(rows=rows, kind=kind):
                    self.check_case(reference, rows, kind)

    def check_case(self, reference, rows, kind):
        torch.manual_seed(1008 + rows)
        partial = torch.randn(rows, 4, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
        if kind == "zeros":
            partial.zero_()
        elif kind == "ties":
            ties = torch.tensor([0., -0., 1.0625, 1.1875, -1.0625, -1.1875, 448., -448.],
                                device="cuda", dtype=torch.bfloat16)
            partial.copy_(ties.repeat(partial.numel() // ties.numel()).reshape_as(partial))
        elif kind == "nonfinite":
            partial[..., :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.],
                                            device="cuda", dtype=torch.bfloat16)
            partial[:, 0, :, 0] = float("nan")
        counts = torch.full((rows, 4), 1, device="cuda", dtype=torch.int32)
        lse = torch.zeros((rows, 4, 16, 16), device="cuda", dtype=torch.float32)
        bf16 = torch.empty((rows, 64, 128), device="cuda", dtype=torch.bfloat16)
        aligned_m = ((rows + 3) // 4 + 1) * 4
        outputs = [torch.empty_like(bf16, dtype=torch.uint8) for _ in range(3)]
        scales = [torch.empty_strided((rows, 64), (1, aligned_m), device="cuda", dtype=torch.int32)
                  for _ in range(3)]
        reference._combine_mxfp8[(rows, 4)](
            partial, lse, counts, outputs[0], scales[0], ALIGNED_M=aligned_m,
            LOG2E=LOG2E_F32, num_warps=4,
        )
        current._combine_mxfp8[(rows, 4)](
            partial, lse, counts, outputs[1], scales[1], ALIGNED_M=aligned_m,
            LOG2E=LOG2E_F32, num_warps=4,
        )
        _q8kv4_combine[(rows, 4)](
            partial, lse, counts, bf16, bf16.stride(0), bf16.stride(1),
            LOG2E=LOG2E_F32, num_warps=4,
        )
        current._quantize_bf16_attention_mxfp8[(rows, 4)](
            bf16, outputs[2], scales[2], ALIGNED_M=aligned_m, num_warps=4,
        )
        for index in (1, 2):
            self.assertTrue(torch.equal(outputs[0], outputs[index]), "FP8 bytes differ")
            self.assertTrue(torch.equal(scales[0], scales[index]), "packed scales differ")
        if kind == "random":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                current._quantize_bf16_attention_mxfp8[(rows, 4)](
                    bf16, outputs[2], scales[2], ALIGNED_M=aligned_m, num_warps=4,
                )
            pointers = (outputs[2].data_ptr(), scales[2].data_ptr())
            for fill in (0.0, 1.0625):
                partial.fill_(fill)
                bf16.fill_(fill)
                reference._combine_mxfp8[(rows, 4)](
                    partial, lse, counts, outputs[0], scales[0], ALIGNED_M=aligned_m,
                    LOG2E=LOG2E_F32, num_warps=4,
                )
                graph.replay()
                self.assertTrue(torch.equal(outputs[0], outputs[2]))
                self.assertTrue(torch.equal(scales[0], scales[2]))
                self.assertEqual(pointers, (outputs[2].data_ptr(), scales[2].data_ptr()))


if __name__ == "__main__":
    unittest.main()

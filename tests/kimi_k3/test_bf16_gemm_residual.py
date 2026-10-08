"""Native K3 residual semantics, immutable inputs, and actual Graph replay."""
import ctypes
import json
import unittest

import torch
from rtp_llm.ops.compute_ops import rtp_llm_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Bf16GemmResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = rtp_llm_ops.cublas_gemm_bf16_fp32_accum_add
        with open('/proc/self/maps') as maps:
            paths = [line.split()[-1] for line in maps if '/libcublas.so.' in line]
        cls.cublas = ctypes.CDLL(paths[0])
        cls.cublas.cublasGetMathMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        cls.cublas.cublasGetMathMode.restype = ctypes.c_int

    def math_mode(self, handle):
        value = ctypes.c_int()
        self.assertEqual(self.cublas.cublasGetMathMode(handle, ctypes.byref(value)), 0)
        return value.value

    def test_native_residual_strides_and_input_immutability(self):
        for rows in (1, 3, 7, 8, 9, 629):
            x = torch.ones(rows, 2, device='cuda', dtype=torch.bfloat16)
            w = torch.tensor([[1., 1. / 256]], device='cuda', dtype=torch.bfloat16).repeat(16, 1)
            residual = -torch.ones(rows, 32, device='cuda', dtype=torch.bfloat16)[:, ::2]
            originals = [t.clone() for t in (x, w, residual)]
            oracle = (x.double() @ w.double().T + residual.double()).bfloat16()
            separate = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(x, w) + residual
            self.assertFalse(torch.equal(separate, oracle))
            for layout in (w, w.T.contiguous().T):
                actual = self.op(x, layout, residual)
                self.assertTrue(torch.equal(actual, oracle))
                self.assertNotEqual(actual.data_ptr(), residual.data_ptr())
            for t, before in zip((x, w, residual), originals):
                self.assertTrue(torch.equal(t, before))

    def test_native_addmm_alignment_and_math_policy_isolation(self):
        torch.manual_seed(927)
        original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            for m, n, k in ((1, 16, 128), (1, 7168, 3584), (7, 7168, 3584), (629, 7168, 3584)):
                x = (torch.randn(m, k, device='cuda') * .125).bfloat16()
                w = (torch.randn(k, n, device='cuda') * .025).bfloat16().T
                residual = (torch.randn(m, n, device='cuda') * .01).bfloat16()
                oracle = x.double() @ w.double().T + residual.double()
                # Native BF16 addmm is the implementation contract. Its
                # rounding is backend/shape dependent; retain FP64 error as
                # evidence rather than assuming an unsupported single cast.
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                reference = residual.clone().addmm_(x, w.T)
                outputs = []
                for flag in (True, False):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
                    handle = torch.cuda.current_blas_handle()
                    before = self.math_mode(handle)
                    actual = self.op(x, w, residual)
                    self.assertEqual(self.math_mode(handle), before)
                    self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, flag)
                    self.assertTrue(torch.equal(actual, reference))
                    error = (actual.double() - oracle).abs()
                    print(json.dumps({"shape": [m, n, k], "policy": flag,
                                      "native_bitwise_equal": True,
                                      "max_abs_fp64": float(error.max()),
                                      "relative_l2_fp64": float(error.norm() / oracle.norm())}))
                    outputs.append(actual)
                self.assertTrue(torch.equal(*outputs))
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original

    def test_graph_replays_update_residual(self):
        x = torch.ones(3, 2, device='cuda', dtype=torch.bfloat16)
        w = torch.tensor([[1., 1. / 256]], device='cuda', dtype=torch.bfloat16).repeat(16, 1)
        residual = torch.zeros(3, 16, device='cuda', dtype=torch.bfloat16)
        self.op(x, w, residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.op(x, w, residual)
        for value in (-1., 0., .5, -1.):
            residual.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            expected = (x.double() @ w.double().T + residual.double()).bfloat16()
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.all(residual == value))

    def test_empty_shapes_and_invalid_residual(self):
        x = torch.empty(2, 0, device='cuda', dtype=torch.bfloat16)
        w = torch.empty(3, 0, device='cuda', dtype=torch.bfloat16)
        residual = torch.ones(2, 3, device='cuda', dtype=torch.bfloat16)
        self.assertTrue(torch.equal(self.op(x, w, residual), residual))
        self.assertEqual(self.op(x[:0], w, residual[:0]).shape, (0, 3))
        with self.assertRaisesRegex(RuntimeError, 'residual must be bfloat16'):
            self.op(x, w, residual.float())
        with self.assertRaisesRegex(RuntimeError, 'residual shape'):
            self.op(x, w, residual[:, :2])


if __name__ == '__main__':
    unittest.main()

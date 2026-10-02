"""K3 BF16 GEMM: FP32 reduction contract without global policy changes."""
import ctypes
import json
import unittest

import torch
import torch.nn.functional as F
from rtp_llm.ops.compute_ops import rtp_llm_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Bf16GemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = rtp_llm_ops.cublas_gemm_bf16_fp32_accum
        # Inspect the already-loaded library instead of selecting another ABI.
        with open('/proc/self/maps') as maps:
            paths = [line.split()[-1] for line in maps if '/libcublas.so.' in line]
        cls.cublas = ctypes.CDLL(paths[0])
        cls.cublas.cublasGetMathMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        cls.cublas.cublasGetMathMode.restype = ctypes.c_int

    def math_mode(self, handle):
        mode = ctypes.c_int()
        status = self.cublas.cublasGetMathMode(handle, ctypes.byref(mode))
        self.assertEqual(status, 0)
        return mode.value

    def test_exact_dyadic_strides_and_old_fp32_operator(self):
        torch.manual_seed(6491)
        xbase = (torch.randint(-2, 3, (9, 7168*2), device='cuda').float()/8).bfloat16()
        wbase = (torch.randint(-2, 3, (96, 7168*2), device='cuda').float()/8).bfloat16()
        x, w = xbase[:, ::2], wbase[:, ::2]
        oracle = x.double() @ w.double().T
        for layout in (w, w.contiguous(), w.T.contiguous().T):
            with self.subTest(stride=layout.stride()):
                actual = self.op(x, layout)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(actual, oracle.bfloat16()))
        old = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(x, w)
        self.assertEqual(old.dtype, torch.float32)
        self.assertTrue(torch.equal(old, oracle.float()))

    def test_fp64_forward_error_and_pytorch_policy_isolation(self):
        torch.manual_seed(6492)
        original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        reports = []
        try:
            for m, n, k in ((1,96,512), (7,192,7168), (117,96,1536), (629,64,512)):
                x = (torch.randn(m,k,device='cuda')*.125).bfloat16()
                w = (torch.randn(k,n,device='cuda')*.025).bfloat16().T
                oracle = x.double() @ w.double().T
                # FP32 dot-product forward-error bound plus final BF16 rounding.
                # Products of normal BF16 values fit exactly in FP32; these
                # generated inputs are far from overflow and underflow.
                unit32 = 2.**-24
                gamma = k*unit32/(1-k*unit32)
                bound = gamma*(x.double().abs() @ w.double().abs().T)
                unit_bf16 = 2.**-8
                bound = (1+unit_bf16)*bound + unit_bf16*oracle.abs()
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                reference = F.linear(x,w)
                outputs = []
                for flag in (True,False):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
                    handle = torch.cuda.current_blas_handle()
                    before = self.math_mode(handle)
                    actual = self.op(x,w)
                    self.assertEqual(self.math_mode(handle),before)
                    self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,flag)
                    delta = (actual.double()-oracle).abs()
                    self.assertTrue(torch.all(delta <= bound), f'FP32 accumulation bound exceeded for {m,n,k}')
                    outputs.append(actual)
                self.assertTrue(torch.equal(*outputs))
                reports.append({'shape':[m,n,k], 'max_abs_fp64':delta.max().item(), 'relative_l2_fp64':((actual.double()-oracle).norm()/oracle.norm()).item(), 'pytorch_fp32_reduction_bitwise':torch.equal(actual,reference), 'max_abs_pytorch':(actual.float()-reference.float()).abs().max().item()})
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original
        print(json.dumps(reports,indent=2),flush=True)

    def test_graph_replays_changed_inputs_and_weights(self):
        x = (torch.arange(7*512,device='cuda').reshape(7,512)%5-2).to(torch.bfloat16)/8
        w = (torch.arange(96*512,device='cuda').reshape(96,512)%7-3).to(torch.bfloat16)/8
        base_x, base_w = x.clone(), w.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.op(x,w)
        torch.cuda.current_stream().wait_stream(stream)
        before_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            handle = torch.cuda.current_blas_handle()
            before_mode = self.math_mode(handle)
            result = self.op(x,w)
            self.assertEqual(self.math_mode(handle),before_mode)
        for xs, ws in ((1.,1.), (.5,2.), (-1.,.5), (2.,-1.), (0.,1.), (1.,1.)):
            x.copy_(base_x*xs); w.copy_(base_w*ws)
            graph.replay()
            oracle = x.double()@w.double().T
            self.assertTrue(torch.equal(result,oracle.bfloat16()))
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,before_flag)

    def test_empty_and_zero_inner_dimension(self):
        for m, n, k in ((0,3,7), (3,0,7), (3,4,0)):
            x = torch.empty(m,k,device='cuda',dtype=torch.bfloat16)
            w = torch.empty(n,k,device='cuda',dtype=torch.bfloat16)
            out = self.op(x,w)
            self.assertEqual(tuple(out.shape),(m,n))
            if k == 0: self.assertTrue(torch.equal(out,torch.zeros_like(out)))

    def test_invalid_inputs_fail_explicitly(self):
        x = torch.zeros(2,4,device='cuda',dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError,'bfloat16'): self.op(x.float(),x)
        with self.assertRaisesRegex(RuntimeError,'inner dimensions'): self.op(x,x[:,:3])
        with self.assertRaisesRegex(RuntimeError,'CUDA'): self.op(x.cpu(),x.cpu())

    def test_k3_linear_dispatch_preserves_public_factory(self):
        from rtp_llm.models_py.modules.kimi_k3.attention import linear
        from rtp_llm.models_py.modules.kimi_k3.linear import KimiK3Bf16Linear, bf16_linear
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear

        torch.manual_seed(6493)
        weight = (torch.randint(-2,3,(16,32),device='cuda').float()/8).bfloat16()
        x = (torch.randint(-2,3,(3,2,16),device='cuda').float()/8).bfloat16()
        registry = tuple(LinearFactory._strategies)
        k3 = linear({'projection':weight},'projection')
        old = LinearFactory.create_linear_from_weights({'projection':weight},'projection')
        self.assertIs(type(k3),KimiK3Bf16Linear)
        self.assertIs(type(old),CudaF16Linear)
        self.assertEqual(registry,tuple(LinearFactory._strategies))
        self.assertEqual(k3.weight.data_ptr(),weight.data_ptr())
        expected = (x.double() @ weight.double()).bfloat16()
        self.assertTrue(torch.equal(k3(x),expected))
        self.assertTrue(torch.equal(bf16_linear(x.cpu(),weight.T.cpu()),expected.cpu()))
        self.assertEqual(tuple(k3(x[:0]).shape),(0,2,32))


if __name__ == '__main__':
    unittest.main(verbosity=2)

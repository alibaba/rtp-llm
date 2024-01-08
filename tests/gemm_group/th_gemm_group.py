from typing import List
import unittest
import os
import torch


def random_tensor(shape, dtype, device):
    return torch.randn(size = shape, dtype=dtype, device=device)

class TestGemm(unittest.TestCase):
    
    def setUp(self) -> None:
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/libth_transformer.so")
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

        self.group_gemm = torch.ops.group_gemm_ops.group_gemm
       
        torch.manual_seed(734876213)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        self.ms = [1, 2, 4, 1024, 2048]
        self.ns = [1024, 2048, 4096]
        self.ks = [8, 16, 32]
        self.bs = [1,2,4]

    def group_gemm_test_helper(self, compute_type, count:int, ms:List[int], ns:List[int], ks:List[int], rtol:float,
                               atol:float):

        As_ref = [random_tensor((m, k), dtype=compute_type, device="cuda") for m, k in zip(ms, ks)]
        Bs_ref = [random_tensor((k, n), dtype=compute_type, device="cuda") for k, n in zip(ks, ns)]
        Cs_ref = [torch.zeros((m, n), dtype=compute_type, device="cuda") for m, n in zip(ms, ns)]
        
        self.starter.record()
        FT_result = self.group_gemm(As_ref, Bs_ref)
        self.ender.record()
        torch.cuda.synchronize()
        curr_time = self.starter.elapsed_time(self.ender)
        print(f"group gemm time is {curr_time}")

        for i in range(count):
            Cs_ref[i] = torch.mm(As_ref[i], Bs_ref[i])
            torch.testing.assert_close(FT_result[i], Cs_ref[i], rtol=rtol, atol=atol, check_dtype=False)

    def test_fp16_group_gemm(self):
        for m, n, k, count in zip(self.ms, self.ns, self.ks, self.bs):      
            ms = [m]*count
            ns = [n]*count
            ks = [k]*count
            print(f"ms is {ms}")
            print(f"ns is {ns}")
            print(f"ks is {ks}")
            self.group_gemm_test_helper(torch.float16, count, ms, ns, ks, rtol=0.001, atol=0.002)


if __name__ == '__main__':
    unittest.main()

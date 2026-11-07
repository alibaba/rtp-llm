"""FastKernel1D direct launches must match the JIT path bitwise."""

import unittest

import torch
import triton

from rtp_llm.models_py.modules.dsv41._fast_launch import FastKernel1D
from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import _strided_q_rmsnorm_kernel
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_kv import _swa_kv_kernel


def _cache_size(jit_fn):
    return sum(len(dc[0]) for dc in jit_fn.device_caches.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFastLaunch(unittest.TestCase):
    def test_strided_q_rmsnorm_bitwise(self):
        with torch.inference_mode():
            fast = FastKernel1D(
                _strided_q_rmsnorm_kernel, num_warps=8, enable_fp_fusion=False
            )
            for rows in (1, 3, 16, 17, 1000, 126981):
                x = torch.randn(126981, 5120, device="cuda", dtype=torch.bfloat16)[
                    :rows
                ]
                w = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
                out_jit = torch.empty_like(x)
                _strided_q_rmsnorm_kernel[(rows,)](
                    x,
                    w,
                    out_jit,
                    x.stride(0),
                    D=5120,
                    EPS=1e-6,
                    BLOCK=8192,
                    num_warps=8,
                    enable_fp_fusion=False,
                )
                out_fast = torch.empty_like(x)
                fast(rows, x, w, out_fast, x.stride(0), 5120, 1e-6, 8192)
                self.assertTrue(
                    torch.equal(out_jit.view(torch.int16), out_fast.view(torch.int16))
                )

    def test_swa_kv_bitwise(self):
        with torch.inference_mode():
            fast = FastKernel1D(_swa_kv_kernel, num_warps=4, enable_fp_fusion=False)
            for rows in (1, 5, 16, 33, 2049):
                raw = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
                freqs_cis = torch.randn(rows, 32, device="cuda", dtype=torch.complex64)
                freqs = torch.view_as_real(freqs_cis.reshape(-1, 32))
                out_jit = torch.empty_like(raw)
                _swa_kv_kernel[(rows,)](
                    raw,
                    weight,
                    freqs,
                    out_jit,
                    raw.stride(0),
                    freqs.stride(0),
                    1e-6,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                out_fast = torch.empty_like(raw)
                fast(
                    rows,
                    raw,
                    weight,
                    freqs,
                    out_fast,
                    raw.stride(0),
                    freqs.stride(0),
                    1e-6,
                )
                self.assertTrue(
                    torch.equal(out_jit.view(torch.int16), out_fast.view(torch.int16))
                )

    def test_no_extra_compiles(self):
        # Direct launches must not grow the triton kernel cache: every call
        # after warmup reuses the warmed CompiledKernel.
        fast = FastKernel1D(_swa_kv_kernel, num_warps=4, enable_fp_fusion=False)
        with torch.inference_mode():
            for rows in (1, 16, 17):
                raw = torch.zeros((rows, 512), device="cuda", dtype=torch.bfloat16)
                weight = torch.zeros(512, device="cuda", dtype=torch.bfloat16)
                freqs = torch.zeros((rows, 32, 2), device="cuda", dtype=torch.float32)
                out = torch.empty_like(raw)
                fast(rows, raw, weight, freqs, out, raw.stride(0), freqs.stride(0), 1e-6)
            size = _cache_size(_swa_kv_kernel)
            for rows in (3, 48, 1001):
                raw = torch.zeros((rows, 512), device="cuda", dtype=torch.bfloat16)
                weight = torch.zeros(512, device="cuda", dtype=torch.bfloat16)
                freqs = torch.zeros((rows, 32, 2), device="cuda", dtype=torch.float32)
                out = torch.empty_like(raw)
                fast(rows, raw, weight, freqs, out, raw.stride(0), freqs.stride(0), 1e-6)
            self.assertEqual(size, _cache_size(_swa_kv_kernel))


if __name__ == "__main__":
    unittest.main()

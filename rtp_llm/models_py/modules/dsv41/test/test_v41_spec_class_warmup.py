"""The spec-class prewarm must leave nothing to compile in serving shapes."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import _strided_q_rmsnorm_kernel
from rtp_llm.models_py.modules.dsv41.dsv41_kernel_jit_warmup import (
    prewarm_v41_spec_class_jit,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_kv import _swa_kv_kernel


def _cache_size(jit_fn):
    return sum(len(dc[0]) for dc in jit_fn.device_caches.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSpecClassWarmup(unittest.TestCase):
    def test_no_compile_after_prewarm(self):
        device = torch.device("cuda", torch.cuda.current_device())
        prewarm_v41_spec_class_jit(device=device)
        sizes = {
            k: _cache_size(k)
            for k in (_strided_q_rmsnorm_kernel, _swa_kv_kernel)
        }
        self.assertTrue(all(v > 0 for v in sizes.values()))
        with torch.inference_mode():
            # Serving-like odd shapes and pool-sliced (unaligned) tensors.
            for rows in (3, 33, 1001):
                x = torch.randn(rows, 5120, device=device, dtype=torch.bfloat16)
                w = torch.randn(5120, device=device, dtype=torch.bfloat16)
                y = torch.empty_like(x)
                _strided_q_rmsnorm_kernel[(rows,)](
                    x,
                    w,
                    y,
                    x.stride(0),
                    D=5120,
                    EPS=1e-6,
                    BLOCK=8192,
                    num_warps=8,
                    enable_fp_fusion=False,
                )
                raw = torch.randn(rows, 512, device=device, dtype=torch.bfloat16)
                weight = torch.randn(512, device=device, dtype=torch.bfloat16)
                freqs = torch.randn(rows, 32, 2, device=device, dtype=torch.float32)
                out = torch.empty_like(raw)
                _swa_kv_kernel[(rows,)](
                    raw,
                    weight,
                    freqs,
                    out,
                    raw.stride(0),
                    freqs.stride(0),
                    1e-6,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        for k, v in sizes.items():
            self.assertEqual(v, _cache_size(k))


if __name__ == "__main__":
    unittest.main()

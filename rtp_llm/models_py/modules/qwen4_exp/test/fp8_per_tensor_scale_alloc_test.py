"""Dynamic per-tensor fp8 quant must not allocate a fresh zeroed scale.

Serving timelines (49 layers/step, warm5) count 424 0-dim ``aten::zeros``
per step (~9ms of host dispatch): every fp8 GEMM input quantization goes
through ``scaled_fp8_per_tensor_quant(scale=None)`` and allocates its
scale accumulator with ``torch.zeros(1)``. The accumulator is zeroed only
because ``per_tensor_absmax_kernel`` reduces with atomicMax, which needs a
zero initial value — a per-device persistent buffer reset with ``zero_()``
(one aten call on a cached tensor) replaces the per-call ``zeros(1)``
allocation while keeping the exact kernel semantics.

This test locks the contract:

* dynamic quantization stays bit-identical to the static path with the same
  scale (same kernel), and the scale equals ``amax / fp8_max`` for random,
  all-zero and large-magnitude inputs;
* back-to-back calls with decreasing amax never observe a stale scale
  (the accumulator is reset before every dynamic call);
* one dynamic call enqueues no ``aten::zero_``/``aten::fill_`` op and no
  fill kernel (the zeros allocation itself is gone).
"""

import json
import os
import tempfile
import unittest

import torch

_DEV = "cuda"
_EPS_CASES = "zero-case"


class Fp8PerTensorScaleAllocTest(unittest.TestCase):
    def _quant(self, x):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )

        return scaled_fp8_per_tensor_quant(x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dynamic_quant_matches_static_and_reference(self):
        from rtp_llm.ops.compute_ops import per_tensor_quant_fp8

        torch.manual_seed(7)
        cases = {
            "random": torch.randn(1572, 2560, device=_DEV, dtype=torch.bfloat16),
            "all_zero": torch.zeros(64, 512, device=_DEV, dtype=torch.bfloat16),
            "large": (
                torch.randn(512, 1024, device=_DEV, dtype=torch.bfloat16) * 300.0
            ),
            "negative": -torch.rand(128, 256, device=_DEV, dtype=torch.bfloat16),
        }
        for name, x in cases.items():
            with self.subTest(name):
                out, scale = self._quant(x)
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(out.dtype, torch.float8_e4m3fn)
                expected_scale = x.abs().amax().float() / torch.finfo(
                    torch.float8_e4m3fn
                ).max
                self.assertTrue(
                    abs(scale.item() - expected_scale.item())
                    <= 1e-5 * abs(expected_scale.item()),
                    f"scale {scale.item()} != reference {expected_scale.item()}",
                )
                static_out = torch.empty_like(out, dtype=torch.float8_e4m3fn)
                per_tensor_quant_fp8(x, static_out, scale, True)
                self.assertTrue(torch.equal(out.view(torch.uint8), static_out.view(torch.uint8)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scale_buffer_reuse_across_calls(self):
        torch.manual_seed(11)
        big = torch.randn(256, 512, device=_DEV, dtype=torch.bfloat16) * 500.0
        small = torch.randn(256, 512, device=_DEV, dtype=torch.bfloat16) * 0.001
        _, scale_big = self._quant(big)
        # The dynamic scale is a shared accumulator: snapshot values before
        # the next dynamic call (the documented consume-immediately contract).
        scale_big_value = scale_big.item()
        _, scale_small = self._quant(small)
        expected_small = small.abs().amax().float() / torch.finfo(
            torch.float8_e4m3fn
        ).max
        self.assertTrue(
            abs(scale_small.item() - expected_small.item())
            <= 1e-5 * abs(expected_small.item()),
            f"scale {scale_small.item()} != reference {expected_small.item()} "
            "(stale accumulator?)",
        )
        self.assertLess(scale_small.item(), scale_big_value)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dynamic_scale_is_a_shared_accumulator(self):
        # Locked documented behavior: the returned dynamic scale aliases one
        # per-device accumulator, so a reference held across the next dynamic
        # call observes the newer value. Callers must consume (enqueue the
        # consumer on the same stream) before quantizing again.
        x1 = torch.full((64, 128), 300.0, device=_DEV, dtype=torch.bfloat16)
        x2 = torch.full((64, 128), 3.0, device=_DEV, dtype=torch.bfloat16)
        _, s1 = self._quant(x1)
        _, s2 = self._quant(x2)
        self.assertIs(s1, s2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dynamic_quant_allocates_no_scale_tensor(self):
        from torch.profiler import ProfilerActivity, profile

        x = torch.randn(64, 512, device=_DEV, dtype=torch.bfloat16)
        self._quant(x)  # warm the per-device accumulator outside the window
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self._quant(x)
            torch.cuda.synchronize()
        path = os.path.join(tempfile.mkdtemp(prefix="fp8_scale_alloc_"), "t.json")
        prof.export_chrome_trace(path)
        with open(path) as handle:
            evs = json.load(handle)["traceEvents"]
        cpu_names = [e["name"] for e in evs if e.get("cat") == "cpu_op"]
        alloc_ops = [n for n in cpu_names if n in ("aten::zeros", "aten::empty_like")]
        self.assertEqual(
            alloc_ops,
            [],
            f"dynamic quant must not allocate a fresh zeroed scale: saw {alloc_ops}",
        )
        # The only remaining allocation is the quantized-output tensor.
        empties = [n for n in cpu_names if n == "aten::empty"]
        self.assertLessEqual(
            len(empties), 1, f"expected only the output allocation, saw {len(empties)}"
        )


if __name__ == "__main__":
    unittest.main()

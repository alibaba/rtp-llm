"""Manual CUDA-event microbenchmark for K3 MoonViT fused Q/K RoPE."""

import json
import math
import statistics
import unittest

import torch

from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
    maybe_fused_apply_rope,
)


def _eager_qk_rope(q, k, freqs):
    freqs = freqs[:, None, :]

    def rotate(x):
        pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * freqs).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


def _measure_us(fn, warmup=50, repeats=40, calls_per_repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(calls_per_repeat):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / calls_per_repeat)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_us": min(samples),
        "max_us": max(samples),
        "repeats": repeats,
        "calls_per_repeat": calls_per_repeat,
        "samples_us": samples,
    }


@unittest.skipUnless(torch.cuda.is_available(), "K3 RoPE benchmark requires CUDA")
class KimiK3RopeMicrobenchmark(unittest.TestCase):
    def test_fused_against_eager(self):
        torch.manual_seed(17)
        device_name = torch.cuda.get_device_name()
        for length in (256, 1024, 4096):
            with self.subTest(length=length):
                shape = (length, 12, 128)
                q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                angles = torch.randn(length, 64, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)

                fused = maybe_fused_apply_rope(q, k, freqs)
                self.assertIsNotNone(fused)
                eager = _eager_qk_rope(q, k, freqs)
                torch.testing.assert_close(fused[0], eager[0], rtol=0.02, atol=0.02)
                torch.testing.assert_close(fused[1], eager[1], rtol=0.02, atol=0.02)

                fused_time = _measure_us(lambda: maybe_fused_apply_rope(q, k, freqs))
                eager_time = _measure_us(lambda: _eager_qk_rope(q, k, freqs))
                print(
                    json.dumps(
                        {
                            "device": device_name,
                            "torch": torch.__version__,
                            "shape": shape,
                            "dtype": "bfloat16",
                            "fused": fused_time,
                            "eager": eager_time,
                            "median_speedup": (
                                eager_time["median_us"] / fused_time["median_us"]
                            ),
                            "median_time_reduction_pct": 100
                            * (
                                1
                                - fused_time["median_us"] / eager_time["median_us"]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()

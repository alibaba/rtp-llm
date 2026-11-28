"""The fused residual must use the input device's stream with a different current GPU."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _engram_inject_triton as native
from rtp_llm.models_py.modules.dsv41.engram import gated_engram_residual


class EngramLaunchDeviceTest(unittest.TestCase):
    @torch.inference_mode()
    def test_other_current_device_uses_input_device_stream_and_restores_caller(self):
        from triton.runtime import driver

        self.assertGreaterEqual(
            torch.cuda.device_count(), 2, "requires two reserved CUDA devices"
        )
        dev = torch.device("cuda", 1)
        actual_launches = []
        native_kernel = native.engram_inject_kernel
        stream = torch.cuda.Stream(device=dev)

        class CheckedLaunch:
            def __getitem__(self, grid):
                launch = native_kernel[grid]

                def run(*args, **kwargs):
                    current = driver.active.get_current_device()
                    stream_id = driver.active.get_current_stream(current)
                    actual_launches.append((current, stream_id))
                    # Detect the faulty dispatch before issuing an unsafe
                    # launch with pointers owned by the other context.
                    if current != dev.index or stream_id != stream.cuda_stream:
                        raise AssertionError(
                            f"wrong Triton launch device or stream: {current}, {stream_id}"
                        )
                    return launch(*args, **kwargs)

                return run

        with torch.cuda.stream(stream):
            rng = torch.Generator(device=dev).manual_seed(201)
            h, kv, q, k = [
                torch.randn(s, generator=rng, device=dev, dtype=torch.bfloat16)
                for s in ((32, 4, 5120), (32, 25600), (4, 5120), (4, 5120))
            ]
            mask = torch.arange(32, device=dev) % 3 != 0
            expected = gated_engram_residual(h.clone(), kv, q, k, 1e-20, mask)
            with torch.cuda.device(0), patch.object(
                native, "engram_inject_kernel", CheckedLaunch()
            ):
                ambient = torch.cuda.current_stream()
                out = gated_engram_residual(h, kv, q, k, 1e-20, mask, out=h)
                self.assertEqual(torch.cuda.current_device(), 0)
                self.assertEqual(torch.cuda.current_stream(), ambient)
                self.assertEqual(out.data_ptr(), h.data_ptr())
            self.assertTrue(torch.equal(out, expected))
        self.assertEqual(actual_launches, [(dev.index, stream.cuda_stream)])

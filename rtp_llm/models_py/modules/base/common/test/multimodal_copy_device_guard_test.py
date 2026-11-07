from unittest import TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class MultimodalCopyDeviceGuardTest(TestCase):
    def setUp(self) -> None:
        # The dedicated target requests two locked GPUs. Fail on resource
        # misconfiguration instead of silently skipping the DeviceGuard check.
        self.assertGreaterEqual(torch.cuda.device_count(), 2)

    def test_copy_on_noncurrent_device_preserves_device_and_stream(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.cuda.device(0):
                target_stream = torch.cuda.Stream(device=1)
                with torch.cuda.stream(target_stream):
                    dst = torch.zeros((131, 8), device="cuda:1", dtype=dtype)
                    srcs = [
                        torch.full((1, 8), i + 1, device="cuda:1", dtype=dtype)
                        for i in range(65)
                    ]
                    offsets = list(range(1, 130, 2))
                    expected = dst.clone()
                    for src, offset in zip(srcs, offsets):
                        expected[offset : offset + 1] = src

                    torch.cuda.set_device(0)
                    caller_stream = torch.cuda.current_stream(0)
                    rtp_llm_ops.fused_multimodal_copy_(dst, srcs, offsets)
                    self.assertEqual(torch.cuda.current_device(), 0)
                    self.assertEqual(torch.cuda.current_stream(0), caller_stream)
                    self.assertEqual(torch.cuda.current_stream(1), target_stream)
                    target_stream.synchronize()
                    torch.testing.assert_close(dst, expected, rtol=0, atol=0)

    def test_validation_failure_restores_device(self):
        with torch.cuda.device(0):
            dst = torch.zeros((4, 8), device="cuda:1", dtype=torch.bfloat16)
            src = torch.ones((1, 8), device="cuda:0", dtype=torch.bfloat16)
            self.assertEqual(torch.cuda.current_device(), 0)
            with self.assertRaisesRegex(RuntimeError, "must be on dst device"):
                rtp_llm_ops.fused_multimodal_copy_(dst, [src], [0])
            self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == "__main__":
    main()

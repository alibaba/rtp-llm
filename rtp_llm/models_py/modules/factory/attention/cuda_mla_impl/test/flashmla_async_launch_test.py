"""Catch device-wide barriers that defeat cross-layer history prefetch."""

import json
import os
import unittest
from pathlib import Path

import torch
from flash_mla import flash_mla_sparse_fwd


class FlashMlaAsyncLaunchTest(unittest.TestCase):
    def test_attention_does_not_wait_for_an_independent_stream(self):
        torch.manual_seed(1729)
        q = torch.randn(13920, 64, 576, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(1553664, 1, 576, device="cuda", dtype=torch.bfloat16)
        indices = torch.randint(
            1553664, (13920, 1, 2048), device="cuda", dtype=torch.int32
        )

        def attend():
            return flash_mla_sparse_fwd(q, kv, indices, 256**-0.5, d_v=512)[0]

        # Warm both kernel initialization and allocator pools before concurrency.
        for _ in range(3):
            reference = attend()
        producer = torch.cuda.Stream()
        producer_end = torch.cuda.Event(enable_timing=True)
        attention_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.cuda.stream(producer):
                # One SM, independent of Attention inputs. A warmed Attention
                # must finish before this deliberately much longer GPU task.
                torch.cuda._sleep(500_000_000)
                producer_end.record()
            actual = attend()
            attention_end.record()
            torch.cuda.synchronize()
        margin_ms = attention_end.elapsed_time(producer_end)
        output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
        output.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(output / "flashmla_async_launch.json"))
        (output / "flashmla_async_launch_result.json").write_text(
            json.dumps({"attention_before_producer_end_ms": margin_ms}, indent=2)
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertGreater(margin_ms, 0, "FlashMLA synchronized an unrelated stream")


if __name__ == "__main__":
    unittest.main()

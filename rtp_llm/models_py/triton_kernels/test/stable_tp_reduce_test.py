"""GPU precision guard for rank-ordered BF16 transport and FP32 summation."""

import json
import os
import statistics
import time
import unittest
from pathlib import Path

import torch
from rtp_llm.models_py.triton_kernels.common.stable_tp_reduce import (
    _sum_rank_order,
    sum_bf16_rank_order,
)


def reference_jit_sum(parts, output):
    _sum_rank_order[((output.numel() + 1023) // 1024,)](
        parts,
        output,
        output.numel(),
        parts.shape[0],
        1024,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )


class StableTpReduceTest(unittest.TestCase):
    def test_compiled_submission(self):
        records = []
        for size in (129, 81 * 6144, 419 * 6144, 9889 * 6144):
            parts = torch.randn(4, size, device="cuda", dtype=torch.bfloat16)
            expected, output = torch.empty_like(parts[0]), torch.empty_like(parts[0])
            reference_jit_sum(parts, expected)
            sum_bf16_rank_order(parts, output)
            torch.testing.assert_close(
                output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            row = {"elements": size}
            for name, fn in (
                ("jit", reference_jit_sum),
                ("compiled", sum_bf16_rank_order),
            ):
                for _ in range(20):
                    fn(parts, output)
                samples = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    for _ in range(100):
                        fn(parts, output)
                    samples.append((time.perf_counter() - before) * 1e4)
                torch.cuda.synchronize()
                row[name + "_cpu_us"] = statistics.median(samples)
            records.append(row)
        # New pointers and a nondefault stream must be used by every invocation.
        for device in range(min(torch.cuda.device_count(), 2)):
            with torch.cuda.device(device):
                parts = torch.randn(4, 8193, device="cuda", dtype=torch.bfloat16)
                expected, output = torch.empty_like(parts[0]), torch.empty_like(
                    parts[0]
                )
                reference_jit_sum(parts, expected)
                sum_bf16_rank_order(parts, output)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    parts = parts.clone()
                    output = torch.empty_like(output)
                    torch.cuda._sleep(50_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    sum_bf16_rank_order(parts, output)
                    self.assertFalse(pending.query())
                stream.synchronize()
                torch.testing.assert_close(
                    output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
                # Contiguous BF16 views can still have unaligned addresses.
                unaligned = torch.empty(
                    4 * 8193 + 1, device="cuda", dtype=torch.bfloat16
                )[1:].view(4, 8193)
                unaligned.copy_(parts)
                sum_bf16_rank_order(unaligned, output)
                torch.testing.assert_close(
                    output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "stable_tp_submission.json"
        ).write_text(json.dumps(records, indent=2))

    def test_graph_capture_and_wide_rank_stride(self):
        parts = torch.randn(4, 8193, device="cuda", dtype=torch.bfloat16)
        expected, output = torch.empty_like(parts[0]), torch.empty_like(parts[0])
        reference_jit_sum(parts, expected)
        sum_bf16_rank_order(parts, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sum_bf16_rank_order(parts, output)
        graph.replay()
        torch.testing.assert_close(
            output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
        # Only one block is launched, but its second rank is beyond INT32_MAX.
        size = 2**31 + 128
        parts = torch.empty(2, size, device="cuda", dtype=torch.bfloat16)
        parts[0, :1024].fill_(1)
        parts[1, :1024].fill_(3)
        output = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
        kernel = _sum_rank_order.warmup(
            torch.bfloat16,
            torch.bfloat16,
            size,
            2,
            1024,
            grid=(1, 1, 1),
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
        kernel[(1, 1, 1)](parts, output, size, 2, 1024)
        self.assertTrue(torch.equal(output, torch.full_like(output, 4)))

    def test_bf16_subnormal_and_extreme_values(self):
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        parts = torch.stack((values, values.roll(1), -values, values.roll(3)))
        expected = torch.zeros_like(values, dtype=torch.float32)
        for part in parts:
            expected.add_(part)
        output = torch.empty_like(values)
        sum_bf16_rank_order(parts, output)
        finite = torch.isfinite(expected)
        mismatch = finite & (
            output.view(torch.int16) != expected.bfloat16().view(torch.int16)
        )
        if mismatch.any():
            print(
                "BF16 boundary mismatch",
                {
                    "actual_bits": output[mismatch].view(torch.int16)[:8].tolist(),
                    "expected_bits": expected.bfloat16()[mismatch]
                    .view(torch.int16)[:8]
                    .tolist(),
                    "input_bits": parts[:, mismatch].view(torch.int16)[:, :8].tolist(),
                },
            )
        torch.testing.assert_close(
            output[finite].view(torch.int16),
            expected.bfloat16()[finite].view(torch.int16),
            rtol=0,
            atol=0,
        )

    def test_exact_rank_order_and_cancellation(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.manual_seed(531012)
        for world in (1, 2, 4, 8):
            for rows in (1, 509, 16381):
                parts = torch.randn(
                    world, rows, 129, device="cuda", dtype=torch.bfloat16
                )
                pattern = (1.0, 16777216.0, -16777216.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                for rank in range(world):
                    parts[rank, 0, 0] = pattern[rank]
                    parts[rank, 0, 1] = -0.0
                expected = torch.zeros(rows, 129, device="cuda", dtype=torch.float32)
                for part in parts:
                    expected.add_(part)
                expected = expected.bfloat16()
                output = torch.empty_like(expected)
                sum_bf16_rank_order(parts, output)
                torch.testing.assert_close(
                    output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )
                if world >= 4:
                    self.assertEqual(output[0, 0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()

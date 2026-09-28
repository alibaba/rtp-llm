"""Real-GPU DFlash2 convolution correctness, graph replay and timing tests."""

import json
import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.dflash2_conv import grouped_conv


def reference_grouped_conv(hidden, delta, base, width, group_size):
    """Independent dense block oracle, with no flattened-row addressing."""
    batch, channels = hidden.shape[0] // width, hidden.shape[1]
    x = hidden.float().reshape(batch, width, channels)
    coefficients = delta.float().repeat_interleave(group_size, dim=-1)
    coefficients = coefficients.reshape(batch, width, base.shape[0], channels)
    result = torch.zeros_like(x)
    for tap in range(min(base.shape[0], width)):
        result[:, tap:] += (coefficients[:, tap:, tap] + base[tap].float()) * x[
            :, : width - tap
        ]
    return result.reshape_as(hidden).to(hidden.dtype)


class DFlash2ConvGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These targets are explicit hardware tests. A missing accelerator must
        # never be reported as a passing (entirely skipped) platform result.
        if not torch.cuda.is_available():
            raise RuntimeError("DFlash2 convolution GPU tests require CUDA or ROCm")

    def test_empty_rows_and_out_alias_contract(self):
        base = torch.ones(2, 16, device="cuda")
        empty = grouped_conv(
            torch.empty(0, 16, device="cuda"),
            torch.empty(0, 2, 1, device="cuda"),
            base,
            8,
            16,
        )
        self.assertEqual(empty.shape, (0, 16))
        hidden = torch.randn(8, 16, device="cuda")
        delta = torch.ones(8, 2, 1, device="cuda")
        with self.assertRaisesRegex(ValueError, "alias"):
            grouped_conv(hidden, delta, base, 8, 16, out=hidden)

    def test_math_and_strided_coefficients(self):
        for batch, width, channels, groups, taps in (
            (1, 1, 80, 16, 2),
            (3, 2, 80, 16, 4),
            (2, 3, 512, 16, 2),
            (3, 8, 5120, 16, 2),
            (2, 7, 96, 3, 3),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(
                    shape=(batch, width, channels, groups, taps), dtype=dtype
                ):
                    hidden = torch.randn(
                        batch * width, channels, device="cuda", dtype=dtype
                    )
                    # Production coefficients[:, side] are not row-contiguous.
                    delta = torch.randn(
                        batch * width,
                        2,
                        taps,
                        channels // groups,
                        device="cuda",
                        dtype=dtype,
                    )[:, 1]
                    base = torch.randn(taps, channels, device="cuda", dtype=dtype)
                    expected = reference_grouped_conv(
                        hidden, delta, base, width, groups
                    )
                    actual = grouped_conv(hidden, delta, base, width, groups)
                    error = (actual.float() - expected.float()).abs()
                    print(
                        json.dumps(
                            {
                                "check": "dflash2_conv_fp32_accumulation",
                                "shape": [batch, width, channels, groups, taps],
                                "dtype": str(dtype),
                                "max_abs_error": error.max().item(),
                                "mean_abs_error": error.mean().item(),
                            }
                        ),
                        flush=True,
                    )
                    # Both paths use FP32 operations and a single output cast.
                    # BF16 permits one output ULP if FP32 rounding straddles a
                    # representable-value midpoint; no token determinism gate.
                    tolerance = 8e-3 if dtype == torch.bfloat16 else 2e-6
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )

    def test_no_cross_request_reads(self):
        width, channels = 3, 16
        hidden = torch.arange(6 * channels, device="cuda", dtype=torch.float32).reshape(
            6, channels
        )
        delta = torch.zeros(6, 2, 1, device="cuda")
        base = torch.stack(
            (torch.zeros(channels, device="cuda"), torch.ones(channels, device="cuda"))
        )
        actual = grouped_conv(hidden, delta, base, width, channels)
        self.assertEqual(actual[0].abs().sum().item(), 0)
        self.assertEqual(actual[width].abs().sum().item(), 0)
        torch.testing.assert_close(actual[1:width], hidden[: width - 1])
        torch.testing.assert_close(actual[width + 1 :], hidden[width:-1])

    def test_graph_replay_reads_updated_inputs(self):
        width, channels = 8, 512
        hidden = torch.randn(3 * width, channels, device="cuda", dtype=torch.bfloat16)
        delta = torch.randn(
            3 * width, 2, 2, channels // 16, device="cuda", dtype=hidden.dtype
        )[:, 0]
        base = torch.randn(2, channels, device="cuda", dtype=hidden.dtype)
        out = torch.empty_like(hidden)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                grouped_conv(hidden, delta, base, width, 16, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            grouped_conv(hidden, delta, base, width, 16, out=out)
        hidden.normal_()
        delta.normal_()
        graph.replay()
        torch.testing.assert_close(
            out, reference_grouped_conv(hidden, delta, base, width, 16)
        )

    def test_performance(self):
        # Timings are evidence, not an architecture-dependent speed threshold.
        for batch in (1, 8, 32):
            hidden = torch.randn(batch * 8, 5120, device="cuda", dtype=torch.bfloat16)
            delta = torch.randn(batch * 8, 2, 320, device="cuda", dtype=hidden.dtype)
            base = torch.randn(2, 5120, device="cuda", dtype=hidden.dtype)
            out = torch.empty_like(hidden)
            for _ in range(10):
                grouped_conv(hidden, delta, base, 8, 16, out=out)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(100):
                grouped_conv(hidden, delta, base, 8, 16, out=out)
            end.record()
            end.synchronize()
            print(
                json.dumps(
                    {
                        "kernel": "dflash2_grouped_conv",
                        "batch": batch,
                        "gpu": torch.cuda.get_device_name(),
                        "milliseconds": start.elapsed_time(end) / 100,
                        "backend": "rocm" if torch.version.hip else "cuda",
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()

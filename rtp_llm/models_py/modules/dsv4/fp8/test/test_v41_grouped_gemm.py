"""Focused dispatch, ragged numerical, stream and graph fallback contracts."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


def load():
    path = Path(__file__).resolve().parents[1] / "_v41_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("grouped_index_contract", path)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    return op


op = load()


class HostContractTest(unittest.TestCase):
    def test_import_and_cpu_fallback_have_no_cuda_or_compiler_side_effect(self):
        with mock.patch.object(torch.cuda, "_lazy_init") as init, mock.patch.object(
            op.ctypes, "CDLL"
        ) as cdll, mock.patch.object(op.subprocess, "run") as compiler:
            fresh = load()
            x, w = torch.empty(2, 512), torch.empty(128, 512)
            self.assertIsNone(fresh.try_grouped_index_gemm(x, w, (2,)))
            self.assertFalse(fresh.warmup(w))
        init.assert_not_called()
        cdll.assert_not_called()
        compiler.assert_not_called()

    def test_host_rows_are_bounded_and_never_coerce_device_metadata(self):
        self.assertTrue(op._rows_supported((0, 1, 7, 0), 8))
        self.assertTrue(op._rows_supported((), 0))
        self.assertTrue(op._rows_supported((65536,), 65536))
        for rows, total in (
            ([True], 1),
            ([1.0], 1),
            ([-1, 2], 1),
            ([1], 2),
            ([65537], 65537),
            ([0] * 1025, 0),
            (torch.tensor([1]), 1),
        ):
            with self.subTest(rows=type(rows), total=total):
                self.assertFalse(op._rows_supported(rows, total))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA13 SM100")
class CUDAContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (10, 0):
            raise unittest.SkipTest("requires SM100")
        cls.env = mock.patch.dict(os.environ, {"DSV41_GROUPED_INDEX_GEMM": "1"})
        cls.env.start()
        torch.manual_seed(1931)
        cls.weight = torch.randn(128, 512, device="cuda").bfloat16()
        if not op.warmup(cls.weight):
            cls.env.stop()
            raise AssertionError("grouped index JIT did not warm up")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def reference(self, x, rows):
        return torch.cat([F.linear(part, self.weight) for part in x.split(rows)])

    def test_ragged_random_and_boundary_inputs_are_bitexact(self):
        for rows in (
            (0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 0),
            tuple(8 * i + 1 for i in range(32)),
            (256,) * 256,
            (0,) * 1024,
        ):
            with self.subTest(segments=len(rows), total=sum(rows)):
                x = torch.randn(sum(rows), 512, device="cuda").bfloat16()
                expected = self.reference(x, rows)
                actual = op.try_grouped_index_gemm(x, self.weight, rows)
                self.assertIsNotNone(actual)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )

    def test_output_validation_cold_disabled_and_unsupported(self):
        x = torch.randn(12, 512, device="cuda").bfloat16()
        out = torch.empty(12, 128, dtype=torch.bfloat16, device="cuda")
        self.assertIs(op.try_grouped_index_gemm(x, self.weight, (5, 7), out=out), out)
        with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_load"
        ) as build:
            self.assertIsNone(op.try_grouped_index_gemm(x, self.weight, (5, 7)))
        build.assert_not_called()
        with mock.patch.dict(os.environ, {"DSV41_GROUPED_INDEX_GEMM": "0"}):
            self.assertIsNone(op.try_grouped_index_gemm(x, self.weight, (5, 7)))
        self.assertIsNone(
            op.try_grouped_index_gemm(
                x, self.weight, torch.tensor([5, 7], device="cuda")
            )
        )
        self.assertIsNone(op.try_grouped_index_gemm(x[:, ::2], self.weight, (5, 7)))
        self.assertIsNone(op.try_grouped_index_gemm(x, self.weight, (6, 7)))
        with self.assertRaises(ValueError):
            op.try_grouped_index_gemm(
                x, self.weight, (5, 7), out=x.view(-1)[: 12 * 128].view(12, 128)
            )
        with self.assertRaises(ValueError):
            op.try_grouped_index_gemm(x, self.weight, (5, 7), out=out.float())

    def test_current_side_stream_and_pointer_table_lifetime(self):
        rows = (17, 31, 67, 127)
        stream = torch.cuda.Stream()
        x = torch.randn(sum(rows), 512, device="cuda").bfloat16()
        expected = self.reference(x, rows)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(2000000)
            actual = op.try_grouped_index_gemm(x, self.weight, rows)
            # Churn same-size tables immediately after enqueueing their use.
            for _ in range(8):
                other = op.try_grouped_index_gemm(x + 1, self.weight, rows)
                del other
        del x
        torch.cuda.current_stream().wait_stream(stream)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_capture_returns_none_before_upload_and_caller_fallback_replays(self):
        rows = (7, 17)
        x = torch.randn(sum(rows), 512, device="cuda").bfloat16()
        expected = self.reference(x, rows)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.reference(x, rows)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with mock.patch.object(
            op, "_execute", side_effect=AssertionError("capture uploaded pointers")
        ):
            with torch.cuda.graph(graph, stream=stream):
                actual = op.try_grouped_index_gemm(x, self.weight, rows)
                if actual is None:
                    actual = self.reference(x, rows)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )
        x.add_(1)
        expected = self.reference(x, rows)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )


if __name__ == "__main__":
    with torch.inference_mode():
        unittest.main()

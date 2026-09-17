"""Pinned native TopK tests. Long lengths are scorer inputs, not generation."""

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.deepselect import sampler_topk, topk
from rtp_llm.models_py.modules.dsv41.native_aot import native_identity


class DeepSelectGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.getuid() != 0
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
        assert torch.version.cuda.startswith("13.")
        import deep_select

        cls.backend_file = Path(deep_select.__file__).resolve()
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if destination:
            (Path(destination) / "deepselect_components.json").write_text(
                json.dumps(
                    {
                        "scope": "native selection and Graph; long candidate lengths do not certify real 1M generation",
                        "backend": str(cls.backend_file),
                        "backend_sha256": hashlib.sha256(
                            cls.backend_file.read_bytes()
                        ).hexdigest(),
                        "native_identity": native_identity("deep-select"),
                        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                        "observations": cls.records,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def validate_selection(self, source, result, lengths, k):
        result.check()
        for row, length in enumerate(lengths):
            keep = min(k, length)
            selected = result.indices[row]
            indices = selected[selected >= 0].long()
            self.assertEqual(indices.numel(), keep)
            self.assertEqual(indices.unique().numel(), keep)
            self.assertTrue(((indices >= 0) & (indices < length)).all())
            expected_values = source[row, :length].topk(keep).values.sort().values
            actual_values = source[row, indices].sort().values
            torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
            torch.testing.assert_close(
                result.values[row][selected >= 0], source[row, indices], rtol=0, atol=0
            )
        self.records.append(
            {
                "test": self.id(),
                "dtype": str(source.dtype),
                "shape": list(source.shape),
                "lengths": lengths,
                "k": k,
                "status": result.status.cpu().tolist(),
                "indices_sha256": hashlib.sha256(
                    result.indices.contiguous().cpu().numpy().tobytes()
                ).hexdigest(),
            }
        )

    def test_short_empty_rows_ties_infinities_and_safe_sentinel(self):
        values = (torch.arange(3 * 137, device="cuda").reshape(3, 137) % 19).bfloat16()
        values[2, 3], values[2, 11] = torch.inf, -torch.inf
        ends = torch.tensor([0, 7, 137], dtype=torch.int32, device="cuda")
        for k in (1, 3, 512, 2048, 4096):
            result = topk(values, k, end=ends, sorted_index=True)
            self.validate_selection(values, result, ends.tolist(), k)
            self.assertTrue((result.indices[0] == -1).all())
            self.assertEqual(result.indices.stride(0) * 4 % 32, 0)

    def test_nan_is_rejected_even_in_vendor_short_row_branch(self):
        values = torch.ones((4, 137), dtype=torch.bfloat16, device="cuda")
        values[0, 0], values[1, 100], values[2, 100] = torch.nan, torch.nan, torch.nan
        ends = torch.tensor([7, 137, 7, -1], dtype=torch.int32, device="cuda")
        result = topk(values, 512, end=ends)
        self.assertEqual(result.status.tolist(), [1, 1, 0, 1])
        self.assertTrue((result.indices[[0, 1, 3]] == -1).all())
        with self.assertRaisesRegex(RuntimeError, "NaN, length"):
            result.check()

    def test_caller_output_rejects_misalignment_overlap_unpadded_tail_and_alias(self):
        values = torch.arange(274, device="cuda").reshape(2, 137).float()
        storage = torch.empty(64, dtype=torch.int32, device="cuda")
        invalid = (
            storage.as_strided((2, 3), (8, 1), 1),
            storage.as_strided((2, 3), (0, 1)),
            storage.as_strided((2, 17), (8, 1)),
            torch.empty(11, dtype=torch.int32, device="cuda").as_strided(
                (2, 3), (8, 1)
            ),
            values.view(torch.int32).as_strided((2, 3), (8, 1)),
        )
        for output in invalid:
            with self.subTest(
                stride=output.stride(), offset=output.storage_offset()
            ), self.assertRaises(ValueError):
                topk(values, output.shape[1], output_idx=output)
        guard = torch.full((2, 16), -99, dtype=torch.int32, device="cuda")
        result = topk(values, 3, output_idx=guard[:, :3])
        self.validate_selection(values, result, [137, 137], 3)
        self.assertTrue((guard[:, 8:] == -99).all())

    def test_sampler_fp32_preserves_values_and_sorts_without_sampling(self):
        storage = torch.arange(2 * 129283, dtype=torch.float32, device="cuda").reshape(
            2, 129283
        )
        values = storage[:, 1:129281]
        result = sampler_topk(values, 37)
        self.validate_selection(values, result, [129280, 129280], 37)
        self.assertTrue((result.values[:, :-1] > result.values[:, 1:]).all())
        with self.assertRaisesRegex(ValueError, "FP32"):
            sampler_topk(values.bfloat16(), 37)

    def test_cluster_512k_and_1m_graph_refreshes_lengths_and_values(self):
        for width in (524288, 1048576):
            values = ((torch.arange(width, device="cuda") * 17) % 257).bfloat16()[
                None, :
            ]
            lengths = torch.full((1,), width, dtype=torch.int32, device="cuda")
            output_idx = torch.empty((1, 512), dtype=torch.int32, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    topk(values, 512, end=lengths, output_idx=output_idx)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = topk(values, 512, end=lengths, output_idx=output_idx)
            for length in (7, 0, 65537, width):
                values.neg_()
                lengths.fill_(length)
                graph.replay()
                self.validate_selection(values, result, [length], 512)

    def test_native_kernel_is_observed(self):
        values = torch.arange(32768, dtype=torch.float32, device="cuda")[None, :]
        topk(values, 512).check()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            topk(values, 512).check()
        names = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertTrue(any("topk_select" in name for name in names), names)
        self.records.append({"test": self.id(), "cuda_kernels": names})
        destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if destination:
            profile.export_chrome_trace(
                str(Path(destination) / "deepselect_trace.json")
            )

    def test_strided_input_matches_sorted_baseline_values(self):
        for dtype in (torch.bfloat16, torch.float32):
            values = torch.randn((3, 2049), device="cuda", dtype=dtype)[:, 1::2]
            values[0, 17] = torch.inf
            values[1, 23] = -torch.inf
            actual = topk(values, 37, sorted_index=True)
            self.validate_selection(values, actual, [1024] * 3, 37)

    def test_rejects_malformed_vendor_output_before_gather(self):
        import deep_select

        values = torch.arange(521, device="cuda", dtype=torch.float32)[None, :]
        patterns = ([0, 0, 1], [0, 521, 1], [0, -2, 1], [0, -1, 1],
                    [0x3F3F3F3F, 1, 2])
        for pattern in patterns:
            def write_malformed(*args, **kwargs):
                kwargs["output_idx"].copy_(torch.tensor([pattern], device="cuda", dtype=torch.int32))

            with self.subTest(pattern=pattern), patch.object(deep_select, "topk", write_malformed):
                result = topk(values, 3)
                self.assertEqual(result.status.tolist(), [1])
                self.assertTrue((result.indices == -1).all())
                self.assertTrue((result.values == -torch.inf).all())

    def test_graph_recovers_after_nan_and_invalid_end(self):
        values = torch.randn((3, 16384), device="cuda", dtype=torch.bfloat16)
        ends = torch.full((3,), 16384, device="cuda", dtype=torch.int32)
        output = torch.empty((3, 512), device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                topk(values, 512, end=ends, output_idx=output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = topk(values, 512, end=ends, output_idx=output)
        pointers = (result.indices.data_ptr(), result.values.data_ptr(), result.status.data_ptr())
        for lengths, bad, expected in (
            ([16384, -1, 0], True, [1, 1, 0]),
            ([513, 16384, 7], False, [0, 0, 0]),
            ([16385, 0, 16384], False, [1, 0, 0]),
            ([16384, 16384, 16384], False, [0, 0, 0]),
        ):
            values.normal_()
            if bad:
                values[0, 1025] = torch.nan
            ends.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            output.fill_(-777)
            graph.replay()
            self.assertEqual(result.status.tolist(), expected)
            self.assertEqual(
                (result.indices.data_ptr(), result.values.data_ptr(), result.status.data_ptr()),
                pointers,
            )
            for row, rejected in enumerate(expected):
                if rejected:
                    self.assertTrue((result.indices[row] == -1).all())
                    self.assertTrue((result.values[row] == -torch.inf).all())
                else:
                    indices = result.indices[row]
                    indices = indices[indices >= 0].long()
                    self.assertEqual(indices.unique().numel(), min(lengths[row], 512))
                    expected_values = values[row, :lengths[row]].topk(min(lengths[row], 512)).values
                    torch.testing.assert_close(
                        values[row, indices].sort().values, expected_values.sort().values,
                        rtol=0, atol=0,
                    )


if __name__ == "__main__":
    unittest.main()

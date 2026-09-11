"""Pinned native TopK tests. Long lengths are scorer inputs, not generation."""

import hashlib
import json
import os
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

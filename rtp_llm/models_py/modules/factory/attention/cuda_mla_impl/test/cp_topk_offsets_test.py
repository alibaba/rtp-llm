import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_topk_offsets_triton import (
    is_supported,
    offset_cp_topk,
)


class CpTopkOffsetsCpuTest(unittest.TestCase):
    def test_fallback_preserves_invalid_indices_and_inputs(self) -> None:
        indices = torch.tensor([[0, -1, 7], [-2, 3, 1]], dtype=torch.int32)
        offsets = torch.tensor([5, 100], dtype=torch.int32)
        original = indices.clone()
        with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "1"}):
            result = offset_cp_topk(indices, offsets)
        torch.testing.assert_close(
            result, torch.tensor([[5, -1, 12], [-1, 103, 101]], dtype=torch.int32)
        )
        torch.testing.assert_close(indices, original)
        self.assertFalse(is_supported(indices, offsets))

    def test_disabled_path_promotes_before_adding(self) -> None:
        indices = torch.tensor([[2**31 - 1, -1]], dtype=torch.int32)
        offsets = torch.tensor([11], dtype=torch.int64)
        with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "0"}):
            result = offset_cp_topk(indices, offsets)
        torch.testing.assert_close(
            result, torch.tensor([[2**31 + 10, -1]], dtype=torch.int64)
        )


class CpTopkOffsetsCudaTest(unittest.TestCase):
    def test_default_fuses_launches_and_explicit_optout_matches(self) -> None:
        indices = torch.randint(-3, 2048, (17, 129), device="cuda", dtype=torch.int32)
        offsets = torch.arange(17, device="cuda", dtype=torch.int64) * 1000
        measurements = []
        for enabled in ("0", "1", "unset"):
            with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": enabled}):
                if enabled == "unset":
                    os.environ.pop("RTP_CP_FUSED_TOPK_OFFSETS")
                offset_cp_topk(indices, offsets)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    actual = offset_cp_topk(indices, offsets)
                    torch.cuda.synchronize()
                names = [
                    event.name
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
                measurements.append({"enabled": enabled, "gpu_events": names})
                expected = (indices + offsets[:, None]).masked_fill(indices < 0, -1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        self.assertEqual(len(measurements[0]["gpu_events"]), 3)
        self.assertEqual(len(measurements[1]["gpu_events"]), 1)
        self.assertEqual(len(measurements[2]["gpu_events"]), 1)
        output = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "cp_topk_launches.json"
        )
        output.write_text(json.dumps(measurements, indent=2))

    def test_integer_boundaries_match_native_promotion_and_overflow(self) -> None:
        for index_dtype in (torch.int32, torch.int64):
            for offset_dtype in (torch.int32, torch.int64):
                with self.subTest(index_dtype=index_dtype, offset_dtype=offset_dtype):
                    limits = torch.iinfo(index_dtype)
                    indices = torch.tensor(
                        [[limits.max, -1, 0], [limits.min, 3, 1]],
                        device="cuda",
                        dtype=index_dtype,
                    )
                    offsets = torch.tensor([11, -7], device="cuda", dtype=offset_dtype)
                    with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "0"}):
                        expected = offset_cp_topk(indices, offsets)

                    with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "1"}):
                        actual = offset_cp_topk(indices, offsets)

                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fused_matches_reference_for_strides_dtypes_and_empty_shapes(self) -> None:
        for rows, width in ((0, 7), (3, 0), (1, 1), (7, 129), (1252, 2048)):
            for index_dtype, offset_dtype in (
                (torch.int32, torch.int32),
                (torch.int32, torch.int64),
                (torch.int64, torch.int32),
                (torch.int64, torch.int64),
            ):
                with self.subTest(
                    shape=(rows, width), dtypes=(index_dtype, offset_dtype)
                ):
                    indices = torch.randint(
                        -3, 1000, (rows, width * 2), device="cuda", dtype=index_dtype
                    )[:, ::2]
                    offsets = (
                        torch.arange(rows * 2, device="cuda", dtype=offset_dtype)[::2]
                        * 101
                    )
                    if offset_dtype == torch.int64:
                        offsets += 2**31
                    original = indices.clone()
                    expected = (indices + offsets[:, None]).masked_fill(indices < 0, -1)
                    with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "1"}):
                        actual = offset_cp_topk(indices, offsets)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(indices, original, rtol=0, atol=0)

    def test_graph_replay_reads_updated_indices_and_offsets(self) -> None:
        indices = torch.tensor(
            [[0, -1, 7], [2, 3, -1]], device="cuda", dtype=torch.int32
        )
        offsets = torch.tensor([0, 100], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict(os.environ, {"RTP_CP_FUSED_TOPK_OFFSETS": "1"}):
            with torch.cuda.stream(stream):
                offset_cp_topk(indices, offsets)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = offset_cp_topk(indices, offsets)
            indices.copy_(torch.tensor([[-1, 5, 2], [4, -2, 0]], device="cuda"))
            offsets.copy_(torch.tensor([90, 300], device="cuda"))
            graph.replay()
        expected = torch.tensor(
            [[-1, 95, 92], [304, -1, 300]], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

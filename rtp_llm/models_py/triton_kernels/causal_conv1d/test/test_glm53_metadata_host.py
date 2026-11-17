"""Host-mirror contract for native and GPU-built attention metadata."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d.glm53_causal_conv1d import (
    prepare_causal_conv1d_metadata,
)


class Glm53MetadataHostTest(unittest.TestCase):
    def test_missing_and_empty_mirrors_match_populated_mirror(self):
        boundaries = torch.tensor([0, 1, 520, 520], dtype=torch.int32)
        reference = prepare_causal_conv1d_metadata(
            boundaries, "cpu", query_start_loc_cpu=boundaries
        )
        for mirror in (None, torch.empty(0, dtype=torch.int32)):
            with self.subTest(mirror=mirror):
                actual = prepare_causal_conv1d_metadata(
                    boundaries, "cpu", query_start_loc_cpu=mirror
                )
                self.assertEqual(actual.total, reference.total)
                self.assertEqual(actual.sequence_boundaries_host, (0, 1, 520, 520))
                torch.testing.assert_close(actual.batch_ptr, reference.batch_ptr)
                torch.testing.assert_close(
                    actual.token_chunk_offset_ptr, reference.token_chunk_offset_ptr
                )

    def test_populated_host_mirror_avoids_device_readback(self):
        host = torch.tensor([0, 3, 11], dtype=torch.int32)
        device = torch.empty(3, device="meta", dtype=torch.int32)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("readback")):
            actual = prepare_causal_conv1d_metadata(
                device, "cpu", query_start_loc_cpu=host
            )
        self.assertEqual(actual.sequence_boundaries_host, (0, 3, 11))

    def test_rejects_nonempty_mismatched_or_non_cpu_mirror(self):
        boundaries = torch.tensor([0, 3, 11], dtype=torch.int32)
        for mirror in (
            torch.tensor([0, 3], dtype=torch.int32),
            torch.empty(3, device="meta", dtype=torch.int32),
        ):
            with self.subTest(mirror=mirror):
                with self.assertRaisesRegex(ValueError, "matching CPU mirror"):
                    prepare_causal_conv1d_metadata(
                        boundaries, "cpu", query_start_loc_cpu=mirror
                    )


if __name__ == "__main__":
    unittest.main()

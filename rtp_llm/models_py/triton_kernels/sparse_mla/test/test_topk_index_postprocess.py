"""Correctness tests for sparse-MLA Top-K index postprocessing."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.topk_index_postprocess import (
    fused_stage1_request_indices,
    fused_stage2_global_indices,
)


def _make_case(
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    col_ids = torch.arange(cols, device="cuda", dtype=torch.int32).view(1, -1)
    valid_lengths = torch.clamp((row_ids * 8 + 1) // 4, min=0, max=cols)
    raw_indices = torch.where(
        col_ids < valid_lengths.view(-1, 1),
        col_ids.expand(rows, cols),
        torch.full((rows, cols), -1, device="cuda", dtype=torch.int32),
    ).contiguous()
    ragged_offsets = ((row_ids // 97) * 4096).contiguous()
    workspace_offsets = ((row_ids // 193) * 32768).contiguous()
    return raw_indices, ragged_offsets, workspace_offsets


def _reference(
    raw_indices: torch.Tensor,
    ragged_offsets: torch.Tensor,
    workspace_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    request_local = torch.where(
        raw_indices >= 0,
        raw_indices + ragged_offsets.view(-1, 1),
        raw_indices,
    )
    attention_global = torch.where(
        request_local < 0,
        -1,
        request_local + workspace_offsets.view(-1, 1),
    )
    return request_local, attention_global


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TopKIndexPostprocessTest(unittest.TestCase):
    def _check_case(self, rows: int, cols: int) -> None:
        raw, ragged_offsets, workspace_offsets = _make_case(rows, cols)
        expected_request, expected_global = _reference(
            raw,
            ragged_offsets,
            workspace_offsets,
        )

        request_output = torch.empty_like(raw)
        actual_request = fused_stage1_request_indices(
            raw,
            ragged_offsets,
            output=request_output,
        )
        self.assertIs(actual_request, request_output)

        global_output = torch.empty_like(raw)
        actual_global = fused_stage2_global_indices(
            expected_request,
            workspace_offsets,
            output=global_output,
        )
        self.assertIs(actual_global, global_output)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_request, expected_request, rtol=0, atol=0)
        torch.testing.assert_close(actual_global, expected_global, rtol=0, atol=0)

    def test_small_non_power_of_two(self) -> None:
        self._check_case(7, 13)

    def test_prime_rows(self) -> None:
        self._check_case(257, 512)

    def test_glm5_layer6_shape(self) -> None:
        self._check_case(6954, 2048)

    def test_empty_rows(self) -> None:
        self._check_case(0, 2048)

    def test_negative_sentinel_contract(self) -> None:
        raw = torch.tensor(
            [[0, 3, -1, -2], [5, -1, 7, -9]],
            dtype=torch.int32,
            device="cuda",
        )
        ragged_offsets = torch.tensor([10, 100], dtype=torch.int32, device="cuda")
        workspace_offsets = torch.tensor(
            [1000, 2000], dtype=torch.int32, device="cuda"
        )
        expected_request, expected_global = _reference(
            raw,
            ragged_offsets,
            workspace_offsets,
        )
        actual_request = fused_stage1_request_indices(raw, ragged_offsets)
        assert actual_request is not None
        actual_global = fused_stage2_global_indices(
            actual_request,
            workspace_offsets,
        )
        assert actual_global is not None
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_request, expected_request, rtol=0, atol=0)
        torch.testing.assert_close(actual_global, expected_global, rtol=0, atol=0)
        self.assertEqual(int(actual_request[0, 3].item()), -2)
        self.assertEqual(int(actual_global[0, 3].item()), -1)

    def test_unsupported_inputs_return_none(self) -> None:
        raw_cpu = torch.zeros((2, 4), dtype=torch.int32)
        offsets_cpu = torch.zeros((2,), dtype=torch.int32)
        self.assertIsNone(fused_stage1_request_indices(raw_cpu, offsets_cpu))

        raw_cuda = torch.zeros((2, 8), dtype=torch.int32, device="cuda")
        offsets_cuda = torch.zeros((2,), dtype=torch.int32, device="cuda")
        self.assertIsNone(
            fused_stage2_global_indices(raw_cuda[:, ::2], offsets_cuda)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

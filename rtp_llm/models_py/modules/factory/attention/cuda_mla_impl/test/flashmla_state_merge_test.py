import unittest

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _strided_lse(tokens: int, heads: int) -> torch.Tensor:
    return torch.randn((heads, tokens), device="cuda", dtype=torch.float32).T


def _merge_segmented(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    partial_q_indptr: torch.Tensor,
    destination_starts: torch.Tensor,
) -> None:
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_state_merge import (
        merge_attention_states_segmented_in_place,
    )

    merge_attention_states_segmented_in_place(
        output,
        output_lse,
        partial_output,
        partial_lse,
        partial_q_indptr,
        destination_starts,
    )


def _finite_pairwise_reference(
    accumulator: torch.Tensor,
    accumulator_lse: torch.Tensor,
    partial: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_lse = torch.maximum(accumulator_lse, partial_lse)
    accumulator_scale = torch.exp(accumulator_lse - max_lse)
    partial_scale = torch.exp(partial_lse - max_lse)
    expected = (
        accumulator.float() * accumulator_scale.unsqueeze(-1)
        + partial.float() * partial_scale.unsqueeze(-1)
    ) / (accumulator_scale + partial_scale).unsqueeze(-1)
    return expected, torch.logaddexp(accumulator_lse, partial_lse)


def _metadata(
    partial_q_indptr: tuple[int, ...], destination_starts: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(partial_q_indptr, device="cuda", dtype=torch.int32),
        torch.tensor(destination_starts, device="cuda", dtype=torch.int32),
    )


class FlashMLAStateMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "state merge requires CUDA")
        torch.manual_seed(20260903)

    def _assert_segmented_reference(
        self,
        output_dtype: torch.dtype,
        heads: int,
        partial_q_indptr: tuple[int, ...],
        destination_starts: tuple[int, ...],
        output_tokens: int,
    ) -> None:
        partial_tokens = partial_q_indptr[-1]
        output = torch.randn(
            (output_tokens, heads, 128), device="cuda", dtype=output_dtype
        )
        output_lse = _strided_lse(output_tokens, heads)
        partial_output = torch.randn(
            (partial_tokens, heads, 128), device="cuda", dtype=torch.bfloat16
        )
        partial_lse = _strided_lse(partial_tokens, heads)
        expected_output = output.float().clone()
        expected_lse = output_lse.clone()
        for segment, destination_start in enumerate(destination_starts):
            source_start = partial_q_indptr[segment]
            source_end = partial_q_indptr[segment + 1]
            destination_end = destination_start + source_end - source_start
            merged, merged_lse = _finite_pairwise_reference(
                output[destination_start:destination_end],
                output_lse[destination_start:destination_end],
                partial_output[source_start:source_end],
                partial_lse[source_start:source_end],
            )
            expected_output[destination_start:destination_end] = merged
            expected_lse[destination_start:destination_end] = merged_lse

        _merge_segmented(
            output,
            output_lse,
            partial_output,
            partial_lse,
            *_metadata(partial_q_indptr, destination_starts),
        )
        torch.cuda.synchronize()

        atol = 0.016 if output_dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(
            output.float(), expected_output, rtol=1e-3, atol=atol
        )
        torch.testing.assert_close(output_lse, expected_lse, rtol=2e-5, atol=2e-5)

    def test_noncontiguous_destinations_and_k3_head_layouts_match_reference(
        self,
    ) -> None:
        for output_dtype in (torch.bfloat16, torch.float32):
            for heads in (12, 96):
                with self.subTest(output_dtype=output_dtype, heads=heads):
                    self._assert_segmented_reference(
                        output_dtype,
                        heads=heads,
                        partial_q_indptr=(0, 2, 5, 6),
                        destination_starts=(1, 6, 13),
                        output_tokens=16,
                    )

    def test_empty_and_invalid_lse_semantics(self) -> None:
        heads = 12
        metadata = _metadata((0, 4), (0,))
        finite_partial = torch.ones(
            (4, heads, 128), device="cuda", dtype=torch.bfloat16
        )
        finite_partial_lse = torch.zeros((4, heads), device="cuda", dtype=torch.float32)

        output = torch.randn((4, heads, 128), device="cuda", dtype=torch.float32)
        output_lse = torch.zeros((4, heads), device="cuda", dtype=torch.float32)
        original_output = output.clone()
        empty_partial = torch.full_like(finite_partial, float("nan"))
        empty_partial_lse = torch.full_like(finite_partial_lse, -torch.inf)
        _merge_segmented(
            output, output_lse, empty_partial, empty_partial_lse, *metadata
        )
        torch.testing.assert_close(output, original_output)
        self.assertTrue(torch.equal(output_lse, torch.zeros_like(output_lse)))

        output.fill_(float("nan"))
        output_lse.fill_(-torch.inf)
        _merge_segmented(
            output, output_lse, finite_partial, finite_partial_lse, *metadata
        )
        torch.testing.assert_close(output, torch.ones_like(output))
        self.assertTrue(torch.equal(output_lse, finite_partial_lse))

        output.fill_(float("nan"))
        output_lse.fill_(-torch.inf)
        _merge_segmented(
            output, output_lse, empty_partial, empty_partial_lse, *metadata
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(torch.isneginf(output_lse).all())

        for invalid_side in ("accumulator", "partial"):
            with self.subTest(invalid_side=invalid_side):
                output.zero_()
                output_lse.zero_()
                partial_lse = finite_partial_lse.clone()
                if invalid_side == "accumulator":
                    output_lse.fill_(float("nan"))
                else:
                    partial_lse.fill_(float("nan"))
                _merge_segmented(
                    output,
                    output_lse,
                    finite_partial,
                    partial_lse,
                    *metadata,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.isnan(output).all())
                self.assertTrue(torch.isnan(output_lse).all())

        for positive_infinity_side in ("accumulator", "partial", "both"):
            with self.subTest(positive_infinity_side=positive_infinity_side):
                output.zero_()
                output_lse.zero_()
                partial_lse = finite_partial_lse.clone()
                if positive_infinity_side in ("accumulator", "both"):
                    output_lse.fill_(torch.inf)
                if positive_infinity_side in ("partial", "both"):
                    partial_lse.fill_(torch.inf)
                _merge_segmented(
                    output,
                    output_lse,
                    finite_partial,
                    partial_lse,
                    *metadata,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.isnan(output).all())
                self.assertTrue(torch.isposinf(output_lse).all())

        for nan_side in ("accumulator", "partial"):
            with self.subTest(nan_side=nan_side, other_side="positive_infinity"):
                output.zero_()
                output_lse.fill_(torch.inf if nan_side == "partial" else torch.nan)
                partial_lse = torch.full_like(
                    finite_partial_lse,
                    torch.nan if nan_side == "partial" else torch.inf,
                )
                _merge_segmented(
                    output,
                    output_lse,
                    finite_partial,
                    partial_lse,
                    *metadata,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.isnan(output).all())
                self.assertTrue(torch.isnan(output_lse).all())

    def test_zero_partial_tokens_is_a_noop(self) -> None:
        output = torch.randn((5, 12, 128), device="cuda", dtype=torch.float32)
        output_lse = _strided_lse(5, 12)
        expected_output = output.clone()
        expected_lse = output_lse.clone()

        _merge_segmented(
            output,
            output_lse,
            torch.empty((0, 12, 128), device="cuda", dtype=torch.bfloat16),
            torch.empty((0, 12), device="cuda", dtype=torch.float32),
            *_metadata((0,), ()),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(output_lse, expected_lse)

    def test_b63_segment_boundaries_match_reference(self) -> None:
        q_lengths = tuple(1 + index % 17 for index in range(63))
        partial_q_indptr = [0]
        for q_length in q_lengths:
            partial_q_indptr.append(partial_q_indptr[-1] + q_length)
        destination_starts = tuple(
            2 * partial_q_indptr[index] for index in range(len(q_lengths))
        )

        self._assert_segmented_reference(
            torch.float32,
            heads=12,
            partial_q_indptr=tuple(partial_q_indptr),
            destination_starts=destination_starts,
            output_tokens=2 * partial_q_indptr[-1],
        )

    def test_binding_rejects_non_k3_state_contracts(self) -> None:
        output = torch.empty((2, 12, 128), device="cuda", dtype=torch.float32)
        output_lse = torch.empty((2, 12), device="cuda", dtype=torch.float32)
        partial = torch.empty((1, 12, 128), device="cuda", dtype=torch.bfloat16)
        partial_lse = torch.empty((1, 12), device="cuda", dtype=torch.float32)
        metadata = _metadata((0, 1), (0,))

        with self.assertRaisesRegex(RuntimeError, "128-wide"):
            rtp_llm_ops._flashmla_merge_attention_states_segmented_in_place(
                output[..., :64].contiguous(),
                output_lse,
                partial[..., :64].contiguous(),
                partial_lse,
                *metadata,
            )
        with self.assertRaisesRegex(RuntimeError, "partial output must be BF16"):
            rtp_llm_ops._flashmla_merge_attention_states_segmented_in_place(
                output,
                output_lse,
                partial.float(),
                partial_lse,
                *metadata,
            )

    def test_binding_rejects_unsupported_lse_layout(self) -> None:
        output = torch.empty((1, 12, 128), device="cuda", dtype=torch.float32)
        partial = torch.empty((1, 12, 128), device="cuda", dtype=torch.bfloat16)
        valid_lse = torch.empty((1, 12), device="cuda", dtype=torch.float32)
        zero_stride_lse = torch.empty(
            (1, 1), device="cuda", dtype=torch.float32
        ).expand(1, 12)
        metadata = _metadata((0, 1), (0,))

        for output_lse, partial_lse in (
            (zero_stride_lse, valid_lse),
            (valid_lse, zero_stride_lse),
        ):
            with self.subTest(
                output_stride=output_lse.stride(),
                partial_stride=partial_lse.stride(),
            ), self.assertRaisesRegex(RuntimeError, "non-overlapping"):
                rtp_llm_ops._flashmla_merge_attention_states_segmented_in_place(
                    output, output_lse, partial, partial_lse, *metadata
                )

    def test_binding_rejects_malformed_metadata(self) -> None:
        output = torch.zeros((2, 12, 128), device="cuda", dtype=torch.float32)
        output_lse = torch.zeros((2, 12), device="cuda", dtype=torch.float32)
        partial = torch.zeros((1, 12, 128), device="cuda", dtype=torch.bfloat16)
        partial_lse = torch.zeros((1, 12), device="cuda", dtype=torch.float32)
        valid_indptr, valid_destinations = _metadata((0, 1), (0,))

        cases = (
            (
                valid_indptr.view(1, 2),
                valid_destinations,
                "partial_q_indptr must be a contiguous int32 vector",
            ),
            (
                valid_indptr.to(torch.int64),
                valid_destinations,
                "partial_q_indptr must be a contiguous int32 vector",
            ),
            (
                valid_indptr,
                valid_destinations.view(1, 1),
                "destination_starts must be a contiguous int32 vector",
            ),
            (
                valid_indptr,
                valid_destinations.to(torch.int64),
                "destination_starts must be a contiguous int32 vector",
            ),
            (
                valid_indptr,
                torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                r"S\+1 indptr entries and S destinations",
            ),
        )
        for partial_q_indptr, destination_starts, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                _merge_segmented(
                    output,
                    output_lse,
                    partial,
                    partial_lse,
                    partial_q_indptr,
                    destination_starts,
                )


if __name__ == "__main__":
    unittest.main()

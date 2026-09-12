from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.distributed.sequence_parallel import (
    local_physical_token_view,
    mask_physical_padding_slots_,
    sequence_parallel_layout,
    sequence_parallel_layout_from_attention_inputs,
)


class SequenceParallelLayoutTest(TestCase):
    def test_layout_is_derived_from_attention_inputs(self) -> None:
        attention_inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=True,
            is_cuda_graph=True,
            input_lengths=torch.zeros(8, dtype=torch.int32),
            logical_request_count=3,
            coordinated_request_count=3,
            logical_token_count=12,
            coordinated_token_count=12,
        )

        layout = sequence_parallel_layout_from_attention_inputs(
            attention_inputs,
            physical_tokens=32,
            world_size=8,
            rank=2,
        )

        self.assertEqual(layout.mode, "target_verify")
        self.assertEqual(layout.requests.tokens_per_request, 4)
        self.assertEqual(layout.requests.logical_requests, 3)
        self.assertEqual(layout.requests.coordinated_requests, 3)
        self.assertEqual(layout.requests.physical_requests, 8)
        self.assertEqual(layout.requests.graph_batch_size, 8)
        self.assertEqual(layout.tokens.logical_tokens, 12)
        self.assertEqual(layout.tokens.coordinated_tokens, 12)
        self.assertEqual(layout.tokens.physical_tokens, 32)

    def test_prefill_padding_uses_token_granularity(self) -> None:
        layout = sequence_parallel_layout(
            mode="prefill",
            logical_requests=2,
            coordinated_requests=2,
            physical_requests=3,
            tokens_per_request=0,
            logical_tokens=13,
            coordinated_tokens=13,
            physical_tokens=16,
            world_size=8,
            rank=7,
        )

        self.assertEqual(layout.requests.tensor_parallel_padding_requests, 1)
        self.assertEqual(layout.tokens.tensor_parallel_padding_tokens, 3)
        self.assertEqual(layout.tokens.local_tokens, 2)
        self.assertEqual(layout.tokens.local_start, 14)
        self.assertEqual(layout.tokens.local_valid_tokens, 0)

    def test_decode_padding_uses_request_granularity(self) -> None:
        layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=3,
            coordinated_requests=3,
            physical_requests=8,
            tokens_per_request=1,
            logical_tokens=3,
            coordinated_tokens=3,
            physical_tokens=8,
            world_size=8,
            rank=2,
            graph_batch_size=8,
        )

        self.assertEqual(layout.requests.tensor_parallel_padding_requests, 5)
        self.assertEqual(layout.tokens.tensor_parallel_padding_tokens, 5)
        self.assertEqual(layout.requests.graph_batch_size, 8)
        self.assertEqual(layout.tokens.local_valid_tokens, 1)

    def test_target_verify_preserves_request_width(self) -> None:
        layout = sequence_parallel_layout(
            mode="target_verify",
            logical_requests=1,
            coordinated_requests=1,
            physical_requests=2,
            tokens_per_request=4,
            logical_tokens=4,
            coordinated_tokens=4,
            physical_tokens=8,
            world_size=8,
            rank=1,
            graph_batch_size=2,
        )

        self.assertEqual(layout.requests.tensor_parallel_padding_requests, 1)
        self.assertEqual(layout.tokens.tensor_parallel_padding_tokens, 4)
        self.assertEqual(layout.tokens.local_tokens, 1)
        self.assertEqual(layout.tokens.local_start, 1)
        self.assertEqual(layout.tokens.local_valid_tokens, 1)

    def test_replica_and_tp_padding_are_accounted_separately(self) -> None:
        layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=2,
            coordinated_requests=5,
            physical_requests=8,
            tokens_per_request=1,
            logical_tokens=2,
            coordinated_tokens=5,
            physical_tokens=8,
            world_size=8,
            rank=7,
        )

        self.assertEqual(layout.requests.coordination_padding_requests, 3)
        self.assertEqual(layout.requests.tensor_parallel_padding_requests, 3)
        self.assertEqual(layout.tokens.tensor_parallel_padding_tokens, 3)
        self.assertEqual(layout.tokens.total_padding_tokens, 6)

    def test_local_token_slice_is_an_allocation_free_view(self) -> None:
        layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=3,
            coordinated_requests=3,
            physical_requests=8,
            tokens_per_request=1,
            logical_tokens=3,
            coordinated_tokens=3,
            physical_tokens=8,
            world_size=4,
            rank=2,
        )
        source = torch.arange(16).reshape(8, 2)

        actual = local_physical_token_view(source, layout)

        torch.testing.assert_close(actual, source[4:6])
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(
            actual.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )

    def test_published_physical_shape_must_match_model_input(self) -> None:
        attention_inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=False,
            input_lengths=torch.zeros(8, dtype=torch.int32),
            logical_request_count=3,
            coordinated_request_count=3,
            physical_request_count=8,
            logical_token_count=3,
            coordinated_token_count=3,
            physical_token_count=8,
        )

        with self.assertRaisesRegex(ValueError, "physical token count"):
            sequence_parallel_layout_from_attention_inputs(
                attention_inputs,
                physical_tokens=16,
                world_size=8,
                rank=0,
            )

    def test_cache_slot_mask_covers_tp_and_coordination_padding(self) -> None:
        slot_mapping = torch.arange(8, dtype=torch.int64)

        actual = mask_physical_padding_slots_(
            slot_mapping,
            logical_tokens=3,
            physical_tokens=8,
        )

        torch.testing.assert_close(
            actual,
            torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]),
        )

    def test_cache_slot_mask_covers_an_all_dummy_rank(self) -> None:
        slot_mapping = torch.arange(4, dtype=torch.int64)

        actual = mask_physical_padding_slots_(
            slot_mapping,
            logical_tokens=0,
            physical_tokens=4,
        )

        torch.testing.assert_close(actual, torch.full((4,), -1))


if __name__ == "__main__":
    main()

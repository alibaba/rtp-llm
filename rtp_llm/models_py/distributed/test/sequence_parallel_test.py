from unittest import TestCase, main

import torch

from rtp_llm.models_py.distributed.sequence_parallel import (
    sequence_parallel_layout,
    shard_physical_tokens,
)


class SequenceParallelLayoutTest(TestCase):
    def test_prefill_padding_uses_token_granularity(self) -> None:
        layout = sequence_parallel_layout(
            mode="prefill",
            logical_requests=2,
            physical_requests=3,
            tokens_per_request=0,
            logical_tokens=13,
            physical_tokens=16,
            world_size=8,
            rank=7,
        )

        self.assertEqual(layout.padding_requests, 1)
        self.assertEqual(layout.padding_tokens, 3)
        self.assertEqual(layout.local_tokens, 2)
        self.assertEqual(layout.local_start, 14)
        self.assertEqual(layout.local_valid_tokens, 0)

    def test_decode_padding_uses_request_granularity(self) -> None:
        layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=3,
            physical_requests=8,
            tokens_per_request=1,
            logical_tokens=3,
            physical_tokens=8,
            world_size=8,
            rank=2,
            graph_batch_size=8,
        )

        self.assertEqual(layout.padding_requests, 5)
        self.assertEqual(layout.padding_tokens, 5)
        self.assertEqual(layout.graph_batch_size, 8)
        self.assertEqual(layout.local_valid_tokens, 1)

    def test_target_verify_preserves_request_width(self) -> None:
        layout = sequence_parallel_layout(
            mode="target_verify",
            logical_requests=1,
            physical_requests=8,
            tokens_per_request=4,
            logical_tokens=4,
            physical_tokens=32,
            world_size=8,
            rank=1,
            graph_batch_size=8,
        )

        self.assertEqual(layout.padding_requests, 7)
        self.assertEqual(layout.padding_tokens, 28)
        self.assertEqual(layout.local_tokens, 4)
        self.assertEqual(layout.local_start, 4)
        self.assertEqual(layout.local_valid_tokens, 0)

    def test_shard_is_a_contiguous_physical_slice(self) -> None:
        layout = sequence_parallel_layout(
            mode="decode",
            logical_requests=3,
            physical_requests=8,
            tokens_per_request=1,
            logical_tokens=3,
            physical_tokens=8,
            world_size=4,
            rank=2,
        )
        source = torch.arange(16).reshape(8, 2)

        actual = shard_physical_tokens(source, layout)

        torch.testing.assert_close(actual, source[4:6])
        self.assertTrue(actual.is_contiguous())


if __name__ == "__main__":
    main()

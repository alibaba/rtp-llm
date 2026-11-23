"""Cross-layer metadata sharing must preserve fresh request geometry."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.step_metadata import (
    cached_indexer_padding,
    cached_main_padded_lengths,
    cached_query_geometry,
    cached_step_value,
)


class StepMetadataTest(unittest.TestCase):
    def test_padding_separates_distinct_block_tables(self):
        shared = SimpleNamespace(metadata_cache={})
        visible = torch.arange(1, 13, dtype=torch.int32).view(3, 4)
        compressed = visible // 4
        state = torch.tensor([[0, 2], [0, 0], [0, 0]], dtype=torch.int32)
        indexer = torch.tensor([[0, 0], [0, 3], [0, 0]], dtype=torch.int32)
        main = torch.tensor([[0, 0], [0, 2], [0, 6]], dtype=torch.int32)
        side = cached_indexer_padding(shared, visible, compressed, state, indexer)
        self.assertIs(
            side, cached_indexer_padding(shared, visible, compressed, state, indexer)
        )
        torch.testing.assert_close(side[0].flatten(), torch.tensor([True, True, False]))
        torch.testing.assert_close(side[1], visible * side[0])
        torch.testing.assert_close(side[2], compressed * side[0])
        actual = cached_main_padded_lengths(shared, side[1], main)
        torch.testing.assert_close(
            actual, visible * visible.new_tensor([[0], [1], [0]])
        )
        other = cached_indexer_padding(shared, visible, compressed, main, indexer)
        torch.testing.assert_close(
            other[0].flatten(), torch.tensor([False, True, True])
        )
        self.assertIsNot(side, other)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_replay_updates_indexer_and_main_padding(self):
        for batch in (2, 4, 8, 16):
            for query in (1, 2, 3, 4):
                prefixes = torch.zeros(batch, device="cuda", dtype=torch.int32)
                state = prefixes.new_zeros((batch, 3))
                indexer = prefixes.new_zeros((batch, 4))
                main = prefixes.new_zeros((batch, 5))

                def forward():
                    context = SimpleNamespace(metadata_cache={})
                    visible, compressed, _, _ = cached_query_geometry(
                        context, prefixes, query, 4
                    )
                    side = cached_indexer_padding(
                        context, visible, compressed, state, indexer
                    )
                    masked = cached_main_padded_lengths(context, side[1], main)
                    for _ in range(11):
                        self.assertIs(
                            side,
                            cached_indexer_padding(
                                context, visible, compressed, state, indexer
                            ),
                        )
                        self.assertIs(
                            masked, cached_main_padded_lengths(context, side[1], main)
                        )
                    return side, masked

                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    side, masked = forward()
                rows = torch.arange(batch, device="cuda")
                offsets = torch.arange(1, query + 1, device="cuda", dtype=torch.int32)
                for replay, length in enumerate((0, 127, 8191, 10239, 32768)):
                    prefixes.copy_(length + rows)
                    state.zero_()
                    indexer.zero_()
                    main.zero_()
                    if replay:
                        state[(rows + replay) % 3 == 0, -1] = 3
                        indexer[(rows + replay) % 3 == 1, 0] = 5
                        main[(rows + replay) % 2 == 0, 2] = 7
                    graph.replay()
                    active = (state.ne(0).any(1) | indexer.ne(0).any(1))[:, None]
                    visible = prefixes[:, None] + offsets
                    torch.testing.assert_close(side[0], active)
                    torch.testing.assert_close(side[1], torch.where(active, visible, 0))
                    torch.testing.assert_close(
                        side[2], torch.where(active, visible // 4, 0)
                    )
                    torch.testing.assert_close(
                        masked,
                        torch.where(active & main.ne(0).any(1)[:, None], visible, 0),
                    )
                graph.reset()

    def assert_geometry(self, actual, prefix, query, ratio):
        visible = prefix[:, None] + torch.arange(1, query + 1, device=prefix.device)
        torch.testing.assert_close(actual[0], visible.to(torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(
            actual[1], (visible // ratio).to(torch.int32), rtol=0, atol=0
        )
        if actual[2] is not None:
            positions = (visible - 1).reshape(-1).to(torch.int64)
            torch.testing.assert_close(actual[2], positions, rtol=0, atol=0)
            torch.testing.assert_close(
                actual[3], positions - positions.remainder(ratio), rtol=0, atol=0
            )

    def test_factory_runs_once_per_forward_and_supports_uncached_callers(self):
        calls = []

        def make():
            calls.append(1)
            return object()

        first = SimpleNamespace(metadata_cache={})
        shared = [cached_step_value(first, ("example",), make) for _ in range(12)]
        self.assertTrue(all(value is shared[0] for value in shared))
        self.assertEqual(len(calls), 1)
        second = SimpleNamespace(metadata_cache={})
        self.assertIsNot(cached_step_value(second, ("example",), make), shared[0])
        uncached = SimpleNamespace(metadata_cache=None)
        self.assertIsNot(
            cached_step_value(uncached, ("example",), make),
            cached_step_value(uncached, ("example",), make),
        )

    def check_geometry(self, device):
        for batch in (1, 2, 4, 8, 16):
            for query in (1, 2, 3, 4):
                for ratio in (4, 8):
                    prefix = (
                        torch.arange(batch, device=device, dtype=torch.int32) + 8191
                    )
                    context = SimpleNamespace(metadata_cache={})
                    values = [
                        cached_query_geometry(context, prefix, query, ratio)
                        for _ in range(12)
                    ]
                    self.assertTrue(all(value is values[0] for value in values))
                    self.assert_geometry(values[0], prefix, query, ratio)
                    other = cached_query_geometry(context, prefix.clone(), query, ratio)
                    self.assertIsNot(other, values[0])
                    torch.testing.assert_close(other[0], values[0][0], rtol=0, atol=0)

    def test_decode_and_all_mtp_query_lengths_cpu(self):
        self.check_geometry("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_decode_and_all_mtp_query_lengths_cuda(self):
        self.check_geometry("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_replay_refreshes_shared_values(self):
        for batch in (2, 4, 8, 16):
            for query in (1, 2, 3, 4):
                prefix = torch.zeros(batch, device="cuda", dtype=torch.int32)

                def forward():
                    context = SimpleNamespace(metadata_cache={})
                    result = cached_query_geometry(context, prefix, query, 4)
                    for _ in range(11):
                        self.assertIs(
                            cached_query_geometry(context, prefix, query, 4), result
                        )
                    return result

                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = forward()
                for length in (0, 127, 8191, 10239, 32768):
                    prefix.copy_(
                        torch.arange(batch, device="cuda", dtype=torch.int32) + length
                    )
                    graph.replay()
                    self.assert_geometry(actual, prefix, query, 4)
                graph.reset()


if __name__ == "__main__":
    unittest.main()

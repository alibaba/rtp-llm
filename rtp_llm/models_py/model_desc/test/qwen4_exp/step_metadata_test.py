"""Cross-layer metadata sharing must preserve fresh request geometry."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.step_metadata import (
    cached_indexer_padding,
    cached_main_padded_lengths,
    cached_prefill_geometry,
    cached_query_geometry,
    cached_step_value,
    check_prefill_metadata_once,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks


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

    def check_prefill(self, device):
        device = torch.device(device)
        inputs = SimpleNamespace(
            input_lengths=torch.tensor([5, 2], dtype=torch.int32),
            prefix_lengths=torch.tensor([128, 256], dtype=torch.int32),
        )
        cache = {}
        contexts = [
            SimpleNamespace(main_inputs=inputs, metadata_cache=cache) for _ in range(12)
        ]
        values = [
            cached_prefill_geometry(ctx, [5, 2], [128, 256], 4, device)
            for ctx in contexts
        ]
        self.assertTrue(all(value is values[0] for value in values))
        geometry = values[0]
        self.assertEqual(geometry.cu_seqlens.tolist(), [0, 5, 7])
        self.assertEqual(geometry.cu_kv_seqlens.tolist(), [0, 133, 391])
        self.assertEqual(
            geometry.logical_positions.tolist(), [128, 129, 130, 131, 132, 256, 257]
        )
        self.assertEqual(
            geometry.visible_lengths.tolist(),
            [[129, 130, 131, 132, 133], [257, 258, 0, 0, 0]],
        )
        self.assertEqual(
            geometry.compressed_lengths.tolist(),
            [[32, 32, 32, 33, 33], [64, 64, 0, 0, 0]],
        )
        self.assertEqual(geometry.final_compressed_lengths.tolist(), [33, 64])
        self.assertEqual(geometry.prefixes.dtype, torch.int64)
        self.assertEqual(geometry.prefixes.tolist(), [128, 256])
        context = contexts[0]
        other_ratio = cached_prefill_geometry(context, [5, 2], [128, 256], 8, device)
        self.assertIsNot(other_ratio, geometry)
        self.assertEqual(other_ratio.final_compressed_lengths.tolist(), [16, 32])

        # Each host value snapshot must distinguish an entry at the same address.
        inputs.input_lengths.copy_(torch.tensor([2, 1], dtype=torch.int32))
        shorter = cached_prefill_geometry(context, [2, 1], [128, 256], 4, device)
        self.assertIsNot(shorter, geometry)
        self.assertEqual(shorter.cu_seqlens.tolist(), [0, 2, 3])
        self.assertEqual(shorter.logical_positions.tolist(), [128, 129, 256])

        inputs.prefix_lengths.add_(128)
        shifted = cached_prefill_geometry(context, [2, 1], [256, 384], 4, device)
        self.assertIsNot(shifted, shorter)
        self.assertEqual(shifted.prefixes.tolist(), [256, 384])
        self.assertEqual(shifted.logical_positions.tolist(), [256, 257, 384])

        next_context = SimpleNamespace(main_inputs=inputs, metadata_cache={})
        fresh = cached_prefill_geometry(next_context, [2, 1], [256, 384], 4, device)
        self.assertIsNot(fresh, shifted)
        self.assertEqual(
            fresh.logical_positions.tolist(), shifted.logical_positions.tolist()
        )
        uncached = SimpleNamespace(main_inputs=inputs, metadata_cache=None)
        self.assertIsNot(
            cached_prefill_geometry(uncached, [2, 1], [256, 384], 4, device),
            cached_prefill_geometry(uncached, [2, 1], [256, 384], 4, device),
        )

    def test_prefill_request_geometry_cpu(self):
        self.check_prefill("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_prefill_request_geometry_cuda(self):
        self.check_prefill("cuda")

    def test_prefill_receipts_require_success_and_refresh_mutated_metadata(self):
        context = SimpleNamespace(
            metadata_cache={},
            is_mtp_draft=False,
            main_inputs=SimpleNamespace(
                is_prefill=True, is_target_verify=False, is_cuda_graph=False
            ),
        )
        table = torch.tensor([1], dtype=torch.int32)
        calls = []

        def validate(*, scoped=True):
            checks = DeviceChecks()
            checks.prefill_metadata_readonly = scoped

            def add():
                calls.append(1)
                checks.add((table <= 0).any(), "unallocated", ValueError)

            check_prefill_metadata_once(context, checks, ("pages", 2), (table,), add)
            return checks

        first, pending = validate(), validate()
        self.assertEqual(len(calls), 2)
        self.assertFalse(pending.verified)
        first.verify()
        pending.verify()
        for _ in range(11):
            validate().verify()
        self.assertEqual(len(calls), 2)
        validate(scoped=False).verify()
        self.assertEqual(len(calls), 3)
        table.zero_()
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "unallocated"):
                validate().verify()
        self.assertEqual(len(calls), 5)
        table.fill_(1)
        context.metadata_cache = {}
        validate().verify()
        self.assertEqual(len(calls), 6)

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
        expect_triton = torch.version.hip is None and (
            torch.cuda.get_device_capability() == (12, 0)
        )
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
                if expect_triton:
                    self.assertIsNotNone(
                        actual[2], "SM120 must produce query positions"
                    )
                    self.assertIsNotNone(actual[3], "SM120 must produce group starts")
                for length in (0, 127, 8191, 10239, 32768):
                    prefix.copy_(
                        torch.arange(batch, device="cuda", dtype=torch.int32) + length
                    )
                    graph.replay()
                    self.assert_geometry(actual, prefix, query, 4)
                graph.reset()


if __name__ == "__main__":
    unittest.main()

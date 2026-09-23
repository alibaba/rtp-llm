"""Exact DSpARK metadata checks, including CUDA Graph input mutation."""

import importlib.util
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

# This module has no model/extension dependency, so CPU gate checks stay local.
_spec = importlib.util.spec_from_file_location(
    "dspark_metadata_test_module",
    Path(__file__).resolve().parents[1] / "_dspark_metadata_triton.py",
)
metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metadata)


def _scalar_reference(
    query,
    prefix,
    active,
    table,
    *,
    gamma,
    window_size,
    entries_per_block,
    tokens_per_block
):
    query = query.cpu().reshape(-1).tolist()
    prefix, active, table = (
        prefix.cpu().tolist(),
        active.cpu().tolist(),
        table.cpu().tolist(),
    )
    topk = ((window_size + gamma + 127) // 128) * 128

    def slot(request, position):
        if position < 0 or position // tokens_per_block >= len(table[request]):
            return -1
        block = table[request][position // tokens_per_block]
        return (
            block * entries_per_block + position % entries_per_block
            if block > 0
            else -1
        )

    def int32(value):
        return (value + (1 << 31)) % (1 << 32) - (1 << 31)

    query_slots, indices, lengths = [], [], []
    for request, end in enumerate(prefix):
        committed = min(end, window_size)
        lengths.append(int32(committed + gamma) if active[request] else 0)
        row = []
        for column in range(topk):
            position = -1
            if active[request]:
                if column < committed:
                    position = end - committed + column
                elif column < committed + gamma:
                    position = end + column - committed
            row.append(int32(slot(request, position)))
        for offset in range(gamma):
            query_slots.append(slot(request, query[request * gamma + offset]))
            indices.append(row)
    return metadata.DSparkMetadata(
        torch.tensor(query_slots, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int32).reshape(len(prefix) * gamma, topk),
        torch.tensor(lengths, dtype=torch.int32),
    )


def _inputs(batch=3, gamma=5, device="cpu"):
    prefix = torch.arange(batch, dtype=torch.int32) * 17
    query = prefix[:, None].long() + torch.arange(gamma)
    active = torch.ones(batch, dtype=torch.bool)
    table = torch.arange(1, batch * 4 + 1, dtype=torch.int32).view(batch, 4)
    return tuple(t.to(device) for t in (query, prefix, active, table))


_GEOMETRY = dict(gamma=5, window_size=128, entries_per_block=256, tokens_per_block=8192)


class DSparkMetadataCPU(unittest.TestCase):
    def test_cpu_returns_none_without_launch_or_input_mutation(self):
        inputs = _inputs()
        before = tuple(t.clone() for t in inputs)
        with patch.object(metadata, "_dspark_metadata_kernel") as kernel:
            self.assertFalse(metadata.is_supported(*inputs, **_GEOMETRY))
            self.assertIsNone(metadata.try_build_dspark_metadata(*inputs, **_GEOMETRY))
            kernel.__getitem__.assert_not_called()
        for actual, expected in zip(inputs, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gate_can_disable_without_inspecting_inputs(self):
        with patch.dict(os.environ, {"DSV4_FUSED_DSPARK_METADATA": "0"}):
            self.assertIsNone(
                metadata.try_build_dspark_metadata(None, None, None, None, **_GEOMETRY)
            )

    def test_reference_noncausal_block_boundary_and_inactive_write_contract(self):
        inputs = (
            torch.tensor([[15, 16, -1], [1, 2, 3]], dtype=torch.int64),
            torch.tensor([15, 1], dtype=torch.int32),
            torch.tensor([True, False]),
            torch.tensor([[2, 5], [7, 0]], dtype=torch.int32),
        )
        result = _scalar_reference(
            *inputs, gamma=3, window_size=4, entries_per_block=8, tokens_per_block=16
        )
        self.assertEqual(result.query_slots.tolist(), [23, 40, -1, 57, 58, 59])
        self.assertEqual(
            result.global_indices[0, :8].tolist(), [19, 20, 21, 22, 23, 40, 41, -1]
        )
        self.assertTrue(torch.equal(result.global_indices[0], result.global_indices[2]))
        self.assertTrue((result.global_indices[3:] == -1).all())
        self.assertEqual(result.topk_length.tolist(), [7, 0])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkMetadataCUDA(unittest.TestCase):
    def setUp(self):
        gate = patch.dict(os.environ, {"DSV4_FUSED_DSPARK_METADATA": "1"})
        gate.start()
        self.addCleanup(gate.stop)

    def _check(self, inputs, geometry, result=None):
        expected = _scalar_reference(*inputs, **geometry)
        if result is None:
            result = metadata.try_build_dspark_metadata(*inputs, **geometry)
        self.assertIsNotNone(result)
        for actual, reference in zip(result, expected):
            self.assertEqual(actual.dtype, reference.dtype)
            self.assertEqual(actual.shape, reference.shape)
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.device, inputs[0].device)
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        return result

    def test_varied_geometry_padding_boundaries_and_integer_dtypes(self):
        rng = random.Random(41)
        for batch, gamma, window, entries, tokens in (
            (1, 1, 0, 8, 16),
            (3, 3, 7, 16, 64),
            (5, 5, 128, 256, 8192),
            (4, 7, 255, 64, 16),
            (8, 5, 511, 256, 8192),
            (2, 33, 257, 17, 31),
            (4, 5, 128, 134, 65536),
            (8, 5, 128, 134, 65536),
        ):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(
                    batch=batch,
                    gamma=gamma,
                    window=window,
                    entries=entries,
                    tokens=tokens,
                    dtype=dtype,
                ):
                    prefix_values = [
                        0,
                        tokens - 1,
                        tokens,
                        2 * tokens + 1,
                        -1,
                        -gamma - 1,
                        tokens * 4 - 1,
                        1,
                    ]
                    prefix = torch.tensor(prefix_values[:batch], dtype=dtype)
                    query = prefix[:, None] + torch.arange(gamma, dtype=dtype)
                    query[-1, -1] = -1
                    active = torch.tensor(
                        [int(i % 3 != 1) for i in range(batch)], dtype=dtype
                    )
                    table = torch.tensor(
                        [
                            [rng.choice([-1, 0, 1, 2, 7]) for _ in range(4)]
                            for _ in range(batch)
                        ],
                        dtype=dtype,
                    )
                    table[0, 0] = 2
                    inputs = tuple(t.cuda() for t in (query, prefix, active, table))
                    geometry = dict(
                        gamma=gamma,
                        window_size=window,
                        entries_per_block=entries,
                        tokens_per_block=tokens,
                    )
                    self._check(inputs, geometry)
                    self._check((inputs[0].flatten(), *inputs[1:]), geometry)

    def test_query_writes_do_not_use_active_attention_mask(self):
        inputs = list(_inputs(batch=2, gamma=3, device="cuda"))
        inputs[2].zero_()
        result = self._check(inputs, dict(_GEOMETRY, gamma=3))
        self.assertTrue((result.query_slots >= 0).all().item())
        self.assertTrue((result.global_indices == -1).all().item())
        self.assertTrue((result.topk_length == 0).all().item())

    def test_int64_positions_and_query_slot_precision(self):
        inputs = (
            torch.tensor([[0, (1 << 33), -1]], dtype=torch.int64, device="cuda"),
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            torch.tensor([True], device="cuda"),
            torch.tensor([[1 << 30]], dtype=torch.int64, device="cuda"),
        )
        result = self._check(
            inputs,
            dict(gamma=3, window_size=1, entries_per_block=16, tokens_per_block=16),
        )
        self.assertEqual(result.query_slots.cpu().tolist(), [1 << 34, -1, -1])

    def test_empty_batch_does_not_launch(self):
        inputs = _inputs(batch=0, device="cuda")
        with patch.object(metadata, "_dspark_metadata_kernel") as kernel:
            self._check(inputs, _GEOMETRY)
            kernel.__getitem__.assert_not_called()

    def test_unsupported_inputs_return_none_without_launch(self):
        query, prefix, active, table = _inputs(device="cuda")
        strided_query = torch.empty(3, 10, dtype=torch.int64, device="cuda")[:, ::2]
        cases = [
            ((query.float(), prefix, active, table), _GEOMETRY),
            ((query, prefix, active.float(), table), _GEOMETRY),
            ((strided_query, prefix, active, table), _GEOMETRY),
            ((query[:, :-1], prefix, active, table), _GEOMETRY),
            ((query, prefix, active, table[:1]), _GEOMETRY),
            ((query, prefix, active, table[:, :0]), _GEOMETRY),
            ((query, prefix, active, table.cpu()), _GEOMETRY),
            ((query, prefix, active, table), dict(_GEOMETRY, gamma=0)),
            ((query, prefix, active, table), dict(_GEOMETRY, window_size=-1)),
            ((query, prefix, active, table), dict(_GEOMETRY, entries_per_block=0)),
            ((query, prefix, active, table), dict(_GEOMETRY, tokens_per_block=0)),
        ]
        with patch.object(metadata, "_dspark_metadata_kernel") as kernel:
            for inputs, geometry in cases:
                self.assertIsNone(
                    metadata.try_build_dspark_metadata(*inputs, **geometry)
                )
            kernel.__getitem__.assert_not_called()

    def test_kernel_errors_are_not_hidden_by_fallback(self):
        with patch.object(metadata, "_dspark_metadata_kernel") as kernel:
            kernel.__getitem__.return_value.side_effect = RuntimeError("launch failed")
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                metadata.try_build_dspark_metadata(*_inputs(device="cuda"), **_GEOMETRY)

    def test_non_default_stream_producer_and_consumer_event_ordering(self):
        geometry = dict(
            gamma=5, window_size=128, entries_per_block=134, tokens_per_block=65536
        )
        host_prefix = torch.tensor([65535, 65536, 131071, -1], dtype=torch.int32)
        host_query = host_prefix[:, None].long() + torch.arange(5)
        host_active = torch.tensor([True, False, True, True])
        host_table = torch.tensor(
            [[2, 3, 0], [4, 0, 5], [0, 7, 8], [9, -1, 0]], dtype=torch.int32
        )
        templates = tuple(
            t.cuda() for t in (host_query, host_prefix, host_active, host_table)
        )
        inputs = tuple(torch.empty_like(t) for t in templates)
        producer, operator, consumer = (torch.cuda.Stream() for _ in range(3))
        input_ready, output_ready, consumed = (torch.cuda.Event() for _ in range(3))
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            for target, values in zip(inputs, templates):
                target.copy_(values)
            inputs[0].add_(1)
            inputs[1].add_(1)
            input_ready.record()
        with torch.cuda.stream(operator):
            operator.wait_event(input_ready)
            result = metadata.try_build_dspark_metadata(*inputs, **geometry)
            output_ready.record()
        with torch.cuda.stream(consumer):
            consumer.wait_event(output_ready)
            copies = metadata.DSparkMetadata(*(tensor.clone() for tensor in result))
            consumed.record()
        consumed.synchronize()
        expected = _scalar_reference(
            host_query + 1, host_prefix + 1, host_active, host_table, **geometry
        )
        for actual, reference in zip(copies, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def test_cuda_graph_replay_reads_changed_prefix_query_table_and_activity(self):
        for gamma in (3, 5):
            with self.subTest(gamma=gamma):
                geometry = dict(
                    gamma=gamma, window_size=7, entries_per_block=8, tokens_per_block=16
                )
                inputs = _inputs(batch=3, gamma=gamma, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        metadata.try_build_dspark_metadata(*inputs, **geometry)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = metadata.try_build_dspark_metadata(*inputs, **geometry)
                pointers = tuple(t.data_ptr() for t in result)
                for round_id in range(2):
                    prefix = torch.tensor(
                        [15 + round_id, 31 + round_id, -1], dtype=torch.int32
                    )
                    query = prefix[:, None].long() + torch.arange(gamma)
                    query[2].fill_(-1)
                    table = torch.tensor(
                        [
                            [1 + round_id, 5, 0, -1],
                            [0, -1, 7 + round_id, 3],
                            [9, 2, 0, 1],
                        ],
                        dtype=torch.int32,
                    )
                    active = torch.tensor([round_id == 0, round_id != 0, True])
                    for target, values in zip(inputs, (query, prefix, active, table)):
                        target.copy_(values)
                    graph.replay()
                    torch.cuda.synchronize()
                    self._check(inputs, geometry, result=result)
                    self.assertEqual(tuple(t.data_ptr() for t in result), pointers)


if __name__ == "__main__":
    unittest.main()

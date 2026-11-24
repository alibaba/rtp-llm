"""Slice byte contracts and graph buffer lifetimes; GPU runs are caller-owned.

The three-way descriptor in CUDA tests uses three allocations on one GPU. It
tests peer-pointer indexing/uneven tails, not actual multi-GPU P2P transport.
Importing this test module does not query or initialize CUDA.
"""

import unittest
import uuid
from contextlib import ExitStack, nullcontext
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _engram_quantized_slice as sliced
from rtp_llm.models_py.modules.dsv41 import _engram_quantized_triton as direct


def byte_oracle(weight, scales, ids):
    valid = (ids >= 0) & (ids < weight.shape[0])
    safe = ids.clamp(0, weight.shape[0] - 1).reshape(-1)
    q = weight.view(torch.uint8).index_select(0, safe).reshape(*ids.shape, 256)
    s = scales.index_select(0, safe).reshape(*ids.shape, 8)
    q[~valid], s[~valid] = 0, 127
    groups = s.reshape(ids.shape[0], 48, 4).to(torch.int64)
    packed = sum(groups[..., i] << (8 * i) for i in range(4)).to(torch.int32)
    storage = torch.full(
        (48, (ids.shape[0] + 3) // 4 * 4), 0x7F7F7F7F, dtype=torch.int32
    )
    storage[:, : ids.shape[0]] = packed.T
    return q.reshape(ids.shape[0], 6144), storage


def partitions(tokens):
    boundaries = [tokens * index // 8 for index in range(9)]
    return list(zip(boundaries[:-1], boundaries[1:]))


class SliceMetadataTest(unittest.TestCase):
    def test_positive_request_and_exact_slice_bounds(self):
        for tokens in (1, 37, 384, 385):
            for start, end in partitions(tokens):
                for grid in (64, 80, 96, 120, 124, 128, 132, 148):
                    sliced._validate_range(tokens, start, end - start, grid)
            sliced._validate_range(tokens, tokens, 0, 80)
        for args in (
            (0, 0, 0, 80),
            (37, -1, 1, 80),
            (37, 38, 0, 80),
            (37, 17, 21, 80),
            (37, 0, -1, 80),
            (37, True, 0, 80),
            (37, 0, True, 80),
            (37, 0, 1, True),
            (37, 0, 1, 0),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sliced._validate_range(*args)

    def test_allocation_delegates_and_rejects_empty_or_cpu(self):
        for ids in (
            torch.zeros((0, 24), dtype=torch.int64),
            torch.zeros((37, 24), dtype=torch.int64),
        ):
            with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
                with self.assertRaises(ValueError):
                    sliced.allocate_outputs(ids)
        sentinel = object()
        with patch.object(
            direct, "allocate_outputs", return_value=sentinel
        ) as allocate:
            self.assertIs(sliced.allocate_outputs("ids"), sentinel)
            allocate.assert_called_once_with("ids")

    def test_prepare_fails_before_launch_in_capture(self):
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(sliced, "launch") as launch:
            with self.assertRaisesRegex(RuntimeError, "outside capture"):
                sliced.prepare(None, None, None, None, 0, 1)
            launch.assert_not_called()

    def test_launch_preserves_full_layout_and_actual_grid_without_allocation(self):
        tokens = 37
        weight = torch.empty((17, 256), dtype=torch.uint8)
        scales = torch.empty((17, 8), dtype=torch.uint8)
        backing = torch.empty((tokens * 2, 48), dtype=torch.int64)
        ids = backing[::2, ::2]
        q = torch.empty((tokens, 6144), dtype=torch.float8_e4m3fn)
        storage = torch.empty((48, 40), dtype=torch.int32)
        outputs = direct.QuantizedLookupOutputs(q, storage.T[:tokens], storage)
        calls = []
        sentinel = object()

        class Kernel:
            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    calls.append((grid, args, kwargs))
                    return sentinel

                return invoke

        def cpu_metadata(tensor, dtype, ndim):
            return (
                isinstance(tensor, torch.Tensor)
                and tensor.dtype == dtype
                and tensor.ndim == ndim
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(direct, "_tensor", side_effect=cpu_metadata)
            )
            stack.enter_context(
                patch.object(direct, "supports_device", return_value=True)
            )
            stack.enter_context(
                patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext())
            )
            stack.enter_context(
                patch.object(sliced, "_lookup_quantized_slice_kernel", Kernel())
            )
            stack.enter_context(
                patch.object(
                    direct,
                    "_prepare_kernel_cached",
                    side_effect=AssertionError("driver mutation"),
                )
            )
            stack.enter_context(
                patch.object(torch, "empty", side_effect=AssertionError("allocation"))
            )
            self.assertIs(
                sliced.launch(weight, scales, ids, outputs, 17, 20, grid=132), sentinel
            )
            # IDs/output may be CPU only because the CUDA metadata gate is
            # explicitly mocked here; no driver or kernel has been invoked.
            bad = direct.QuantizedLookupOutputs(q[:1], storage.T[:1], storage)
            with self.assertRaises(ValueError):
                sliced.launch(weight, scales, ids, bad, 17, 20)
        grid, args, options = calls[0]
        self.assertEqual(grid, (132,))
        self.assertIs(args[2], ids)
        self.assertIs(args[3], outputs.values)
        self.assertIs(args[4], outputs.scales)
        self.assertEqual((args[5], args[7], args[8], args[9]), (37, 96, 2, 40))
        self.assertEqual(args[12:14], (17, 20))
        self.assertEqual(options, {"GRID": 132, "SHARDED": False, "num_warps": 1})
        self.assertEqual(len(calls), 1)

    def test_oracle_keeps_native_bytes_and_neutral_padding(self):
        weight = torch.arange(256).to(torch.uint8).reshape(1, 256)
        scales = torch.tensor([[0, 1, 128, 255, 127, 126, 125, 124]], dtype=torch.uint8)
        ids = torch.zeros((1, 24), dtype=torch.int64)
        ids[0, 1:3] = torch.tensor([-1, 1])
        q, sf = byte_oracle(weight, scales, ids)
        self.assertTrue(torch.equal(q[:, :256], weight))
        self.assertFalse(q[:, 256:768].any().item())
        self.assertEqual(sf[0, 0].item(), -8388352)
        self.assertTrue(torch.all(sf[2:6, 0] == 0x7F7F7F7F).item())
        self.assertTrue(torch.all(sf[:, 1:] == 0x7F7F7F7F).item())


class QuantizedSliceCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires a caller-reserved CUDA GPU")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("requires SM100 or SM103")
        from rtp_llm.models_py.modules.dsv41._engram_scale_shards import (
            ShardedScaleDescriptor,
            _ScaleShardOwner,
        )
        from rtp_llm.models_py.modules.dsv41.engram import _PinnedSharedTable

        self.device = torch.device("cuda", torch.cuda.current_device())
        rows = torch.arange(17)[:, None]
        self.weight = ((rows * 17 + torch.arange(256)[None, :] * 29) % 256).to(
            torch.uint8
        )
        self.scales = ((rows * 13 + torch.arange(8)[None, :] * 31) % 256).to(
            torch.uint8
        )
        self.scales[0, :2] = torch.tensor([0, 255], dtype=torch.uint8)
        self.pinned = _PinnedSharedTable(
            self.weight.view(torch.float8_e4m3fn),
            self.scales,
            "quantized-slice-" + uuid.uuid4().hex,
            self.device,
        )
        self.weight_uva = self.pinned.cuda_view(self.pinned.weight)
        self.local_scales = self.scales.to(self.device)
        # Same-device synthetic peer allocations exercise pointer arithmetic.
        # Keep all three, not merely shard0, alive until every graph is reset.
        self.peer_allocations = []
        for rank in range(3):
            shard = torch.full((6, 8), 173, dtype=torch.uint8, device=self.device)
            count = min(6, 17 - rank * 6)
            shard[:count].copy_(self.scales[rank * 6 : rank * 6 + count])
            self.peer_allocations.append(shard)
        pointers = torch.tensor(
            [shard.data_ptr() for shard in self.peer_allocations],
            dtype=torch.uint64,
            device=self.device,
        )
        owner = _ScaleShardOwner(
            self.peer_allocations[0],
            pointers,
            tuple(self.peer_allocations),
            None,
            "same-device-test-only",
            17,
            3,
            0,
            6,
            (0, 1, 2),
        )
        self.descriptor = ShardedScaleDescriptor(
            self.peer_allocations[0], pointers, 17, 3, 0, 6, owner
        )
        torch.cuda.synchronize(self.device)

    def tearDown(self):
        if hasattr(self, "device"):
            torch.cuda.synchronize(self.device)

    def _ids(self, tokens, generation):
        ids = (torch.arange(tokens * 24).reshape(tokens, 24) * 7 + generation) % 17
        ids[:, -4:] = torch.tensor([-1, 17, -(2**63), 2**63 - 1])
        return ids

    def _buffers(self, tokens):
        aligned = (tokens + 3) // 4 * 4
        q_guard = torch.empty(
            tokens * 6144 + 512, dtype=torch.uint8, device=self.device
        )
        sf_guard = torch.empty(
            48 * aligned + 128, dtype=torch.int32, device=self.device
        )
        q = q_guard[256:-256].reshape(tokens, 6144).view(torch.float8_e4m3fn)
        storage = sf_guard[64:-64].reshape(48, aligned)
        return direct.QuantizedLookupOutputs(q, storage.T[:tokens], storage), (
            q_guard,
            sf_guard,
        )

    def _poison(self, guards):
        guards[0].fill_(0xA5)
        guards[1].fill_(-1)

    def _assert(self, outputs, guards, expected, covered, padding_written):
        q = outputs.values.view(torch.uint8).cpu()
        sf = outputs.scale_storage.cpu()
        tokens = q.shape[0]
        self.assertTrue(torch.equal(q[covered], expected[0][covered]))
        self.assertTrue(
            torch.equal(sf[:, :tokens][:, covered], expected[1][:, :tokens][:, covered])
        )
        self.assertTrue(torch.all(q[~covered] == 0xA5).item())
        self.assertTrue(torch.all(sf[:, :tokens][:, ~covered] == -1).item())
        self.assertTrue(
            torch.all(sf[:, tokens:] == (0x7F7F7F7F if padding_written else -1)).item()
        )
        for region, value in (
            (guards[0][:256], 0xA5),
            (guards[0][-256:], 0xA5),
            (guards[1][:64], -1),
            (guards[1][-64:], -1),
        ):
            self.assertTrue(torch.all(region == value).item())

    @torch.inference_mode()
    def test_individual_neighbors_padding_and_complete_pair(self):
        for scales in (self.local_scales, self.descriptor):
            for tokens in (1, 37, 384, 385):
                backing = torch.empty(
                    (tokens * 2, 48), dtype=torch.int64, device=self.device
                )
                live = backing[::2, ::2]
                cpu_ids = self._ids(tokens, 0)
                live.copy_(cpu_ids)
                expected = byte_oracle(self.weight, self.scales, cpu_ids)
                outputs, guards = self._buffers(tokens)
                spans = [(0, 0), *partitions(tokens), (tokens, tokens)]
                for start, end in spans:
                    sliced.prepare(
                        self.weight_uva, scales, live, outputs, start, end - start
                    )
                    self._poison(guards)
                    sliced.launch(
                        self.weight_uva, scales, live, outputs, start, end - start
                    )
                    torch.cuda.synchronize(self.device)
                    covered = torch.zeros(tokens, dtype=torch.bool)
                    covered[start:end] = True
                    self._assert(
                        outputs,
                        guards,
                        expected,
                        covered,
                        end > start and end == tokens,
                    )
                self._poison(guards)
                for start, end in reversed(partitions(tokens)):
                    sliced.launch(
                        self.weight_uva, scales, live, outputs, start, end - start
                    )
                torch.cuda.synchronize(self.device)
                self._assert(
                    outputs,
                    guards,
                    expected,
                    torch.ones(tokens, dtype=torch.bool),
                    True,
                )
                whole = direct.allocate_outputs(live)
                direct.launch_epilogue(self.weight_uva, scales, live, whole)
                torch.cuda.synchronize(self.device)
                self.assertTrue(
                    torch.equal(
                        whole.values.view(torch.uint8), outputs.values.view(torch.uint8)
                    )
                )
                self.assertTrue(torch.equal(whole.scale_storage, outputs.scale_storage))

    @torch.inference_mode()
    def test_changed_graph_inputs_sizes_and_auxiliary_joins(self):
        capture = torch.cuda.Stream(device=self.device)
        auxiliary = torch.cuda.Stream(device=self.device)
        graphs = []
        try:
            for scales in (self.local_scales, self.descriptor):
                for tokens in (1, 37, 384, 385):
                    backing = torch.empty(
                        (tokens * 2, 48), dtype=torch.int64, device=self.device
                    )
                    live = backing[::2, ::2]
                    live.copy_(self._ids(tokens, 0))
                    outputs, guards = self._buffers(tokens)
                    for start, end in partitions(tokens):
                        sliced.prepare(
                            self.weight_uva, scales, live, outputs, start, end - start
                        )
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture):
                        for start, end in partitions(tokens):
                            auxiliary.wait_stream(capture)
                            with torch.cuda.stream(auxiliary):
                                sliced.launch(
                                    self.weight_uva,
                                    scales,
                                    live,
                                    outputs,
                                    start,
                                    end - start,
                                )
                            capture.wait_stream(auxiliary)
                        outputs.values.record_stream(capture)
                        outputs.scales.record_stream(capture)
                    graphs.append((graph, live, outputs, guards, tokens))
            for graph, live, outputs, guards, tokens in reversed(graphs):
                addresses = (outputs.values.data_ptr(), outputs.scales.data_ptr())
                for generation in (1, 2, 3):
                    cpu_ids = self._ids(tokens, generation)
                    live.copy_(cpu_ids)
                    self._poison(guards)
                    torch.cuda.synchronize(self.device)
                    graph.replay()
                    torch.cuda.synchronize(self.device)
                    self._assert(
                        outputs,
                        guards,
                        byte_oracle(self.weight, self.scales, cpu_ids),
                        torch.ones(tokens, dtype=torch.bool),
                        True,
                    )
                    self.assertEqual(
                        addresses,
                        (outputs.values.data_ptr(), outputs.scales.data_ptr()),
                    )
        finally:
            torch.cuda.synchronize(self.device)
            for graph, *_ in graphs:
                graph.reset()


if __name__ == "__main__":
    unittest.main()

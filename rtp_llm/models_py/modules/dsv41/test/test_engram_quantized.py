"""Native FP8 lookup bytes, graph inputs, default WKV and prefetch ownership.

Small synthetic tables exercise correctness only, never performance. CUDA tests
must run via the repository's guarded GPU test runner. CPU metadata tests do not
establish B200 runtime or numerical validation.
"""

import gc
import os
import unittest
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import _engram_decode_prefetch as prefetch
from rtp_llm.models_py.modules.dsv41 import _engram_quantized_triton as direct


def byte_oracle(weight, scales, ids):
    """CPU indexing and explicit little-endian packing, independent of Triton."""
    rows = ids.shape[0]
    valid = (ids >= 0) & (ids < weight.shape[0])
    safe = ids.clamp(0, weight.shape[0] - 1).reshape(-1)
    values = weight.view(torch.uint8).index_select(0, safe).reshape(rows, 24, 256)
    factors = scales.view(torch.uint8).index_select(0, safe).reshape(rows, 24, 8)
    values[~valid] = 0
    factors[~valid] = 127
    factors = factors.reshape(rows, 48, 4).to(torch.int64)
    words = sum(factors[:, :, byte] << (byte * 8) for byte in range(4)).to(torch.int32)
    storage = torch.full((48, (rows + 3) // 4 * 4), 0x7F7F7F7F, dtype=torch.int32)
    storage[:, :rows] = words.T
    return values.reshape(rows, 6144), storage


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertTrue(
        torch.equal(
            actual.detach().cpu().contiguous().view(torch.uint8),
            expected.detach().cpu().contiguous().view(torch.uint8),
        )
    )


class QuantizedMetadataTest(unittest.TestCase):
    def test_sm100_and_sm103_share_capability_gate_and_tuning(self):
        # Metadata only: this never establishes numerical/runtime B200 support.
        try:
            for capability, supported in (
                ((10, 0), True),
                ((10, 3), True),
                ((9, 0), False),
                ((8, 9), False),
                ((12, 0), False),
            ):
                direct._supports_cuda_device.cache_clear()
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ) as query:
                    self.assertEqual(direct.supports_device("cuda:0"), supported)
                    query.assert_called_once()
                self.assertEqual(direct._gather_grid(148), 148)
            direct._supports_cuda_device.cache_clear()
            with patch.object(torch.cuda, "get_device_capability") as query:
                self.assertFalse(direct.supports_device("cpu"))
                query.assert_not_called()
        finally:
            direct._supports_cuda_device.cache_clear()

    def test_explicit_cta_budgets_are_preserved(self):
        with patch.object(torch.cuda, "get_device_properties") as query:
            for grid in (1, 80, 120, 128, 132, 148, 160):
                self.assertEqual(direct._gather_grid(grid), grid)
            for invalid in (None, 0, -1, True, 148.0, "148"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    direct._gather_grid(invalid)
            query.assert_not_called()

    def test_default_grid_uses_cached_device_sm_count(self):
        direct._device_sm_count.cache_clear()
        try:
            with patch.object(
                torch.cuda,
                "get_device_properties",
                side_effect=[
                    SimpleNamespace(multi_processor_count=148),
                    SimpleNamespace(multi_processor_count=132),
                ],
            ) as query:
                self.assertEqual(direct._gather_grid(None, "cuda:0"), 148)
                self.assertEqual(direct._gather_grid(None, "cuda:0"), 148)
                self.assertEqual(direct._gather_grid(None, "cuda:1"), 132)
                self.assertEqual(query.call_count, 2)
        finally:
            direct._device_sm_count.cache_clear()

    def test_cpu_and_wrong_geometry_reject_before_allocation(self):
        invalid = [
            torch.zeros((3, 24), dtype=torch.int64),
            torch.zeros((3, 23), dtype=torch.int64),
            torch.zeros((0, 24), dtype=torch.int64),
            torch.zeros((3, 24), dtype=torch.int32),
        ]
        with patch.object(
            torch, "empty", side_effect=AssertionError("unexpected allocation")
        ):
            for indices in invalid:
                with self.subTest(shape=tuple(indices.shape), dtype=indices.dtype):
                    with self.assertRaises(ValueError):
                        direct.allocate_outputs(indices)

    def test_byte_oracle_neutral_invalid_scale_and_word_order(self):
        weight = torch.zeros((1, 256), dtype=torch.uint8)
        scales = torch.tensor([[0, 1, 128, 255, 127, 126, 125, 124]], dtype=torch.uint8)
        ids = torch.zeros((1, 24), dtype=torch.int64)
        ids[0, 1] = -1
        values, packed = byte_oracle(weight, scales, ids)
        self.assertEqual(packed[0, 0].item(), -8388352)  # 0xff800100
        self.assertEqual(packed[2, 0].item(), 0x7F7F7F7F)
        self.assertTrue(torch.all(packed[:, 1:] == 0x7F7F7F7F).item())
        self.assertFalse(values.any().item())

    def test_tuple_prefetch_records_both_outputs_and_releases_on_consume(self):
        from rtp_llm.models_py.modules.dsv41.engram import _EngramLookupWork

        class TrackedTensor(torch.Tensor):
            def record_stream(self, stream):
                self.recorded.append(stream)

        values = torch.empty((2, 6144), dtype=torch.float8_e4m3fn).as_subclass(
            TrackedTensor
        )
        scales = torch.empty((2, 48), dtype=torch.int32).as_subclass(TrackedTensor)
        values.recorded, scales.recorded = [], []
        ids = torch.zeros((2, 24), dtype=torch.int64)
        hidden = torch.zeros((2, 3, 64), dtype=torch.bfloat16)
        stream, done = Mock(), Mock()
        work = _EngramLookupWork(object(), ids, (values, scales), done)
        with patch.object(torch.cuda, "current_stream", return_value=stream):
            returned = work.consume(ids, hidden)
        self.assertIs(returned[0], values)
        self.assertIs(returned[1], scales)
        self.assertEqual(values.recorded, [stream])
        self.assertEqual(scales.recorded, [stream])
        stream.wait_event.assert_called_once_with(done)
        done.synchronize.assert_not_called()
        self.assertIsNone(work.rows)
        self.assertIsNone(work.embedding)

    def test_decode_tuple_consumer_on_new_stream_records_both_outputs(self):
        class TrackedTensor(torch.Tensor):
            def record_stream(self, stream):
                self.recorded.append(stream)

        values = torch.empty((2, 6144), dtype=torch.float8_e4m3fn).as_subclass(
            TrackedTensor
        )
        scales = torch.empty((2, 48), dtype=torch.int32).as_subclass(TrackedTensor)
        values.recorded, scales.recorded = [], []
        work = prefetch._LookupWork.__new__(prefetch._LookupWork)
        work.ids = torch.zeros((2, 24), dtype=torch.int64)
        work.rows, work.joined = (values, scales), set()
        work.current = Mock()
        work.module = SimpleNamespace(embed_tokens=object())
        work.state = SimpleNamespace(stream=Mock(), matches=Mock(return_value=True))
        consumer = Mock()
        with patch.object(torch.cuda, "current_stream", return_value=consumer):
            returned = work.consume(
                work.ids, torch.zeros((2, 3, 64), dtype=torch.bfloat16)
            )
            work.wait(consumer)
        self.assertIs(returned, work.rows)
        consumer.wait_stream.assert_called_once_with(work.state.stream)
        self.assertEqual(values.recorded, [consumer])
        self.assertEqual(scales.recorded, [consumer])

    def test_ordinary_linear_keeps_legacy_lookup_route(self):
        from rtp_llm.models_py.modules.dsv41.engram import (
            Engram,
            _lookup_for_projection,
        )

        embedding = Mock()
        embedding._quantized_lookup_supported = True
        module = Engram(
            SimpleNamespace(n_hash_cols=24, head_dim=256),
            0,
            embedding,
            torch.nn.Linear(6144, 256, bias=False),
            torch.ones((3, 64)),
            torch.ones((3, 64)),
            1e-20,
        )
        self.assertFalse(module._direct_fp8_lookup)
        ids = torch.zeros((2, 24), dtype=torch.int64)
        _lookup_for_projection(module, ids, ids.device)
        embedding.assert_called_once_with(ids, ids.device)
        embedding.lookup_quantized.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class QuantizedLookupCUDA(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("native MXFP8 experiment requires SM100 or SM103")
        from rtp_llm.models_py.modules.dsv41.engram import (
            HostEngramEmbedding,
            _PinnedSharedTable,
        )

        torch.manual_seed(20261025)
        self.environment = patch.dict(os.environ, {"DSV41_ENGRAM_DEVICE_SCALES": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        row = torch.arange(4096, dtype=torch.int64)[:, None]
        col = torch.arange(256, dtype=torch.int64)[None, :]
        self.weight = (
            ((row * 17 + col * 29 + row // 16) % 256)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        self.scales = ((row * 13 + torch.arange(8)[None, :] * 31) % 256).to(torch.uint8)
        self.pinned = _PinnedSharedTable(
            self.weight, self.scales, "quantized-byte-test-" + uuid.uuid4().hex, "cuda"
        )
        self.embedding = HostEngramEmbedding(
            self.pinned.weight, self.pinned.scales, pinned=self.pinned
        )
        self.addCleanup(torch.cuda.synchronize)

    def _ids(self, tokens, generation):
        ids = (
            torch.arange(tokens * 24, dtype=torch.int64).reshape(tokens, 24) * 37
            + generation * 53
        ) % 4096
        ids[::3, ::4] = -1
        ids[1::7, 1::5] = 4096
        return ids

    def _assert_output(self, output, expected_values, expected_storage):
        tokens = expected_values.shape[0]
        self.assertEqual(output.values.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.values.shape, (tokens, 6144))
        self.assertTrue(output.values.is_contiguous())
        self.assertEqual(output.scales.dtype, torch.int32)
        self.assertEqual(output.scales.shape, (tokens, 48))
        self.assertEqual(output.scales.stride(), (1, (tokens + 3) // 4 * 4))
        assert_bytes(self, output.values.view(torch.uint8), expected_values)
        assert_bytes(self, output.scale_storage, expected_storage)

    @torch.inference_mode()
    def test_changed_graph_inputs_all_bytes_tail_strides_invalid_and_padding(self):
        for tokens in (1, 7, 19, 32, 37, 384, 385, 1024):
            ids_storage = torch.empty(
                (tokens * 2 + 2, 50), dtype=torch.int64, device="cuda"
            )
            live = ids_storage[1 : 1 + tokens * 2 : 2, 1:49:2]
            self.assertFalse(live.is_contiguous())
            live.copy_(self._ids(tokens, 0))
            with self.subTest(tokens=tokens):
                buffers = direct.allocate_outputs(live)
                prepared = direct.prepare_lookup(*self.embedding._uva, live, buffers)
                self.assertEqual(
                    prepared["resources"]["epilogue"]["grid"],
                    [self.embedding._num_sms, 1, 1],
                )
                if tokens == 19 and self.embedding._num_sms == 148:
                    resource = prepared["resources"]["epilogue"]
                    self.assertEqual(resource["block_rows"], 4)
                    self.assertEqual(resource["num_warps"], 1)
                if tokens == 384:
                    resource = prepared["resources"]["epilogue"]
                    self.assertEqual(resource["block_rows"], 16)
                    self.assertEqual(resource["num_warps"], 4)
                    self.assertEqual(
                        resource["loop_iterations_max"],
                        (tokens * 24 - 1) // (self.embedding._num_sms * 16) + 1,
                    )
                graph = torch.cuda.CUDAGraph()
                pointers = (buffers.values.data_ptr(), buffers.scales.data_ptr())
                try:
                    with torch.cuda.graph(graph):
                        direct.launch_epilogue(*self.embedding._uva, live, buffers)
                    previous = None
                    for generation in range(4):
                        cpu_ids = self._ids(tokens, generation + 1)
                        expected_values, expected_storage = byte_oracle(
                            self.weight, self.scales, cpu_ids
                        )
                        if previous is not None:
                            self.assertFalse(torch.equal(previous, expected_values))
                        previous = expected_values
                        live.copy_(cpu_ids)
                        buffers.values.view(torch.uint8).fill_(0xA5)
                        buffers.scale_storage.fill_(-1)
                        graph.replay()
                        torch.cuda.synchronize()
                        self._assert_output(buffers, expected_values, expected_storage)
                        self.assertEqual(
                            (buffers.values.data_ptr(), buffers.scales.data_ptr()),
                            pointers,
                        )
                finally:
                    torch.cuda.synchronize()
                    graph.reset()

    @torch.inference_mode()
    def test_output_guards_and_rejected_alias_or_dtype(self):
        tokens, aligned = 37, 40
        live = self._ids(tokens, 1).cuda()
        value_owner = torch.full(
            (tokens * 6144 + 32,), 0xA5, dtype=torch.uint8, device="cuda"
        )
        values = value_owner[16:-16].reshape(tokens, 6144).view(torch.float8_e4m3fn)
        scale_owner = torch.full(
            (48 * aligned + 8,), 0x12345678, dtype=torch.int32, device="cuda"
        )
        storage = scale_owner[4:-4].reshape(48, aligned)
        output = direct.QuantizedLookupOutputs(values, storage.T[:tokens], storage)
        expected_values, expected_storage = byte_oracle(
            self.weight, self.scales, live.cpu()
        )
        direct.launch_epilogue(*self.embedding._uva, live, output)
        torch.cuda.synchronize()
        self._assert_output(output, expected_values, expected_storage)
        self.assertTrue(torch.all(value_owner[:16] == 0xA5).item())
        self.assertTrue(torch.all(value_owner[-16:] == 0xA5).item())
        self.assertTrue(torch.all(scale_owner[:4] == 0x12345678).item())
        self.assertTrue(torch.all(scale_owner[-4:] == 0x12345678).item())
        bad = direct.QuantizedLookupOutputs(values.float(), output.scales, storage)
        with self.assertRaises(ValueError):
            direct.launch_epilogue(*self.embedding._uva, live, bad)
        alias = live.untyped_storage()
        aliased_values = torch.empty(0, dtype=torch.float8_e4m3fn, device="cuda").set_(
            alias, 0, (tokens, 6144), (6144, 1)
        )
        with self.assertRaises(ValueError):
            direct.launch_epilogue(
                *self.embedding._uva,
                live,
                direct.QuantizedLookupOutputs(aliased_values, output.scales, storage)
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class QuantizedEngramIntegrationCUDA(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("requires actual SM100 or SM103")
        from rtp_llm.models_py.modules.dsv41.engram import (
            Engram,
            EngramLayout,
            HostEngramEmbedding,
            _PinnedSharedTable,
        )
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        self.environment = patch.dict(
            os.environ,
            {"DSV41_ENGRAM_DEVICE_SCALES": "1", "DSV41_DECODE_ENGRAM_PREFETCH": "1"},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("DSV41_ENGRAM_DIRECT_FP8", None)
        torch.manual_seed(20261025)
        self.weight = (torch.randn(4096, 256) * 16).to(torch.float8_e4m3fn)
        self.scales = torch.randint(117, 122, (4096, 8), dtype=torch.uint8)
        self.pinned = _PinnedSharedTable(
            self.weight,
            self.scales,
            "quantized-default-test-" + uuid.uuid4().hex,
            "cuda",
        )
        self.embedding = HostEngramEmbedding(
            self.pinned.weight, self.pinned.scales, pinned=self.pinned
        )
        layout = EngramLayout(
            dict(
                engram_layer_ids=[1],
                engram_num_embeddings=[4096],
                engram_max_ngram_size=4,
                engram_n_heads=8,
                engram_head_dim=256,
                engram_compressed_vocab_size=32,
                engram_pad_token_id=2,
                engram_vocab_size=32,
            )
        )
        weight = (torch.randn(256, 6144, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        scales = torch.ones(8, 192, device="cuda").to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(weight, scales)
        self.module = Engram(
            layout,
            0,
            self.embedding,
            linear,
            torch.randn(3, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(3, 64, device="cuda", dtype=torch.bfloat16),
            1e-20,
        )
        self.assertTrue(self.embedding._quantized_lookup_supported)
        self.assertTrue(self.module._direct_fp8_lookup)
        self.model = torch.nn.Module()
        self.model.layers = [
            SimpleNamespace(engram=None, engram_hashes=None),
            SimpleNamespace(engram=self.module, engram_hashes=None),
        ]
        self.addCleanup(torch.cuda.synchronize)

    def _reference(self, ids, hidden, mask):
        from rtp_llm.models_py.modules.dsv41.engram import gated_engram_residual

        q, storage = byte_oracle(self.weight, self.scales, ids.cpu())
        q = q.to("cuda").view(torch.float8_e4m3fn)
        scale_storage = storage.cuda()
        scale_view = scale_storage.T[: ids.shape[0]]
        kv = self.module.wkv.forward_quantized(q, scale_view)
        return gated_engram_residual(
            hidden.clone(),
            kv,
            self.module.q_weight,
            self.module.k_weight,
            self.module.eps,
            mask,
        )

    def _run(self, ids, hidden, mask, metadata, asynchronous=True, abandon=False):
        self.model.layers[1].engram_hashes = ids
        with patch.dict(
            os.environ, {"DSV41_DECODE_ENGRAM_PREFETCH": "1" if asynchronous else "0"}
        ):
            plan = prefetch.DecodeEngramPrefetch(self.model, metadata)
        try:
            plan.before_layer(0)
            if abandon:
                return None
            return self.module(hidden, ids, mask)
        finally:
            plan.close()

    @torch.inference_mode()
    def test_default_sync_and_decode_prefetch_use_native_pair(self):
        ids = torch.randint(4096, (7, 24), device="cuda", dtype=torch.int64)
        hidden = torch.randn(7, 3, 64, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(7, dtype=torch.bool, device="cuda")
        mask[::3] = False
        expected = self._reference(ids, hidden, mask)
        metadata = SimpleNamespace(batch_size=7, q_len_per_req=1)
        with patch.object(
            self.module.wkv,
            "_quantize_input",
            side_effect=AssertionError("unexpected BF16 requantization"),
        ):
            for asynchronous in (False, True):
                actual = self._run(ids, hidden.clone(), mask, metadata, asynchronous)
                torch.cuda.synchronize()
                assert_bytes(self, actual, expected)
                self.assertIsNone(self.module._lookup_work)
                assert_bytes(self, actual[~mask], hidden[~mask])

    @torch.inference_mode()
    def test_explicit_diagnostic_off_uses_legacy_requantization(self):
        from rtp_llm.models_py.modules.dsv41.engram import Engram

        with patch.dict(os.environ, {"DSV41_ENGRAM_DIRECT_FP8": "0"}):
            legacy = Engram(
                self.module.layout,
                0,
                self.embedding,
                self.module.wkv,
                self.module.q_weight,
                self.module.k_weight,
                self.module.eps,
            )
        self.assertFalse(legacy._direct_fp8_lookup)
        ids = torch.randint(4096, (7, 24), device="cuda", dtype=torch.int64)
        hidden = torch.randn(7, 3, 64, device="cuda", dtype=torch.bfloat16)
        with patch.object(
            legacy.wkv, "_quantize_input", wraps=legacy.wkv._quantize_input
        ) as quantize:
            actual = legacy(hidden.clone(), ids)
        quantize.assert_called_once()
        expected = self.module(hidden.clone(), ids)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.001)

    @torch.inference_mode()
    def test_two_graph_sizes_changed_ids_masks_poison_and_scale_owner_reload(self):
        graphs = []
        try:
            for rows in (1, 37):
                ids = torch.randint(4096, (rows, 24), device="cuda", dtype=torch.int64)
                hidden = torch.randn(rows, 3, 64, device="cuda", dtype=torch.bfloat16)
                mask = torch.ones(rows, dtype=torch.bool, device="cuda")
                metadata = SimpleNamespace(batch_size=rows, q_len_per_req=1)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self._run(ids, hidden, mask, metadata)
                    self._run(ids, hidden, mask, metadata)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with patch.object(
                    self.module.wkv,
                    "_quantize_input",
                    side_effect=AssertionError("requantization in capture"),
                ):
                    with torch.cuda.graph(graph, stream=stream):
                        actual = self._run(ids, hidden, mask, metadata)
                graphs.append(
                    (graph, ids, hidden, mask, metadata, actual, actual.data_ptr())
                )
            for generation in range(4):
                for graph, ids, hidden, mask, metadata, actual, pointer in graphs[
                    :: (-1 if generation % 2 else 1)
                ]:
                    ids.copy_(
                        torch.randint(4096, ids.shape, device="cuda", dtype=torch.int64)
                    )
                    ids[::3, ::4] = -1
                    hidden.normal_()
                    mask.fill_(True)
                    mask[generation % mask.numel()] = False
                    actual.fill_(float("nan"))
                    metadata._engram_decode_tables.validate()
                    graph.replay()
                    expected = self._reference(ids, hidden, mask)
                    torch.cuda.synchronize()
                    assert_bytes(self, actual, expected)
                    self.assertEqual(actual.data_ptr(), pointer)
                    self.assertIsNone(self.module._lookup_work)
            metadata = graphs[0][4]
            original = self.embedding._device_scale_lease
            reference = weakref.ref(original)
            self.embedding._device_scale_lease = object()
            del original
            gc.collect()
            with self.assertRaisesRegex(RuntimeError, "recreate decode CUDA graphs"):
                metadata._engram_decode_tables.validate()
            self.assertIsNotNone(reference())
            self.embedding._device_scale_lease = reference()
            metadata._engram_decode_tables.validate()
        finally:
            torch.cuda.synchronize()
            for graph, *_ in graphs:
                graph.reset()

    @torch.inference_mode()
    def test_abandoned_decode_prefetch_joins_pair_and_clears_pending(self):
        ids = torch.randint(4096, (37, 24), device="cuda", dtype=torch.int64)
        hidden = torch.randn(37, 3, 64, device="cuda", dtype=torch.bfloat16)
        metadata = SimpleNamespace(batch_size=37, q_len_per_req=1)
        self._run(ids, hidden, None, metadata)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                self._run(ids, hidden, None, metadata, abandon=True)
            graph.replay()
            torch.cuda.synchronize()
            self.assertIsNone(self.module._lookup_work)
        finally:
            torch.cuda.synchronize()
            graph.reset()


if __name__ == "__main__":
    unittest.main()

"""Decode prefetch ownership, changed graph inputs, batch switching and reload."""

import ast
import gc
import os
import unittest
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import _engram_decode_prefetch as prefetch


class DecodePrefetchCPU(unittest.TestCase):
    def test_default_off_does_not_touch_cuda_or_layers(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            torch.cuda, "current_stream"
        ) as current:
            work = prefetch.DecodeEngramPrefetch(object(), object())
            work.before_layer(0)
            work.close()
        current.assert_not_called()

    def test_cold_capture_falls_back_without_creating_stream(self):
        module = SimpleNamespace(
            q_weight=SimpleNamespace(is_cuda=True),
            embed_tokens=SimpleNamespace(_uva=object(), _pinned=object()),
        )
        with patch.dict(os.environ, {"DSV41_DECODE_ENGRAM_PREFETCH": "1"}), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(torch.cuda, "Stream") as stream:
            self.assertIsNone(prefetch.prepare(module))
        stream.assert_not_called()

    def test_replay_preparation_rejects_reload_before_metadata_update(self):
        root = Path(__file__).resolve().parents[1]
        for source in ("decode/decode_fmha_impl.py", "fp8/decode/decode_fmha_impl.py"):
            tree = ast.parse((root/source).read_text())
            method = next(
                node for cls in tree.body if isinstance(cls, ast.ClassDef)
                for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "prepare_cuda_graph"
            )
            for node in ast.walk(method):
                if isinstance(node, ast.arg):
                    node.annotation = None
            method.returns = None
            namespace = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), source, "exec"), namespace)
            owner = SimpleNamespace(validate=Mock(side_effect=RuntimeError("stale table")))
            runner = SimpleNamespace(metadata=SimpleNamespace(_engram_decode_tables=owner))
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "stale table"):
                namespace["prepare_cuda_graph"](runner, None)
            owner.validate.assert_called_once()

    def test_graph_owns_old_table_and_rejects_reload(self):
        model = torch.nn.Module()
        embedding = SimpleNamespace(_uva=object(), _pinned=object())
        module = SimpleNamespace(embed_tokens=embedding)
        model.layers = [SimpleNamespace(engram=module)]
        state = prefetch._LookupStream.__new__(prefetch._LookupStream)
        state.embedding, state.uva, state.pinned = (
            embedding, embedding._uva, embedding._pinned
        )
        resources = prefetch._GraphTables(model, ((0, module, state),))
        reference = weakref.ref(state)
        resources.validate()
        original_uva = embedding._uva
        for change in ("uva", "embedding", "module"):
            with self.subTest(change=change):
                if change == "uva":
                    embedding._uva = object()
                elif change == "embedding":
                    module.embed_tokens = object()
                else:
                    model.layers[0].engram = object()
                with self.assertRaisesRegex(RuntimeError, "recreate decode CUDA graphs"):
                    resources.validate()
                embedding._uva = original_uva
                module.embed_tokens = embedding
                model.layers[0].engram = module
        del state
        gc.collect()
        self.assertIsNotNone(reference())
        del resources
        gc.collect()
        self.assertIsNone(reference())

    def test_close_joins_abandoned_work_and_clears_only_its_own_pending(self):
        pending = object()
        module = SimpleNamespace(_lookup_work=pending)
        work = SimpleNamespace(module=module, wait=Mock())
        plan = prefetch.DecodeEngramPrefetch.__new__(prefetch.DecodeEngramPrefetch)
        plan.work = [work]
        plan.close()
        work.wait.assert_called_once()
        self.assertIs(module._lookup_work, pending)
        module._lookup_work = work
        plan.work = [work]
        plan.close()
        self.assertIsNone(module._lookup_work)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class DecodePrefetchCUDA(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules.dsv41.engram import (
            Engram, EngramLayout, HostEngramEmbedding, NgramHashState,
            _PinnedSharedTable,
        )

        self.context = patch.dict(os.environ, {"DSV41_DECODE_ENGRAM_PREFETCH": "1"})
        self.context.start()
        self.addCleanup(self.context.stop)
        torch.manual_seed(20261022)
        self.layout = EngramLayout(dict(
            engram_layer_ids=[1, 14], engram_num_embeddings=[4000, 4000],
            engram_max_ngram_size=4, engram_n_heads=2, engram_head_dim=32,
            engram_compressed_vocab_size=32, engram_pad_token_id=2,
            engram_vocab_size=32,
        ))
        self.hash = NgramHashState(self.layout, token_map=list(range(32)), device="cuda")
        self.model = torch.nn.Module()
        self.model.layers = [SimpleNamespace(engram=None, engram_hashes=None) for _ in range(15)]
        self.sources = []
        for index, layer in enumerate((1, 14)):
            weight = torch.randn(4000, 32).to(torch.float8_e4m3fn)
            scales = torch.randint(125, 129, (4000, 1), dtype=torch.uint8)
            pinned = _PinnedSharedTable(weight, scales, "decode-test-"+uuid.uuid4().hex, "cuda")
            embedding = HostEngramEmbedding(pinned.weight, pinned.scales, pinned=pinned)
            module = Engram(
                self.layout, index, embedding,
                torch.nn.Linear(192, 96, bias=False, device="cuda", dtype=torch.bfloat16),
                torch.randn(2, 32, device="cuda", dtype=torch.bfloat16),
                torch.randn(2, 32, device="cuda", dtype=torch.bfloat16), 1e-20,
            )
            self.model.layers[layer].engram = module
            self.sources.append((weight, scales))

    def _run(self, windows, hidden, metadata, asynchronous):
        hashes = self.hash(windows)
        for index, layer in enumerate((1, 14)):
            self.model.layers[layer].engram_hashes = hashes[:, index].contiguous()
        plan = prefetch.DecodeEngramPrefetch(self.model, metadata) if asynchronous else None
        outputs = []
        try:
            for index, layer in enumerate(self.model.layers):
                if plan is not None:
                    plan.before_layer(index)
                if layer.engram is not None:
                    outputs.append(layer.engram(hidden.clone(), layer.engram_hashes, windows[:, 0] >= 0))
            return outputs
        finally:
            if plan is not None:
                plan.close()

    @torch.inference_mode()
    def test_replay_changes_hashes_padding_and_switches_batch_graphs(self):
        graphs = []
        for batch, span in ((1, 1), (1, 6), (7, 6), (64, 6)):
            rows = batch * span
            metadata = SimpleNamespace(batch_size=batch, q_len_per_req=span)
            windows = torch.full((rows, 4), 7, device="cuda", dtype=torch.int32)
            hidden = torch.randn(rows, 2, 32, device="cuda", dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._run(windows, hidden, metadata, True)
                expected = self._run(windows, hidden, metadata, False)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self._run(windows, hidden, metadata, True)
            graphs.append((graph, metadata, windows, hidden, outputs, [x.data_ptr() for x in outputs]))
        for replay in range(12):
            # Revisit smaller graphs after larger ones; output storage is per graph.
            for graph, metadata, windows, hidden, outputs, pointers in graphs[::(-1 if replay % 2 else 1)]:
                windows.fill_((replay * 7 + 3) % 32)
                windows[::3] = -1
                hidden.normal_()
                metadata._engram_decode_tables.validate()
                graph.replay()
                expected = self._run(windows, hidden, metadata, False)
                torch.cuda.synchronize()
                self.assertEqual([x.data_ptr() for x in outputs], pointers)
                for actual, reference in zip(outputs, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_consumer_on_another_stream_and_abandoned_capture_work(self):
        metadata = SimpleNamespace(batch_size=1, q_len_per_req=6)
        windows = torch.full((6, 4), 7, device="cuda", dtype=torch.int32)
        hidden = torch.randn(6, 2, 32, device="cuda", dtype=torch.bfloat16)
        self._run(windows, hidden, metadata, True)
        origin, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        origin.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=origin):
            hashes = self.hash(windows)
            for index, layer in enumerate((1, 14)):
                self.model.layers[layer].engram_hashes = hashes[:, index].contiguous()
            plan = prefetch.DecodeEngramPrefetch(self.model, metadata)
            plan.before_layer(0)
            consumer.wait_stream(origin)
            with torch.cuda.stream(consumer):
                layer = self.model.layers[1]
                output = layer.engram(hidden.clone(), layer.engram_hashes)
            origin.wait_stream(consumer)
            # Publish L14 but deliberately omit its consumer: close must join it.
            plan.before_layer(12)
            plan.close()
        for token in (3, 11, 7):
            windows.fill_(token)
            graph.replay()
            expected = self._run(windows, hidden, metadata, False)[0]
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertTrue(all(layer.engram._lookup_work is None for layer in self.model.layers if layer.engram is not None))

    @torch.inference_mode()
    def test_reload_keeps_old_uva_alive_and_recaptures_new_table(self):
        from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding, _PinnedSharedTable

        metadata = SimpleNamespace(batch_size=2, q_len_per_req=6)
        windows = torch.full((12, 4), 7, device="cuda", dtype=torch.int32)
        hidden = torch.randn(12, 2, 32, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._run(windows, hidden, metadata, True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            old_outputs = self._run(windows, hidden, metadata, True)
        graph.replay()
        old_values = [x.clone() for x in old_outputs]
        torch.cuda.synchronize()
        module = self.model.layers[14].engram
        old_pinned = weakref.ref(module.embed_tokens._pinned)
        weight, scales = self.sources[1]
        replacement = _PinnedSharedTable((-weight.float()).to(torch.float8_e4m3fn), scales,
                                        "decode-reload-"+uuid.uuid4().hex, "cuda")
        module.embed_tokens = HostEngramEmbedding(replacement.weight, replacement.scales, pinned=replacement)
        gc.collect()
        self.assertIsNotNone(old_pinned())
        with self.assertRaisesRegex(RuntimeError, "recreate decode CUDA graphs"):
            metadata._engram_decode_tables.validate()
        # The retained old capture is still memory-safe until the runner drops it.
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(old_outputs, old_values):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        new_metadata = SimpleNamespace(batch_size=2, q_len_per_req=6)
        with torch.cuda.stream(stream):
            self._run(windows, hidden, new_metadata, True)
        stream.synchronize()
        new_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(new_graph, stream=stream):
            new_outputs = self._run(windows, hidden, new_metadata, True)
        new_graph.replay()
        expected = self._run(windows, hidden, new_metadata, False)
        torch.cuda.synchronize()
        for actual, reference in zip(new_outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertFalse(torch.equal(new_outputs[1], old_values[1]))
        del graph, metadata
        gc.collect()
        self.assertIsNone(old_pinned())


if __name__ == "__main__":
    unittest.main()

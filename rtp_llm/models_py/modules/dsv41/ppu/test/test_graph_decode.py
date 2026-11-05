"""Device decode continuation over native pages, tested against the old runner."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.decode import DecodeStepOrchestrator
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import bounded_workspace
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import (
    PPUDecodeGraphImpl,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_decode import GraphDecodeRunner
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import run_prefill
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class GraphDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selection = patch.dict(
            os.environ,
            {"RTP_V41_PPU_MOE_BACKEND": "grouped", "RTP_V41_PPU_MOE_PRECISION": "w4a4"},
        )
        selection.start()
        cls.addClassCleanup(selection.stop)
        cls.config = synthetic_config(max_position_embeddings=48)
        geo = cls.geometry = V41Geometry.from_config(cls.config)
        cls.ops = PPUOps(
            ep_size=1,
            num_routed_experts=geo.n_routed_experts,
            num_local_experts=geo.n_routed_experts,
            hidden_size=geo.hidden_size,
            inter_dim=geo.moe_inter_dim,
            swiglu_limit=geo.swiglu_limit,
            mhc_config=geo.mhc_config(),
            sm_scale=geo.sm_scale(),
            window=geo.window,
        )
        with tempfile.TemporaryDirectory(prefix="v41-graph-decode-") as root:
            checkpoint = write_synthetic_checkpoint(cls.config, Path(root), seed=827)
            cls.bundle = load_ppu_weights(cls.config, checkpoint)
        cls.bundle.bind(cls.ops)

    def seeded(self):
        histories = [list(range(9)), list(range(20, 32))]
        cache = PPUKVCache(self.config)
        run_prefill(
            self.bundle,
            self.ops,
            cache,
            torch.tensor(histories[0] + histories[1]),
            cu_seqlens=torch.tensor([0, 9, 21]),
        )
        reference = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=cache
        )
        for row, tokens in enumerate(histories):
            reference.add_request(row, tokens)
        native = NativePools(self.config, batch=2, blocks=20)
        NativeCacheBridge(self.config, native, native.inputs).export_requests(
            cache, [len(tokens) for tokens in histories]
        )
        self.set_lengths(native, [9, 12])
        impl = PPUDecodeGraphImpl(
            self.config, native, native.inputs, device="cpu", batch_size=2
        )
        core = PPUModelCore(self.config, self.bundle, self.ops, device="cpu")
        runner = GraphDecodeRunner(core, max_batch_size=2)
        return runner, impl, native, reference, histories

    @staticmethod
    def set_lengths(native, lengths):
        for entry in native.inputs.values():
            entry.is_prefill = False
            entry.input_lengths = torch.ones(len(lengths), dtype=torch.int32)
            entry.sequence_lengths = torch.tensor(lengths, dtype=torch.int32)
            entry.sequence_lengths_plus_1_device = entry.sequence_lengths + 1

    def inputs(self, native, tokens, histories):
        windows = []
        ngram = self.geometry.engram_max_ngram_size
        for row, token in enumerate(tokens):
            recent = histories[row][-(ngram - 1) :][::-1]
            windows.append([token] + recent + [-1] * (ngram - 1 - len(recent)))
        return NS(
            input_ids=torch.tensor(tokens, dtype=torch.int32),
            attention_inputs=native.inputs,
            engram_token_windows=torch.tensor(windows, dtype=torch.int32),
        )

    def assert_valid_cache(self, native, cache, lengths):
        actual_bridge = NativeCacheBridge(self.config, native, native.inputs)
        for row, length in enumerate(lengths):
            for owner, tag in cache.pool_keys():
                key = owner, tag
                positions = actual_bridge._positions(key, length)
                if tag in ("global_kv_1", "global_kv_2", "indexer_kv"):
                    slots = cache._slots_for_blocks(key, row, positions.numel())
                else:
                    slots = cache.slots_at_positions(owner, tag, row, positions)
                expected = cache._gather(key, slots)
                actual = actual_bridge._copy_rows(key, row, positions)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_tensor_decode_continues_both_compression_phases(self):
        runner, impl, native, reference, histories = self.seeded()
        for step, tokens in enumerate(([35, 40], [36, 41], [37, 42], [38, 43])):
            lengths = [len(history) for history in histories]
            self.set_lengths(native, lengths)
            impl.prepare_cuda_graph(native.inputs, valid_request_count=2)
            actual = runner.forward(self.inputs(native, tokens, histories), impl)
            expected = reference.step(dict(enumerate(tokens)))
            torch.testing.assert_close(actual, expected.hidden, rtol=0.005, atol=0.005)
            for row, token in enumerate(tokens):
                histories[row].append(token)
            self.assert_valid_cache(
                native, reference.cache.cache, [len(history) for history in histories]
            )
        runner.validate()

    def test_padding_executes_without_writing_another_request(self):
        runner, impl, native, reference, histories = self.seeded()
        before = {key: value.kv_cache_base.clone() for key, value in native.handles.items()}
        impl.prepare_cuda_graph(native.inputs, valid_request_count=0)
        output = runner.forward(self.inputs(native, [35, 40], histories), impl)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        for key, value in native.handles.items():
            self.assertTrue(torch.equal(value.kv_cache_base, before[key]), key)

        impl.prepare_cuda_graph(native.inputs, valid_request_count=1)
        output = runner.forward(self.inputs(native, [35, 40], histories), impl)
        expected = reference.step({0: 35})
        torch.testing.assert_close(output[:1], expected.hidden, rtol=0.005, atol=0.005)
        self.assertTrue(torch.equal(output[1:], torch.zeros_like(output[1:])))
        # Every second request's physical block remains byte-identical.
        for (owner, tag), handle in native.handles.items():
            blocks = native.inputs[tag].kv_cache_block_id_device[1].long()
            self.assertTrue(
                torch.equal(handle.kv_cache_base[blocks], before[(owner, tag)][blocks]),
                (owner, tag),
            )

    def test_attention_reads_only_topk_plus_window(self):
        runner, impl, native, reference, histories = self.seeded()
        selected = torch.tensor([[0, 1, -1, -1], [0, 2, 4, -1]], dtype=torch.int32)
        kv, indices, lengths = bounded_workspace(
            selected,
            cache=impl.cache,
            layer_id=6,
            source_layer=6,
            ratio=1,
            positions=torch.tensor([8, 11]),
            valid_mask=torch.tensor([True, False]),
            window=self.geometry.window,
        )
        self.assertEqual(kv.shape, (2 * 640, 1, 512))
        self.assertTrue(torch.equal(lengths, torch.full((2,), 640, dtype=torch.int32)))
        first = indices[0, 0]
        self.assertEqual(int((first >= 0).sum()), 2 + self.geometry.window)
        self.assertTrue(bool((first[first >= 0] < 640).all()))
        self.assertEqual(int(indices[1, 0, 0]), 640)
        self.assertTrue(torch.equal(kv[640:], torch.zeros_like(kv[640:])))


class GraphCodebookTest(unittest.TestCase):
    def test_prewarmed_encode_decode_never_recreates_host_constants(self):
        device = torch.device("cpu")
        values = torch.tensor(
            [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        )
        _quant.prepare_e2m1_codebooks(device)
        original = _quant._e2m1_codebooks(device)
        with patch.object(torch, "tensor", side_effect=AssertionError("host constant recreated")):
            _quant.prepare_e2m1_codebooks(device)
            encoded = _quant.encode_fp4_e2m1(values)
            decoded = _quant.decode_fp4_e2m1(encoded)
        retained = _quant._e2m1_codebooks(device)
        self.assertIs(retained[0], original[0])
        self.assertIs(retained[1], original[1])
        torch.testing.assert_close(decoded.float(), values, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.signbit(decoded), torch.signbit(values)))

    def test_cold_capture_fails_before_device_allocation(self):
        # An unused explicit device key exercises the guard without allocating
        # accelerator memory, even in the CPU contract-test process.
        device = torch.device("cuda:63")
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with patch.object(torch, "tensor", side_effect=AssertionError("cold H2D copy")):
                with self.assertRaisesRegex(RuntimeError, "prepared before Graph capture"):
                    _quant.prepare_e2m1_codebooks(device)


if __name__ == "__main__":
    unittest.main()

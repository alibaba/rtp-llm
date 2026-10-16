"""Small unregistered-table tests for CPU prefetch plus real Engram graphs."""

import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
)
from rtp_llm.models_py.modules.dsv41.engram_graph import CpuEngramGraphState
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


class Sm90EngramGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("test_sm90_engram_graph requires CUDA SM90")

    def setUp(self):
        torch.manual_seed(4127)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.layout = EngramLayout(dict(
            engram_layer_ids=[0, 1], engram_num_embeddings=[4000, 4000],
            engram_max_ngram_size=4, engram_n_heads=4, engram_head_dim=32,
            engram_compressed_vocab_size=32, engram_pad_token_id=2,
            engram_vocab_size=32,
        ))
        self.hash_state = NgramHashState(
            self.layout, token_map=list(range(32)), device=self.device
        )
        self.layers = []
        k, n = self.layout.n_hash_cols * self.layout.head_dim, 5 * 5120
        for layer_index in range(2):
            table = torch.randn(4000, 32).to(torch.float8_e4m3fn)
            table_scales = torch.randint(124, 130, (4000, 1), dtype=torch.uint8)
            # Plain CPU tensors: never create shared memory or register pages.
            embedding = HostEngramEmbedding(table, table_scales)
            weight = (torch.randn(n, k) * 0.02).to(torch.float8_e4m3fn)
            scales = torch.randint(125, 129, (n // 32, k // 32), dtype=torch.uint8)
            projection = V41MXFP8Linear(
                weight.to(self.device), scales.view(torch.float8_e8m0fnu).to(self.device)
            )
            self.assertIsNotNone(projection._sm90_linear)
            engram = Engram(
                self.layout, layer_index, embedding, projection,
                torch.randn(4, 5120, device=self.device).bfloat16(),
                torch.randn(4, 5120, device=self.device).bfloat16(), 1e-6,
            )
            self.layers.append(types.SimpleNamespace(engram=engram))
        self.model = types.SimpleNamespace(
            v4=types.SimpleNamespace(
                embed=types.SimpleNamespace(weight=torch.empty(0, device=self.device)),
                layers=self.layers,
            ),
            _engram_hash_state=self.hash_state,
            _engram_layers=((0, 0), (1, 1)),
            _v4_args=types.SimpleNamespace(v41_config={"image_token_id": 31}),
        )

    def _windows(self, count, shift=0):
        windows = (torch.arange(count * 4).reshape(count, 4) + shift) % 31
        if count:
            windows[0, 2] = -1
        if count > 1:
            windows[1, 0] = 31  # Image token must disable residual injection.
        if count > 2:
            windows[2, 0] = -1
        return windows.to(torch.int32)

    def _state(self, capacity):
        inputs = types.SimpleNamespace(
            input_ids=torch.zeros(capacity, dtype=torch.int32, device=self.device),
            engram_token_windows=self._windows(capacity),
        )
        with mock.patch(
            "rtp_llm.models_py.modules.dsv41.engram._register_shared_host_memory",
            side_effect=AssertionError("Graph prefetch must not register host tables"),
        ):
            return CpuEngramGraphState(self.model, inputs)

    @torch.inference_mode()
    def test_fixed_buffers_changed_histories_and_cleared_bucket_tail(self):
        for capacity in (1, 2, 4):
            state = self._state(capacity)
            addresses = {key: value.data_ptr() for key, value in state.rows.items()}
            mask_address = state.token_mask.data_ptr()
            for count in (capacity, 1, 0, capacity):
                for cuda_input in (False, True):
                    windows = self._windows(count, shift=7 + count)
                    source = windows.to(self.device) if cuda_input else windows
                    with mock.patch.object(
                        self.hash_state, "hash_token_windows",
                        wraps=self.hash_state.hash_token_windows,
                    ) as hashed:
                        state.prepare(source)
                        self.assertEqual(hashed.call_count, 1)
                    hashes = self.hash_state.hash_token_windows(
                        windows, dead_mask=windows == 31
                    )
                    for layer_id, layer in enumerate(self.layers):
                        expected = layer.engram.embed_tokens(hashes[:, layer_id], "cpu")
                        torch.testing.assert_close(
                            state.rows[layer_id][:count].cpu(), expected, rtol=0, atol=0
                        )
                        self.assertEqual(state.rows[layer_id].data_ptr(), addresses[layer_id])
                        self.assertEqual(int(torch.count_nonzero(state.rows[layer_id][count:])), 0)
                    mask = torch.zeros(capacity, dtype=torch.bool)
                    mask[:count] = (windows[:, 0] >= 0) & (windows[:, 0] != 31)
                    torch.testing.assert_close(state.token_mask.cpu(), mask, rtol=0, atol=0)
                    self.assertEqual(state.token_mask.data_ptr(), mask_address)

    @torch.inference_mode()
    def test_reject_invalid_history_before_mutating_buffers(self):
        state = self._state(2)
        snapshot = {key: value.clone() for key, value in state.rows.items()}
        for bad in (torch.zeros(3, 4, dtype=torch.int32),
                    torch.zeros(2, 3, dtype=torch.int32),
                    torch.zeros(2, 4), torch.full((2, 4), 32, dtype=torch.int64)):
            with self.assertRaises((ValueError, TypeError)):
                state.prepare(bad)
            for key in snapshot:
                torch.testing.assert_close(state.rows[key], snapshot[key], rtol=0, atol=0)
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA Graph capture"):
                state.prepare(self._windows(2))

    @torch.inference_mode()
    def test_real_graph_replay_matches_original_cpu_lookup_chain_bitwise(self):
        for capacity in (1, 2, 4):
            state = self._state(capacity)
            # Each layer has an independent projection; run their normal chained
            # residuals, not a surrogate lookup-only graph.
            hidden = torch.randn(capacity, 4, 5120, device=self.device).bfloat16()
            original = hidden.clone()
            hashes = self.hash_state.hash_token_windows(self._windows(capacity))

            def prefetched_forward():
                value = hidden
                for layer_id, layer in enumerate(self.layers):
                    value = layer.engram(
                        value, hashes[:, layer_id], state.token_mask,
                        embedding_rows=state.rows[layer_id],
                    )
                return value

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    hidden.copy_(original)
                    prefetched_forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = prefetched_forward()
            for count, shift in ((capacity, 3), (1, 19), (capacity, 11)):
                windows = self._windows(count, shift=shift)
                if shift == 11:
                    windows[0, 0] = 31
                    if count > 1:
                        windows[1, 0] = 7
                state.prepare(windows)
                original.normal_()
                hidden.copy_(original)
                graph.replay()
                torch.cuda.synchronize(self.device)
                graph_result = captured.clone()
                padded = torch.full((capacity, 4), -1, dtype=torch.int32)
                padded[:count] = windows
                # The ordinary model hashes on CUDA before its CPU lookup;
                # compare that path with the prefetch state's CPU hash too.
                padded_cuda = padded.to(self.device)
                fresh_hashes = self.hash_state.hash_token_windows(
                    padded_cuda, dead_mask=padded_cuda == 31
                )
                expected = original.clone()
                for layer_id, layer in enumerate(self.layers):
                    expected = layer.engram(expected, fresh_hashes[:, layer_id], state.token_mask)
                torch.testing.assert_close(graph_result, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    graph_result[~state.token_mask], original[~state.token_mask], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()

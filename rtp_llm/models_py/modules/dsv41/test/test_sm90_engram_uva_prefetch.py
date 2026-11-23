"""Bounded SM90 UVA async lookup, including changed-input graph replay.

Reuses only the existing small-table fixture (132 KiB registered host table).
No checkpoint/shared production table is created, duplicated, or registered.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41._engram_decode_prefetch import DecodeEngramPrefetch
from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding, NgramHashState
from rtp_llm.models_py.modules.dsv41.engram_prefetch import UvaEngramPrefetch
from rtp_llm.models_py.modules.dsv41.test import test_sm90_engram as fixture_module


def _masked_cpu_lookup_reference(table, scales, indices):
    """Decode FP8/E8M0 rows while independently applying the UVA ID mask."""
    indices = indices.to(device="cpu", dtype=torch.int64)
    valid = (indices >= 0) & (indices < table.shape[0])
    safe = indices.clamp(0, table.shape[0] - 1).flatten()
    rows = table.view(torch.uint8).index_select(0, safe).view(table.dtype).float()
    exponents = scales.view(torch.uint8).index_select(0, safe).float()
    rows = rows * torch.exp2(exponents - 127).repeat_interleave(32, dim=1)
    rows = rows.bfloat16().view(*indices.shape, table.shape[1])
    return rows.masked_fill(~valid[..., None], 0)


class Sm90EngramUvaPrefetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.Sm90EngramTest.setUpClass()

    def setUp(self):
        fixture = fixture_module.Sm90EngramTest()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture
        self.device = fixture.device
        self.model = fixture.model
        self.embedding = self.model.embed_tokens
        self.hasher = NgramHashState(
            fixture.layout, token_map=[*range(12), 31], device=self.device
        )
        self.prefetch = UvaEngramPrefetch(self.device, (0, 1))
        # GPU scale captures retain their tables through decode metadata, just
        # as the model forward does before starting the auxiliary lookup stream.
        self.graph_model = torch.nn.Module()
        self.graph_model.layers = [
            SimpleNamespace(engram=self.model, engram_hashes=None) for _ in (0, 1)
        ]
        self.graph_metadata = {}
        self.addCleanup(self.graph_metadata.clear)
        self.addCleanup(self.prefetch.finish)

    def _lookups(self, hashes):
        return ((0, self.embedding, hashes), (1, self.embedding, hashes))

    def _run_prefetched(self, hidden, windows):
        # Hashing and masks are inside the captured body, so replay must use
        # the latest complete token history rather than a cached first request.
        hashes = self.hasher.hash_token_windows(windows, dead_mask=windows == 12)
        ids = hashes[:, 0].contiguous()
        mask = (windows[:, 0] >= 0) & (windows[:, 0] != 12)
        metadata = self.graph_metadata.setdefault(
            windows.shape[0],
            SimpleNamespace(batch_size=windows.shape[0], q_len_per_req=1),
        )
        owner = DecodeEngramPrefetch(self.graph_model, metadata)
        try:
            if (
                self.embedding._device_scale_lease is not None
                and torch.cuda.is_current_stream_capturing()
            ):
                self.assertIsNotNone(owner._capture_permit)
            self.prefetch.begin(self._lookups(ids))
            output = hidden
            for layer_id in (0, 1):
                output = self.model(
                    output, ids, mask, embedding_rows=self.prefetch.wait(layer_id)
                )
            return output
        finally:
            try:
                self.prefetch.finish()
            finally:
                owner.close()

    def _run_eager(self, hidden, windows):
        hashes = self.hasher.hash_token_windows(windows, dead_mask=windows == 12)
        ids = hashes[:, 0].contiguous()
        mask = (windows[:, 0] >= 0) & (windows[:, 0] != 12)
        output = hidden
        for _ in (0, 1):
            output = self.model(output, ids, mask)
        return output

    @torch.inference_mode()
    def test_verify_prepare_and_real_block_consumer_follow_changed_graph_inputs(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
        from rtp_llm.models_py.modules.dsv41.block import Block
        from rtp_llm.models_py.modules.dsv41.engram import EngramLayout

        with patch.dict("os.environ", {"DSV41_ENGRAM_DEVICE_SCALES": "0"}):
            embedding = HostEngramEmbedding(
                self.fixture.pinned.weight,
                self.fixture.pinned.scales,
                pinned=self.fixture.pinned,
            )
        self.model.embed_tokens = embedding
        self.assertIsNone(embedding._device_scale_lease)
        layout = EngramLayout(
            dict(
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[4000, 4000],
                engram_max_ngram_size=4,
                engram_n_heads=4,
                engram_head_dim=32,
                engram_compressed_vocab_size=32,
                engram_pad_token_id=2,
                engram_vocab_size=13,
            )
        )
        state = NgramHashState(layout, token_map=[*range(12), 31], device=self.device)
        prefetch = UvaEngramPrefetch(self.device, (1, 14))
        self.addCleanup(prefetch.finish)
        indices = ((1, 0), (14, 1))
        layers = [SimpleNamespace(engram=None) for _ in range(15)]
        for target, _ in indices:
            layers[target] = SimpleNamespace(
                layer_id=target, engram=self.model, engram_prefetch=None
            )
        owner = SimpleNamespace(
            _engram_attachment=SimpleNamespace(layer_indices=indices, hash_state=state),
            _v4_args=SimpleNamespace(v41_config={"image_token_id": 12}),
            v4=SimpleNamespace(
                embed=SimpleNamespace(weight=self.model.q_weight), layers=layers
            ),
            _engram_prefetch=prefetch,
            _engram_prefetch_max_rows=256,
            _engram_layers=indices,
            _engram_verify_prefetch_enabled=True,
        )
        attn = SimpleNamespace(is_prefill=True, is_target_verify=True)

        def run(hidden, windows, enabled):
            owner._engram_verify_prefetch_enabled = enabled
            try:
                DeepSeekV41Model._prepare_engram(
                    owner,
                    SimpleNamespace(
                        engram_token_windows=windows, attention_inputs=attn
                    ),
                )
                output = hidden
                for target, _ in indices:
                    self.assertIs(
                        layers[target].engram_prefetch, prefetch if enabled else None
                    )
                    output = Block._inject_engram(layers[target], output)
                return output
            finally:
                prefetch.finish()
                for target, _ in indices:
                    layers[target].engram_prefetch = None

        for tokens in (6, 48, 128):
            windows = torch.zeros(tokens, 4, dtype=torch.int32, device=self.device)
            initial = torch.randn(tokens, 4, 5120, device=self.device).bfloat16()
            hidden = initial.clone()
            for _ in range(3):
                hidden.copy_(initial)
                run(hidden, windows, True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run(hidden, windows, True)
            for turn in range(3):
                values = (torch.arange(tokens * 4).reshape(tokens, 4) + turn) % 12
                values[tokens // 2 :] = -1
                values[0, 2] = -1
                if turn == 2:
                    values[0, 0] = 12
                windows.copy_(values)
                initial.normal_()
                hidden.copy_(initial)
                graph.replay()
                torch.cuda.synchronize(self.device)
                expected = run(initial.clone(), windows, False)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            graph = output = None

    @torch.inference_mode()
    def test_eager_and_graph_changed_history_mask_and_bucket_are_bitwise_equal(self):
        for capacity in (1, 2, 4):
            windows = torch.zeros(capacity, 4, dtype=torch.int32, device=self.device)
            hidden = torch.randn(capacity, 4, 5120, device=self.device).bfloat16()
            original = hidden.clone()
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    hidden.copy_(original)
                    self._run_prefetched(hidden, windows)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = self._run_prefetched(hidden, windows)
            for case, live in enumerate((capacity, 1, 0, capacity)):
                fresh = (
                    torch.arange(capacity * 4).reshape(capacity, 4) + case * 3
                ) % 12
                fresh[live:] = -1
                if live:
                    fresh[0, 2] = -1  # Truncate an ngram at a request boundary.
                if live and case == 3:
                    fresh[0, 0] = 12  # Change which rows are masked.
                windows.copy_(fresh)
                original.normal_()
                hidden.copy_(original)
                graph.replay()
                torch.cuda.synchronize(self.device)
                result = captured.clone()
                expected = self._run_eager(original.clone(), windows)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                actual_eager = self._run_prefetched(original.clone(), windows)
                torch.testing.assert_close(actual_eager, expected, rtol=0, atol=0)
                invalid = (windows[:, 0] < 0) | (windows[:, 0] == 12)
                torch.testing.assert_close(
                    result[invalid], original[invalid], rtol=0, atol=0
                )
                self.assertEqual(self.prefetch.rows, {})
                self.assertIsNone(self.prefetch._origin)

    @torch.inference_mode()
    def test_prefetch_matches_cpu_rows_and_explicit_out(self):
        cpu = HostEngramEmbedding(self.fixture.table, self.fixture.table_scales)
        table_rows = self.fixture.table.shape[0]
        self.assertEqual(table_rows, 4000)
        for count in (0, 1, 7):
            ids = torch.randint(0, table_rows, (count, 12), device=self.device)
            if count:
                ids[0, :3] = torch.tensor(
                    [-1, table_rows, table_rows + 7], device=self.device
                )
            host_ids = ids.cpu()
            expected = _masked_cpu_lookup_reference(
                self.fixture.table, self.fixture.table_scales, host_ids
            )
            # The production CPU API accepts table-domain IDs. Check those
            # values independently, preserving its upper-bound error behavior.
            valid = (host_ids >= 0) & (host_ids < table_rows)
            safe_expected = cpu(host_ids.clamp(0, table_rows - 1), "cpu")
            safe_expected.masked_fill_(~valid[..., None], 0)
            torch.testing.assert_close(expected, safe_expected, rtol=0, atol=0)
            self.prefetch.begin(self._lookups(ids))
            try:
                for layer_id in (0, 1):
                    rows = self.prefetch.wait(layer_id)
                    torch.testing.assert_close(rows.cpu(), expected, rtol=0, atol=0)
            finally:
                self.prefetch.finish()
        with self.assertRaisesRegex(RuntimeError, "registered layer tables"):
            self.prefetch.begin(((0, cpu, ids),))
        self.assertIsNone(self.prefetch._origin)

    @torch.inference_mode()
    def test_failure_after_first_lookup_joins_stream_and_allows_next_request(self):
        ids = torch.zeros(1, 12, dtype=torch.int64, device=self.device)

        class FailingLookup:
            _uva = self.embedding._uva
            weight = self.embedding.weight

            def __call__(self, *args, **kwargs):
                raise RuntimeError("injected lookup failure")

        with self.assertRaisesRegex(RuntimeError, "injected lookup failure"):
            self.prefetch.begin(((0, self.embedding, ids), (1, FailingLookup(), ids)))
        self.assertEqual(self.prefetch.rows, {})
        self.assertIsNone(self.prefetch._origin)
        self.prefetch.begin(self._lookups(ids))
        # Simulate a consumer failing before waiting on either layer.
        self.prefetch.finish()
        torch.cuda.synchronize(self.device)
        self.assertEqual(self.prefetch.rows, {})


if __name__ == "__main__":
    unittest.main()

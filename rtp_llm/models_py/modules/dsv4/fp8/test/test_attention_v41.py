"""CPU contracts for V4.1 compression, query scaling, and shared selections."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention_v41_module
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.dsv4.fp8.attention import _configure_flash_mla_l2_persist
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    AttentionV41FP8,
    compress_pairs,
    mask_candidate_logits,
    rms_norm,
    rope_only,
    select_candidate_blocks,
)


class AttentionV41Test(unittest.TestCase):
    @staticmethod
    def _chunk_meta_fixture(ratio, window_size, *, pool_bound=True):
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = 2
        attn.compress_ratio = ratio
        attn.window_size = window_size
        attn._shared_attention = {"layers": {2: attn}}
        attn._source_pool = lambda region: object() if pool_bound else None
        return attn

    @staticmethod
    def _chunk_meta_reference(globals_by_req, swa, swa_starts, req_ids):
        per_request = []
        offset = 0
        for (g, _), sw, start in zip(globals_by_req, swa, swa_starts):
            per_request.append((offset, g.shape[0], start))
            offset += g.shape[0] + sw.shape[0]
        return tuple(
            torch.tensor(
                [[per_request[request][column]] for request in req_ids.tolist()],
                dtype=torch.long,
            )
            for column in range(3)
        )

    def test_prefill_chunk_meta_ragged_requests_use_global_cp_lengths(self):
        for cp_on in (False, True):
            for ratio in (1, 2):
                for window_size in (1, 8):
                    with self.subTest(
                        cp_on=cp_on, ratio=ratio, window_size=window_size
                    ):
                        lengths = [1, 5, 3, 8, 2, 7]
                        prefixes = [
                            0,
                            1,
                            window_size - 1,
                            window_size,
                            window_size + 1,
                            2 * window_size + 3,
                        ]
                        tails = [min(p, window_size - 1) for p in prefixes]
                        globals_by_req = [
                            (torch.empty((p + n) // ratio, 4), None)
                            for p, n in zip(prefixes, lengths)
                        ]
                        swa = [
                            torch.empty(n + tail, 4) for n, tail in zip(lengths, tails)
                        ]
                        starts = [p - tail for p, tail in zip(prefixes, tails)]
                        req_ids = torch.repeat_interleave(
                            torch.arange(len(lengths)), torch.tensor(lengths)
                        ).flip(0)
                        expected = self._chunk_meta_reference(
                            globals_by_req, swa, starts, req_ids
                        )
                        common = SimpleNamespace(
                            cp_on=cp_on,
                            # Deliberately different from global lengths:
                            # CP's local shard lengths must not size the cache.
                            input_lengths=torch.tensor(
                                [1] * len(lengths) if cp_on else lengths,
                                dtype=torch.int32,
                            ),
                            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                            cp_ctx=SimpleNamespace(
                                input_lengths_global=torch.tensor(
                                    lengths, dtype=torch.int32
                                )
                            ),
                        )
                        attn = self._chunk_meta_fixture(ratio, window_size)
                        with patch.object(
                            torch,
                            "tensor",
                            side_effect=AssertionError("unexpected host tensor upload"),
                        ):
                            actual = attn._prefill_chunk_meta(
                                globals_by_req,
                                swa,
                                starts,
                                req_ids,
                                torch.device("cpu"),
                                common=common,
                            )
                        for result, reference in zip(actual, expected):
                            self.assertEqual(result.dtype, torch.long)
                            torch.testing.assert_close(result, reference)

    def test_prefill_chunk_meta_pool_free_keeps_new_boundary_shapes(self):
        lengths = [1, 3, 2]
        prefixes = [7, 8, 13]
        req_ids = torch.tensor([2, 0, 1, 1, 2, 1])
        swa = [torch.empty(n, 4) for n in lengths]
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                globals_by_req = [
                    (torch.empty((p + n) // ratio - p // ratio, 4), None)
                    for p, n in zip(prefixes, lengths)
                ]
                expected = self._chunk_meta_reference(
                    globals_by_req, swa, prefixes, req_ids
                )
                attn = self._chunk_meta_fixture(ratio, 8, pool_bound=False)
                with patch.object(
                    torch,
                    "tensor",
                    side_effect=AssertionError("unexpected host tensor upload"),
                ):
                    actual = attn._prefill_chunk_meta(
                        globals_by_req,
                        swa,
                        prefixes,
                        req_ids,
                        torch.device("cpu"),
                        # Pool-free warmup must use actual shapes, and need
                        # not have the production pool's length metadata.
                        common=SimpleNamespace(),
                    )
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference)

    def test_prefill_chunk_meta_single_request_keeps_scalar_fast_path(self):
        attn = self._chunk_meta_fixture(2, 8)
        req_ids = torch.zeros(3, dtype=torch.long)
        with patch.object(
            torch, "tensor", side_effect=AssertionError("unexpected host tensor upload")
        ):
            actual = attn._prefill_chunk_meta(
                [(torch.empty(9, 4), None)],
                [torch.empty(10, 4)],
                [8],
                req_ids,
                torch.device("cpu"),
                common=SimpleNamespace(),
            )
        self.assertEqual(actual, (0, 9, 8))
        self.assertTrue(all(isinstance(value, int) for value in actual))

    def test_prefill_chunk_meta_cache_survives_consumers_until_next_forward(self):
        attn = self._chunk_meta_fixture(2, 8)
        globals_by_req = [(torch.empty(2, 4), None), (torch.empty(3, 4), None)]
        swa = [torch.empty(4, 4), torch.empty(6, 4)]
        req_ids = torch.tensor([0, 1, 1])
        common = SimpleNamespace(
            cp_on=False,
            input_lengths=torch.tensor([4, 6]),
            prefix_lengths=torch.zeros(2, dtype=torch.long),
        )
        actual = attn._prefill_chunk_meta(
            globals_by_req, swa, [0, 0], req_ids, "cpu", common=common
        )
        consumer = self._chunk_meta_fixture(2, 8)
        consumer.layer_id = 3
        consumer._shared_attention = attn._shared_attention
        attn._shared_attention["layers"][3] = consumer
        consumer._begin_forward()
        # A consumer must return the same tuple without re-reading metadata.
        self.assertIs(
            consumer._prefill_chunk_meta(None, None, None, None, None, common=None),
            actual,
        )
        attn._begin_forward()
        self.assertNotIn("prefill_chunk_meta", attn._shared_attention)
        next_meta = attn._prefill_chunk_meta(
            [(torch.empty(1, 4), None)],
            [torch.empty(2, 4)],
            [0],
            torch.zeros(2, dtype=torch.long),
            "cpu",
            common=SimpleNamespace(),
        )
        self.assertEqual(next_meta, (0, 1, 0))
        self.assertIsNot(next_meta, actual)

    def test_small_cp_x_gather_ceiling_covers_mid_size_batches(self):
        # The single-shot raw-x all-gather must cover the shared-prefix batch
        # band (8x8192 = 65536 global rows and every smaller multi-request
        # forward) while leaving large-context single requests on the tiled
        # owner-projected path.
        for padded, expected in [
            (8192, True),
            (16384, True),
            (32768, True),
            (40960, True),
            (49152, True),
            (57344, True),
            (65536, True),
            (65537, False),
            (131072, False),
        ]:
            with self.subTest(padded=padded):
                ctx = SimpleNamespace(padded_seq_len=padded)
                self.assertEqual(
                    attention_v41_module._use_small_cp_x_gather(ctx), expected
                )
        with patch.object(attention_v41_module, "_SMALL_CP_X_GATHER_MAX_ROWS", 32768):
            ctx = SimpleNamespace(padded_seq_len=49152)
            self.assertFalse(attention_v41_module._use_small_cp_x_gather(ctx))
            self.assertTrue(
                attention_v41_module._use_small_cp_x_gather(
                    SimpleNamespace(padded_seq_len=32768)
                )
            )

    def test_cp_hidden_tiles_restore_padded_multiple_requests(self):
        for cp_size in (2, 4):
            lengths = (5, 19, 1, 32)
            chunks = tuple(
                2 * ((n + 2 * cp_size - 1) // (2 * cp_size)) for n in lengths
            )
            local = [[] for _ in range(cp_size)]
            expected = []
            for request, (length, chunk) in enumerate(zip(lengths, chunks)):
                values = torch.arange(length) + request * 100
                padded = torch.full((cp_size * chunk,), -1)
                padded[:length] = values
                halves = padded.reshape(2 * cp_size, chunk // 2)
                for rank in range(cp_size):
                    local[rank].extend((halves[rank], halves[-rank - 1]))
                expected.append(values)
            local = [torch.cat(parts) for parts in local]
            context = SimpleNamespace(
                cp_size=cp_size,
                chunk_lengths_per_req=chunks,
                input_lengths_global_host=lengths,
            )
            for tile_rows in (1, 3, 7, 32768):
                with patch.object(
                    attention_v41_module, "_PRODUCE_GLOBAL_TILE_ROWS", tile_rows
                ):
                    plan = list(attention_v41_module._prefill_x_tile_plan(context))
                actual = torch.cat(
                    [local[rank][start : start + count] for rank, start, count in plan]
                )
                torch.testing.assert_close(actual, torch.cat(expected))
                self.assertTrue(all(0 < count <= tile_rows for _, _, count in plan))

    def test_cp_gather_updates_input_with_symm_memory_wrapper(self):
        from rtp_llm.models_py.distributed import collective_torch as collectives

        value = torch.tensor([[1.0, 2.0]])
        source = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, kv_cache_sharded=True)
        )

        class SymmetricCommunicator:
            def should_torch_symm_mem_allreduce(self, tensor):
                return True

            def all_reduce(self, tensor, out=None):
                reduced = tensor * 4
                if out is not None:
                    out.copy_(reduced)
                    return out
                return reduced

        symm = SimpleNamespace(
            get_symm_mem_communicator=lambda: SymmetricCommunicator()
        )
        with patch.object(
            collectives, "_get_rocm_rccl", return_value=None
        ), patch.object(collectives, "_get_symm_mem", return_value=symm):
            actual = AttentionV41FP8._gather_shards(source, value)
            projected = torch.tensor([[3.0, 5.0]])
            AttentionV41FP8._prefill_output_all_reduce(
                SimpleNamespace(tp_size=4), projected
            )
        self.assertIs(actual, value)
        torch.testing.assert_close(value, torch.tensor([[4.0, 8.0]]))
        torch.testing.assert_close(projected, torch.tensor([[12.0, 20.0]]))

    def test_v41_metadata_allocates_swa_and_keeps_owner_block_tables(self):
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.decode.forward import (
            decode_metadata_compress_ratios,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            allocate_decode_metadata_fp8,
        )

        args = SimpleNamespace(
            v41_config={"kv_source_layer_ids": [2, 8, 14, 20]},
            compress_ratios=[0, 0, 2, 1],
            n_layers=4,
        )
        specs = {
            SWA_KV: (256, 128, 3),
            CSA_KV: (64, 128, 3),
            HCA_KV: (128, 128, 3),
            INDEXER_KV: (128, 128, 3),
            CSA_STATE: (8, 128, 3),
        }
        meta = allocate_decode_metadata_fp8(
            max_batch_size=2,
            q_len=6,
            window_size=128,
            head_dim=512,
            max_seq_len=256,
            compress_ratios=decode_metadata_compress_ratios(args),
            index_topk=512,
            device=torch.device("cpu"),
            paged_pool_specs=specs,
        )
        self.assertEqual(meta.slot_mapping_compressed, {})
        self.assertEqual(set(meta.pool_block_tables), set(specs))
        torch.testing.assert_close(
            meta.req_id_per_token, torch.tensor([0] * 6 + [1] * 6, dtype=torch.int32)
        )
        args.v41_config = None
        args.compress_ratios = [0, 4, 128, 0]
        self.assertEqual(decode_metadata_compress_ratios(args), [0, 4, 128, 0])

    def test_pair_pooling_is_channelwise_and_does_not_round_before_norm(self):
        torch.manual_seed(71)
        values = torch.randn(7, 2, 512) * 7
        scores = torch.randn_like(values) * 2
        norm = torch.randn(512).bfloat16()
        actual = compress_pairs(values, scores, norm, 1e-6)
        expected = torch.empty(7, 512)
        for group in range(7):
            for channel in range(512):
                p = torch.softmax(scores[group, :, channel], dim=0)
                expected[group, channel] = (values[group, :, channel] * p).sum()
        expected = rms_norm(expected, norm, 1e-6).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rounded_early = rms_norm(
            (values * scores.softmax(1)).sum(1).bfloat16(), norm, 1e-6
        )
        self.assertTrue((actual != rounded_early).any())

    def test_rope_does_not_apply_post_q_normalization(self):
        q = torch.tensor([[[3.0, 4.0, 5.0, 6.0]]], dtype=torch.bfloat16)
        rotated = rope_only(q.clone(), torch.tensor([[1j]]), 2)
        torch.testing.assert_close(
            rotated, torch.tensor([[[3.0, 4.0, -6.0, 5.0]]], dtype=q.dtype)
        )
        torch.testing.assert_close(
            rotated.float().square().sum(), q.float().square().sum()
        )

    def test_topk_consumers_reuse_source_selection_without_reindexing(self):
        selected = torch.tensor([[2, 0, -1]], dtype=torch.int32)
        consumer = SimpleNamespace(
            is_index_source=False,
            index_source_layer_id=20,
            _shared_attention={"topk": {20: selected}},
        )
        actual = AttentionV41FP8._select_indices(consumer, None, None, None, None)
        self.assertIs(actual, selected)

    def test_short_context_selects_completed_groups_only(self):
        source = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=2,
            kv_source_layer_id=2,
            layer_id=2,
            compress_ratio=2,
            index_topk=4,
            _shared_attention={"global": {2: [(None, torch.zeros(3, 128))]}},
        )
        positions = torch.tensor([0, 1, 2, 3, 4, 5])
        actual = AttentionV41FP8._select_indices(
            source, torch.zeros(6, 1), None, positions, torch.zeros(6, dtype=torch.long)
        )
        expected = torch.tensor(
            [
                [-1, -1, -1, -1],
                [0, -1, -1, -1],
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, -1, -1],
                [0, 1, 2, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_continuation_pair_uses_previous_fp32_state(self):
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = 2
        attn.kv_source_layer_id = 2
        attn.compress_ratio = 2
        attn.head_dim = 4
        attn.rope_head_dim = 2
        attn.eps = 1e-6
        attn.global_wkv = torch.eye(4, dtype=torch.bfloat16)
        attn.global_wgate = torch.eye(4, dtype=torch.bfloat16) * 0.125
        attn.global_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.index_wk = torch.eye(4, dtype=torch.bfloat16)
        attn.index_k_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.freqs_cis = torch.ones(8, 1, dtype=torch.complex64)
        attn._shared_attention = {"layers": {2: attn}}
        attn._source_pool = lambda region: None
        attn._write_states = lambda *args: None
        first = torch.tensor([[1.25, -2.5, 4.75, 3.0]], dtype=torch.bfloat16)
        second = torch.tensor([[4.0, 0.5, -1.5, 2.0]], dtype=torch.bfloat16)
        previous = torch.cat((first.float(), first.float() * 0.125), -1)
        attn._read_state = lambda *args: previous
        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._linear_bf16_bf16_fp32",
            side_effect=lambda x, w: torch.nn.functional.linear(x.float(), w.float()),
        ):
            attn._produce_global(
                second,
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([1]),
                prefill=True,
            )
        actual, _ = attn._shared_attention["global"][2][0]
        values = torch.stack((first.float(), second.float()), 1)
        expected = compress_pairs(values, values * 0.125, attn.global_norm, attn.eps)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_candidates_use_block_max_and_pin_newest(self):
        logits = torch.tensor([[8.0, 9.0, 3.0, 4.0, 1.0, -torch.inf]])
        candidates = select_candidate_blocks(logits, torch.tensor([5]), 2, 2)
        self.assertEqual(set(candidates[0].tolist()), {0, 2})
        later_logits = torch.tensor([[1.0, 2.0, 100.0, 200.0, 0.0, -torch.inf]])
        mask_candidate_logits(later_logits, candidates, 2)
        self.assertEqual(later_logits.argmax(-1).item(), 1)
        self.assertTrue(torch.isneginf(later_logits[0, 2:4]).all())

    def test_cp4_assigns_each_ratio2_entry_exactly_once(self):
        positions = torch.arange(2048, dtype=torch.int64)
        request = torch.zeros_like(positions)
        # Physical owner blocks span eight kernel pages.
        bt = torch.arange(1, 17).view(1, -1)
        results = [
            cp_kv_slot_mapping(
                positions, bt, request, 128, 64, 2, 4, rank, owner_tokens_per_block=1024
            )
            for rank in range(4)
        ]
        owned = torch.stack([slots >= 0 for slots in results]).sum(0)
        self.assertTrue((owned[positions % 2 == 0] == 0).all())
        self.assertTrue((owned[positions % 2 == 1] == 1).all())
        self.assertTrue((results[0][1024:] == -1).all())
        self.assertTrue((results[1][:1024] == -1).all())
        self.assertTrue((results[2] == -1).all())
        self.assertTrue((results[3] == -1).all())

    def _kv_source_fixture(self, state_writes):
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = 2
        attn.kv_source_layer_id = 2
        attn.compress_ratio = 2
        attn.head_dim = 4
        attn.rope_head_dim = 2
        attn.eps = 1e-6
        attn.global_wkv = torch.eye(4, dtype=torch.bfloat16)
        attn.global_wgate = torch.eye(4, dtype=torch.bfloat16) * 0.125
        attn.global_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.index_wk = torch.eye(4, dtype=torch.bfloat16)
        attn.index_k_norm = torch.ones(4, dtype=torch.bfloat16)
        attn.freqs_cis = torch.ones(16, 1, dtype=torch.complex64)
        attn._shared_attention = {"layers": {2: attn}}
        attn._source_pool = lambda region: None
        # [B, 2 * head_dim] previous-pair state per request.
        previous = torch.arange(2 * 8, dtype=torch.float32).view(2, 8)
        attn._read_state = lambda positions, req_ids: previous

        def record_writes(values, scores, positions, req_ids, seq_ends):
            state_writes.append((positions.clone(), values.clone(), scores.clone()))

        attn._write_states = record_writes
        return attn

    def test_produce_global_tiling_is_bit_identical_across_tile_sizes(self):
        # Chunked gather-project: tiles cut at arbitrary rows (including inside
        # ratio-2 pairs and across requests); the seam carries the pair state,
        # so every tile size produces the same pooled globals and the same
        # concatenated per-token state writes as the single-shot projection.
        torch.manual_seed(11)
        rows = 9
        x_full = torch.randn(rows, 4).bfloat16()
        positions = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)
        req_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
        starts = torch.tensor([3, 7], dtype=torch.long)
        lengths = torch.tensor([4, 5], dtype=torch.long)

        def run(tile_rows, streamed=False):
            state_writes = []
            attn = self._kv_source_fixture(state_writes)
            projected = torch.cat((x_full.float(), x_full.float() * 0.125), -1)
            with patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor._linear_bf16_bf16_fp32",
                side_effect=lambda x, w: torch.nn.functional.linear(
                    x.float(), w.float()
                ),
            ), patch.object(
                attention_v41_module, "_PRODUCE_GLOBAL_TILE_ROWS", tile_rows
            ):
                attn._produce_global(
                    x_full[:1] if streamed else x_full,
                    positions,
                    req_ids,
                    starts,
                    lengths,
                    prefill=True,
                    projected_tiles=(
                        (
                            (start, projected[start : start + tile_rows])
                            for start in range(0, rows, tile_rows)
                        )
                        if streamed
                        else None
                    ),
                )
            return attn._shared_attention["global"][2], state_writes

        reference_globals, reference_writes = run(4096)
        self.assertEqual(len(reference_writes), 1)
        for tile_rows, streamed in [
            (n, streamed) for n in (1, 2, 3, 4, 5, 8) for streamed in (False, True)
        ]:
            with self.subTest(tile_rows=tile_rows, streamed=streamed):
                tiled_globals, tiled_writes = run(tile_rows, streamed)
                self.assertEqual(len(tiled_globals), len(reference_globals))
                for (g_ref, k_ref), (g_tile, k_tile) in zip(
                    reference_globals, tiled_globals
                ):
                    torch.testing.assert_close(g_tile, g_ref, rtol=0, atol=0)
                    torch.testing.assert_close(k_tile, k_ref, rtol=0, atol=0)
                pos_ref = torch.cat([w[0] for w in reference_writes])
                values_ref = torch.cat([w[1] for w in reference_writes])
                scores_ref = torch.cat([w[2] for w in reference_writes])
                pos_tile = torch.cat([w[0] for w in tiled_writes])
                values_tile = torch.cat([w[1] for w in tiled_writes])
                scores_tile = torch.cat([w[2] for w in tiled_writes])
                torch.testing.assert_close(pos_tile, pos_ref, rtol=0, atol=0)
                torch.testing.assert_close(values_tile, values_ref, rtol=0, atol=0)
                torch.testing.assert_close(scores_tile, scores_ref, rtol=0, atol=0)

    @torch.inference_mode()
    def test_rms_norm_native_matches_reference(self):
        # Native RMSNorm contract (migrated from the dev-lane math test): on a
        # Blackwell CUDA device the bf16 sites run the framework
        # ``rtp_llm_ops.rmsnorm`` kernel; the result stays within one BF16
        # ulp of this reference formula with a >=99.9% bit-exact rate.
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability(0)[0] != 10
        ):
            self.skipTest("native rmsnorm requires a Blackwell CUDA device")
        device = torch.device("cuda")
        shapes = [(2, 5120), (7, 1280), (2048, 512), (3, 128), (1, 5120)]
        for index, shape in enumerate(shapes):
            torch.manual_seed(83 + index)
            hidden = torch.randn(*shape, device=device).bfloat16()
            weight = (torch.randn(shape[-1], device=device) * 0.5 + 1).bfloat16()
            native = rms_norm(hidden, weight, 1e-20)
            reference = (
                hidden.float()
                * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-20)
                * weight.float()
            ).to(hidden.dtype)
            ai = native.view(torch.int16).to(torch.int32)
            bi = reference.view(torch.int16).to(torch.int32)
            ulp = (
                torch.where(ai >= 0, ai, -32768 - ai)
                - torch.where(bi >= 0, bi, -32768 - bi)
            ).abs()
            exact = (native == reference).float().mean().item()
            self.assertGreaterEqual(exact, 0.999, (shape, exact))
            self.assertLessEqual(int(ulp.max().item()), 1, (shape, int(ulp.max())))

    @torch.inference_mode()
    def test_rms_norm_fp32_boundary_keeps_reference_path(self):
        # FP32-boundary sites (compressor projections) and CPU tensors keep
        # the reference formula bit-exactly; only bf16 CUDA input pairs take
        # the native kernel.
        torch.manual_seed(92)
        hidden = torch.randn(4, 512).bfloat16()
        weight = torch.randn(512).bfloat16()
        mixed = rms_norm(hidden.float(), weight, 1e-6)
        expected = (
            hidden.float()
            * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        )
        self.assertEqual(mixed.dtype, torch.float32)
        self.assertTrue(torch.equal(mixed, expected))

    def test_begin_forward_drops_cross_layer_index_plan(self):
        attn = AttentionV41FP8.__new__(AttentionV41FP8)
        torch.nn.Module.__init__(attn)
        attn.layer_id = 2
        attn._shared_attention = {
            "layers": {2: attn},
            "prefill_index_plan": object(),
            "prefill_chunk_meta": object(),
        }
        attn._begin_forward()
        self.assertNotIn("prefill_index_plan", attn._shared_attention)
        self.assertNotIn("prefill_chunk_meta", attn._shared_attention)
        self.assertEqual(attn._shared_attention["global"], {})


class PrefillCandidatesIntegrationTest(unittest.TestCase):
    def test_source_slice_writes_in_place_and_four_consumers_reuse_bitmap(self):
        candidates = torch.full((6, 2), -7, dtype=torch.int32)
        flags = torch.full((6, 3), -9, dtype=torch.int32)
        shared = {"candidates": candidates}
        shared["prefill_candidate_mask"] = (candidates, flags, 2)
        rows = slice(1, 4)
        logits = torch.zeros(3, 17)
        visible = torch.tensor([3, 9, 17])
        expected_ids = torch.tensor([[0, 1], [2, 4], [6, 8]], dtype=torch.int32)
        expected_flags = torch.tensor([[3, 0, 0], [20, 0, 0], [320, 0, 0]])

        def select(logits, visible, block_size, topk, *, out, flags, build_bitmap=True):
            self.assertEqual(
                out.untyped_storage().data_ptr(),
                candidates.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                flags.untyped_storage().data_ptr(),
                shared["prefill_candidate_mask"][1].untyped_storage().data_ptr(),
            )
            out.copy_(expected_ids)
            flags.copy_(expected_flags)
            return out, flags

        def mask(logits, chunk_flags, block_size):
            torch.testing.assert_close(chunk_flags, expected_flags.to(torch.int32))
            self.assertEqual(
                chunk_flags.untyped_storage().data_ptr(),
                flags.untyped_storage().data_ptr(),
            )
            logits[:, 0] = -torch.inf
            return True

        helpers = attention_v41_module.prefill_candidates
        with patch.object(
            helpers, "select_candidates", side_effect=select
        ), patch.object(
            helpers, "build_flags", side_effect=AssertionError("Bitmap must be reused")
        ), patch.object(
            helpers, "mask_candidates", side_effect=mask
        ) as mask_call:
            attention_v41_module._apply_prefill_candidates(
                shared, logits, visible, rows, 2, 2, True
            )
            for _ in range(4):
                consumer_logits = torch.ones_like(logits)
                attention_v41_module._apply_prefill_candidates(
                    shared, consumer_logits, visible, rows, 2, 2, False
                )
                self.assertTrue(torch.isneginf(consumer_logits[:, 0]).all())
            self.assertEqual(mask_call.call_count, 4)
        torch.testing.assert_close(candidates[rows], expected_ids)
        self.assertTrue((candidates[[0, 4, 5]] == -7).all())
        self.assertTrue((flags[[0, 4, 5]] == -9).all())
        source = SimpleNamespace(layer_id=2, _shared_attention=shared)
        shared["layers"] = {2: source}
        AttentionV41FP8._begin_forward(source)
        self.assertNotIn("prefill_candidate_mask", shared)
        self.assertIsNone(shared["candidates"])

    def test_ragged_source_scatter_restores_candidates_and_bitmap_row_order(self):
        candidates = torch.full((5, 2), -1, dtype=torch.int32)
        bitmap = torch.full((5, 3), -9, dtype=torch.int32)
        shared = {
            "candidates": candidates,
            "prefill_candidate_mask": (candidates, bitmap, 2),
        }
        rows = torch.tensor([4, 0, 2])
        expected_ids = torch.tensor([[0, 2], [1, 3], [4, 5]], dtype=torch.int32)
        expected_flags = torch.tensor([[5, 0, 0], [10, 0, 0], [48, 0, 0]])

        def select(logits, visible, block_size, topk, *, out, flags, build_bitmap=True):
            self.assertNotEqual(
                out.untyped_storage().data_ptr(),
                candidates.untyped_storage().data_ptr(),
            )
            out.copy_(expected_ids)
            flags.copy_(expected_flags)
            return out, flags

        with patch.object(
            attention_v41_module.prefill_candidates,
            "select_candidates",
            side_effect=select,
        ):
            attention_v41_module._apply_prefill_candidates(
                shared, torch.zeros(3, 12), torch.tensor([5, 8, 12]), rows, 2, 2, True
            )
        torch.testing.assert_close(candidates[rows], expected_ids)
        torch.testing.assert_close(bitmap[rows], expected_flags.to(torch.int32))
        self.assertTrue((candidates[[1, 3]] == -1).all())
        self.assertTrue((bitmap[[1, 3]] == -9).all())

    def test_source_fallback_discards_partial_bitmap_and_consumer_rebuilds(self):
        candidates = torch.full((4, 3), -1, dtype=torch.int32)
        candidates[:2] = torch.tensor([[0, 2, -1], [1, 2, -1]])
        shared = {
            "candidates": candidates,
            # Simulate an earlier source chunk that has initialized only
            # half the complete cross-layer bitmap.
            "prefill_candidate_mask": (
                candidates,
                torch.full((4, 1), 123, dtype=torch.int32),
                2,
            ),
        }
        logits = torch.tensor([[5.0, 2.0, 1.0, -torch.inf], [3.0, 4.0, 5.0, 6.0]])
        visible = torch.tensor([3, 4])
        rows = torch.tensor([3, 2])
        expected = select_candidate_blocks(logits, visible, 2, 3)
        helpers = attention_v41_module.prefill_candidates
        with patch.object(helpers, "select_candidates", return_value=None):
            attention_v41_module._apply_prefill_candidates(
                shared, logits, visible, rows, 2, 3, True
            )
        self.assertNotIn("prefill_candidate_mask", shared)
        torch.testing.assert_close(candidates[rows, :2], expected)
        self.assertTrue((candidates[rows, 2] == -1).all())
        built = torch.tensor([[5], [6]], dtype=torch.int32)
        consumer = torch.arange(8, dtype=torch.float32).view(2, 4)
        reference = mask_candidate_logits(consumer.clone(), candidates[rows], 2)
        with patch.object(
            helpers, "build_flags", return_value=built
        ) as build, patch.object(
            helpers, "mask_candidates", return_value=False
        ) as fused_mask:
            attention_v41_module._apply_prefill_candidates(
                shared, consumer, visible, rows, 2, 3, False
            )
        torch.testing.assert_close(build.call_args.args[0], candidates[rows])
        self.assertEqual(build.call_args.args[1:], (4, 2))
        self.assertIs(fused_mask.call_args.args[1], built)
        torch.testing.assert_close(consumer, reference)

    def test_single_and_ragged_topk_chunk_rows_map_to_original_output(self):
        for single_request in (True, False):
            for use_fused_topk in (False, True):
                with self.subTest(single=single_request, fused=use_fused_topk):
                    req_ids = torch.tensor(
                        [0, 0, 0, 0, 0] if single_request else [1, 0, 1, 0, 1]
                    )
                    positions = torch.tensor([4, 2, 5, 0, 3])
                    x = torch.ones(5, 2)
                    qr = torch.stack((torch.arange(1, 6), torch.ones(5)), dim=1)
                    keys = [7] if single_request else [5, 7]
                    globals_by_req = [
                        (
                            None,
                            attention_v41_module.prefill_indexer.PrefillIndexerKeys(
                                torch.empty(n, 64, dtype=torch.int8),
                                torch.empty(n, dtype=torch.int32),
                            ),
                        )
                        for n in keys
                    ]
                    attn = SimpleNamespace(
                        is_index_source=True,
                        index_source_layer_id=20,
                        kv_source_layer_id=20,
                        layer_id=20,
                        index_topk=2,
                        index_n_heads=1,
                        index_head_dim=2,
                        rope_head_dim=0,
                        compress_ratio=1,
                        index_wq=None,
                        index_weights=torch.ones(1, 2),
                        freqs_cis=torch.ones(8, 1, dtype=torch.complex64),
                        _lin=lambda weight, value: value,
                        v41_config=dict(
                            candidate_source_layer_id=20,
                            candidate_topk_blocks=1,
                            candidate_block_size=2,
                        ),
                        _shared_attention={"global": {20: globals_by_req}},
                    )
                    selections = []

                    def score(q, q_sf, k, k_sf, weights, visible):
                        sign = torch.where(q[:, 0, 0].long() % 2 == 0, 1.0, -1.0)
                        columns = torch.arange(k.shape[0])
                        logits = (columns[None] + 1) * sign[:, None]
                        return logits.masked_fill(
                            columns[None] >= visible[:, None], -torch.inf
                        )

                    def select(logits, visible, topk):
                        if not use_fused_topk:
                            return None
                        values, ids = logits.topk(topk, dim=-1)
                        return torch.where(values.isfinite(), ids, -1).int()

                    def candidates(
                        shared, logits, visible, rows, block_size, topk, publish
                    ):
                        self.assertTrue(publish)
                        selections.append(rows)

                    with patch.object(
                        attention_v41_module,
                        "rope_only",
                        side_effect=lambda q, *args: q,
                    ), patch.object(
                        attention_v41_module.prefill_indexer,
                        "quantize_indexer_q",
                        side_effect=lambda q: (q, torch.zeros(5, 1)),
                    ), patch.object(
                        attention_v41_module.prefill_indexer,
                        "logits_chunk_rows",
                        return_value=2,
                    ), patch.object(
                        attention_v41_module.prefill_indexer,
                        "score_indexer_chunk",
                        side_effect=score,
                    ), patch.object(
                        attention_v41_module.prefill_topk,
                        "try_select_tokens",
                        side_effect=select,
                    ), patch.object(
                        attention_v41_module,
                        "_apply_prefill_candidates",
                        side_effect=candidates,
                    ):
                        actual = AttentionV41FP8._select_indices(
                            attn, x, qr, positions, req_ids
                        )
                    expected = []
                    for row, position in enumerate(positions.tolist()):
                        ids = list(range(position + 1))
                        if (row + 1) % 2 == 0:
                            ids.reverse()
                        expected.append((ids[:2] + [-1, -1])[:2])
                    torch.testing.assert_close(
                        actual, torch.tensor(expected, dtype=torch.int32)
                    )
                    if single_request:
                        self.assertEqual(
                            selections, [slice(0, 2), slice(2, 4), slice(4, 6)]
                        )
                    else:
                        for actual_rows, expected_rows in zip(
                            selections, ([1, 3], [0, 2], [4])
                        ):
                            torch.testing.assert_close(
                                actual_rows, torch.tensor(expected_rows)
                            )


class FlashMlaL2PersistEnvironmentTest(unittest.TestCase):
    def setUp(self):
        _configure_flash_mla_l2_persist.cache_clear()
        self.addCleanup(_configure_flash_mla_l2_persist.cache_clear)

    def test_default_disables_native_l2_persist(self):
        with patch.dict(os.environ, {"RTP_V41_TEST_UNRELATED": "preserve"}):
            os.environ.pop("FLASH_MLA_NO_L2_PERSIST", None)
            _configure_flash_mla_l2_persist()
            _configure_flash_mla_l2_persist()
            self.assertEqual(os.environ["FLASH_MLA_NO_L2_PERSIST"], "1")
            self.assertEqual(os.environ["RTP_V41_TEST_UNRELATED"], "preserve")

    def test_explicit_disable_is_preserved(self):
        with patch.dict(os.environ, {"FLASH_MLA_NO_L2_PERSIST": "1"}):
            _configure_flash_mla_l2_persist()
            _configure_flash_mla_l2_persist()
            self.assertEqual(os.environ["FLASH_MLA_NO_L2_PERSIST"], "1")

    def test_explicit_enable_removes_presence_flag_only_once(self):
        with patch.dict(
            os.environ,
            {"FLASH_MLA_NO_L2_PERSIST": "0", "RTP_V41_TEST_UNRELATED": "preserve"},
        ):
            _configure_flash_mla_l2_persist()
            self.assertNotIn("FLASH_MLA_NO_L2_PERSIST", os.environ)
            # Later Attention instances must not treat the now-absent flag as
            # an unset default and disable the user's explicit L2 policy.
            _configure_flash_mla_l2_persist()
            self.assertNotIn("FLASH_MLA_NO_L2_PERSIST", os.environ)
            self.assertEqual(os.environ["RTP_V41_TEST_UNRELATED"], "preserve")


if __name__ == "__main__":
    unittest.main()

"""Semantic IDs, timestamped layouts and the QSA coordinate cache contract."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_ROPE_POS_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
    semantic_token_ids,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_position_state import (
    QSAPositionState,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks
from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_processor import (
    Qwen4VisualInput,
    assemble_visual_embedding,
)


def position_state(prefixes, lengths, positions, pool, table, *, graph=False):
    device = pool.device
    prefixes = torch.as_tensor(prefixes, device=device, dtype=torch.int32)
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.as_tensor(lengths, device=device, dtype=torch.int32).cumsum(0)
    inputs = SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_cuda_graph=graph,
        combo_position_ids=positions.to(device=device, dtype=torch.int32).reshape(-1),
        cu_seqlens_device=cu,
        prefix_lengths_device=prefixes,
        prefix_lengths=prefixes,
    )
    cache = SimpleNamespace(
        tag=INDEXER_ROPE_POS_TAG, seq_size_per_block=128, kv_cache_base=pool
    )
    context = SimpleNamespace(
        main_inputs=inputs,
        position_state_cache=cache,
        position_state_inputs=SimpleNamespace(kv_cache_block_id_device=table),
    )
    state = QSAPositionState(context, 4)
    if not graph:
        checks = DeviceChecks()
        state.add_checks(checks)
        checks.verify()
    return state


class Qwen4MultimodalTest(unittest.TestCase):
    def check_mixed_vision_batch(self, device):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_vit as vit,
        )
        from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_mixin import (
            Qwen4ExpImageEmbedding,
        )
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        config = vit.Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(17)
        adapter = Qwen4ExpImageEmbedding.__new__(Qwen4ExpImageEmbedding)
        with patch.object(vit, "default_attn_impl", "sdpa"):
            adapter.visual = vit.Qwen3_5MoeVisionModel(config).eval().to(device)
        adapter.mm_processor = self.processor()
        adapter.text_embedding = torch.randn(256, 32)
        image = Qwen4VisualInput(torch.randn(16, 12), torch.tensor([[1, 4, 4]]))
        video = Qwen4VisualInput(
            torch.randn(16, 12), torch.tensor([[2, 2, 4]]), [0.1, 0.5]
        )
        expected = [adapter.embedding(data) for data in (image, video)]
        with patch.object(
            adapter.visual, "forward", wraps=adapter.visual.forward
        ) as forward:
            actual = adapter.batched_embedding(
                [image, video], [MMUrlType.IMAGE, MMUrlType.VIDEO]
            )
        self.assertEqual(forward.call_count, 1)
        for result, reference in zip(actual, expected):
            for left, right in zip(result, reference):
                torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)

    def test_mixed_image_video_reuses_one_shared_vision_forward_cpu(self):
        self.check_mixed_vision_batch("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mixed_image_video_reuses_one_shared_vision_forward_cuda(self):
        self.check_mixed_vision_batch("cuda")

    def assemble(self, grid, timestamps=None, device="cpu"):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
            Qwen3_5MoeImageEmbedding,
        )

        grid = torch.tensor([grid])
        tower = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))
        positions = Qwen3_5MoeImageEmbedding.get_position_ids(tower, grid)[0]
        rows = int(grid.prod()) // 4
        features = torch.arange(rows * 4, device=device, dtype=torch.float32).reshape(
            rows, 4
        )
        text = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4) / 1024
        result = assemble_visual_embedding(
            Qwen4VisualInput(torch.empty(0), grid, timestamps),
            features,
            positions,
            self.processor(),
            text,
        )
        return result, features, text

    def test_image_reuses_shared_spatial_coordinates_and_complete_tensor(self):
        (embedding, positions, ids), features, text = self.assemble([1, 4, 6])
        self.assertEqual(ids.tolist(), [10, *([20] * 6), 11])
        torch.testing.assert_close(
            embedding, torch.cat([text[10:11], features, text[11:12]]), atol=0, rtol=0
        )
        self.assertEqual(
            positions.tolist(),
            [
                [0, 0, 0],
                *[[1, h + 1, w + 1] for h in range(2) for w in range(3)],
                [4, 4, 4],
            ],
        )

    def test_video_keeps_timestamp_embeddings_frame_order_and_mrope(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                (embedding, positions, ids), features, text = self.assemble(
                    [2, 4, 4], [0.25, 1.75], device
                )
                tokens, expected, expected_positions = [], [], []
                offset = 0
                for frame, stamp in enumerate((0.25, 1.75)):
                    prefix = [ord(c) for c in f"<{stamp:.1f} seconds>"] + [10]
                    tokens.extend([*prefix, 21, 21, 21, 21, 11])
                    expected.extend(
                        [
                            text[prefix].to(device),
                            features[frame * 4 : (frame + 1) * 4],
                            text[11:12].to(device),
                        ]
                    )
                    expected_positions.extend(
                        [[value] * 3 for value in range(offset, offset + len(prefix))]
                    )
                    offset += len(prefix)
                    expected_positions.extend(
                        [
                            [offset, offset + h, offset + w]
                            for h, w in ((0, 0), (0, 1), (1, 0), (1, 1))
                        ]
                    )
                    offset += 2
                    expected_positions.append([offset] * 3)
                    offset += 1
                self.assertEqual(ids.tolist(), tokens)
                self.assertEqual(positions.tolist(), expected_positions)
                torch.testing.assert_close(
                    embedding, torch.cat(expected), atol=0, rtol=0
                )
                self.assertEqual(positions.device.type, "cpu")
                self.assertEqual(ids.dtype, torch.int32)

    def test_video_rejects_inconsistent_frame_metadata(self):
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.assemble([2, 4, 4], [0.0])

    def test_rpc_keeps_existing_tensor_fields_without_an_extra_protocol(self):
        from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
        from rtp_llm.server.vit_rpc_server import merge_embedding_results, trans_output

        result, _, _ = self.assemble([2, 4, 4], [0.0, 1.0])
        embedding, positions, ids = result
        output = trans_output(
            merge_embedding_results(
                [MMEmbeddingRes([embedding], [positions], [ids])] * 2
            )
        )
        self.assertNotIn(8, output.DESCRIPTOR.fields_by_number)
        self.assertEqual(list(output.split_size), [embedding.shape[0]] * 2)
        self.assertEqual(list(output.multimodal_extra_input[0].shape), [ids.numel()])
        self.assertEqual(
            output.multimodal_extra_input[0].int32_data, ids.numpy().tobytes()
        )
        self.assertEqual(
            output.multimodal_pos_id.int32_data,
            torch.cat([positions] * 2).numpy().tobytes(),
        )

    def test_cache_hash_includes_semantic_ids_and_positions_without_readback(self):
        from rtp_llm.multimodal.mm_process_engine import _feature_hashes_from_result

        result = (
            [torch.ones(2, 4)],
            [torch.zeros(2, 3, dtype=torch.int32)],
            [torch.tensor([10, 20], dtype=torch.int32)],
        )
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("coordinate readback")
        ):
            reference = _feature_hashes_from_result(result)[0]
            torch.testing.assert_close(
                reference, _feature_hashes_from_result(result)[0]
            )
            for other in (
                (result[0], [result[1][0] + 1], result[2]),
                (result[0], result[1], [result[2][0] + 1]),
            ):
                self.assertFalse(
                    torch.equal(reference, _feature_hashes_from_result(other)[0])
                )

    def test_shared_embedding_cache_round_trip_keeps_complete_tensor_result(self):
        from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
        from rtp_llm.multimodal.mm_remote_embedding_cache import MMRemoteEmbeddingCache
        from rtp_llm.multimodal.test.mm_cache_test_store import Store

        store = Store()
        remote = MMRemoteEmbeddingCache(
            store,
            max_object_bytes=65536,
            max_inflight_bytes=1048576,
            read_timeout_ms=500,
        )
        self.addCleanup(remote.close)
        cache = MMEmbeddingCache(0, 64, on_cpu_evict=remote.submit_eviction)
        self.addCleanup(cache.clear)
        result = (
            [torch.ones(2, 4)],
            [torch.zeros(2, 3, dtype=torch.int32)],
            [torch.tensor([10, 20], dtype=torch.int32)],
        )
        for key in ("image_a", "image_b"):
            _, entry = cache.try_acquire(key)
            cache.complete(key, entry, result)
        self.assertTrue(store.saved.wait(2))
        loaded = remote.load("image_a")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 3)
        for actual, expected in zip(loaded, result):
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)

    def processor(self):
        return SimpleNamespace(
            image_processor=SimpleNamespace(merge_size=2),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=10,
            vision_end_token_id=11,
            tokenizer=SimpleNamespace(
                encode=lambda text, **kwargs: [ord(c) for c in text]
            ),
        )

    def test_semantic_ids_are_independent_from_cache_hashes(self):
        original = torch.tensor([7, -10, 800000, 8, -20, 9], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=original,
            multimodal_inputs=SimpleNamespace(
                mm_features_locs=torch.tensor([1, 4]),
                mm_extra_input=[
                    torch.tensor([20, 20], dtype=torch.int32),
                    torch.tensor([21], dtype=torch.int32),
                ],
                multimodal_features=[torch.zeros(2, 4), torch.zeros(1, 4)],
            ),
        )
        result = semantic_token_ids(inputs)
        self.assertEqual(result.tolist(), [7, 20, 20, 8, 21, 9])
        self.assertEqual(original.tolist(), [7, -10, 800000, 8, -20, 9])

    def test_semantic_marker_covers_only_the_remaining_feature_rows(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for flat in (False, True):
                with self.subTest(device=device, tp_flattened=flat):
                    features = torch.zeros(6, 4, device=device)[2:4]
                    marker = torch.tensor([21, 21], dtype=torch.int32, device=device)
                    if flat:
                        marker = marker.reshape(-1)
                    inputs = SimpleNamespace(
                        input_ids=torch.tensor(
                            [7, -10, -11, 8], dtype=torch.int32, device=device
                        ),
                        multimodal_inputs=SimpleNamespace(
                            mm_features_locs=torch.tensor([1], device=device),
                            mm_extra_input=[marker],
                            multimodal_features=[features],
                        ),
                    )
                    with patch.object(
                        torch.Tensor,
                        "item",
                        side_effect=AssertionError("semantic marker host read"),
                    ):
                        actual = semantic_token_ids(inputs)
                    self.assertEqual(actual.tolist(), [7, 21, 21, 8])

    def check_tail(self, device):
        generator = torch.Generator().manual_seed(31)
        coords = torch.randint(
            0, 400, (270, 3), generator=generator, dtype=torch.int32
        ).to(device)
        pool = torch.full((4, 24), -7, device=device, dtype=torch.int32)
        table = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
        first = position_state([0], [137], coords[:137], pool, table)
        expected = coords[torch.arange(137, device=device) // 4 * 4]
        self.assertTrue(torch.equal(first.group_positions(), expected))
        first.stage()
        for prefix, length in [
            (137, 3),
            (140, 1),
            (141, 2),
            (143, 4),
            (147, 113),
            (260, 3),
        ]:
            state = position_state(
                [prefix], [length], coords[prefix : prefix + length], pool, table
            )
            starts = torch.arange(prefix, prefix + length, device=device) // 4 * 4
            self.assertTrue(
                torch.equal(state.group_positions(), coords[starts]), (prefix, length)
            )
            original = pool.clone()
            state.stage()
            state.rollback()
            self.assertTrue(torch.equal(pool, original))
            state.stage()
        self.assertTrue(
            torch.equal(
                pool[0], torch.full((24,), -7, device=device, dtype=torch.int32)
            )
        )

    def test_coordinate_checks_are_deferred_and_shared_without_pool_aliasing(self):
        positions = torch.arange(12, dtype=torch.int32)
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            combo_position_ids=positions,
            cu_seqlens_device=torch.tensor([0, 4], dtype=torch.int32),
            prefix_lengths_device=torch.tensor([0], dtype=torch.int32),
        )
        table = torch.tensor([[1]], dtype=torch.int32)
        metadata = {}

        def context(pool):
            return SimpleNamespace(
                main_inputs=inputs,
                metadata_cache=metadata,
                position_state_cache=SimpleNamespace(
                    tag=INDEXER_ROPE_POS_TAG, seq_size_per_block=128, kv_cache_base=pool
                ),
                position_state_inputs=SimpleNamespace(kv_cache_block_id_device=table),
            )

        first_pool = torch.zeros((2, 24), dtype=torch.int32)
        second_pool = torch.zeros_like(first_pool)
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("unexpected scalar read")
        ):
            first = QSAPositionState(context(first_pool), 4)
            second = QSAPositionState(context(second_pool), 4)
            checks = DeviceChecks()
            self.assertTrue(first.add_checks(checks))
        self.assertIs(first.geometry, second.geometry)
        with self.assertRaisesRegex(RuntimeError, "pre-write checks"):
            first.stage()
        checks.verify()
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("repeated scalar read")
        ):
            self.assertFalse(second.add_checks(DeviceChecks()))
        first.stage()
        self.assertTrue(torch.equal(second_pool, torch.zeros_like(second_pool)))
        second.stage()
        self.assertTrue(torch.equal(first_pool, second_pool))
        inputs.combo_position_ids = -positions - 1
        invalid = QSAPositionState(context(torch.zeros_like(first_pool)), 4)
        invalid_checks = DeviceChecks()
        invalid.add_checks(invalid_checks)
        with self.assertRaisesRegex(ValueError, "invalid coordinates"):
            invalid_checks.verify()
        with self.assertRaisesRegex(RuntimeError, "pre-write checks"):
            invalid.stage()

    def check_shared_geometry_layer_history(self, device):
        positions = torch.arange(21, dtype=torch.int32, device=device) + 2000
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            combo_position_ids=positions,
            cu_seqlens_device=torch.tensor([0, 7], dtype=torch.int32, device=device),
            prefix_lengths_device=torch.tensor([1], dtype=torch.int32, device=device),
        )
        table = torch.tensor([[1]], dtype=torch.int32, device=device)
        metadata = {}
        pools = [
            torch.arange(48, dtype=torch.int32, device=device).reshape(2, 24),
            (torch.arange(48, dtype=torch.int32, device=device) + 1000).reshape(2, 24),
        ]
        before = [pool.clone() for pool in pools]
        states = [
            QSAPositionState(
                SimpleNamespace(
                    main_inputs=inputs,
                    metadata_cache=metadata,
                    position_state_cache=SimpleNamespace(
                        tag=INDEXER_ROPE_POS_TAG,
                        seq_size_per_block=128,
                        kv_cache_base=pool,
                    ),
                    position_state_inputs=SimpleNamespace(
                        kv_cache_block_id_device=table
                    ),
                ),
                4,
            )
            for pool in pools
        ]
        checks = DeviceChecks()
        states[0].add_checks(checks)
        checks.verify()
        states[1].add_checks(DeviceChecks())
        for state, original in zip(states, before):
            expected = torch.cat(
                (original[1, :3].repeat(3, 1), positions.reshape(7, 3)[3].repeat(4, 1))
            )
            torch.testing.assert_close(state.group_positions(), expected)
            state.stage()
        states[0].rollback()
        torch.testing.assert_close(pools[0], before[0])
        self.assertFalse(torch.equal(pools[1], before[1]))
        states[1].rollback()
        torch.testing.assert_close(pools[1], before[1])

    def test_shared_geometry_keeps_layer_history_and_rollback_cpu(self):
        self.check_shared_geometry_layer_history("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_shared_geometry_keeps_layer_history_and_rollback_cuda(self):
        self.check_shared_geometry_layer_history("cuda")

    def test_pure_text_ple_needs_no_device_scalar_read(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
            ple_input_ids,
        )

        ids = torch.tensor([7, 8], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=ids,
            embedding_inputs=SimpleNamespace(text_tokens_mask=torch.ones_like(ids)),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[], mm_extra_input=[]
            ),
        )
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("text mask readback")
        ):
            self.assertIs(ple_input_ids(inputs, 9), ids)
        inputs.embedding_inputs.text_tokens_mask = torch.ones(3, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "size does not match"):
            ple_input_ids(inputs, 9)

    def test_coordinate_tail_and_page_snapshots_cpu(self):
        self.check_tail("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_coordinate_tail_and_page_snapshots_cuda(self):
        self.check_tail("cuda")

    def test_packed_requests_do_not_share_coordinate_history(self):
        pool = torch.zeros((5, 24), dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        a = torch.arange(33, dtype=torch.int32).reshape(11, 3)
        b = 100 + torch.arange(45, dtype=torch.int32).reshape(15, 3)
        state = position_state([0, 0], [11, 15], torch.cat([a, b]), pool, table)
        state.stage()
        next_a, next_b = torch.tensor([[70, 71, 72]]), torch.tensor([[170, 171, 172]])
        tail = position_state(
            [11, 15], [1, 1], torch.cat([next_a, next_b]), pool, table
        )
        self.assertTrue(torch.equal(tail.group_positions(), torch.stack([a[8], b[12]])))

    def test_qsa_compression_across_visual_prefill_decode_boundary(self):
        from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import (
            QSARuntimeFixture,
        )

        fixture = QSARuntimeFixture()
        fixture.setUp()
        coords = [[17, 20 + i // 3, 30 + i % 3] for i in range(7)]
        positions = torch.zeros((8, 24), dtype=torch.int32)

        def attach(context):
            context.main_inputs.prefix_lengths_device = (
                context.main_inputs.prefix_lengths
            )
            return replace(
                context,
                position_state_cache=SimpleNamespace(
                    tag=INDEXER_ROPE_POS_TAG,
                    seq_size_per_block=8,
                    kv_cache_base=positions,
                ),
                position_state_inputs=context.indexer_state_inputs,
            )

        prefill = attach(
            fixture._single_request_context(
                fixture.kv_base,
                fixture.state_base,
                is_prefill=True,
                input_length=7,
                sequence_length=0,
                position_ids=coords,
            )
        )
        raw = torch.randn(7, fixture.D, dtype=torch.bfloat16)
        indexer = fixture._decode_indexer()
        _, cos, sin, _ = prefill.write_prefill_indexer_cache(
            raw, indexer=indexer, rope_config=fixture.rope_config
        )
        decode = attach(
            fixture._single_request_context(
                fixture.kv_base,
                fixture.state_base,
                is_prefill=False,
                input_length=1,
                sequence_length=7,
                position_ids=[[40, 40, 40]],
            )
        )
        next_raw = torch.randn(1, fixture.D, dtype=torch.bfloat16)
        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score.qsa_paged_indexer_score",
            return_value=torch.tensor([[0.25, 3.0]]),
        ):
            decode.select_decode_tokens(
                torch.randn(1, 2, fixture.D, dtype=torch.bfloat16),
                next_raw,
                indexer=indexer,
                rope_config=fixture.rope_config,
            )
        expected = fixture._expected(
            torch.cat([raw[4:], next_raw]), cos[0, 4], sin[0, 4], 4
        )
        pool = fixture.kv_base.view(torch.bfloat16).view(7, 2, fixture.D)
        torch.testing.assert_close(pool[3, 1], expected, atol=0, rtol=0)

    def test_ple_uses_semantic_ids_with_visual_mask(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import (
            ple_input_ids,
        )

        inputs = SimpleNamespace(
            input_ids=torch.tensor([7, -100, 800000, 8]),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([1, 0, 0, 1])
            ),
            multimodal_inputs=SimpleNamespace(
                mm_features_locs=torch.tensor([1]),
                mm_extra_input=[torch.tensor([20, 20], dtype=torch.int32)],
                multimodal_features=[torch.zeros(2, 4)],
            ),
        )
        self.assertEqual(ple_input_ids(inputs, 9).tolist(), [7, 20, 20, 8])

    def test_draft_graph_normalizes_target_visual_position_offset(self):
        from unittest.mock import Mock

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
            DraftPrefillGraphReplay,
        )

        groups = {}
        for tag in ("full", "indexer_kv", "indexer_state"):
            groups[tag] = SimpleNamespace(
                input_lengths=torch.tensor([3], dtype=torch.int32),
                input_lengths_device=torch.tensor([3], dtype=torch.int32),
                prefix_lengths=torch.tensor([137], dtype=torch.int32),
                prefix_lengths_device=torch.tensor([137], dtype=torch.int32),
                cu_seqlens_device=torch.tensor([0, 3], dtype=torch.int32),
                kv_cache_kernel_block_id=torch.tensor([[1, 2]], dtype=torch.int32),
                kv_cache_block_id_device=torch.tensor([[1, 2]], dtype=torch.int32),
                combo_position_ids=torch.tensor(
                    [40, 40, 40, 41, 41, 41, 42, 42, 42, 0, 0, 0], dtype=torch.int32
                ),
            )
        impl = Mock()
        replay = DraftPrefillGraphReplay(
            {"full": impl},
            [
                ("full", 8, 128, False),
                ("indexer_kv", 8, 32, True),
                ("indexer_state", 8, 128, False),
            ],
            3,
            4,
        )
        replay.prepare_cuda_graph(groups)
        for value in groups.values():
            self.assertEqual(
                value.combo_position_ids.reshape(-1, 3).tolist(),
                [[137] * 3, [138] * 3, [139] * 3, [0] * 3],
            )
        impl.prepare_cuda_graph.assert_called_once_with(groups["full"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_coordinate_graph_replay_uses_dynamic_input(self):
        pool = torch.zeros((3, 24), device="cuda", dtype=torch.int32)
        table = torch.tensor([[1, 2]], device="cuda", dtype=torch.int32)
        positions = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
        prefixes = torch.zeros(1, device="cuda", dtype=torch.int32)
        lengths = torch.full((1,), 4, device="cuda", dtype=torch.int32)
        warm = position_state(prefixes, lengths, positions, pool, table, graph=True)
        warm.stage()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state = position_state(
                prefixes, lengths, positions, pool, table, graph=True
            )
            groups = state.group_positions()
            state.stage()
        for offset in (100, 230, 700):
            positions.copy_(
                torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
                + offset
            )
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(groups, positions[:1].expand(4, -1)))
            self.assertTrue(torch.equal(pool[1].reshape(8, 3)[:4], positions))


if __name__ == "__main__":
    unittest.main()

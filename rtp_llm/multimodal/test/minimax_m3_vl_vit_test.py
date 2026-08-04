import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import (
    minimax_m3_vl_rope as rope_module,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import (
    minimax_m3_vl_vit as vit_module,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
    MiniMaxM3VLDeployWeightInfo,
    MiniMaxM3VLImageEmbedding,
    _MiniMaxM3VLPreprocessBuffers,
    _MiniMaxM3VLVisionGraphCache,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_vit import (
    CLIPAttention,
    MiniMaxM3VLVisionModel,
    VisionConfig,
    _apply_rope,
    get_fused_qkv_checkpoint_names,
)


class MiniMaxM3VLWorkEstimateTest(unittest.TestCase):
    def setUp(self):
        self.embedding = object.__new__(MiniMaxM3VLImageEmbedding)
        self.embedding.mm_processor = SimpleNamespace(
            patch_size=14,
            max_pixels=451584,
        )
        self.embedding.temporal_patch_size = 2
        self.embedding.merge_size = 2
        self.embedding.visual = SimpleNamespace(
            dtype=torch.bfloat16,
            vision_config=VisionConfig(),
        )

    def test_image_work_estimate_is_exact(self):
        raw = torch.zeros(3, 16, 16, dtype=torch.uint8)
        estimate = self.embedding.estimate_work((raw, (448, 448), None))

        self.assertEqual(estimate.input_patches, 1024)
        self.assertEqual(estimate.output_tokens, 258)
        self.assertEqual(estimate.max_attention_segment, 1024)
        self.assertEqual(estimate.attention_work, 1024**2)
        self.assertEqual(estimate.estimated_workspace_bytes, 1024 * 40960)

    def test_video_work_estimate_includes_padding_and_timestamps(self):
        raw = torch.zeros(5, 1, 1, 3, dtype=torch.uint8)
        timestamps = [[1, 2], [3], [4, 5, 6]]
        estimate = self.embedding.estimate_work((raw, (448, 448), timestamps))

        self.assertEqual(estimate.input_patches, 3 * 1024)
        self.assertEqual(estimate.output_tokens, 768 + 6 + 6)
        self.assertEqual(estimate.max_attention_segment, 3 * 1024)
        self.assertEqual(estimate.attention_work, (3 * 1024) ** 2)

    def test_timestamp_group_mismatch_is_rejected(self):
        raw = torch.zeros(5, 1, 1, 3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "timestamp group count"):
            self.embedding.estimate_work((raw, (448, 448), [[1], [2]]))

    def test_batch_budget_uses_existing_media_cap(self):
        budget = self.embedding.get_batch_work_budget(32)

        self.assertEqual(budget.input_patches, 32 * 2304)
        self.assertEqual(budget.output_tokens, 32 * 578)
        self.assertEqual(budget.max_attention_segment, 4 * 2304)
        self.assertEqual(budget.attention_work, 32 * (2304**2))
        self.assertIsNone(self.embedding.get_batch_work_budget(1 << 30))


class MiniMaxM3VLGpuFoldTest(unittest.TestCase):
    class _FakeVisual(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            self.dtype = torch.bfloat16

    def setUp(self):
        self.embedding = object.__new__(MiniMaxM3VLImageEmbedding)
        self.embedding.mm_processor = SimpleNamespace(
            patch_size=2,
            image_mean=(0.25, 0.5, 0.75),
            image_std=(0.5, 0.25, 0.125),
            rescale_factor=1.0 / 255.0,
        )
        self.embedding.temporal_patch_size = 2
        self.embedding.merge_size = 2
        self.embedding.visual = self._FakeVisual()
        self.embedding._preprocess_buffers = _MiniMaxM3VLPreprocessBuffers(
            self.embedding.mm_processor.image_mean,
            self.embedding.mm_processor.image_std,
        )

    def _reference_fold(self, frames_nchw, target_hw):
        p = self.embedding.mm_processor
        frames = frames_nchw.float()
        frames = torchvision.transforms.functional.resize(
            frames,
            list(target_hw),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )
        video = frames.unsqueeze(0) * p.rescale_factor
        mean = torch.tensor(p.image_mean).view(1, 1, 3, 1, 1)
        std = torch.tensor(p.image_std).view(1, 1, 3, 1, 1)
        video = (video - mean) / std

        temporal_patch_size = self.embedding.temporal_patch_size
        pad_n = (
            temporal_patch_size - video.shape[1] % temporal_patch_size
        ) % temporal_patch_size
        if pad_n:
            video = torch.cat([video, video[:, -1:].repeat(1, pad_n, 1, 1, 1)], dim=1)

        batch, frames, channel, height, width = video.shape
        patch_size = p.patch_size
        merge_size = self.embedding.merge_size
        grid_t = frames // temporal_patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches = video.view(
            batch,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        ).to(torch.bfloat16)

    def test_fold_matches_previous_layout_and_reuses_device_constants(self):
        torch.manual_seed(20260803)
        raw = torch.randint(0, 256, (1, 3, 5, 7), dtype=torch.uint8)
        expected = self._reference_fold(raw, (8, 8))
        mean_ptr = self.embedding._preprocess_buffers.image_mean.data_ptr()
        std_ptr = self.embedding._preprocess_buffers.image_std.data_ptr()

        actual, grid_thw = self.embedding._gpu_fold(raw, (8, 8))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            grid_thw, torch.tensor([[1, 4, 4]], dtype=torch.long)
        )
        self.assertEqual(
            self.embedding._preprocess_buffers.image_mean.data_ptr(), mean_ptr
        )
        self.assertEqual(
            self.embedding._preprocess_buffers.image_std.data_ptr(), std_ptr
        )

    def test_fold_writes_directly_into_packed_destination_slice(self):
        raw = torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(1, 3, 4, 4)
        expected = self._reference_fold(raw, (4, 4))
        destination = torch.full(
            (expected.shape[0] + 4, expected.shape[1]),
            torch.nan,
            dtype=torch.bfloat16,
        )

        actual, _ = self.embedding._gpu_fold(
            raw,
            (4, 4),
            pixel_values_out=destination[: expected.shape[0]],
        )

        self.assertEqual(actual.data_ptr(), destination.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isnan(destination[expected.shape[0] :]).all().item())


class MiniMaxM3VLVisionAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260727)
        self.config = VisionConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )

    def _position_embeddings(
        self, sequence_length, device=None, head_dim=None, rotary_dim=None
    ):
        if head_dim is None:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
        if rotary_dim is None:
            rotary_dim = head_dim
        angles = torch.randn(sequence_length, 1, rotary_dim // 2, device=device)
        return (
            angles.cos().repeat(1, 1, 2),
            angles.sin().repeat(1, 1, 2),
        )

    def _unfused_reference(
        self, attention, hidden_states, position_embeddings, offsets
    ):
        sequence_length = hidden_states.shape[0]
        embed_dim = attention.embed_dim
        projections = []
        for index in range(3):
            start = index * embed_dim
            end = start + embed_dim
            projection = F.linear(
                hidden_states,
                attention.qkv_proj.weight[start:end],
                attention.qkv_proj.bias[start:end],
            )
            projections.append(
                projection.view(
                    sequence_length,
                    attention.num_heads,
                    attention.head_dim,
                )
            )

        q, k, v = projections
        q, k = _apply_rope(q, k, *position_embeddings)
        output = attention._segmented_sdpa(q, k, v, offsets)
        return attention.out_proj(output.reshape(sequence_length, embed_dim))

    def test_fused_qkv_matches_unfused_projection(self):
        attention = CLIPAttention(self.config)
        hidden_states = torch.randn(7, self.config.hidden_size)
        offsets = (0, 3, 7)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
        position_embeddings = self._position_embeddings(7)

        expected = self._unfused_reference(
            attention, hidden_states, position_embeddings, offsets
        )
        actual = attention(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen=4,
            segment_offsets=offsets,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(attention.last_backend, "sdpa")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fused_qkv_rope_matches_eager_and_packs_outputs(self):
        sequence_length = 11
        num_heads = 2
        head_dim = 80
        rotary_dim = 78
        qkv = torch.randn(
            sequence_length,
            3,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        angles = torch.randn(
            sequence_length,
            1,
            rotary_dim // 2,
            device="cuda",
        )
        cos = angles.cos().repeat(1, 1, 2)
        sin = angles.sin().repeat(1, 1, 2)

        expected_q, expected_k = _apply_rope(qkv[:, 0], qkv[:, 1], cos, sin)
        expected_v = qkv[:, 2].contiguous()
        actual = rope_module.fused_qkv_rope(qkv, cos, sin)
        self.assertIsNotNone(actual)
        actual_q, actual_k, actual_v = actual

        torch.testing.assert_close(actual_q, expected_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_k, expected_k, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
        self.assertTrue(actual_q.is_contiguous())
        self.assertTrue(actual_k.is_contiguous())
        self.assertTrue(actual_v.is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_packed_attention_matches_segmented_sdpa(self):
        config = VisionConfig(
            hidden_size=160,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=320,
        )
        attention = CLIPAttention(config).cuda().to(torch.bfloat16)
        backend = vit_module._select_attention_backend(torch.empty(1, device="cuda"))
        if backend == "sdpa":
            self.skipTest("packed vision attention backend is unavailable")

        hidden_states = torch.randn(
            16,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        offsets = (0, 7, 16)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        position_embeddings = self._position_embeddings(
            16,
            device="cuda",
            head_dim=attention.head_dim,
            rotary_dim=78,
        )

        expected = self._unfused_reference(
            attention, hidden_states, position_embeddings, offsets
        )
        vision_model = MiniMaxM3VLVisionModel(config).cuda().to(torch.bfloat16)
        attention_context = vision_model._prepare_attention_context(
            hidden_states, cu_seqlens
        )
        actual = attention(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen=9,
            segment_offsets=offsets,
            attention_context=attention_context,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(attention.last_backend, backend)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_packed_attention_handles_representative_image_sequence(self):
        config = VisionConfig(
            hidden_size=1280,
            num_hidden_layers=1,
            num_attention_heads=16,
            intermediate_size=5120,
        )
        attention = CLIPAttention(config).cuda().to(torch.bfloat16)
        sequence_length = 2204
        hidden_states = torch.randn(
            sequence_length,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        offsets = (0, sequence_length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        position_embeddings = self._position_embeddings(
            sequence_length,
            device="cuda",
            head_dim=attention.head_dim,
            rotary_dim=78,
        )
        vision_model = MiniMaxM3VLVisionModel(config).cuda().to(torch.bfloat16)
        attention_context = vision_model._prepare_attention_context(
            hidden_states, cu_seqlens
        )
        if attention_context.backend == "sdpa":
            self.skipTest("packed vision attention backend is unavailable")

        actual = hidden_states
        for _ in range(32):
            actual = attention(
                actual,
                cu_seqlens,
                position_embeddings,
                max_seqlen=sequence_length,
                segment_offsets=offsets,
                attention_context=attention_context,
            )

        self.assertEqual(actual.shape, (sequence_length, config.hidden_size))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertEqual(attention.last_backend, attention_context.backend)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_replays_packed_vision_model(self):
        config = VisionConfig(
            hidden_size=160,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=320,
            patch_size=14,
        )
        model = MiniMaxM3VLVisionModel(config).cuda().to(torch.bfloat16).eval()
        patch_dim = 3 * 2 * config.patch_size * config.patch_size
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        sample = torch.randn(4, patch_dim, device="cuda", dtype=torch.bfloat16)
        backend = vit_module._select_attention_backend(sample)
        if backend not in ("fa4", "flash_attn"):
            self.skipTest(f"vision backend {backend} is not graph-enabled")

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(3):
                model(sample, grid_thw)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        static_input = sample.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_output = model(static_input, grid_thw)

        replay_input = torch.randn_like(sample)
        expected = model(replay_input, grid_thw)
        static_input.copy_(replay_input)
        graph.replay()

        torch.testing.assert_close(
            static_output,
            expected,
            rtol=2e-2,
            atol=2e-2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_graph_cache_captures_second_shape_and_replays(self):
        config = VisionConfig(
            hidden_size=160,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=320,
            patch_size=14,
        )
        model = MiniMaxM3VLVisionModel(config).cuda().to(torch.bfloat16).eval()
        patch_dim = 3 * 2 * config.patch_size * config.patch_size
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        sample = torch.randn(4, patch_dim, device="cuda", dtype=torch.bfloat16)
        backend = vit_module._select_attention_backend(sample)
        if backend not in ("fa4", "flash_attn", "flashinfer"):
            self.skipTest(f"vision backend {backend} is not graph-enabled")

        graph_context = model.prepare_cuda_graph_attention_context(
            grid_thw,
            sample.device,
            sample.dtype,
        )
        graph_context_output = model(
            sample,
            grid_thw,
            attention_context=graph_context,
        )
        eager_output = model(sample, grid_thw)
        torch.testing.assert_close(
            graph_context_output,
            eager_output,
            rtol=2e-2,
            atol=2e-2,
        )

        cache = _MiniMaxM3VLVisionGraphCache(model, max_entries=2, capture_after=2)
        inputs = [sample, torch.randn_like(sample), torch.randn_like(sample)]
        for index, graph_input in enumerate(inputs):
            with self.subTest(index=index):
                expected = model(graph_input, grid_thw)
                actual = cache.run(graph_input, grid_thw)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        self.assertEqual(
            cache.stats(),
            {"hit": 1, "miss": 2, "capture": 1, "fallback": 0},
        )

        stream_inputs = [torch.randn_like(sample), torch.randn_like(sample)]
        stream_expected = [model(value, grid_thw) for value in stream_inputs]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        stream_actual = []
        for stream, value in zip(streams, stream_inputs):
            with torch.cuda.stream(stream):
                stream_actual.append(cache.run(value, grid_thw))
        torch.cuda.synchronize()
        for actual, expected in zip(stream_actual, stream_expected):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(
            cache.stats(),
            {"hit": 3, "miss": 2, "capture": 1, "fallback": 0},
        )

        packed_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
        packed_input = torch.randn(8, patch_dim, device="cuda", dtype=torch.bfloat16)
        expected = model(packed_input, packed_grid_thw)
        actual = cache.run(packed_input, packed_grid_thw)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertEqual(
            cache.stats(),
            {"hit": 3, "miss": 2, "capture": 1, "fallback": 0},
        )

    def test_segments_do_not_attend_to_each_other(self):
        attention = CLIPAttention(self.config)
        hidden_states = torch.randn(7, self.config.hidden_size)
        changed_hidden_states = hidden_states.clone()
        changed_hidden_states[3:] += 100.0
        offsets = (0, 3, 7)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
        position_embeddings = self._position_embeddings(7)

        original = attention(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen=4,
            segment_offsets=offsets,
        )
        changed = attention(
            changed_hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen=4,
            segment_offsets=offsets,
        )

        torch.testing.assert_close(original[:3], changed[:3], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[3:], changed[3:]))

    def test_attention_metadata_is_computed_from_segment_shapes(self):
        model = MiniMaxM3VLVisionModel(self.config)
        cu_seqlens, max_seqlen, offsets = model._compute_attention_metadata(
            [[1, 2, 3], [2, 1, 2]], torch.device("cpu")
        )

        torch.testing.assert_close(
            cu_seqlens, torch.tensor([0, 6, 10], dtype=torch.int32)
        )
        self.assertEqual(max_seqlen, 6)
        self.assertEqual(offsets, (0, 6, 10))


class MiniMaxM3VLFusedWeightTest(unittest.TestCase):
    def test_fused_qkv_checkpoint_name_mapping(self):
        live_name = (
            "vision_tower.vision_model.encoder.layers.3." "self_attn.qkv_proj.weight"
        )
        self.assertEqual(
            get_fused_qkv_checkpoint_names(live_name),
            (
                "vision_tower.vision_model.encoder.layers.3." "self_attn.q_proj.weight",
                "vision_tower.vision_model.encoder.layers.3." "self_attn.k_proj.weight",
                "vision_tower.vision_model.encoder.layers.3." "self_attn.v_proj.weight",
            ),
        )
        self.assertIsNone(
            get_fused_qkv_checkpoint_names(
                "vision_tower.vision_model.encoder.layers.3."
                "self_attn.out_proj.weight"
            )
        )

    def test_deploy_weight_info_concatenates_qkv(self):
        fused_weight = (
            "vision_tower.vision_model.encoder.layers.0." "self_attn.qkv_proj.weight"
        )
        fused_bias = (
            "vision_tower.vision_model.encoder.layers.0." "self_attn.qkv_proj.bias"
        )
        out_weight = (
            "vision_tower.vision_model.encoder.layers.0." "self_attn.out_proj.weight"
        )

        class FakeVitWeights:
            weight_names = [fused_weight, fused_bias, out_weight]
            ckpt_prefix = ""

        weight_info = MiniMaxM3VLDeployWeightInfo(
            vit_config=None, vit_weights=FakeVitWeights()
        ).get_weight_info()
        weights = {weight.name: weight for weight in weight_info.weights}

        checkpoint_names = [
            info.tensor_name(None) for info in weights[fused_weight].weights
        ]
        self.assertEqual(
            checkpoint_names,
            list(get_fused_qkv_checkpoint_names(fused_weight)),
        )

        q = torch.full((2, 3), 1.0)
        k = torch.full((2, 3), 2.0)
        v = torch.full((2, 3), 3.0)
        merged = weights[fused_weight].process_fun([q, k, v])
        torch.testing.assert_close(merged, torch.cat([q, k, v], dim=0))
        self.assertEqual(
            [info.tensor_name(None) for info in weights[out_weight].weights],
            [out_weight],
        )


if __name__ == "__main__":
    unittest.main()

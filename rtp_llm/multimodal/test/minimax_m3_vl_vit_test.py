import unittest

import torch
import torch.nn.functional as F

from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import (
    minimax_m3_vl_rope as rope_module,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import (
    minimax_m3_vl_vit as vit_module,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
    MiniMaxM3VLDeployWeightInfo,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_vit import (
    CLIPAttention,
    MiniMaxM3VLVisionModel,
    VisionConfig,
    _apply_rope,
    get_fused_qkv_checkpoint_names,
)


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

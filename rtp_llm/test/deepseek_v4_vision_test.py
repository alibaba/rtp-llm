import io
from types import SimpleNamespace
from unittest import TestCase, main, mock

import torch
from PIL import Image

from rtp_llm.models.deepseek_v4 import DeepSeekV4
from rtp_llm.models.deepseek_v4_vision import (
    IMAGE_END,
    IMAGE_START,
    Aligner,
    DeepSeekV4VisionEmbedding,
    DeepSeekV4VisionWeights,
    RMSNorm,
    build_image_attention_spans,
    build_image_block,
)
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class DeepSeekV4VisionTest(TestCase):
    def test_image_block_alignment(self):
        for start_pos in (0, 1, 2, 3, 18, 24, 405):
            types, permutation = build_image_block(8, 12, start_pos)
            image_start = int((types == IMAGE_START).nonzero().item())
            self.assertEqual((start_pos + image_start) % 4, 3)
            self.assertEqual(int(types[-1]), IMAGE_END)
            self.assertEqual(permutation.numel(), 8 * 12)

    def test_tiny_encoder_uses_multimodal_config(self):
        vision_config = {
            "hidden_size": 8,
            "vision_n_layers": 1,
            "vision_dim": 16,
            "vision_n_heads": 2,
            "vision_inter_dim": 32,
            "vision_patch_size": 2,
            "vision_rope_theta": 10000.0,
            "vision_downsample_ratio": 1,
            "vision_max_n_token": 32,
            "vision_min_pixels": 16,
            "vision_max_wh_ratio": 8,
        }
        encoder = DeepSeekV4VisionEmbedding(
            SimpleNamespace(config=vision_config),
            SimpleNamespace(compute_dtype=torch.float32),
        )
        with mock.patch(
            "rtp_llm.models.deepseek_v4_vision.F.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            output = encoder.image_embedding([Image.new("RGB", (4, 4))], start_pos=3)[0]
        self.assertEqual(output.shape, (10, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(sdpa.call_args.args[0].dim(), 4)

    def test_loader_preserves_fp32_vision_norm_weights(self):
        vision_config = {
            "hidden_size": 8,
            "vision_n_layers": 1,
            "vision_dim": 16,
            "vision_n_heads": 2,
            "vision_inter_dim": 32,
            "vision_patch_size": 2,
            "vision_rope_theta": 10000.0,
            "vision_downsample_ratio": 1,
        }
        encoder = DeepSeekV4VisionEmbedding(
            SimpleNamespace(config=vision_config),
            SimpleNamespace(compute_dtype=torch.bfloat16),
        )
        parts = {
            "vision": encoder.vision,
            "aligner": encoder.aligner,
            "image_start": encoder.image_start,
            "image_end": encoder.image_end,
            "image_newline": encoder.image_newline,
            "image_pad": encoder.image_pad,
        }
        vit_weights = DeepSeekV4VisionWeights(parts)
        loaded = {
            name: torch.ones_like(
                dict(encoder.named_parameters())[name], dtype=torch.bfloat16
            )
            for name in vit_weights.weight_names
        }
        owner = SimpleNamespace(
            mm_part=encoder,
            weight=SimpleNamespace(
                get_global_weight_or_none=lambda name: loaded.get(name)
            ),
        )

        DeepSeekV4._load_mm_weight(
            owner,
            SimpleNamespace(vit_weights=vit_weights),
            torch.bfloat16,
            "cpu",
        )

        norm_weights = [
            module.weight for module in encoder.modules() if isinstance(module, RMSNorm)
        ]
        self.assertTrue(norm_weights)
        self.assertTrue(all(weight.dtype == torch.float32 for weight in norm_weights))
        self.assertEqual(encoder.vision.patch_embed.proj.weight.dtype, torch.bfloat16)

    def test_aligner_layout_matches_unfold(self):
        aligner = Aligner(
            {
                "vision_dim": 2,
                "vision_downsample_ratio": 3,
                "hidden_size": 4,
            }
        )
        source = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4 * 5, 2)
        with mock.patch.object(aligner.w1, "forward", wraps=aligner.w1.forward) as w1:
            aligner(source, 4, 5)
        actual = w1.call_args.args[0]

        chw = source.view(4, 5, 2).permute(2, 0, 1)
        chw = torch.nn.functional.pad(chw, (0, 1, 0, 2))
        expected = (
            torch.nn.functional.unfold(chw.unsqueeze(0), 3, stride=3)
            .squeeze(0)
            .transpose(0, 1)
        )
        torch.testing.assert_close(actual, expected)

    def test_embedding_cache_is_scoped_by_image_start_phase(self):
        vision_config = {
            "hidden_size": 8,
            "vision_n_layers": 1,
            "vision_dim": 16,
            "vision_n_heads": 2,
            "vision_inter_dim": 32,
            "vision_patch_size": 2,
            "vision_rope_theta": 10000.0,
            "vision_downsample_ratio": 1,
            "vision_max_n_token": 32,
            "vision_min_pixels": 16,
            "vision_max_wh_ratio": 8,
        }
        encoder = DeepSeekV4VisionEmbedding(
            SimpleNamespace(config=vision_config),
            SimpleNamespace(compute_dtype=torch.float32),
        )
        image_bytes = io.BytesIO()
        Image.new("RGB", (4, 4)).save(image_bytes, format="PNG")
        payload = image_bytes.getvalue()
        phase1 = SimpleNamespace(image_block_start_mod4=1)
        phase2 = SimpleNamespace(image_block_start_mod4=2)

        with mock.patch(
            "rtp_llm.models.deepseek_v4_vision.get_bytes_io_from_url",
            side_effect=lambda *_args, **_kwargs: io.BytesIO(payload),
        ), mock.patch.object(
            encoder, "mm_process", wraps=encoder.mm_process
        ) as process:
            first, _ = encoder.mm_embedding("cache-test-image", 1, configs=phase1)
            again, _ = encoder.mm_embedding("cache-test-image", 1, configs=phase1)
            other_phase, _ = encoder.mm_embedding("cache-test-image", 1, configs=phase2)

        self.assertIs(first, again)
        self.assertEqual(process.call_count, 2)
        self.assertNotEqual(first.size(0), other_phase.size(0))

    def test_image_attention_spans_keep_original_position_across_reuse(self):
        raw_spans = torch.tensor([[0, 2, 12], [1, 8, 20]])
        spans = build_image_attention_spans(
            raw_spans, torch.tensor([5, 4]), device="cpu"
        )
        torch.testing.assert_close(spans, torch.tensor([[0, 3, 11], [1, 11, 19]]))

        with self.assertRaisesRegex(RuntimeError, "too deep inside an image block"):
            build_image_attention_spans(
                torch.tensor([[0, 2, 300]]),
                torch.tensor([200]),
                device="cpu",
            )

    def test_mm_process_engine_preserves_image_start_phase(self):
        class FakeMMPart:
            def __init__(self):
                self.starts = []

            def mm_embedding(self, url, mm_type, download_headers, configs):
                self.starts.append(configs.image_block_start_mod4)
                rows = 6 if url == "first" else 2
                return torch.zeros(rows, 1), None

        mm_part = FakeMMPart()
        model = SimpleNamespace(
            model_config=SimpleNamespace(
                mm_model_config=SimpleNamespace(mm_position_ids_style=0),
                mm_related_params=SimpleNamespace(support_batch=False),
            ),
            mm_part=mm_part,
        )
        engine = MMProcessEngine(model, SimpleNamespace(download_headers=""))
        output = engine.submit(
            ["first", "second"],
            preprocess_configs=[
                [-1, -1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, -1, 3],
            ],
        )

        self.assertEqual(mm_part.starts, [1, 3])
        self.assertEqual([feature.size(0) for feature in output.embeddings], [6, 2])


if __name__ == "__main__":
    main()

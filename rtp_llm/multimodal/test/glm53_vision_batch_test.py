"""Attention isolation, packed media ordering, budgets and inference kernels."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.glm5_3_flash_mixin import (
    Glm53FlashImageEmbedding,
    Glm53FlashVisionAttention,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate


def reference_norm(x, weight, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(
        x.dtype
    ) * weight


class VisionBatchTest(unittest.TestCase):
    def test_group_attention_preserves_segment_isolation(self):
        torch.manual_seed(13)
        attention = Glm53FlashVisionAttention(
            SimpleNamespace(hidden_size=32, num_heads=4, attention_bias=True)
        )
        x = torch.randn(18, 32)
        freqs = torch.randn(18, 4)
        lengths = [4, 4, 6, 4]
        with torch.inference_mode():
            q, k, v = attention.qkv(x).view(18, 3, 4, 8).unbind(1)
            q, k = attention._apply_rope(
                attention.q_norm(q), attention.k_norm(k), freqs
            )
            outputs = []
            start = 0
            for length in lengths:
                values = [
                    t[start : start + length].transpose(0, 1).unsqueeze(0)
                    for t in (q, k, v)
                ]
                outputs.append(
                    F.scaled_dot_product_attention(*values).squeeze(0).transpose(0, 1)
                )
                start += length
            expected = attention.proj(torch.cat(outputs).reshape(x.shape))
            with patch.object(
                F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
            ) as sdpa:
                actual = attention(x, lengths, freqs)
                self.assertEqual(sdpa.call_count, 3)
            torch.testing.assert_close(actual, expected)
            altered = x.clone()
            altered[4:8] *= 100
            changed = attention(altered, lengths, freqs)
            torch.testing.assert_close(changed[:4], actual[:4], rtol=0, atol=0)
            torch.testing.assert_close(changed[8:], actual[8:], rtol=0, atol=0)

    def test_packed_media_restore_order_and_timestamps_and_split_budget(self):
        class Visual:
            config = SimpleNamespace(
                hidden_size=4, intermediate_size=8, out_hidden_size=4
            )
            spatial_merge_size = 2
            patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1)))
            calls = []

            def __call__(self, pixels, grid):
                self.calls.append(grid.tolist())
                return pixels.reshape(-1, 4, 2).mean(1)

        emb = Glm53FlashImageEmbedding.__new__(Glm53FlashImageEmbedding)
        emb.visual = Visual()
        emb.special_token_ids = {"image_start": 11, "image_end": 12}
        emb._encode_timestamp = lambda t: [100 + int(t)]
        data = [
            (torch.full((16, 2), 3.0), torch.tensor([[1, 4, 4]])),
            (torch.full((8, 2), 7.0), torch.tensor([[2, 2, 2]]), [0, 1]),
            (torch.full((4, 2), 9.0), torch.tensor([[1, 2, 2]])),
        ]
        expected = [emb.embedding(x) for x in data]
        emb.visual.calls.clear()
        actual = emb.batched_embedding(data, [None] * 3)
        self.assertEqual(len(emb.visual.calls), 1)
        self.assertEqual(emb.visual.calls[0], [[2, 2, 2], [1, 2, 2], [1, 4, 4]])
        for left, right in zip(actual, expected):
            self.assertIsNone(left[1])
            for a, b in zip(left[0] + left[2], right[0] + right[2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        emb.visual.calls.clear()
        emb.get_batch_work_budget = lambda _: MMWorkEstimate(input_patches=10)
        actual = emb.batched_embedding(data, [None] * 3)
        self.assertEqual(len(emb.visual.calls), 3)
        self.assertEqual(emb.batched_embedding([], []), [])
        with self.assertRaises(ValueError):
            emb.batched_embedding(data, [None])

    def test_model_default_and_explicit_batch_override(self):
        config = VitConfig()
        self.assertEqual(config.embedding_scheduler_args()["max_batch_size"], 1)
        self.assertEqual(
            config.embedding_scheduler_args(default_gpu_batch=True)["max_batch_size"], 8
        )
        config.use_gpu_batch = False
        self.assertEqual(
            config.embedding_scheduler_args(default_gpu_batch=True)["max_batch_size"], 1
        )
        config.use_gpu_batch = True
        self.assertEqual(config.embedding_scheduler_args()["max_batch_size"], 8)

    def test_existing_environment_and_cli_override_model_default(self):
        import os

        from rtp_llm.server.server_args.server_args import EnvArgumentParser
        from rtp_llm.server.server_args.vit_group_args import init_vit_group_args

        for env_value, args, expected in (
            (None, [], None),
            ("0", [], False),
            ("1", [], True),
            ("1", ["--use_gpu_batch", "0"], False),
        ):
            with patch.dict(os.environ):
                os.environ.pop("VIT_USE_GPU_BATCH", None)
                if env_value is not None:
                    os.environ["VIT_USE_GPU_BATCH"] = env_value
                config = VitConfig()
                parser = EnvArgumentParser()
                parser.set_root_config(SimpleNamespace(vit_config=config))
                init_vit_group_args(parser, config)
                parser.parse_args(args)
                self.assertIs(config.use_gpu_batch, expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VisionKernelTest(unittest.TestCase):
    def test_rms_and_qk_rope_intermediate_rounding(self):
        from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.vision_kernels import (
            qk_norm_rope,
            rms_norm,
        )

        torch.manual_seed(17)
        records = []
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for tokens in (1, 7, 129, 1024):
                    for width in (64, 1024):
                        x = torch.randn(tokens, width, device="cuda", dtype=dtype) * 3
                        weight = torch.randn(width, device="cuda", dtype=dtype)
                        expected = reference_norm(x, weight, 1e-6)
                        actual = rms_norm(x, weight, 1e-6)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    qkv = torch.randn(tokens, 3, 16, 64, device="cuda", dtype=dtype)
                    qw = torch.randn(64, device="cuda", dtype=dtype)
                    kw = torch.randn_like(qw)
                    freqs = torch.randn(tokens, 32, device="cuda")
                    emb = torch.cat((freqs, freqs), -1)
                    cos, sin = emb.cos(), emb.sin()
                    actual = qk_norm_rope(qkv, qw, kw, cos, sin)
                    for index, weight in enumerate((qw, kw)):
                        norm = reference_norm(qkv[:, index], weight, 1e-5).float()
                        half1, half2 = norm.chunk(2, -1)
                        expected = (
                            norm * cos[:, None]
                            + torch.cat((-half2, half1), -1) * sin[:, None]
                        ).to(dtype)
                        torch.testing.assert_close(
                            actual[index], expected, rtol=0, atol=0
                        )
                        records.append(
                            dict(
                                dtype=str(dtype),
                                tokens=tokens,
                                max_abs=float(
                                    (actual[index].float() - expected.float())
                                    .abs()
                                    .max()
                                ),
                            )
                        )
        print("KERNEL_PRECISION", records)


if __name__ == "__main__":
    unittest.main()

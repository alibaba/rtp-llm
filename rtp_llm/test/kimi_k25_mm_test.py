"""Minimal contract tests for Kimi-K2.5 multimodal pieces.

Covers two pieces of behavior that previously had silent breakage:

1. ``KimiK25ImageEmbedding.image_embedding`` must return a per-image
   ``List[Tensor]`` shaped ``(num_tokens, hidden)``, so callers that
   index ``image_embedding([img])[0]`` get the full projected sequence
   for that image rather than a single row of a concatenated tensor.

2. ``KimiK25Renderer`` must reject ``video_url`` content parts up front:
   the in-tree image processor is image-only and the ckpt-side processor
   is unavailable in this build.

These tests stub the heavy ViT + projector + processor so they run
without checkpoint weights or a GPU.
"""

from unittest import TestCase, main
from unittest.mock import MagicMock

import torch
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_image_processor import (
    KimiK25VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_vit import (
    KimiK25ImageEmbedding,
)
from rtp_llm.openai.api_datatype import (
    ChatMessage,
    ContentPart,
    ContentPartTypeEnum,
    ImageURL,
    RoleEnum,
)
from rtp_llm.openai.renderers.kimi_k25_renderer import KimiK25Renderer

_HIDDEN = 8
_TOKENS_PER_IMAGE = 5


class _StubKimiK25ImageEmbedding(KimiK25ImageEmbedding):
    """Subclass that overrides `_device` / `_data_type` so we don't need a
    real `vision_tower.patch_embed.proj.weight` to satisfy the properties."""

    @property
    def _device(self):
        return torch.device("cpu")

    @property
    def _data_type(self):
        return torch.float32


def _build_stub_embedding() -> KimiK25ImageEmbedding:
    """Return a KimiK25ImageEmbedding with vision_tower / mm_projector /
    image_processor stubbed out. Avoids loading any HF weights."""

    obj = _StubKimiK25ImageEmbedding.__new__(_StubKimiK25ImageEmbedding)
    obj.media_token_id = 163605

    # vision_tower is invoked as `self.vision_tower(pixel_values, grid_thws)`
    # in image_embedding and is expected to return a List[Tensor] (one
    # per image) shaped to feed mm_projector_forward, which concatenates
    # along dim 0, runs the projector, then splits per image.
    def _fake_vt_forward(pixel_values, grid_thws):
        n_images = grid_thws.shape[0]
        return [
            torch.full(
                (_TOKENS_PER_IMAGE, 1, _HIDDEN),
                fill_value=float(i + 1),
                dtype=torch.float32,
            )
            for i in range(n_images)
        ]

    obj.vision_tower = _fake_vt_forward  # type: ignore[assignment]

    # mm_projector_forward reshapes to (-1, hidden). Identity projector
    # preserves per-image token counts so we can assert exact shapes.
    class _IdProjector(torch.nn.Module):
        def forward(self, x):
            return x

    obj.mm_projector = _IdProjector()

    obj.image_processor = MagicMock()

    def _fake_preprocess(medias, return_tensors=None):
        n = len(medias)
        return {
            "pixel_values": torch.zeros(n, 3, 14, 14),
            "grid_thws": torch.ones(n, 3, dtype=torch.int64),
        }

    obj.image_processor.preprocess.side_effect = _fake_preprocess
    return obj


class KimiK25ImageEmbeddingShapeTest(TestCase):
    def setUp(self) -> None:
        self.emb = _build_stub_embedding()

    def test_image_embedding_returns_list_per_image(self):
        img = Image.new("RGB", (14, 14))
        out = self.emb.image_embedding([img, img])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for t in out:
            self.assertEqual(t.shape, (_TOKENS_PER_IMAGE, _HIDDEN))

    def test_image_embedding_first_index_is_full_sequence(self):
        # Regression for the original bug where image_embedding returned
        # a single concatenated tensor; callers indexing `[0]` then got
        # one row instead of the per-image (tokens, hidden) block.
        img = Image.new("RGB", (14, 14))
        first = self.emb.image_embedding([img])[0]
        self.assertEqual(first.dim(), 2)
        self.assertEqual(first.shape, (_TOKENS_PER_IMAGE, _HIDDEN))


class KimiK25RendererVideoGuardTest(TestCase):
    def _make_renderer(self) -> KimiK25Renderer:
        renderer = KimiK25Renderer.__new__(KimiK25Renderer)
        return renderer

    def test_video_part_rejected(self):
        renderer = self._make_renderer()
        msg = ChatMessage(
            role=RoleEnum.user,
            content=[
                ContentPart(type=ContentPartTypeEnum.text, text="describe"),
                ContentPart(
                    type=ContentPartTypeEnum.video_url,
                    video_url=ImageURL(url="http://example.com/v.mp4"),
                ),
            ],
        )
        with self.assertRaises(ValueError) as cm:
            renderer._collect_and_rewrite([msg])
        self.assertIn("video", str(cm.exception).lower())

    def test_image_part_accepted(self):
        renderer = self._make_renderer()
        msg = ChatMessage(
            role=RoleEnum.user,
            content=[
                ContentPart(
                    type=ContentPartTypeEnum.image_url,
                    image_url=ImageURL(url="http://example.com/i.png"),
                ),
            ],
        )
        rewritten, mm_input = renderer._collect_and_rewrite([msg])
        self.assertEqual(len(rewritten), 1)
        self.assertEqual(mm_input.urls, ["http://example.com/i.png"])


class KimiK25VisionProcessorVideoGuardTest(TestCase):
    def test_video_media_dict_rejected(self):
        proc = KimiK25VisionProcessor()
        with self.assertRaises(ValueError):
            proc.preprocess({"type": "video", "video": "ignored"})


if __name__ == "__main__":
    main()

import asyncio
import math
import os
import struct
import threading
import zlib
from io import BytesIO
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rtp_llm.openai.api_datatype import ChatCompletionRequest
import rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor as kimi_k3_image_processor
import rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit as kimi_k3_vit
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    K3_MAX_IMAGE_FILE_SIZE_KB,
    K3_MAX_IMAGE_PIXELS,
    KimiK3VisionProcessor,
    _navit_resize_image,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_moonvit import (
    MoonViT3dPretrainedModel,
    MoonVision3dPatchEmbed,
    apply_rope,
    tpool_patch_merger,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
    maybe_fused_apply_rope,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
    KimiK3PatchMergerMLPV2,
    KimiK3VisionConfig,
    mm_projector_forward,
)
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_eagle3 import KimiK3Eagle3Model
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
from rtp_llm.multimodal.multimodal_util import MMUrlType


def _image_bytes(width=8, height=6, mode="RGB", color=(1, 2, 3)):
    data = BytesIO()
    Image.new(mode, (width, height), color).save(data, format="PNG")
    return data.getvalue()


def _png_with_declared_size(width, height):
    """PNG whose IHDR declares a size PIL reads without decoding any pixel."""

    def chunk(tag, body):
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00"))
    )


def _mm_input(tensor, url=""):
    return SimpleNamespace(mm_type=MMUrlType.IMAGE, tensor=tensor, url=url)


def _chat_request(content):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": content}]}
    )


def _media_request(media_type, *urls):
    return _chat_request(
        [{"type": media_type, media_type: {"url": url}} for url in urls]
    )


def _tiny_vision_config(**overrides):
    values = {
        "vt_hidden_size": 4,
        "qkv_hidden_size": 4,
        "vt_intermediate_size": 8,
        "vt_num_attention_heads": 1,
        "vt_num_hidden_layers": 1,
        "init_pos_emb_height": 2,
        "init_pos_emb_width": 2,
    }
    values.update(overrides)
    return KimiK3VisionConfig(**values)


class KimiK3VisionProcessorTest(TestCase):
    def setUp(self):
        self.processor = KimiK3VisionProcessor()

    @staticmethod
    def media(image):
        return {"type": "image", "image": image}

    def test_resize_padding_grid_and_token_count(self):
        media = self.media(Image.new("RGB", (225, 223)))
        result = self.processor.preprocess(media)

        self.assertEqual(self.processor.media_proc_cfg["in_patch_limit"], 65536)
        self.assertEqual(self.processor.media_tokens_calculator(media), 72)
        self.assertTrue(torch.equal(result.grid_thws, torch.tensor([[1, 16, 18]])))
        self.assertEqual(result.pixel_values.shape, (288, 3, 14, 14))
        self.assertEqual(result.pixel_values.dtype, torch.float32)
        self.assertEqual(media["image"].size, (225, 223))

    def test_transparent_image_modes_use_chessboard_after_resize(self):
        rgba = Image.fromarray(np.zeros((28, 28, 4), dtype=np.uint8), "RGBA")
        palette = Image.new("P", (28, 28), 0)
        palette.putpalette([255, 0, 0, 0, 0, 0] + [0] * (256 * 3 - 6))
        palette.info["transparency"] = 0

        for mode, image in (("RGBA", rgba), ("palette", palette)):
            with self.subTest(mode=mode):
                result = self.processor.preprocess(self.media(image))
                patches = result.pixel_values.reshape(2, 2, 3, 14, 14)
                self.assertEqual(float(patches[0, 0, 0, 0, 0]), 1.0)
                self.assertAlmostEqual(
                    float(patches[0, 0, 0, 0, 8]),
                    180 / 127.5 - 1,
                    places=6,
                )

    def test_transparent_bg_fill_stage_changes_composite_order(self):
        rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba[..., 3] = 128  # semi-transparent, so the blend order is observable
        image = Image.fromarray(rgba, "RGBA")

        outputs = {}
        for stage in ("after_resize", "before_resize"):
            processor = KimiK3VisionProcessor(
                {"transparent_bg_fill_stage": stage, "in_patch_limit": 64}
            )
            outputs[stage] = processor.preprocess(self.media(image)).pixel_values

        # in_patch_limit forces a 2x downscale, so compositing before the resize
        # blends the chessboard into the resampled pixels instead of after.
        self.assertFalse(torch.equal(outputs["after_resize"], outputs["before_resize"]))

    def test_invalid_transparent_bg_fill_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            KimiK3VisionProcessor({"transparent_bg_fill_stage": "at_the_end"})

    def test_multiple_images_keep_input_order(self):
        medias = [
            self.media(Image.new("RGB", (28, 28), (0, 0, 0))),
            self.media(Image.new("RGB", (28, 28), (255, 255, 255))),
        ]
        result = self.processor.preprocess(medias)

        self.assertEqual(result.pixel_values.shape, (8, 3, 14, 14))
        self.assertEqual(float(result.pixel_values[0, 0, 0, 0]), -1.0)
        self.assertEqual(float(result.pixel_values[4, 0, 0, 0]), 1.0)

    def test_navit_resize_boundaries(self):
        cases = (
            ((3585, 3585, 65536), (3585, 3585, 27, 27, 16641)),
            ((28, 28, 3), (24, 24, 4, 4, 1)),
            ((100000, 100, 65536), (7167, 7, 1, 21, 256)),
        )
        result_keys = (
            "new_width",
            "new_height",
            "pad_width",
            "pad_height",
            "num_tokens",
        )
        for (width, height, in_patch_limit), expected in cases:
            with self.subTest(width=width, height=height):
                resize = _navit_resize_image(
                    width=width,
                    height=height,
                    patch_size=14,
                    merge_kernel_size=2,
                    in_patch_limit=in_patch_limit,
                    patch_limit_on_one_side=512,
                    fixed_output_tokens=None,
                )
                self.assertEqual(
                    tuple(resize[key] for key in result_keys),
                    expected,
                )

class KimiK3PreprocessInputTest(TestCase):
    def setUp(self):
        self.vit_config = SimpleNamespace(download_headers='{"X-Test": "value"}')

    def test_tensor_bytes_are_decoded(self):
        raw = _image_bytes()
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

        image = KimiK3ImageEmbedding.preprocess_input(
            [_mm_input(tensor)], self.vit_config
        )

        self.assertEqual(image.size, (8, 6))
        self.assertEqual(image.mode, "RGB")

    def test_oversized_pixel_count_is_rejected_before_decode(self):
        # A 54-byte file: under PIL's own 89 MP default, so without the guard
        # image.copy() succeeds and allocates 201 MB of pixels.
        raw = _png_with_declared_size(8192, 8192)
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "pixel count"):
            KimiK3ImageEmbedding.preprocess_input([_mm_input(tensor)], self.vit_config)
        # A stock 48 MP phone photo must stay under the limit.
        self.assertLess(8000 * 6000, K3_MAX_IMAGE_PIXELS)

    def test_oversized_tensor_is_rejected_before_any_copy(self):
        # A direct model RPC call skips the renderer preflight, so this is the only
        # place the shared per-image byte cap can still catch the payload.
        oversized = torch.empty(
            K3_MAX_IMAGE_FILE_SIZE_KB * 1024 + 1, dtype=torch.uint8
        )
        with patch.object(
            kimi_k3_vit, "BytesIO", side_effect=AssertionError("copied before check")
        ) as no_copy:
            with self.assertRaisesRegex(ValueError, "image bytes exceed"):
                KimiK3ImageEmbedding.preprocess_input(
                    [_mm_input(oversized)], self.vit_config
                )
        no_copy.assert_not_called()

    def test_tensor_requires_flat_uint8_bytes(self):
        for tensor in (
            torch.zeros(4, dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.uint8),
        ):
            with self.subTest(dtype=tensor.dtype, shape=tuple(tensor.shape)):
                with self.assertRaisesRegex(ValueError, "1-D uint8"):
                    KimiK3ImageEmbedding.preprocess_input(
                        [_mm_input(tensor)], self.vit_config
                    )

    def test_url_download_uses_configured_headers(self):
        raw = _image_bytes()
        empty = torch.empty(0, dtype=torch.uint8)
        with patch.object(
            kimi_k3_vit,
            "get_bytes_io_from_url",
            return_value=BytesIO(raw),
        ) as download:
            image = KimiK3ImageEmbedding.preprocess_input(
                [_mm_input(empty, "https://example.com/image.png")],
                self.vit_config,
            )

        self.assertEqual(image.size, (8, 6))
        download.assert_called_once_with(
            "https://example.com/image.png",
            self.vit_config.download_headers,
            max_file_size_kb=K3_MAX_IMAGE_FILE_SIZE_KB,
        )

    def test_embedding_holds_mm_lock(self):
        def image_embedding(images):
            self.assertTrue(kimi_k3_vit.mm_lock.locked())
            return [torch.zeros(1)]

        embedding = SimpleNamespace(
            _data_type=torch.float32,
            image_embedding=image_embedding,
        )
        KimiK3ImageEmbedding.embedding(embedding, "image")

    def test_batched_embedding_holds_mm_lock(self):
        def image_embedding(images):
            self.assertTrue(kimi_k3_vit.mm_lock.locked())
            return [torch.zeros(1) for _ in images]

        embedding = SimpleNamespace(
            _data_type=torch.float32,
            image_embedding=image_embedding,
        )
        results = KimiK3ImageEmbedding.batched_embedding(
            embedding, ["image_a", "image_b"], [MMUrlType.IMAGE, MMUrlType.IMAGE]
        )
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0][1])

    def test_image_embedding_keeps_grid_metadata_on_cpu(self):
        grid_thws = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        captured = {}

        class Processor:
            @staticmethod
            def preprocess(_medias, return_tensors):
                self.assertEqual(return_tensors, "pt")
                return {
                    "pixel_values": torch.ones(4, 3, 2, 2),
                    "grid_thws": grid_thws,
                }

        def vision_tower(pixel_values, grid_metadata):
            captured["pixel_values"] = pixel_values
            captured["grid_thws"] = grid_metadata
            return [torch.ones(1, 4, 2, dtype=pixel_values.dtype)]

        embedding = SimpleNamespace(
            image_processor=Processor(),
            _device=torch.device("cpu"),
            _data_type=torch.float16,
            vision_tower=vision_tower,
            mm_projector=nn.Identity(),
        )
        output = KimiK3ImageEmbedding.image_embedding(embedding, [object()])

        self.assertEqual(captured["pixel_values"].dtype, torch.float16)
        self.assertIs(captured["grid_thws"], grid_thws)
        self.assertEqual(captured["grid_thws"].device.type, "cpu")
        self.assertEqual(output[0].dtype, torch.float16)


class KimiK3MultimodalEmbeddingTest(TestCase):
    HASH_IDS = (2140422385, -1781747402)

    def setUp(self):
        self.model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(self.model)
        self.model.embedding_weight = torch.arange(
            24, dtype=torch.float32
        ).reshape(8, 3)
        self.model.embed_tokens = EmbeddingTorch(self.model.embedding_weight)
        self.model.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def _embed(self, input_ids, features, location):
        features = torch.tensor(features, dtype=torch.float32)
        multimodal_inputs = SimpleNamespace(
            multimodal_features=[features],
            mm_features_locs_host=torch.tensor([location], dtype=torch.int32),
        )
        actual = self.model._embed(
            torch.tensor(input_ids, dtype=torch.int32), multimodal_inputs
        )
        return actual, features

    def test_hash_token_ids_are_masked_before_visual_features_are_injected(self):
        actual, features = self._embed(
            [5, *self.HASH_IDS, 7],
            [[101, 102, 103], [201, 202, 203]],
            1,
        )
        expected = torch.cat(
            (
                self.model.embedding_weight[5:6],
                features,
                self.model.embedding_weight[7:8],
            )
        )
        torch.testing.assert_close(actual, expected)

    def test_negative_location_skips_visual_features_in_reused_prefix(self):
        actual, features = self._embed(
            [*self.HASH_IDS, 7],
            [[101, 102, 103], [201, 202, 203], [301, 302, 303]],
            -1,
        )
        expected = torch.stack(
            (
                features[1],
                features[2],
                self.model.embedding_weight[7],
            )
        )
        torch.testing.assert_close(actual, expected)


class KimiK3Eagle3MultimodalEmbeddingTest(TestCase):
    # Feature values stay well outside the embedding table's range (0..191) so a
    # misplaced row can never compare equal to a text embedding.
    IMAGE_A = torch.tensor([[1000.0, 1001.0, 1002.0], [2000.0, 2001.0, 2002.0]])
    IMAGE_B = torch.tensor([[3000.0, 3001.0, 3002.0], [4000.0, 4001.0, 4002.0]])

    def setUp(self):
        self.embedding_weight = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)
        self.model = KimiK3Eagle3Model.__new__(KimiK3Eagle3Model)
        nn.Module.__init__(self.model)
        self.model.embedding = EmbeddingTorch(self.embedding_weight)
        self.model.multimodal_embedding_injector = MultimodalEmbeddingInjector()

    def _embed(self, input_ids, features, locations, cu_seqlens):
        boundaries = torch.tensor(cu_seqlens, dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=torch.tensor(input_ids, dtype=torch.int32),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features,
                mm_features_locs_host=torch.tensor(locations, dtype=torch.int32),
            ),
            attention_inputs=SimpleNamespace(
                cu_seqlens=boundaries, cu_seqlens_host=boundaries
            ),
        )
        return self.model._embed_shifted_multimodal(inputs)

    def test_draft_prefill_shifts_and_injects_image_features(self):
        # Target [10, hash_0, hash_1, 11] shifts to [hash_0, hash_1, 11, sampled_12],
        # while mm_features_locs still describes the target position (1).
        output = self._embed(
            [-101, -102, 11, 12], [self.IMAGE_A], [1], cu_seqlens=[0, 4]
        )

        self.assertTrue(torch.equal(output[0:2], self.IMAGE_A))
        self.assertTrue(torch.equal(output[2], self.embedding_weight[11]))
        self.assertTrue(torch.equal(output[3], self.embedding_weight[12]))

    def test_draft_prefill_drops_the_row_shifted_out_of_the_window(self):
        # Target [hash_0, hash_1, 11] starts with the image, so the shift pushes
        # row 0 before token 0: it has no draft slot and must be dropped.
        output = self._embed(
            [-102, 11, 12, 13], [self.IMAGE_A], [0], cu_seqlens=[0, 4]
        )

        self.assertTrue(torch.equal(output[0], self.IMAGE_A[1]))
        self.assertTrue(torch.equal(output[1], self.embedding_weight[11]))
        self.assertTrue(torch.equal(output[2], self.embedding_weight[12]))
        self.assertTrue(torch.equal(output[3], self.embedding_weight[13]))

    def test_draft_prefill_clamps_features_to_their_own_request(self):
        # Two 3-token requests; B's image sits at its own first token (loc 3 ==
        # request start), where a global "loc - 1" would land in A's last slot.
        output = self._embed(
            [-101, -102, 20, -202, 22, 21],
            [self.IMAGE_A, self.IMAGE_B],
            [1, 3],
            cu_seqlens=[0, 3, 6],
        )

        self.assertTrue(torch.equal(output[0:2], self.IMAGE_A))
        # Request A's sampled token must survive untouched.
        self.assertTrue(torch.equal(output[2], self.embedding_weight[20]))
        # Request B keeps its own start: row 0 is shifted out, row 1 lands there.
        self.assertTrue(torch.equal(output[3], self.IMAGE_B[1]))
        self.assertTrue(torch.equal(output[4], self.embedding_weight[22]))
        self.assertTrue(torch.equal(output[5], self.embedding_weight[21]))


class KimiK3MediaPreflightTest(TestCase):
    def test_preflight_rejects_image_when_full_decode_fails(self):
        data = BytesIO()
        Image.new("RGB", (64, 64), "red").save(
            data, format="TIFF", compression="tiff_lzw"
        )
        raw = bytearray(data.getvalue())
        # The header remains readable, but the LZW pixel stream is invalid.
        raw[16] ^= 0xFF

        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            return_value=BytesIO(raw),
        ):
            with self.assertRaisesRegex(ValueError, "could not be decoded"):
                kimi_k3_image_processor.preflight_kimi_k3_images(["image"])

    def test_preflight_rejects_oversized_pixel_count(self):
        raw = _image_bytes()
        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            return_value=BytesIO(raw),
        ), patch.object(
            kimi_k3_image_processor,
            "K3_MAX_IMAGE_PIXELS",
            1,
        ):
            with self.assertRaisesRegex(ValueError, "pixel count"):
                kimi_k3_image_processor.preflight_kimi_k3_images(["image"])

    def test_shared_preflight_reuses_downloaded_bytes_and_reports_sizes(self):
        raw = _image_bytes(
            width=31,
            height=17,
            mode="RGBA",
            color=(255, 0, 0, 0),
        )
        url = "http://example.com/transparent.png"
        download_headers = '{"Authorization": "Bearer test"}'

        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            return_value=BytesIO(raw),
        ) as download:
            tensors, sizes = kimi_k3_image_processor.preflight_kimi_k3_images(
                [url], download_headers
            )

        self.assertEqual(sizes, [(31, 17)])
        self.assertEqual(tensors[0].cpu().numpy().tobytes(), raw)
        download.assert_called_once_with(
            url,
            download_headers,
            max_file_size_kb=K3_MAX_IMAGE_FILE_SIZE_KB,
        )

    def test_preflight_does_not_read_or_seek_shared_cached_stream(self):
        raw = _image_bytes(width=19, height=13)
        url = "http://example.com/cached.png"

        class SharedCachedBytesIO(BytesIO):
            def read(self, *args, **kwargs):
                raise AssertionError("shared cache stream must not be read")

            def seek(self, *args, **kwargs):
                raise AssertionError("shared cache stream must not be seeked")

        shared = SharedCachedBytesIO(raw)
        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            return_value=shared,
        ) as download:
            tensors, sizes = kimi_k3_image_processor.preflight_kimi_k3_images(
                [url, url]
            )

        self.assertEqual(sizes, [(19, 13), (19, 13)])
        self.assertEqual(
            [tensor.cpu().numpy().tobytes() for tensor in tensors], [raw, raw]
        )
        self.assertEqual(download.call_count, 2)


class KimiK3RendererTest(TestCase):
    def setUp(self):
        self.renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        self.renderer.vit_config = SimpleNamespace(
            download_headers='{"Authorization": "Bearer test"}',
        )
        self.renderer.max_seq_len = 0
        self.renderer._image_processor = KimiK3VisionProcessor()

    def test_video_url_is_rejected_as_k3(self):
        request = _media_request("video_url", "http://example.com/video.mp4")

        with self.assertRaisesRegex(ValueError, "only text and image_url.*video_url"):
            self.renderer.render_chat(request)

    def test_tool_result_image_is_rejected(self):
        message = {
            "role": "tool",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.png"},
                }
            ],
        }

        with self.assertRaisesRegex(
            ValueError, "image_url content only in user messages.*'tool'"
        ):
            self.renderer._collect_and_rewrite([message])

    def test_render_chat_serializes_messages_and_renders_template_once(self):
        request = _chat_request("hello")
        template_calls = []

        def apply_chat_template(messages, **kwargs):
            template_calls.append((messages, kwargs))
            return [101, 102] if kwargs["tokenize"] else "rendered prompt"

        self.renderer.tokenizer = SimpleNamespace(
            apply_chat_template=apply_chat_template
        )
        rendered = self.renderer.render_chat(request)

        self.assertEqual(rendered.input_ids, [101, 102])
        self.assertEqual(rendered.multimodal_inputs, [])
        # The prompt string is only needed for debug output, where the endpoints
        # decode input_ids instead; rendering the template twice would be wasteful.
        self.assertEqual(rendered.rendered_prompt, "")
        self.assertEqual(
            [kwargs["tokenize"] for _, kwargs in template_calls],
            [True],
        )
        for messages, kwargs in template_calls:
            self.assertIsInstance(messages[0], dict)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[0]["content"], "hello")
            self.assertTrue(kwargs["add_generation_prompt"])
            self.assertEqual(kwargs["image_prompts"], [])

    def test_preflight_reuses_bytes_and_injects_real_size_prompt(self):
        raw = _image_bytes(
            width=31,
            height=17,
            mode="RGBA",
            color=(255, 0, 0, 0),
        )

        class Tokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs):
                prompt = f"before {kwargs['image_prompts'][0]} after"
                return list(prompt.encode()) if kwargs["tokenize"] else prompt

        self.renderer.tokenizer = Tokenizer()
        request = _media_request("image_url", "http://example.com/transparent.png")

        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            return_value=BytesIO(raw),
        ) as download:
            rendered = self.renderer.render_chat(request)

        download.assert_called_once_with(
            "http://example.com/transparent.png",
            self.renderer.vit_config.download_headers,
            max_file_size_kb=K3_MAX_IMAGE_FILE_SIZE_KB,
        )
        # This tokenizer stub encodes the template output byte-by-byte, so the token
        # ids carry the prompt the real image size was injected into.
        self.assertEqual(
            bytes(rendered.input_ids).decode(),
            "before <|media_begin|>image 31x17<|media_content|>"
            "<|media_pad|><|media_end|> after",
        )
        self.assertEqual(
            rendered.multimodal_inputs[0].tensor.cpu().numpy().tobytes(), raw
        )

    def test_async_preflight_is_concurrent_and_off_event_loop(self):
        barrier = threading.Barrier(2)
        main_thread = threading.get_ident()
        worker_threads = []

        def preflight(url, download_headers):
            worker_threads.append(threading.get_ident())
            barrier.wait(timeout=2)
            return torch.tensor([len(url)], dtype=torch.uint8), (8, 6)

        self.renderer.tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: (
                [1, 2] if kwargs["tokenize"] else "rendered"
            )
        )
        request = _media_request(
            "image_url",
            "http://example.com/0.png",
            "http://example.com/1.png",
        )

        with patch.object(
            kimi_k3_image_processor,
            "_preflight_kimi_k3_image",
            side_effect=preflight,
        ):
            rendered = asyncio.run(self.renderer.render_chat_async(request))

        self.assertEqual(len(rendered.multimodal_inputs), 2)
        self.assertEqual(len(worker_threads), 2)
        self.assertTrue(all(thread != main_thread for thread in worker_threads))

    def test_preflight_rejects_image_count_before_download(self):
        with patch.object(
            kimi_k3_image_processor, "K3_MAX_IMAGES_PER_REQUEST", 1
        ), patch.object(
            kimi_k3_image_processor, "_preflight_kimi_k3_image"
        ) as preflight:
            with self.assertRaisesRegex(ValueError, "image count"):
                kimi_k3_image_processor.preflight_kimi_k3_images(
                    ["first", "second"]
                )
        preflight.assert_not_called()

    def test_preflight_rejects_total_image_bytes(self):
        with patch.object(
            kimi_k3_image_processor, "K3_MAX_TOTAL_IMAGE_BYTES", 1
        ), patch.object(
            kimi_k3_image_processor,
            "_preflight_kimi_k3_image",
            return_value=(torch.zeros(2, dtype=torch.uint8), (8, 6)),
        ):
            with self.assertRaisesRegex(ValueError, "image bytes"):
                kimi_k3_image_processor.preflight_kimi_k3_images(["image"])

    def test_render_rejects_expanded_visual_tokens_over_context(self):
        self.renderer.max_seq_len = 1
        with self.assertRaisesRegex(ValueError, "expanded multimodal input"):
            self.renderer._validate_visual_token_budget([1], [(56, 56)])


class KimiK3VisionConfigWiringTest(TestCase):
    def test_load_vision_config_propagates_multimodal_fields(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3

        checkpoint_config = {
            "vision_config": {
                "merge_kernel_size": 2,
                "vt_hidden_size": 1024,
                "_name_or_path": "ignored",
            },
            "media_placeholder_token_id": 42,
        }
        config = ModelConfig()
        KimiK3._load_vision_config(config, checkpoint_config)

        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[42]])
        self.assertEqual(
            config.mm_related_params.special_token_ids["image_token_index"], 42
        )
        self.assertEqual(
            config.mm_related_params.config["vision_config"],
            {"merge_kernel_size": 2, "vt_hidden_size": 1024},
        )

    def test_checkpoint_without_vision_config_stays_text_only(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3

        config = ModelConfig()
        KimiK3._load_vision_config(config, {"media_placeholder_token_id": 42})

        # Claiming multimodal would build a MoonViT from defaults and only fail
        # later in weight loading, with no vision_tower.* tensors in the ckpt.
        self.assertFalse(config.mm_model_config.is_multimodal)

    def test_vision_config_without_media_token_id_is_rejected(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3

        with self.assertRaisesRegex(ValueError, "media_placeholder_token_id"):
            KimiK3._load_vision_config(
                ModelConfig(), {"vision_config": {"vt_hidden_size": 1024}}
            )


class KimiK3MoonViTTest(TestCase):
    @staticmethod
    def _reference_block(block, hidden, rope):
        normed = F.rms_norm(
            hidden, (hidden.shape[-1],), block.norm0.weight, eps=block.norm0.eps
        )
        qkv = F.linear(normed, block.wqkv.weight).view(
            hidden.shape[0], 3, block.num_heads, block.head_dim
        )
        query, key, value = torch.unbind(qkv, dim=1)

        rope = rope.unsqueeze(1)
        query = torch.view_as_real(
            torch.view_as_complex(query.float().view(*query.shape[:-1], -1, 2))
            * rope
        ).flatten(-2)
        key = torch.view_as_real(
            torch.view_as_complex(key.float().view(*key.shape[:-1], -1, 2))
            * rope
        ).flatten(-2)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            block.head_dim
        )
        attention = torch.matmul(torch.softmax(scores, dim=-1), value)
        attention = attention.transpose(0, 1).reshape(hidden.shape[0], -1)
        hidden = hidden + F.linear(attention, block.wo.weight)

        normed = F.rms_norm(
            hidden, (hidden.shape[-1],), block.norm1.weight, eps=block.norm1.eps
        )
        mlp = F.linear(normed, block.mlp.fc0.weight)
        mlp = F.gelu(mlp, approximate="tanh")
        mlp = F.linear(mlp, block.mlp.fc1.weight)
        return hidden + mlp

    def test_default_config_matches_checkpoint(self):
        config = KimiK3VisionConfig()

        self.assertEqual(
            KimiK3VisionConfig(merge_kernel_size=2).merge_kernel_size, [2, 2]
        )
        self.assertEqual(config.vt_hidden_size, 1024)
        self.assertEqual(config.qkv_hidden_size, 1536)
        self.assertEqual(config.vt_intermediate_size, 4096)
        self.assertEqual(config.vt_num_attention_heads, 12)
        self.assertEqual(config.vt_num_hidden_layers, 27)
        self.assertEqual(config.pos_emb_interpolation_mode, "bilinear")
        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.mlp_type, "mlp2")
        self.assertEqual(config.mm_projector_type, "patchmergerv2")
        self.assertFalse(config.attn_bias)
        self.assertFalse(config.linear_bias)
        self.assertFalse(config.patch_embed_proj_bias)

    def test_position_embedding_interpolation_ignores_config_mode(self):
        # Upstream declares pos_emb_interpolation_mode (the ckpt says "bilinear")
        # but never wires it into the module, so it always interpolates bicubic.
        patch_embed = MoonVision3dPatchEmbed(
            _tiny_vision_config(pos_emb_interpolation_mode="nearest")
        )
        with patch.object(F, "interpolate", wraps=F.interpolate) as interpolate:
            patch_embed.pos_emb._interp(3, 3)

        self.assertEqual(patch_embed.pos_emb.interpolation_mode, "bicubic")
        self.assertEqual(interpolate.call_args.kwargs["mode"], "bicubic")

    def test_tiny_model_structure_and_packed_forward(self):
        config = _tiny_vision_config(
            vt_hidden_size=64,
            qkv_hidden_size=96,
            vt_intermediate_size=128,
            vt_num_attention_heads=3,
        )
        model = MoonViT3dPretrainedModel(config)
        block = model.encoder.blocks[0]

        self.assertIsInstance(block.norm0, nn.RMSNorm)
        self.assertIsInstance(block.norm1, nn.RMSNorm)
        self.assertIsInstance(model.encoder.final_layernorm, nn.RMSNorm)
        self.assertIsNone(block.norm0.eps)
        self.assertIsNone(block.norm1.eps)
        self.assertIsNone(model.encoder.final_layernorm.eps)
        self.assertEqual(block.wqkv.weight.shape, (288, 64))
        self.assertEqual(block.wo.weight.shape, (64, 96))
        self.assertIsNone(block.wqkv.bias)
        self.assertIsNone(block.wo.bias)
        self.assertIsNone(block.mlp.fc0.bias)
        self.assertIsNone(block.mlp.fc1.bias)
        self.assertIsNone(model.patch_embed.proj.bias)

        first_image = torch.linspace(-1.0, 1.0, 4 * 3 * 14 * 14).reshape(
            4, 3, 14, 14
        )
        second_image = torch.linspace(1.0, -0.5, 8 * 3 * 14 * 14).reshape(
            8, 3, 14, 14
        )
        first_grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        second_grid = torch.tensor([[1, 2, 4]], dtype=torch.int64)
        output = model(first_image, first_grid)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, (1, 4, 64))

        packed_output = model(
            torch.cat([first_image, second_image], dim=0),
            torch.cat([first_grid, second_grid]),
        )
        self.assertEqual(
            [item.shape for item in packed_output], [(1, 4, 64), (2, 4, 64)]
        )
        second_output = model(second_image, second_grid)
        torch.testing.assert_close(packed_output[0], output[0])
        torch.testing.assert_close(packed_output[1], second_output[0])

    def test_patch_and_position_embedding_numerics(self):
        config = _tiny_vision_config(
            patch_size=2,
            num_channels=1,
            init_pos_emb_time=1,
        )
        patch_embed = MoonVision3dPatchEmbed(config)
        position_weight = (
            torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) / 10
        )
        with torch.no_grad():
            patch_embed.proj.weight.fill_(1)
            patch_embed.pos_emb.weight.copy_(position_weight)
            patch_embed.pos_emb.time_weight.zero_()

        pixel_values = torch.stack(
            [torch.full((1, 2, 2), float(value)) for value in range(1, 5)]
        )
        output = patch_embed(
            pixel_values, torch.tensor([[1, 2, 2]], dtype=torch.int64)
        )
        expected = (
            torch.arange(4, 17, 4, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, 4)
            + position_weight.flatten(end_dim=1)
        )
        torch.testing.assert_close(output, expected)

    def test_block_numerics(self):
        block = MoonViT3dPretrainedModel(_tiny_vision_config()).encoder.blocks[0]
        with torch.no_grad():
            block.norm0.weight.fill_(1.0)
            block.norm1.weight.fill_(0.9)
            for parameter in (
                block.wqkv.weight,
                block.wo.weight,
                block.mlp.fc0.weight,
                block.mlp.fc1.weight,
            ):
                parameter.copy_(
                    torch.linspace(-0.2, 0.2, parameter.numel()).reshape_as(parameter)
                )

        hidden = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, -1.0, 2.0, -2.0],
                [2.0, -2.0, 1.0, -1.0],
            ]
        )
        rope_angles = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        rope = torch.polar(torch.ones_like(rope_angles), rope_angles)
        cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

        expected = self._reference_block(block, hidden, rope)
        actual = block(hidden, cu_seqlens, rope, 4)
        torch.testing.assert_close(actual, expected)

    @skipUnless(torch.cuda.is_available(), "CUDA is required for fused RoPE")
    def test_fused_rope_matches_eager_with_joint_output_buffer(self):
        seq_len, num_heads, head_dim = 17, 3, 128
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                qkv = torch.randn(
                    seq_len,
                    3,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                )
                qkv_before = qkv.clone()
                query, key, _ = torch.unbind(qkv, dim=1)
                angles = torch.randn(
                    seq_len,
                    head_dim // 2,
                    device="cuda",
                    dtype=torch.float32,
                )
                freqs = torch.polar(torch.ones_like(angles), angles)

                with patch.dict(os.environ, {"KIMI_K3_FUSED_ROPE": "0"}):
                    expected_query, expected_key = apply_rope(query, key, freqs)
                    self.assertIsNone(maybe_fused_apply_rope(query, key, freqs))
                with patch.dict(os.environ, {"KIMI_K3_FUSED_ROPE": "1"}):
                    actual_query, actual_key = apply_rope(query, key, freqs)

                rtol, atol = (
                    (1e-3, 3e-3) if dtype == torch.float16 else (1e-2, 3e-2)
                )
                torch.testing.assert_close(
                    actual_query, expected_query, rtol=rtol, atol=atol
                )
                torch.testing.assert_close(
                    actual_key, expected_key, rtol=rtol, atol=atol
                )
                self.assertTrue(actual_query.is_contiguous())
                self.assertTrue(actual_key.is_contiguous())
                self.assertEqual(
                    actual_query.untyped_storage().data_ptr(),
                    actual_key.untyped_storage().data_ptr(),
                )
                torch.testing.assert_close(qkv, qkv_before, rtol=0, atol=0)

    def test_temporal_and_spatial_merge_numerics(self):
        output = tpool_patch_merger(
            torch.arange(16, dtype=torch.float32).view(16, 1),
            torch.tensor([[2, 2, 4]], dtype=torch.int64),
            merge_kernel_size=(2, 2),
        )

        self.assertEqual(len(output), 1)
        torch.testing.assert_close(
            output[0],
            torch.tensor(
                [
                    [[4.0], [5.0], [8.0], [9.0]],
                    [[6.0], [7.0], [10.0], [11.0]],
                ]
            ),
        )


class KimiK3PatchMergerMLPV2Test(TestCase):
    def setUp(self):
        self.config = KimiK3VisionConfig(
            vt_hidden_size=8,
            mm_hidden_size=8,
            text_hidden_size=16,
        )
        self.projector = KimiK3PatchMergerMLPV2(self.config)

    def test_structure_and_checkpoint_names(self):
        self.assertEqual(self.projector.proj[0].weight.shape, (32, 32))
        self.assertEqual(self.projector.proj[2].weight.shape, (16, 32))
        self.assertEqual(self.projector.post_norm.weight.shape, (16,))
        self.assertEqual(
            set(self.projector.state_dict()),
            {
                "proj.0.weight",
                "proj.2.weight",
                "post_norm.weight",
            },
        )

    def test_mm_projector_forward_preserves_image_boundaries(self):
        vision_outputs = [
            torch.arange(64, dtype=torch.float32).reshape(2, 4, 8) / 64,
            torch.arange(96, dtype=torch.float32).reshape(3, 4, 8) / 96 + 1,
        ]
        output = mm_projector_forward(self.projector, vision_outputs)
        expected = [self.projector(features) for features in vision_outputs]

        self.assertEqual([item.shape for item in output], [(2, 16), (3, 16)])
        for actual, projected in zip(output, expected):
            torch.testing.assert_close(actual, projected)

    def test_mm_projector_forward_single_image_skips_cat(self):
        image_features = torch.randn(2, 4, 8)
        expected = self.projector(image_features)
        with patch.object(torch, "cat", wraps=torch.cat) as cat:
            output = mm_projector_forward(self.projector, [image_features])

        cat.assert_not_called()
        self.assertEqual(len(output), 1)
        torch.testing.assert_close(output[0], expected)

    def test_projector_numerics(self):
        with torch.no_grad():
            self.projector.proj[0].weight.copy_(torch.eye(32))
            self.projector.proj[2].weight.copy_(torch.eye(32)[:16])
            self.projector.post_norm.weight.fill_(1)

        flattened = torch.arange(32, dtype=torch.float32).view(1, 32) / 32
        output = self.projector(flattened.view(1, 4, 8))
        expected = F.rms_norm(
            F.gelu(flattened)[:, :16],
            (16,),
            self.projector.post_norm.weight,
            eps=self.config.projector_ln_eps,
        )
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    main()

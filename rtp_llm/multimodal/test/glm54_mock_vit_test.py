import asyncio
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.glm54_mock_vit_input import GLM54_MOCK_IMAGE_TOKEN
from rtp_llm.multimodal.media_network import (
    MediaNetwork,
    UnsupportedMediaNetworkPolicy,
    read_media_source,
)
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.glm54_mock_vit import (
    Glm54MockImageEmbedding,
    Glm54MockVitMixin,
)
from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.glm5_3_flash_mixin import (
    Glm53FlashVisionModel,
    glm5_smart_resize,
)
from rtp_llm.multimodal.multimodal_util import (
    _default_request_get,
    collect_download_timing,
    get_bytes_io_from_url,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.custom_renderer import ThinkStatus
from rtp_llm.openai.renderers.glm54_mock_vit_renderer import Glm54MockVitRenderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolStreamStatus,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters


def vision_config():
    return SimpleNamespace(
        attention_bias=True,
        depth=2,
        hidden_size=32,
        in_channels=3,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        projection_intermediate_size=32,
        rms_norm_eps=1e-6,
        spatial_merge_size=2,
        swiglu_limit=7.0,
        temporal_patch_size=2,
    )


def image_part(url, **preprocess_config):
    return {
        "type": "image_url",
        "image_url": {"url": url},
        "preprocess_config": preprocess_config,
    }


def chat_request(*parts):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": list(parts)}]}
    )


class Glm54MockVitTest(unittest.TestCase):
    def setUp(self):
        self.renderer = object.__new__(Glm54MockVitRenderer)
        self.renderer.chat_template = (
            "{% for message in messages %}{% for part in message.content %}"
            "{{ part.text }}{% endfor %}{% endfor %}"
        )
        self.renderer.tokenizer = Mock()
        self.renderer.tokenizer.encode.return_value = [8, 9]

    def test_native_http_loader_bounds_and_partitions_cached_responses(self):
        requests = []
        paths = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                source = self.headers.get("X-Test-Source", "unknown")
                requests.append(source)
                paths.append(self.path)
                payload = source.encode() * 600
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        config = VitConfig()
        config.mm_preprocess_max_workers = 2
        config.url_cache_item_num = 8
        # Loopback stands in for a public image origin in this hermetic test.
        with patch.object(MediaNetwork, "_configure_internal_policy", return_value=None):
            network = MediaNetwork(config)
        self.addCleanup(network.close)
        url = f"http://127.0.0.1:{server.server_port}/image with space.jpg"
        red = '{"X-Test-Source":"red"}'
        blue = '{"X-Test-Source":"blue"}'

        with (
            network.use(time.monotonic() + 5),
            patch("rtp_llm.multimodal.media_network.getproxies", return_value={}),
        ):
            self.assertEqual(read_media_source(url, red, 8).read(), b"red" * 600)
            with collect_download_timing() as timing:
                self.assertEqual(read_media_source(url, red, 8).read(), b"red" * 600)
            self.assertEqual(timing.elapsed_ms, 0)
            self.assertEqual(read_media_source(url, blue, 8).read(), b"blue" * 600)
            with self.assertRaisesRegex(FtRuntimeException, MMErr.FILE_TOO_LARGE):
                read_media_source(url, red, 1)
        self.assertEqual(requests, ["red", "blue"])
        self.assertEqual(paths, ["/image%20with%20space.jpg"] * 2)

        with (
            network.use(time.monotonic() - 1),
            patch("rtp_llm.multimodal.media_network.getproxies", return_value={}),
        ):
            with self.assertRaisesRegex(FtRuntimeException, MMErr.DL_TIMEOUT):
                read_media_source(url, red, 8)

        with (
            network.use(time.monotonic() + 5),
            patch(
                "rtp_llm.multimodal.media_network.getproxies",
                return_value={"https": "http://proxy.example.test"},
            ),
            patch(
                "rtp_llm.multimodal.media_network.get_bytes_io_from_url",
                return_value=BytesIO(b"proxied"),
            ) as legacy,
        ):
            self.assertEqual(read_media_source(url, red, 8).read(), b"proxied")
        legacy.assert_called_once()
        self.assertEqual(requests, ["red", "blue"])

    def test_legacy_http_loader_checks_denylist_on_redirects(self):
        paths = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                paths.append(self.path)
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header(
                        "Location", f"http://localhost:{self.server.server_port}/blocked"
                    )
                    self.send_header("Content-Length", "0")
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        backend_url = f"http://127.0.0.1:{server.server_port}"
        legacy_url = f"http://media.example.test:{server.server_port}"
        ssrf_check = SimpleNamespace(
            safe_request_get=Mock(),
            rewrite_oss_url=Mock(
                side_effect=lambda url: url.replace(legacy_url, backend_url)
            ),
            check_ssrf=Mock(return_value=True),
        )
        with patch.dict(os.environ, {"MM_DOWNLOAD_DOMAIN_DENYLIST": '["localhost"]'}):
            with patch(
                "rtp_llm.multimodal.multimodal_util.REQUEST_GET", _default_request_get
            ):
                with self.assertRaisesRegex(
                    FtRuntimeException, "download domain is blocked"
                ):
                    get_bytes_io_from_url(f"{backend_url}/redirect")
            with (
                patch("rtp_llm.multimodal.multimodal_util.REQUEST_GET", None),
                patch(
                    "rtp_llm.multimodal.multimodal_util.import_module",
                    return_value=ssrf_check,
                ),
            ):
                self.assertEqual(
                    get_bytes_io_from_url(f"{legacy_url}/allowed").read(), b""
                )
                with self.assertRaisesRegex(
                    FtRuntimeException, "download domain is blocked"
                ):
                    get_bytes_io_from_url(f"{legacy_url}/redirect")
        self.assertEqual(paths, ["/redirect", "/allowed", "/redirect"])
        ssrf_check.safe_request_get.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in ssrf_check.check_ssrf.call_args_list],
            [f"{backend_url}/allowed", f"{backend_url}/redirect"],
        )

    def test_native_http_loader_declines_legacy_private_policy(self):
        policy = Mock()
        with patch(
            "rtp_llm.multimodal.media_network.import_module",
            return_value=SimpleNamespace(),
        ):
            with self.assertRaises(UnsupportedMediaNetworkPolicy):
                MediaNetwork._configure_internal_policy(policy)
        policy.set_networks.assert_not_called()

    def test_mock_preprocess_passes_image_size_limit_to_transport(self):
        config = VitConfig()
        config.mm_image_max_file_size_kb = 3
        url = "https://example.test/image.jpg"
        image = MultimodalInput(url, MMUrlType.IMAGE, torch.empty(0), MMPreprocessConfig())
        processor_config = {
            "image_processor": {
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "min_image_tokens": 1,
                "max_image_tokens": 100,
            }
        }
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.glm5_3_flash_mixin.read_media_source",
            side_effect=ValueError("download finished"),
        ) as download:
            with self.assertRaisesRegex(ValueError, "download finished"):
                Glm54MockImageEmbedding.preprocess_input(
                    [image], config, None, None, processor_config
                )
        download.assert_called_once_with(url, config.download_headers, 3)

    def test_renderer_preserves_image_order_and_preprocess_options(self):
        first_url = "https://example.test/first.png"
        second_url = "https://example.test/second.png"
        request = chat_request(
            image_part(first_url, min_pixels=64, max_pixels=256, mm_timeout_ms=250),
            {"type": "text", "text": " between "},
            image_part(second_url),
        )
        original_request = request.model_dump()

        result = self.renderer.render_chat(request)

        self.assertEqual(result.input_ids, [8, 9])
        self.assertEqual(request.model_dump(), original_request)
        self.assertEqual(
            result.rendered_prompt,
            f"{GLM54_MOCK_IMAGE_TOKEN}\n between {GLM54_MOCK_IMAGE_TOKEN}\n",
        )
        self.renderer.tokenizer.encode.assert_called_once_with(result.rendered_prompt)
        self.assertEqual(
            [item.url for item in result.multimodal_inputs], [first_url, second_url]
        )
        self.assertTrue(
            all(item.mm_type == MMUrlType.IMAGE for item in result.multimodal_inputs)
        )
        first_config = result.multimodal_inputs[0].mm_preprocess_config
        self.assertEqual(
            (
                first_config.min_pixels,
                first_config.max_pixels,
                first_config.mm_timeout_ms,
            ),
            (64, 256, 250),
        )
        second_config = result.multimodal_inputs[1].mm_preprocess_config
        self.assertEqual(second_config.min_pixels, -1)

    def test_renderer_rejects_invalid_media_before_tokenizing(self):
        url = "https://example.test/image.png"
        invalid_parts = (
            {"type": "image_url"},
            image_part("  "),
            {"type": "video_url"},
            {"type": "audio_url"},
            {"type": "igraph"},
            {"type": "text"},
            image_part(url, resized_height=32),
            image_part(url, crop_positions="invalid"),
            image_part(url, max_long_side_pixel=128),
            image_part(url, fps=2),
            image_part(url, min_frames=1),
            image_part(url, min_pixels=0),
            image_part(url, min_pixels=256, max_pixels=128),
            image_part(url, max_pixels=2**31),
        )
        for invalid_part in invalid_parts:
            with self.subTest(part=invalid_part):
                with self.assertRaises(FtRuntimeException) as failure:
                    self.renderer.render_chat(chat_request(invalid_part))
                self.assertEqual(
                    failure.exception.exception_type,
                    ExceptionType.MM_WRONG_FORMAT_ERROR,
                )
        self.renderer.tokenizer.encode.assert_not_called()

    def test_renderer_rejects_an_image_missing_from_the_rendered_prompt(self):
        self.renderer.chat_template = "prompt with no image marker"
        request = chat_request(image_part("https://example.test/a"))
        with self.assertRaises(FtRuntimeException) as failure:
            self.renderer.render_chat(request)
        self.assertEqual(
            failure.exception.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR
        )

    def test_renderer_matches_images_to_reordered_tool_results(self):
        self.renderer.chat_template = (
            "{% for call in messages[0].tool_calls %}"
            "{% for message in messages[1:] if message.tool_call_id == call.id %}"
            "{{ message.tool_call_id }}:{% for part in message.content %}"
            "{{ part.text }}{% endfor %}{% endfor %}{% endfor %}"
        )
        request = ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "image", "arguments": "{}"},
                            }
                            for call_id in ("A", "B")
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "B",
                        "content": [image_part("https://example.test/b", min_pixels=64)],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "A",
                        "content": [image_part("https://example.test/a", min_pixels=128)],
                    },
                ]
            }
        )

        result = self.renderer.render_chat(request)

        self.assertEqual(
            result.rendered_prompt,
            f"A:{GLM54_MOCK_IMAGE_TOKEN}\nB:{GLM54_MOCK_IMAGE_TOKEN}\n",
        )
        self.assertEqual(
            [item.url for item in result.multimodal_inputs],
            ["https://example.test/a", "https://example.test/b"],
        )
        self.assertEqual(
            [item.mm_preprocess_config.min_pixels for item in result.multimodal_inputs],
            [128, 64],
        )

    def test_renderer_rejects_duplicate_and_reserved_placeholders(self):
        self.renderer.chat_template = (
            "{{ messages[0].content[0].text }}{{ messages[0].content[0].text }}"
        )
        with self.assertRaisesRegex(FtRuntimeException, "repeated an image placeholder"):
            self.renderer.render_chat(chat_request(image_part("https://example.test/a")))

        with self.assertRaisesRegex(FtRuntimeException, "reserved for image inputs"):
            self.renderer.render_chat(
                chat_request({"type": "text", "text": GLM54_MOCK_IMAGE_TOKEN})
            )
        self.renderer.tokenizer.encode.assert_not_called()

    def test_direct_answer_parser_removes_only_leading_closing_tags(self):
        request = chat_request({"type": "text", "text": "Describe the image."})
        request.enable_thinking = False
        parser = self.renderer._create_reasoning_parser(request)
        for text, expected in (
            ("Answer", ("", "Answer")),
            ("  Answer", ("", "  Answer")),
            ("</think></think>Answer</think>", ("", "Answer</think>")),
            ("\n</think>\n</think>\nAnswer", ("", "\nAnswer")),
            ("</think></think>", ("", "")),
            ("<think>reasoning</think>Answer", ("reasoning", "Answer")),
        ):
            with self.subTest(text=text):
                self.assertEqual(parser.parse_non_stream(text), expected)

        for chunk, expected in (
            ("\n", ("", "")),
            ("</th", ("", "")),
            ("ink>\n</think></", ("", "")),
            ("think>\nAnswer", ("", "\nAnswer")),
            ("</think>", ("", "</think>")),
        ):
            with self.subTest(chunk=chunk):
                self.assertEqual(parser.parse_stream_chunk(chunk), expected)

        parser = self.renderer._create_reasoning_parser(request)
        self.assertEqual(parser.parse_stream_chunk("\n"), ("", ""))
        self.assertEqual(parser.parse_stream_chunk(" Answer"), ("", "\n Answer"))

        parser = self.renderer._create_reasoning_parser(request)
        self.assertEqual(parser.parse_stream_chunk("\n"), ("", ""))
        self.assertEqual(parser.parse_stream_chunk("<thi"), ("", ""))
        self.assertEqual(
            parser.parse_stream_chunk("nk>trace</think>Answer"), ("trace", "Answer")
        )

    def test_stream_end_preserves_incomplete_prefixes_and_whitespace(self):
        request = chat_request({"type": "text", "text": "Describe the image."})
        request.enable_thinking = False
        for chunks, expected in (
            (("\n",), "\n"),
            (("\n", "</"), "\n</"),
            (("\n", "<thi"), "\n<thi"),
            (("</think>", "\n"), "\n"),
            (("\n", "</think>"), ""),
        ):
            with self.subTest(chunks=chunks):
                parser = self.renderer._create_reasoning_parser(request)
                status = ReasoningToolStreamStatus(request, None, parser)
                status.output = SimpleNamespace(
                    aux_info=SimpleNamespace(
                        input_len=1,
                        output_len=2,
                        reuse_len=0,
                        multimodal_lengths={},
                    )
                )
                for chunk in chunks:
                    self.assertEqual(parser.parse_stream_chunk(chunk), ("", ""))
                response = asyncio.run(
                    self.renderer._flush_buffer([status], [], True, [ThinkStatus()])
                )
                self.assertEqual(response.choices[0].delta.content, expected)
                self.assertEqual(parser.parse_non_stream("".join(chunks)), ("", expected))

    def test_resize_fits_aligned_upscale_under_maximum(self):
        bounds = {
            "temporal_patch_size": 2,
            "factor": 28,
            "min_pixels": 2 * 14_000,
            "max_pixels": 2 * 16_000,
        }
        self.assertEqual(glm5_smart_resize(28, 42, **bounds), (112, 140))
        self.assertEqual(glm5_smart_resize(42, 28, **bounds), (140, 112))

    def test_native_entry_points_reject_invalid_media_before_download(self):
        embedding = object.__new__(Glm54MockImageEmbedding)
        pixels = torch.empty(0)
        url = "https://example.test/image.png"
        invalid_inputs = (
            MultimodalInput("", MMUrlType.DEFAULT, pixels, MMPreprocessConfig()),
            MultimodalInput(url, MMUrlType.AUDIO, pixels, MMPreprocessConfig()),
            MultimodalInput(url, MMUrlType.IMAGE, pixels, MMPreprocessConfig(width=32)),
            MultimodalInput(
                url,
                MMUrlType.IMAGE,
                pixels,
                MMPreprocessConfig(crop_positions=[0.5] * 4),
            ),
            MultimodalInput(
                url,
                MMUrlType.IMAGE,
                pixels,
                MMPreprocessConfig(min_pixels=128, max_pixels=64),
            ),
        )
        validate = (
            embedding.validate_inputs,
            lambda inputs: embedding.preprocess_input(inputs, None, None, None, None),
        )
        for item in invalid_inputs:
            for entry_point in validate:
                with self.subTest(item=item, entry_point=entry_point):
                    with self.assertRaises(FtRuntimeException) as failure:
                        entry_point([item])
                    self.assertEqual(
                        failure.exception.exception_type,
                        ExceptionType.MM_WRONG_FORMAT_ERROR,
                    )

    def test_estimator_counts_parameters_without_allocating_weights(self):
        config = vision_config()
        params = VitParameters()
        params.config = {
            "vision_config": vars(config),
            "swiglu_limit": config.swiglu_limit,
        }
        meta_visual = Glm54MockVitMixin._get_mm_module(params, VitConfig())
        cpu_visual = Glm53FlashVisionModel(config)

        self.assertEqual(
            dict(meta_visual.state_dict()).keys(), dict(cpu_visual.state_dict()).keys()
        )
        self.assertTrue(
            all(parameter.is_meta for parameter in meta_visual.parameters())
        )
        self.assertTrue(meta_visual.rotary_inv_freq.is_meta)
        param_count = sum(parameter.numel() for parameter in cpu_visual.parameters())
        self.assertEqual(
            Glm54MockVitMixin.eval_mm_model_param_count(params, "", ""), param_count
        )
        self.assertEqual(
            Glm54MockVitMixin.eval_mm_model_size(params, "", ""), param_count * 2
        )

    def test_packed_visual_forward_preserves_each_image_and_zero_pads(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            visual = Glm53FlashVisionModel(vision_config()).eval()
        embedding = object.__new__(Glm54MockImageEmbedding)
        embedding.visual = visual
        embedding.padding = 8
        config = visual.config
        patch_elements = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        generator = torch.Generator().manual_seed(52)
        grids = (
            torch.tensor([[1, 2, 4]]),
            torch.tensor([[1, 2, 2]]),
            torch.tensor([[1, 2, 2]]),
        )
        data = [
            (torch.randn((int(grid.prod()), patch_elements), generator=generator), grid)
            for grid in grids
        ]

        result = embedding.batched_embedding(data, [MMUrlType.IMAGE] * len(data))

        self.assertEqual(len(result), len(data))
        with torch.inference_mode():
            for (features, position_ids, layout), (pixels, grid) in zip(result, data):
                expected = visual(pixels, grid)
                self.assertEqual(len(features), 1)
                self.assertEqual(features[0].shape, (expected.shape[0], 24))
                self.assertIsNone(position_ids)
                self.assertIsNone(layout)
                torch.testing.assert_close(
                    features[0][:, : config.out_hidden_size],
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )
                padding = features[0][:, config.out_hidden_size :]
                torch.testing.assert_close(padding, torch.zeros_like(padding))


if __name__ == "__main__":
    unittest.main()

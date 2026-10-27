"""Exercise the D3 HTTP/worker wiring with the official protocol and a mocked RPC."""

import asyncio
import base64
import io
import json
import math
import shutil
import tempfile
import unittest
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from fastapi.testclient import TestClient

from rtp_llm.frontend.d3 import MAX_OPTIONS, answer_codes
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import FrontendWorker, d3_executor, d3_model_name
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.metrics import AccMetrics, GaugeMetrics
from rtp_llm.models.downstream_modules.d3_config import D3Config
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    MMUrlType,
)


class FakeTokenizer:
    """Deterministic chat-template tokenizer; a maximal run of capitals is one token."""

    def __init__(self):
        self.runs = {}
        self.rendered = []

    def get_real_tokenizer(self):
        return self

    def _token_of(self, run):
        if run not in self.runs:
            self.runs[run] = 100000 + len(self.runs)
        return self.runs[run]

    def _encode(self, text):
        ids, run = [], ""
        for char in text + "\0":
            if char.isascii() and char.isupper():
                run += char
                continue
            if run:
                ids.append(self._token_of(run))
                run = ""
            if char != "\0":
                ids.append(1000 + ord(char))
        return ids

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self._encode(item) for item in text]}
        return {"input_ids": self._encode(text)}

    def encode(self, text, add_special_tokens=True):
        return self._encode(text)

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
        **_,
    ):
        parts = []
        for message in conversation:
            content = message["content"]
            if isinstance(content, list):
                body = "".join(
                    (
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        if part["type"] == "image"
                        else (
                            "<|vision_start|><|video_pad|><|vision_end|>"
                            if part["type"] == "video"
                            else part["text"]
                        )
                    )
                    for part in content
                )
            else:
                body = content
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        self.rendered.append(rendered)
        return rendered


def image_url(width=64, height=64):
    from PIL import Image

    image = Image.new("RGB", (width, height), (96, 160, 224))
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{data}"


def video_url(width=64, height=64, frames=24, fps=24, color=(96, 160, 224)):
    """A small constant-color MP4 clip as a strict data URL (PyAV encode)."""
    import av
    import numpy as np

    with io.BytesIO() as buffer:
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        array = np.empty((height, width, 3), dtype=np.uint8)
        array[:, :] = color
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        data = base64.b64encode(buffer.getvalue()).decode()
    return f"data:video/mp4;base64,{data}"


def official_group_text(meta):
    """One group's official expansion string, from its parsed URL metadata.

    Official ``Qwen3VLProcessor.replace_video_token`` renders each 2-frame
    group as its timestamp ("<X.X seconds>") followed by the group's
    vision-wrapped pad tokens; the timestamp is the mean of the group's two
    frame times (official ``_calculate_timestamps``).
    """
    first, second = meta["frames_indices"]
    time = (first / meta["fps"] + second / meta["fps"]) / 2
    return (
        f"<{time:.1f} seconds>"
        + "<|vision_start|>"
        + "<|video_pad|>" * meta["tokens"]
        + "<|vision_end|>"
    )


def official_replacement(text, expansions):
    """Official ``get_text_with_replacements`` over the bare video token.

    Scans the ORIGINAL text for bare ``<|video_pad|>`` occurrences and
    replaces the i-th occurrence with the i-th expansion, keeping every other
    character (the template's outer markers) and never rescanning insertions.
    """
    import re

    pattern = re.compile(re.escape("<|video_pad|>"))
    out, last = [], 0
    for match, expansion in zip(pattern.finditer(text), expansions):
        out.append(text[last : match.start()])
        out.append(expansion)
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def d3_config(tokenizer, **overrides):
    codes, token_ids = answer_codes(tokenizer)
    values = dict(
        format_version=1,
        format_id="d3-code-readout-v1",
        codes=tuple(codes),
        token_ids=tuple(token_ids),
        temperature=1.0,
        attention_mode="noncausal_full_attention",
        pooling="last",
        max_length=None,
        readout_dtype="float32",
        hidden_size=2560,
    )
    values.update(overrides)
    return D3Config(**values)


def systemone_request(**extra):
    return {
        "model": "ignored",
        "state": {"text": "context"},
        "questions": {
            "pick": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"red": "warm", "blue": "cool"},
            },
            "check": {"type": "noul", "instructions": "Does it match?"},
            "grade": {"type": "score", "criteria": ["low", "mid", "high"]},
        },
        **extra,
    }


class D3SystemOneTest(unittest.TestCase):
    def setUp(self):
        reporter = patch("rtp_llm.frontend.frontend_server.kmonitor.report")
        self.report = reporter.start()
        self.addCleanup(reporter.stop)
        self.worker = object.__new__(FrontendWorker)
        self.worker.tokenizer = FakeTokenizer()
        self.worker.kev_config = None
        self.worker.jev_config = None
        self.worker.jev_executor = None
        self.worker._d3_request_context = ContextVar("test_d3_request_context")
        self.worker._jev_request_context = ContextVar("test_jev_request_context")
        from rtp_llm.frontend.d3 import D3Executor

        self.worker.d3_executor = D3Executor(
            d3_config(self.worker.tokenizer),
            self.worker._d3_rows,
            tokenizer=self.worker.tokenizer,
            model_name="d3-mini",
            batch_rows=2,
            timeout_seconds=10,
        )
        self.enqueue = AsyncMock()
        self.worker.backend_rpc_server_visitor = SimpleNamespace(
            batch_enqueue=self.enqueue
        )
        self.enqueue_rows(input_len=100)

    def enqueue_rows(self, *, input_len=100, reuse_len=0, logits=None, finished=True):
        async def respond(inputs, *, routing_key):
            return [
                self.output(
                    logits=logits,
                    finished=finished,
                    input_len=input_len,
                    reuse_len=reuse_len,
                )
                for _ in inputs
            ]

        self.enqueue.side_effect = respond

    def output(self, *, logits=None, finished=True, input_len=0, reuse_len=0):
        if logits is None:
            # Code 'B' (index 1) wins every question.
            logits = torch.zeros((1, MAX_OPTIONS), dtype=torch.float32)
            logits[0, 1] = 4.0
        return GenerateOutputs(
            generate_outputs=[
                GenerateOutput(
                    finished=finished,
                    custom_output=logits,
                    aux_info=AuxInfo(input_len=input_len, reuse_len=reuse_len),
                )
            ]
        )

    def server(self):
        server = object.__new__(FrontendServer)
        server._frontend_worker = self.worker
        server._global_controller = MagicMock()
        server._global_controller.max_concurrency = 4
        server._access_logger = MagicMock()
        server.rank_id = "0"
        server.server_id = "0"
        server.is_embedding = False
        return server

    def client(self, server):
        owner = object.__new__(FrontendApp)
        owner.frontend_server = server
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        return TestClient(owner.create_app())

    def assert_metrics(self, *expected):
        self.assertCountEqual(
            [call.args[0] for call in self.report.call_args_list], expected
        )

    def test_systemone_runs_official_protocol_over_batch_rpc(self):
        self.enqueue_rows(input_len=100)
        result = asyncio.run(self.worker.systemone(systemone_request()))
        self.assertEqual(result["model"], "d3-mini")
        self.assertEqual(set(result), {"model", "answers", "usage"})
        self.assertEqual(list(result["answers"]), ["pick", "check", "grade"])
        e4 = math.exp(4.0)
        expected_blue = e4 / (1.0 + e4)
        self.assertEqual(result["answers"]["pick"]["choice"], "blue")
        self.assertAlmostEqual(
            result["answers"]["pick"]["probabilities"]["blue"], expected_blue, places=12
        )
        self.assertAlmostEqual(
            result["answers"]["check"]["noul"], expected_blue, places=12
        )
        # grade: three options [0, 4, 0] -> p1 = e^4/(2+e^4), p2 = 1/(2+e^4).
        self.assertAlmostEqual(
            result["answers"]["grade"]["score"],
            e4 / (2.0 + e4) + 2.0 / (2.0 + e4),
            places=12,
        )
        self.assertEqual(result["usage"], {"input_tokens": 300, "output_tokens": 0})
        # batch_rows=2: three questions split into two RPCs, one routing key.
        self.assertEqual(
            [len(call.args[0]) for call in self.enqueue.await_args_list], [2, 1]
        )
        self.assertEqual(
            len({call.kwargs["routing_key"] for call in self.enqueue.await_args_list}),
            1,
        )

    def test_rpc_rows_are_independent_prefill_only_requests(self):
        self.enqueue_rows(input_len=100)
        asyncio.run(self.worker.systemone(systemone_request()))
        items = [item for call in self.enqueue.await_args_list for item in call.args[0]]
        self.assertEqual(len(items), 3)
        self.assertEqual(len({item.request_id for item in items}), 3)
        self.assertTrue(all(0 <= item.request_id < (1 << 63) for item in items))
        for item in items:
            self.assertEqual(item.token_ids.dtype, torch.int32)
            self.assertEqual(item.token_ids.device.type, "cpu")
            self.assertEqual(item.generate_config.max_new_tokens, 0)
            self.assertFalse(item.generate_config.is_streaming)
            self.assertEqual(item.mm_inputs, [])
            self.assertEqual(item.prefix_length, 0)
            self.assertIs(item.tokenizer, self.worker.tokenizer)
        # The prompt is the official chat template with thinking disabled.
        # The rendered list starts with the answer-codes probes (d3_config()
        # and the D3Executor constructor each call answer_codes, which renders
        # the probe conversation); the question prompts are the LAST three.
        prompt = self.worker.tokenizer.rendered[-3]
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertIn("Options:\nA: red: warm\nB: blue: cool", prompt)
        self.assertIn('State:\n{"text": "context"}', prompt)
        self.assertEqual(
            items[0].token_ids[-1].item(),
            self.worker.tokenizer._encode("\n")[-1],
        )

    def test_bounded_batches_split_complete_rows_only(self):
        request = systemone_request()
        request["questions"] = {f"q{i}": {"type": "noul"} for i in range(5)}
        self.enqueue_rows(input_len=10)
        result = asyncio.run(self.worker.systemone(request))
        self.assertEqual(
            [len(call.args[0]) for call in self.enqueue.await_args_list], [2, 2, 1]
        )
        self.assertEqual(len(result["answers"]), 5)
        self.assertEqual(result["usage"], {"input_tokens": 50, "output_tokens": 0})

    def test_images_flow_through_multimodal_rpc_with_official_budget(self):
        url = image_url(1024, 1024)
        request = systemone_request(images=[url])
        expanded = 500
        self.enqueue_rows(input_len=expanded)
        result = asyncio.run(self.worker.systemone(request))
        items = [item for call in self.enqueue.await_args_list for item in call.args[0]]
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(len(item.mm_inputs), 1)
            media = item.mm_inputs[0]
            self.assertEqual(media.url, url)
            self.assertEqual(media.mm_type, MMUrlType.IMAGE)
            self.assertEqual(media.tensor.numel(), 0)
            self.assertEqual(media.mm_preprocess_config.min_pixels, 65536)
            self.assertEqual(media.mm_preprocess_config.max_pixels, 1638400)
        for prompt in self.worker.tokenizer.rendered[-3:]:
            self.assertEqual(prompt.count("<|image_pad|>"), 1)
        self.assertEqual(
            result["usage"], {"input_tokens": 3 * expanded, "output_tokens": 0}
        )

    def test_videos_flow_through_multimodal_rpc_as_group_entries(self):
        from rtp_llm.multimodal.multimodal_util import parse_d3_video_group_url

        url = video_url(frames=24, fps=24)  # -> 4 sampled frames, 2 groups
        request = systemone_request(videos=[url])
        expanded = 500
        self.enqueue_rows(input_len=expanded)
        result = asyncio.run(self.worker.systemone(request))
        items = [item for call in self.enqueue.await_args_list for item in call.args[0]]
        self.assertEqual(len(items), 3)
        metas = []
        for item in items:
            self.assertEqual(len(item.mm_inputs), 2)
            for media in item.mm_inputs:
                self.assertEqual(media.mm_type, MMUrlType.VIDEO)
                meta = parse_d3_video_group_url(media.url)
                # 4 frames of 64x64 stay 64x64 (already factor-aligned, within
                # the per-frame budget): one 4-token group per 2 frames.
                self.assertEqual(meta["total_groups"], 2)
                self.assertEqual(meta["tokens"], 4)
                self.assertEqual(meta["height"], 64)
                self.assertEqual(meta["width"], 64)
                self.assertEqual(meta["fps"], 24.0)
                self.assertEqual(tuple(media.tensor.shape), (2, 3, 64, 64))
                self.assertEqual(media.tensor.dtype, torch.uint8)
                # The URL is self-contained: its lossless payload carries the
                # SAME producer-resized bytes as the in-process tensor (the
                # RPC drops the tensor; only the URL crosses).
                self.assertEqual(
                    meta["frames"],
                    bytes(media.tensor.contiguous().numpy().tobytes()),
                )
                metas.append(meta)
        self.assertEqual(sorted(m["group_index"] for m in metas), [0, 0, 0, 1, 1, 1])
        for metas_one_row in (metas[:2], metas[2:4], metas[4:6]):
            self.assertEqual(
                [m["frames_indices"] for m in metas_one_row],
                [[0, 8], [15, 23]],
            )
        # The rendered template carries the BARE placeholder (one per video);
        # the official replacement swaps only that token for the expansion,
        # keeping the template's OUTER vision markers around it. Each enqueued
        # row must be exactly the tokenization of that official result —
        # expansion reconstructed here independently from the parsed group
        # metadata (timestamps [0,8]/24 -> "<0.2 seconds>", [15,23]/24 ->
        # "<0.8 seconds>").
        for prompt, item, metas_one_row in zip(
            self.worker.tokenizer.rendered[-3:],
            items,
            (metas[:2], metas[2:4], metas[4:6]),
        ):
            self.assertEqual(prompt.count("<|video_pad|>"), 1)
            expansion = "".join(official_group_text(m) for m in metas_one_row)
            expected = official_replacement(prompt, [expansion])
            self.assertIn("<0.2 seconds><|vision_start|>", expected)
            self.assertIn("<0.8 seconds><|vision_start|>", expected)
            self.assertEqual(expected.count("<|video_pad|>"), 8)  # 2 groups x 4 pads
            self.assertEqual(expected.count("<|vision_start|>"), 3)  # 2 groups + outer
            self.assertEqual(expected.count("<|vision_end|>"), 3)
            self.assertEqual(
                item.token_ids.tolist(),
                self.worker.tokenizer.encode(expected, add_special_tokens=False),
            )
        self.assertEqual(
            result["usage"], {"input_tokens": 3 * expanded, "output_tokens": 0}
        )

    def test_mixed_image_and_video_rows_keep_prompt_order(self):
        from rtp_llm.multimodal.multimodal_util import parse_d3_video_group_url

        request = systemone_request(
            images=[image_url(1024, 1024)], videos=[video_url()]
        )
        self.enqueue_rows(input_len=300)
        asyncio.run(self.worker.systemone(request))
        items = [item for call in self.enqueue.await_args_list for item in call.args[0]]
        for item in items:
            kinds = [media.mm_type for media in item.mm_inputs]
            # Images first, then the video's groups, in prompt placeholder order.
            self.assertEqual(kinds, [MMUrlType.IMAGE, MMUrlType.VIDEO, MMUrlType.VIDEO])
        for prompt, item in zip(self.worker.tokenizer.rendered[-3:], items):
            image_at = prompt.index("<|image_pad|>")
            video_at = prompt.index("<|video_pad|>")
            self.assertLess(image_at, video_at)
            # Both media keep their template wrappers: the image keeps its own
            # pair, and the video's expansion adds its per-group markers inside
            # the template's outer pair (official replaces the bare token only).
            metas = [
                parse_d3_video_group_url(media.url) for media in item.mm_inputs[1:]
            ]
            expansion = "".join(official_group_text(m) for m in metas)
            expected = official_replacement(prompt, [expansion])
            self.assertEqual(prompt.count("<|image_pad|>"), 1)
            self.assertEqual(prompt.count("<|video_pad|>"), 1)
            self.assertEqual(expected.count("<|vision_start|>"), 1 + 2 + 1)
            self.assertEqual(expected.count("<|vision_end|>"), 1 + 2 + 1)
            self.assertEqual(
                item.token_ids.tolist(),
                self.worker.tokenizer.encode(expected, add_special_tokens=False),
            )

    def test_over_limit_video_question_names_both_media_kinds(self):
        from rtp_llm.frontend.d3 import D3Executor

        video = SimpleNamespace(
            # _video_group_entries encodes (and slices) the real frames, so the
            # patched video must carry a real [32, 3, 64, 64] uint8 tensor.
            frames=torch.zeros(32, 3, 64, 64, dtype=torch.uint8),
            fps=24.0,
            indices=list(range(32)),
            total_frames=480,
            height=64,
            width=64,
            timestamps=[0.5] * 16,
            group_tokens=4,
            key="k",
            tokens=16 * 4,
            groups=16,
        )
        with patch("rtp_llm.frontend.d3.load_video", return_value=video):
            executor = D3Executor(
                d3_config(self.worker.tokenizer, max_length=200),
                self.worker._d3_rows,
                tokenizer=self.worker.tokenizer,
                model_name="d3-mini",
            )
            result = executor.prepare(
                dict(
                    systemone_request(
                        images=[image_url(64, 64)],
                        videos=["data:video/mp4;base64,AAAA"],
                    )
                )
            )
        self.assertIn("image(s) and 1 video(s)", result.errors["pick"]["message"])
        self.assertIn("nothing was truncated", result.errors["pick"]["message"])

    def test_invalid_requests_map_to_422_with_official_detail(self):
        cases = [
            ([], "expected a JSON object"),
            ({}, "questions is required"),
            (dict(systemone_request(), extra=1), "'extra' not understood"),
            (
                dict(
                    systemone_request(),
                    videos=["data:video/avi;base64,AAAA"],
                ),
                "video data URLs take the types",
            ),
            (
                dict(systemone_request(), questions={"bad": {"type": "choice"}}),
                "invalid_questions",
            ),
        ]
        for request, marker in cases:
            with self.subTest(request=request):
                server = self.server()
                self.report.reset_mock()
                response = asyncio.run(server.systemone(request))
                self.assertEqual(response.status_code, 422)
                self.assertIn(marker, json.dumps(json.loads(response.body)["detail"]))
                self.enqueue.assert_not_awaited()
                self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
                self.assertEqual(
                    self.report.call_args_list[-1].args[2]["error_code"],
                    "INVALID_PARAMS",
                )
                server._global_controller.increment.assert_called_once_with()
                server._global_controller.decrement.assert_called_once_with()

    def test_invalid_questions_detail_body_is_official(self):
        server = self.server()
        request = dict(
            systemone_request(), questions={"bad": "nope", "q": {"type": "noul"}}
        )
        response = asyncio.run(server.systemone(request))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            json.loads(response.body)["detail"],
            {"invalid_questions": {"bad": "a question is an object with a type"}},
        )

    def test_render_invalid_questions_reject_the_whole_request_before_running(self):
        # Official contract: a literal placeholder token on a media path is a
        # render-phase invalid_question, and EVERY invalid_question rejects the
        # WHOLE request with the 422 invalid_questions body before anything
        # runs — the valid question in the same request never reaches the RPC.
        server = self.server()
        request = dict(
            systemone_request(
                images=[image_url(64, 64)],
                questions={
                    "bad": {
                        "type": "noul",
                        "instructions": "before <|video_pad|> after",
                    },
                    "good": {"type": "noul"},
                },
            ),
        )
        response = asyncio.run(server.systemone(request))
        self.assertEqual(response.status_code, 422)
        detail = json.loads(response.body)["detail"]
        self.assertEqual(list(detail["invalid_questions"]), ["bad"])
        self.assertIn(
            "literal image or video placeholder token",
            detail["invalid_questions"]["bad"],
        )
        self.enqueue.assert_not_awaited()

    def test_two_videos_expand_independently_in_order(self):
        # Independent-review P2: a 1-token group's expansion is itself a
        # placeholder-shaped span, so the expansion must stay indexed per
        # video — an earlier video's groups can never eat a later video's
        # placeholder. End-to-end: a 32x32 video (2 groups x 1 token, the
        # trigger) followed by a longer 64x64 clip (2 groups x 4 tokens,
        # distinct timestamps).
        from rtp_llm.multimodal.multimodal_util import parse_d3_video_group_url

        tiny = video_url(width=32, height=32, frames=24, fps=24)
        longer = video_url(width=64, height=64, frames=48, fps=24)
        request = systemone_request(videos=[tiny, longer])
        asyncio.run(self.worker.systemone(request))
        rows = [item for call in self.enqueue.await_args_list for item in call.args[0]]
        self.assertEqual(len(rows), 3)
        for row in rows:
            metas = [parse_d3_video_group_url(media.url) for media in row.mm_inputs]
            # Prompt placeholder order: the first video's groups, then the
            # second video's groups.
            self.assertEqual([m["group_index"] for m in metas], [0, 1, 0, 1])
            self.assertEqual([m["total_groups"] for m in metas], [2, 2, 2, 2])
            self.assertEqual([m["height"] for m in metas], [32, 32, 64, 64])
            # Each video's groups carry that video's content-derived key.
            self.assertEqual(metas[0]["video_key"], metas[1]["video_key"])
            self.assertEqual(metas[2]["video_key"], metas[3]["video_key"])
            self.assertNotEqual(metas[0]["video_key"], metas[2]["video_key"])
            # The URL payloads carry each group's exact resized frames.
            self.assertEqual(metas[0]["frames_bytes"], 2 * 3 * 32 * 32)
            self.assertEqual(metas[2]["frames_bytes"], 2 * 3 * 64 * 64)
        for prompt, row in zip(self.worker.tokenizer.rendered[-3:], rows):
            # The template renders one BARE placeholder per video; the official
            # replacement swaps the i-th occurrence for the i-th video's
            # expansion, so each video's groups land only in its own slot.
            self.assertEqual(prompt.count("<|video_pad|>"), 2)
            metas = [parse_d3_video_group_url(media.url) for media in row.mm_inputs]
            expansions = [
                "".join(official_group_text(m) for m in metas[:2]),
                "".join(official_group_text(m) for m in metas[2:]),
            ]
            expected = official_replacement(prompt, expansions)
            # 2 groups x 1 pad + 2 groups x 4 pads, no leftover placeholder.
            self.assertEqual(expected.count("<|video_pad|>"), 2 * 1 + 2 * 4)
            # Timestamp groups stay with their own video, in order: the tiny
            # video's 0.2/0.8 seconds entirely before the longer clip's
            # 0.3/1.6 seconds (the old bug reordered/nested them).
            self.assertLess(
                expected.index("<0.2 seconds>"), expected.index("<0.8 seconds>")
            )
            self.assertLess(
                expected.index("<0.8 seconds>"), expected.index("<0.3 seconds>")
            )
            self.assertLess(
                expected.index("<0.3 seconds>"), expected.index("<1.6 seconds>")
            )
            # Each video's whole expansion stays inside its template wrapper:
            # per video, its groups' markers plus the outer one.
            self.assertEqual(expected.count("<|vision_start|>"), 3 + 3)
            self.assertEqual(expected.count("<|vision_end|>"), 3 + 3)
            self.assertEqual(
                row.token_ids.tolist(),
                self.worker.tokenizer.encode(expected, add_special_tokens=False),
            )

    def test_over_limit_question_answers_error_and_siblings_still_run(self):
        request = systemone_request()
        request["questions"]["huge"] = {
            "type": "choice",
            "instructions": "many",
            "criteria": {f"option-{i}": f"description {i}" for i in range(250)},
        }
        prepared = self.worker.d3_executor.prepare(request)
        grade_len = [
            row.planned_tokens for row in prepared.rows if row.question_id == "grade"
        ][0]
        self.worker.d3_executor.max_row_tokens = grade_len
        self.enqueue_rows(input_len=100)
        result = asyncio.run(self.worker.systemone(request))
        error = result["answers"]["huge"]
        self.assertEqual(error["error"], "max_length_exceeded")
        self.assertEqual(error["type"], "choice")
        self.assertIn("nothing was truncated", error["message"])
        self.assertEqual(result["answers"]["pick"]["choice"], "blue")
        self.assertEqual(result["usage"]["output_tokens"], 0)
        self.assertEqual(result["usage"]["input_tokens"], 300)

    def test_malformed_backend_results_are_server_errors(self):
        # Structural corruption (wrong readout shape) is a whole-request 500;
        # NON-FINITE logits are NOT structural — the landed P3 semantics route
        # them through the executor's probabilities(), which reports the
        # official per-question invalid_model_output error on a normal 200.
        structural = [
            torch.zeros(MAX_OPTIONS),
            torch.zeros((1, MAX_OPTIONS - 1)),
        ]
        for logits in structural:
            with self.subTest(shape=tuple(logits.shape)):
                self.report.reset_mock()
                server = self.server()
                self.enqueue_rows(logits=logits)
                response = asyncio.run(server.systemone(systemone_request()))
                self.assertEqual(response.status_code, 500)
                self.assert_metrics(AccMetrics.QPS_METRIC, AccMetrics.ERROR_QPS_METRIC)
                server._global_controller.decrement.assert_called_once_with()
                server._access_logger.log_exception_access.assert_called_once()
        nonfinite = [
            torch.full((1, MAX_OPTIONS), float("nan")),
            torch.full((1, MAX_OPTIONS), float("inf")),
        ]
        for logits in nonfinite:
            with self.subTest(shape=tuple(logits.shape)):
                self.report.reset_mock()
                server = self.server()
                self.enqueue_rows(logits=logits)
                response = asyncio.run(server.systemone(systemone_request()))
                self.assertEqual(response.status_code, 200)
                body = json.loads(response.body)
                self.assertEqual(
                    body["usage"], {"input_tokens": 300, "output_tokens": 0}
                )
                for qid in ("pick", "check", "grade"):
                    self.assertEqual(
                        body["answers"][qid]["error"], "invalid_model_output"
                    )
                    self.assertIn("non-finite logit", body["answers"][qid]["message"])
                # A 200 counts as success: framework QPS + success QPS +
                # the framework latency gauge (all three are reported).
                self.assert_metrics(
                    AccMetrics.QPS_METRIC,
                    AccMetrics.SUCCESS_QPS_METRIC,
                    GaugeMetrics.LANTENCY_METRIC,
                )
                server._global_controller.decrement.assert_called_once_with()
        for outputs in ([], [GenerateOutputs(generate_outputs=[])]):
            with self.subTest(outputs=len(outputs)):
                self.report.reset_mock()
                server = self.server()
                self.enqueue.return_value = outputs
                self.enqueue.side_effect = None
                response = asyncio.run(server.systemone(systemone_request()))
                self.assertEqual(response.status_code, 500)
                server._global_controller.decrement.assert_called_once_with()

    def test_unfinished_rows_and_prefix_reuse_are_server_errors(self):
        for kwargs, marker in (
            ({"finished": False}, "invalid readout logits"),
            ({"reuse_len": 7}, "must not reuse"),
        ):
            with self.subTest(marker=marker):
                self.report.reset_mock()
                server = self.server()
                self.enqueue_rows(**kwargs)
                response = asyncio.run(server.systemone(systemone_request()))
                self.assertEqual(response.status_code, 500)
                # The Starlette body is bytes: decode/parse it, never
                # json.dumps the raw body.
                self.assertIn(marker, json.loads(response.body)["message"])

    def test_cancellation_drains_the_rpc_and_releases_the_service_slot(self):
        server = self.server()
        request = dict(systemone_request())
        request["questions"] = {"a": {"type": "noul"}, "b": {"type": "noul"}}

        async def run():
            entered = 0
            stopped = 0
            all_entered = asyncio.Event()

            async def blocked(inputs, *, routing_key):
                nonlocal entered, stopped
                entered += 1
                all_entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped += 1
                return [self.output() for _ in inputs]

            self.enqueue.side_effect = blocked
            task = asyncio.create_task(server.systemone(request))
            await asyncio.wait_for(all_entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return stopped, entered

        stopped, entered = asyncio.run(run())
        self.assertEqual(entered, 1)
        self.assertEqual(stopped, 1)
        self.assertIsNone(self.worker._d3_request_context.get(None))
        server._global_controller.decrement.assert_called_once_with()

    def test_concurrent_requests_keep_routing_keys_and_headers_separate(self):
        first_headers = {"x-request-id": "first", "traceparent": "trace-first"}
        second_headers = {"x-request-id": "second", "traceparent": "trace-second"}

        async def run():
            self.enqueue_rows(input_len=7)
            return await asyncio.gather(
                self.worker.systemone(systemone_request(), headers=first_headers),
                self.worker.systemone(systemone_request(), headers=second_headers),
            )

        first, second = asyncio.run(run())
        self.assertIsNone(self.worker._d3_request_context.get(None))
        routes = {}
        for call in self.enqueue.await_args_list:
            for item in call.args[0]:
                routes.setdefault(item.headers["x-request-id"], set()).add(
                    call.kwargs["routing_key"]
                )
        self.assertEqual(set(routes), {"first", "second"})
        self.assertEqual(len(routes["first"]), 1)
        self.assertEqual(len(routes["second"]), 1)
        self.assertTrue(routes["first"].isdisjoint(routes["second"]))
        self.assertEqual(first["model"], second["model"])

    def test_d3_deployment_keeps_decide_endpoints_on_kev_errors(self):
        from rtp_llm.models.downstream_modules.kev import KevRequestError

        server = self.server()
        response = asyncio.run(server.decide({"kind": "choice", "question": "q"}))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.body))
        response = server.decide_info()
        self.assertEqual(response.status_code, 404)
        with self.assertRaises(KevRequestError):
            # worker.decide is async — await it, or the coroutine object never
            # runs and no exception is ever raised.
            asyncio.run(self.worker.decide({"kind": "choice", "question": "q"}))

    def test_http_route_serves_the_official_response(self):
        self.enqueue_rows(input_len=42)
        server = self.server()
        client = self.client(server)
        response = client.post("/v1/systemone", json=systemone_request())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], "d3-mini")
        self.assertEqual(body["usage"], {"input_tokens": 126, "output_tokens": 0})
        self.assertEqual(body["answers"]["pick"]["choice"], "blue")
        bad = client.post("/v1/systemone", json={"questions": {}})
        self.assertEqual(bad.status_code, 422)
        self.assertIn("questions is required", bad.json()["detail"])


class D3ExecutorFactoryTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.root = tempfile.mkdtemp(prefix="d3-ckpt-")
        self.addCleanup(shutil.rmtree, self.root, True)
        codes, token_ids = answer_codes(self.tokenizer)
        self.decision = {
            "format_version": 1,
            "format_id": "d3-code-readout-v1",
            "prompt": "d3",
            "codes": list(codes),
            "token_ids": list(token_ids),
            "temperature": 1.0,
            "attention_mode": "noncausal_full_attention",
            "pooling": "last",
            "max_length": None,
            "readout_dtype": "float32",
        }
        with open(f"{self.root}/config.json", "w") as file:
            json.dump(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "hidden_size": 2560,
                        "layer_types": ["linear_attention", "full_attention"],
                    },
                },
                file,
            )

    def sidecar(self):
        with open(f"{self.root}/decision_config.json", "w") as file:
            json.dump(self.decision, file)

    def test_no_sidecar_leaves_other_deployments_untouched(self):
        self.assertIsNone(
            d3_executor(
                self.root,
                lambda rows: [],
                tokenizer=self.tokenizer,
                max_seq_len=None,
                kev_config=None,
                jev_config=None,
                timeout_seconds=None,
            )
        )
        self.assertEqual(d3_model_name(self.root), self.root.rstrip("/").split("/")[-1])

    def test_sidecar_builds_an_executor_named_after_the_manifest(self):
        self.sidecar()
        with open(f"{self.root}/MODEL_MANIFEST.json", "w") as file:
            json.dump(
                {"schema": "d3-package-manifest/1", "model_name": "d3-mini"}, file
            )
        with patch.dict("os.environ", {"D3_MAX_ROWS_PER_RPC": "3"}):
            executor = d3_executor(
                self.root,
                lambda rows: [],
                tokenizer=self.tokenizer,
                max_seq_len=1024,
                kev_config=None,
                jev_config=None,
                timeout_seconds=5,
            )
        self.assertIsNotNone(executor)
        self.assertEqual(executor.model_name, "d3-mini")
        self.assertEqual(executor.batch_rows, 3)
        self.assertEqual(executor.max_row_tokens, 1024)

    def test_manifest_conflicts_are_refused(self):
        self.sidecar()
        for kev, jev in (({"max_options": 4}, None), (None, {"schema_version": 1})):
            with self.subTest(kev=bool(kev), jev=bool(jev)):
                with self.assertRaisesRegex(ValueError, "cannot combine D3"):
                    d3_executor(
                        self.root,
                        lambda rows: [],
                        tokenizer=self.tokenizer,
                        max_seq_len=None,
                        kev_config=kev,
                        jev_config=jev,
                        timeout_seconds=None,
                    )

    def test_bad_row_limit_env_is_refused(self):
        self.sidecar()
        with patch.dict("os.environ", {"D3_MAX_ROWS_PER_RPC": "0"}):
            with self.assertRaisesRegex(ValueError, "D3_MAX_ROWS_PER_RPC"):
                d3_executor(
                    self.root,
                    lambda rows: [],
                    tokenizer=self.tokenizer,
                    max_seq_len=None,
                    kev_config=None,
                    jev_config=None,
                    timeout_seconds=None,
                )


if __name__ == "__main__":
    unittest.main()

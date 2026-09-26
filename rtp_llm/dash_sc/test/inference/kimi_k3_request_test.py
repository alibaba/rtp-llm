"""K3 DashSc request tests using the public gRPC message shape."""

import json
import unittest
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.dash_sc.codec import DashScRequestControls, SamplingParams
from rtp_llm.dash_sc.inference.servicer import (
    _normalize_k3_image_prompt,
    _resolve_k3_dash_sc_thinking,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2


class _K3Tokenizer:
    _markers = {
        "<|kimi_image_placeholder|>": [22, 11],
        "<|media_pad|>": [163605],
        "<|media_begin|>": [20],
        "<|media_end|>": [21],
    }

    def encode(self, text):
        return self._markers[text]

    def decode(self, *_args, **_kwargs):
        raise AssertionError("K3 request normalization must preserve ordinary token IDs")


def _request_with_media(*parts):
    request = predict_v2_pb2.ModelInferRequest()
    request.parameters["payload"].string_param = json.dumps(
        {"input": {"messages": [{"role": "user", "content": list(parts)}]}}
    )
    return request


class KimiK3DashScThinkingTest(unittest.TestCase):
    def test_structured_output_defaults_to_response_channel(self) -> None:
        cases = (
            (SamplingParams(), True),
            (SamplingParams(response_format='{"type":"json_object"}'), False),
            (SamplingParams(json_format=True), False),
        )
        for sampling, expected in cases:
            with self.subTest(sampling=sampling):
                controls = DashScRequestControls()
                config = sampling.to_generate_config(request_controls=controls)
                self.assertEqual(
                    _resolve_k3_dash_sc_thinking(sampling, controls, config),
                    expected,
                )

    def test_explicit_controls_take_precedence(self) -> None:
        sampling = SamplingParams(response_format='{"type":"json_object"}')
        for controls, expected in (
            (DashScRequestControls(enable_thinking=True), True),
            (DashScRequestControls(reasoning_effort="high"), True),
            (DashScRequestControls(enable_thinking=False), False),
            (DashScRequestControls(max_new_think_tokens=0), False),
        ):
            with self.subTest(controls=controls):
                config = sampling.to_generate_config(request_controls=controls)
                self.assertEqual(
                    _resolve_k3_dash_sc_thinking(sampling, controls, config),
                    expected,
                )


class KimiK3DashScMediaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _K3Tokenizer()

    def test_one_placeholder_per_image_preserves_other_tokens(self) -> None:
        request = _request_with_media(
            {"image": "https://example.com/a.jpg", "min_pixels": 50176},
            {"image": "https://example.com/b.jpg"},
        )
        input_ids = [7, 22, 11, 8, 900, 901, 22, 11, 9]
        with patch(
            "rtp_llm.multimodal.multimodal_util.get_bytes_io_from_url",
            side_effect=AssertionError("frontend must not download images"),
        ) as download:
            token_ids, mm_inputs = _normalize_k3_image_prompt(
                request, input_ids, self.tokenizer
            )

        self.assertEqual(token_ids, [7, 163605, 8, 900, 901, 163605, 9])
        self.assertEqual(
            [item.url for item in mm_inputs],
            ["https://example.com/a.jpg", "https://example.com/b.jpg"],
        )
        self.assertEqual(mm_inputs[0].mm_preprocess_config.min_pixels, 50176)
        self.assertTrue(all(item.tensor.numel() == 0 for item in mm_inputs))
        download.assert_not_called()

    def test_expanded_media_span_collapses_to_one_pad(self) -> None:
        request = _request_with_media({"image": "https://example.com/a.jpg"})

        token_ids, mm_inputs = _normalize_k3_image_prompt(
            request, [7, 20, 640, 480, 163605, 21, 8], self.tokenizer
        )

        self.assertEqual(token_ids, [7, 163605, 8])
        self.assertEqual(len(mm_inputs), 1)

    def test_rejects_mismatched_image_count(self) -> None:
        request = _request_with_media(
            {"image": "https://example.com/a.jpg"},
            {"image": "https://example.com/b.jpg"},
        )

        with self.assertRaisesRegex(FtRuntimeException, "count does not match"):
            _normalize_k3_image_prompt(request, [7, 22, 11, 8], self.tokenizer)

    def test_rejects_non_image_media(self) -> None:
        for media_type in ("video", "audio"):
            with self.subTest(media_type=media_type):
                request = _request_with_media(
                    {media_type: "https://example.com/media"}
                )
                with self.assertRaisesRegex(FtRuntimeException, "only image"):
                    _normalize_k3_image_prompt(request, [7], self.tokenizer)


if __name__ == "__main__":
    unittest.main()

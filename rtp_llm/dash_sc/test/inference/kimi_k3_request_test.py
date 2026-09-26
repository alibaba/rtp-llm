"""K3 DashSc request tests using the public gRPC message shape."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.dash_sc.codec import (
    DashScRequestControls,
    SamplingParams,
    parse_sampling_params,
)
from rtp_llm.dash_sc.inference.request_adapter import (
    DashScRequestContext,
    _build_mm_inputs_from_request,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.models.kimi_k3.kimi_k3_dash_sc_adapter import (
    KimiK3DashScRequestAdapter,
    _normalize_k3_image_prompt,
    _resolve_k3_dash_sc_thinking,
)
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
)


class _K3Tokenizer:
    _markers = {
        "<|kimi_image_placeholder|>": [22, 11],
        "<|media_pad|>": [163605],
        "<|media_begin|>": [20],
        "<|media_end|>": [21],
        "<|open|>think<|sep|>": [10, 12, 13],
        "<|open|>response<|sep|>": [10, 11, 13],
        "<|close|>think<|sep|><|open|>response<|sep|>": [14, 12, 13, 10, 11, 13],
    }

    def encode(self, text):
        return self._markers[text]

    def decode(self, *_args, **_kwargs):
        raise AssertionError(
            "K3 request normalization must preserve ordinary token IDs"
        )


def _request_with_media(*parts):
    request = predict_v2_pb2.ModelInferRequest()
    request.parameters["payload"].string_param = json.dumps(
        {"input": {"messages": [{"role": "user", "content": list(parts)}]}}
    )
    return request


class KimiK3DashScThinkingTest(unittest.TestCase):
    def test_completion_limit_reaches_k3_contract_from_grpc_request(self) -> None:
        cases = (
            (None, None, 131072),
            ("max_completion_tokens", 128, 128),
            ("max_completion_tokens", 131072, 131072),
            ("max_new_tokens", 1048576, 131072),
        )
        for field, value, expected in cases:
            with self.subTest(field=field, value=value):
                request = predict_v2_pb2.ModelInferRequest()
                if field is not None:
                    request.parameters[field].int64_param = value
                sampling = parse_sampling_params(request)
                config = sampling.to_generate_config(
                    request_controls=DashScRequestControls()
                )
                apply_kimi_k3_request_contract(
                    config, specified_fields=sampling.specified_fields, thinking=True
                )
                self.assertEqual(config.max_new_tokens, expected)

    def test_structured_output_does_not_override_request_thinking(self) -> None:
        cases = (
            (SamplingParams(), True),
            (SamplingParams(response_format='{"type":"json_object"}'), True),
            (SamplingParams(json_format=True), True),
        )
        for sampling, expected in cases:
            with self.subTest(sampling=sampling):
                controls = DashScRequestControls()
                config = sampling.to_generate_config(request_controls=controls)
                self.assertEqual(
                    _resolve_k3_dash_sc_thinking(sampling, controls, config),
                    expected,
                )

    def test_explicit_mode_is_separate_from_zero_thinking_budget(self) -> None:
        sampling = SamplingParams(max_new_think_tokens=0)
        controls = DashScRequestControls(enable_thinking=True)
        config = sampling.to_generate_config(request_controls=controls)
        self.assertTrue(_resolve_k3_dash_sc_thinking(sampling, controls, config))

    def test_prompt_channel_resolves_missing_controls(self) -> None:
        sampling = SamplingParams(json_format=True)
        controls = DashScRequestControls()
        config = sampling.to_generate_config(request_controls=controls)
        self.assertTrue(_resolve_k3_dash_sc_thinking(sampling, controls, config, True))
        self.assertFalse(
            _resolve_k3_dash_sc_thinking(sampling, controls, config, False)
        )

    def test_unified_grammar_wins_over_response_format(self) -> None:
        grammar = {"type": "structural_tag", "format": {"type": "any_text"}}
        request = predict_v2_pb2.ModelInferRequest()
        request.parameters["unified_structural_tag"].string_param = json.dumps(grammar)
        request.parameters["response_format"].string_param = '{"type":"json_object"}'
        sampling = parse_sampling_params(request)
        self.assertTrue(sampling.has_unified_structural_tag)
        config = sampling.to_generate_config()
        self.assertIsNone(config.response_format)
        self.assertEqual(config.structural_tag, grammar)

    def test_legacy_grammar_does_not_need_scope(self) -> None:
        request = predict_v2_pb2.ModelInferRequest()
        request.parameters["tool_call_structural_tag"].string_param = json.dumps(
            {"type": "structural_tag", "format": {"type": "any_text"}}
        )
        sampling = parse_sampling_params(request)
        self.assertFalse(sampling.has_unified_structural_tag)
        self.assertIsNotNone(sampling.structural_tag)

    def test_adapter_installs_upstream_grammar_without_disabling_thinking(self) -> None:
        from xgrammar.builtin_structural_tag import get_model_structural_tag

        grammar = get_model_structural_tag("kimi_k3", reasoning="enabled").model_dump(
            mode="json", exclude_none=True
        )
        grammar["format"]["elements"][0]["content"]["max_tokens"] = 9
        request = predict_v2_pb2.ModelInferRequest()
        request.parameters["unified_structural_tag"].string_param = json.dumps(grammar)
        sampling = parse_sampling_params(request)
        controls = DashScRequestControls(enable_thinking=True, max_new_think_tokens=9)
        config = sampling.to_generate_config(request_controls=controls)
        prepared = KimiK3DashScRequestAdapter().prepare(
            DashScRequestContext(
                request=request,
                input_ids=[7, 10, 12, 13],
                sampling=sampling,
                controls=controls,
                generate_config=config,
                tokenizer=_K3Tokenizer(),
            )
        )
        config.add_thinking_params(
            _K3Tokenizer(),
            SimpleNamespace(
                think_mode="disabled",
                think_end_token_id=-1,
                think_end_tag="<|close|>think<|sep|>",
            ),
            enable_thinking=True,
            input_ids=prepared.input_ids,
        )
        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 9)
        self.assertEqual(config.structural_tag, grammar)

    def test_adapter_constrains_legacy_and_unified_grammar_the_same_way(self) -> None:
        grammar = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {
                        "type": "tag",
                        "begin": "",
                        "content": {"type": "any_text"},
                        "end": "<|close|>think<|sep|>",
                    },
                    {"type": "any_text"},
                ],
            },
        }
        plans = []
        for key in ("tool_call_structural_tag", "unified_structural_tag"):
            request = predict_v2_pb2.ModelInferRequest()
            request.parameters[key].string_param = json.dumps(grammar)
            sampling = parse_sampling_params(request)
            controls = DashScRequestControls(
                enable_thinking=True, max_new_think_tokens=1
            )
            config = sampling.to_generate_config(request_controls=controls)
            prepared = KimiK3DashScRequestAdapter().prepare(
                DashScRequestContext(
                    request,
                    [7, 10, 12, 13],
                    sampling,
                    controls,
                    config,
                    _K3Tokenizer(),
                    mm_inputs=[],
                    messages=[],
                )
            )
            final_constraint = config.finalize_response_format()
            elements = config.structural_tag["format"]["elements"]
            self.assertEqual(elements[0]["content"]["max_tokens"], 1)
            self.assertEqual(elements[1]["begin"], "<|open|>response<|sep|>")
            self.assertEqual(
                elements[1]["content"]["excludes"], ["<|open|>", "<|close|>"]
            )
            self.assertEqual(prepared.prompt_token_offset, 3)
            self.assertEqual(config.end_think_token_ids, [14, 12, 13, 10, 11, 13])
            self.assertEqual(final_constraint.value, grammar)
            plans.append(config.structural_tag)
        self.assertEqual(plans[0], plans[1])

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

    def test_adapter_reuses_preparsed_messages_and_media(self) -> None:
        sampling, controls = SamplingParams(), DashScRequestControls()
        media = [SimpleNamespace(mm_type=1)]
        context = DashScRequestContext(
            predict_v2_pb2.ModelInferRequest(),
            [22, 11, 10, 12, 13],
            sampling,
            controls,
            sampling.to_generate_config(),
            self.tokenizer,
            mm_inputs=media,
            messages=[],
        )
        with patch(
            "rtp_llm.models.kimi_k3.kimi_k3_dash_sc_adapter.parse_messages_from_request",
            side_effect=AssertionError("payload reparsed"),
        ), patch(
            "rtp_llm.models.kimi_k3.kimi_k3_dash_sc_adapter._build_mm_inputs_from_request",
            side_effect=AssertionError("media rebuilt"),
        ):
            prepared = KimiK3DashScRequestAdapter().prepare(context)
        self.assertIs(prepared.mm_inputs, media)
        self.assertEqual(prepared.input_ids, [163605, 10, 12, 13])
        self.assertEqual(prepared.prompt_token_offset, 3)

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
            mm_inputs = _build_mm_inputs_from_request(request)
            token_ids = _normalize_k3_image_prompt(input_ids, self.tokenizer, mm_inputs)

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

        mm_inputs = _build_mm_inputs_from_request(request)
        token_ids = _normalize_k3_image_prompt(
            [7, 20, 640, 480, 163605, 21, 8], self.tokenizer, mm_inputs
        )

        self.assertEqual(token_ids, [7, 163605, 8])
        self.assertEqual(len(mm_inputs), 1)

    def test_rejects_mismatched_image_count(self) -> None:
        request = _request_with_media(
            {"image": "https://example.com/a.jpg"},
            {"image": "https://example.com/b.jpg"},
        )

        with self.assertRaisesRegex(FtRuntimeException, "count does not match"):
            _normalize_k3_image_prompt(
                [7, 22, 11, 8],
                self.tokenizer,
                _build_mm_inputs_from_request(request),
            )

    def test_rejects_non_image_media(self) -> None:
        for media_type in ("video", "audio"):
            with self.subTest(media_type=media_type):
                request = _request_with_media({media_type: "https://example.com/media"})
                with self.assertRaisesRegex(FtRuntimeException, "only image"):
                    _normalize_k3_image_prompt(
                        [7],
                        self.tokenizer,
                        _build_mm_inputs_from_request(request),
                    )


if __name__ == "__main__":
    unittest.main()

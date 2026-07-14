"""Unit tests for ``rtp_llm.dash_sc.codec`` (request parsing + response builders)."""

from __future__ import annotations

import json
import struct
from unittest import TestCase, main

import torch

from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import (
    _DEFAULT_MAX_COMPLETION_TOKENS,
    _DEFAULT_MAX_NEW_TOKENS,
    _DEFAULT_MAX_THINKING_TOKENS,
    DashErrorSpec,
    DashScParameterError,
    LLMFinishReason,
    OtherParams,
    SamplingParams,
    build_dash_error_response,
    build_stream_response_from_generate_outputs,
    parse_dash_sc_grpc_request,
    parse_input_ids_from_request,
    parse_max_new_tokens_for_proxy,
    parse_other_params,
    parse_sampling_params,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.inference.servicer import stream_log_tag
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def _unpack_int64_le(raw: bytes) -> list[int]:
    return [int(x) for x in struct.unpack("<%dq" % (len(raw) // 8), raw)]


def _tool_call_structural_tag() -> dict:
    return {
        "format": {
            "type": "triggered_tags",
            "triggers": ["<｜DSML｜invoke"],
            "tags": [
                {
                    "type": "tag",
                    "begin": '<｜DSML｜invoke name="get_weather">',
                    "content": {
                        "type": "json_schema",
                        "json_schema": {"type": "object"},
                    },
                    "end": "</｜DSML｜invoke>",
                }
            ],
        }
    }


def _dashscope_sequence_tool_call_structural_tag() -> dict:
    return {
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "<｜DSML｜tool_calls>\n"},
                {
                    "type": "tags_with_separator",
                    "tags": [
                        {
                            "type": "tag",
                            "begin": '<｜DSML｜invoke name="get_weather">',
                            "content": {
                                "type": "json_schema",
                                "json_schema": {"type": "object"},
                            },
                            "end": "</｜DSML｜invoke>",
                        }
                    ],
                    "separator": "\n",
                    "at_least_one": True,
                    "stop_after_first": True,
                },
                {"type": "const_string", "value": "\n</｜DSML｜tool_calls>"},
            ],
        }
    }


def _dashscope_sequence_as_tag_structural_tag() -> dict:
    tag = _dashscope_sequence_tool_call_structural_tag()
    begin, content, end = tag["format"]["elements"]
    return {
        "format": {
            "type": "tag",
            "begin": begin["value"],
            "content": content,
            "end": end["value"],
        }
    }


def _add_tensor(
    req: predict_v2_pb2.ModelInferRequest,
    name: str,
    datatype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = datatype
    inp.shape[:] = shape
    req.raw_input_contents.append(raw)


class DashScGrpcRequestTest(TestCase):
    def test_parse_input_ids_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        raw = struct.pack("<3i", 10, 20, 30)
        _add_tensor(req, "input_ids", "INT32", [3], raw)
        self.assertEqual(parse_input_ids_from_request(req), [10, 20, 30])

    def test_parse_input_ids_int64(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        raw = struct.pack("<2q", 7, -1)
        _add_tensor(req, "input_ids", "INT64", [2], raw)
        self.assertEqual(parse_input_ids_from_request(req), [7, -1])

    def test_parse_input_ids_missing_tensor(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_input_ids_raw_index_mismatch(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        inp = req.inputs.add()
        inp.name = "input_ids"
        inp.datatype = "INT32"
        inp.shape.append(1)
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_input_ids_bad_length(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [2], b"\x01\x02\x03")
        self.assertIsNone(parse_input_ids_from_request(req))

    def test_parse_sampling_defaults(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        sp = parse_sampling_params(req)
        self.assertIsInstance(sp, SamplingParams)
        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.top_k, 0)
        self.assertEqual(sp.top_p, 1.0)
        self.assertEqual(sp.stop_words_list, ())

    def test_parse_sampling_scalars(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", 128))
        _add_tensor(req, "num_return_sequences", "INT32", [1], struct.pack("<i", 2))
        _add_tensor(req, "top_p", "FP32", [1], struct.pack("<f", 0.9))
        _add_tensor(req, "top_k", "INT32", [1], struct.pack("<i", 50))
        _add_tensor(req, "temperature", "FP32", [1], struct.pack("<f", 0.7))
        _add_tensor(req, "min_new_tokens", "INT32", [1], struct.pack("<i", 3))
        _add_tensor(req, "seed", "INT64", [1], struct.pack("<q", 999))
        _add_tensor(req, "repetition_penalty", "FP32", [1], struct.pack("<f", 1.1))
        _add_tensor(req, "frequency_penalty", "FP32", [1], struct.pack("<f", 0.2))
        _add_tensor(req, "presence_penalty", "FP32", [1], struct.pack("<f", 0.3))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_completion_tokens, 128)
        self.assertEqual(sp.num_return_sequences, 2)
        self.assertAlmostEqual(sp.top_p, 0.9)
        self.assertEqual(sp.top_k, 50)
        self.assertAlmostEqual(sp.temperature, 0.7)
        self.assertEqual(sp.min_new_tokens, 3)
        self.assertEqual(sp.random_seed, 999)
        self.assertAlmostEqual(sp.repetition_penalty, 1.1)
        self.assertAlmostEqual(sp.frequency_penalty, 0.2)
        self.assertAlmostEqual(sp.presence_penalty, 0.3)

    def test_parse_sampling_dashscope_aliases(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "n", "INT32", [1], struct.pack("<i", 1))
        _add_tensor(req, "min_length", "INT32", [1], struct.pack("<i", 2))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.num_return_sequences, 1)
        self.assertEqual(sp.min_new_tokens, 2)

    def test_parse_sampling_response_format_parameters(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )
        sp = parse_sampling_params(req)
        self.assertEqual(json.loads(sp.response_format), {"type": "json_object"})
        self.assertEqual(
            json.loads(sp.to_generate_config().response_format), {"type": "json_object"}
        )

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["json_format"].bool_param = True
        config = parse_sampling_params(req).to_generate_config()
        self.assertTrue(config.json_format)
        config.validate()
        self.assertEqual(config.json_schema, '{"type":"object"}')

    def test_build_model_infer_request_carries_response_format(self) -> None:
        req = build_model_infer_request(
            request_id="test",
            model_name="default",
            input_ids=[1, 2, 3],
            sampling=SamplingParams(
                max_new_tokens=16,
                response_format=json.dumps({"type": "json_object"}),
            ),
        )

        sp = parse_sampling_params(req)

        self.assertEqual(json.loads(sp.response_format), {"type": "json_object"})
        self.assertEqual(
            json.loads(sp.to_generate_config().response_format), {"type": "json_object"}
        )

    def test_build_model_infer_request_carries_structural_tag_response_format(
        self,
    ) -> None:
        response_format = {
            "type": "structural_tag",
            "format": _dashscope_sequence_tool_call_structural_tag()["format"],
        }
        normalized_response_format = {
            "type": "structural_tag",
            "format": _dashscope_sequence_as_tag_structural_tag()["format"],
        }
        req = build_model_infer_request(
            request_id="test",
            model_name="default",
            input_ids=[1, 2, 3],
            sampling=SamplingParams(
                max_new_tokens=16,
                response_format=json.dumps(response_format, ensure_ascii=False),
            ),
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self.assertIsNone(sp.response_format)
        self.assertIsNone(config.response_format)
        self.assertEqual(
            json.loads(sp.structural_tag),
            {"format": normalized_response_format["format"]},
        )
        self.assertEqual(
            json.loads(config.structural_tag),
            {"format": normalized_response_format["format"]},
        )

    def test_parse_sampling_response_format_array_compat(self) -> None:
        cases = [
            [{"type": "json_object"}, {"type": "json_schema", "json_schema": {}}],
            [json.dumps({"type": "json_object"})],
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = json.dumps(payload)
                sp = parse_sampling_params(req)
                self.assertEqual(
                    json.loads(sp.response_format), {"type": "json_object"}
                )

    def test_parse_sampling_bad_response_format_json_is_rejected(self) -> None:
        cases = [
            "not-json",
            json.dumps(["not-json"], ensure_ascii=False),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = payload

                with self.assertRaisesRegex(DashScParameterError, "response_format"):
                    parse_sampling_params(req)

    def _assert_guided_json_response_format(self, response_format, schema) -> None:
        self.assertEqual(
            json.loads(response_format),
            {"type": "json_schema", "json_schema": {"schema": schema}},
        )

    def test_parse_sampling_guided_json_list_sets_response_format_json_schema(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "人物": {"type": "array", "items": {"type": "string"}},
                "其他实体": {
                    "type": "array",
                    "items": {"type": "string"},
                    "enum": ["我国", "社会主义事业", "两个文明建设"],
                },
            },
            "required": ["人物", "其他实体"],
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["guided_json"].string_param = json.dumps(
            [schema], ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self._assert_guided_json_response_format(sp.response_format, schema)
        self._assert_guided_json_response_format(config.response_format, schema)
        self.assertIsNone(config.json_schema)

    def test_parse_sampling_guided_json_list_string_sets_response_format_json_schema(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"地点": {"type": "array", "items": {"type": "string"}}},
            "required": ["地点"],
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["guided_json"].string_param = json.dumps(
            [json.dumps(schema, ensure_ascii=False)], ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self._assert_guided_json_response_format(sp.response_format, schema)
        self._assert_guided_json_response_format(config.response_format, schema)
        self.assertIsNone(config.json_schema)

    def test_parse_sampling_guided_json_overrides_response_format(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}}
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "json_object"}
        )
        req.parameters["guided_json"].string_param = json.dumps([schema])

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self._assert_guided_json_response_format(sp.response_format, schema)
        self._assert_guided_json_response_format(config.response_format, schema)
        self.assertIsNone(config.json_schema)

    def test_parse_sampling_guided_json_from_nested_dash_parameters(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"parameters": {"guided_json": [schema]}}
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self._assert_guided_json_response_format(sp.response_format, schema)
        self._assert_guided_json_response_format(config.response_format, schema)
        self.assertIsNone(config.json_schema)

    def test_parse_sampling_response_format_from_nested_dash_parameters(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "parameters": {
                    "response_format": [{"type": "json_object"}],
                    "enable_thinking": False,
                }
            }
        )

        sp = parse_sampling_params(req)
        op = parse_other_params(req)

        self.assertEqual(json.loads(sp.response_format), {"type": "json_object"})
        self.assertIs(op.enable_thinking, False)

    def test_build_model_infer_request_preserves_json_controls(self) -> None:
        req = build_model_infer_request(
            request_id="r-json",
            model_name="m",
            input_ids=[1, 2, 3],
            sampling=SamplingParams(
                response_format='{"type":"json_object"}',
                json_format=True,
            ),
            enable_thinking=False,
        )

        sp = parse_sampling_params(req)
        op = parse_other_params(req)

        self.assertEqual(json.loads(sp.response_format), {"type": "json_object"})
        self.assertTrue(sp.json_format)
        self.assertIs(op.enable_thinking, False)

    def test_parse_sampling_tool_call_structural_tag_parameter(self) -> None:
        tag = _tool_call_structural_tag()
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self.assertEqual(json.loads(sp.structural_tag), tag)
        self.assertEqual(json.loads(config.structural_tag), tag)
        self.assertIsNone(config.response_format)
        self.assertFalse(config.json_format)

    def test_parse_sampling_normalizes_dashscope_sequence_structural_tag(self) -> None:
        tag = _dashscope_sequence_tool_call_structural_tag()
        normalized = _dashscope_sequence_as_tag_structural_tag()
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)

        self.assertEqual(json.loads(sp.structural_tag), normalized)
        self.assertEqual(json.loads(sp.to_generate_config().structural_tag), normalized)

    def test_parse_sampling_tool_call_structural_tag_unwraps_dashscope_list(
        self,
    ) -> None:
        tag = _tool_call_structural_tag()
        ignored = {
            "format": {
                "type": "triggered_tags",
                "triggers": ["<ignored>"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<ignored>",
                        "end": "</ignored>",
                    }
                ],
            }
        }
        cases = [
            {
                "tool_call_structural_tag": json.dumps(
                    [json.dumps(tag, ensure_ascii=False)], ensure_ascii=False
                )
            },
            {
                "tool_call_structural_tag": json.dumps(
                    [tag, ignored], ensure_ascii=False
                )
            },
            {
                "ds_header_attributes": json.dumps(
                    {
                        "parameters": {
                            "tool_call_structural_tag": [
                                json.dumps(tag, ensure_ascii=False)
                            ]
                        }
                    },
                    ensure_ascii=False,
                )
            },
        ]

        for params in cases:
            with self.subTest(params=tuple(params)):
                req = predict_v2_pb2.ModelInferRequest()
                for name, value in params.items():
                    req.parameters[name].string_param = value

                sp = parse_sampling_params(req)

                self.assertEqual(json.loads(sp.structural_tag), tag)

    def test_parse_sampling_empty_tool_call_structural_tag_list_is_ignored(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = "[]"

        sp = parse_sampling_params(req)

        self.assertIsNone(sp.structural_tag)

    def test_parse_sampling_bad_tool_call_structural_tag_json_is_rejected(
        self,
    ) -> None:
        cases = [
            "not-json",
            json.dumps(["not-json"], ensure_ascii=False),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = payload

                with self.assertRaisesRegex(
                    DashScParameterError, "tool_call_structural_tag"
                ):
                    parse_sampling_params(req)

    def test_parse_sampling_structural_tag_top_level_shape_validation(
        self,
    ) -> None:
        cases = [
            {},
            {"foo": "bar"},
            {"structures": []},
        ]
        for tag in cases:
            with self.subTest(tag=tag):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    tag, ensure_ascii=False
                )

                with self.assertRaisesRegex(
                    DashScParameterError, "invalid tool_call_structural_tag"
                ):
                    parse_sampling_params(req)

    def test_parse_sampling_structural_tag_alias_and_nested_dash_parameters(
        self,
    ) -> None:
        tag = {
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<tool_call>",
                        "content": {"type": "json_schema", "json_schema": {}},
                        "end": "</tool_call>",
                    }
                ],
            }
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"parameters": {"structural_tag": tag}}, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        self.assertEqual(json.loads(sp.structural_tag), tag)

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )
        sp = parse_sampling_params(req)
        self.assertEqual(json.loads(sp.to_generate_config().structural_tag), tag)

    def test_build_model_infer_request_carries_tool_call_structural_tag(self) -> None:
        tag = _tool_call_structural_tag()
        req = build_model_infer_request(
            request_id="r-structural-tag",
            model_name="m",
            input_ids=[1, 2, 3],
            sampling=SamplingParams(
                max_new_tokens=16,
                structural_tag=json.dumps(tag, ensure_ascii=False),
            ),
        )

        self.assertIn("tool_call_structural_tag", req.parameters)
        sp = parse_sampling_params(req)
        self.assertEqual(json.loads(sp.structural_tag), tag)
        self.assertEqual(json.loads(sp.to_generate_config().structural_tag), tag)

    def test_parse_sampling_max_completion_tokens_parameter_alias_wins(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["max_tokens"].int64_param = 200
        req.parameters["max_completion_tokens"].int64_param = 100
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_total_tokens, 200)
        self.assertEqual(sp.max_completion_tokens, 100)
        self.assertFalse(hasattr(sp, "max_new_tokens_from_completion_alias"))

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["max_tokens"].int64_param = 64
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_total_tokens, 64)

    def test_parse_sampling_top_p_as_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "top_p", "INT32", [1], struct.pack("<i", 1))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.top_p, 1.0)

    def test_downstream_proxy_max_new_tokens_negative_uses_completion_fallback(
        self,
    ) -> None:
        """The proxy keeps its own validation contract, while the inference
        boundary canonicalizes the historical wire alias exactly once."""
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        self.assertEqual(parse_max_new_tokens_for_proxy(req), (-1, False))

        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS)

    def test_downstream_proxy_completion_alias_preserves_independent_limits(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", 50))
        _add_tensor(req, "max_tokens", "INT32", [1], struct.pack("<i", 80))
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 11))

        sp = parse_sampling_params(req)
        gc = sp.to_generate_config()

        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_total_tokens, 80)
        self.assertEqual(sp.max_completion_tokens, 44)
        self.assertEqual(sp.max_new_think_tokens, 11)
        self.assertEqual(gc.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(gc.max_tokens, 80)
        self.assertEqual(gc.max_completion_tokens, 44)
        self.assertEqual(gc.max_thinking_tokens, 11)

    def test_downstream_proxy_completion_alias_uses_normalized_think_window(
        self,
    ) -> None:
        for completion, max_think, expected in (
            (50, 2, 53),
            (100, 2, 103),
            (100, 51, 54),
        ):
            with self.subTest(completion=completion, max_think=max_think):
                req = predict_v2_pb2.ModelInferRequest()
                _add_tensor(
                    req,
                    "max_new_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", completion),
                )
                _add_tensor(
                    req,
                    "max_new_think_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", max_think),
                )
                req.parameters["enable_thinking"].bool_param = True

                sp = parse_sampling_params(req)

                self.assertEqual(sp.max_completion_tokens, expected)
                self.assertEqual(
                    sp.to_generate_config().max_completion_tokens, expected
                )

    def test_downstream_proxy_completion_alias_ignores_disabled_think_window(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", 100))
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 51))
        req.parameters["enable_thinking"].bool_param = False

        sp = parse_sampling_params(req)

        self.assertEqual(sp.max_completion_tokens, 100)

    def test_explicit_completion_tensor_wins_over_downstream_wire_alias(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", 1))
        _add_tensor(req, "max_completion_tokens", "INT32", [1], struct.pack("<i", 100))

        sp = parse_sampling_params(req)

        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_completion_tokens, 100)

    def test_parse_sampling_openai_compat_max_new_tokens_negative_uses_default(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-envoy-original-path": "/compatible-mode/v1/chat/completions"}
        )
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        sp = parse_sampling_params(req)

        self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sp.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS)

    def test_non_positive_max_completion_is_normalized_once_for_backend(
        self,
    ) -> None:
        for value in (-2, -1, 0):
            with self.subTest(value=value):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["max_completion_tokens"].int64_param = value
                sp = parse_sampling_params(req)
                self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
                self.assertEqual(
                    sp.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS
                )
                self.assertEqual(
                    sp.to_generate_config().max_completion_tokens,
                    _DEFAULT_MAX_COMPLETION_TOKENS,
                )
                self.assertFalse(hasattr(sp, "raw_max_completion_tokens"))

    def test_backend_ignores_upstream_only_thinking_budget(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 1))
        req.parameters["thinking_budget"].int64_param = 50
        req.parameters["max_completion_tokens"].int64_param = -1
        req.parameters["enable_thinking"].bool_param = True

        _, sampling, other = parse_dash_sc_grpc_request(req)

        self.assertIsNotNone(sampling)
        self.assertIsNotNone(other)
        self.assertEqual(sampling.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS)
        self.assertEqual(
            sampling.to_generate_config(other=other).max_completion_tokens,
            _DEFAULT_MAX_COMPLETION_TOKENS,
        )
        self.assertIsNone(sampling.max_new_think_tokens)
        self.assertIsNone(other.max_new_think_tokens)
        self.assertFalse(hasattr(other, "thinking_budget"))
        self.assertTrue(other.enable_thinking)

    def test_non_positive_completion_uses_fallback(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 1))
        req.parameters["max_completion_tokens"].int64_param = 0

        _, sampling, _ = parse_dash_sc_grpc_request(req)

        self.assertIsNotNone(sampling)
        self.assertEqual(sampling.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS)
        self.assertEqual(
            sampling.to_generate_config().max_completion_tokens,
            _DEFAULT_MAX_COMPLETION_TOKENS,
        )

    def test_parse_sampling_max_completion_tokens_non_positive_blocks_legacy_aliases(
        self,
    ) -> None:
        for value in (-2, -1, 0):
            with self.subTest(value=value):
                req = predict_v2_pb2.ModelInferRequest()
                _add_tensor(
                    req,
                    "max_completion_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", value),
                )
                _add_tensor(
                    req,
                    "max_new_tokens",
                    "INT32",
                    [1],
                    struct.pack("<i", -1),
                )
                req.parameters["max_tokens"].int64_param = 1

                sp = parse_sampling_params(req)

                self.assertEqual(sp.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS)
                self.assertEqual(sp.max_total_tokens, 1)
                self.assertEqual(
                    sp.max_completion_tokens, _DEFAULT_MAX_COMPLETION_TOKENS
                )
                generate_config = sp.to_generate_config(
                    other=OtherParams(enable_thinking=True, max_new_think_tokens=50)
                )
                self.assertEqual(
                    generate_config.max_new_tokens, _DEFAULT_MAX_NEW_TOKENS
                )
                self.assertEqual(generate_config.max_tokens, 1)
                self.assertEqual(
                    generate_config.max_completion_tokens,
                    _DEFAULT_MAX_COMPLETION_TOKENS,
                )
                self.assertEqual(generate_config.max_thinking_tokens, 50)

    def test_completion_alias_thinking_budget_keeps_backend_limit(
        self,
    ) -> None:
        sampling = SamplingParams(
            max_new_tokens=100,
            max_completion_tokens=100,
        )
        other = OtherParams(enable_thinking=True, max_new_think_tokens=10)

        generate_config = sampling.to_generate_config(other=other)

        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertEqual(generate_config.max_completion_tokens, 100)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

    def test_completion_alias_thinking_budget_respects_max_tokens_cap(
        self,
    ) -> None:
        sampling = SamplingParams(
            max_new_tokens=100,
            max_total_tokens=105,
            max_completion_tokens=100,
        )
        other = OtherParams(enable_thinking=True, max_new_think_tokens=10)

        generate_config = sampling.to_generate_config(other=other)

        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertEqual(generate_config.max_tokens, 105)
        self.assertEqual(generate_config.max_completion_tokens, 100)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

    def test_explicit_max_new_tokens_thinking_budget_keeps_backend_limit(
        self,
    ) -> None:
        sampling = SamplingParams(max_new_tokens=100)
        other = OtherParams(enable_thinking=True, max_new_think_tokens=10)

        generate_config = sampling.to_generate_config(other=other)

        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

    def test_parse_sampling_max_new_think_tokens_zero(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 0))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_think_tokens, 0)
        # The servicer later turns zero budget into no-thinking while retaining
        # think boundary ids for C++ static masking.
        self.assertEqual(sp.to_generate_config().max_thinking_tokens, 0)

    def test_parse_sampling_max_think_length_priority_and_negative(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 0))
        _add_tensor(req, "max_think_length", "INT32", [1], struct.pack("<i", -1))
        sp = parse_sampling_params(req)
        # raw value stored; to_generate_config maps negative to the default budget.
        self.assertEqual(sp.max_new_think_tokens, -1)
        self.assertEqual(
            sp.to_generate_config().max_thinking_tokens,
            _DEFAULT_MAX_THINKING_TOKENS,
        )

    def test_build_request_writes_thinking_controls(self) -> None:
        req = build_model_infer_request(
            request_id="test",
            model_name="default",
            input_ids=[1, 2],
            sampling=SamplingParams(max_new_tokens=3, max_new_think_tokens=7),
            enable_thinking=False,
        )

        ids, sp, op = parse_dash_sc_grpc_request(req)

        self.assertEqual(ids, [1, 2])
        self.assertIsNotNone(sp)
        self.assertIsNotNone(op)
        self.assertEqual(sp.max_new_think_tokens, 7)
        self.assertIs(op.enable_thinking, False)

    def test_parse_sampling_legacy_top_k_parameter(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        p = predict_v2_pb2.InferParameter()
        p.int64_param = 88
        req.parameters["top_k"].CopyFrom(p)
        sp = parse_sampling_params(req)
        self.assertEqual(sp.top_k, 88)

    def test_parse_sampling_stop_words_list_2d(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        flat = [1, 2, 3, 4]
        raw = struct.pack("<%di" % len(flat), *flat)
        _add_tensor(req, "stop_words_list", "INT32", [2, 2], raw)
        sp = parse_sampling_params(req)
        self.assertEqual(sp.stop_words_list, ((1, 2), (3, 4)))
        self.assertEqual(sp.stop_words_list_py(), [[1, 2], [3, 4]])

    def test_sampling_params_n_alias(self) -> None:
        sp = SamplingParams(num_return_sequences=5)
        self.assertEqual(sp.n, 5)

    def test_parse_other_params_default(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        op = parse_other_params(req)
        self.assertEqual(op, OtherParams(return_input_ids=False))

    def test_parse_other_params_bool(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "return_input_ids", "BOOL", [1], b"\x01")
        op = parse_other_params(req)
        self.assertTrue(op.return_input_ids)

    def test_parse_other_params_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "return_input_ids", "INT32", [1], struct.pack("<i", 0))
        op = parse_other_params(req)
        self.assertFalse(op.return_input_ids)

    def test_parse_other_params_thinking_controls(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "x-ds-llm-thinking": "false",
                "x-dashscope-inner-timeout": 1800,
                "x-ds-request-priority": "10",
                "user_id": "u1",
                "x-dashscope-apikeyid": "ak1",
            }
        )
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 0))
        op = parse_other_params(req)
        self.assertFalse(op.return_input_ids)
        self.assertIs(op.enable_thinking, False)
        self.assertEqual(op.max_new_think_tokens, 0)
        self.assertEqual(op.timeout_ms, 1_800_000)
        self.assertEqual(op.traffic_reject_priority, 10)
        self.assertEqual(
            op.request_headers, {"user_id": "u1", "x-dashscope-apikeyid": "ak1"}
        )

    def test_parse_other_params_reasoning_effort_max_alias(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["reasoning_effort"].string_param = "max"
        op = parse_other_params(req)
        self.assertEqual(op.reasoning_effort, "xhigh")

    def test_parse_other_params_reasoning_effort_from_dashscope_attrs(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"parameters": {"reasoning_effort": {"effort": "max"}}}
        )
        op = parse_other_params(req)
        self.assertEqual(op.reasoning_effort, "xhigh")

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"body": {"reasoning_effort": "max"}}
        )
        op = parse_other_params(req)
        self.assertEqual(op.reasoning_effort, "xhigh")

    def test_backend_only_consumes_proxy_normalized_think_budget(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 1))
        req.parameters["enable_thinking"].bool_param = True
        req.parameters["thinking_budget"].int64_param = 100
        req.parameters["max_new_think_tokens"].int64_param = 101
        _, sampling, other = parse_dash_sc_grpc_request(req)

        self.assertIsNotNone(sampling)
        self.assertIsNotNone(other)
        assert sampling is not None and other is not None
        self.assertIs(other.enable_thinking, True)
        self.assertEqual(other.max_new_think_tokens, 101)
        self.assertIsNone(sampling.max_new_think_tokens)
        self.assertFalse(hasattr(other, "thinking_budget"))

    def test_parse_dash_sc_grpc_request_ok(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        _add_tensor(req, "top_k", "INT32", [1], struct.pack("<i", 10))
        _add_tensor(req, "return_input_ids", "BOOL", [1], b"\x01")
        ids, sp, op = parse_dash_sc_grpc_request(req)
        self.assertEqual(ids, [1, 2])
        self.assertIsNotNone(sp)
        self.assertIsNotNone(op)
        assert sp is not None and op is not None
        self.assertEqual(sp.top_k, 10)
        self.assertTrue(op.return_input_ids)

    def test_parse_dash_sc_grpc_request_no_input_ids(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        ids, sp, op = parse_dash_sc_grpc_request(req)
        self.assertIsNone(ids)
        self.assertIsNone(sp)
        self.assertIsNone(op)

    def test_sampling_to_generate_config(self) -> None:
        sp = SamplingParams(
            max_new_tokens=64,
            top_k=1,
            max_new_think_tokens=128,
            stop_words_list=((42,),),
        )
        gc = sp.to_generate_config(other=OtherParams(return_input_ids=True))
        self.assertEqual(gc.max_new_tokens, 64)
        self.assertEqual(gc.top_k, 1)
        self.assertEqual(gc.max_thinking_tokens, 128)
        self.assertEqual(gc.stop_words_list, [[42]])
        self.assertTrue(gc.return_input_ids)


class BuildStreamResponseFromGenerateOutputsTest(TestCase):
    def test_empty_generate_outputs_raises(self) -> None:
        go = GenerateOutputs(generate_outputs=[])
        with self.assertRaises(ValueError) as ctx:
            build_stream_response_from_generate_outputs(
                dash_sc_request_id="r1",
                model_name="m",
                go=go,
                request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r1"),
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_basic_generated_ids_finish_aux(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8, 9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=10, reuse_len=4),
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="req-a",
            model_name="mdl",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=99, trace_id="req-a"),
            return_input_ids=False,
        )
        self.assertFalse(resp.error_message)
        infer = resp.infer_response
        self.assertEqual(infer.id, "req-a")
        self.assertEqual(infer.model_name, "mdl")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [7, 8, 9])
        self.assertEqual(list(infer.outputs[0].shape), [1, 3])
        self.assertEqual(
            _unpack_int64_le(by_name["finish_reason"]),
            [LLMFinishReason.STOP],
        )
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_num"]), [10])
        self.assertEqual(_unpack_int32_le(by_name["prompt_cached_token_num"]), [4])
        self.assertIn("prompt_token_num", infer.parameters)
        self.assertIn("prompt_cached_token_num", infer.parameters)
        self.assertTrue(infer.parameters["prompt_token_num"].HasField("int64_param"))
        self.assertTrue(
            infer.parameters["prompt_cached_token_num"].HasField("int64_param")
        )
        self.assertEqual(
            infer.parameters["prompt_cached_token_num"].int64_param,
            4,
        )
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 10)

    def test_dash_error_response_uses_inner_error_fields(self) -> None:
        resp = build_dash_error_response(
            "req-error",
            "glm",
            error_spec=DashErrorSpec(
                error_no=8,
                finish_reason=LLMFinishReason.STOP_ENGINE_PARAM,
                status_code=400,
                status_name="InvalidParameter",
            ),
            status_message="invalid\nmax_new_tokens",
        )

        self.assertFalse(resp.error_message)
        infer = resp.infer_response
        self.assertEqual(infer.id, "req-error")
        self.assertEqual(infer.model_name, "glm")
        self.assertEqual(infer.parameters["error_no"].int64_param, 8)
        payload = json.loads(infer.parameters["error_msg"].string_param)
        self.assertEqual(payload["status_code"], 400)
        self.assertIsInstance(payload["status_code"], int)
        self.assertEqual(payload["status_name"], "InvalidParameter")
        self.assertIn("max_new_tokens", payload["status_message"])
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(
            _unpack_int64_le(by_name["finish_reason"]),
            [LLMFinishReason.STOP_ENGINE_PARAM],
        )
        self.assertEqual(by_name["finished"], b"\x01")
        self.assertNotIn("generated_ids", by_name)
        self.assertNotIn("token_ids", by_name)

    def test_dash_error_status_code_is_json_number(self) -> None:
        for status_code in (400, 413, 422, 500, 503, 504):
            with self.subTest(status_code=status_code):
                resp = build_dash_error_response(
                    "req-error",
                    "glm",
                    error_spec=DashErrorSpec(
                        error_no=8,
                        finish_reason=LLMFinishReason.STOP_ENGINE_PARAM,
                        status_code=status_code,
                        status_name="InvalidParameter",
                    ),
                    status_message="invalid max_new_tokens",
                )
                error_msg = resp.infer_response.parameters["error_msg"].string_param

                self.assertIn(f'"status_code":{status_code}', error_msg)
                self.assertNotIn(f'"status_code":"{status_code}"', error_msg)
                self.assertEqual(json.loads(error_msg)["status_code"], status_code)

    def test_finish_reason_length_override_repro_p1(self) -> None:
        """P1 repro: when generation finishes because ``max_new_tokens`` was
        reached, the wire protocol currently has no way to signal 'length' —
        ``finished=True`` always maps to 0 (stop), so dashscope-serving
        collapses every cutoff into ``finish_reason='stop'``.

        Expected fix: codec exposes ``LLMFinishReason.LENGTH = 1`` and
        ``build_stream_response_from_generate_outputs`` takes a
        ``finish_reason_override`` argument the caller can set when the
        cumulative output reaches the per-request budget."""
        out = GenerateOutput(
            output_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=4, reuse_len=0, output_len=3),
        )
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
            finish_reason_override=LLMFinishReason.LENGTH,
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(
            _unpack_int64_le(by_name["finish_reason"]),
            [LLMFinishReason.LENGTH],
        )

    def test_not_finished_finish_reason_two(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([1], dtype=torch.int32),
            finished=False,
            aux_info=None,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=0, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(
            _unpack_int64_le(by_name["finish_reason"]),
            [LLMFinishReason.STREAMING],
        )
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_num"]), [0])
        self.assertEqual(_unpack_int32_le(by_name["prompt_cached_token_num"]), [0])
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 0)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 0)

    def test_missing_aux_info_uses_request_prompt_len_for_usage(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([1], dtype=torch.int32),
            finished=False,
            aux_info=None,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=0, trace_id="r"),
            request_input_ids=[10, 11, 12],
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_num"]), [3])
        self.assertEqual(_unpack_int32_le(by_name["prompt_cached_token_num"]), [0])
        self.assertEqual(infer.parameters["prompt_token_num"].int64_param, 3)
        self.assertEqual(infer.parameters["prompt_cached_token_num"].int64_param, 0)

    def test_output_ids_2d_uses_first_row(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([[100, 101]], dtype=torch.int32),
            finished=True,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [100, 101])

    def test_return_input_ids_prepends_prompt_tensor(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([5], dtype=torch.int32),
            finished=True,
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
            request_input_ids=[1, 2, 3],
            return_input_ids=True,
        )
        infer = resp.infer_response
        names = [infer.outputs[i].name for i in range(len(infer.outputs))]
        self.assertEqual(
            names,
            [
                "prompt_token_ids",
                "generated_ids",
                "finish_reason",
                "finished",
                "prompt_token_num",
                "prompt_cached_token_num",
            ],
        )
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["prompt_token_ids"]), [1, 2, 3])
        self.assertEqual(list(infer.outputs[0].shape), [1, 3])

    def test_missing_output_ids_empty_generated(self) -> None:
        out = GenerateOutput(output_ids=None, finished=False, aux_info=AuxInfo())
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(by_name["generated_ids"], struct.pack("<i", 0))
        self.assertEqual(list(infer.outputs[0].shape), [1, 0])


class PrependToGeneratedIdsTensorTest(TestCase):
    def _build_infer_with_generated_ids(self, ids: list[int]):
        out = GenerateOutput(
            output_ids=torch.tensor(ids, dtype=torch.int32) if ids else None,
            finished=True,
            aux_info=AuxInfo(input_len=0, reuse_len=0),
        )
        go = GenerateOutputs(generate_outputs=[out])
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=go,
            request_log_tag=stream_log_tag(request_id_numeric=1, trace_id="r"),
        )
        return resp.infer_response

    def test_prepend_list_success(self) -> None:
        infer = self._build_infer_with_generated_ids([7, 8, 9])
        self.assertTrue(prepend_to_generated_ids_tensor(infer, [100, 200]))
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(
            _unpack_int32_le(by_name["generated_ids"]), [100, 200, 7, 8, 9]
        )
        gen_out = next(o for o in infer.outputs if o.name == "generated_ids")
        self.assertEqual(list(gen_out.shape), [1, 5])

    def test_prepend_empty_list_noop(self) -> None:
        infer = self._build_infer_with_generated_ids([7, 8, 9])
        self.assertFalse(prepend_to_generated_ids_tensor(infer, []))
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [7, 8, 9])

    def test_prepend_to_empty_tensor_noop(self) -> None:
        infer = self._build_infer_with_generated_ids([])
        self.assertFalse(prepend_to_generated_ids_tensor(infer, [100]))
        gen_out = next(o for o in infer.outputs if o.name == "generated_ids")
        self.assertEqual(list(gen_out.shape), [1, 0])

    def test_prepend_without_generated_ids_output_returns_false(self) -> None:
        resp = predict_v2_pb2.ModelStreamInferResponse()
        self.assertFalse(prepend_to_generated_ids_tensor(resp.infer_response, [100]))


class StreamLogTagTest(TestCase):
    def test_stream_log_tag_format(self) -> None:
        self.assertEqual(
            stream_log_tag(request_id_numeric=-1, trace_id="tid"),
            "request_id=-1 trace_id=tid",
        )


if __name__ == "__main__":
    main()

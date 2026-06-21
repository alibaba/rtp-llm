"""Unit tests for ``rtp_llm.dash_sc.codec`` (request parsing + response builders)."""

from __future__ import annotations

import json
import struct
from unittest import TestCase, main

import torch

from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import (
    FINISH_REASON_USE_PARAMETER_STATUS,
    DashScParameterError,
    MultimodalPart,
    OtherParams,
    SamplingParams,
    build_error_response,
    build_stream_response_from_generate_outputs,
    parse_dash_sc_grpc_request,
    parse_input_ids_from_request,
    parse_multimodal_parts_from_request,
    parse_other_params,
    parse_sampling_params,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.inference.servicer import stream_log_tag
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    MMUrlType,
)


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
        self.assertEqual(sp.max_new_tokens, 32000)
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
        self.assertEqual(sp.max_new_tokens, 128)
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

    def _set_force_at_least_one_env(self, value: str | None) -> None:
        """Idempotent env toggle with auto-cleanup so subTest variants stay isolated."""
        import os
        from rtp_llm.dash_sc.codec import _FORCE_AT_LEAST_ONE_ENV_KEY

        prev = os.environ.get(_FORCE_AT_LEAST_ONE_ENV_KEY)
        if value is None:
            os.environ.pop(_FORCE_AT_LEAST_ONE_ENV_KEY, None)
        else:
            os.environ[_FORCE_AT_LEAST_ONE_ENV_KEY] = value
        self.addCleanup(
            lambda: (
                os.environ.__setitem__(_FORCE_AT_LEAST_ONE_ENV_KEY, prev)
                if prev is not None
                else os.environ.pop(_FORCE_AT_LEAST_ONE_ENV_KEY, None)
            )
        )

    def test_force_at_least_one_env_off_leaves_structural_tag_untouched(self) -> None:
        """Env required: structural_tag set but env off → no force."""
        self._set_force_at_least_one_env(None)
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
                "at_least_one": False,
            }
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        self.assertEqual(
            json.loads(sp.structural_tag)["format"]["at_least_one"], False
        )

    def test_force_at_least_one_no_op_on_format_without_at_least_one_slot(
        self,
    ) -> None:
        """Env on + structural_tag is plain json_schema → force_at_least_one
        runs but mechanically no-ops (json_schema has no ``at_least_one``
        slot). Regression guard: the walker must NOT inject the field on
        unrelated formats."""
        self._set_force_at_least_one_env("1")
        tag = {
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
            }
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        self.assertEqual(json.loads(sp.structural_tag), tag)

    def test_force_at_least_one_env_on_overrides_triggered_tags(self) -> None:
        self._set_force_at_least_one_env("1")
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
                "at_least_one": False,
                "stop_after_first": False,
            }
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        result_format = json.loads(sp.structural_tag)["format"]
        self.assertTrue(result_format["at_least_one"])
        self.assertFalse(result_format["stop_after_first"])
        # to_generate_config must reflect the override too (string passthrough).
        cfg_format = json.loads(sp.to_generate_config().structural_tag)["format"]
        self.assertTrue(cfg_format["at_least_one"])

    def test_force_at_least_one_env_on_overrides_response_format_path(self) -> None:
        self._set_force_at_least_one_env("true")
        response_format = {
            "type": "structural_tag",
            "structural_tag": {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<tool_call>"],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": "<tool_call>get_current_weather",
                            "content": {
                                "type": "json_schema",
                                "json_schema": {"type": "object"},
                                "style": "glm_xml",
                            },
                            "end": "</tool_call>",
                        }
                    ],
                    "at_least_one": False,
                    "stop_after_first": False,
                    "excludes": ["<think>", "</think>"],
                },
            },
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["response_format"].string_param = json.dumps(
            response_format, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        self.assertIsNone(sp.response_format)
        result_format = json.loads(sp.structural_tag)["format"]
        self.assertTrue(result_format["at_least_one"])

    def test_force_at_least_one_env_on_overrides_nested_tags_with_separator(
        self,
    ) -> None:
        self._set_force_at_least_one_env("yes")
        tag = _dashscope_sequence_tool_call_structural_tag()
        # dashscope-sequence wrapper -> tag + tags_with_separator after adapt;
        # at_least_one lives at format.content.at_least_one.
        tag["format"]["elements"][1]["at_least_one"] = False
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            tag, ensure_ascii=False
        )

        sp = parse_sampling_params(req)
        adapted = json.loads(sp.structural_tag)
        self.assertEqual(adapted["format"]["type"], "tag")
        self.assertTrue(adapted["format"]["content"]["at_least_one"])

    def test_parse_sampling_max_completion_tokens_parameter_alias_wins(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["max_tokens"].int64_param = 200
        req.parameters["max_completion_tokens"].int64_param = 100
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, 100)
        self.assertTrue(sp.max_new_tokens_from_completion_alias)
        self.assertEqual(sp.max_total_tokens, 200)

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["max_tokens"].int64_param = 64
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, 64)
        self.assertFalse(sp.max_new_tokens_from_completion_alias)

    def test_parse_sampling_top_p_as_int32(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "top_p", "INT32", [1], struct.pack("<i", 1))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.top_p, 1.0)

    def test_parse_sampling_max_new_tokens_negative_keeps_signal_repro_p3(
        self,
    ) -> None:
        """P3 repro: a negative ``max_new_tokens`` is silently clamped to 0,
        which the backend later rejects with ``FtRuntimeException`` (HTTP 500).
        Expected: codec preserves the signed value so the caller can decide
        between rejecting up-front or substituting a server default."""
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))
        sp = parse_sampling_params(req)
        self.assertEqual(sp.max_new_tokens, -1)

    def test_parse_sampling_openai_compat_max_new_tokens_negative_uses_default(
        self,
    ) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-envoy-original-path": "/compatible-mode/v1/chat/completions"}
        )
        _add_tensor(req, "max_new_tokens", "INT32", [1], struct.pack("<i", -1))

        sp = parse_sampling_params(req)

        self.assertEqual(sp.max_new_tokens, 32000)
        self.assertFalse(sp.max_new_tokens_from_completion_alias)

    def test_parse_sampling_max_completion_tokens_non_positive_preserves_error_repro(
        self,
    ) -> None:
        """Non-positive max_completion_tokens must be rejected before enqueue."""
        for value in (-1, 0):
            with self.subTest(value=value):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["max_completion_tokens"].int64_param = value
                sp = parse_sampling_params(req)
                self.assertEqual(sp.max_new_tokens, value)
                self.assertTrue(sp.max_new_tokens_from_completion_alias)

    def test_parse_sampling_max_completion_tokens_non_positive_blocks_legacy_aliases(
        self,
    ) -> None:
        for value in (-1, 0):
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

                sp = parse_sampling_params(req)

                self.assertEqual(sp.max_new_tokens, value)
                self.assertTrue(sp.max_new_tokens_from_completion_alias)

    def test_completion_alias_thinking_budget_keeps_backend_limit(
        self,
    ) -> None:
        sampling = SamplingParams(
            max_new_tokens=100,
            max_new_tokens_from_completion_alias=True,
        )
        other = OtherParams(enable_thinking=True, max_new_think_tokens=10)

        generate_config = sampling.to_generate_config(other=other)

        self.assertEqual(generate_config.max_new_tokens, 100)
        self.assertEqual(generate_config.max_thinking_tokens, 10)

    def test_completion_alias_thinking_budget_respects_max_tokens_cap(
        self,
    ) -> None:
        sampling = SamplingParams(
            max_new_tokens=100,
            max_new_tokens_from_completion_alias=True,
            max_total_tokens=105,
        )
        other = OtherParams(enable_thinking=True, max_new_think_tokens=10)

        generate_config = sampling.to_generate_config(other=other)

        self.assertEqual(generate_config.max_new_tokens, 100)
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
        # raw value stored; to_generate_config maps negative → INT32_MAX
        self.assertEqual(sp.max_new_think_tokens, -1)
        self.assertEqual(sp.to_generate_config().max_thinking_tokens, 2_147_483_647)

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

    def test_parse_other_params_thinking_mode(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["thinking_mode"].string_param = "adaptive"
        self.assertEqual(parse_other_params(req).thinking_mode, "adaptive")

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"x-ds-llm-thinking-mode": "enabled"}
        )
        self.assertEqual(parse_other_params(req).thinking_mode, "enabled")

        req.parameters["thinking_mode"].string_param = "invalid"
        self.assertIsNone(parse_other_params(req).thinking_mode)

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

    def test_parse_other_params_dashscope_body_thinking_aliases(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["enable_thinking"].bool_param = False
        req.parameters["thinking_budget"].int64_param = 100
        op = parse_other_params(req)
        self.assertIs(op.enable_thinking, False)
        self.assertEqual(op.max_new_think_tokens, 100)

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

    @staticmethod
    def _set_payload(
        req: predict_v2_pb2.ModelInferRequest,
        payload_obj: object,
        key: str = "payload",
    ) -> None:
        req.parameters[key].string_param = json.dumps(payload_obj)

    # ------------------------------------------------------------------
    # OpenAI shape (gpt3_serving build_payload output)
    # ------------------------------------------------------------------

    def test_parse_multimodal_parts_image_video_audio(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://x.png"},
                                },
                                {
                                    "type": "video_url",
                                    "video_url": {"url": "http://y.mp4"},
                                },
                                {
                                    "type": "audio_url",
                                    "audio_url": {"url": "http://z.wav"},
                                },
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [
                MultimodalPart(url="http://x.png", mm_type=MMUrlType.IMAGE),
                MultimodalPart(url="http://y.mp4", mm_type=MMUrlType.VIDEO),
                MultimodalPart(url="http://z.wav", mm_type=MMUrlType.AUDIO),
            ],
        )

    def test_parse_multimodal_parts_skips_text(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe this"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://a.png"},
                                },
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://a.png", mm_type=MMUrlType.IMAGE)],
        )

    def test_parse_multimodal_parts_missing_payload(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        self.assertEqual(parse_multimodal_parts_from_request(req), [])

    def test_parse_multimodal_parts_invalid_json(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["payload"].string_param = "not json"
        # Fail-open: returns [] instead of raising.
        self.assertEqual(parse_multimodal_parts_from_request(req), [])

    def test_parse_multimodal_parts_url_as_string(self) -> None:
        # Defensive: hand-built clients may pass image_url as plain string.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": "http://b.jpg"}
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://b.jpg", mm_type=MMUrlType.IMAGE)],
        )

    def test_parse_multimodal_parts_top_level_messages(self) -> None:
        # Tolerate clients that send {'messages': [...]} without the 'input' wrapper.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "http://c.png"}}
                        ],
                    }
                ]
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://c.png", mm_type=MMUrlType.IMAGE)],
        )

    def test_parse_multimodal_parts_bare_messages_list(self) -> None:
        # dashllm __messages__ path: bare list without dict wrapper.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://d.png"}}
                    ],
                }
            ],
            key="__messages__",
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://d.png", mm_type=MMUrlType.IMAGE)],
        )

    # ------------------------------------------------------------------
    # Dashscope native shape (multimodal_serving / MMGPT3Item / ocr/*.json)
    # ------------------------------------------------------------------

    def test_parse_multimodal_parts_native_image_video_audio(self) -> None:
        # Native shape: no ``type`` field, modality keyed directly.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": "http://x.png"},
                                {"video": "http://y.mp4"},
                                {"audio": "http://z.wav"},
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [
                MultimodalPart(url="http://x.png", mm_type=MMUrlType.IMAGE),
                MultimodalPart(url="http://y.mp4", mm_type=MMUrlType.VIDEO),
                MultimodalPart(url="http://z.wav", mm_type=MMUrlType.AUDIO),
            ],
        )

    def test_parse_multimodal_parts_native_with_inline_config(self) -> None:
        # ocr/request_for_general.json: per-part min_pixels/max_pixels inline.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": "http://ocr.jpg",
                                    "min_pixels": 3136,
                                    "max_pixels": 6422528,
                                    "enable_rotate": False,
                                },
                                {"text": "describe"},
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [
                MultimodalPart(
                    url="http://ocr.jpg",
                    mm_type=MMUrlType.IMAGE,
                    min_pixels=3136,
                    max_pixels=6422528,
                )
            ],
        )

    def test_parse_multimodal_parts_native_video_as_list_flattens(self) -> None:
        # MMGPT3Item.video can be List[str] (frame sequence); each frame becomes
        # its own VIDEO MultimodalPart so the engine doesn't drop frames.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "video": [
                                        "http://f1.jpg",
                                        "http://f2.jpg",
                                        "http://f3.jpg",
                                    ],
                                    "fps": 2,
                                    "max_frames": 32,
                                },
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [
                MultimodalPart(
                    url="http://f1.jpg",
                    mm_type=MMUrlType.VIDEO,
                    fps=2,
                    max_frames=32,
                ),
                MultimodalPart(
                    url="http://f2.jpg",
                    mm_type=MMUrlType.VIDEO,
                    fps=2,
                    max_frames=32,
                ),
                MultimodalPart(
                    url="http://f3.jpg",
                    mm_type=MMUrlType.VIDEO,
                    fps=2,
                    max_frames=32,
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Wrapping & alternative parameter key
    # ------------------------------------------------------------------

    def test_parse_multimodal_parts_full_http_body_wrapping(self) -> None:
        # Canonical dashscope HTTP body shape (see ocr/request_for_general.json):
        # header sibling + payload wrapper around input.messages.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "header": {"request_id": "ocr_test"},
                "payload": {
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"image": "http://full.png"},
                                    {"text": "describe"},
                                ],
                            }
                        ]
                    },
                    "parameters": {"max_tokens": 2000},
                },
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://full.png", mm_type=MMUrlType.IMAGE)],
        )

    def test_parse_multimodal_parts_double_underscore_messages_key(self) -> None:
        # multimodal_serving/server/dserv_stream_worker_for_vl.py reads
        # parameters['__messages__'] instead of parameters['payload'].
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"image": "http://dserv.png"}],
                        }
                    ]
                }
            },
            key="__messages__",
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://dserv.png", mm_type=MMUrlType.IMAGE)],
        )

    def test_parse_multimodal_parts_payload_key_wins_over_messages(self) -> None:
        # If both keys are present, ``payload`` (gpt3_serving outbound) wins;
        # this matches _MULTIMODAL_PARAMETER_KEYS priority order.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "messages": [
                    {"role": "user", "content": [{"image": "http://from-payload.png"}]}
                ]
            },
            key="payload",
        )
        self._set_payload(
            req,
            {
                "messages": [
                    {"role": "user", "content": [{"image": "http://from-double.png"}]}
                ]
            },
            key="__messages__",
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [MultimodalPart(url="http://from-payload.png", mm_type=MMUrlType.IMAGE)],
        )

    # ------------------------------------------------------------------
    # Per-part config edge cases
    # ------------------------------------------------------------------

    def test_parse_multimodal_parts_nested_preprocess_config(self) -> None:
        # RTP-LLM's OpenAI ContentPart.preprocess_config nesting works too.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://nested.png"},
                                    "preprocess_config": {
                                        "min_pixels": 100,
                                        "max_pixels": 200,
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(
            parts,
            [
                MultimodalPart(
                    url="http://nested.png",
                    mm_type=MMUrlType.IMAGE,
                    min_pixels=100,
                    max_pixels=200,
                )
            ],
        )

    def test_parse_multimodal_parts_inline_overrides_nested(self) -> None:
        # When both nested ``preprocess_config`` and inline keys are set, inline wins.
        req = predict_v2_pb2.ModelInferRequest()
        self._set_payload(
            req,
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "image": "http://both.png",
                                    "preprocess_config": {"min_pixels": 100},
                                    "min_pixels": 9999,
                                }
                            ],
                        }
                    ]
                }
            },
        )
        parts = parse_multimodal_parts_from_request(req)
        self.assertEqual(parts[0].min_pixels, 9999)


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
        self.assertEqual(_unpack_int64_le(by_name["finish_reason"]), [0])
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

    def test_error_response_uses_business_status_frame(self) -> None:
        resp = build_error_response(
            "req-error",
            "invalid max_new_tokens: 0; must be greater than 0",
            status_code=400,
            status_name="InvalidParameter",
        )

        self.assertFalse(resp.error_message)
        infer = resp.infer_response
        self.assertEqual(infer.id, "req-error")
        self.assertEqual(infer.parameters["status_code"].int64_param, 400)
        self.assertEqual(
            infer.parameters["status_name"].string_param,
            "InvalidParameter",
        )
        self.assertIn(
            "max_new_tokens",
            infer.parameters["status_message"].string_param,
        )
        payload = json.loads(infer.parameters["__messages__"].string_param)
        self.assertEqual(payload["header"]["status_code"], 400)
        self.assertEqual(payload["header"]["status_name"], "InvalidParameter")
        self.assertTrue(payload["header"]["finished"])
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        generated = next(out for out in infer.outputs if out.name == "generated_ids")
        self.assertEqual(list(generated.shape), [1, 0])
        self.assertEqual(by_name["generated_ids"], struct.pack("<i", 0))
        self.assertEqual(
            _unpack_int64_le(by_name["finish_reason"]),
            [FINISH_REASON_USE_PARAMETER_STATUS],
        )

    def test_finish_reason_length_override_repro_p1(self) -> None:
        """P1 repro: when generation finishes because ``max_new_tokens`` was
        reached, the wire protocol currently has no way to signal 'length' —
        ``finished=True`` always maps to 0 (stop), so dashscope-serving
        collapses every cutoff into ``finish_reason='stop'``.

        Expected fix: codec exposes ``FINISH_REASON_LENGTH = 1`` and
        ``build_stream_response_from_generate_outputs`` takes a
        ``finish_reason_override`` argument the caller can set when the
        cumulative output reaches the per-request budget."""
        from rtp_llm.dash_sc.codec import FINISH_REASON_LENGTH

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
            finish_reason_override=FINISH_REASON_LENGTH,
        )
        infer = resp.infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int64_le(by_name["finish_reason"]), [1])

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
        self.assertEqual(_unpack_int64_le(by_name["finish_reason"]), [2])
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

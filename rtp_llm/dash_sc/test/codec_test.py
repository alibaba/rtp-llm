"""Unit tests for ``rtp_llm.dash_sc.codec`` (request parsing + response builders)."""

from __future__ import annotations

import json
import math
import struct
from unittest import TestCase, main

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import (
    FINISH_REASON_USE_PARAMETER_STATUS,
    DashScParameterError,
    OtherParams,
    SamplingParams,
    build_error_response,
    build_stream_response_from_generate_outputs,
    parse_dash_sc_grpc_request,
    parse_input_ids_from_request,
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


def _unpack_fp32_le(raw: bytes) -> list[float]:
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


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
        self.assertEqual(sp.max_new_tokens, 131072)
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

    def test_parse_logprobs_tensors_and_generate_config_mapping(self) -> None:
        for shape in ([], [1], [1, 1]):
            with self.subTest(shape=shape):
                req = predict_v2_pb2.ModelInferRequest()
                _add_tensor(req, "logprobs", "BOOL", shape, b"\x01")
                _add_tensor(req, "top_logprobs", "INT32", shape, struct.pack("<i", 5))

                sp = parse_sampling_params(req)
                config = sp.to_generate_config()

                self.assertTrue(sp.return_logprobs)
                self.assertEqual(sp.top_logprobs, 5)
                self.assertTrue(config.return_logprobs)
                self.assertEqual(config.top_logprobs, 5)

    def test_batch_wrapped_false_logprobs_preserves_disabled_path(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "logprobs", "BOOL", [1, 1], b"\x00")
        _add_tensor(req, "top_logprobs", "INT32", [1, 1], struct.pack("<i", 0))

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self.assertFalse(sp.return_logprobs)
        self.assertEqual(sp.top_logprobs, 0)
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.top_logprobs, 0)

    def test_online_disabled_defaults_with_ordinary_top_k(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["logprobs"].bool_param = False
        req.parameters["top_logprobs"].int64_param = 0
        req.parameters["top_k"].int64_param = 5
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "body": {
                    "input": {"prompt": "今天中午吃什么？"},
                    "parameters": {"max_length": 1000, "top_k": 5},
                }
            },
            ensure_ascii=False,
        )

        sp = parse_sampling_params(req)
        config = sp.to_generate_config()

        self.assertEqual(sp.top_k, 5)
        self.assertFalse(sp.return_logprobs)
        self.assertEqual(sp.top_logprobs, 0)
        self.assertFalse(config.return_logprobs)
        self.assertEqual(config.top_logprobs, 0)

    def test_positive_top_logprobs_still_requires_logprobs(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["logprobs"].bool_param = False
        req.parameters["top_logprobs"].int64_param = 5

        with self.assertRaisesRegex(
            DashScParameterError, "top_logprobs requires logprobs=true"
        ):
            parse_sampling_params(req)

    def test_parse_online_nested_logprobs_with_thinking_controls(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {
                "body": {
                    "parameters": {
                        "logprobs": True,
                        "top_logprobs": 5,
                    }
                }
            }
        )
        _add_tensor(req, "min_length", "INT32", [1], struct.pack("<i", 3))
        _add_tensor(req, "max_new_think_tokens", "INT32", [1], struct.pack("<i", 128))

        sp = parse_sampling_params(req)

        self.assertEqual(sp.min_new_tokens, 3)
        self.assertEqual(sp.max_new_think_tokens, 128)
        self.assertTrue(sp.return_logprobs)
        self.assertEqual(sp.top_logprobs, 5)

    def test_parse_return_logprobs_parameter_alias(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["return_logprobs"].bool_param = True
        req.parameters["top_logprobs"].int64_param = 2

        sp = parse_sampling_params(req)

        self.assertTrue(sp.return_logprobs)
        self.assertEqual(sp.top_logprobs, 2)

    def test_parameter_n_fallback_preserves_non_logprobs_backend_fanout(
        self,
    ) -> None:
        for param_name in ("n", "num_return_sequences"):
            for value in (0, 1):
                for return_logprobs in (False, True):
                    with self.subTest(
                        param_name=param_name,
                        value=value,
                        return_logprobs=return_logprobs,
                    ):
                        req = predict_v2_pb2.ModelInferRequest()
                        req.parameters[param_name].int64_param = value
                        if return_logprobs:
                            req.parameters["logprobs"].bool_param = True

                        sp = parse_sampling_params(req)
                        config = sp.to_generate_config()
                        expected = value if return_logprobs else 0

                        self.assertEqual(sp.num_return_sequences, expected)
                        self.assertEqual(config.num_return_sequences, expected)
                        self.assertIs(sp.return_logprobs, return_logprobs)
                        self.assertIs(config.return_logprobs, return_logprobs)

    def test_parameter_n_defaults_do_not_enable_backend_fanout(self) -> None:
        for return_logprobs in (False, True):
            with self.subTest(return_logprobs=return_logprobs):
                req = predict_v2_pb2.ModelInferRequest()
                if return_logprobs:
                    req.parameters["logprobs"].bool_param = True

                sp = parse_sampling_params(req)
                config = sp.to_generate_config()

                self.assertEqual(sp.num_return_sequences, 0)
                self.assertEqual(config.num_return_sequences, 0)

    def test_parameter_n_greater_than_one_is_rejected_for_single_result_wire(
        self,
    ) -> None:
        for param_name in ("n", "num_return_sequences"):
            for return_logprobs in (False, True):
                with self.subTest(
                    param_name=param_name,
                    return_logprobs=return_logprobs,
                ):
                    req = predict_v2_pb2.ModelInferRequest()
                    req.parameters[param_name].int64_param = 2
                    if return_logprobs:
                        req.parameters["logprobs"].bool_param = True

                    with self.assertRaisesRegex(
                        DashScParameterError,
                        "DashScope response does not support n > 1",
                    ):
                        parse_sampling_params(req)

    def test_parameter_n_alias_conflict_is_rejected(self) -> None:
        for return_logprobs in (False, True):
            with self.subTest(return_logprobs=return_logprobs):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["n"].int64_param = 1
                req.parameters["num_return_sequences"].int64_param = 0
                if return_logprobs:
                    req.parameters["logprobs"].bool_param = True

                with self.assertRaisesRegex(
                    DashScParameterError,
                    "conflicting n and num_return_sequences parameters",
                ):
                    parse_sampling_params(req)

    def test_input_tensor_n_keeps_existing_mapping_and_logprobs_validation(
        self,
    ) -> None:
        for tensor_name in ("n", "num_return_sequences"):
            for value in (0, 1, 2):
                for return_logprobs in (False, True):
                    with self.subTest(
                        tensor_name=tensor_name,
                        value=value,
                        return_logprobs=return_logprobs,
                    ):
                        req = predict_v2_pb2.ModelInferRequest()
                        _add_tensor(
                            req,
                            tensor_name,
                            "INT32",
                            [1],
                            struct.pack("<i", value),
                        )
                        if return_logprobs:
                            req.parameters["logprobs"].bool_param = True

                        if return_logprobs and value > 1:
                            with self.assertRaisesRegex(
                                DashScParameterError,
                                "logprobs does not support n > 1",
                            ):
                                parse_sampling_params(req)
                            continue

                        sp = parse_sampling_params(req)
                        config = sp.to_generate_config()
                        self.assertEqual(sp.num_return_sequences, value)
                        self.assertEqual(config.num_return_sequences, value)

    def test_logprobs_validation(self) -> None:
        invalid_cases = []

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["top_logprobs"].int64_param = 1
        invalid_cases.append((req, "requires logprobs=true"))

        for value in (-1, 21):
            req = predict_v2_pb2.ModelInferRequest()
            req.parameters["logprobs"].bool_param = True
            req.parameters["top_logprobs"].int64_param = value
            invalid_cases.append((req, "between 0 and 20"))

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["logprobs"].bool_param = True
        _add_tensor(req, "n", "INT32", [1], struct.pack("<i", 2))
        invalid_cases.append((req, "does not support n > 1"))

        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "logprobs", "BOOL", [2], b"\x01\x00")
        invalid_cases.append((req, "must contain exactly one value"))

        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "logprobs", "BOOL", [1], b"\x02")
        invalid_cases.append((req, "boolean scalar"))

        req = predict_v2_pb2.ModelInferRequest()
        _add_tensor(req, "top_logprobs", "INT32", [1], struct.pack("<2i", 1, 2))
        invalid_cases.append((req, "integer scalar"))

        for req, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(DashScParameterError, message):
                    parse_sampling_params(req)

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

        self.assertEqual(sp.max_new_tokens, 131072)
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
        # Negative sentinel values are preserved through the conversion layer.
        self.assertEqual(sp.max_new_think_tokens, -1)
        self.assertEqual(sp.to_generate_config().max_thinking_tokens, -1)

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

    def test_build_request_writes_logprobs_controls(self) -> None:
        req = build_model_infer_request(
            request_id="test-logprobs",
            model_name="default",
            input_ids=[1, 2],
            sampling=SamplingParams(return_logprobs=True, top_logprobs=4),
        )

        _, sp, _ = parse_dash_sc_grpc_request(req)

        self.assertIsNotNone(sp)
        assert sp is not None
        self.assertTrue(sp.return_logprobs)
        self.assertEqual(sp.top_logprobs, 4)

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

    def test_parse_other_params_debug_parameter(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["debug"].bool_param = True
        self.assertTrue(parse_other_params(req).debug)

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["debug"].bool_param = False
        self.assertFalse(parse_other_params(req).debug)

        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"parameters": {"debug": True}}
        )
        self.assertTrue(parse_other_params(req).debug)

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

    def test_stream_response_preserves_negative_thinking_budget(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1),
        )
        config = GenerateConfig(max_thinking_tokens=-1)
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="req-negative-budget",
            model_name="mdl",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag=stream_log_tag(
                request_id_numeric=1, trace_id="req-negative-budget"
            ),
            generate_config=config,
        )

        self.assertEqual(
            resp.infer_response.parameters["max_new_think_tokens"].int64_param,
            -1,
        )


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

    def test_serializes_compact_logprob_tensors(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1, -0.2], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor([[7, 70], [8, 80]], dtype=torch.int64),
            top_logprobs=torch.tensor(
                [[-0.1, -1.1], [-0.2, -1.2]], dtype=torch.float32
            ),
            finished=True,
        )
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="logprobs",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
        )
        infer = resp.infer_response
        metadata = {item.name: item for item in infer.outputs}
        raw = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }

        self.assertEqual(
            [item.name for item in infer.outputs[:5]],
            [
                "generated_ids",
                "finish_reason",
                "finished",
                "prompt_token_num",
                "prompt_cached_token_num",
            ],
        )
        self.assertEqual(list(metadata["token_logprobs"].shape), [1, 2])
        self.assertEqual(metadata["token_logprobs"].datatype, "FP32")
        self.assertEqual(list(metadata["top_logprob_token_ids"].shape), [1, 2, 2])
        self.assertEqual(metadata["top_logprob_token_ids"].datatype, "INT32")
        self.assertEqual(list(metadata["top_logprobs"].shape), [1, 2, 2])
        self.assertEqual(_unpack_int32_le(raw["generated_ids"]), [7, 8])
        self.assertEqual(_unpack_int32_le(raw["top_logprob_token_ids"]), [7, 70, 8, 80])
        for actual, expected in zip(
            _unpack_fp32_le(raw["token_logprobs"]), [-0.1, -0.2]
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            _unpack_fp32_le(raw["top_logprobs"]), [-0.1, -1.1, -0.2, -1.2]
        ):
            self.assertAlmostEqual(actual, expected)
        gateway_logprobs = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(len(gateway_logprobs), 2)
        self.assertAlmostEqual(gateway_logprobs[0]["7"], -0.1)
        self.assertAlmostEqual(gateway_logprobs[0]["70"], -1.1)
        self.assertAlmostEqual(gateway_logprobs[1]["8"], -0.2)
        self.assertAlmostEqual(gateway_logprobs[1]["80"], -1.2)

    def test_rewritten_or_misaligned_logprobs_are_rejected(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8], dtype=torch.int32),
            token_logprobs=torch.tensor([[-0.1], [-0.2]], dtype=torch.float32),
            finished=True,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_stream_response_from_generate_outputs(
                dash_sc_request_id="bad-shape",
                model_name="m",
                go=GenerateOutputs(generate_outputs=[out]),
                request_log_tag="tag",
                generate_config=GenerateConfig(return_logprobs=True),
            )

        out.token_logprobs = torch.tensor([-0.1, -0.2], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "rewritten generated_ids"):
            build_stream_response_from_generate_outputs(
                dash_sc_request_id="bad-rewrite",
                model_name="m",
                go=GenerateOutputs(generate_outputs=[out]),
                request_log_tag="tag",
                token_ids=[7],
                generate_config=GenerateConfig(return_logprobs=True),
            )

    def test_top_logprobs_zero_serializes_empty_aligned_top_tensors(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1, -0.2], dtype=torch.float32),
            top_logprob_token_ids=torch.empty((2, 0), dtype=torch.int32),
            top_logprobs=torch.empty((2, 0), dtype=torch.float32),
            finished=True,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="top-zero",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
        ).infer_response
        metadata = {item.name: item for item in infer.outputs}
        raw = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }

        self.assertEqual(list(metadata["top_logprob_token_ids"].shape), [1, 2, 0])
        self.assertEqual(list(metadata["top_logprobs"].shape), [1, 2, 0])
        self.assertEqual(raw["top_logprob_token_ids"], b"")
        self.assertEqual(raw["top_logprobs"], b"")
        gateway_logprobs = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(len(gateway_logprobs), 2)
        self.assertEqual(list(gateway_logprobs[0]), ["7"])
        self.assertEqual(list(gateway_logprobs[1]), ["8"])

    def test_mtp_uncommitted_tail_is_dropped_for_every_accept_length(self) -> None:
        """MTP may expose P+1 verification rows but commit only 0..P tokens.

        Keep this matrix explicit: these are the production failures reported
        as 0/1, 1/2, 2/3, 3/4, 4/5 and 5/6 shape mismatches.
        """
        for accepted in range(6):
            for compact in (False, True):
                with self.subTest(accepted=accepted, compact=compact):
                    self._assert_mtp_uncommitted_tail_is_dropped(
                        accepted=accepted, compact=compact
                    )

    def _assert_mtp_uncommitted_tail_is_dropped(
        self, *, accepted: int, compact: bool
    ) -> None:
        generated_ids = list(range(100, 100 + accepted))
        probability_rows = accepted + 1
        token_logprobs = torch.arange(probability_rows, dtype=torch.float32).neg()
        top_ids = torch.arange(
            100,
            100 + probability_rows * 5,
            dtype=torch.int32,
        ).reshape(probability_rows, 5)
        if accepted:
            top_ids[:accepted, 0] = torch.tensor(generated_ids, dtype=torch.int32)
        top_values = (
            torch.arange(probability_rows * 5, dtype=torch.float32)
            .neg()
            .reshape(probability_rows, 5)
        )
        out = GenerateOutput(
            output_ids=torch.tensor(generated_ids, dtype=torch.int32),
            token_logprobs=token_logprobs,
            top_logprob_token_ids=top_ids,
            top_logprobs=top_values,
            finished=accepted == 0,
            logprobs_offset=0 if compact else None,
            logprobs_count=probability_rows if compact else None,
        )

        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id=f"mtp-accept-{accepted}",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=5),
        ).infer_response
        metadata = {item.name: item for item in infer.outputs}
        self.assertEqual(list(metadata["token_logprobs"].shape), [1, accepted])
        self.assertEqual(
            list(metadata["top_logprob_token_ids"].shape),
            [1, accepted, 5],
        )
        records = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(len(records), accepted)
        for token_id, record in zip(generated_ids, records):
            self.assertTrue(all(isinstance(key, str) for key in record))
            self.assertIn(str(token_id), record)

    def test_compact_mtp_tail_after_reasoning_prefix_is_dropped(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([10, 11], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.2, -0.3], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor([[11], [12]], dtype=torch.int32),
            top_logprobs=torch.tensor([[-0.2], [-0.3]], dtype=torch.float32),
            logprobs_offset=1,
            logprobs_count=2,
        )

        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="compact-prefix-tail",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=1),
        ).infer_response

        records = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(records[0], {"10": 0.0})
        self.assertAlmostEqual(records[1]["11"], -0.2, places=6)
        self.assertEqual(out.logprobs_offset, 1)
        self.assertEqual(out.logprobs_count, 1)

    def test_compact_thinking_offset_drops_one_elided_tail(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([10, 11], dtype=torch.int32),
            logprobs_offset=3,
            logprobs_count=0,
        )

        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="compact-thinking-tail",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=0),
        ).infer_response

        records = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(records, [{"10": 0.0}, {"11": 0.0}])
        self.assertEqual(out.logprobs_offset, 2)
        self.assertEqual(out.logprobs_count, 0)

    def test_logprob_shape_mismatch_other_than_one_mtp_tail_is_rejected(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1, -0.2, -0.3]),
            top_logprob_token_ids=torch.tensor([[7], [8], [9]], dtype=torch.int32),
            top_logprobs=torch.tensor([[-0.1], [-0.2], [-0.3]]),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_stream_response_from_generate_outputs(
                dash_sc_request_id="mtp-invalid-extra-rows",
                model_name="m",
                go=GenerateOutputs(generate_outputs=[out]),
                request_log_tag="tag",
                generate_config=GenerateConfig(return_logprobs=True, top_logprobs=1),
            )

    def test_synthetic_frame_omits_logprob_payload(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor([[7, 70]], dtype=torch.int32),
            top_logprobs=torch.tensor([[-0.1, -1.1]], dtype=torch.float32),
            finished=False,
        )
        resp = build_stream_response_from_generate_outputs(
            dash_sc_request_id="synthetic",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
            token_ids=[128822, 271],
            include_logprobs=False,
        )
        infer = resp.infer_response
        names = [item.name for item in infer.outputs]

        self.assertEqual(_unpack_int32_le(infer.raw_output_contents[0]), [128822, 271])
        self.assertNotIn("token_logprobs", names)
        self.assertNotIn("top_logprob_token_ids", names)
        self.assertNotIn("top_logprobs", names)
        self.assertNotIn("logprobs", infer.parameters)

    def test_forced_think_close_frame_has_aligned_logprob_payload(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor([[7, 70]], dtype=torch.int32),
            top_logprobs=torch.tensor([[-0.1, -1.1]], dtype=torch.float32),
            finished=False,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="forced-think-close",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=5),
            token_ids=[128822, 271],
            include_logprobs=False,
            include_forced_token_logprobs=True,
        ).infer_response
        metadata = {item.name: item for item in infer.outputs}

        self.assertEqual(list(metadata["token_logprobs"].shape), [1, 2])
        self.assertEqual(list(metadata["top_logprob_token_ids"].shape), [1, 2, 5])
        wire_rows = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(len(wire_rows), 2)
        self.assertEqual(wire_rows[0]["128822"], 0.0)
        self.assertEqual(wire_rows[1]["271"], 0.0)
        self.assertTrue(all(len(row) == 5 for row in wire_rows))

    def test_compact_thinking_only_frame_materializes_placeholders(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([10, 11, 128822], dtype=torch.int32),
            logprobs_offset=3,
            logprobs_count=0,
            finished=False,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="compact-thinking-only",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
        ).infer_response

        metadata = {item.name: item for item in infer.outputs}
        raw = {
            item.name: infer.raw_output_contents[index]
            for index, item in enumerate(infer.outputs)
        }
        self.assertEqual(list(metadata["token_logprobs"].shape), [1, 3])
        self.assertEqual(_unpack_fp32_le(raw["token_logprobs"]), [0.0, 0.0, 0.0])
        self.assertEqual(
            _unpack_int32_le(raw["top_logprob_token_ids"]),
            [10, 0, 11, 0, 128822, 0],
        )
        top_values = _unpack_fp32_le(raw["top_logprobs"])
        self.assertEqual(top_values[::2], [0.0, 0.0, 0.0])
        self.assertTrue(all(math.isinf(value) for value in top_values[1::2]))

        wire_rows = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(
            wire_rows,
            [
                {"10": 0.0, "0": -math.inf},
                {"11": 0.0, "0": -math.inf},
                {"128822": 0.0, "0": -math.inf},
            ],
        )

    def test_reasoning_prefix_uses_placeholders_and_content_keeps_logprobs(
        self,
    ) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([10, 128822, 271, 20], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.13, -0.20], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor(
                [[271, 2], [20, 120]],
                dtype=torch.int32,
            ),
            top_logprobs=torch.tensor(
                [[-0.13, -1.13], [-0.20, -1.20]],
                dtype=torch.float32,
            ),
            logprobs_offset=2,
            logprobs_count=2,
            finished=False,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="reasoning-content-boundary",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True, top_logprobs=2),
        ).infer_response

        metadata = {item.name: item for item in infer.outputs}
        raw = {
            item.name: infer.raw_output_contents[index]
            for index, item in enumerate(infer.outputs)
        }
        token_values = _unpack_fp32_le(raw["token_logprobs"])
        self.assertEqual(token_values[:2], [0.0, 0.0])
        self.assertAlmostEqual(token_values[2], -0.13)
        self.assertAlmostEqual(token_values[3], -0.20)
        self.assertEqual(list(metadata["token_logprobs"].shape), [1, 4])
        self.assertEqual(
            _unpack_int32_le(raw["top_logprob_token_ids"]),
            [10, 0, 128822, 0, 271, 2, 20, 120],
        )
        top_values = _unpack_fp32_le(raw["top_logprobs"])
        self.assertEqual(top_values[0], 0.0)
        self.assertTrue(math.isinf(top_values[1]))
        self.assertEqual(top_values[2], 0.0)
        self.assertTrue(math.isinf(top_values[3]))
        self.assertAlmostEqual(top_values[4], -0.13)
        self.assertAlmostEqual(top_values[6], -0.20)

        wire_rows = json.loads(infer.parameters["logprobs"].string_param)
        self.assertEqual(wire_rows[0]["10"], 0.0)
        self.assertEqual(wire_rows[1]["128822"], 0.0)
        self.assertAlmostEqual(wire_rows[2]["271"], -0.13)
        self.assertAlmostEqual(wire_rows[3]["20"], -0.20)

    def test_disabled_request_ignores_stale_logprob_tensors(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1], dtype=torch.float32),
            top_logprob_token_ids=torch.tensor([[7]], dtype=torch.int32),
            top_logprobs=torch.tensor([[-0.1]], dtype=torch.float32),
            finished=True,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="disabled-stale",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=False),
        ).infer_response

        names = [item.name for item in infer.outputs]
        self.assertNotIn("token_logprobs", names)
        self.assertNotIn("top_logprob_token_ids", names)
        self.assertNotIn("top_logprobs", names)
        self.assertNotIn("logprobs", infer.parameters)

    def test_stream_response_returns_dash_debug_info_when_requested(self) -> None:
        config = GenerateConfig(max_new_tokens=128, max_thinking_tokens=100)
        unfinished = GenerateOutput(
            output_ids=torch.tensor([7], dtype=torch.int32), finished=False
        )
        unfinished_infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="debug-streaming",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[unfinished]),
            request_log_tag="tag",
            generate_config=config,
            debug=True,
        ).infer_response
        self.assertNotIn("debug_info", unfinished_infer.parameters)

        finished = GenerateOutput(
            output_ids=torch.tensor([8], dtype=torch.int32), finished=True
        )
        finished_infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="debug-finished",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[finished]),
            request_log_tag="tag",
            generate_config=config,
            debug=True,
        ).infer_response
        self.assertEqual(
            json.loads(finished_infer.parameters["debug_info"].string_param),
            {
                "llm_params": {
                    "max_new_tokens": 128,
                    "max_new_think_tokens": 100,
                }
            },
        )

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
        # Empty INT32 tensors carry one filler element to keep raw_output_contents
        # aligned; consumers must trust the declared shape.
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [0])
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

    def test_prepend_refuses_to_misalign_logprob_outputs(self) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([7, 8], dtype=torch.int32),
            token_logprobs=torch.tensor([-0.1, -0.2], dtype=torch.float32),
            finished=True,
        )
        infer = build_stream_response_from_generate_outputs(
            dash_sc_request_id="r",
            model_name="m",
            go=GenerateOutputs(generate_outputs=[out]),
            request_log_tag="tag",
            generate_config=GenerateConfig(return_logprobs=True),
        ).infer_response

        self.assertFalse(prepend_to_generated_ids_tensor(infer, [100]))
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [7, 8])
        self.assertEqual(len(_unpack_fp32_le(by_name["token_logprobs"])), 2)


class StreamLogTagTest(TestCase):
    def test_stream_log_tag_format(self) -> None:
        self.assertEqual(
            stream_log_tag(request_id_numeric=-1, trace_id="tid"),
            "request_id=-1 trace_id=tid",
        )


if __name__ == "__main__":
    main()

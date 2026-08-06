"""XGrammar request-safety tests for DashSc request parsing."""

from __future__ import annotations

import json
from unittest import TestCase, main

from rtp_llm.dash_sc.codec import DashScParameterError, parse_sampling_params
from rtp_llm.dash_sc.proto import predict_v2_pb2


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


def _deepseek_xml_structural_tag(schema: dict) -> dict:
    return {
        "format": {
            "type": "tag",
            "begin": "<result>",
            "content": {
                "type": "json_schema",
                "json_schema": schema,
                "style": "deepseek_xml",
            },
            "end": "</result>",
        }
    }


class XGrammarSafetyTest(TestCase):
    def test_rejects_deepseek_xml_length_constraint_above_128(self) -> None:
        for keyword in ("minLength", "maxLength"):
            with self.subTest(keyword=keyword):
                schema = {
                    "type": "object",
                    "properties": {
                        "result": {
                            "anyOf": [
                                {"type": "string", keyword: 129},
                                {"type": "null"},
                            ]
                        }
                    },
                }
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    _deepseek_xml_structural_tag(schema)
                )

                with self.assertRaisesRegex(
                    DashScParameterError,
                    rf"deepseek_xml.*{keyword}=129.*must be <= 128",
                ):
                    parse_sampling_params(req)

    def test_allows_deepseek_xml_length_constraint_at_128(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "minLength": 128,
                    "maxLength": 128,
                }
            },
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            _deepseek_xml_structural_tag(schema)
        )

        sampling = parse_sampling_params(req)

        self.assertEqual(
            json.loads(sampling.structural_tag)["format"]["content"]["style"],
            "deepseek_xml",
        )

    def test_rejects_unsafe_deepseek_xml_from_response_format(self) -> None:
        tag = _deepseek_xml_structural_tag({"type": "string", "maxLength": 1000})
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["response_format"].string_param = json.dumps(
            {"type": "structural_tag", "format": tag["format"]}
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=1000"
        ):
            parse_sampling_params(req)

    def test_deepseek_xml_uses_raw_schema_not_response_format_envelope(self) -> None:
        schema = {
            "schema": {},
            "type": "string",
            "maxLength": 129,
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            _deepseek_xml_structural_tag(schema)
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_implicit_deepseek_xml_json_schema_format(self) -> None:
        tag = _deepseek_xml_structural_tag({"type": "string", "maxLength": 129})
        del tag["format"]["content"]["type"]
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(tag)

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_referenced_outer_string_constraint(self) -> None:
        schema = {
            "type": "object",
            "properties": {"result": {"$ref": "#/$defs/result"}},
            "$defs": {"result": {"type": "string", "minLength": 129}},
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            _deepseek_xml_structural_tag(schema)
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*minLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_root_array_string_constraint(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            _deepseek_xml_structural_tag(
                {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 129},
                }
            )
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_empty_type_array_constraint(self) -> None:
        schema = {
            "type": [],
            "properties": {"result": {"type": "string", "maxLength": 129}},
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(
            _deepseek_xml_structural_tag(schema)
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_deepseek_xml_only_checks_outer_xml_string_constraints(self) -> None:
        safe_schemas = (
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "maxLength": 129}},
                    }
                },
            },
            {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 129},
                    }
                },
            },
            {
                "type": "object",
                "$defs": {"unused": {"type": "string", "maxLength": 129}},
            },
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "maxLength": 129},
                    "value": {
                        "type": "string",
                        "default": {"maxLength": 129},
                    },
                },
            },
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "pattern": ".*",
                        "maxLength": 129,
                    }
                },
            },
        )
        for schema in safe_schemas:
            with self.subTest(schema=schema):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    _deepseek_xml_structural_tag(schema)
                )

                sampling = parse_sampling_params(req)

                self.assertIsNotNone(sampling.structural_tag)

    def test_rejects_deepseek_xml_from_nested_dashscope_sequence(self) -> None:
        tag = _dashscope_sequence_tool_call_structural_tag()
        tag["type"] = "structural_tag"
        content = tag["format"]["elements"][1]["tags"][0]["content"]
        content["style"] = "deepseek_xml"
        content["json_schema"] = {
            "type": "object",
            "properties": {"result": {"type": "string", "maxLength": 129}},
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["ds_header_attributes"].string_param = json.dumps(
            {"parameters": {"tool_call_structural_tag": tag}}
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_from_implicit_nested_formats(self) -> None:
        tag = _dashscope_sequence_tool_call_structural_tag()
        del tag["format"]["type"]
        nested_tag = tag["format"]["elements"][1]["tags"][0]
        del nested_tag["type"]
        content = nested_tag["content"]
        del content["type"]
        content["style"] = "deepseek_xml"
        content["json_schema"] = {
            "type": "object",
            "properties": {"result": {"type": "string", "minLength": 129}},
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(tag)

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*minLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_after_sanitized_empty_schema_format(self) -> None:
        tag = {
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "json_schema"},
                    {
                        "type": "json_schema",
                        "json_schema": {"type": "string", "maxLength": 129},
                        "style": "deepseek_xml",
                    },
                ],
            }
        }
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = json.dumps(tag)

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_rejects_deepseek_xml_from_implicit_dispatch(self) -> None:
        unsafe_format = {
            "json_schema": {"type": "string", "maxLength": 129},
            "style": "deepseek_xml",
        }
        for trigger in ("go", 123):
            with self.subTest(trigger=trigger):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    {"format": {"rules": [[trigger, unsafe_format]]}}
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"deepseek_xml.*maxLength=129"
                ):
                    parse_sampling_params(req)

    def test_rejects_deepseek_xml_through_numeric_token_formats(self) -> None:
        unsafe_format = {
            "type": "json_schema",
            "json_schema": {"type": "string", "maxLength": 129},
            "style": "deepseek_xml",
        }
        formats = (
            {
                "type": "tag",
                "begin": {"token": 1.0},
                "content": unsafe_format,
                "end": "</x>",
            },
            {
                "type": "token_triggered_tags",
                "trigger_tokens": [1.0],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<x>",
                        "content": unsafe_format,
                        "end": "</x>",
                    }
                ],
            },
            {
                "type": "token_dispatch",
                "rules": [[1.0, unsafe_format]],
            },
            {
                "type": "token_dispatch",
                "rules": [[-1, unsafe_format]],
            },
        )
        for structural_format in formats:
            with self.subTest(structural_format=structural_format):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    {"format": structural_format}
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"deepseek_xml.*maxLength=129"
                ):
                    parse_sampling_params(req)

    def test_deep_structural_format_is_checked_without_python_recursion(self) -> None:
        structural_format = json.dumps(
            {
                "type": "json_schema",
                "json_schema": {"type": "string", "maxLength": 129},
                "style": "deepseek_xml",
            }
        )
        for _ in range(600):
            structural_format = (
                '{"type":"optional","content":' + structural_format + "}"
            )
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["tool_call_structural_tag"].string_param = (
            '{"format":' + structural_format + "}"
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"deepseek_xml.*maxLength=129"
        ):
            parse_sampling_params(req)

    def test_implicit_structural_format_respects_parser_precedence(self) -> None:
        unsafe_format = {
            "json_schema": {"type": "string", "maxLength": 129},
            "style": "deepseek_xml",
        }
        safe_formats = (
            {"value": "fixed", "elements": [unsafe_format]},
            {
                "begin": "<x>",
                "content": {"value": "fixed"},
                "end": "</x>",
                "elements": [unsafe_format],
            },
            {"excludes": [], "content": unsafe_format},
        )
        for structural_format in safe_formats:
            with self.subTest(structural_format=structural_format):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    {"format": structural_format}
                )

                sampling = parse_sampling_params(req)

                self.assertIsNotNone(sampling.structural_tag)

    def test_implicit_structural_format_falls_through_invalid_candidates(self) -> None:
        unsafe_schema = {"type": "string", "maxLength": 129}
        unsafe_format = {
            "json_schema": unsafe_schema,
            "style": "deepseek_xml",
        }
        safe_tag = {
            "begin": "<x>",
            "content": {"value": "safe"},
            "end": "</x>",
        }
        formats = (
            {
                "begin": {},
                "content": {"type": "const_string", "value": "safe"},
                "end": "</x>",
                "json_schema": unsafe_schema,
                "style": "deepseek_xml",
            },
            {
                "elements": [{}],
                "content": unsafe_format,
            },
            {"json_schema": {}, "style": "bogus", "content": unsafe_format},
            {"json_schema": {}, "any_order": "bad", "content": unsafe_format},
            {
                "triggers": ["<x>"],
                "tags": [safe_tag],
                "at_least_one": "bad",
                "content": unsafe_format,
            },
            {
                "tags": [safe_tag],
                "separator": "",
                "stop_after_first": "bad",
                "content": unsafe_format,
            },
            {
                "triggers": ["<x>"],
                "tags": [{**safe_tag, "type": "unknown"}],
                "content": unsafe_format,
            },
        )
        for structural_format in formats:
            with self.subTest(structural_format=structural_format):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    {"format": structural_format}
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"deepseek_xml.*maxLength=129"
                ):
                    parse_sampling_params(req)

    def test_rejects_plain_json_unbounded_min_length_above_2000(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["response_format"].string_param = json.dumps(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "experience": {"type": "string", "minLength": 2001}
                        },
                    },
                },
            }
        )

        with self.assertRaisesRegex(
            DashScParameterError,
            r"unbounded JSON schema.*minLength=2001.*must be <= 2000",
        ):
            parse_sampling_params(req)

    def test_rejects_plain_json_string_schema_above_2000(self) -> None:
        schema = {"type": "string", "minLength": 2001}
        containers = (
            json.dumps(schema),
            {"name": "result", "schema": json.dumps(schema)},
        )
        for container in containers:
            with self.subTest(container=container):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = json.dumps(
                    {"type": "json_schema", "json_schema": container}
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"unbounded JSON schema.*minLength=2001"
                ):
                    parse_sampling_params(req)

    def test_rejects_plain_json_effectively_unbounded_lengths(self) -> None:
        schemas = (
            {
                "type": [],
                "properties": {"result": {"type": "string", "minLength": 2001}},
            },
            {"type": "string", "minLength": 50000, "maxLength": -1},
            {"type": "string", "minLength": 50000, "maxLength": 4_294_967_295},
        )
        for schema in schemas:
            with self.subTest(schema=schema):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = json.dumps(
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                    }
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"unbounded JSON schema.*minLength"
                ):
                    parse_sampling_params(req)

    def test_rejects_plain_json_length_in_structural_tag(self) -> None:
        schemas = (
            (
                {},
                {"type": "string", "minLength": 2001},
            ),
            (
                {"style": "json"},
                {
                    "type": "array",
                    "items": {"type": "string", "minLength": 2001},
                },
            ),
        )
        for extra_format, schema in schemas:
            with self.subTest(extra_format=extra_format, schema=schema):
                tag = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": schema,
                        **extra_format,
                    }
                }
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    tag
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"unbounded JSON schema.*minLength=2001"
                ):
                    parse_sampling_params(req)

    def test_rejects_plain_json_length_in_nested_deepseek_schema(self) -> None:
        nested_schemas = (
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "minLength": 50000}},
                    }
                },
            },
            {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 50000},
                    }
                },
            },
        )
        for schema in nested_schemas:
            with self.subTest(schema=schema):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    _deepseek_xml_structural_tag(schema)
                )

                with self.assertRaisesRegex(
                    DashScParameterError, r"unbounded JSON schema.*minLength=50000"
                ):
                    parse_sampling_params(req)

    def test_deepseek_recursive_schema_safety_check_terminates(self) -> None:
        schemas = (
            (
                False,
                {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/node"}},
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {"next": {"$ref": "#/$defs/node"}},
                        }
                    },
                },
            ),
            (
                True,
                {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/node"}},
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "value": {
                                    "type": "string",
                                    "minLength": 50000,
                                },
                                "next": {"$ref": "#/$defs/node"},
                            },
                        }
                    },
                },
            ),
        )
        for should_reject, schema in schemas:
            with self.subTest(should_reject=should_reject):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["tool_call_structural_tag"].string_param = json.dumps(
                    _deepseek_xml_structural_tag(schema)
                )

                if should_reject:
                    with self.assertRaisesRegex(
                        DashScParameterError,
                        r"unbounded JSON schema.*minLength=50000",
                    ):
                        parse_sampling_params(req)
                else:
                    self.assertIsNotNone(parse_sampling_params(req).structural_tag)

    def test_plain_json_ignores_non_compiled_length_fields(self) -> None:
        safe_schemas = (
            {"type": "integer", "minLength": 2001},
            {"type": "string", "default": {"minLength": 2001}},
            {"type": "string", "const": "fixed", "minLength": 2001},
            {"type": "string", "pattern": ".*", "minLength": 2001},
            {"type": "string", "format": "email", "minLength": 2001},
            {"type": "string", "format": {}, "minLength": 2001},
            {"type": "string", "pattern": [], "minLength": 2001},
            {
                "type": "object",
                "$defs": {"unused": {"type": "string", "minLength": 2001}},
            },
        )
        for schema in safe_schemas:
            with self.subTest(schema=schema):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = json.dumps(
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                    }
                )

                sampling = parse_sampling_params(req)

                self.assertIsNotNone(sampling.response_format)

    def test_rejects_guided_json_unbounded_min_length_above_2000(self) -> None:
        req = predict_v2_pb2.ModelInferRequest()
        req.parameters["guided_json"].string_param = json.dumps(
            [{"type": "string", "minLength": 2001}]
        )

        with self.assertRaisesRegex(
            DashScParameterError, r"unbounded JSON schema.*minLength=2001"
        ):
            parse_sampling_params(req)

    def test_plain_json_uses_incident_specific_min_length_guard(self) -> None:
        safe_schemas = (
            {"type": "string", "minLength": 200},
            {"type": "string", "minLength": 2000},
            {"type": "string", "maxLength": 120000},
            {"type": "string", "minLength": 50000, "maxLength": 50000},
        )
        for schema in safe_schemas:
            with self.subTest(schema=schema):
                req = predict_v2_pb2.ModelInferRequest()
                req.parameters["response_format"].string_param = json.dumps(
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                    }
                )

                sampling = parse_sampling_params(req)

                self.assertEqual(
                    json.loads(sampling.response_format)["json_schema"]["schema"],
                    schema,
                )


if __name__ == "__main__":
    main()

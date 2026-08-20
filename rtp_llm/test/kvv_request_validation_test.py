#!/usr/bin/env python3

import unittest

from pydantic import ValidationError

from rtp_llm.openai.api_datatype import ChatCompletionRequest


def _tool(name: str = "get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _request(messages, **kwargs):
    return ChatCompletionRequest.model_validate({"messages": messages, **kwargs})


class KvvRequestValidationTest(unittest.TestCase):
    def test_tool_description_is_optional(self):
        tool = _tool()
        del tool["function"]["description"]
        tool["function"]["strict"] = True

        request = _request([{"role": "user", "content": "hello"}], tools=[tool])

        self.assertIsNone(request.tools[0].function.description)
        self.assertIs(request.tools[0].function.strict, True)

    def test_template_thinking_alias_controls_thinking(self):
        enabled = _request(
            [{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": True},
        )
        disabled = _request(
            [{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": False},
        )
        alias_precedence = _request(
            [{"role": "user", "content": "hello"}],
            chat_template_kwargs={"thinking": True, "enable_thinking": False},
        )

        self.assertTrue(enabled.enable_thinking_requested())
        self.assertTrue(disabled.disable_thinking())
        self.assertTrue(alias_precedence.enable_thinking_requested())
        self.assertFalse(alias_precedence.disable_thinking())

    def test_official_thinking_object_is_preserved_and_disables_thinking(self):
        request = _request(
            [{"role": "user", "content": "hello"}],
            thinking={"type": "disabled"},
        )

        self.assertTrue(request.disable_thinking())
        self.assertEqual(
            request.model_dump(exclude_none=True, mode="json")["thinking"],
            {"type": "disabled"},
        )

    def test_official_thinking_effort_takes_precedence(self):
        request = _request(
            [{"role": "user", "content": "hello"}],
            thinking={"type": "enabled", "keep": "all", "effort": "low"},
            reasoning_effort="max",
        )

        self.assertTrue(request.enable_thinking_requested())
        self.assertEqual(request.thinking.effort, "low")

    def test_sampling_penalties_are_preserved_and_track_explicit_fields(self):
        omitted = _request([{"role": "user", "content": "hello"}])
        explicit = _request(
            [{"role": "user", "content": "hello"}],
            presence_penalty=0.5,
            frequency_penalty=0.25,
        )

        self.assertNotIn("presence_penalty", omitted.model_fields_set)
        self.assertNotIn("frequency_penalty", omitted.model_fields_set)
        self.assertEqual(explicit.presence_penalty, 0.5)
        self.assertEqual(explicit.frequency_penalty, 0.25)
        self.assertIn("presence_penalty", explicit.model_fields_set)
        self.assertIn("frequency_penalty", explicit.model_fields_set)

    def test_dynamic_tools_are_preserved_from_multiple_system_messages(self):
        request = _request(
            [
                {"role": "system", "content": "", "tools": [_tool("weather")]},
                {"role": "user", "content": "hello"},
                {"role": "system", "content": None, "tools": [_tool("time")]},
            ],
            tool_choice="required",
        )

        dumped = request.model_dump(exclude_none=True, mode="json")
        self.assertEqual(
            [
                message["tools"][0]["function"]["name"]
                for message in dumped["messages"]
                if message.get("tools")
            ],
            ["weather", "time"],
        )

    def test_dynamic_tool_strict_false_is_preserved(self):
        tool = _tool()
        tool["function"]["strict"] = False
        request = _request([{"role": "system", "content": "", "tools": [tool]}])
        self.assertIs(request.messages[0].tools[0]["function"]["strict"], False)

    def test_dynamic_tool_parameters_and_strict_types_are_validated(self):
        invalid_functions = [
            {"name": "bad_parameters", "parameters": []},
            {
                "name": "bad_strict",
                "parameters": {"type": "object"},
                "strict": "false",
            },
        ]
        for function in invalid_functions:
            with self.subTest(function=function), self.assertRaises(ValidationError):
                _request(
                    [
                        {
                            "role": "system",
                            "content": "",
                            "tools": [{"type": "function", "function": function}],
                        }
                    ]
                )

    def test_dynamic_tools_require_system_role_and_empty_content(self):
        bad_messages = [
            [{"role": "user", "content": "", "tools": [_tool()]}],
            [{"role": "assistant", "content": "", "tools": [_tool()]}],
            [{"role": "system", "content": "not empty", "tools": [_tool()]}],
        ]
        for messages in bad_messages:
            with self.subTest(messages=messages), self.assertRaises(ValidationError):
                _request(messages)

    def test_dynamic_tool_shape_is_validated(self):
        valid = _tool()
        malformed = []
        for field in ("type", "function"):
            tool = _tool()
            del tool[field]
            malformed.append(tool)
        for field in ("name", "parameters"):
            tool = _tool()
            del tool["function"][field]
            malformed.append(tool)
        malformed.extend(
            [
                {"type": "bogus", "function": {"name": "x"}},
                None,
            ]
        )

        for tool in malformed:
            with self.subTest(tool=tool), self.assertRaises(ValidationError):
                _request([{"role": "system", "content": "", "tools": [tool]}])

        with self.assertRaises(ValidationError):
            _request([{"role": "system", "content": "", "tools": valid}])

    def test_dynamic_tool_name_contract(self):
        for name in ("1bad", "bad@name", "", "a" * 257):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                _request([{"role": "system", "content": "", "tools": [_tool(name)]}])

        request = _request(
            [{"role": "system", "content": "", "tools": [_tool("a" * 256)]}]
        )
        self.assertEqual(request.messages[0].tools[0]["function"]["name"], "a" * 256)

    def test_duplicate_names_are_rejected_across_all_tool_locations(self):
        cases = [
            (
                [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [_tool("dup"), _tool("dup")],
                    }
                ],
                None,
            ),
            (
                [
                    {"role": "system", "content": "", "tools": [_tool("dup")]},
                    {"role": "system", "content": "", "tools": [_tool("dup")]},
                ],
                None,
            ),
            (
                [{"role": "system", "content": "", "tools": [_tool("dup")]}],
                [_tool("dup")],
            ),
        ]
        for messages, tools in cases:
            with self.subTest(messages=messages, tools=tools), self.assertRaises(
                ValidationError
            ):
                _request(messages, tools=tools)

    def test_tool_messages_require_tool_call_id(self):
        with self.assertRaises(ValidationError):
            _request([{"role": "tool", "content": "result"}])

        request = _request(
            [{"role": "tool", "content": "result", "tool_call_id": "call_1"}]
        )
        self.assertEqual(request.messages[0].tool_call_id, "call_1")

    def test_tool_choice_matches_kvv_empty_tools_contract(self):
        for tool_choice in ("none", "auto"):
            with self.subTest(tool_choice=tool_choice):
                request = _request(
                    [{"role": "user", "content": "hello"}],
                    tool_choice=tool_choice,
                )
                self.assertEqual(request.tool_choice, tool_choice)

        for tools in (None, []):
            with self.subTest(tools=tools), self.assertRaises(ValidationError):
                _request(
                    [{"role": "user", "content": "hello"}],
                    tools=tools,
                    tool_choice="required",
                )

        with self.assertRaises(ValidationError):
            _request(
                [{"role": "user", "content": "hello"}],
                tools=[_tool()],
                tool_choice="bogus",
            )

    def test_json_schema_requires_name_schema_and_strict_boolean(self):
        invalid_formats = [
            {"type": "bogus"},
            {"type": "json_schema"},
            {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
            {"type": "json_schema", "json_schema": {"name": "object"}},
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "object",
                    "schema": "not-an-object",
                },
            },
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "object",
                    "strict": "yes",
                    "schema": {"type": "object"},
                },
            },
        ]
        for response_format in invalid_formats:
            with self.subTest(response_format=response_format), self.assertRaises(
                ValidationError
            ):
                _request(
                    [{"role": "user", "content": "hello"}],
                    response_format=response_format,
                )

        request = _request(
            [{"role": "user", "content": "hello"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "object",
                    "strict": False,
                    "schema": {"type": "object"},
                },
            },
        )
        self.assertIs(request.response_format.json_schema.strict, False)


if __name__ == "__main__":
    unittest.main()

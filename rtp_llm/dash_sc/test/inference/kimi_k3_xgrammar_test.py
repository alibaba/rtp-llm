"""Real Python xgrammar contract for K3 XTML admission and decoding."""

import json
import unittest

import xgrammar as xgr
from xgrammar.builtin_structural_tag import get_model_structural_tag

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.response_format_compiler import ResponseFormatPlan
from rtp_llm.dash_sc.inference.grammar_validator import GrammarValidator
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer


class KimiK3XGrammarTest(unittest.TestCase):
    def test_auto_tools_keep_plain_response_and_constrain_xml_argument_type(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        tokenizer = xgr.TokenizerInfo(
            [chr(i) for i in range(128)], stop_token_ids=[0]
        )
        compiler = xgr.GrammarCompiler(tokenizer, max_threads=1)
        renderer = KimiK3Renderer.__new__(KimiK3Renderer)
        call_prefix = (
            '<|close|>response<|sep|><|open|>tools<|sep|>'
            '<|open|>call tool="get_weather" index="1"<|sep|>'
        )
        call_suffix = (
            "<|close|>argument<|sep|><|close|>call<|sep|>"
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        )

        for thinking, tool_choice in (
            (False, "auto"),
            (True, "auto"),
            (False, None),
            (True, None),
        ):
            with self.subTest(thinking=thinking, tool_choice=tool_choice):
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "Weather?"}],
                        "tools": [tool],
                        "tool_choice": tool_choice,
                        "thinking": {
                            "type": "enabled" if thinking else "disabled"
                        },
                    }
                )
                config = GenerateConfig(max_thinking_tokens=8)
                renderer.apply_chat_completion_constraints(request, config)
                plan = ResponseFormatPlan.compile(
                    config, renderer.get_reasoning_format()
                )
                self.assertIsNotNone(plan.engine_constraint)
                grammar = compiler.compile_structural_tag(
                    json.dumps(plan.engine_constraint.value)
                )

                def accepts(text: str) -> bool:
                    matcher = xgr.GrammarMatcher(
                        grammar, terminate_without_stop_token=True
                    )
                    return matcher.accept_string(text) and matcher.is_terminated()

                prefix = (
                    "brief<|close|>think<|sep|><|open|>response<|sep|>"
                    if thinking
                    else ""
                )
                self.assertTrue(
                    accepts(
                        prefix
                        + "ok<|close|>response<|sep|><|close|>message<|sep|>"
                    )
                )
                self.assertTrue(
                    accepts(
                        prefix
                        + call_prefix
                        + '<|open|>argument key="city" type="string"<|sep|>SF'
                        + call_suffix
                    )
                )
                self.assertFalse(
                    accepts(
                        prefix
                        + call_prefix
                        + '<|open|>argument key="city" type="number"<|sep|>1'
                        + call_suffix
                    )
                )
                self.assertFalse(
                    accepts(
                        prefix
                        + call_prefix
                        + '<|open|>argument key="city" type="array"<|sep|>[]'
                        + call_suffix
                    )
                )

    def test_xtml_tool_schema_compiles_and_rejects_wrong_argument_type(self) -> None:
        tool = {
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        }
        structural_tag = get_model_structural_tag(
            "kimi_k3", tools=[tool], tool_choice="required", reasoning=False
        )
        spec = structural_tag.model_dump_json()

        # DashSc must accept the same grammar that the execution side compiles.
        validator = GrammarValidator.__new__(GrammarValidator)
        tokenizer = xgr.TokenizerInfo(
            [chr(i) for i in range(128)], stop_token_ids=[0]
        )
        validator._tokenizer_info_json = tokenizer.serialize_json()
        validator._compile_threads = 1
        validator._cache_limit_bytes = -1
        validator._disable_any_whitespace = False
        validator._backend = validator._build_backend()
        validator._compile("structural_tag", spec)

        compiled_grammar = validator._backend.compile_structural_tag(spec)
        prefix = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
        )
        suffix = (
            "<|close|>argument<|sep|><|close|>call<|sep|>"
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        )

        def accepts(argument_type: str) -> bool:
            matcher = xgr.GrammarMatcher(
                compiled_grammar, terminate_without_stop_token=True
            )
            text = (
                prefix
                + f'<|open|>argument key="city" type="{argument_type}"<|sep|>SF'
                + suffix
            )
            return matcher.accept_string(text) and matcher.is_terminated()

        self.assertTrue(accepts("string"))
        self.assertFalse(accepts("number"))


if __name__ == "__main__":
    unittest.main()

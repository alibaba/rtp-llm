"""Real Python xgrammar contract for K3 XTML admission and decoding."""

import unittest

import xgrammar as xgr
from xgrammar.builtin_structural_tag import get_model_structural_tag

from rtp_llm.dash_sc.inference.grammar_validator import GrammarValidator


class KimiK3XGrammarTest(unittest.TestCase):
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

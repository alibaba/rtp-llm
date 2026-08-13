import copy
from pathlib import Path
from unittest import TestCase, main

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.chatglm47_renderer import ChatGlm47Renderer


def _tool_call(call_id, name="lookup", value="value"):
    tool_call = {
        "type": "function",
        "function": {"name": name, "arguments": f'{{"query": "{value}"}}'},
    }
    if call_id is not None:
        tool_call["id"] = call_id
    return tool_call


def _tool_result(call_id, content):
    result = {"role": "tool", "content": content}
    if call_id is not None:
        result["tool_call_id"] = call_id
    return result


def _history(calls, results):
    return [
        {"role": "user", "content": "look these up"},
        {"role": "assistant", "tool_calls": calls},
        *results,
    ]


class GlmToolHistoryTest(TestCase):
    def setUp(self):
        self.renderer = ChatGlm45Renderer.__new__(ChatGlm45Renderer)

    def test_parallel_results_are_ordered_by_tool_call_id(self):
        messages = _history(
            [_tool_call("call_a", value="a"), _tool_call("call_b", value="b")],
            [_tool_result("call_b", "result-b"), _tool_result("call_a", "result-a")],
        )
        original = copy.deepcopy(messages)

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(
            [item["content"] for item in processed[-2:]],
            ["result-a", "result-b"],
        )
        self.assertEqual(
            processed[1]["tool_calls"][0]["function"]["arguments"], {"query": "a"}
        )
        self.assertEqual(messages, original)

    def test_same_name_parallel_calls_still_use_ids(self):
        messages = _history(
            [_tool_call("call_a", value="first"), _tool_call("call_b", value="second")],
            [
                _tool_result("call_b", "second-result"),
                _tool_result("call_a", "first-result"),
            ],
        )

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(
            [item["content"] for item in processed[-2:]],
            ["first-result", "second-result"],
        )

    def test_each_tool_round_is_matched_locally(self):
        messages = _history(
            [_tool_call("round1_a"), _tool_call("round1_b")],
            [_tool_result("round1_b", "r1-b"), _tool_result("round1_a", "r1-a")],
        )
        messages.extend(
            [
                {"role": "assistant", "content": "continue"},
                {"role": "user", "content": "again"},
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call("round2_a"), _tool_call("round2_b")],
                },
                _tool_result("round2_b", "r2-b"),
                _tool_result("round2_a", "r2-a"),
            ]
        )

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(
            [message["content"] for message in processed if message["role"] == "tool"],
            ["r1-a", "r1-b", "r2-a", "r2-b"],
        )

    def test_single_legacy_history_without_ids_is_preserved(self):
        messages = _history(
            [_tool_call(None, value="a")],
            [_tool_result(None, "result-a")],
        )

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(processed[-1]["content"], "result-a")

    def test_rejects_single_half_ids(self):
        for call_id, result_id in (("call_a", None), (None, "call_a")):
            with self.subTest(call_id=call_id, result_id=result_id), self.assertRaises(
                ValueError
            ):
                self.renderer._preprocess_messages(
                    _history(
                        [_tool_call(call_id)],
                        [_tool_result(result_id, "result-a")],
                    )
                )

    def test_rejects_parallel_history_without_ids(self):
        with self.assertRaises(ValueError):
            self.renderer._preprocess_messages(
                _history(
                    [_tool_call(None, value="a"), _tool_call(None, value="b")],
                    [_tool_result(None, "result-a"), _tool_result(None, "result-b")],
                )
            )

    def test_glm5_renderer_inherits_history_association(self):
        renderer = ChatGlm47Renderer.__new__(ChatGlm47Renderer)
        processed = renderer._preprocess_messages(
            _history(
                [_tool_call("call_a"), _tool_call("call_b")],
                [
                    _tool_result("call_b", "result-b"),
                    _tool_result("call_a", "result-a"),
                ],
            )
        )
        self.assertEqual(
            [item["content"] for item in processed[-2:]],
            ["result-a", "result-b"],
        )

    def test_real_glm_prompt_is_identical_for_reversed_results(self):
        renderer = ChatGlm47Renderer.__new__(ChatGlm47Renderer)
        template_path = (
            Path(__file__).parent
            / "model_test/fake_test/testdata/glm45/tokenizer/chat_template.jinja"
        )
        renderer.chat_template = template_path.read_text()
        calls = [_tool_call("call_a", value="a"), _tool_call("call_b", value="b")]
        canonical = ChatCompletionRequest(
            messages=_history(
                calls,
                [
                    _tool_result("call_a", "result-a"),
                    _tool_result("call_b", "result-b"),
                ],
            )
        )
        reversed_results = ChatCompletionRequest(
            messages=_history(
                calls,
                [
                    _tool_result("call_b", "result-b"),
                    _tool_result("call_a", "result-a"),
                ],
            )
        )

        canonical_prompt = renderer._build_prompt(canonical)
        reversed_prompt = renderer._build_prompt(reversed_results)

        self.assertEqual(reversed_prompt, canonical_prompt)
        self.assertLess(
            canonical_prompt.index("result-a"), canonical_prompt.index("result-b")
        )

    def test_rejects_duplicate_call_ids(self):
        with self.assertRaises(ValueError):
            self.renderer._preprocess_messages(
                _history(
                    [_tool_call("call_a"), _tool_call("call_a")],
                    [_tool_result("call_a", "one"), _tool_result("call_b", "two")],
                )
            )

    def test_rejects_duplicate_result_ids(self):
        with self.assertRaises(ValueError):
            self.renderer._preprocess_messages(
                _history(
                    [_tool_call("call_a"), _tool_call("call_b")],
                    [_tool_result("call_a", "one"), _tool_result("call_a", "two")],
                )
            )

    def test_rejects_parallel_half_ids(self):
        cases = (
            (
                [_tool_call("call_a"), _tool_call(None)],
                [_tool_result("call_a", "one"), _tool_result("call_b", "two")],
            ),
            (
                [_tool_call("call_a"), _tool_call("call_b")],
                [_tool_result("call_a", "one"), _tool_result(None, "two")],
            ),
            (
                [_tool_call("call_a"), _tool_call("call_b")],
                [_tool_result(None, "one"), _tool_result(None, "two")],
            ),
        )
        for calls, results in cases:
            with self.subTest(calls=calls, results=results), self.assertRaises(
                ValueError
            ):
                self.renderer._preprocess_messages(_history(calls, results))

    def test_rejects_unknown_or_missing_results(self):
        cases = (
            [_tool_result("call_a", "one"), _tool_result("call_c", "three")],
            [_tool_result("call_a", "one")],
            [
                _tool_result("call_a", "one"),
                _tool_result("call_b", "two"),
                _tool_result("call_c", "three"),
            ],
        )
        calls = [_tool_call("call_a"), _tool_call("call_b")]
        for results in cases:
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.renderer._preprocess_messages(_history(calls, results))

    def test_rejects_mismatched_single_call_ids(self):
        with self.assertRaises(ValueError):
            self.renderer._preprocess_messages(
                _history([_tool_call("call_a")], [_tool_result("call_b", "result")])
            )

    def test_rejects_orphan_or_interrupted_results(self):
        cases = (
            [_tool_result("call_a", "orphan")],
            [
                {"role": "assistant", "tool_calls": [_tool_call("call_a")]},
                {"role": "user", "content": "interrupt"},
                _tool_result("call_a", "late"),
            ],
        )
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.renderer._preprocess_messages(messages)


if __name__ == "__main__":
    main()

import json
from unittest import IsolatedAsyncioTestCase, main

from pydantic import ValidationError

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    RoleEnum,
    ToolCall,
    UsageInfo,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject


async def _responses(*responses):
    for response in responses:
        yield response


def _logprobs(token):
    return ChoiceLogprobs(
        content=[
            ChatCompletionTokenLogprob(
                token=token,
                bytes=None,
                logprob=-0.1,
                top_logprobs=[],
            )
        ]
    )


def _choice(
    index,
    *,
    content=None,
    reasoning_content=None,
    tool_calls=None,
    finish_reason=None,
    logprobs=None,
):
    return ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
        logprobs=logprobs,
    )


def _tool_call(*, tool_id=None, name=None, arguments=None):
    return ToolCall(
        index=0,
        id=tool_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


class OpenaiEndpointCollectResponseTest(IsolatedAsyncioTestCase):
    async def _collect(self, *responses):
        return await OpenaiEndpoint._collect_complete_response(
            _responses(*responses),
            debug_info=None,
            model_name="test-model",
        )

    async def test_reordered_choices_merge_by_index_and_sort(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(1, content="B1", logprobs=_logprobs("b1")),
                    _choice(0, content="A1", logprobs=_logprobs("a1")),
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        content="A2",
                        finish_reason=FinisheReason.stop,
                        logprobs=_logprobs("a2"),
                    ),
                    _choice(
                        1,
                        content="B2",
                        finish_reason=FinisheReason.length,
                        logprobs=_logprobs("b2"),
                    ),
                ]
            ),
        )

        self.assertEqual([choice.index for choice in result.choices], [0, 1])
        self.assertEqual(
            [choice.message.content for choice in result.choices],
            ["A1A2", "B1B2"],
        )
        self.assertEqual(
            [choice.finish_reason for choice in result.choices],
            [FinisheReason.stop, FinisheReason.length],
        )
        self.assertEqual(
            [item.token for item in result.choices[0].logprobs.content],
            ["a1", "a2"],
        )
        self.assertEqual(
            [item.token for item in result.choices[1].logprobs.content],
            ["b1", "b2"],
        )

    async def test_sparse_choices_and_first_reasoning_are_preserved(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[_choice(1, reasoning_content="reason-1")]
            ),
            StreamResponseObject(choices=[]),
            StreamResponseObject(choices=[_choice(0, content="A")]),
            StreamResponseObject(
                choices=[
                    _choice(1, content="B", reasoning_content="reason-2")
                ]
            ),
        )

        self.assertEqual([choice.index for choice in result.choices], [0, 1])
        self.assertEqual(result.choices[0].message.content, "A")
        self.assertEqual(result.choices[1].message.content, "B")
        self.assertEqual(
            result.choices[1].message.reasoning_content,
            "reason-1reason-2",
        )
        self.assertEqual(result.choices[1].message.role, RoleEnum.assistant)

    async def test_reordered_tool_fragments_do_not_cross_choices(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(
                                tool_id="tool-a",
                                name="echo",
                                arguments='{"value":"A',
                            )
                        ],
                    ),
                    _choice(
                        1,
                        tool_calls=[
                            _tool_call(
                                tool_id="tool-b",
                                name="echo",
                                arguments='{"value":"B',
                            )
                        ],
                    ),
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(1, tool_calls=[_tool_call(arguments='"}')]),
                    _choice(0, tool_calls=[_tool_call(arguments='"}')]),
                ]
            ),
        )

        first_tool = result.choices[0].message.tool_calls[0]
        second_tool = result.choices[1].message.tool_calls[0]
        self.assertEqual(first_tool.id, "tool-a")
        self.assertEqual(second_tool.id, "tool-b")
        self.assertEqual(json.loads(first_tool.function.arguments), {"value": "A"})
        self.assertEqual(json.loads(second_tool.function.arguments), {"value": "B"})

    async def test_empty_metadata_chunk_is_allowed(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[],
                usage=UsageInfo(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                ),
            ),
            StreamResponseObject(choices=[_choice(0, content="done")]),
        )

        self.assertEqual(result.choices[0].message.content, "done")
        self.assertEqual(result.usage.total_tokens, 5)

    async def test_invalid_and_duplicate_choice_indexes_are_rejected(self):
        for invalid_index in (-1, True, 1.0, "3"):
            with self.subTest(index=invalid_index):
                with self.assertRaises(ValidationError):
                    _choice(invalid_index, content="invalid")

        invalid_choice = ChatCompletionResponseStreamChoice.model_construct(
            index=-1,
            delta=DeltaMessage(content="invalid"),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            await self._collect(StreamResponseObject(choices=[invalid_choice]))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(0, content="first"),
                        _choice(0, content="second"),
                    ]
                )
            )


if __name__ == "__main__":
    main()

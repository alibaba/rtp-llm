import json
from unittest import IsolatedAsyncioTestCase, main

from pydantic import ValidationError

from rtp_llm.openai.api_datatype import (
    ChatCompletionExtraOutputs,
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
from rtp_llm.utils.base_model_datatypes import AuxInfo


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
    function_call=None,
    tool_calls=None,
    finish_reason=None,
    logprobs=None,
):
    return ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(
            content=content,
            reasoning_content=reasoning_content,
            function_call=function_call,
            tool_calls=tool_calls,
        ),
        finish_reason=finish_reason,
        logprobs=logprobs,
    )


def _tool_call(*, index=0, tool_id=None, name=None, arguments=None):
    return ToolCall(
        index=index,
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

    async def test_legacy_function_call_fragments_are_merged(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        function_call=FunctionCall(
                            name="lookup",
                            arguments='{"city":"Bei',
                        ),
                    )
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        function_call=FunctionCall(name=None, arguments='jing"}'),
                        finish_reason=FinisheReason.function_call,
                    )
                ]
            ),
        )

        function_call = result.choices[0].message.function_call
        self.assertEqual(function_call.name, "lookup")
        self.assertEqual(json.loads(function_call.arguments), {"city": "Beijing"})

    async def test_missing_tool_index_merges_only_unambiguous_call(self):
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
                    )
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[_tool_call(index=None, arguments='"}')],
                    )
                ]
            ),
        )

        self.assertEqual(len(result.choices[0].message.tool_calls), 1)
        tool_call = result.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.index, 0)
        self.assertEqual(tool_call.id, "tool-a")
        self.assertEqual(json.loads(tool_call.function.arguments), {"value": "A"})

    async def test_missing_tool_identity_is_rejected_when_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "missing index and id"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=0, tool_id="tool-a", name="first"),
                                _tool_call(index=1, tool_id="tool-b", name="second"),
                            ],
                        )
                    ]
                ),
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[_tool_call(index=None, arguments="{}")],
                        )
                    ]
                ),
            )

    async def test_multiple_identityless_tool_deltas_in_one_chunk_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing index and id"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=None, name="first"),
                                _tool_call(index=None, name="second"),
                            ],
                        )
                    ]
                )
            )

    async def test_tool_fragments_can_continue_by_id(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(
                                index=3,
                                tool_id="tool-a",
                                name="echo",
                                arguments='{"value":"A',
                            )
                        ],
                    )
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(
                                index=None,
                                tool_id="tool-a",
                                arguments='"}',
                            )
                        ],
                    )
                ]
            ),
        )

        self.assertEqual(len(result.choices[0].message.tool_calls), 1)
        tool_call = result.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.index, 3)
        self.assertEqual(json.loads(tool_call.function.arguments), {"value": "A"})

    async def test_empty_tool_ids_do_not_alias_distinct_indexes(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(index=0, tool_id="", name="first"),
                            _tool_call(index=1, tool_id="", name="second"),
                        ],
                    )
                ]
            )
        )

        tool_calls = result.choices[0].message.tool_calls
        self.assertEqual([tool.index for tool in tool_calls], [0, 1])
        self.assertEqual(
            [tool.function.name for tool in tool_calls], ["first", "second"]
        )

    async def test_empty_tool_id_fragment_uses_unique_existing_call(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(
                                index=0,
                                tool_id="tool-a",
                                name="echo",
                                arguments="{",
                            )
                        ],
                    )
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(index=None, tool_id="", arguments="}")
                        ],
                    )
                ]
            ),
        )

        tool_calls = result.choices[0].message.tool_calls
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].id, "tool-a")
        self.assertEqual(tool_calls[0].function.arguments, "{}")

    async def test_tool_calls_are_sorted_by_index_across_chunks(self):
        result = await self._collect(
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(index=1, tool_id="tool-b", name="second")
                        ],
                    )
                ]
            ),
            StreamResponseObject(
                choices=[
                    _choice(
                        0,
                        tool_calls=[
                            _tool_call(index=0, tool_id="tool-a", name="first")
                        ],
                    )
                ]
            ),
        )

        tool_calls = result.choices[0].message.tool_calls
        self.assertEqual([tool.index for tool in tool_calls], [0, 1])
        self.assertEqual([tool.id for tool in tool_calls], ["tool-a", "tool-b"])

    async def test_multiple_id_only_tool_calls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing index"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(
                                    index=None, tool_id="tool-a", name="first"
                                ),
                                _tool_call(
                                    index=None, tool_id="tool-b", name="second"
                                ),
                            ],
                        )
                    ]
                )
            )

    async def test_late_identity_after_identityless_fragment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing index and id"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=None, name="echo", arguments="{")
                            ],
                        )
                    ]
                ),
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=2, tool_id="tool-a", arguments="}")
                            ],
                        )
                    ]
                ),
            )

    async def test_identityless_fragment_is_not_bound_by_later_list_order(self):
        with self.assertRaisesRegex(ValueError, "missing index and id"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=None, name="unknown", arguments="prefix")
                            ],
                        )
                    ]
                ),
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=0, tool_id="tool-a", name="first"),
                                _tool_call(index=1, tool_id="tool-b", name="second"),
                            ],
                        )
                    ]
                ),
            )

    async def test_partial_tool_identities_do_not_split_one_call(self):
        fragment_pairs = (
            (
                _tool_call(index=2, name="echo", arguments="{"),
                _tool_call(index=None, tool_id="tool-a", arguments="}"),
            ),
            (
                _tool_call(index=None, tool_id="tool-a", name="echo", arguments="{"),
                _tool_call(index=2, arguments="}"),
            ),
        )
        for first, second in fragment_pairs:
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "identity cannot be resolved"):
                    await self._collect(
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[first])]
                        ),
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[second])]
                        ),
                    )

    async def test_collect_does_not_mutate_input_fragments(self):
        first_function = FunctionCall(name="lookup", arguments="{")
        second_function = FunctionCall(name=None, arguments="}")
        first_tool = _tool_call(
            index=0,
            tool_id="tool-a",
            name="echo",
            arguments="{",
        )
        second_tool = _tool_call(index=None, arguments="}")

        first_response = StreamResponseObject(
            choices=[
                _choice(
                    0,
                    function_call=first_function,
                    tool_calls=[first_tool],
                )
            ]
        )
        second_response = StreamResponseObject(
            choices=[
                _choice(
                    0,
                    function_call=second_function,
                    tool_calls=[second_tool],
                )
            ]
        )
        before = [
            first_response.choices[0].model_dump_json(),
            second_response.choices[0].model_dump_json(),
        ]

        result = await self._collect(first_response, second_response)

        self.assertEqual(result.choices[0].message.function_call.arguments, "{}")
        self.assertEqual(
            result.choices[0].message.tool_calls[0].function.arguments, "{}"
        )
        self.assertEqual(first_function.arguments, "{")
        self.assertEqual(second_function.arguments, "}")
        self.assertEqual(first_tool.function.arguments, "{")
        self.assertEqual(second_tool.function.arguments, "}")
        self.assertIsNone(second_tool.index)
        self.assertEqual(
            before,
            [
                first_response.choices[0].model_dump_json(),
                second_response.choices[0].model_dump_json(),
            ],
        )

    async def test_conflicting_legacy_function_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting function call name"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            function_call=FunctionCall(
                                name="lookup", arguments="{"
                            ),
                        )
                    ]
                ),
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            function_call=FunctionCall(
                                name="delete", arguments="}"
                            ),
                        )
                    ]
                ),
            )

    async def test_conflicting_tool_name_and_type_are_rejected(self):
        conflict_deltas = (
            _tool_call(
                index=0,
                tool_id="tool-a",
                name="delete",
                arguments="}",
            ),
            ToolCall(
                index=0,
                id="tool-a",
                type="custom",
                function=FunctionCall(name="lookup", arguments="}"),
            ),
        )
        for conflict in conflict_deltas:
            with self.subTest(conflict=conflict):
                with self.assertRaisesRegex(ValueError, "conflicting tool call"):
                    await self._collect(
                        StreamResponseObject(
                            choices=[
                                _choice(
                                    0,
                                    tool_calls=[
                                        _tool_call(
                                            index=0,
                                            tool_id="tool-a",
                                            name="lookup",
                                            arguments="{",
                                        )
                                    ],
                                )
                            ]
                        ),
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[conflict])]
                        ),
                    )

    async def test_invalid_tool_indexes_are_rejected_at_collector_boundary(self):
        for invalid_index in (-1, True, 1.0, "3"):
            with self.subTest(index=invalid_index):
                invalid_tool = ToolCall.model_construct(
                    index=invalid_index,
                    id="tool-a",
                    type="function",
                    function=FunctionCall(name="lookup", arguments="{}"),
                )
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    await self._collect(
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[invalid_tool])]
                        )
                    )

    async def test_invalid_tool_ids_are_rejected_at_collector_boundary(self):
        for invalid_id in (True, 1, [], {}):
            with self.subTest(tool_id=invalid_id):
                invalid_tool = ToolCall.model_construct(
                    index=0,
                    id=invalid_id,
                    type="function",
                    function=FunctionCall(name="lookup", arguments="{}"),
                )
                with self.assertRaisesRegex(ValueError, "string or null"):
                    await self._collect(
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[invalid_tool])]
                        )
                    )

    async def test_duplicate_tool_identity_in_one_chunk_is_rejected(self):
        duplicate_pairs = (
            (
                _tool_call(index=0, tool_id="tool-a", name="lookup"),
                _tool_call(index=0, tool_id="tool-b", name="lookup"),
            ),
            (
                _tool_call(index=0, tool_id="tool-a", name="lookup"),
                _tool_call(index=1, tool_id="tool-a", name="lookup"),
            ),
        )
        for first, second in duplicate_pairs:
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, "duplicate tool call"):
                    await self._collect(
                        StreamResponseObject(
                            choices=[_choice(0, tool_calls=[first, second])]
                        )
                    )

    async def test_later_logprobs_are_deep_copied(self):
        first_logprobs = _logprobs("first")
        second_logprobs = _logprobs("second")
        result = await self._collect(
            StreamResponseObject(
                choices=[_choice(0, logprobs=first_logprobs)]
            ),
            StreamResponseObject(
                choices=[_choice(0, logprobs=second_logprobs)]
            ),
        )

        second_logprobs.content[0].token = "mutated"
        self.assertEqual(
            [item.token for item in result.choices[0].logprobs.content],
            ["first", "second"],
        )
        self.assertIsNot(
            result.choices[0].logprobs.content[1], second_logprobs.content[0]
        )

    async def test_response_metadata_objects_are_deep_copied(self):
        usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        aux_info = AuxInfo(cum_log_probs=[-0.5], multimodal_lengths={0: 7})
        extra_outputs = ChatCompletionExtraOutputs(output_ids=[[11, 12]])
        response = StreamResponseObject(
            choices=[_choice(0, content="done")],
            usage=usage,
            aux_info=aux_info,
            extra_outputs=extra_outputs,
        )

        result = await self._collect(response)
        usage.total_tokens = 99
        aux_info.cum_log_probs[0] = -9.0
        aux_info.multimodal_lengths[0] = 99
        extra_outputs.output_ids[0][0] = 99

        self.assertEqual(result.usage.total_tokens, 3)
        self.assertEqual(result.aux_info.cum_log_probs, [-0.5])
        self.assertEqual(result.aux_info.multimodal_lengths, {0: 7})
        self.assertEqual(result.extra_outputs.output_ids, [[11, 12]])

    async def test_conflicting_tool_index_and_id_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting index and id"):
            await self._collect(
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=0, tool_id="tool-a", name="first"),
                                _tool_call(index=1, tool_id="tool-b", name="second"),
                            ],
                        )
                    ]
                ),
                StreamResponseObject(
                    choices=[
                        _choice(
                            0,
                            tool_calls=[
                                _tool_call(index=0, tool_id="tool-b", arguments="{}")
                            ],
                        )
                    ]
                ),
            )

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
        for invalid_index in (-1, True, 1.0, "3"):
            invalid_choice = ChatCompletionResponseStreamChoice.model_construct(
                index=invalid_index,
                delta=DeltaMessage(content="invalid"),
            )
            with self.subTest(constructed_index=invalid_index):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    await self._collect(
                        StreamResponseObject(choices=[invalid_choice])
                    )

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

from unittest import IsolatedAsyncioTestCase, main

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    UsageInfo,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject


class OpenaiEndpointCollectTest(IsolatedAsyncioTestCase):
    async def test_joins_reasoning_and_content_fragments(self):
        async def response_generator():
            deltas = [
                DeltaMessage(role=RoleEnum.assistant, reasoning_content="first "),
                DeltaMessage(reasoning_content="second "),
                DeltaMessage(content="answer"),
            ]
            for index, delta in enumerate(deltas):
                yield StreamResponseObject(
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=delta,
                            finish_reason=(
                                FinisheReason.stop
                                if index == len(deltas) - 1
                                else None
                            ),
                        )
                    ],
                    usage=UsageInfo(completion_tokens=index + 1),
                )

        complete = await OpenaiEndpoint._collect_complete_response(
            response_generator(),
            debug_info=None,
        )

        self.assertEqual(
            complete.choices[0].message.reasoning_content,
            "first second ",
        )
        self.assertEqual(complete.choices[0].message.content, "answer")
        self.assertEqual(complete.choices[0].finish_reason, FinisheReason.stop)
        self.assertEqual(complete.usage.completion_tokens, 3)


if __name__ == "__main__":
    main()

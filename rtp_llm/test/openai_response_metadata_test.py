import itertools
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from rtp_llm.openai.api_datatype import (
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    UsageInfo,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject


def _choice(content: str) -> ChatCompletionResponseStreamChoice:
    return ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=content),
    )


async def _choice_stream(request_index: int):
    for chunk_index in range(3):
        yield StreamResponseObject(
            choices=[_choice(f"{request_index}:{chunk_index}")],
            usage=UsageInfo(
                prompt_tokens=1,
                completion_tokens=chunk_index + 1,
                total_tokens=chunk_index + 2,
            ),
        )


class OpenaiResponseMetadataTest(IsolatedAsyncioTestCase):
    async def test_interleaved_streams_use_request_local_metadata(self):
        clock = itertools.count(1_000)
        with patch(
            "rtp_llm.openai.api_datatype.time.time",
            side_effect=lambda: next(clock),
        ):
            streams = [
                OpenaiEndpoint._complete_stream_response(
                    _choice_stream(request_index),
                    debug_info=None,
                    model_name="test-model",
                )
                for request_index in range(64)
            ]

            chunks_by_request = [[] for _ in streams]
            for _ in range(3):
                for request_index, stream in enumerate(streams):
                    chunks_by_request[request_index].append(await anext(stream))

            aggregates = [
                await stream.gen_complete_response_once() for stream in streams
            ]

        request_ids = []
        for chunks, aggregate in zip(chunks_by_request, aggregates):
            metadata = {(response.id, response.created) for response in chunks}
            metadata.add((aggregate.id, aggregate.created))
            self.assertEqual(len(metadata), 1)
            request_ids.append(aggregate.id)

        self.assertEqual(len(set(request_ids)), len(request_ids))
        for stream in streams:
            await stream.aclose()

    async def test_non_stream_responses_use_distinct_metadata(self):
        first = await OpenaiEndpoint._collect_complete_response(
            _choice_stream(0),
            debug_info=None,
            model_name="test-model",
        )
        second = await OpenaiEndpoint._collect_complete_response(
            _choice_stream(1),
            debug_info=None,
            model_name="test-model",
        )

        self.assertNotEqual(first.id, second.id)


if __name__ == "__main__":
    main()

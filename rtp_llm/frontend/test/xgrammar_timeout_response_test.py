import unittest
from itertools import product

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.openai.api_datatype import UsageInfo
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import StreamResponseObject
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


class XgrammarTimeoutResponseTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_stream_and_complete_response(self):
        for num_sequences, aux_enabled in product((0, 2), (False, True)):
            with self.subTest(num_sequences=num_sequences, aux_enabled=aux_enabled):
                worker = object.__new__(FrontendWorker)
                config = GenerateConfig(
                    num_return_sequences=num_sequences, aux_info=aux_enabled
                )
                chunks = []
                for timed_out in (False, True):
                    outputs = [
                        GenerateOutput(
                            finished=timed_out,
                            aux_info=AuxInfo(xgrammar_update_timeout=timed_out),
                        )
                        for _ in range(max(num_sequences, 1))
                    ]
                    response = GenerateResponse(
                        generate_texts=["text"] * len(outputs),
                        generate_outputs=GenerateOutputs(generate_outputs=outputs),
                    )
                    chunk = worker._format_response_new(response, config)
                    chunks.append(chunk)
                    serialized_aux = chunk.model_dump(mode="json")["aux_info"]
                    if aux_enabled:
                        rows = serialized_aux if num_sequences else [serialized_aux]
                        self.assertTrue(
                            all(
                                row["xgrammar_update_timeout"] == timed_out
                                for row in rows
                            )
                        )
                    else:
                        self.assertFalse(serialized_aux)

                if not aux_enabled:
                    continue
                complete = await FrontendWorker.collect_complete_response(
                    CompleteResponseAsyncGenerator.generate_from_list(chunks),
                    incremental=True,
                    batch_infer=False,
                    num_return_sequences=num_sequences,
                )
                serialized_aux = complete.model_dump(mode="json")["aux_info"]
                rows = serialized_aux if num_sequences else [serialized_aux]
                self.assertTrue(all(row["xgrammar_update_timeout"] for row in rows))

    async def test_openai_stream_and_complete_response(self):
        for aux_enabled in (False, True):
            with self.subTest(aux_enabled=aux_enabled):

                async def choices():
                    for timed_out in (False, True):
                        yield StreamResponseObject(
                            aux_info=(
                                AuxInfo(xgrammar_update_timeout=timed_out)
                                if aux_enabled
                                else None
                            ),
                            usage=UsageInfo(
                                prompt_tokens=2, completion_tokens=2, total_tokens=4
                            ),
                        )

                responses = OpenaiEndpoint._complete_stream_response(choices(), None)
                chunks = [chunk async for chunk in responses]
                complete = await responses.gen_complete_response_once()
                for response, expected_timeout in zip(
                    [*chunks, complete], (False, True, True)
                ):
                    serialized = response.model_dump(mode="json", exclude_none=True)
                    if aux_enabled:
                        self.assertEqual(
                            serialized["aux_info"]["xgrammar_update_timeout"],
                            expected_timeout,
                        )
                    else:
                        self.assertNotIn("aux_info", serialized)


if __name__ == "__main__":
    unittest.main()

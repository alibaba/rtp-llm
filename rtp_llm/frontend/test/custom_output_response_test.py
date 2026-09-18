import asyncio
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_worker import BatchPipelineResponse, FrontendWorker
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


class CustomOutputResponseTest(unittest.TestCase):
    def format_response(self, values, count, batch=False):
        outputs = [
            GenerateOutput(
                finished=True,
                aux_info=AuxInfo(),
                custom_output=torch.tensor(value) if value is not None else None,
                input_ids=torch.tensor([[1, 2]]),
                output_ids=torch.tensor([[3]]),
            )
            for value in values
        ]
        response = GenerateResponse(
            generate_texts=["text"] * len(values),
            generate_outputs=GenerateOutputs(generate_outputs=outputs),
        )
        worker = object.__new__(FrontendWorker)
        config = GenerateConfig(
            num_return_sequences=count, return_input_ids=True, return_output_ids=True
        )
        if not batch:
            return worker._format_response_new(response, config)
        worker.generate_env_config = None
        worker.pipeline = SimpleNamespace(
            batch_infer=AsyncMock(return_value=[response]),
            tokenizer=[],
            _special_tokens=None,
            create_generate_config=lambda *args, **kwargs: config,
        )
        return asyncio.run(worker.batch_infer(["prompt"], 1, {})).response_batch[0]

    def test_single_multi_and_batch_preserve_scores_across_chunks(self):
        for count, batch in product((0, 1, 2), (False, True)):
            with self.subTest(count=count, batch=batch):
                values = [[[-0.5, 1.5]], [[0.0, 2.0]]][: max(count, 1)]
                first = self.format_response(values, count, batch)
                self.assertEqual(first.custom_output, values if count else values[0])
                if count == 0:
                    self.assertEqual(first.input_ids, [[1, 2]])
                    self.assertEqual(first.output_ids, [[3]])
                later = [[[0.0, 1.0]], [[2.0, 3.0]]][: max(count, 1)]
                absent = self.format_response([None] * max(count, 1), count, batch)
                chunks = [first, self.format_response(later, count, batch), absent]
                if batch:
                    chunks = [
                        BatchPipelineResponse(response_batch=[chunk])
                        for chunk in chunks
                    ]
                complete = asyncio.run(
                    FrontendWorker.collect_complete_response(
                        CompleteResponseAsyncGenerator.generate_from_list(chunks),
                        incremental=True,
                        batch_infer=batch,
                        num_return_sequences=count,
                    )
                )
                result = complete.response_batch[0] if batch else complete
                self.assertEqual(result.custom_output, later if count else later[0])
                self.assertNotIn("custom_output", absent.model_dump(exclude_none=True))

    def test_integer_output_keeps_json_number_type(self):
        result = self.format_response([[2147483647]], 0)
        self.assertIs(type(result.custom_output[0]), int)
        self.assertEqual(result.custom_output, [2147483647])


if __name__ == "__main__":
    unittest.main()

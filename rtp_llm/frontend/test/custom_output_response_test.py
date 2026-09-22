import asyncio
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_worker import (
    FrontendWorker,
    MultiSequencesPipelineResponse,
    PipelineResponse,
)
from rtp_llm.ops import RoleType
from rtp_llm.structure.request_constants import request_id_field_name
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
    def format_response(self, values, count, batch=False, num_beams=1):
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
            num_return_sequences=count,
            num_beams=num_beams,
            return_input_ids=True,
            return_output_ids=True,
        )
        if not batch:
            return worker._format_response_new(response, config)
        worker.generate_env_config = None
        worker.backend_rpc_server_visitor = SimpleNamespace(
            pd_sep_config=SimpleNamespace(role_type=RoleType.PDFUSION),
            host_service=SimpleNamespace(service_available=False),
        )
        worker.pipeline = SimpleNamespace(
            batch_infer_prepared=AsyncMock(return_value=[response]),
            tokenizer=[],
            _special_tokens=None,
            create_generate_config=lambda *args, **kwargs: config,
        )
        result = worker.inference(
            True,
            prompt_batch=["prompt"],
            generate_config=config.model_dump(),
            **{request_id_field_name: 1},
        )
        return asyncio.run(
            CompleteResponseAsyncGenerator.get_last_value(result)
        ).response_batch[0]

    def test_single_multi_preserve_prefill_output(self):
        for count, matrix in product((0, 1, 2), (False, True)):
            values = [[[-0.5, 1.5]], [[2.0, 3.0]]] if matrix else [[-0.5], [1.5]]
            values = values[: max(count, 1)]
            first = self.format_response(values, count)
            self.assertEqual(first.custom_output, values if count > 0 else values[0])
            absent = self.format_response([None] * max(count, 1), count)
            complete = asyncio.run(
                FrontendWorker.collect_complete_response(
                    CompleteResponseAsyncGenerator.generate_from_list([first, absent]),
                    incremental=True,
                    batch_infer=False,
                    num_return_sequences=count,
                )
            )
            self.assertEqual(complete.custom_output, first.custom_output)
            self.assertNotIn("custom_output", absent.model_dump(exclude_none=True))

    def test_batch_infer_uses_shared_response_contract(self):
        for count, with_scores in product((0, 1, 2), (False, True)):
            with self.subTest(count=count, with_scores=with_scores):
                values = [[[-0.5, 1.5]], [[2.0, 3.0]]][: max(count, 1)]
                if not with_scores:
                    values = [None] * len(values)
                result = self.format_response(values, count, batch=True)
                expected = self.format_response(values, count)
                self.assertEqual(result.model_dump(), expected.model_dump())
                self.assertIsInstance(
                    result,
                    MultiSequencesPipelineResponse if count > 0 else PipelineResponse,
                )
                self.assertEqual(
                    result.custom_output,
                    (values if count > 0 else values[0]) if with_scores else None,
                )
                if not with_scores:
                    self.assertNotIn(
                        "custom_output", result.model_dump(exclude_none=True)
                    )

    def test_batch_infer_preserves_beam_responses(self):
        result = self.format_response([None, None], 0, batch=True, num_beams=2)
        self.assertIsInstance(result, PipelineResponse)
        self.assertEqual(result.response, "text")
        self.assertEqual(result.aux_info["beam_responses"], ["text", "text"])

    def test_integer_output_keeps_json_number_type(self):
        result = self.format_response([[2147483647]], 0)
        self.assertIs(type(result.custom_output[0]), int)
        self.assertEqual(result.custom_output, [2147483647])


if __name__ == "__main__":
    unittest.main()

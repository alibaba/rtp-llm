import unittest

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_worker import BatchPipelineResponse, FrontendWorker
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)


class CustomOutputResponseTest(unittest.TestCase):
    def setUp(self):
        # Serialization needs no tokenizer or backend; invoke the real formatter.
        self.worker = object.__new__(FrontendWorker)

    def format_response(self, values, num_return_sequences):
        response = GenerateResponse(
            generate_texts=[str(index) for index in range(len(values))],
            generate_outputs=GenerateOutputs(
                generate_outputs=[
                    GenerateOutput(
                        finished=True,
                        aux_info=AuxInfo(),
                        custom_output=(
                            torch.tensor(value) if value is not None else None
                        ),
                    )
                    for value in values
                ]
            ),
        )
        return self.worker._format_response_new(
            response, GenerateConfig(num_return_sequences=num_return_sequences)
        )

    def test_raw_single_and_multi_sequence_preserve_values(self):
        values = [[[-0.5, 1.5]], [[0.25, 0.75]]]
        for count in (0, 1, 2):
            with self.subTest(num_return_sequences=count):
                selected = values[: max(count, 1)]
                response = self.format_response(selected, count)
                expected = selected if count else selected[0]
                self.assertEqual(response.model_dump()["custom_output"], expected)

    def test_multi_sequence_missing_value_keeps_index(self):
        response = self.format_response([[[0.25]], None, [[0.75]]], 3)
        self.assertEqual(
            response.model_dump()["custom_output"], [[[0.25]], None, [[0.75]]]
        )

    def test_unconfigured_output_is_absent(self):
        for count in (0, 1, 2):
            with self.subTest(num_return_sequences=count):
                response = self.format_response([None] * max(count, 1), count)
                self.assertNotIn(
                    "custom_output", response.model_dump(exclude_none=True)
                )

    def test_batch_response_preserves_single_and_multi_sequence_outputs(self):
        batch = BatchPipelineResponse(
            response_batch=[
                self.format_response([[[0.25]]], 0),
                self.format_response([[[0.5]], [[0.75]]], 2),
            ]
        )
        values = [
            item["custom_output"] for item in batch.model_dump()["response_batch"]
        ]
        self.assertEqual(values, [[[0.25]], [[[0.5]], [[0.75]]]])


if __name__ == "__main__":
    unittest.main()

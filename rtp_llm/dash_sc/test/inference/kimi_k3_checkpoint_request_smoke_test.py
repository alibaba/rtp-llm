"""K3 checkpoint-backed DashSc request through the public stream servicer.

The tokenizer, protobuf request, K3 request controls and response builder are
real. Only the unavailable K3 text-model forward is replayed.
"""

import json
import os
import struct
import unittest

import torch
from transformers import AutoTokenizer

from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.models.kimi_k3.kimi_k3_dash_sc_adapter import KimiK3DashScRequestAdapter
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class _TerminalBackend:
    def __init__(self, output_id):
        self.output_id = output_id
        self.last_input = None

    async def enqueue(self, generate_input):
        self.last_input = generate_input

        async def response():
            yield GenerateOutputs(
                generate_outputs=[
                    GenerateOutput(
                        output_ids=torch.tensor([self.output_id], dtype=torch.int32),
                        finished=True,
                        aux_info=AuxInfo(
                            input_len=generate_input.input_length,
                            output_len=1,
                            step_output_len=1,
                        ),
                    )
                ]
            )

        return response()


class _GrpcContext:
    def invocation_metadata(self):
        return ()

    def peer(self):
        return "ipv4:127.0.0.1:1234"

    def code(self):
        return None

    def is_active(self):
        return True

    def details(self):
        return ""


class KimiK3DashScCheckpointSmokeTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["K3_CKPT_PATH"], trust_remote_code=True
        )

    async def test_two_images_preserve_request_order_and_native_defaults(self):
        first_url = "https://example.com/first.png"
        second_url = "https://example.com/second.png"
        placeholder = self.tokenizer.encode("<|kimi_image_placeholder|>")
        media_pad = self.tokenizer.encode("<|media_pad|>")
        self.assertTrue(placeholder)
        self.assertEqual(len(media_pad), 1)
        prefix = self.tokenizer.encode("Describe: ")
        between = self.tokenizer.encode(" then ")
        suffix = self.tokenizer.encode(" please")
        original_ids = prefix + placeholder + between + placeholder + suffix
        expected_ids = prefix + media_pad + between + media_pad + suffix

        request = predict_v2_pb2.ModelInferRequest()
        request.id = "k3-two-images"
        request.model_name = "kimi_k3"
        tensor = request.inputs.add()
        tensor.name = "input_ids"
        tensor.datatype = "INT32"
        tensor.shape.append(len(original_ids))
        request.raw_input_contents.append(
            struct.pack(f"<{len(original_ids)}i", *original_ids)
        )
        request.parameters["enable_thinking"].bool_param = False
        request.parameters["payload"].string_param = json.dumps(
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"image": first_url},
                                {"text": "then"},
                                {"image": second_url},
                            ],
                        }
                    ]
                }
            }
        )
        output_id = self.tokenizer.encode("answer")[-1]
        backend = _TerminalBackend(output_id)
        servicer = DashScInferenceServicer(
            backend_visitor=backend,
            tokenizer=self.tokenizer,
            request_adapter=KimiK3DashScRequestAdapter(),
        )

        async def one_request():
            yield request

        responses = [
            response
            async for response in servicer.ModelStreamInfer(
                one_request(), _GrpcContext()
            )
        ]

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].error_message, "")
        self.assertIsNotNone(backend.last_input)
        self.assertEqual(backend.last_input.token_ids.reshape(-1).tolist(), expected_ids)
        self.assertEqual(
            [item.url for item in backend.last_input.mm_inputs],
            [first_url, second_url],
        )
        config = backend.last_input.generate_config
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_new_tokens, 131072)
        self.assertEqual(config.temperature, 0.6)
        self.assertEqual(config.top_p, 0.95)

    async def test_negative_top_logprobs_is_rejected_before_backend_enqueue(self):
        input_ids = self.tokenizer.encode("Question?")
        request = predict_v2_pb2.ModelInferRequest()
        request.id = "k3-negative-top-logprobs"
        request.model_name = "kimi_k3"
        tensor = request.inputs.add()
        tensor.name = "input_ids"
        tensor.datatype = "INT32"
        tensor.shape.append(len(input_ids))
        request.raw_input_contents.append(
            struct.pack(f"<{len(input_ids)}i", *input_ids)
        )
        request.parameters["logprobs"].bool_param = True
        request.parameters["top_logprobs"].int64_param = -1
        backend = _TerminalBackend(self.tokenizer.encode("answer")[-1])
        servicer = DashScInferenceServicer(
            backend_visitor=backend,
            tokenizer=self.tokenizer,
            request_adapter=KimiK3DashScRequestAdapter(),
        )

        async def one_request():
            yield request

        responses = [
            response
            async for response in servicer.ModelStreamInfer(
                one_request(), _GrpcContext()
            )
        ]
        self.assertEqual(len(responses), 1)
        payload = json.loads(
            responses[0].infer_response.parameters["error_msg"].string_param
        )
        self.assertEqual(payload["status_code"], 400)
        self.assertIn("top_logprobs", payload["status_message"])
        self.assertIsNone(backend.last_input)


if __name__ == "__main__":
    unittest.main()

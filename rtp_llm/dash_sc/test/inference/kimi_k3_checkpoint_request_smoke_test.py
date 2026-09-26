"""K3 checkpoint-backed DashSc request through the public stream servicer.

The tokenizer, protobuf request, K3 request controls and response builder are
real. Only the unavailable K3 text-model forward is replayed.
"""

import base64
import io
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import grpc
import torch
from PIL import Image
from transformers import AutoTokenizer

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    MMUrlType,
)


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


class KimiK3DashScCheckpointSmokeTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def image_data_url(color, size):
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        data = buffer.getvalue()
        return "data:image/png;base64," + base64.b64encode(data).decode(), data

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ["K3_CKPT_PATH"], trust_remote_code=True
        )

    async def send_grpc_request(self, servicer, request):
        server = grpc.aio.server()
        predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
            servicer, server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        self.assertGreater(port, 0)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

                async def one_request():
                    yield request

                return [
                    response
                    async for response in stub.ModelStreamInfer(one_request())
                ]
        finally:
            await server.stop(0)

    async def test_two_images_preserve_request_order_and_native_defaults(self):
        first_url, first_bytes = self.image_data_url((128, 64, 32), (28, 28))
        second_url, second_bytes = self.image_data_url((32, 64, 128), (56, 28))
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
            model_type="kimi_k3",
        )

        responses = await self.send_grpc_request(servicer, request)

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

        # Continue this same DashSc request through the production K3 visual
        # mixin and C++ placeholder expansion. The replay backend only stands
        # in for the separately owned K3 text-model forward.
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        model_config = ModelConfig()
        model_config.model_type = "kimi_k3"
        model_config.ckpt_path = str(checkpoint)
        model_config.data_type = "bf16"
        configure_kimi_k3_multimodal(
            model_config, json.loads((checkpoint / "config.json").read_text())
        )
        self.assertEqual(model_config.mm_model_config.mm_sep_tokens, [media_pad])

        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.disable_access_log = True
        vit_config.mm_cache_item_num = 0
        engine_config = SimpleNamespace(
            load_config=SimpleNamespace(load_method=LoadMethod.AUTO),
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        )
        engine = MultimodalMixinFactory.create_multimodal_process_engine(
            model_config, engine_config, vit_config, device="cuda:0"
        )
        self.addCleanup(engine.stop)
        urls = [item.url for item in backend.last_input.mm_inputs]
        data = [first_bytes, second_bytes]
        for url, image_bytes in zip(urls, data):
            self.assertEqual(base64.b64decode(url.partition(",")[2]), image_bytes)
        result = engine.mm_embedding_cpp(
            ["", ""],
            [MMUrlType.IMAGE, MMUrlType.IMAGE],
            [torch.tensor(list(image_bytes), dtype=torch.uint8) for image_bytes in data],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000] for _ in data],
        )
        self.assertEqual(len(result.embeddings), 2)
        self.assertEqual(result.position_ids, [])
        self.assertEqual(result.extra_input, [])
        pad_id = media_pad[0]
        for image_bytes, features, vision_tokens in zip(
            data, result.embeddings, (1, 2)
        ):
            image = Image.open(io.BytesIO(image_bytes))
            prompt_ids = self.tokenizer.encode(
                engine.mm_part.image_processor.make_image_prompt(*image.size)
            )
            pad_index = prompt_ids.index(pad_id)
            self.assertEqual(prompt_ids.count(pad_id), 1)
            self.assertEqual(
                tuple(features.shape),
                (len(prompt_ids) - 1 + vision_tokens, 7168),
            )
            self.assertTrue(torch.isfinite(features).all().item())
            torch.testing.assert_close(
                features[:pad_index].cpu(),
                engine.mm_part._word_embedding_weight[prompt_ids[:pad_index]].to(
                    features.dtype
                ),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                features[pad_index + vision_tokens :].cpu(),
                engine.mm_part._word_embedding_weight[prompt_ids[pad_index + 1 :]].to(
                    features.dtype
                ),
                rtol=0,
                atol=0,
            )

        binary = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/cpp/multimodal_processor/test/kimi_k3_feature_expand_check"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            token_ids = backend.last_input.token_ids.reshape(-1).to(torch.int32).cpu()
            token_path = Path(temp_dir) / "token_ids.raw"
            token_path.write_bytes(token_ids.numpy().tobytes())
            command = [
                str(binary),
                str(pad_id),
                str(token_path),
                str(token_ids.numel()),
            ]
            for index, (url, features) in enumerate(zip(urls, result.embeddings)):
                feature_path = Path(temp_dir) / f"image_{index}.raw"
                feature_path.write_bytes(
                    features.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                command.extend(
                    (
                        url,
                        str(int(MMUrlType.IMAGE)),
                        str(feature_path),
                        str(features.shape[0]),
                    )
                )
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

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
            model_type="kimi_k3",
        )

        responses = await self.send_grpc_request(servicer, request)
        self.assertEqual(len(responses), 1)
        payload = json.loads(
            responses[0].infer_response.parameters["error_msg"].string_param
        )
        self.assertEqual(payload["status_code"], 400)
        self.assertIn("top_logprobs", payload["status_message"])
        self.assertIsNone(backend.last_input)


if __name__ == "__main__":
    unittest.main()

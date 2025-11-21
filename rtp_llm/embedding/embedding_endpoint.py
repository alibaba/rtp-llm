import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

import grpc
import numpy as np
import torch

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.utils import create_custom_module


def tensor_pb_to_torch(tensor_pb) -> Optional[torch.Tensor]:
    data_type = tensor_pb.data_type
    shape = list(tensor_pb.shape)
    if data_type == tensor_pb.DataType.FP32:
        dtype = torch.float32
        buffer = tensor_pb.fp32_data
    elif data_type == tensor_pb.DataType.INT32:
        dtype = torch.int32
        buffer = tensor_pb.int32_data
    elif data_type == tensor_pb.DataType.FP16:
        dtype = torch.float16
        buffer = tensor_pb.fp16_data
    elif data_type == tensor_pb.DataType.BF16:
        dtype = torch.bfloat16
        buffer = tensor_pb.bf16_data
    else:
        logging.warning(f"Unsupported data type: {data_type}")
        return None
    if not buffer:
        logging.warning("Empty data buffer")
        return None
    try:
        torch_tensor = torch.frombuffer(buffer, dtype=dtype).clone().reshape(shape)
        return torch_tensor
    except Exception as e:
        logging.warning(f"Failed to convert buffer to tensor: {e}")
        return None


class EmbeddingEndpoint(object):
    def __init__(self, config: GptInitModelParameters, tokenizer: BaseTokenizer):
        self.renderer = create_custom_module(
            config.task_type, config, tokenizer
        ).renderer
        # 创建到服务器的连接

        self.address = f"localhost:{g_worker_info.embedding_rpc_server_port}"
        logging.info(f"embedding endpoint connect to rpc addresses: {self.address}")

    async def handle(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if isinstance(request, str):
            request = json.loads(request)
        try:
            formate_request = self.renderer.render_request(request)
            batch_input = self.renderer.create_input(formate_request)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
        try:
            batch_output = await self.decode(batch_input)
            response = await self.renderer.render_response(
                formate_request, batch_input, batch_output
            )
            logable_response = await self.renderer.render_log_response(response)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, str(e))
        return response, logable_response

    async def decode(self, input: EngineInputs):
        output = EngineOutputs(outputs=None, input_length=0)
        await asyncio.to_thread(self.decode_grpc, input, output)
        return output

    def decode_grpc(self, input: EngineInputs, output: EngineOutputs):
        options = [
            ("grpc.max_metadata_size", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(self.address, options=options)
        stub = pb2_grpc.EmbeddingRpcServiceStub(channel)
        multimodal_features = []
        for feature in input.multimodal_inputs:
            multimodal_features.append(
                pb2.MultimodalInputPB(
                    multimodal_type=feature.mm_type, multimodal_url=feature.url
                )
            )
        request = pb2.EmbeddingInputPB(
            token_ids=input.token_ids.tolist(),  # 示例token ids
            token_type_ids=input.token_type_ids.tolist(),  # 示例segment ids
            input_lengths=input.input_lengths.tolist(),  # 输入长度
            request_id=1,  # 唯一请求ID
            multimodal_features=multimodal_features,
        )
        try:
            # 调用 decode 方法（返回一个响应流）
            response = stub.decode(request)  # 注意这里是流式响应
            if response.output_is_tensor:
                tensor_pb = response.output_t
                result = tensor_pb_to_torch(tensor_pb)
            else:
                result = []
                for output_map_iter in response.output_map:
                    tensor_map = {}
                    for key, tensor_pb in output_map_iter.items():
                        torch_tensor = tensor_pb_to_torch(tensor_pb)
                        tensor_map[key] = torch_tensor
                    result.append(tensor_map)
            output.outputs = result
            output.input_length = input.input_length
            return result
        except grpc.RpcError as e:
            logging.warning(f"RPC failed: {e.code()}: {e.details()}")

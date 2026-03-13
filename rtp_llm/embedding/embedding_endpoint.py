import json
import logging
from typing import Any, Dict, Optional, Tuple

import grpc
import numpy as np
import torch
from fastapi import Request
from fastapi.responses import ORJSONResponse

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.base_endpoint import BaseEndpoint
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.concurrency_controller import ConcurrencyController
from rtp_llm.utils.util import AtomicCounter

USAGE_HEADER = "USAGE"


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


class EmbeddingEndpoint(BaseEndpoint):
    def __init__(
        self,
        model_config,
        grpc_config,
        server_config,
        tokenizer: BaseTokenizer,
        global_controller: ConcurrencyController,
        access_logger: AccessLogger,
        rank_id: str = "0",
        server_id: str = "0",
        active_requests: Optional[AtomicCounter] = None,
    ):
        super().__init__(
            global_controller=global_controller,
            access_logger=access_logger,
            rank_id=rank_id,
            server_id=server_id,
            active_requests=active_requests,
        )
        self.renderer = create_custom_module(model_config, tokenizer).renderer
        # 创建到服务器的连接
        self.address = f"localhost:{server_config.embedding_rpc_server_port}"
        logging.info(f"embedding endpoint connect to rpc addresses: {self.address}")
        self.options = []
        client_config = grpc_config.get_client_config()
        if client_config is not None:
            for key, value in client_config.items():
                self.options.append((key, value))
        logging.info(f"embedding endpoint grpc options: {self.options}")

    def _check_request(self, request: Any, req_id: int) -> Dict[str, Any]:
        if isinstance(request, str):
            request = json.loads(request)
        assert isinstance(request, dict)
        request[request_id_field_name] = req_id
        return request

    async def _handle_request_impl(
        self, request: Any, raw_request: Request
    ) -> ORJSONResponse:
        """Embedding path: obtain request_dict, then embedding() + _finish_with_response. Single try/except."""
        request_dict = None
        try:
            request_dict = await self._obtain_request_dict(request, raw_request)

            result, logable_result = await self.embedding(request_dict)
            usage = result.get("usage", {}) or {}
            headers = {USAGE_HEADER: json.dumps(usage)}
            return self._finish_with_response(
                request_dict,
                result,
                logable_for_log=logable_result,
                headers=headers,
            )
        except BaseException as e:
            self._global_controller.decrement()
            req_for_error = (
                request_dict
                if request_dict is not None
                else self._convert_to_dict(request)
            )
            return self._handle_exception(req_for_error, e)

    async def embedding(
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
            output = EngineOutputs(outputs=None, input_length=0)
            channel = grpc.aio.insecure_channel(self.address, options=self.options)
            stub = pb2_grpc.EmbeddingRpcServiceStub(channel)
            multimodal_features = [
                pb2.MultimodalInputPB(multimodal_type=f.mm_type, multimodal_url=f.url)
                for f in batch_input.multimodal_inputs
            ]
            pb_request = pb2.EmbeddingInputPB(
                token_ids=batch_input.token_ids.tolist(),
                token_type_ids=batch_input.token_type_ids.tolist(),
                input_lengths=batch_input.input_lengths.tolist(),
                request_id=1,
                multimodal_features=multimodal_features,
            )
            try:
                grpc_response = await stub.embedding(pb_request)
                if grpc_response.output_is_tensor:
                    result = tensor_pb_to_torch(grpc_response.output_t)
                else:
                    result = []
                    for output_map_iter in grpc_response.output_map:
                        tensor_map = {
                            k: tensor_pb_to_torch(tensor_pb)
                            for k, tensor_pb in output_map_iter.items()
                        }
                        result.append(tensor_map)
                output.outputs = result
                output.input_length = batch_input.input_length
            except grpc.RpcError as e:
                logging.warning(f"RPC failed: {e.code()}: {e.details()}")
                raise
            finally:
                await channel.close()

            response = await self.renderer.render_response(
                formate_request, batch_input, output
            )
            logable_response = await self.renderer.render_log_response(response)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, str(e))
        return response, logable_response

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import grpc
import numpy as np
import torch

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import (
    EmbeddingInputException,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.py_config_modules import EmbeddingConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.utils import create_custom_module
from rtp_llm.ops import RoleType
from rtp_llm.server.host_service import HostService, HostServiceArgs
from rtp_llm.utils.grpc_util import trans_from_tensor

# Do not sit in a long reconnect backoff after a backend restart.
_SHARED_CHANNEL_GRPC_OPTIONS = (
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.min_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 1000),
)
# Aggressive keepalive is opt-in: a missed ACK can fail an active RPC even
# without an RPC deadline, while the paused backend may later finish its work.
_KEEPALIVE_GRPC_OPTIONS = (
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
)


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
    def __init__(
        self,
        model_config,
        grpc_config,
        server_config,
        tokenizer: BaseTokenizer,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.renderer = create_custom_module(model_config, tokenizer).renderer
        # 创建到服务器的连接
        self.address = f"localhost:{server_config.embedding_rpc_server_port}"
        logging.info(f"embedding endpoint connect to rpc addresses: {self.address}")
        self.options = []
        client_config = grpc_config.get_client_config()
        if client_config is not None:
            for key, value in client_config.items():
                self.options.append((key, value))
        host_args = HostServiceArgs.create_from_env()
        self.host_service = HostService(host_args)

        embedding_config = embedding_config or EmbeddingConfig()
        # Reuse one HTTP/2 connection per event loop instead of a new channel per request.
        self.share_channel: bool = embedding_config.embedding_share_grpc_channel
        # The RPC deadline is opt-in. EmbeddingRpcServer does not observe gRPC
        # cancellation yet, so an expired deadline only fails the frontend call
        # while the backend still finishes the request.
        rpc_timeout_s = embedding_config.embedding_rpc_timeout_s
        self.rpc_timeout: Optional[float] = (
            rpc_timeout_s if rpc_timeout_s is not None and rpc_timeout_s > 0 else None
        )
        # Only wait for a reconnecting channel when a deadline bounds the wait;
        # otherwise fail fast like the per-request channel does.
        self.wait_for_ready: bool = self.share_channel and self.rpc_timeout is not None
        channel_options = _SHARED_CHANNEL_GRPC_OPTIONS if self.share_channel else ()
        if embedding_config.embedding_grpc_keepalive:
            channel_options += _KEEPALIVE_GRPC_OPTIONS
        option_keys = {key for key, _ in self.options}
        for key, value in channel_options:
            if key not in option_keys:
                self.options.append((key, value))
        # id(loop) -> (loop, channel). The loop is kept to detect a reused id.
        self._channels: Dict[
            int, Tuple[asyncio.AbstractEventLoop, grpc.aio.Channel]
        ] = {}
        logging.info(
            f"embedding endpoint grpc options: {self.options}, "
            f"share_channel: {self.share_channel}, rpc_timeout: {self.rpc_timeout}, "
            f"wait_for_ready: {self.wait_for_ready}"
        )

    def _get_shared_channel(self) -> grpc.aio.Channel:
        # grpc.aio channels are bound to the event loop that created them.
        loop = asyncio.get_running_loop()
        entry = self._channels.get(id(loop))
        if entry is not None and entry[0] is loop:
            return entry[1]
        # Drop channels of loops that are gone; they cannot be awaited any more.
        for loop_id, (owner_loop, _) in list(self._channels.items()):
            if owner_loop.is_closed():
                self._channels.pop(loop_id, None)
        channel = grpc.aio.insecure_channel(self.address, options=self.options)
        self._channels[id(loop)] = (loop, channel)
        return channel

    async def close(self) -> None:
        """Close this loop's channel and prune entries whose owner loop is closed."""
        current_loop = asyncio.get_running_loop()
        for loop_id, (owner_loop, channel) in list(self._channels.items()):
            if owner_loop is current_loop:
                del self._channels[loop_id]
                await channel.close()
            elif owner_loop.is_closed():
                self._channels.pop(loop_id, None)

    async def embedding(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if isinstance(request, str):
            request = json.loads(request)
        try:
            profile_config = self._extract_profile_config(request)
            formate_request = self.renderer.render_request(request)
            batch_input = self.renderer.create_input(formate_request)
        except EmbeddingInputException:
            raise
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
        try:
            batch_output = await self.generate_embeddings(batch_input, profile_config)
            response = await self.renderer.render_response(
                formate_request, batch_input, batch_output
            )
            logable_response = await self.renderer.render_log_response(response)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, str(e))
        return response, logable_response

    @staticmethod
    def _extract_profile_config(request: Dict[str, Any]) -> Dict[str, Any]:
        def as_dict(value):
            return value if isinstance(value, dict) else {}

        def as_bool(value, default=False):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value) if value is not None else default

        def as_int(value, default=1):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        extra_configs = as_dict(request.get("extra_configs"))
        generate_config = as_dict(request.get("generate_config"))
        config = {**extra_configs, **generate_config}
        return {
            "gen_timeline": as_bool(config.get("gen_timeline", False)),
            "profile_step": as_int(config.get("profile_step", 1)),
            "profile_trace_name": re.sub(
                r"[^A-Za-z0-9_-]", "", str(config.get("profile_trace_name", ""))
            ),
        }

    async def generate_embeddings(
        self, input: EngineInputs, profile_config: Optional[Dict[str, Any]] = None
    ):
        output = EngineOutputs(outputs=None, input_length=0)
        await self.generate_embeddings_grpc(input, output, profile_config)
        return output

    async def generate_embeddings_grpc(
        self,
        input: EngineInputs,
        output: EngineOutputs,
        profile_config: Optional[Dict[str, Any]] = None,
    ):
        profile_config = profile_config or {}
        if self.share_channel:
            channel = self._get_shared_channel()
        else:
            channel = grpc.aio.insecure_channel(self.address, options=self.options)
        stub = pb2_grpc.EmbeddingRpcServiceStub(channel)
        multimodal_features = []

        vit_role_addr = ""
        if input.multimodal_inputs and self.host_service:
            role_addrs = self.host_service.get_backend_role_addrs([RoleType.VIT])
            if role_addrs:
                vit_role_addr = role_addrs[0].ip + ":" + str(role_addrs[0].grpc_port)

        for feature in input.multimodal_inputs:
            preprocess_config = pb2.MMPreprocessConfigPB(
                width=feature.mm_preprocess_config.width,
                height=feature.mm_preprocess_config.height,
                min_pixels=feature.mm_preprocess_config.min_pixels,
                max_pixels=feature.mm_preprocess_config.max_pixels,
                fps=feature.mm_preprocess_config.fps,
                min_frames=feature.mm_preprocess_config.min_frames,
                max_frames=feature.mm_preprocess_config.max_frames,
                crop_positions=feature.mm_preprocess_config.crop_positions,
                mm_timeout_ms=feature.mm_preprocess_config.mm_timeout_ms,
            )
            multimodal_features.append(
                pb2.MultimodalInputPB(
                    multimodal_type=feature.mm_type,
                    multimodal_url=feature.url,
                    multimodal_tensor=trans_from_tensor(feature.tensor),
                    mm_preprocess_config=preprocess_config,
                )
            )
        request = pb2.EmbeddingInputPB(
            token_ids=input.token_ids.tolist(),  # 示例token ids
            token_type_ids=input.token_type_ids.tolist(),  # 示例segment ids
            input_lengths=input.input_lengths.tolist(),  # 输入长度
            request_id=1,  # 唯一请求ID
            multimodal_features=multimodal_features,
            vit_role_addr=vit_role_addr,
            gen_timeline=profile_config.get("gen_timeline", False),
            profile_step=profile_config.get("profile_step", 1),
            profile_trace_name=profile_config.get("profile_trace_name", ""),
        )
        try:
            response = await stub.embedding(
                request, timeout=self.rpc_timeout, wait_for_ready=self.wait_for_ready
            )
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
            raise
        finally:
            if not self.share_channel:
                await channel.close()

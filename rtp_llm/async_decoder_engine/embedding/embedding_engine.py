import asyncio
import logging
from typing import Optional

import grpc
import numpy as np
import torch
from typing_extensions import override

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.ops import MultimodalInputCpp, RtpEmbeddingOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


def tensor_pb_to_torch(tensor_pb) -> Optional[torch.Tensor]:
    # 获取数据类型和形状
    data_type = tensor_pb.data_type
    shape = list(tensor_pb.shape)  # 转换为 list[int]

    # 根据数据类型选择对应的字节字段
    if data_type == tensor_pb.DataType.FP32:
        dtype_np = np.float32
        buffer = tensor_pb.fp32_data
    elif data_type == tensor_pb.DataType.INT32:
        dtype_np = np.int32
        buffer = tensor_pb.int32_data
    elif data_type == tensor_pb.DataType.FP16:
        # FP16 需要特殊处理（numpy 没有 float16 的 frombuffer 直接支持）
        buffer = tensor_pb.fp16_data
        np_array = np.frombuffer(buffer, dtype=np.uint16).view(np.float16)
        torch_tensor = torch.from_numpy(np_array).reshape(shape)
        return torch_tensor
    elif data_type == tensor_pb.DataType.BF16:
        # BF16 需要转换为 float32 再处理
        buffer = tensor_pb.bf16_data
        np_array = np.frombuffer(buffer, dtype=np.uint16)
        np_array = np_array.astype(np.float32) / 32768.0  # 简化的 BF16 转换
        torch_tensor = torch.from_numpy(np_array).reshape(shape)
        return torch_tensor
    else:
        print(f"Unsupported data type: {data_type}")
        return None

    # 检查数据是否为空
    if not buffer:
        print("Empty data buffer")
        return None

    # 将字节数据转换为 numpy 数组
    try:
        np_array = np.frombuffer(buffer, dtype=dtype_np)
    except Exception as e:
        print(f"Failed to parse buffer: {e}")
        return None

    # 检查形状是否匹配
    if np.prod(shape) != np_array.size:
        print(f"Shape {shape} does not match data size {np_array.size}")
        return None

    # 转换为 PyTorch Tensor 并调整形状
    torch_tensor = torch.from_numpy(np_array).reshape(shape)
    return torch_tensor


class EmbeddingCppEngine(BaseEngine):
    def __init__(self, model):
        logging.info("creating cpp embedding engine")
        self.model = model
        assert (
            self.model.custom_module is not None
        ), "embedding custom module should not be None"
        # self.cpp_handler = self.model.custom_module.create_cpp_handler()
        self.cpp_engine = RtpEmbeddingOp()

    @override
    def stop(self) -> None:
        self.cpp_engine.stop()

    @override
    def start(self):
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model)
        else:
            self.mm_engine = None
        self.cpp_engine.init(self.model, self.mm_engine)
        # self.model.custom_module.handler.init_cpp_handler()

    def decode_grpc(self, input: EngineInputs):
        channel = grpc.insecure_channel("localhost:27001")
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
            return result
        except grpc.RpcError as e:
            print(f"RPC failed: {e.code()}: {e.details()}")

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        multimodal_inputs = [
            MultimodalInputCpp(i.url, i.tensor, int(i.mm_type))
            for i in inputs.multimodal_inputs
        ]
        results = self.decode_grpc(inputs)
        # results = self.cpp_engine.decode(
        #     inputs.token_ids,
        #     inputs.token_type_ids,
        #     inputs.input_lengths,
        #     0,
        #     multimodal_inputs,
        # )
        outputs.outputs = results
        outputs.input_length = inputs.input_length

    @override
    async def decode(self, input: EngineInputs) -> EngineOutputs:
        output = EngineOutputs(outputs=None, input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output

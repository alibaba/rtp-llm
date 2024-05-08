import asyncio
import sys

from typing import Any
import grpc

from maga_transformer.cpp.proto.model_rpc_service_pb2_grpc import ModelRpcServiceStub
from maga_transformer.models.base_model import GenerateInput, GenerateOutput, AuxInfo
from maga_transformer.cpp.proto.model_rpc_service_pb2 import TensorPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateConfigPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateInputPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import AuxInfoPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateOutputPB
import numpy as np
import torch
from maga_transformer.distribute.worker_info import g_master_info


def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.token_ids.extend(input_py.token_ids.tolist())
    generate_config_pb = GenerateConfigPB()
    generate_config_pb = input_pb.generate_config

    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.num_return_sequences = input_py.generate_config.num_return_sequences
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    trans_option(generate_config_pb, input_py.generate_config, "top_k")
    trans_option(generate_config_pb, input_py.generate_config, "top_p")
    trans_option(generate_config_pb, input_py.generate_config, "temperature")
    trans_option(generate_config_pb, input_py.generate_config,
                 "repetition_penalty")
    trans_option(generate_config_pb, input_py.generate_config, "random_seed")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_decay")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_min")
    trans_option(generate_config_pb, input_py.generate_config,
                 "top_p_reset_ids")
    trans_option(generate_config_pb, input_py.generate_config, "task_id")
    trans_option(generate_config_pb, input_py.generate_config, "adapter_name")

    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_incremental = input_py.generate_config.return_incremental
    generate_config_pb.return_hidden_states = input_py.generate_config.return_hidden_states

    trans_option(input_pb, input_py, "lora_id")
    input_pb.prefix_length = input_py.prefix_length
    return input_pb


def trans_tensor(t: TensorPB):
    if t.data_type == TensorPB.DataType.FLOAT32:
        return torch.tensor(list(t.data_float32),
                            dtype=torch.float32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.INT32:
        return torch.tensor(list(t.data_int32),
                            dtype=torch.int32).reshape(list(t.shape))
    else:
        raise Exception("unkown error type")


def trans_output(output_pb: GenerateOutputPB):
    output_py = GenerateOutput()
    output_py.finished = output_pb.finished
    output_py.aux_info = AuxInfo(cost_time=output_pb.aux_info.cost_time_ms,
                                 iter_count=output_pb.aux_info.iter_count,
                                 input_len=output_pb.aux_info.input_len,
                                 output_len=output_pb.aux_info.output_len,
                                 reuse_len=output_pb.aux_info.reuse_len)
    output_py.output_ids = trans_tensor(output_pb.output_ids)
    if output_pb.HasField('hidden_states'):
        output_py.hidden_states = trans_tensor(output_pb.hidden_states)
    if output_pb.HasField('loss'):
        output_py.loss = trans_tensor(output_pb.loss)
    if output_pb.HasField('logits'):
        output_py.logits = trans_tensor(output_pb.logits)
    return output_py


class ModelRpcClient(object):

    def __init__(self, address: str = f'localhost:{g_master_info.model_rpc_port}'):
        # 创建到服务器的连接
        self._address = address

    async def enqueue(self, input: GenerateInput) -> GenerateOutput:
        input_pb = trans_input(input)
        response_iterator = None
        try:
            async with grpc.aio.insecure_channel(self._address) as channel:
                stub = ModelRpcServiceStub(channel)
                response_iterator = stub.generate_stream(input_pb)
                # 调用服务器方法并接收流式响应
                count = 0
                async for response in response_iterator.__aiter__():
                    count += 1
                    yield trans_output(response)
                    # print(f"Received response:{type(response)} {response.finished}")
        except grpc.RpcError as e:
            if response_iterator:
                response_iterator.cancel()
            print(f"RPC failed: {e.code()}, {e.details()}")

    def stop(self):
        self.rtp_llm_op.stop()


if __name__ == '__main__':
    client = ModelRpcClient()
    input = GenerateInput()
    asyncio.run(client.generate_stream(input))

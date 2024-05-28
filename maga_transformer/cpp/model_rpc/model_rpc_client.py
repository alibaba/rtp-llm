
import sys
from typing import Any, Optional
import asyncio
import numpy as np
import logging
import grpc
import torch

from maga_transformer.utils.util import AtomicCounter
from maga_transformer.cpp.proto.model_rpc_service_pb2_grpc import ModelRpcServiceStub
from maga_transformer.models.base_model import GenerateInput, GenerateOutput, GenerateOutputs, AuxInfo
from maga_transformer.cpp.proto.model_rpc_service_pb2 import TensorPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateConfigPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateInputPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import AuxInfoPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateOutputPB, GenerateOutputsPB
from maga_transformer.distribute.worker_info import g_master_info
from maga_transformer.utils.model_weight import LoraResource, LoraResourceHolder

request_counter = AtomicCounter()

def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.request_id = request_counter.increment()
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())
    input_pb.lora_id = input_py.lora_id

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.num_return_sequences = input_py.generate_config.num_return_sequences
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    generate_config_pb.top_k = input_py.generate_config.top_k
    generate_config_pb.top_p = input_py.generate_config.top_p 
    generate_config_pb.temperature = input_py.generate_config.temperature
    generate_config_pb.repetition_penalty = input_py.generate_config.repetition_penalty
    trans_option(generate_config_pb, input_py.generate_config, "random_seed")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_decay")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_min")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_reset_ids")
    trans_option(generate_config_pb, input_py.generate_config, "task_id")

    generate_config_pb.select_tokens_id.extend(input_py.generate_config.select_tokens_id)
    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_incremental = input_py.generate_config.return_incremental
    generate_config_pb.return_hidden_states = input_py.generate_config.return_hidden_states
    generate_config_pb.is_streaming = input_py.generate_config.is_streaming
    generate_config_pb.timeout_ms = input_py.generate_config.timeout_ms
    
    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    return input_pb


def trans_tensor(t: TensorPB):
    if t.data_type == TensorPB.DataType.FP32:
        return torch.frombuffer(t.fp32_data, dtype=torch.float32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.INT32:
        return torch.frombuffer(t.int32_data, dtype=torch.int32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.FP16:
        return torch.frombuffer(t.fp16_data, dtype=torch.float16).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.BF16:
        return torch.frombuffer(t.bf16_data, dtype=torch.bfloat16).reshape(list(t.shape))
    else:
        raise Exception("unkown error type")
    

def trans_output(input_py: GenerateInput, outputs_pb: GenerateOutputsPB):
    logging.debug("outputs_pb = ", outputs_pb)
    outputs_py = GenerateOutputs()
    for output_pb in outputs_pb.generate_outputs:
        output_py = GenerateOutput()
        output_py.finished = output_pb.finished
        output_py.aux_info = AuxInfo(cost_time=output_pb.aux_info.cost_time_us / 1000.0,
                                    iter_count=output_pb.aux_info.iter_count,
                                    input_len=output_pb.aux_info.input_len,
                                    reuse_len=output_pb.aux_info.reuse_len,
                                    prefix_len=output_pb.aux_info.prefix_len,
                                    output_len=output_pb.aux_info.output_len
                                    )
        if output_pb.aux_info.HasField('cum_log_probs'):
            output_py.aux_info.cum_log_probs = trans_tensor(output_pb.aux_info.cum_log_probs).tolist()
        output_py.output_ids = trans_tensor(output_pb.output_ids)
        output_py.input_ids = input_py.token_ids.reshape(1, -1)
        if output_pb.HasField('hidden_states'):
            output_py.hidden_states = trans_tensor(output_pb.hidden_states)
        if output_pb.HasField('loss'):
            output_py.loss = trans_tensor(output_pb.loss)
        if output_pb.HasField('logits'):
            output_py.logits = trans_tensor(output_pb.logits)
        outputs_py.generate_outputs.append(output_py)
        
    return outputs_py


class ModelRpcClient(object):

    def __init__(self, lora_resource: LoraResource, address: Optional[str] = None):
        self._lora_resource = lora_resource
        # 创建到服务器的连接
        if not address:
            address = f'localhost:{g_master_info.model_rpc_port}'
        self._address = address

    async def enqueue(self, input: GenerateInput) -> GenerateOutput:
        lora_resource_holder = None
        if input.generate_config.adapter_name is not None:
            lora_resource_holder = LoraResourceHolder(self._lora_resource, input.generate_config.adapter_name)
            input.lora_id = lora_resource_holder.lora_id
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
                    yield trans_output(input, response)
        except grpc.RpcError as e:
            # TODO(xinfei.sxf) 非流式的请求无法取消了
            if response_iterator:
                response_iterator.cancel()
            logging.warning(f"request: [{input_pb.request_id}] RPC failed: {e.code()}, {e.details()}")
            raise e
        finally:
           if lora_resource_holder is not None:
               self.lora_resource_holder.release()

    def stop(self):
        self.rtp_llm_op.stop()


if __name__ == '__main__':
    client = ModelRpcClient()
    input = GenerateInput()
    asyncio.run(client.generate_stream(input))

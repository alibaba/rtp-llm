
import sys
from typing import Any, Optional, AsyncGenerator
import asyncio
import numpy as np
import functools
import logging
import torch
import grpc
from grpc import StatusCode

from maga_transformer.utils.util import AtomicCounter
from maga_transformer.cpp.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from maga_transformer.models.base_model import GenerateInput, GenerateOutput, GenerateOutputs, AuxInfo
from maga_transformer.cpp.proto.model_rpc_service_pb2 import TensorPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import MMPreprocessConfigPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import MulitmodalInputPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateInputPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import GenerateOutputsPB
from maga_transformer.cpp.proto.model_rpc_service_pb2 import ErrorDetailsPB
from maga_transformer.distribute.worker_info import g_master_info
from maga_transformer.distribute.worker_info import g_worker_info
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

request_counter = AtomicCounter()

MAX_GRPC_TIMEOUT_SECONDS = 60 * 3

def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)

def trans_option_cast(pb_object, py_object, name, func):
    if getattr(py_object, name):
        getattr(pb_object, name).value = func(getattr(py_object, name))

def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    # The stream id cannot use the request id because the request may contain prompt batch.
    input_pb.request_id = request_counter.increment()
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())

    trans_multimodal_input(input_py, input_pb)

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.num_return_sequences = input_py.generate_config.num_return_sequences
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    generate_config_pb.top_k = input_py.generate_config.top_k
    generate_config_pb.top_p = input_py.generate_config.top_p
    generate_config_pb.temperature = input_py.generate_config.temperature
    generate_config_pb.sp_edit = input_py.generate_config.sp_edit
    generate_config_pb.force_disable_sp_run = input_py.generate_config.force_disable_sp_run
    generate_config_pb.repetition_penalty = input_py.generate_config.repetition_penalty
    trans_option(generate_config_pb, input_py.generate_config, "no_repeat_ngram_size")
    trans_option(generate_config_pb, input_py.generate_config, "random_seed")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_decay")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_min")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_reset_ids")
    trans_option(generate_config_pb, input_py.generate_config, "adapter_name")
    trans_option_cast(generate_config_pb, input_py.generate_config, "task_id", functools.partial(str))

    generate_config_pb.select_tokens_id.extend(input_py.generate_config.select_tokens_id)
    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_incremental = input_py.generate_config.return_incremental
    generate_config_pb.return_hidden_states = input_py.generate_config.return_hidden_states
    generate_config_pb.is_streaming = input_py.generate_config.is_streaming
    generate_config_pb.timeout_ms = input_py.generate_config.timeout_ms
    if input_py.generate_config.sp_advice_prompt_token_ids:
        generate_config_pb.sp_advice_prompt_token_ids.extend(input_py.generate_config.sp_advice_prompt_token_ids)
    generate_config_pb.return_all_probs = input_py.generate_config.return_all_probs
    generate_config_pb.can_use_pd_separation = input_py.generate_config.can_use_pd_separation
    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    return input_pb

def trans_multimodal_input(input_py: GenerateInput, input_pb: GenerateInputPB):
    for mm_input in input_py.mm_inputs:
        mm_input_pb = MulitmodalInputPB()
        mm_input_pb.multimodal_url = mm_input.url
        mm_input_pb.multimodal_type = mm_input.mm_type
        mm_preprocess_config_pb = mm_input_pb.mm_preprocess_config
        mm_preprocess_config_pb.width = mm_input.config.width
        mm_preprocess_config_pb.height = mm_input.config.height
        mm_preprocess_config_pb.min_pixels = mm_input.config.min_pixels
        mm_preprocess_config_pb.max_pixels = mm_input.config.max_pixels
        mm_preprocess_config_pb.fps = mm_input.config.fps
        mm_preprocess_config_pb.min_frames = mm_input.config.min_frames
        mm_preprocess_config_pb.max_frames = mm_input.config.max_frames
        input_pb.multimodal_inputs.append(mm_input_pb)

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


def trans_output(input_py: GenerateInput, outputs_pb: GenerateOutputsPB) -> GenerateOutputs:
    logging.debug("outputs_pb = " +  str(outputs_pb))
    outputs_py = GenerateOutputs()
    for output_pb in outputs_pb.generate_outputs:
        output_py = GenerateOutput()
        output_py.finished = output_pb.finished
        output_py.aux_info = AuxInfo(cost_time=output_pb.aux_info.cost_time_us / 1000.0,
                                    iter_count=output_pb.aux_info.iter_count,
                                    input_len=output_pb.aux_info.input_len,
                                    reuse_len=output_pb.aux_info.reuse_len,
                                    prefix_len=output_pb.aux_info.prefix_len,
                                    output_len=output_pb.aux_info.output_len,
                                    step_output_len=output_pb.aux_info.step_output_len,
                                    fallback_tokens=output_pb.aux_info.fallback_tokens,
                                    fallback_times=output_pb.aux_info.fallback_times,
                                    pd_sep=output_pb.aux_info.pd_sep,
                                    first_token_cost_time=output_pb.aux_info.first_token_cost_time_us / 1000.0
                                    )
        # TODO(xinfei.sxf) cum_log_probs is not right, ignore it temporarily
        if output_pb.aux_info.HasField('cum_log_probs'):
            output_py.aux_info.cum_log_probs = trans_tensor(output_pb.aux_info.cum_log_probs).tolist()
        output_py.output_ids = trans_tensor(output_pb.output_ids)
        output_py.input_ids = input_py.token_ids.reshape(1, -1)
        if output_pb.HasField('hidden_states'):
            output_py.hidden_states = trans_tensor(output_pb.hidden_states)
        if output_pb.HasField('loss'):
            # when calculate_loss 1, result should be one element
            if input_py.generate_config.calculate_loss == 1:
                output_py.loss = trans_tensor(output_pb.loss)[0]
            else:
                output_py.loss = trans_tensor(output_pb.loss)
        if output_pb.HasField('logits'):
            output_py.logits = trans_tensor(output_pb.logits)
        if output_pb.HasField('all_probs'):
            output_py.all_probs = trans_tensor(output_pb.all_probs)
        outputs_py.generate_outputs.append(output_py)

    return outputs_py


class ModelRpcClient(object):

    def __init__(self, config: GptInitModelParameters, address: Optional[str] = None):
        # 创建到服务器的连接
        if not address:
            address = f'localhost:{g_worker_info.rpc_server_port}'
        logging.info("client connect to rpc address: " + address)
        self._address = address
        self.model_config = config

    async def enqueue(self, input: GenerateInput) -> AsyncGenerator[GenerateOutputs, None]:
        input_pb = trans_input(input)
        response_iterator = None
        request_timeout_ms = input.generate_config.timeout_ms
        rpc_timeout_ms = self.model_config.max_rpc_timeout_ms \
                            if self.model_config.max_rpc_timeout_ms > 0 else MAX_GRPC_TIMEOUT_SECONDS
        if request_timeout_ms == None or request_timeout_ms <= 0:
            grpc_timeout_seconds = rpc_timeout_ms
        else:
            grpc_timeout_seconds = request_timeout_ms / 1000

        try:
            async with grpc.aio.insecure_channel(self._address) as channel:
                stub = RpcServiceStub(channel)
                response_iterator = stub.GenerateStreamCall(input_pb, timeout=grpc_timeout_seconds)
                # 调用服务器方法并接收流式响应
                count = 0
                async for response in response_iterator.__aiter__():
                    count += 1
                    yield trans_output(input, response)
        except grpc.RpcError as e:
            # TODO(xinfei.sxf) 非流式的请求无法取消了
            if response_iterator:
                response_iterator.cancel()
            error_details = ErrorDetailsPB()
            metadata = e.trailing_metadata()
            if 'grpc-status-details-bin' in metadata and error_details.ParseFromString(metadata['grpc-status-details-bin']):
                logging.error(f"request: [{input_pb.request_id}] RPC failed: "
                              f"{e.code()}, {e.details()}, detail error code is "
                              f"{ExceptionType.from_value(error_details.error_code)}")
                raise FtRuntimeException(ExceptionType(error_details.error_code), error_details.error_message)
            else:
                logging.error(f"request: [{input_pb.request_id}] RPC failed: "
                              f"error code is {e.code()}, detail is {e.details()}")
                if e.code() == StatusCode.DEADLINE_EXCEEDED:
                    raise FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, e.details())
                elif e.code() == StatusCode.CANCELLED:
                    raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
                else:
                    raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())
        except Exception as e:
            logging.error(f'rpc unknown error:{str(e)}')
            raise e
        finally:
            if response_iterator:
                response_iterator.cancel()

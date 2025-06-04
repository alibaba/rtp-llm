import sys
import os
from typing import Any, Optional, AsyncGenerator
import asyncio
import numpy as np
import functools
import logging
import torch
import grpc
from grpc import StatusCode

from rtp_llm.utils.util import AtomicCounter
from rtp_llm.cpp.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.models.base_model import GenerateInput, GenerateOutput, GenerateOutputs, AuxInfo, GenerateConfig
from rtp_llm.cpp.proto.model_rpc_service_pb2 import TensorPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import MMPreprocessConfigPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import MultimodalInputPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import GenerateOutputsPB
from rtp_llm.cpp.proto.model_rpc_service_pb2 import ErrorDetailsPB
from rtp_llm.distribute.worker_info import g_master_info, WorkerInfo
from rtp_llm.distribute.worker_info import g_worker_info, g_parallel_info
from rtp_llm.config.exceptions import FtRuntimeException, ExceptionType
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.utils.grpc_util import trans_option, trans_option_cast, trans_tensor
from rtp_llm.distribute.gang_info import get_gang_info, GangInfo
from rtp_llm.utils.concurrency_controller import ConcurrencyException, get_global_controller
from rtp_llm.utils.multimodal_util import maybe_hash_url

MAX_GRPC_TIMEOUT_SECONDS = 3600

def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.request_id = input_py.request_id
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())

    trans_multimodal_input(input_py, input_pb, input_py.generate_config)

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.max_thinking_tokens = input_py.generate_config.max_thinking_tokens
    generate_config_pb.end_think_token_ids.extend(input_py.generate_config.end_think_token_ids)
    generate_config_pb.in_think_mode = input_py.generate_config.in_think_mode
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.num_return_sequences = input_py.generate_config.num_return_sequences
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    generate_config_pb.top_k = input_py.generate_config.top_k
    generate_config_pb.top_p = input_py.generate_config.top_p
    generate_config_pb.temperature = input_py.generate_config.temperature
    generate_config_pb.sp_edit = input_py.generate_config.sp_edit
    generate_config_pb.force_disable_sp_run = input_py.generate_config.force_disable_sp_run
    generate_config_pb.force_sp_accept = input_py.generate_config.force_sp_accept
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
    generate_config_pb.return_cum_log_probs = input_py.generate_config.return_cum_log_probs
    generate_config_pb.return_all_probs = input_py.generate_config.return_all_probs
    generate_config_pb.return_softmax_probs = input_py.generate_config.return_softmax_probs
    generate_config_pb.can_use_pd_separation = input_py.generate_config.can_use_pd_separation
    generate_config_pb.gen_timeline = input_py.generate_config.gen_timeline
    generate_config_pb.global_request_id = input_py.generate_config.global_request_id

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    return input_pb

def trans_multimodal_input(input_py: GenerateInput, input_pb: GenerateInputPB, generate_config: GenerateConfig):
    resized_shape = [-1, -1]
    if generate_config.resized_shape:
        if len(generate_config.resized_shape) != 2:
            logging.info("Resized shape must be a list with 2 positive int, refering width and height")
        else:
            resized_shape = generate_config.resized_shape
    for mm_input in input_py.mm_inputs:
        mm_input_pb = MultimodalInputPB()
        mm_input_pb.multimodal_url = maybe_hash_url(mm_input.url)
        mm_input_pb.multimodal_type = mm_input.mm_type
        mm_preprocess_config_pb = mm_input_pb.mm_preprocess_config
        mm_preprocess_config_pb.width = mm_input.config.width if mm_input.config.width != -1 else resized_shape[0]
        mm_preprocess_config_pb.height = mm_input.config.height if mm_input.config.height != -1 else resized_shape[1]
        mm_preprocess_config_pb.min_pixels = mm_input.config.min_pixels
        mm_preprocess_config_pb.max_pixels = mm_input.config.max_pixels
        mm_preprocess_config_pb.fps = mm_input.config.fps
        mm_preprocess_config_pb.min_frames = mm_input.config.min_frames
        mm_preprocess_config_pb.max_frames = mm_input.config.max_frames
        input_pb.multimodal_inputs.append(mm_input_pb)


def trans_output(input_py: GenerateInput, outputs_pb: GenerateOutputsPB) -> GenerateOutputs:
    logging.debug("outputs_pb = " +  str(outputs_pb))
    outputs_py = GenerateOutputs()
    for output_pb in outputs_pb.generate_outputs:
        output_py = GenerateOutput()
        output_py.finished = output_pb.finished
        output_py.aux_info = AuxInfo(cost_time=output_pb.aux_info.cost_time_us / 1000.0,
                                    first_token_cost_time=output_pb.aux_info.first_token_cost_time_us / 1000.0,
                                    wait_time=output_pb.aux_info.wait_time_us / 1000.0,
                                    iter_count=output_pb.aux_info.iter_count,
                                    input_len=output_pb.aux_info.input_len,
                                    reuse_len=output_pb.aux_info.reuse_len,
                                    prefix_len=output_pb.aux_info.prefix_len,
                                    output_len=output_pb.aux_info.output_len,
                                    step_output_len=output_pb.aux_info.step_output_len,
                                    fallback_tokens=output_pb.aux_info.fallback_tokens,
                                    fallback_times=output_pb.aux_info.fallback_times,
                                    pd_sep=output_pb.aux_info.pd_sep)
        # TODO(xinfei.sxf) cum_log_probs is not right, ignore it temporarily
        if output_pb.aux_info.HasField('cum_log_probs'):
            output_py.aux_info.cum_log_probs = trans_tensor(output_pb.aux_info.cum_log_probs).tolist()
        if output_pb.aux_info.HasField('softmax_probs'):
            output_py.aux_info.softmax_probs = trans_tensor(output_pb.aux_info.softmax_probs).tolist()
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
        self._addresses = []
        # for test usage
        hack_ep_single_entry = bool(int(os.environ.get('HACK_EP_SINGLE_ENTRY', 0)))
        logging.info(f"hack ep single entry: {hack_ep_single_entry}")
        if (g_parallel_info.dp_size > 1) and (not hack_ep_single_entry):
            members_info_str = f"[world_rank: {g_parallel_info.world_rank}]"+ \
                f"[tp_size: {g_parallel_info.tp_size}] all members: " + "{"
            members = get_gang_info().members
            for member in members:
                members_info_str += f"{member}\n"
                if member.local_rank % g_parallel_info.tp_size == 0:
                    self._addresses.append(f'{member.ip}:{member.rpc_server_port}')
            members_info_str += "}"
            logging.info(f"{members_info_str}")
        else:
            self._addresses = [address]
        logging.info(f"client connect to rpc addresses: {self._addresses}")
        self.model_config = config

    async def enqueue(self, input_py: GenerateInput) -> AsyncGenerator[GenerateOutputs, None]:
        request_timeout_ms = input_py.generate_config.timeout_ms
        rpc_timeout_ms = self.model_config.max_rpc_timeout_ms \
                            if self.model_config.max_rpc_timeout_ms > 0 else MAX_GRPC_TIMEOUT_SECONDS * 1000
        if request_timeout_ms == None or request_timeout_ms <= 0:
            grpc_timeout_seconds = rpc_timeout_ms / 1000
        else:
            grpc_timeout_seconds = request_timeout_ms / 1000
        input_py.generate_config.timeout_ms = (int)(grpc_timeout_seconds * 1000)
        input_pb = trans_input(input_py)
        response_iterator = None
        try:
            async with grpc.aio.insecure_channel(self._addresses[input_py.request_id % len(self._addresses)]) as channel:
                stub = RpcServiceStub(channel)
                response_iterator = stub.GenerateStreamCall(input_pb, timeout=grpc_timeout_seconds)
                # 调用服务器方法并接收流式响应
                count = 0
                async for response in response_iterator.__aiter__():
                    count += 1
                    yield trans_output(input_py, response)
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

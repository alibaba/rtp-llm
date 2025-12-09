import asyncio
import functools
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Optional, List, Any, Tuple

import grpc
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleType
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.distribute.worker_info import g_parallel_info, g_worker_info
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateConfig,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.utils.grpc_util import trans_option, trans_option_cast, trans_tensor

MAX_GRPC_TIMEOUT_SECONDS = 3600

# Thread pool for executing gRPC streams in isolated threads
# max_workers = ceil((concurrency_limit / max(1, frontend_server_count)) * 1.2)
_concurrency_limit = StaticConfig.concurrency_config.concurrency_limit
_frontend_server_count = max(1, StaticConfig.server_config.frontend_server_count)
_calculated_workers = math.ceil((_concurrency_limit / _frontend_server_count) * 1.2)
GRPC_STREAM_EXECUTOR = ThreadPoolExecutor(
    max_workers=_calculated_workers,
    thread_name_prefix="grpc-stream-"
)
logging.info(f"GRPC_STREAM_EXECUTOR workers: {_calculated_workers}")


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}


def trans_role_type(role_type: RoleType) -> RoleAddrPB.RoleType:
    if role_type == RoleType.PDFUSION:
        return RoleAddrPB.RoleType.PDFUSION
    elif role_type == RoleType.PREFILL:
        return RoleAddrPB.RoleType.PREFILL
    elif role_type == RoleType.DECODE:
        return RoleAddrPB.RoleType.DECODE
    elif role_type == RoleType.VIT:
        return RoleAddrPB.RoleType.VIT
    elif role_type == RoleType.FRONTEND:
        return RoleAddrPB.RoleType.FRONTEND


def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.request_id = input_py.request_id
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())

    trans_multimodal_input(input_py, input_pb, input_py.generate_config)
    # check generate config is valid before enter into engine
    input_py.generate_config.validate()

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.max_thinking_tokens = (
        input_py.generate_config.max_thinking_tokens
    )
    generate_config_pb.end_think_token_ids.extend(
        input_py.generate_config.end_think_token_ids
    )
    generate_config_pb.in_think_mode = input_py.generate_config.in_think_mode
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.variable_num_beams.extend(
        input_py.generate_config.variable_num_beams
    )
    generate_config_pb.num_return_sequences = (
        input_py.generate_config.num_return_sequences
    )
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    generate_config_pb.top_k = input_py.generate_config.top_k
    generate_config_pb.top_p = input_py.generate_config.top_p
    generate_config_pb.temperature = input_py.generate_config.temperature
    generate_config_pb.sp_edit = input_py.generate_config.sp_edit
    generate_config_pb.force_disable_sp_run = (
        input_py.generate_config.force_disable_sp_run
    )
    generate_config_pb.force_sp_accept = input_py.generate_config.force_sp_accept
    generate_config_pb.repetition_penalty = input_py.generate_config.repetition_penalty
    generate_config_pb.presence_penalty = input_py.generate_config.presence_penalty
    generate_config_pb.frequency_penalty = input_py.generate_config.frequency_penalty
    generate_config_pb.do_sample = input_py.generate_config.do_sample
    trans_option(generate_config_pb, input_py.generate_config, "no_repeat_ngram_size")
    trans_option(generate_config_pb, input_py.generate_config, "random_seed")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_decay")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_min")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_reset_ids")
    trans_option(generate_config_pb, input_py.generate_config, "adapter_name")
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "task_id", functools.partial(str)
    )

    generate_config_pb.select_tokens_id.extend(
        input_py.generate_config.select_tokens_id
    )
    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_incremental = input_py.generate_config.return_incremental
    generate_config_pb.return_hidden_states = (
        input_py.generate_config.return_hidden_states
    )
    generate_config_pb.return_all_hidden_states = (
        input_py.generate_config.return_all_hidden_states
    )
    generate_config_pb.hidden_states_cut_dim = (
        input_py.generate_config.hidden_states_cut_dim
    )
    generate_config_pb.normalized_hidden_states = (
        input_py.generate_config.normalized_hidden_states
    )
    generate_config_pb.is_streaming = input_py.generate_config.is_streaming
    generate_config_pb.timeout_ms = input_py.generate_config.timeout_ms
    if input_py.generate_config.sp_advice_prompt_token_ids:
        generate_config_pb.sp_advice_prompt_token_ids.extend(
            input_py.generate_config.sp_advice_prompt_token_ids
        )
    generate_config_pb.return_cum_log_probs = (
        input_py.generate_config.return_cum_log_probs
    )
    generate_config_pb.return_all_probs = input_py.generate_config.return_all_probs
    generate_config_pb.return_softmax_probs = (
        input_py.generate_config.return_softmax_probs
    )
    generate_config_pb.can_use_pd_separation = (
        input_py.generate_config.can_use_pd_separation
    )
    generate_config_pb.gen_timeline = input_py.generate_config.gen_timeline
    generate_config_pb.profile_step = input_py.generate_config.profile_step
    generate_config_pb.global_request_id = input_py.generate_config.global_request_id
    generate_config_pb.inter_request_id = input_py.generate_config.inter_request_id
    generate_config_pb.ignore_eos = input_py.generate_config.ignore_eos
    generate_config_pb.reuse_cache = input_py.generate_config.reuse_cache
    generate_config_pb.enable_3fs = input_py.generate_config.enable_3fs
    generate_config_pb.enable_memory_block_cache = (
        input_py.generate_config.enable_memory_block_cache
    )

    trans_option_cast(
        generate_config_pb, input_py.generate_config, "trace_id", functools.partial(str)
    )

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    for role_addr in input_py.generate_config.role_addrs:
        role_addr_pb = RoleAddrPB()
        role_addr_pb.role = trans_role_type(role_addr.role)
        role_addr_pb.ip = role_addr.ip
        role_addr_pb.http_port = role_addr.http_port
        role_addr_pb.grpc_port = role_addr.grpc_port

        generate_config_pb.role_addrs.append(role_addr_pb)

    return input_pb


def trans_multimodal_input(
    input_py: GenerateInput, input_pb: GenerateInputPB, generate_config: GenerateConfig
):
    resized_shape = [-1, -1]
    if generate_config.resized_shape:
        if len(generate_config.resized_shape) != 2:
            logging.info(
                "Resized shape must be a list with 2 positive int, refering width and height"
            )
        else:
            resized_shape = generate_config.resized_shape
    for mm_input in input_py.mm_inputs:
        mm_input_pb = MultimodalInputPB()
        mm_input_pb.multimodal_url = mm_input.url
        mm_input_pb.multimodal_type = mm_input.mm_type
        mm_preprocess_config_pb = mm_input_pb.mm_preprocess_config
        mm_preprocess_config_pb.width = (
            mm_input.config.width if mm_input.config.width != -1 else resized_shape[0]
        )
        mm_preprocess_config_pb.height = (
            mm_input.config.height if mm_input.config.height != -1 else resized_shape[1]
        )
        mm_preprocess_config_pb.min_pixels = mm_input.config.min_pixels
        mm_preprocess_config_pb.max_pixels = mm_input.config.max_pixels
        mm_preprocess_config_pb.fps = mm_input.config.fps
        mm_preprocess_config_pb.min_frames = mm_input.config.min_frames
        mm_preprocess_config_pb.max_frames = mm_input.config.max_frames
        input_pb.multimodal_inputs.append(mm_input_pb)


# 假设 trans_tensor 函数将 Protobuf 的 TensorPB 转换为 numpy array
# from .utils import trans_tensor


def trans_output(
    input_py: GenerateInput, outputs_pb: GenerateOutputsPB, stream_state: StreamState
) -> GenerateOutputs:
    logging.debug("outputs_pb = %s", outputs_pb)
    output_pb = outputs_pb.flatten_output
    num_outputs = len(output_pb.finished)

    if num_outputs == 0:
        return GenerateOutputs()

    logits_index = input_py.generate_config.logits_index
    aux_info_flag = input_py.generate_config.aux_info

    all_output_ids = (
        trans_tensor(output_pb.output_ids)
        if output_pb.HasField("output_ids")
        and (len(output_pb.output_ids.shape) > 0 and output_pb.output_ids.shape[0] > 0)
        else None
    )
    all_hidden_states = (
        trans_tensor(output_pb.hidden_states)
        if output_pb.HasField("hidden_states")
        and len(output_pb.hidden_states.shape) > 0
        and output_pb.hidden_states.shape[0] > 0
        else None
    )
    all_all_hidden_states = (
        trans_tensor(output_pb.all_hidden_states)
        if output_pb.HasField("all_hidden_states")
        and len(output_pb.all_hidden_states.shape) > 0
        and output_pb.all_hidden_states.shape[0] > 0
        else None
    )
    all_loss = (
        trans_tensor(output_pb.loss)
        if output_pb.HasField("loss")
        and len(output_pb.loss.shape) > 0
        and output_pb.loss.shape[0] > 0
        else None
    )
    all_logits = (
        trans_tensor(output_pb.logits)
        if output_pb.HasField("logits")
        and len(output_pb.logits.shape) > 0
        and output_pb.logits.shape[0] > 0
        else None
    )
    all_all_probs = (
        trans_tensor(output_pb.all_probs)
        if output_pb.HasField("all_probs")
        and len(output_pb.all_probs.shape) > 0
        and output_pb.all_probs.shape[0] > 0
        else None
    )

    outputs_py = GenerateOutputs()
    input_token_ids = input_py.token_ids.reshape(1, -1)

    # 遍历每个 beam/output
    for i in range(num_outputs):
        output_py = GenerateOutput()
        output_py.finished = output_pb.finished[i]
        current_aux_info = None
        if aux_info_flag and len(output_pb.aux_info) > i:
            aux_info_pb = output_pb.aux_info[i]
            current_aux_info = AuxInfo(
                cost_time=aux_info_pb.cost_time_us / 1000.0,
                first_token_cost_time=aux_info_pb.first_token_cost_time_us / 1000.0,
                wait_time=aux_info_pb.wait_time_us / 1000.0,
                iter_count=aux_info_pb.iter_count,
                input_len=aux_info_pb.input_len,
                prefix_len=aux_info_pb.prefix_len,
                output_len=aux_info_pb.output_len,
                step_output_len=aux_info_pb.step_output_len,
                fallback_tokens=aux_info_pb.fallback_tokens,
                fallback_times=aux_info_pb.fallback_times,
                pd_sep=aux_info_pb.pd_sep,
                reuse_len=aux_info_pb.total_reuse_len,
                local_reuse_len=aux_info_pb.local_reuse_len,
                remote_reuse_len=aux_info_pb.remote_reuse_len,
                prefill_total_reuse_len=aux_info_pb.prefill_total_reuse_len,
                prefill_local_reuse_len=aux_info_pb.prefill_local_reuse_len,
                prefill_remote_reuse_len=aux_info_pb.prefill_remote_reuse_len,
                decode_total_reuse_len=aux_info_pb.decode_total_reuse_len,
                decode_local_reuse_len=aux_info_pb.decode_local_reuse_len,
                decode_remote_reuse_len=aux_info_pb.decode_remote_reuse_len,
                aux_string=aux_info_pb.aux_string,
                role_addrs=input_py.generate_config.role_addrs,
            )
            if aux_info_pb.HasField("cum_log_probs"):
                current_aux_info.cum_log_probs = trans_tensor(
                    aux_info_pb.cum_log_probs
                ).tolist()
            if aux_info_pb.HasField("softmax_probs"):
                current_aux_info.softmax_probs = trans_tensor(
                    aux_info_pb.softmax_probs
                ).tolist()

            output_py.aux_info = current_aux_info

        if all_output_ids is not None:
            output_py.output_ids = all_output_ids[i]
        output_py.input_ids = input_token_ids

        if all_hidden_states is not None:
            output_py.hidden_states = all_hidden_states[i]

        if all_all_hidden_states is not None:
            output_py.all_hidden_states = all_all_hidden_states[i]

        if all_loss is not None:
            loss_slice = all_loss[i]
            if input_py.generate_config.calculate_loss == 1:
                output_py.loss = (
                    loss_slice[0]
                    if hasattr(loss_slice, "__len__") and len(loss_slice) > 0
                    else loss_slice
                )
            else:
                output_py.loss = loss_slice

        if all_logits is not None:
            output_py.logits = all_logits[i]

        if all_all_probs is not None:
            output_py.all_probs = all_all_probs[i]

        if (
            logits_index is not None
            and all_logits is not None
            and current_aux_info
            and current_aux_info.output_len == logits_index
        ):
            stream_state.cached_logits_dict[i] = output_py.logits

        if output_py.finished and i in stream_state.cached_logits_dict:
            output_py.logits = stream_state.cached_logits_dict[i]

        outputs_py.generate_outputs.append(output_py)

    return outputs_py


def _handle_grpc_stream_in_thread(
    address: str,
    options: List[Tuple[str, Any]],
    input_pb: GenerateInputPB,
    timeout: float,
    response_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    error_info: Dict[str, Optional[Exception]],
    error_event: threading.Event
) -> None:
    """
    Handle gRPC streaming in a separate thread.
    
    This function runs in a thread pool executor and handles the synchronous
    gRPC streaming operations, putting responses into an async queue for
    consumption by the main event loop.
    
    Args:
        address: gRPC server address (host:port)
        options: gRPC channel options
        input_pb: Protocol buffer input message
        timeout: Request timeout in seconds
        response_queue: Async queue for thread-safe communication
        loop: Event loop for scheduling coroutines
        error_info: Shared dict for error information
        error_event: Event to signal error occurrence
        
    Raises:
        Nothing (all exceptions are captured and stored in error_info)
    """
    channel: Optional[grpc.Channel] = None
    
    try:
        # Create gRPC channel and stub
        channel = grpc.insecure_channel(address, options=options)
        stub = RpcServiceStub(channel)
        
        # Start streaming call
        response_iterator = stub.GenerateStreamCall(input_pb, timeout=timeout)
        
        # Stream responses to queue
        for response in response_iterator:
            asyncio.run_coroutine_threadsafe(
                response_queue.put(response), loop
            ).result()
            
    except Exception as exc:
        # Capture error information for main thread
        error_info["exception"] = exc
        error_event.set()
        logging.error(f"gRPC stream error in thread: {type(exc).__name__}: {exc}")
        
    finally:
        # Always signal end of stream
        try:
            asyncio.run_coroutine_threadsafe(
                response_queue.put(None), loop
            ).result()
        except Exception as exc:
            logging.error(f"Failed to send end signal: {type(exc).__name__}: {exc}")
            
        # Clean up channel
        if channel:
            try:
                channel.close()
            except Exception as exc:
                logging.error(f"Failed to close channel: {type(exc).__name__}: {exc}")


async def _handle_grpc_stream_in_executor(
    loop: asyncio.AbstractEventLoop,
    address: str,
    input_pb: GenerateInputPB,
    input_py: GenerateInput,
    options: List[Tuple[str, Any]],
    timeout: float,
    stream_state: StreamState
) -> AsyncGenerator[GenerateOutputs, None]:
    """
    Handle gRPC streaming responses using thread pool executor.
    
    This function bridges synchronous gRPC streaming with asynchronous
    Python by running the blocking operations in a thread pool.
    
    Args:
        loop: Event loop for scheduling async operations
        address: gRPC server address (host:port)
        input_pb: Protocol buffer input message
        input_py: Python input object for output transformation
        options: gRPC channel options
        timeout: Request timeout in seconds
        stream_state: State for caching logits across stream chunks
        
    Yields:
        GenerateOutputs: Transformed output objects from the stream
        
    Raises:
        FtRuntimeException: For gRPC errors with proper error codes
        Exception: For other unexpected errors
    """
    # Thread-safe communication primitives
    response_queue: asyncio.Queue = asyncio.Queue()
    error_event = threading.Event()
    error_info: Dict[str, Optional[Exception]] = {"exception": None}
    
    # Submit streaming task to thread pool
    stream_future = loop.run_in_executor(
        GRPC_STREAM_EXECUTOR,
        _handle_grpc_stream_in_thread,
        address,
        options,
        input_pb,
        timeout,
        response_queue,
        loop,
        error_info,
        error_event
    )
    
    try:
        # Process responses as they arrive
        while True:
            response = await response_queue.get()
            
            # Check for stream termination
            if response is None:
                break
                
            # Check for errors from the thread
            if error_event.is_set():
                raise error_info["exception"]
                
            # Transform and yield the response
            yield trans_output(input_py, response, stream_state)
            
    finally:
        # Ensure thread completion
        try:
            await stream_future
        except Exception as exc:
            logging.warning(f"Thread pool future error: {type(exc).__name__}: {exc}")
            # Re-raise if we don't have a more specific error
            if not error_event.is_set():
                raise


class ModelRpcClient(object):

    def __init__(self, config: GptInitModelParameters, address: Optional[str] = None):
        # 创建到服务器的连接
        if not address:
            address = f"localhost:{g_worker_info.rpc_server_port}"
        self._addresses = []
        # for test usage
        hack_ep_single_entry = config.py_env_configs.py_eplb_config.hack_ep_single_entry
        logging.info(f"hack ep single entry: {hack_ep_single_entry}")
        if (g_parallel_info.dp_size > 1) and (not hack_ep_single_entry):
            members_info_str = (
                f"[world_rank: {g_parallel_info.world_rank}]"
                + f"[tp_size: {g_parallel_info.tp_size}] all members: "
                + "{"
            )
            members = get_gang_info().members
            for member in members:
                members_info_str += f"{member}\n"
                if member.local_rank % g_parallel_info.tp_size == 0:
                    self._addresses.append(f"{member.ip}:{member.rpc_server_port}")
            members_info_str += "}"
            logging.info(f"{members_info_str}")
        else:
            self._addresses = [address]
        # last rank as ffn service, no be entry
        if config.gpt_init_params.ffn_disaggregate_config.enable_ffn_disaggregate:
            serving_ranks = (
                config.gpt_init_params.ffn_disaggregate_config.attention_tp_size
                * config.gpt_init_params.ffn_disaggregate_config.attention_dp_size
            )
            self._addresses = self._addresses[:serving_ranks]
        logging.info(f"client connect to rpc addresses: {self._addresses}")
        self.model_config = config
        self.options = []
        client_config = config.gpt_init_params.grpc_config.get_client_config()

        for key, value in client_config.items():
            self.options.append((key, value))
        logging.info(f"client options: {self.options}")

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        request_timeout_ms = input_py.generate_config.timeout_ms
        rpc_timeout_ms = (
            self.model_config.max_rpc_timeout_ms
            if self.model_config.max_rpc_timeout_ms > 0
            else MAX_GRPC_TIMEOUT_SECONDS * 1000
        )
        if request_timeout_ms == None or request_timeout_ms <= 0:
            grpc_timeout_seconds = rpc_timeout_ms / 1000
        else:
            grpc_timeout_seconds = request_timeout_ms / 1000
        input_py.generate_config.timeout_ms = (int)(grpc_timeout_seconds * 1000)
        input_pb = trans_input(input_py)
        response_iterator = None
        stream_state = StreamState()

        address_list = self._addresses

        for role_addr in input_py.generate_config.role_addrs:
            if (
                (
                    self.model_config.decode_entrance
                    and role_addr.role == RoleType.DECODE
                )
                or role_addr.role == RoleType.PDFUSION
                or (
                    not self.model_config.decode_entrance
                    and role_addr.role == RoleType.PREFILL
                )
            ):
                if role_addr.ip != "":
                    address_list = [role_addr.ip + ":" + str(role_addr.grpc_port)]
                    break

        try:
            loop = asyncio.get_event_loop()
            async for response in _handle_grpc_stream_in_executor(
                    loop,
                    address_list[input_py.request_id % len(address_list)],
                    input_py,
                    input_pb,
                    self.options,
                    grpc_timeout_seconds,
                    stream_state):
                yield response

        except grpc.RpcError as e:
            # TODO(xinfei.sxf) 非流式的请求无法取消了
            if response_iterator:
                response_iterator.cancel()
            error_details = ErrorDetailsPB()
            metadata = e.trailing_metadata()
            if "grpc-status-details-bin" in metadata and error_details.ParseFromString(
                metadata["grpc-status-details-bin"]
            ):
                logging.error(
                    f"request: [{input_pb.request_id}] RPC failed: "
                    f"{e.code()}, {e.details()}, detail error code is "
                    f"{ExceptionType.from_value(error_details.error_code)}"
                )
                raise FtRuntimeException(
                    ExceptionType(error_details.error_code), error_details.error_message
                )
            else:
                logging.error(
                    f"request: [{input_pb.request_id}] RPC failed: "
                    f"error code is {e.code()}, detail is {e.details()}"
                )
                if e.code() == StatusCode.DEADLINE_EXCEEDED:
                    raise FtRuntimeException(
                        ExceptionType.GENERATE_TIMEOUT, e.details()
                    )
                elif e.code() == StatusCode.CANCELLED:
                    raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
                else:
                    raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())
        except Exception as e:
            logging.error(f"rpc unknown error:{str(e)}")
            raise e
        finally:
            if response_iterator:
                response_iterator.cancel()

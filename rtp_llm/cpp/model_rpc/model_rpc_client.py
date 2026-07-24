import functools
import json
import logging
import os
import time
from typing import AsyncGenerator

import grpc
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleType
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    FetchRequestPB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
    RoleTypePB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.server.request_headers import (
    extract_correlation_request_id,
    extract_trace_id,
)
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateConfig,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool
from rtp_llm.utils.grpc_util import trans_option, trans_option_cast, trans_tensor


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}


def _is_finished_response(outputs_pb: GenerateOutputsPB) -> bool:
    finished = outputs_pb.flatten_output.finished
    return bool(finished) and all(finished)


def trans_role_type(role_type: RoleType) -> int:
    return role_type.value


def _trans_jsonable_option(config_pb, config, field_name):
    if not hasattr(config_pb, field_name):
        return
    value = getattr(config, field_name, None)
    if value is None:
        return
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    getattr(config_pb, field_name).value = value


def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.request_id = input_py.request_id
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())
    input_pb.start_time = int(time.time() * 1_000_000)
    input_pb.group_size = input_py.group_size
    if hasattr(input_py, "group_id") and input_py.group_id != -1:
        input_pb.group_id.value = input_py.group_id

    request_info = getattr(input_py, "request_info", None)
    if request_info is not None:
        input_pb.request_info.frontend_ip = (
            getattr(request_info, "frontend_ip", "") or ""
        )
        input_pb.request_info.dash_ip = getattr(request_info, "dash_ip", "") or ""
        input_pb.request_info.trace_id = getattr(request_info, "trace_id", "") or ""
        input_pb.request_info.request_id = getattr(request_info, "request_id", "") or ""
        input_pb.request_info.source_role = (
            getattr(request_info, "source_role", "") or ""
        )
    if not input_pb.request_info.trace_id:
        input_pb.request_info.trace_id = str(
            input_py.generate_config.trace_id
            or extract_trace_id(getattr(input_py, "headers", None))
            or ""
        )
    if not input_pb.request_info.request_id:
        input_pb.request_info.request_id = extract_correlation_request_id(
            getattr(input_py, "headers", None)
        ) or str(input_pb.request_info.trace_id or input_py.request_id)

    trans_multimodal_input(input_py, input_pb, input_py.generate_config)
    # check generate config is valid before enter into engine
    input_py.generate_config.validate()

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.max_thinking_tokens = (
        input_py.generate_config.max_thinking_tokens
    )
    if hasattr(generate_config_pb, "begin_think_token_ids"):
        generate_config_pb.begin_think_token_ids.extend(
            input_py.generate_config.begin_think_token_ids
        )
    if hasattr(generate_config_pb, "end_think_token_ids"):
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
    _trans_jsonable_option(generate_config_pb, input_py.generate_config, "json_schema")
    _trans_jsonable_option(generate_config_pb, input_py.generate_config, "regex")
    _trans_jsonable_option(generate_config_pb, input_py.generate_config, "ebnf")
    _trans_jsonable_option(
        generate_config_pb, input_py.generate_config, "structural_tag"
    )
    _trans_jsonable_option(
        generate_config_pb, input_py.generate_config, "response_format"
    )
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
    generate_config_pb.profile_trace_name = input_py.generate_config.profile_trace_name
    generate_config_pb.global_request_id = input_py.generate_config.global_request_id
    generate_config_pb.ignore_eos = input_py.generate_config.ignore_eos
    generate_config_pb.reuse_cache = input_py.generate_config.reuse_cache
    generate_config_pb.enable_memory_cache = (
        input_py.generate_config.enable_memory_cache
    )
    generate_config_pb.enable_device_cache = (
        input_py.generate_config.enable_device_cache
    )
    generate_config_pb.enable_remote_cache = (
        input_py.generate_config.enable_remote_cache
    )
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "trace_id", functools.partial(str)
    )
    trans_option(generate_config_pb, input_py.generate_config, "group_timeout")

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    for role_addr in input_py.generate_config.role_addrs:
        role_addr_pb = RoleAddrPB()
        role_addr_pb.role = role_addr.role.name
        role_addr_pb.role_type = trans_role_type(role_addr.role)
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
                pd_sep=aux_info_pb.pd_sep,
                reuse_len=aux_info_pb.total_reuse_len,
                local_reuse_len=aux_info_pb.local_reuse_len,
                remote_reuse_len=aux_info_pb.remote_reuse_len,
                memory_reuse_len=aux_info_pb.memory_reuse_len,
                prefill_total_reuse_len=aux_info_pb.prefill_total_reuse_len,
                prefill_local_reuse_len=aux_info_pb.prefill_local_reuse_len,
                prefill_remote_reuse_len=aux_info_pb.prefill_remote_reuse_len,
                prefill_memory_reuse_len=aux_info_pb.prefill_memory_reuse_len,
                decode_total_reuse_len=aux_info_pb.decode_total_reuse_len,
                decode_local_reuse_len=aux_info_pb.decode_local_reuse_len,
                decode_remote_reuse_len=aux_info_pb.decode_remote_reuse_len,
                decode_memory_reuse_len=aux_info_pb.decode_memory_reuse_len,
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


class ModelRpcClient(object):

    def __init__(
        self,
        addresses: list[str],
        client_config,
        max_rpc_timeout_ms: int = 0,
        decode_entrance: bool = False,
    ):
        """Initialize ModelRpcClient with addresses.

        Args:
            addresses: List of RPC addresses for data parallel communication
            max_rpc_timeout_ms: Maximum RPC timeout in milliseconds. <= 0 disables
                the gRPC deadline. Callers normally pass pd_sep_config.max_rpc_timeout_ms
                (args: --max_rpc_timeout_ms / env: MAX_RPC_TIMEOUT_MS).
            decode_entrance: Whether this is a decode entrance
        """
        self._addresses = addresses
        self._max_rpc_timeout_ms = max_rpc_timeout_ms
        self._decode_entrance = decode_entrance
        self._options = []
        for key, value in client_config.items():
            self._options.append((key, value))
        self._options.append(("grpc.max_send_message_length", 1024 * 1024 * 1024))
        self._options.append(("grpc.max_receive_message_length", 1024 * 1024 * 1024))
        logging.info(f"client options: {self._options}")

        # Initialize the channel pool
        self._channel_pool = GrpcHostChannelPool(
            options=self._options, cleanup_interval=60  # clean up every minute
        )
        logging.info(f"addresses: {self._addresses}")

    async def close(self):
        await self._channel_pool.close()

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        request_timeout_ms = input_py.generate_config.timeout_ms
        # Prefer per-request timeout; otherwise fall back to the server-side default
        # (pd_sep_config.max_rpc_timeout_ms). effective_ms <= 0 means no gRPC deadline.
        effective_ms = (
            request_timeout_ms
            if request_timeout_ms is not None and request_timeout_ms > 0
            else self._max_rpc_timeout_ms
        )
        input_pb = trans_input(input_py)
        response_iterator = None
        stream_state = StreamState()
        use_fetch_response = bool(getattr(input_py, "enqueued_by_master", False))

        if use_fetch_response:
            address_list = [
                role_addr.ip + ":" + str(role_addr.grpc_port)
                for role_addr in input_py.generate_config.role_addrs
                if role_addr.role == RoleType.PREFILL and role_addr.ip
            ]
            if os.environ.get("FLEXLB_EXPECT_FETCH_RESPONSE") == "1":
                logging.info(
                    "FLEXLB_EXPECT_FETCH_RESPONSE request_id=%s using FetchResponse",
                    input_pb.request_id,
                )
        else:
            address_list = self._addresses
            for role_addr in input_py.generate_config.role_addrs:
                if (
                    (self._decode_entrance and role_addr.role == RoleType.DECODE)
                    or role_addr.role == RoleType.PDFUSION
                    or (
                        not self._decode_entrance and role_addr.role == RoleType.PREFILL
                    )
                ):
                    if role_addr.ip != "":
                        address_list = [role_addr.ip + ":" + str(role_addr.grpc_port)]
                        break

        if not address_list:
            raise ValueError(f"No address found for request: {input_pb.request_id}")
        # Select target address before entering the try block so it is always
        # available to the error handlers below (surfaced in logs only)
        # details to identify which backend peer dropped the connection).
        target_address = address_list[input_py.request_id % len(address_list)]
        logging.debug(
            f"request: [{input_pb.request_id}] send to address: {target_address}"
        )
        stub = None
        stream_done = False
        terminal_seen = False
        try:
            # Get channel from pool
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            grpc_kwargs = {"timeout": effective_ms / 1000.0} if effective_ms > 0 else {}
            if use_fetch_response:
                response_iterator = stub.FetchResponse(
                    FetchRequestPB(request_id=input_pb.request_id), **grpc_kwargs
                )
            else:
                response_iterator = stub.GenerateStreamCall(input_pb, **grpc_kwargs)
            # 调用服务器方法并接收流式响应
            async for response in response_iterator.__aiter__():
                output = trans_output(input_py, response, stream_state)
                if use_fetch_response and _is_finished_response(response):
                    terminal_seen = True
                yield output
            stream_done = True
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
                    f"request: [{input_pb.request_id}] RPC to [{target_address}] failed: "
                    f"{e.code()}, {e.details()}, detail error code is "
                    f"{ExceptionType.from_value(error_details.error_code)}"
                )
                raise FtRuntimeException(
                    ExceptionType(error_details.error_code), error_details.error_message
                )
            else:
                logging.error(
                    f"request: [{input_pb.request_id}] RPC to [{target_address}] failed: "
                    f"error code is {e.code()}, detail is {e.details()}"
                )
                # NOTE: keep the backend peer (target_address) in the log line above
                # ONLY. Do NOT append it to the FtRuntimeException message, which is
                # serialized into the client-facing error response and would leak
                # internal cluster topology (worker ip:port) to callers.
                details = e.details() or ""
                if e.code() == StatusCode.DEADLINE_EXCEEDED:
                    raise FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, details)
                elif e.code() == StatusCode.CANCELLED:
                    raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, details)
                elif e.code() == StatusCode.UNAVAILABLE:
                    lower_details = details.lower()
                    if (
                        "socket closed" in lower_details
                        or "connection reset" in lower_details
                    ):
                        exception_type = ExceptionType.CONNECTION_RESET_BY_PEER
                    elif "timed out" in lower_details or "timeout" in lower_details:
                        exception_type = ExceptionType.CONNECT_TIMEOUT
                    else:
                        exception_type = ExceptionType.CONNECT_FAILED
                    raise FtRuntimeException(exception_type, details)
                else:
                    raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, details)
        except Exception as e:
            logging.error(
                f"request: [{input_pb.request_id}] rpc to [{target_address}] unknown error: {str(e)}"
            )
            raise e
        finally:
            should_cancel = not stream_done and not (
                use_fetch_response and terminal_seen
            )
            if response_iterator and should_cancel:
                response_iterator.cancel()

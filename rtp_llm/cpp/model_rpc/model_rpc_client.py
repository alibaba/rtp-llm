import functools
import logging
import math
from typing import AsyncGenerator, Optional

import grpc
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ReturnAllProbsMode, RoleType
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateInputPB,
    ErrorDetailsPB,
    ErrorCodePB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateConfig,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    RoleAddr,
    current_monotonic_time_s,
    current_unix_time_ms,
    initialize_request_deadlines,
)
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool
from rtp_llm.utils.grpc_util import (
    trans_from_tensor,
    trans_option,
    trans_option_cast,
    trans_tensor,
)

MAX_GRPC_TIMEOUT_SECONDS = 3600


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}


def batch_error_exception_type(error_code: int) -> ExceptionType:
    if error_code == ErrorCodePB.GENERATE_TIMEOUT:
        return ExceptionType.GENERATE_TIMEOUT
    if error_code == ErrorCodePB.CANCELLED:
        return ExceptionType.CANCELLED_ERROR
    return ExceptionType.UNKNOWN_ERROR


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


def trans_input(input_py: GenerateInput, *, timeout_ms: Optional[int] = None):
    input_pb = GenerateInputPB()
    input_pb.request_id = input_py.request_id
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())
    input_pb.batch_group_size = input_py.batch_group_size
    if hasattr(input_py, "batch_group_id") and input_py.batch_group_id != -1:
        input_pb.batch_group_id.value = input_py.batch_group_id
    request_deadline_unix_ms = getattr(
        input_py, "request_deadline_unix_ms", None
    )
    if request_deadline_unix_ms is not None and request_deadline_unix_ms > 0:
        input_pb.request_deadline_unix_ms = request_deadline_unix_ms

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
    random_seed = input_py.generate_config.random_seed
    if random_seed is not None and random_seed != []:
        generate_config_pb.random_seed.value = random_seed
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
    generate_config_pb.timeout_ms = (
        input_py.generate_config.timeout_ms if timeout_ms is None else timeout_ms
    )
    if input_py.generate_config.sp_advice_prompt_token_ids:
        generate_config_pb.sp_advice_prompt_token_ids.extend(
            input_py.generate_config.sp_advice_prompt_token_ids
        )
    generate_config_pb.return_cum_log_probs = (
        input_py.generate_config.return_cum_log_probs
    )
    # dual-write: legacy bool (true if any probs requested) + new int32 mode (offset 1)
    _rapm = input_py.generate_config.return_all_probs
    generate_config_pb.return_all_probs = _rapm != ReturnAllProbsMode.NONE
    generate_config_pb.return_all_probs_mode = _rapm + 1
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
    generate_config_pb.unique_key = input_py.generate_config.unique_key
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "trace_id", functools.partial(str)
    )
    trans_option(generate_config_pb, input_py.generate_config, "batch_group_timeout")
    trans_option(generate_config_pb, input_py.generate_config, "force_batch")

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    # 生成式推荐：组合 token 约束
    generate_config_pb.combo_token_size = input_py.generate_config.combo_token_size
    for i in range(len(input_py.generate_config.banned_combo_token_ids)):
        banned_combo = generate_config_pb.banned_combo_token_ids.rows.add()
        banned_combo.values.extend(
            input_py.generate_config.banned_combo_token_ids[i]
        )

    for role_addr in input_py.generate_config.role_addrs:
        role_addr_pb = RoleAddrPB()
        role_addr_pb.role = trans_role_type(role_addr.role)
        role_addr_pb.ip = role_addr.ip
        role_addr_pb.http_port = role_addr.http_port
        role_addr_pb.grpc_port = role_addr.grpc_port

        generate_config_pb.role_addrs.append(role_addr_pb)

    return input_pb


def get_multimodal_preprocess_value(value: Optional[int], default: int):
    if value is not None and value != -1:
        return value
    else:
        return default


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
        mm_preprocess_config_pb.width = get_multimodal_preprocess_value(
            mm_input.mm_preprocess_config.width, resized_shape[0]
        )
        mm_preprocess_config_pb.height = get_multimodal_preprocess_value(
            mm_input.mm_preprocess_config.height, resized_shape[1]
        )
        mm_preprocess_config_pb.min_pixels = get_multimodal_preprocess_value(
            generate_config.min_pixels, mm_input.mm_preprocess_config.min_pixels
        )
        mm_preprocess_config_pb.max_pixels = get_multimodal_preprocess_value(
            generate_config.max_pixels, mm_input.mm_preprocess_config.max_pixels
        )
        mm_preprocess_config_pb.fps = get_multimodal_preprocess_value(
            generate_config.fps, mm_input.mm_preprocess_config.fps
        )
        mm_preprocess_config_pb.min_frames = get_multimodal_preprocess_value(
            generate_config.min_frames, mm_input.mm_preprocess_config.min_frames
        )
        mm_preprocess_config_pb.max_frames = get_multimodal_preprocess_value(
            generate_config.max_frames, mm_input.mm_preprocess_config.max_frames
        )
        mm_preprocess_config_pb.crop_positions.extend(
            generate_config.crop_positions
            if generate_config.crop_positions is not None
            else mm_input.mm_preprocess_config.crop_positions
        )
        mm_preprocess_config_pb.mm_timeout_ms = get_multimodal_preprocess_value(
            generate_config.mm_timeout_ms, mm_input.mm_preprocess_config.mm_timeout_ms
        )
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
            if len(aux_info_pb.multimodal_lengths) > 0:
                current_aux_info.multimodal_lengths = dict(
                    aux_info_pb.multimodal_lengths
                )

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
            max_rpc_timeout_ms: Maximum RPC timeout in milliseconds
            decode_entrance: Whether this is a decode entrance
        """
        self._addresses = addresses
        self._max_rpc_timeout_ms = max_rpc_timeout_ms
        self._decode_entrance = decode_entrance
        self._options = []
        for key, value in client_config.items():
            self._options.append((key, value))
        logging.info(f"client options: {self._options}")

        # Initialize the channel pool
        self._channel_pool = GrpcHostChannelPool(
            options=self._options, cleanup_interval=60  # clean up every minute
        )
        logging.info(f"addresses: {self._addresses}")

    def _compute_grpc_timeout(
        self,
        timeout_ms,
        request_deadline_monotonic_s: Optional[float] = None,
        monotonic_now_s: Optional[float] = None,
    ) -> float:
        rpc_timeout_ms = (
            self._max_rpc_timeout_ms
            if self._max_rpc_timeout_ms > 0
            else MAX_GRPC_TIMEOUT_SECONDS * 1000
        )
        timeout_limits_ms = [float(rpc_timeout_ms)]
        if timeout_ms is not None and timeout_ms > 0:
            timeout_limits_ms.append(float(timeout_ms))
        if request_deadline_monotonic_s is not None:
            if monotonic_now_s is None:
                monotonic_now_s = current_monotonic_time_s()
            remaining_ms = (
                request_deadline_monotonic_s - monotonic_now_s
            ) * 1000.0
            if remaining_ms <= 0:
                raise FtRuntimeException(
                    ExceptionType.GENERATE_TIMEOUT,
                    "request deadline exhausted before backend RPC",
                )
            timeout_limits_ms.append(remaining_ms)
        return min(timeout_limits_ms) / 1000.0

    @staticmethod
    def _wire_timeout_ms(grpc_timeout_seconds: float) -> int:
        return max(1, math.ceil(grpc_timeout_seconds * 1000.0 - 1e-9))

    def _handle_grpc_error(self, e: grpc.RpcError, request_desc: str) -> None:
        error_details = ErrorDetailsPB()
        metadata = e.trailing_metadata() or {}
        has_error_details = "grpc-status-details-bin" in metadata and error_details.ParseFromString(
            metadata["grpc-status-details-bin"]
        )
        if e.code() == StatusCode.DEADLINE_EXCEEDED:
            message = (
                error_details.error_message
                if has_error_details and error_details.error_message
                else e.details()
            )
            logging.error(f"{request_desc} RPC deadline exceeded: {message}")
            raise FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, message)
        if has_error_details:
            logging.error(
                f"{request_desc} RPC failed: "
                f"{e.code()}, {e.details()}, detail error code is "
                f"{ExceptionType.from_value(error_details.error_code)}"
            )
            raise FtRuntimeException(
                ExceptionType(error_details.error_code), error_details.error_message
            )
        else:
            logging.error(
                f"{request_desc} RPC failed: "
                f"error code is {e.code()}, detail is {e.details()}"
            )
            if e.code() == StatusCode.CANCELLED:
                raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
            else:
                raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        initialize_request_deadlines(
            input_py, current_monotonic_time_s(), current_unix_time_ms()
        )
        request_deadline_s = getattr(
            input_py, "request_deadline_monotonic_s", None
        )
        self._compute_grpc_timeout(
            input_py.generate_config.timeout_ms, request_deadline_s
        )
        response_iterator = None
        stream_state = StreamState()

        address_list = self._addresses

        for role_addr in input_py.generate_config.role_addrs:
            if (
                (self._decode_entrance and role_addr.role == RoleType.DECODE)
                or role_addr.role == RoleType.PDFUSION
                or (not self._decode_entrance and role_addr.role == RoleType.PREFILL)
            ):
                if role_addr.ip != "":
                    address_list = [role_addr.ip + ":" + str(role_addr.grpc_port)]
                    break

        if not address_list:
            raise ValueError(f"No address found for request: {input_py.request_id}")
        logging.debug(
            f"request: [{input_py.request_id}] send to address: {address_list[input_py.request_id % len(address_list)]}"
        )

        try:
            # Select target address
            target_address = address_list[input_py.request_id % len(address_list)]
            logging.debug(f"target_address: {target_address}")
            # Get channel from pool
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            grpc_timeout_seconds = self._compute_grpc_timeout(
                input_py.generate_config.timeout_ms, request_deadline_s
            )
            input_pb = trans_input(
                input_py,
                timeout_ms=self._wire_timeout_ms(grpc_timeout_seconds),
            )
            grpc_timeout_seconds = self._compute_grpc_timeout(
                input_py.generate_config.timeout_ms, request_deadline_s
            )
            input_pb.generate_config.timeout_ms = self._wire_timeout_ms(
                grpc_timeout_seconds
            )

            response_iterator = stub.GenerateStreamCall(
                input_pb, timeout=grpc_timeout_seconds
            )
            # 调用服务器方法并接收流式响应
            async for response in response_iterator.__aiter__():
                yield trans_output(input_py, response, stream_state)
        except grpc.RpcError as e:
            if response_iterator:
                response_iterator.cancel()
            self._handle_grpc_error(e, f"request: [{input_py.request_id}]")
        except Exception as e:
            logging.error(f"rpc unknown error:{str(e)}")
            raise e
        finally:
            if response_iterator:
                response_iterator.cancel()

    async def batch_enqueue(self, inputs: list[GenerateInput]) -> list[GenerateOutputs]:
        if not inputs:
            return []

        monotonic_now_s = current_monotonic_time_s()
        unix_now_ms = current_unix_time_ms()
        for inp in inputs:
            initialize_request_deadlines(inp, monotonic_now_s, unix_now_ms)
            self._compute_grpc_timeout(
                inp.generate_config.timeout_ms,
                getattr(inp, "request_deadline_monotonic_s", None),
                monotonic_now_s,
            )

        if not self._addresses:
            raise ValueError("No address found for batch request")
        target_address = self._addresses[inputs[0].request_id % len(self._addresses)]
        logging.debug(
            f"batch request: [{len(inputs)} items] send to address: {target_address}"
        )

        try:
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            batch_input_pb = BatchGenerateInputPB()
            for inp in inputs:
                batch_input_pb.inputs.append(trans_input(inp))
            final_now_s = current_monotonic_time_s()
            final_item_timeouts = []
            for inp, input_pb in zip(inputs, batch_input_pb.inputs):
                item_timeout = self._compute_grpc_timeout(
                    inp.generate_config.timeout_ms,
                    getattr(inp, "request_deadline_monotonic_s", None),
                    final_now_s,
                )
                input_pb.generate_config.timeout_ms = self._wire_timeout_ms(
                    item_timeout
                )
                final_item_timeouts.append(item_timeout)
            grpc_timeout_seconds = max(final_item_timeouts)
            response = await stub.BatchGenerateCall(
                batch_input_pb, timeout=grpc_timeout_seconds
            )

            results = []
            for i, result_pb in enumerate(response.results):
                if (
                    result_pb.HasField("error_info")
                    and result_pb.error_info.error_message
                ):
                    raise FtRuntimeException(
                        batch_error_exception_type(result_pb.error_info.error_code),
                        f"batch item {i} failed: {result_pb.error_info.error_message}",
                    )
                stream_state = StreamState()
                output = trans_output(inputs[i], result_pb.final_output, stream_state)
                results.append(output)
            return results

        except grpc.RpcError as e:
            self._handle_grpc_error(e, f"batch request: [{len(inputs)} items]")
        except FtRuntimeException:
            raise
        except Exception as e:
            logging.error(f"batch rpc unknown error: {str(e)}")
            raise e

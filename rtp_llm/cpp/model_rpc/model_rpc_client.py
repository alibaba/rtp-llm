import functools
import logging
from collections.abc import Sequence
from typing import AsyncGenerator, Optional

import grpc
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ReturnAllProbsMode, RoleType
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateInputPB,
    ErrorDetailsPB,
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
)
from rtp_llm.utils.grpc_host_channel_pool import GrpcHostChannelPool
from rtp_llm.utils.grpc_util import (
    trans_option,
    trans_option_cast,
    trans_tensor,
)

MAX_GRPC_TIMEOUT_SECONDS = 3600


def _format_host_port(host: str, port: int) -> str:
    port = int(port)
    if not host or port < 1 or port > 65535:
        raise ValueError(f"invalid grpc address host={host}, port={port}")
    if host[0] == "[" or host[-1] == "]":
        if host[0] == "[" and host[-1] == "]":
            return f"{host}:{port}"
        raise ValueError(f"invalid bracketed grpc host: {host}")
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def _split_host_port(address: str) -> tuple[str, int]:
    if address.startswith("["):
        close_pos = address.find("]")
        if close_pos == -1 or close_pos + 1 >= len(address) or address[close_pos + 1] != ":":
            raise ValueError(f"invalid grpc address: {address}")
        host = address[1:close_pos]
        port_text = address[close_pos + 2 :]
    else:
        host, port_text = address.rsplit(":", 1)
    port = int(port_text)
    if not host or port < 1 or port > 65535:
        raise ValueError(f"invalid grpc address: {address}")
    return host, port


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
    input_pb.batch_group_size = input_py.batch_group_size
    if hasattr(input_py, "batch_group_id") and input_py.batch_group_id != -1:
        input_pb.batch_group_id.value = input_py.batch_group_id

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
    generate_config_pb.unique_key = input_py.generate_config.unique_key
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
    trans_option(generate_config_pb, input_py.generate_config, "batch_group_timeout")
    trans_option(generate_config_pb, input_py.generate_config, "force_batch")

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    # 生成式推荐：组合 token 约束
    generate_config_pb.combo_token_size = input_py.generate_config.combo_token_size
    for i in range(len(input_py.generate_config.banned_combo_token_ids)):
        banned_combo = generate_config_pb.banned_combo_token_ids.rows.add()
        banned_combo.values.extend(input_py.generate_config.banned_combo_token_ids[i])

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
    input_py: GenerateInput,
    outputs_pb: GenerateOutputsPB,
    stream_state: StreamState,
    response_role_addrs=None,
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
                role_addrs=response_role_addrs or input_py.generate_config.role_addrs,
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

    def _get_explicit_target_address(self, input_py: GenerateInput) -> str | None:
        for role_addr in input_py.generate_config.role_addrs:
            if (
                (self._decode_entrance and role_addr.role == RoleType.DECODE)
                or role_addr.role == RoleType.PDFUSION
                or (not self._decode_entrance and role_addr.role == RoleType.PREFILL)
            ) and role_addr.ip != "":
                return _format_host_port(role_addr.ip, role_addr.grpc_port)
        return None

    def _build_response_role_addrs(
        self, input_py: GenerateInput, target_address: str
    ) -> list[RoleAddr] | None:
        response_role_addrs = (
            [
                role_addr
                for role_addr in input_py.generate_config.role_addrs
                if role_addr.role == RoleType.DECODE
            ]
            if self._decode_entrance
            else None
        )
        if self._decode_entrance and not response_role_addrs:
            target_ip, target_port = _split_host_port(target_address)
            response_role_addrs = list(response_role_addrs or [])
            response_role_addrs.append(
                RoleAddr(
                    role=RoleType.DECODE,
                    ip=target_ip,
                    http_port=target_port - 1,
                    grpc_port=target_port,
                )
            )
        return response_role_addrs

    def _select_address_and_response_role_addrs(
        self, input_py: GenerateInput
    ) -> tuple[list[str], str, list[RoleAddr] | None]:
        address_list = self._addresses
        explicit_target_address = self._get_explicit_target_address(input_py)
        if explicit_target_address is not None:
            address_list = [explicit_target_address]

        if not address_list:
            raise ValueError(f"No address found for request: {input_py.request_id}")

        target_address = self._select_target_address(address_list, input_py.request_id)
        response_role_addrs = self._build_response_role_addrs(input_py, target_address)

        return address_list, target_address, response_role_addrs

    @staticmethod
    def _select_target_address(address_list: Sequence[str], request_id: int) -> str:
        return address_list[request_id % len(address_list)]

    def _compute_grpc_timeout(self, timeout_ms) -> float:
        rpc_timeout_ms = (
            self._max_rpc_timeout_ms
            if self._max_rpc_timeout_ms > 0
            else MAX_GRPC_TIMEOUT_SECONDS * 1000
        )
        if timeout_ms is None or timeout_ms <= 0:
            return rpc_timeout_ms / 1000
        return timeout_ms / 1000

    def _handle_grpc_error(self, e: grpc.RpcError, request_desc: str) -> None:
        error_details = ErrorDetailsPB()
        metadata = e.trailing_metadata()
        if "grpc-status-details-bin" in metadata and error_details.ParseFromString(
            metadata["grpc-status-details-bin"]
        ):
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
            if e.code() == StatusCode.DEADLINE_EXCEEDED:
                raise FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, e.details())
            elif e.code() == StatusCode.CANCELLED:
                raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
            elif e.code() == StatusCode.RESOURCE_EXHAUSTED:
                # transErrorCodeToGrpc maps only MALLOC_FAILED / DECODE_MALLOC_FAILED
                # to RESOURCE_EXHAUSTED, so reversing it back to MALLOC_ERROR is
                # unambiguous. Needed so prefill's LACK MEM surfaces as
                # 602_MALLOC_ERROR in the frontend error_qps metric rather than
                # collapsing to UNKNOWN_ERROR when grpc-status-details-bin is missing.
                raise FtRuntimeException(ExceptionType.MALLOC_ERROR, e.details())
            else:
                raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        grpc_timeout_seconds = self._compute_grpc_timeout(
            input_py.generate_config.timeout_ms
        )
        input_py.generate_config.timeout_ms = int(grpc_timeout_seconds * 1000)
        input_pb = trans_input(input_py)
        response_iterator = None
        stream_state = StreamState()

        _, target_address, response_role_addrs = (
            self._select_address_and_response_role_addrs(input_py)
        )
        logging.debug(
            f"request: [{input_pb.request_id}] send to address: {target_address}"
        )

        try:
            # Select target address
            logging.debug(f"target_address: {target_address}")
            # Get channel from pool
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            response_iterator = stub.GenerateStreamCall(
                input_pb, timeout=grpc_timeout_seconds
            )
            # 调用服务器方法并接收流式响应
            async for response in response_iterator.__aiter__():
                yield trans_output(
                    input_py,
                    response,
                    stream_state,
                    response_role_addrs=response_role_addrs,
                )
        except grpc.RpcError as e:
            if response_iterator:
                response_iterator.cancel()
            self._handle_grpc_error(e, f"request: [{input_pb.request_id}]")
        except Exception as e:
            logging.error(f"rpc unknown error:{str(e)}")
            raise e
        finally:
            if response_iterator:
                response_iterator.cancel()

    async def batch_enqueue(self, inputs: list[GenerateInput]) -> list[GenerateOutputs]:
        if not inputs:
            return []

        max_timeout_ms = max((inp.generate_config.timeout_ms or 0) for inp in inputs)
        grpc_timeout_seconds = self._compute_grpc_timeout(max_timeout_ms)

        batch_input_pb = BatchGenerateInputPB()
        for inp in inputs:
            inp.generate_config.timeout_ms = int(grpc_timeout_seconds * 1000)
            input_pb = trans_input(inp)
            batch_input_pb.inputs.append(input_pb)

        _, target_address, first_response_role_addrs = (
            self._select_address_and_response_role_addrs(inputs[0])
        )
        response_role_addrs_list = [first_response_role_addrs]
        for inp in inputs[1:]:
            explicit_target_address = self._get_explicit_target_address(inp)
            if (
                explicit_target_address is not None
                and explicit_target_address != target_address
            ):
                raise FtRuntimeException(
                    ExceptionType.UNSUPPORTED_OPERATION,
                    "batch enqueue routed to conflicting explicit backends: %s vs %s"
                    % (target_address, explicit_target_address),
                )
            response_role_addrs = self._build_response_role_addrs(inp, target_address)
            response_role_addrs_list.append(response_role_addrs)
        logging.debug(
            f"batch request: [{len(inputs)} items] send to address: {target_address}"
        )

        try:
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)
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
                        ExceptionType.UNKNOWN_ERROR,
                        f"batch item {i} failed: {result_pb.error_info.error_message}",
                    )
                stream_state = StreamState()
                output = trans_output(
                    inputs[i],
                    result_pb.final_output,
                    stream_state,
                    response_role_addrs=response_role_addrs_list[i],
                )
                results.append(output)
            return results

        except grpc.RpcError as e:
            self._handle_grpc_error(e, f"batch request: [{len(inputs)} items]")
        except FtRuntimeException:
            raise
        except Exception as e:
            logging.error(f"batch rpc unknown error: {str(e)}")
            raise e

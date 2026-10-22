import asyncio
import functools
import json
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import AsyncGenerator

import grpc
import torch
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleType
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateInputPB,
    ErrorCodePB,
    ErrorDetailsPB,
    FetchRequestPB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.server.request_headers import (
    extract_correlation_request_id,
    extract_trace_id,
    resolve_qos_priority,
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

logger = logging.getLogger(__name__)


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}
        self.prefill_aux_info_observed = False


def _is_finished_response(outputs_pb: GenerateOutputsPB) -> bool:
    finished = outputs_pb.flatten_output.finished
    return bool(finished) and all(finished)


def _validate_prefill_only_response(
    input_py: GenerateInput, outputs_pb: GenerateOutputsPB, stream_state: StreamState
) -> None:
    if not input_py.generate_config.is_prefill_only():
        return

    output_pb = outputs_pb.flatten_output
    if output_pb.HasField("output_ids"):
        # Validate even frames without finished/aux_info: neither can hide tokens.
        # Use the existing decoder so scalar, empty and malformed tensors retain
        # the same shape/payload semantics as normal output conversion.
        violation = None
        try:
            token_count = trans_tensor(output_pb.output_ids).numel()
        except ValueError as error:
            violation = f"invalid output_ids: {error}"
        else:
            if token_count != 0:
                violation = f"output_ids contains {token_count} tokens"
        if violation is not None:
            logger.error(
                "prefill-only response contract violation: request_id=%s, %s",
                input_py.request_id,
                violation,
            )
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION,
                f"prefill-only request [{input_py.request_id}] received {violation}",
            )

    aux_infos = output_pb.aux_info
    if not aux_infos:
        return

    stream_state.prefill_aux_info_observed = True
    for index, aux_info in enumerate(aux_infos):
        if aux_info.output_len != 0:
            logger.error(
                "prefill-only response contract violation: request_id=%s, "
                "aux_info[%s].output_len=%s; backend generated tokens",
                input_py.request_id,
                index,
                aux_info.output_len,
            )
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION,
                f"prefill-only request [{input_py.request_id}] received "
                f"aux_info[{index}].output_len={aux_info.output_len}; the backend "
                "generated tokens and may not support GenerateConfigPB.prefill_only",
            )


def _validate_prefill_only_completed(
    input_py: GenerateInput, stream_state: StreamState
) -> None:
    if (
        input_py.generate_config.is_prefill_only()
        and input_py.generate_config.aux_info
        and not stream_state.prefill_aux_info_observed
    ):
        logger.error(
            "prefill-only capability verification failed: request_id=%s "
            "completed without aux_info",
            input_py.request_id,
        )
        raise FtRuntimeException(
            ExceptionType.EXECUTION_EXCEPTION,
            f"prefill-only request [{input_py.request_id}] completed without "
            "aux_info; the backend capability cannot be verified and it may not "
            "support GenerateConfigPB.prefill_only",
        )


_RPC_ERROR_CODE_OVERRIDES = {
    ErrorCodePB.CANCELLED: ExceptionType.CANCELLED_ERROR,
    # The wire enum uses single-L CANCELED, unlike the native/Python code.
    ErrorCodePB.P2P_CONNECTOR_WORKER_READ_CANCELED: (
        ExceptionType.P2P_CONNECTOR_WORKER_READ_CANCELLED
    ),
}


def _rpc_error_code_name(error_code: int) -> str:
    try:
        return ErrorCodePB.Name(error_code)
    except ValueError:
        return f"UNRECOGNIZED_ERROR_CODE_{error_code}"


def _trans_rpc_error_code(error_code: int) -> ExceptionType:
    if error_code in _RPC_ERROR_CODE_OVERRIDES:
        return _RPC_ERROR_CODE_OVERRIDES[error_code]
    try:
        return ExceptionType[_rpc_error_code_name(error_code)]
    except KeyError:
        return ExceptionType.UNKNOWN_ERROR


def trans_role_type(role_type: RoleType) -> RoleAddrPB.RoleType:
    """Map the frontend role to the original RoleAddrPB field-1 enum.

    Keep this explicit instead of depending on the numeric values of two
    independently generated enums remaining aligned.
    """
    if role_type == RoleType.PDFUSION:
        return RoleAddrPB.RoleType.PDFUSION
    if role_type == RoleType.PREFILL:
        return RoleAddrPB.RoleType.PREFILL
    if role_type == RoleType.DECODE:
        return RoleAddrPB.RoleType.DECODE
    if role_type == RoleType.VIT:
        return RoleAddrPB.RoleType.VIT
    if role_type == RoleType.FRONTEND:
        return RoleAddrPB.RoleType.FRONTEND
    raise ValueError(f"unsupported role type: {role_type!r}")


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
    input_pb.priority = resolve_qos_priority(
        getattr(input_py, "headers", None), input_py.generate_config
    )
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
    generate_config_pb.prefill_only = input_py.generate_config.is_prefill_only()
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
    generate_config_pb.frontend_metric_streaming = (
        input_py.generate_config.frontend_metric_streaming
    )
    generate_config_pb.timeout_ms = input_py.generate_config.timeout_ms
    if input_py.generate_config.sp_advice_prompt_token_ids:
        generate_config_pb.sp_advice_prompt_token_ids.extend(
            input_py.generate_config.sp_advice_prompt_token_ids
        )
    generate_config_pb.return_cum_log_probs = (
        input_py.generate_config.return_cum_log_probs
    )
    generate_config_pb.return_all_probs = input_py.generate_config.return_all_probs
    generate_config_pb.return_logprobs = input_py.generate_config.return_logprobs
    generate_config_pb.top_logprobs = input_py.generate_config.top_logprobs
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
    generate_config_pb.capture_hidden_states = (
        input_py.generate_config.capture_hidden_states
    )
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
        proto_role = trans_role_type(role_addr.role)
        # Dual-write the original enum at field 1 and the string extension so
        # both old and new engines decode role addresses during rolling upgrade.
        role_addr_pb.role = proto_role
        role_addr_pb.role_str = role_addr.role.name
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
    _validate_prefill_only_response(input_py, outputs_pb, stream_state)
    logging.debug("outputs_pb = %s", outputs_pb)
    output_pb = outputs_pb.flatten_output
    num_outputs = len(output_pb.finished)

    if num_outputs == 0:
        return GenerateOutputs()

    logits_index = input_py.generate_config.logits_index
    aux_info_flag = input_py.generate_config.aux_info

    def trans_optional_compact_tensor(tensor_pb, dtype):
        shape = list(tensor_pb.shape)
        if not shape:
            return None
        if any(dim == 0 for dim in shape):
            return torch.empty(shape, dtype=dtype)
        return trans_tensor(tensor_pb)

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
    all_token_logprobs = (
        trans_optional_compact_tensor(output_pb.token_logprobs, torch.float32)
        if output_pb.HasField("token_logprobs")
        else None
    )
    all_top_logprob_token_ids = (
        trans_optional_compact_tensor(output_pb.top_logprob_token_ids, torch.int32)
        if output_pb.HasField("top_logprob_token_ids")
        else None
    )
    all_top_logprobs = (
        trans_optional_compact_tensor(output_pb.top_logprobs, torch.float32)
        if output_pb.HasField("top_logprobs")
        else None
    )

    logprobs_offsets = list(output_pb.logprobs_offsets)
    logprobs_counts = list(output_pb.logprobs_counts)
    has_logprobs_placement = bool(logprobs_offsets or logprobs_counts)
    if has_logprobs_placement:
        if len(logprobs_offsets) != num_outputs or len(logprobs_counts) != num_outputs:
            raise ValueError(
                "logprobs_offsets/logprobs_counts must contain one value per output"
            )

    def compact_logprob_slice(tensor, output_index: int, count: int, name: str):
        if tensor is None:
            if count:
                raise ValueError(f"{name} is missing for logprobs_count={count}")
            return None
        if tensor.dim() < 2 or tensor.shape[0] != num_outputs:
            raise ValueError(
                f"{name} must have a leading output dimension of {num_outputs}: "
                f"got {tuple(tensor.shape)}"
            )
        output_tensor = tensor[output_index]
        if output_tensor.shape[0] < count:
            raise ValueError(
                f"{name} has only {output_tensor.shape[0]} padded rows for "
                f"logprobs_count={count}"
            )
        return output_tensor.narrow(0, 0, count)

    outputs_py = GenerateOutputs(
        frontend_metric_only=bool(outputs_pb.frontend_metric_only),
        frontend_context_token_num=(
            int(outputs_pb.frontend_context_token_num.value)
            if outputs_pb.HasField("frontend_context_token_num")
            else None
        ),
        frontend_context_token_num_with_cache=(
            int(outputs_pb.frontend_context_token_num_with_cache.value)
            if outputs_pb.HasField("frontend_context_token_num_with_cache")
            else None
        ),
        frontend_context_execute_time_us=(
            int(outputs_pb.frontend_context_execute_time_us.value)
            if outputs_pb.HasField("frontend_context_execute_time_us")
            else None
        ),
        frontend_context_execute_time_with_cache_us=(
            int(outputs_pb.frontend_context_execute_time_with_cache_us.value)
            if outputs_pb.HasField("frontend_context_execute_time_with_cache_us")
            else None
        ),
        frontend_generate_token_num=(
            int(outputs_pb.frontend_generate_token_num.value)
            if outputs_pb.HasField("frontend_generate_token_num")
            else None
        ),
        frontend_sp_tpot_samples=[
            (int(sample.sequence_id), float(sample.tpot_us))
            for sample in outputs_pb.frontend_sp_tpot_samples
        ],
        frontend_generate_execute_time_us=(
            int(outputs_pb.frontend_generate_execute_time_us.value)
            if outputs_pb.HasField("frontend_generate_execute_time_us")
            else None
        ),
    )
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
                speculative_verify_rounds=aux_info_pb.speculative_verify_rounds,
                speculative_accepted_token_num=aux_info_pb.speculative_accepted_token_num,
                speculative_proposed_draft_tokens=(
                    aux_info_pb.speculative_proposed_draft_tokens
                ),
                context_execute_time_us=aux_info_pb.context_execute_time_us,
                generate_execute_time_us=aux_info_pb.generate_execute_time_us,
                context_execute_time_with_cache_us=(
                    aux_info_pb.context_execute_time_with_cache_us
                ),
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

        if has_logprobs_placement:
            logprobs_offset = int(logprobs_offsets[i])
            logprobs_count = int(logprobs_counts[i])
            if logprobs_offset < 0 or logprobs_count < 0:
                raise ValueError("logprobs offset/count must be non-negative")
            if all_output_ids is not None:
                padded_output_token_count = int(all_output_ids[i].numel())
                if logprobs_offset + logprobs_count > padded_output_token_count:
                    raise ValueError(
                        "compact logprobs placement exceeds the output_ids token span"
                    )
            output_py.logprobs_offset = logprobs_offset
            output_py.logprobs_count = logprobs_count
            output_py.token_logprobs = compact_logprob_slice(
                all_token_logprobs, i, logprobs_count, "token_logprobs"
            )
            output_py.top_logprob_token_ids = compact_logprob_slice(
                all_top_logprob_token_ids,
                i,
                logprobs_count,
                "top_logprob_token_ids",
            )
            output_py.top_logprobs = compact_logprob_slice(
                all_top_logprobs, i, logprobs_count, "top_logprobs"
            )
        else:
            # Legacy RPC responses carried one probability row per output id.
            if all_token_logprobs is not None:
                output_py.token_logprobs = all_token_logprobs[i]
            if all_top_logprob_token_ids is not None:
                output_py.top_logprob_token_ids = all_top_logprob_token_ids[i]
            if all_top_logprobs is not None:
                output_py.top_logprobs = all_top_logprobs[i]

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
        logging.info(f"client options: {self._options}")

        # Initialize the channel pool
        self._channel_pool = GrpcHostChannelPool(
            options=self._options, cleanup_interval=60  # clean up every minute
        )
        logging.info(f"addresses: {self._addresses}")

    def update_addresses(self, addresses: list[str]) -> None:
        self._addresses = list(addresses)
        logging.info("model RPC addresses refreshed: %s", self._addresses)

    async def close(self):
        await self._channel_pool.close()

    async def batch_generate(
        self,
        inputs: list[GenerateInput],
        *,
        preserve_capture_backpressure: bool = False,
    ) -> list[GenerateOutputs | FtRuntimeException]:
        """Run one native BatchGenerateCall, with atomic failure by default.

        Complete non-streaming outputs are requested on copied configs. The
        legacy per-enqueue max_concurrency option is not supported by this API.

        Opt-in requires every input to be strict zero-token capture without
        cache reuse. Only exact capture backpressure is retained at its input
        position, alongside later successes. It certifies non-admission before
        reservation/publication; other item, transport, cardinality and
        conversion failures raise for the whole call, with no partial results.
        """
        if not inputs:
            return []

        batch_inputs = []
        effective_timeouts_ms = []
        for inp in inputs:
            config = inp.generate_config.model_copy(deep=True)
            # Unary batch returns only the terminal output, not token deltas.
            config.is_streaming = False
            request_timeout_ms = config.timeout_ms
            effective_ms = (
                request_timeout_ms
                if request_timeout_ms is not None and request_timeout_ms > 0
                else self._max_rpc_timeout_ms
            )
            if effective_ms > 0:
                config.timeout_ms = int(effective_ms)
                effective_timeouts_ms.append(int(effective_ms))
            elif config.timeout_ms is None:
                config.timeout_ms = 0
            batch_inputs.append(replace(inp, generate_config=config))

        # A finite sibling must not impose its deadline on an unlimited item.
        has_unlimited_request = self._max_rpc_timeout_ms <= 0 and any(
            timeout_ms is None or timeout_ms <= 0
            for timeout_ms in (inp.generate_config.timeout_ms for inp in inputs)
        )
        grpc_timeout_seconds = (
            max(effective_timeouts_ms) / 1000.0
            if effective_timeouts_ms and not has_unlimited_request
            else None
        )
        batch_input_pb = BatchGenerateInputPB()
        for i, inp in enumerate(batch_inputs):
            input_pb = trans_input(inp)
            if preserve_capture_backpressure:
                config_pb = input_pb.generate_config
                if not (
                    config_pb.max_new_tokens == 0
                    and config_pb.prefill_only
                    and config_pb.capture_hidden_states
                    and not config_pb.reuse_cache
                ):
                    raise FtRuntimeException(
                        ExceptionType.INVALID_PARAMS,
                        "preserve_capture_backpressure requires every input to "
                        "use effective max_new_tokens == 0, "
                        "capture_hidden_states=True, and reuse_cache=False "
                        f"(invalid batch item {i})",
                    )
            batch_input_pb.inputs.append(input_pb)

        target_address = self._addresses[inputs[0].request_id % len(self._addresses)]
        response_call = None
        completed = False
        acquire_start = time.monotonic()
        try:
            # Keep GLM's leased pool lifecycle rather than borrowing the main
            # tree's unleased pool API. Acquisition consumes the same deadline.
            async with self._channel_pool.acquire(
                target_address, timeout=grpc_timeout_seconds
            ) as channel:
                stub = RpcServiceStub(channel)
                grpc_kwargs = (
                    {
                        "timeout": max(
                            0.0,
                            grpc_timeout_seconds - (time.monotonic() - acquire_start),
                        )
                    }
                    if grpc_timeout_seconds is not None
                    else {}
                )
                try:
                    response_call = stub.BatchGenerateCall(
                        batch_input_pb, **grpc_kwargs
                    )
                    response = await response_call
                    completed = True
                    if len(response.results) != len(inputs):
                        raise FtRuntimeException(
                            ExceptionType.EXECUTION_EXCEPTION,
                            "batch RPC returned an invalid result cardinality: "
                            f"received {len(response.results)} results for {len(inputs)} requests",
                        )
                    results = []
                    for i, result_pb in enumerate(response.results):
                        legacy_error = result_pb.HasField("error_info") and (
                            result_pb.error_info.error_code != ErrorCodePB.NONE_ERROR
                            or result_pb.error_info.error_message
                        )
                        has_error = legacy_error
                        if result_pb.HasField("error_details"):
                            code = result_pb.error_details.error_code
                            message = result_pb.error_details.error_message
                            try:
                                exception_type = ExceptionType(code)
                                code_name = exception_type.name
                            except ValueError:
                                exception_type = ExceptionType.UNKNOWN_ERROR
                                code_name = f"UNKNOWN({code})"
                            capture_backpressure = (
                                code == ExceptionType.HIDDEN_STATE_CAPTURE_BACKPRESSURE
                            )
                            # Never reinterpret an execution failure as admission rejection.
                            # Conflicting BP envelopes are not safe to retry, even opt-in.
                            if (code == 0 and legacy_error) or (
                                capture_backpressure
                                and (
                                    result_pb.HasField("final_output")
                                    or (
                                        result_pb.HasField("error_info")
                                        and result_pb.error_info.error_code
                                        != ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
                                    )
                                )
                            ):
                                raise FtRuntimeException(
                                    ExceptionType.EXECUTION_EXCEPTION,
                                    f"batch item {i} request [{inputs[i].request_id}] "
                                    f"has conflicting error fields: native={code}, "
                                    f"legacy={result_pb.error_info.error_code}, "
                                    f"final_output={result_pb.HasField('final_output')}; "
                                    f"{message}",
                                )
                            has_error = code != 0 or bool(message)
                            message = f"{code_name}: {message}" if message else code_name
                        elif legacy_error:
                            code = result_pb.error_info.error_code
                            exception_type = _trans_rpc_error_code(code)
                            message = (
                                result_pb.error_info.error_message
                                or _rpc_error_code_name(code)
                            )
                            capture_backpressure = (
                                code == ErrorCodePB.HIDDEN_STATE_CAPTURE_BACKPRESSURE
                            )
                        if has_error:
                            error = FtRuntimeException(
                                exception_type,
                                f"batch item {i} request [{inputs[i].request_id}] failed: {message}",
                            )
                            if preserve_capture_backpressure and capture_backpressure:
                                results.append(error)
                                continue
                            raise error
                        stream_state = StreamState()
                        output = trans_output(
                            batch_inputs[i], result_pb.final_output, stream_state
                        )
                        _validate_prefill_only_completed(batch_inputs[i], stream_state)
                        results.append(output)
                    return results
                finally:
                    if response_call is not None and not completed:
                        response_call.cancel()
        except asyncio.TimeoutError as error:
            raise FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT,
                "timed out waiting for RPC channel capacity",
            ) from error
        except grpc.RpcError as error:
            self._handle_grpc_error(
                error, f"batch request: [{len(inputs)} items]", target_address
            )

    async def batch_enqueue(self, inputs: list[GenerateInput]) -> list[GenerateOutputs]:
        """Compatibility alias retaining atomic batch failure semantics."""
        return await self.batch_generate(inputs)

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
        logging.debug(
            f"request: [{input_pb.request_id}] send to address: {address_list[input_py.request_id % len(address_list)]}"
        )
        stub = None
        stream_done = False
        terminal_seen = False
        resources = AsyncExitStack()
        acquire_start = time.monotonic()
        try:
            # Select target address
            target_address = address_list[input_py.request_id % len(address_list)]
            logging.debug(f"target_address: {target_address}")
            # Get channel from pool
            channel = await resources.enter_async_context(
                self._channel_pool.acquire(
                    target_address,
                    timeout=effective_ms / 1000.0 if effective_ms > 0 else None,
                )
            )
            stub = RpcServiceStub(channel)

            grpc_kwargs = (
                {
                    "timeout": max(
                        0.0, effective_ms / 1000.0 - (time.monotonic() - acquire_start)
                    )
                }
                if effective_ms > 0
                else {}
            )
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
            _validate_prefill_only_completed(input_py, stream_state)
        except asyncio.TimeoutError as e:
            raise FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT,
                "timed out waiting for RPC channel capacity",
            ) from e
        except grpc.RpcError as e:
            # TODO(xinfei.sxf) 非流式的请求无法取消了
            if response_iterator:
                response_iterator.cancel()
            self._handle_grpc_error(
                e, f"request: [{input_pb.request_id}]", target_address
            )
        except Exception as e:
            logging.error(f"rpc unknown error:{str(e)}")
            raise e
        finally:
            should_cancel = not stream_done and not (
                use_fetch_response and terminal_seen
            )
            if response_iterator and should_cancel:
                response_iterator.cancel()
            await resources.aclose()

    def _handle_grpc_error(
        self, e: grpc.RpcError, request_desc: str, target_address: str = ""
    ) -> None:
        # NOTE: keep the backend peer (target_address) in the log lines ONLY.
        # Do NOT append it to the FtRuntimeException message, which is
        # serialized into the client-facing error response and would leak
        # internal cluster topology (worker ip:port) to callers.
        peer_desc = f" to [{target_address}]" if target_address else ""
        error_details = ErrorDetailsPB()
        metadata = dict(e.trailing_metadata() or ())
        if "grpc-status-details-bin" in metadata and error_details.ParseFromString(
            metadata["grpc-status-details-bin"]
        ):
            raw_error_code = error_details.error_code
            try:
                exception_type = ExceptionType(raw_error_code)
                error_code_name = exception_type.name
            except ValueError:
                exception_type = ExceptionType.UNKNOWN_ERROR
                error_code_name = f"UNKNOWN({raw_error_code})"
            logging.error(
                f"{request_desc} RPC{peer_desc} failed: "
                f"{e.code()}, {e.details()}, detail error code is "
                f"{error_code_name}"
            )
            raise FtRuntimeException(exception_type, error_details.error_message)
        else:
            logging.error(
                f"{request_desc} RPC{peer_desc} failed: "
                f"error code is {e.code()}, detail is {e.details()}"
            )
            if e.code() == StatusCode.DEADLINE_EXCEEDED:
                raise FtRuntimeException(
                    ExceptionType.GENERATE_TIMEOUT, e.details()
                )
            elif e.code() == StatusCode.CANCELLED:
                raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
            elif e.code() == StatusCode.UNAVAILABLE:
                details = e.details() or ""
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
                raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())

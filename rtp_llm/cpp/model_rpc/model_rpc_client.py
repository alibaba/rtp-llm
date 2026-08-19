import asyncio
import functools
import json
import logging
import math
import time
from typing import Any, AsyncGenerator, Dict, Optional, Union

import grpc
from google.protobuf.wrappers_pb2 import StringValue
from grpc import StatusCode

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ReturnAllProbsMode, RoleType
from rtp_llm.config.response_format_compiler import validate_engine_ready
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    BatchGenerateInputPB,
    ErrorDetailsPB,
    FetchRequestPB,
    GenerateConfigPB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.server.request_headers import (
    extract_correlation_request_id,
    extract_trace_id,
)
from rtp_llm.telemetry import CURRENT_TRACE_STATE
from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry import start_client_span
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
    trans_from_tensor,
    trans_option,
    trans_option_cast,
    trans_tensor,
)

MAX_GRPC_TIMEOUT_SECONDS = 3600
RPC_CLEANUP_TIMEOUT_SECONDS = 0.1
RPC_SETTLE_TIMEOUT_SECONDS = 5.0
JsonableOption = Optional[Union[str, Dict[str, Any], bool]]


def _selected_pd_separation(
    selected_role: Optional[RoleType], generate_config: GenerateConfig
) -> Optional[bool]:
    if selected_role == RoleType.PDFUSION:
        return False
    if selected_role != RoleType.PREFILL:
        return None
    # Keep this aligned with PrefillRpcServer::GenerateStreamCall. These are
    # the request fields actually sent to the selected Prefill endpoint, so
    # the result describes its real PD-vs-local branch rather than process role.
    return (
        generate_config.max_new_tokens > 1
        and generate_config.num_beams <= 1
        and not generate_config.variable_num_beams
        and generate_config.num_return_sequences <= 1
        and generate_config.can_use_pd_separation
    )


def _engine_reported_finished(outputs: Optional[GenerateOutputs]) -> bool:
    """True when the engine flagged every sub-output of this response as done.

    `finished` comes straight from the engine protobuf (see trans_output), so it
    is the authoritative "generation is over" signal, unlike the renderer-side
    finish_reason which can also fire on renderer-side stop-word truncation
    while the engine keeps generating.
    """
    if outputs is None or not outputs.generate_outputs:
        return False
    return all(bool(out.finished) for out in outputs.generate_outputs)


def _request_completed_normally(trace_state: Any) -> bool:
    """True when renderer or root state proves normal request completion.

    Used to classify a teardown-time GeneratorExit/CancelledError. The renderer
    stops iterating as soon as every sequence has a finish_reason, which fires
    on stop-word truncation too. Context-dependent tokenization can make the
    renderer match a string stop word that the engine's token-level
    stop_words_list misses. The renderer publishes that deliberate completion
    before closing the backend generator. ``settled_ok`` remains a fail-open
    fallback for non-renderer consumers that close only after the root settles.
    """
    if trace_state is None:
        return False
    try:
        return trace_state.renderer_completed is True or trace_state.settled_ok is True
    except Exception:  # noqa: BLE001 - fail-open
        return False


async def _wait_for_rpc_termination(
    response_iterator: Any, timeout_seconds: Optional[float] = None
) -> Any:
    """Waits until grpc.aio has observed the server-side stream termination."""
    try:
        code = getattr(response_iterator, "code", None)
        if code is not None:
            if timeout_seconds is None:
                return await code()
            return await asyncio.wait_for(code(), timeout_seconds)
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        # This helper also runs from enqueue()'s finally block. Swallowing the
        # task cancellation there would turn caller cancellation into success.
        raise
    except Exception:  # noqa: BLE001 - span cleanup must stay fail-open
        pass
    return None


async def _settle_client_span_after_rpc(  # noqa: C901 - request-local lifecycle state machine
    response_iterator: Any,
    client_span: Any,
    outputs: Optional[GenerateOutputs],
    abandoned_event: "asyncio.Event",
    active_deadline: Optional[float] = None,
    include_all_sequences: bool = True,
) -> Any:
    """Settle a CLIENT span without letting observation own an active call.

    ``active_deadline`` is the absolute deadline of the underlying gRPC call.
    Once it expires, gRPC is responsible for producing its terminal status;
    only an explicit abandonment transfers cancellation ownership here. A
    missing deadline deliberately means that active observation has no local
    upper bound, matching ``enqueue()``'s no-deadline contract.
    """
    try:
        can_observe_rpc_status = callable(getattr(response_iterator, "code", None))
    except Exception:  # noqa: BLE001 - optional grpc.aio observation
        can_observe_rpc_status = False
    status = None
    status_task = None
    abandoned_task = None
    cleanup_timed_out = False
    try:
        if can_observe_rpc_status:
            loop = asyncio.get_running_loop()
            status_task = asyncio.create_task(
                _wait_for_rpc_termination(response_iterator)
            )
            abandoned_task = asyncio.create_task(abandoned_event.wait())

            while True:
                if status_task.done():
                    status = status_task.result()
                    break

                if abandoned_event.is_set():
                    # The consumer has handed the transport to this task. Give
                    # cleanup its own bounded window, even if active observation
                    # already ran past the request deadline.
                    cleanup_deadline = loop.time() + RPC_SETTLE_TIMEOUT_SECONDS
                    remaining = max(0.0, cleanup_deadline - loop.time())
                    done, _ = await asyncio.wait({status_task}, timeout=remaining)
                    if done:
                        status = status_task.result()
                    else:
                        try:
                            response_iterator.cancel()
                        except Exception:  # noqa: BLE001 - fail-open cleanup
                            pass
                        done, _ = await asyncio.wait(
                            {status_task}, timeout=RPC_CLEANUP_TIMEOUT_SECONDS
                        )
                        if done:
                            status = status_task.result()
                        else:
                            cleanup_timed_out = True
                    break

                remaining = (
                    active_deadline - loop.time()
                    if active_deadline is not None
                    else None
                )
                wait_set = {status_task, abandoned_task}
                if remaining is None:
                    await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                elif remaining > 0:
                    done, _ = await asyncio.wait(
                        wait_set,
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        # The active consumer still owns the transport. After the
                        # observation deadline, keep waiting for either the real
                        # status or an explicit ownership transfer from aclose().
                        continue
                else:
                    # A configured gRPC deadline should make code() resolve to
                    # DEADLINE_EXCEEDED. Do not add a second local cancellation
                    # at this boundary; wait for that transport terminal state
                    # or for the consumer to abandon ownership.
                    await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if client_span is not None:
            _record_client_rpc_status(client_span, status)
            _record_client_span_usage(
                client_span, outputs, include_all_sequences=include_all_sequences
            )
            _record_client_span_latency(client_span, outputs)
            if status == StatusCode.OK or not can_observe_rpc_status:
                client_span.finish()
            elif cleanup_timed_out:
                client_span.finish(error_type="RpcSettlementTimeout")
            else:
                client_span.finish(error_type="RpcError")
        return status
    except asyncio.CancelledError as cancellation:
        if status_task is not None and status_task.done():
            try:
                status = status_task.result()
            except BaseException:  # noqa: BLE001 - preserve task cancellation
                pass
        if client_span is not None:
            _record_client_rpc_status(client_span, status)
            _record_client_span_usage(
                client_span, outputs, include_all_sequences=include_all_sequences
            )
            _record_client_span_latency(client_span, outputs)
            client_span.finish(error=cancellation, error_type="RpcSettlementCancelled")
        if abandoned_event.is_set():
            try:
                response_iterator.cancel()
            except Exception:  # noqa: BLE001 - fail-open cleanup
                pass
        raise
    finally:
        pending_tasks = []
        for task in (status_task, abandoned_task):
            if task is not None and not task.done():
                task.cancel()
                pending_tasks.append(task)
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)


def _consume_settlement_task(task: "asyncio.Task[Any]") -> None:
    try:
        task.result()
    except BaseException:  # noqa: BLE001 - detached cleanup is fail-open
        logging.debug(
            "client span settlement task ended before RPC status", exc_info=True
        )


def _record_client_rpc_status(client_span: Any, status: Any) -> None:
    if client_span is None or status is None:
        return
    try:
        value = getattr(status, "name", None)
        if value:
            client_span.set_attribute(trace_attrs.RPC_RESPONSE_STATUS_CODE, str(value))
    except Exception:  # noqa: BLE001 - fail-open
        pass


def _record_client_span_usage(
    client_span: Any,
    outputs: Optional[GenerateOutputs],
    *,
    include_all_sequences: bool = True,
) -> None:
    """Writes the per-hop token attributes on the gRPC CLIENT span.

    The platform top-bar Total tokens only aggregates caller-side spans (it
    needs a plain client span carrying gen_ai.span.kind=LLM plus total_tokens),
    so the pair is duplicated here for direct deployments without the gateway.
    Full five-key set (semconv + legacy aliases, mirroring
    telemetry::setUsageTokenAttributes): the tokens tooltip reads
    prompt_tokens/completion_tokens, not total. Fail-open; a span already
    finished drops these writes.
    """
    if client_span is None:
        return
    try:
        generate_outputs = outputs.generate_outputs if outputs is not None else []
        if not generate_outputs:
            return
        if not include_all_sequences:
            generate_outputs = generate_outputs[:1]
        aux_infos = [out.aux_info for out in generate_outputs]
        input_len = aux_infos[0].input_len
        if (
            input_len <= 0
            or any(aux.input_len != input_len for aux in aux_infos)
            or any(aux.output_len <= 0 for aux in aux_infos)
        ):
            return
        output_len = sum(aux.output_len for aux in aux_infos)
        client_span.set_attribute(trace_attrs.GEN_AI_SPAN_KIND, "LLM")
        client_span.set_attribute(trace_attrs.GEN_AI_USAGE_INPUT_TOKENS, input_len)
        client_span.set_attribute(trace_attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_len)
        client_span.set_attribute(trace_attrs.GEN_AI_USAGE_PROMPT_TOKENS, input_len)
        client_span.set_attribute(
            trace_attrs.GEN_AI_USAGE_COMPLETION_TOKENS, output_len
        )
        client_span.set_attribute(
            trace_attrs.GEN_AI_USAGE_TOTAL_TOKENS, input_len + output_len
        )
    except Exception:  # noqa: BLE001 - fail-open
        pass


def _record_client_span_latency(
    client_span: Any, outputs: Optional[GenerateOutputs]
) -> None:
    """Writes Engine TTFT/TPOT on one physical engine CLIENT span.

    Multi-return (n>1) is served by a single physical stream, so every sequence
    shares the prefill that commits the first token: Engine TTFT is written only
    when all sequences agree on it. The per-token decode interval, by contrast,
    is per-sequence and has no unambiguous stream-level value, so Engine TPOT is
    restricted to single-sequence streams instead of silently publishing
    sequence 0 as if it described the whole span.
    """
    if client_span is None:
        return
    try:
        generate_outputs = outputs.generate_outputs if outputs is not None else []
        if not generate_outputs:
            return
        aux_infos = [out.aux_info for out in generate_outputs]
        if any(
            not isinstance(aux.output_len, int)
            or isinstance(aux.output_len, bool)
            or aux.output_len <= 0
            for aux in aux_infos
        ):
            return
        ttft_ms = aux_infos[0].first_token_cost_time
        if (
            not isinstance(ttft_ms, (int, float))
            or isinstance(ttft_ms, bool)
            or not math.isfinite(float(ttft_ms))
            or ttft_ms <= 0
            or any(aux.first_token_cost_time != ttft_ms for aux in aux_infos)
        ):
            return
        client_span.set_attribute(
            trace_attrs.RTP_LLM_ENGINE_TIME_TO_FIRST_TOKEN_MS, float(ttft_ms)
        )

        if len(aux_infos) != 1:
            return
        output_len = aux_infos[0].output_len
        cost_ms = aux_infos[0].cost_time
        if (
            output_len > 1
            and isinstance(cost_ms, (int, float))
            and not isinstance(cost_ms, bool)
            and math.isfinite(float(cost_ms))
            and cost_ms >= ttft_ms
        ):
            client_span.set_attribute(
                trace_attrs.RTP_LLM_ENGINE_TIME_PER_OUTPUT_TOKEN_MS,
                float(cost_ms - ttft_ms) / (output_len - 1),
            )
    except Exception:  # noqa: BLE001 - fail-open
        pass


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}


def _is_finished_response(outputs_pb: GenerateOutputsPB) -> bool:
    finished = outputs_pb.flatten_output.finished
    return bool(finished) and all(finished)


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


def _trans_jsonable_option(
    option_pb: StringValue, name: str, value: JsonableOption
) -> None:
    """Serialize structured config exactly once at the protobuf boundary."""
    if value is None:
        return
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as e:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"{name} must be serializable as JSON: {str(e)}",
            ) from e
    option_pb.value = value


def _trans_jsonable_options(
    config_pb: GenerateConfigPB, config: GenerateConfig
) -> None:
    _trans_jsonable_option(config_pb.json_schema, "json_schema", config.json_schema)
    _trans_jsonable_option(config_pb.regex, "regex", config.regex)
    _trans_jsonable_option(config_pb.ebnf, "ebnf", config.ebnf)
    _trans_jsonable_option(
        config_pb.structural_tag, "structural_tag", config.structural_tag
    )


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
    # Preserve main's regular GenerateConfig validation at the RPC boundary,
    # then assert (without mutating) that the request entrypoint prepared grammar.
    input_py.generate_config.validate()
    validate_engine_ready(input_py.generate_config)

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.max_thinking_tokens = (
        input_py.generate_config.max_thinking_tokens
    )
    generate_config_pb.begin_think_token_ids.extend(
        input_py.generate_config.begin_think_token_ids
    )
    generate_config_pb.end_think_token_ids.extend(
        input_py.generate_config.end_think_token_ids
    )
    generate_config_pb.in_think_mode = input_py.generate_config.in_think_mode
    generate_config_pb.thinking_mode = int(input_py.generate_config.thinking_mode)
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
    _trans_jsonable_options(generate_config_pb, input_py.generate_config)
    trans_option(generate_config_pb, input_py.generate_config, "adapter_name")
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "task_id", functools.partial(str)
    )

    generate_config_pb.select_tokens_id.extend(
        input_py.generate_config.select_tokens_id
    )
    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_prompt_logits = (
        input_py.generate_config.return_prompt_logits
    )
    generate_config_pb.prompt_logits_top_k = (
        input_py.generate_config.prompt_logits_top_k
    )
    generate_config_pb.prompt_logits_start = (
        input_py.generate_config.prompt_logits_start
    )
    generate_config_pb.prompt_logits_end = input_py.generate_config.prompt_logits_end
    generate_config_pb.return_target_logprob = (
        input_py.generate_config.return_target_logprob
    )
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
    generate_config_pb.ignore_eos = input_py.generate_config.ignore_eos
    generate_config_pb.reuse_cache = input_py.generate_config.reuse_cache
    generate_config_pb.enable_host_cache = (
        input_py.generate_config.enable_host_cache
    )
    generate_config_pb.enable_device_cache = (
        input_py.generate_config.enable_device_cache
    )
    generate_config_pb.enable_remote_cache = (
        input_py.generate_config.enable_remote_cache
    )
    generate_config_pb.enable_disk_cache = input_py.generate_config.enable_disk_cache
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "trace_id", functools.partial(str)
    )
    trans_option(generate_config_pb, input_py.generate_config, "group_timeout")

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    # 生成式推荐：组合 token 约束
    generate_config_pb.combo_token_size = input_py.generate_config.combo_token_size
    generate_config_pb.enable_cross_sequence_ban = (
        input_py.generate_config.enable_cross_sequence_ban
    )
    generate_config_pb.cross_seq_diverge_start_combo = (
        input_py.generate_config.cross_seq_diverge_start_combo
    )
    for i in range(len(input_py.generate_config.banned_combo_token_ids)):
        banned_combo = generate_config_pb.banned_combo_token_ids.rows.add()
        banned_combo.values.extend(input_py.generate_config.banned_combo_token_ids[i])

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

    prompt_logits_data = None
    if output_pb.HasField("prompt_logits") and output_pb.prompt_logits.HasField(
        "topk_logprobs"
    ):
        pl_pb = output_pb.prompt_logits
        prompt_logits_data = {
            "topk_logprobs": trans_tensor(pl_pb.topk_logprobs),
            "topk_token_ids": trans_tensor(pl_pb.topk_token_ids),
            "target_logprobs": (
                trans_tensor(pl_pb.target_logprobs)
                if pl_pb.HasField("target_logprobs")
                else None
            ),
            "start_pos": pl_pb.start_pos,
            "end_pos": pl_pb.end_pos,
        }

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
                disk_reuse_len=aux_info_pb.disk_reuse_len,
                prefill_total_reuse_len=aux_info_pb.prefill_total_reuse_len,
                prefill_local_reuse_len=aux_info_pb.prefill_local_reuse_len,
                prefill_remote_reuse_len=aux_info_pb.prefill_remote_reuse_len,
                prefill_memory_reuse_len=aux_info_pb.prefill_memory_reuse_len,
                prefill_disk_reuse_len=aux_info_pb.prefill_disk_reuse_len,
                decode_total_reuse_len=aux_info_pb.decode_total_reuse_len,
                decode_local_reuse_len=aux_info_pb.decode_local_reuse_len,
                decode_remote_reuse_len=aux_info_pb.decode_remote_reuse_len,
                decode_memory_reuse_len=aux_info_pb.decode_memory_reuse_len,
                decode_disk_reuse_len=aux_info_pb.decode_disk_reuse_len,
                speculative_draft_rounds=aux_info_pb.speculative_draft_rounds,
                speculative_accepted_tokens_per_pos=list(
                    aux_info_pb.speculative_accepted_tokens_per_pos
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

        if prompt_logits_data is not None:
            output_py.prompt_logits = prompt_logits_data

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

    async def close(self) -> None:
        await self._channel_pool.close()

    def _compute_grpc_timeout(self, timeout_ms) -> float:
        rpc_timeout_ms = (
            self._max_rpc_timeout_ms
            if self._max_rpc_timeout_ms > 0
            else MAX_GRPC_TIMEOUT_SECONDS * 1000
        )
        if timeout_ms is None or timeout_ms <= 0:
            return rpc_timeout_ms / 1000
        return timeout_ms / 1000

    def _handle_grpc_error(
        self, e: grpc.RpcError, request_desc: str, target_address: str = ""
    ) -> None:
        # NOTE: keep the backend peer (target_address) in the log lines ONLY.
        # Do NOT append it to the FtRuntimeException message, which is
        # serialized into the client-facing error response and would leak
        # internal cluster topology (worker ip:port) to callers.
        peer_desc = f" to [{target_address}]" if target_address else ""
        error_details = ErrorDetailsPB()
        metadata = e.trailing_metadata()
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
                raise FtRuntimeException(ExceptionType.GENERATE_TIMEOUT, e.details())
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

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        request_timeout_ms = input_py.generate_config.timeout_ms
        # Prefer per-request timeout; otherwise fall back to the server-side default
        # (pd_sep_config.max_rpc_timeout_ms). effective_ms <= 0 means no gRPC
        # deadline and therefore no local active-consumer upper bound.
        effective_ms = (
            request_timeout_ms
            if request_timeout_ms is not None and request_timeout_ms > 0
            else self._max_rpc_timeout_ms
        )
        if effective_ms > 0:
            # Write the normalized timeout back so the server-visible timeout_ms
            # matches the client gRPC deadline: engine-side timeout checks and
            # P2P deadlineMs() require a positive timeout_ms.
            input_py.generate_config.timeout_ms = int(effective_ms)
        input_pb = trans_input(input_py)
        response_iterator = None
        rpc_status = None
        stream_state = StreamState()
        include_all_sequences = not input_py.generate_config.has_num_beams()
        use_fetch_response = bool(getattr(input_py, "enqueued_by_master", False))
        selected_role = None

        if use_fetch_response:
            address_list = [
                role_addr.ip + ":" + str(role_addr.grpc_port)
                for role_addr in input_py.generate_config.role_addrs
                if role_addr.role == RoleType.PREFILL and role_addr.ip
            ]
            if address_list:
                # FetchResponse targets the Prefill endpoint the master enqueued
                # on, so the PD attribute follows the same Prefill semantics as
                # the streaming channel below.
                selected_role = RoleType.PREFILL
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
                        selected_role = role_addr.role
                        break

        if not address_list:
            raise ValueError(f"No address found for request: {input_py.request_id}")
        # Select target address before entering the try block so it is always
        # available to the error handlers below (surfaced in logs only)
        # details to identify which backend peer dropped the connection).
        target_address = address_list[input_py.request_id % len(address_list)]
        logging.debug(
            f"request: [{input_py.request_id}] send to address: {target_address}"
        )
        stream_done = False
        terminal_seen = False
        client_settlement_task = None
        client_settlement_abandoned = None
        rpc_deadline = None

        trace_state = CURRENT_TRACE_STATE.get()
        pd_separation = _selected_pd_separation(selected_role, input_py.generate_config)
        if pd_separation is not None and trace_state is not None:
            trace_state.set_attribute(trace_attrs.RTP_LLM_PD_SEP, pd_separation)

        # gRPC CLIENT span: child of the HTTP SERVER span
        # published via CURRENT_TRACE_STATE; W3C traceparent goes into gRPC
        # metadata. Both are no-ops when telemetry is disabled.
        client_span, trace_metadata = start_client_span(
            "rtp_llm.generate_stream_call", target_address
        )
        if client_span is not None:
            client_settlement_abandoned = asyncio.Event()
            # Bailian Unitrace index key (see rtp_llm/telemetry/attributes.py)
            client_span.set_attribute(trace_attrs.REQUEST_ID, str(input_py.request_id))
            client_span.set_attribute(
                trace_attrs.RTP_LLM_REQUEST_ID, input_py.request_id
            )
        last_output = None

        try:
            # Get channel from pool
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            grpc_kwargs = {}
            if effective_ms > 0:
                grpc_kwargs["timeout"] = effective_ms / 1000.0
            if trace_metadata:
                # One injection point covers both channels: W3C traceparent
                # rides gRPC metadata for FetchResponse and GenerateStreamCall.
                grpc_kwargs["metadata"] = trace_metadata
            if effective_ms > 0:
                # grpc.aio starts this timeout when the call is created. The
                # observer uses the same absolute boundary, so time spent
                # receiving application frames is included.
                rpc_deadline = asyncio.get_running_loop().time() + effective_ms / 1000.0
            if use_fetch_response:
                response_iterator = stub.FetchResponse(
                    FetchRequestPB(request_id=input_pb.request_id), **grpc_kwargs
                )
            else:
                response_iterator = stub.GenerateStreamCall(input_pb, **grpc_kwargs)
            # 调用服务器方法并接收流式响应
            async for response in response_iterator.__aiter__():
                output_py = trans_output(input_py, response, stream_state)
                last_output = output_py
                if use_fetch_response and _is_finished_response(response):
                    terminal_seen = True
                if _engine_reported_finished(output_py) and client_span is not None:
                    # The finished application frame is not the gRPC EOF. If it
                    # escapes first, an upstream renderer can close this generator
                    # while the server is still settling the RPC. The application
                    # frame is the data-plane completion boundary; keep terminal
                    # observation off that path and settle the span independently.
                    if client_settlement_task is None:
                        client_settlement_task = asyncio.create_task(
                            _settle_client_span_after_rpc(
                                response_iterator,
                                client_span,
                                output_py,
                                client_settlement_abandoned,
                                active_deadline=rpc_deadline,
                                include_all_sequences=include_all_sequences,
                            )
                        )
                        client_settlement_task.add_done_callback(
                            _consume_settlement_task
                        )
                yield output_py
            stream_done = True
        except grpc.RpcError as e:
            rpc_status = e.code()
            if client_span is not None:
                _record_client_rpc_status(client_span, rpc_status)
                _record_client_span_usage(
                    client_span,
                    last_output,
                    include_all_sequences=include_all_sequences,
                )
                _record_client_span_latency(client_span, last_output)
                client_span.finish(error=e, error_type="RpcError")
            if response_iterator:
                response_iterator.cancel()
            self._handle_grpc_error(
                e, f"request: [{input_pb.request_id}]", target_address
            )
        except (asyncio.CancelledError, GeneratorExit) as e:
            # Client disconnect / stream teardown: these are BaseException
            # subclasses the `except Exception` below never sees, and the
            # finally fallback would end the CLIENT span as OK — yielding
            # contradictory SERVER=Cancelled / CLIENT=OK traces. Cancel the
            # RPC first, settle the span, then re-raise so the cancellation
            # keeps propagating.
            # Which status is truthful cannot be decided from the exception
            # type: both a real disconnect and the renderer deliberately
            # closing a completed response arrive here as GeneratorExit. The
            # renderer milestone distinguishes those paths before root span
            # settlement; a root already settled OK remains a fallback.
            engine_finished = _engine_reported_finished(last_output)
            if response_iterator:
                if not engine_finished:
                    response_iterator.cancel()
                # A CLIENT span covers grpc.aio's terminal RPC state. Engine-finished
                # streams settle independently; after local cancellation, remote
                # handler cleanup can still be asynchronous.
                if (
                    rpc_status is None
                    and client_span is not None
                    and client_settlement_task is None
                ):
                    rpc_status = await _wait_for_rpc_termination(
                        response_iterator, timeout_seconds=RPC_CLEANUP_TIMEOUT_SECONDS
                    )
                if client_settlement_task is None:
                    _record_client_rpc_status(client_span, rpc_status)
            if client_span is not None and client_settlement_task is None:
                # Keep usage for both successful cleanup and a genuinely
                # cancelled stream. The last delivered response is confirmed
                # work; writing it after finish() would be dropped.
                _record_client_span_usage(
                    client_span,
                    last_output,
                    include_all_sequences=include_all_sequences,
                )
                if engine_finished or _request_completed_normally(trace_state):
                    _record_client_span_latency(client_span, last_output)
                    client_span.finish()
                else:
                    client_span.finish(error=e, error_type="Cancelled")
            raise
        except Exception as e:
            if response_iterator:
                response_iterator.cancel()
                if (
                    rpc_status is None
                    and client_span is not None
                    and client_settlement_task is None
                ):
                    rpc_status = await _wait_for_rpc_termination(
                        response_iterator, timeout_seconds=RPC_CLEANUP_TIMEOUT_SECONDS
                    )
                if client_settlement_task is None:
                    _record_client_rpc_status(client_span, rpc_status)
            if client_span is not None and client_settlement_task is None:
                _record_client_span_usage(
                    client_span,
                    last_output,
                    include_all_sequences=include_all_sequences,
                )
                _record_client_span_latency(client_span, last_output)
                client_span.finish(error=e)
            logging.error(
                f"request: [{input_pb.request_id}] rpc to [{target_address}] unknown error: {str(e)}"
            )
            raise e
        finally:
            try:
                if client_span is not None:
                    if (
                        client_settlement_task is not None
                        and not stream_done
                        and client_settlement_abandoned is not None
                    ):
                        client_settlement_abandoned.set()
                    # Normal completion has a detached settlement task. Do not
                    # await it here, otherwise aclose() would reintroduce the
                    # finished-frame blocking regression.
                    if client_settlement_task is not None:
                        if client_settlement_task.done():
                            try:
                                rpc_status = client_settlement_task.result()
                            except BaseException:  # noqa: BLE001 - fail-open
                                pass
                        else:
                            rpc_status = None
                    elif response_iterator and rpc_status is None:
                        rpc_status = await _wait_for_rpc_termination(
                            response_iterator,
                            timeout_seconds=RPC_CLEANUP_TIMEOUT_SECONDS,
                        )
                    if client_settlement_task is None or client_settlement_task.done():
                        _record_client_rpc_status(client_span, rpc_status)
                        _record_client_span_usage(
                            client_span,
                            last_output,
                            include_all_sequences=include_all_sequences,
                        )
                        _record_client_span_latency(client_span, last_output)
                        # success/cancel fallback; idempotent with the error paths above
                        client_span.finish()
            except asyncio.CancelledError as cleanup_cancel:
                if response_iterator:
                    try:
                        response_iterator.cancel()
                    except Exception:  # noqa: BLE001 - preserve caller cancellation
                        pass
                if client_span is not None and client_settlement_task is None:
                    _record_client_rpc_status(client_span, StatusCode.CANCELLED)
                    _record_client_span_usage(
                        client_span,
                        last_output,
                        include_all_sequences=include_all_sequences,
                    )
                    client_span.finish(error=cleanup_cancel, error_type="Cancelled")
                raise
            should_cancel = (
                not stream_done
                and client_settlement_task is None
                and not (use_fetch_response and terminal_seen)
            )
            if response_iterator and should_cancel:
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

        target_address = self._addresses[inputs[0].request_id % len(self._addresses)]
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

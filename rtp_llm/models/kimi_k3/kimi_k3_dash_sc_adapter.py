"""DashSc request semantics for Kimi K3's channel and media protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.dash_sc.codec import (
    DashScRequestControls,
    SamplingParams,
    parse_messages_from_request,
    parse_multimodal_parts_from_request,
)
from rtp_llm.dash_sc.inference.request_adapter import (
    DashScRequestAdapter,
    DashScRequestContext,
    PreparedDashScRequest,
    PromptTokenizer,
    build_multimodal_inputs_from_request,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_count,
    validate_kimi_k3_tool_history,
    validate_kimi_k3_top_logprobs,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType

if TYPE_CHECKING:
    from rtp_llm.ops import MultimodalInput

_INT32_MAX = 2_147_483_647

_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
_K3_MEDIA_PAD = "<|media_pad|>"
_K3_MEDIA_BEGIN = "<|media_begin|>"
_K3_MEDIA_END = "<|media_end|>"
_K3_THINK_TO_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>"


def _resolve_k3_dash_sc_thinking(
    sampling: SamplingParams,
    request_controls: DashScRequestControls,
    generate_config: GenerateConfig,
) -> bool:
    """Apply K3's explicit thinking controls before its structured-output default."""
    if (
        request_controls.enable_thinking is False
        or sampling.max_new_think_tokens == 0
        or request_controls.max_new_think_tokens == 0
    ):
        return False
    if (
        request_controls.enable_thinking is True
        or sampling.max_new_think_tokens is not None
        or request_controls.max_new_think_tokens is not None
        or request_controls.reasoning_effort is not None
    ):
        return True
    response_format = generate_config.response_format
    if response_format is not None and response_format.type != "text":
        return False
    if sampling.json_format or generate_config.structural_tag is not None:
        return False
    return True


def _k3_token_ids(tokenizer: PromptTokenizer | None, text: str) -> list[int]:
    native = tokenizer
    if native is None:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 tokenizer is required for image prompt normalization",
        )
    return list(native.encode(text))


def _k3_marker_offsets(input_ids: list[int], marker: list[int]) -> list[int]:
    if not marker:
        return []
    offsets = []
    cursor = 0
    while cursor <= len(input_ids) - len(marker):
        if input_ids[cursor : cursor + len(marker)] == marker:
            offsets.append(cursor)
            cursor += len(marker)
        else:
            cursor += 1
    return offsets


def _normalize_k3_image_prompt(
    request: predict_v2_pb2.ModelInferRequest,
    input_ids: list[int],
    tokenizer: PromptTokenizer | None,
) -> tuple[list[int], list[MultimodalInput]]:
    """Reduce each upstream image span to one ViT-owned media-pad token."""
    parts = parse_multimodal_parts_from_request(request)
    if not parts:
        return input_ids, []
    if any(part.mm_type != MMUrlType.IMAGE for part in parts):
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 supports only image multimodal inputs",
        )

    placeholder = _k3_token_ids(tokenizer, _K3_IMAGE_PLACEHOLDER)
    media_pad = _k3_token_ids(tokenizer, _K3_MEDIA_PAD)
    if not placeholder or not media_pad:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR,
            "Kimi K3 image markers could not be tokenized",
        )
    placeholders = _k3_marker_offsets(input_ids, placeholder)
    pads = _k3_marker_offsets(input_ids, media_pad)
    if placeholders:
        if pads or len(placeholders) != len(parts):
            raise FtRuntimeException(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                "Kimi K3 image placeholder count does not match image count",
            )
        result = []
        cursor = 0
        for offset in placeholders:
            result.extend(input_ids[cursor:offset])
            result.extend(media_pad)
            cursor = offset + len(placeholder)
        result.extend(input_ids[cursor:])
    else:
        if len(pads) != len(parts):
            raise FtRuntimeException(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                "Kimi K3 media prompt count does not match image count",
            )
        begin = _k3_marker_offsets(input_ids, _k3_token_ids(tokenizer, _K3_MEDIA_BEGIN))
        end_ids = _k3_token_ids(tokenizer, _K3_MEDIA_END)
        end = _k3_marker_offsets(input_ids, end_ids)
        if begin or end:
            if len(begin) != len(parts) or len(end) != len(parts):
                raise FtRuntimeException(
                    ExceptionType.MM_WRONG_FORMAT_ERROR,
                    "Kimi K3 expanded image boundaries do not match image count",
                )
            result = []
            cursor = 0
            for start, pad, finish in zip(begin, pads, end):
                if not cursor <= start < pad < finish:
                    raise FtRuntimeException(
                        ExceptionType.MM_WRONG_FORMAT_ERROR,
                        "Kimi K3 image marker order is invalid",
                    )
                result.extend(input_ids[cursor:start])
                result.extend(media_pad)
                cursor = finish + len(end_ids)
            result.extend(input_ids[cursor:])
        else:
            result = input_ids
    return result, build_multimodal_inputs_from_request(request)


class KimiK3DashScRequestAdapter(DashScRequestAdapter):
    def prepare(self, context: DashScRequestContext) -> PreparedDashScRequest:
        sampling = context.sampling
        controls = context.controls
        config = context.generate_config
        validate_kimi_k3_top_logprobs(sampling.top_logprobs)
        messages = parse_messages_from_request(context.request)
        if messages is not None:
            validate_kimi_k3_tool_history(messages, allow_partial=True)
        input_ids, mm_inputs = _normalize_k3_image_prompt(
            context.request, context.input_ids, context.tokenizer
        )
        thinking = _resolve_k3_dash_sc_thinking(sampling, controls, config)
        config.thinking_mode = (
            ThinkingMode.ENABLED if thinking else ThinkingMode.DISABLED
        )
        config.in_think_mode = thinking
        if thinking:
            request_budget = sampling.max_new_think_tokens
            if request_budget is None:
                request_budget = controls.max_new_think_tokens
            config.max_thinking_tokens = (
                _INT32_MAX
                if request_budget is not None and request_budget < 0
                else request_budget if request_budget is not None else 32000
            )
        apply_kimi_k3_request_contract(
            config, specified_fields=sampling.specified_fields, thinking=thinking
        )
        return PreparedDashScRequest(
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            prompt_token_offset=kimi_k3_pending_prompt_token_count(
                context.tokenizer, input_ids
            ),
            reasoning_format=(
                ReasoningFormat(
                    tag_begin="",
                    tag_end=_K3_THINK_TO_RESPONSE,
                    tag_end_native_encoding=True,
                )
                if thinking
                else None
            ),
        )

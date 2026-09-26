"""DashSc request semantics for Kimi K3's channel and media protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.dash_sc.codec import (
    DashScRequestControls,
    SamplingParams,
    parse_messages_from_request,
)
from rtp_llm.dash_sc.inference.request_adapter import (
    DashScRequestAdapter,
    DashScRequestContext,
    PreparedDashScRequest,
    PromptTokenizer,
    _build_mm_inputs_from_request,
)
from rtp_llm.models.kimi_k3.kimi_k3_grammar import KimiK3GrammarFormat
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_prompt_channel,
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


def _resolve_k3_dash_sc_thinking(
    sampling: SamplingParams,
    request_controls: DashScRequestControls,
    generate_config: GenerateConfig,
    prompt_thinking: bool | None = None,
) -> bool:
    """Resolve request thinking independently of who owns its grammar."""
    if request_controls.enable_thinking is not None:
        return request_controls.enable_thinking
    budget = sampling.max_new_think_tokens
    if budget is None:
        budget = request_controls.max_new_think_tokens
    if budget is not None:
        return budget != 0
    if prompt_thinking is not None:
        return prompt_thinking
    if request_controls.reasoning_effort is not None:
        return True
    if generate_config.thinking_mode in (ThinkingMode.ENABLED, ThinkingMode.DISABLED):
        return generate_config.thinking_mode == ThinkingMode.ENABLED
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
    input_ids: list[int],
    tokenizer: PromptTokenizer | None,
    mm_inputs: list[MultimodalInput],
) -> list[int]:
    """Reduce each upstream image span to one ViT-owned media-pad token."""
    if not mm_inputs:
        return input_ids
    if any(part.mm_type != MMUrlType.IMAGE for part in mm_inputs):
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
        if pads or len(placeholders) != len(mm_inputs):
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
        if len(pads) != len(mm_inputs):
            raise FtRuntimeException(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                "Kimi K3 media prompt count does not match image count",
            )
        begin = _k3_marker_offsets(input_ids, _k3_token_ids(tokenizer, _K3_MEDIA_BEGIN))
        end_ids = _k3_token_ids(tokenizer, _K3_MEDIA_END)
        end = _k3_marker_offsets(input_ids, end_ids)
        if begin or end:
            if len(begin) != len(mm_inputs) or len(end) != len(mm_inputs):
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
    return result


class KimiK3DashScRequestAdapter(DashScRequestAdapter):
    def prepare(self, context: DashScRequestContext) -> PreparedDashScRequest:
        sampling = context.sampling
        controls = context.controls
        config = context.generate_config
        validate_kimi_k3_top_logprobs(getattr(sampling, "top_logprobs", 0))
        messages = context.messages
        if messages is None:
            messages = parse_messages_from_request(context.request)
        if messages is not None:
            validate_kimi_k3_tool_history(messages, allow_partial=True)
        mm_inputs = context.mm_inputs
        if mm_inputs is None:
            mm_inputs = _build_mm_inputs_from_request(
                context.request, messages=messages
            )
        input_ids = _normalize_k3_image_prompt(
            context.input_ids, context.tokenizer, mm_inputs
        )
        prompt_info = kimi_k3_prompt_channel(context.tokenizer, input_ids)
        channel, prompt_token_offset = prompt_info
        thinking = _resolve_k3_dash_sc_thinking(
            sampling,
            controls,
            config,
            prompt_thinking=None if channel is None else channel == "think",
        )
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
        config._structural_tag_is_complete = config.structural_tag is not None
        config._grammar_format = KimiK3GrammarFormat(
            upstream_grammar=config._structural_tag_is_complete,
            prompt_info=prompt_info,
        ).prepare(config, context.tokenizer, input_ids)
        return PreparedDashScRequest(
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            prompt_token_offset=prompt_token_offset,
        )

import json
from math import isclose
from typing import AbstractSet, Any, NoReturn, Protocol, Sequence

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException


def kimi_k3_pending_prompt_token_ids(tokenizer: Any, thinking: bool) -> list[int]:
    """Encode the trusted open generation channel excluded from public usage."""
    channel = "think" if thinking else "response"
    # Use the native tokenizer's default special-token policy for XTML tags,
    # as in K3 prompt rendering; do not pass HF-only encoding kwargs.
    token_ids = tokenizer.encode(f"<|open|>{channel}<|sep|>")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(token_id, int) for token_id in token_ids)
    ):
        raise TypeError("Kimi K3 tokenizer.encode must return a non-empty List[int]")
    return token_ids


def kimi_k3_pending_prompt_token_count(tokenizer: Any, input_ids: list[int]) -> int:
    """Exclude only an exact pending-channel suffix of a pre-tokenized prompt."""
    if tokenizer is None or not input_ids:
        return 0
    for thinking in (False, True):
        pending_ids = kimi_k3_pending_prompt_token_ids(tokenizer, thinking)
        if input_ids[-len(pending_ids) :] == pending_ids:
            return len(pending_ids)
    return 0


class _GenerationConfig(Protocol):
    temperature: Any
    top_p: Any
    presence_penalty: Any
    frequency_penalty: Any
    num_return_sequences: int
    in_think_mode: bool
    max_thinking_tokens: int


_TOP_P_ABS_TOL = 1e-6


def _reject(name: str, value: Any, expected: str) -> NoReturn:
    raise FtRuntimeException(
        ExceptionType.INVALID_PARAMS,
        f"Kimi K3 requires {name} {expected}, got {value!r}",
    )


def _reject_n(value: Any) -> NoReturn:
    raise FtRuntimeException(
        ExceptionType.INVALID_PARAMS,
        f"Range of n should be [1, 1], got {value!r}",
    )


def _scalar_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        _reject(name, value, "to be a scalar")
    return float(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _reject_tool_history(detail: str) -> NoReturn:
    raise FtRuntimeException(
        ExceptionType.INVALID_PARAMS,
        f"Invalid Kimi K3 tool-call history: {detail}",
    )


def validate_kimi_k3_tool_history(
    messages: Sequence[Any], *, allow_partial: bool = False
) -> None:
    """Validate complete OpenAI tool-call turns before K3 prompt rendering.

    Tool results for one parallel assistant call may arrive in any order, but
    every result must match exactly one pending call before the conversation can
    advance.  The validator intentionally accepts either pydantic message models
    or their JSON dictionaries so both OpenAI and DashSc entrypoints can share it.

    DashSc media payloads may omit text-only turns. With ``allow_partial``, check
    the fields of the supplied calls/results without requiring complete pairing
    across turns. The OpenAI renderer always validates the complete history.
    """

    seen_call_ids: set[str] = set()
    pending_call_ids: set[str] = set()

    for message_index, message in enumerate(messages):
        role_value = _field(message, "role")
        role = getattr(role_value, "value", role_value)

        if not allow_partial and pending_call_ids and role != "tool":
            pending = ", ".join(sorted(pending_call_ids))
            _reject_tool_history(
                f"messages[{message_index}] advances the conversation before "
                f"tool results are supplied for: {pending}"
            )

        tool_calls = _field(message, "tool_calls")
        if tool_calls is not None and role != "assistant":
            _reject_tool_history(
                f"messages[{message_index}].tool_calls is only valid for role='assistant'"
            )

        if role == "assistant" and tool_calls:
            if not isinstance(tool_calls, (list, tuple)):
                _reject_tool_history(
                    f"messages[{message_index}].tool_calls must be an array"
                )
            for call_index, tool_call in enumerate(tool_calls):
                path = f"messages[{message_index}].tool_calls[{call_index}]"
                call_id = _field(tool_call, "id")
                if not isinstance(call_id, str) or not call_id.strip():
                    _reject_tool_history(f"{path}.id must be a non-empty string")
                if call_id in seen_call_ids:
                    _reject_tool_history(f"{path}.id duplicates {call_id!r}")
                if _field(tool_call, "type") != "function":
                    _reject_tool_history(f"{path}.type must be 'function'")

                function = _field(tool_call, "function")
                if function is None:
                    _reject_tool_history(f"{path}.function must be an object")
                name = _field(function, "name")
                if not isinstance(name, str) or not name.strip():
                    _reject_tool_history(
                        f"{path}.function.name must be a non-empty string"
                    )
                arguments = _field(function, "arguments")
                if not isinstance(arguments, str) or not arguments.strip():
                    _reject_tool_history(
                        f"{path}.function.arguments must be a non-empty JSON object string"
                    )
                try:
                    decoded_arguments = json.loads(arguments)
                except (TypeError, ValueError) as error:
                    _reject_tool_history(
                        f"{path}.function.arguments must be valid JSON: {error}"
                    )
                if not isinstance(decoded_arguments, dict):
                    _reject_tool_history(
                        f"{path}.function.arguments must decode to a JSON object"
                    )

                seen_call_ids.add(call_id)
                pending_call_ids.add(call_id)

        if role == "tool":
            tool_call_id = _field(message, "tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                _reject_tool_history(
                    f"messages[{message_index}].tool_call_id must be a non-empty string"
                )
            if not allow_partial and tool_call_id not in pending_call_ids:
                _reject_tool_history(
                    f"messages[{message_index}].tool_call_id {tool_call_id!r} "
                    "does not match a pending assistant tool call"
                )
            pending_call_ids.discard(tool_call_id)

    if not allow_partial and pending_call_ids:
        pending = ", ".join(sorted(pending_call_ids))
        _reject_tool_history(f"missing tool results for: {pending}")


def apply_kimi_k3_request_contract(
    config: _GenerationConfig,
    *,
    specified_fields: AbstractSet[str],
    thinking: bool,
) -> None:
    """Apply Kimi K3 generation defaults and validate explicitly supplied values.

    ``specified_fields`` contains canonical, non-null request field names.  Keeping
    presence separate from ``GenerateConfig`` matters because each transport fills
    its own generic defaults before the model-specific contract runs.
    """

    if "temperature" not in specified_fields:
        config.temperature = 1.0 if thinking else 0.6
    else:
        temperature = _scalar_float("temperature", config.temperature)
        if not 0.0 <= temperature <= 1.0:
            _reject("temperature", config.temperature, "in [0, 1]")
        config.temperature = temperature

    if "top_p" not in specified_fields:
        config.top_p = 0.95
    else:
        top_p = _scalar_float("top_p", config.top_p)
        if isclose(top_p, 0.95, rel_tol=0.0, abs_tol=_TOP_P_ABS_TOL):
            # DashSc commonly carries this value as FP32. Canonicalize its
            # round-off so both entry points produce the same GenerateConfig.
            config.top_p = 0.95
        elif isclose(top_p, 1.0, rel_tol=0.0, abs_tol=_TOP_P_ABS_TOL):
            config.top_p = 1.0
        else:
            _reject("top_p", config.top_p, "to be 0.95 or 1.0")

    for name in ("presence_penalty", "frequency_penalty"):
        value = _scalar_float(name, getattr(config, name))
        if value != 0.0:
            _reject(name, value, "to be 0")
        setattr(config, name, 0.0)

    if config.num_return_sequences not in (0, 1):
        _reject_n(config.num_return_sequences)
    if "n" in specified_fields and config.num_return_sequences != 1:
        _reject_n(config.num_return_sequences)

    config.in_think_mode = bool(thinking)
    if not thinking:
        config.max_thinking_tokens = 0

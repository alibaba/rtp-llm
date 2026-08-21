"""Smoke comparer for DashSc gRPC ModelStreamInfer."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import struct
import sys
import time
import inspect
from hashlib import sha1
from typing import Any, List, Optional, Tuple, Union

import grpc
from pydantic import BaseModel
from smoke.common_def import QueryStatus, SmokeException
from smoke.normal_comparer import AuxInfo, NormalComparer
from smoke.utils import no_compare, save_response

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import ServerConfig
from rtp_llm.dash_sc import (
    SamplingParams,
    build_model_infer_request,
    dash_sc_grpc_client_channel_options,
    decode_finish_reason,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory


DASH_SC_GRPC_ENDPOINT = "/__dash_sc_grpc__"
PROMPT_MODE_RAW = "raw"
PROMPT_MODE_DSV4_CHAT = "dsv4_chat"
DSV4_THINKING_MODE_AUTO = "auto"
DSV4_THINKING_MODE_CHAT = "chat"
DSV4_THINKING_MODE_THINKING = "thinking"
_DSV4_THINKING_MODES = {
    DSV4_THINKING_MODE_AUTO,
    DSV4_THINKING_MODE_CHAT,
    DSV4_THINKING_MODE_THINKING,
}


def _model_type_key(model_type: str) -> str:
    return str(model_type or "").replace("-", "_").lower()


def _default_prompt_mode_for_model(model_type: str) -> str:
    if _model_type_key(model_type) == "deepseek_v4":
        return PROMPT_MODE_DSV4_CHAT
    return PROMPT_MODE_RAW


def _tokenizer_encode_to_int_list(tokenizer: Any, prompt: str) -> List[int]:
    input_ids = tokenizer.encode(prompt)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    return [int(x) for x in input_ids]


def _load_dsv4_encoding_module(ckpt_path: str) -> Any:
    encoding_script_path = os.path.abspath(
        os.path.join(ckpt_path.rstrip(os.path.sep), "encoding", "encoding_dsv4.py")
    )
    if not os.path.exists(encoding_script_path):
        raise FileNotFoundError(
            f"DeepSeek-V4 encoding script not found: {encoding_script_path}"
        )
    module_hash = sha1(encoding_script_path.encode("utf-8")).hexdigest()[:12]
    module_name = f"dash_sc_smoke_encoding_dsv4_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, encoding_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {encoding_script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _parse_chat_messages_json(messages_json: str) -> List[dict[str, Any]]:
    try:
        parsed = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"chat_messages_json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("chat_messages_json must be a non-empty JSON list")
    messages: List[dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"chat_messages_json[{idx}] must be an object")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"chat_messages_json[{idx}].role must be a string")
        if "content" not in item:
            raise ValueError(f"chat_messages_json[{idx}].content is required")
        messages.append(dict(item))
    return messages


def _attach_tools_to_system_message(
    messages: List[dict[str, Any]], tools: Optional[List[dict[str, Any]]]
) -> None:
    if not tools:
        return
    tools_data = copy.deepcopy(tools)
    for msg in messages:
        if msg.get("role") == "system":
            msg["tools"] = tools_data
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools_data})


def _resolve_dsv4_thinking_mode(
    requested_mode: str, enable_thinking: Optional[bool]
) -> str:
    if requested_mode not in _DSV4_THINKING_MODES:
        raise ValueError(f"unsupported dsv4_thinking_mode: {requested_mode!r}")
    if requested_mode == DSV4_THINKING_MODE_AUTO:
        return (
            DSV4_THINKING_MODE_THINKING
            if enable_thinking is True
            else DSV4_THINKING_MODE_CHAT
        )
    if requested_mode == DSV4_THINKING_MODE_THINKING and enable_thinking is not True:
        raise ValueError("dsv4_thinking_mode=thinking requires enable_thinking=true")
    if requested_mode == DSV4_THINKING_MODE_CHAT and enable_thinking is True:
        raise ValueError("dsv4_thinking_mode=chat conflicts with enable_thinking=true")
    return requested_mode


def _encode_prompt_for_smoke(
    *,
    tokenizer: Any,
    ckpt_path: str,
    model_type: str,
    prompt: str,
    prompt_mode: Optional[str],
    chat_messages_json: str,
    enable_thinking: Optional[bool],
    dsv4_thinking_mode: str,
    tools: Optional[List[dict[str, Any]]],
    tool_choice: Optional[Any],
) -> List[int]:
    prompt_mode = prompt_mode or _default_prompt_mode_for_model(model_type)
    if prompt_mode == PROMPT_MODE_RAW:
        if chat_messages_json.strip():
            raise ValueError("chat_messages_json requires prompt_mode=dsv4_chat")
        if tools:
            raise ValueError("tools requires prompt_mode=dsv4_chat")
        if tool_choice is not None:
            raise ValueError("tool_choice is not supported by dash-sc gRPC smoke")
        return _tokenizer_encode_to_int_list(tokenizer, prompt)
    if prompt_mode != PROMPT_MODE_DSV4_CHAT:
        raise ValueError(f"unsupported prompt_mode: {prompt_mode!r}")
    if _model_type_key(model_type) != "deepseek_v4":
        raise ValueError("prompt_mode=dsv4_chat requires model_type deepseek_v4")
    if tool_choice is not None:
        raise ValueError("tool_choice is not supported by dash-sc gRPC smoke")

    messages = (
        _parse_chat_messages_json(chat_messages_json)
        if chat_messages_json.strip()
        else [{"role": "user", "content": prompt}]
    )
    _attach_tools_to_system_message(messages, tools)
    thinking_mode = _resolve_dsv4_thinking_mode(
        dsv4_thinking_mode, enable_thinking
    )
    rendered_prompt = _load_dsv4_encoding_module(ckpt_path).encode_messages(
        messages,
        thinking_mode=thinking_mode,
        drop_thinking=True,
        add_default_bos_token=True,
    )
    return _tokenizer_encode_to_int_list(tokenizer, rendered_prompt)


class DashScGrpcQueryInfo(BaseModel):
    prompt: Optional[Union[List[Any], str]] = None
    input_ids: Optional[Union[List[int], str]] = None
    generate_config: GenerateConfig = GenerateConfig()
    prompt_mode: Optional[str] = None
    chat_messages_json: str = ""
    tools: Optional[List[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    tool_call_structural_tag: Optional[Any] = None
    structural_tag: Optional[Any] = None
    dsv4_thinking_mode: str = DSV4_THINKING_MODE_AUTO
    return_input_ids: bool = False
    enable_thinking: Optional[bool] = None
    max_new_think_tokens: Optional[int] = None
    input_ints: Optional[dict[str, int]] = None
    parameter_ints: Optional[dict[str, int]] = None
    parameter_strings: Optional[dict[str, Any]] = None
    ds_header_attributes: Optional[dict[str, Any]] = None
    grpc_timeout_seconds: Optional[float] = None
    stop_when_requirements_met: bool = True
    yield_generator: bool = True


class DashScGrpcResponse(BaseModel):
    response: str = ""
    generated_ids: List[int] = []
    output_ids: List[int] = []
    reasoning_ids: List[int] = []
    content_ids: List[int] = []
    parameters: dict[str, Any] = {}
    reasoning_content: str = ""
    content: str = ""
    aux_info: Optional[AuxInfo] = None
    finish_reason: Optional[int] = None


class DashScGrpcExpected(BaseModel):
    response: Optional[str] = None
    generated_ids: Optional[List[int]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    generated_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    reasoning_ids: Optional[List[int]] = None
    content_ids: Optional[List[int]] = None
    content_startswith: Optional[str] = None
    response_contains: Optional[List[str]] = None
    content_contains: Optional[List[str]] = None
    content_not_contains: Optional[List[str]] = None
    reasoning_contains: Optional[List[str]] = None
    reasoning_not_contains: Optional[List[str]] = None
    required_parameters: Optional[List[str]] = None
    forbidden_parameters: Optional[List[str]] = None
    parameters_equal: Optional[dict[str, Any]] = None
    max_generated_id_occurrences: Optional[dict[str, int]] = None
    generated_ids_prefix: Optional[List[int]] = None
    generated_ids_contains: Optional[List[List[int]]] = None
    min_generated_ids: Optional[int] = None
    min_tokens_after_last_required_subsequence: Optional[int] = None
    aux_info: Optional[AuxInfo] = None
    finish_reason: Optional[int] = None
    min_content_chars: Optional[int] = None
    expected_error_message_contains: Optional[str] = None
    expected_status_code: Optional[int] = None
    json_content: bool = False
    json_object: bool = False
    required_json_keys: Optional[List[str]] = None
    expected_json: Optional[Any] = None


def _scalar_int(v: Any) -> int:
    if isinstance(v, list):
        return int(v[0]) if v else 0
    return int(v)


def _scalar_float(v: Any) -> float:
    if isinstance(v, list):
        return float(v[0]) if v else 1.0
    return float(v)


def _generate_config_to_sampling(
    gc: GenerateConfig,
    max_new_think_tokens: Optional[int],
    structural_tag: Optional[Any] = None,
) -> SamplingParams:
    stop_words = gc.stop_words_list or []
    stop_tuples = tuple(tuple(group) for group in stop_words) if stop_words else tuple()
    seed = gc.random_seed
    if isinstance(seed, list):
        seed = int(seed[0]) if seed else None
    elif seed is not None:
        seed = int(seed)
    kwargs = dict(
        max_new_tokens=_scalar_int(gc.max_new_tokens),
        num_return_sequences=_scalar_int(gc.num_return_sequences or 0),
        top_p=_scalar_float(gc.top_p),
        top_k=_scalar_int(gc.top_k),
        temperature=_scalar_float(gc.temperature),
        min_new_tokens=_scalar_int(gc.min_new_tokens),
        random_seed=seed,
        repetition_penalty=_scalar_float(gc.repetition_penalty),
        frequency_penalty=_scalar_float(gc.frequency_penalty),
        presence_penalty=_scalar_float(gc.presence_penalty),
        stop_words_list=stop_tuples,
    )
    sampling_params = inspect.signature(SamplingParams).parameters
    if "max_new_think_tokens" in sampling_params:
        kwargs["max_new_think_tokens"] = max_new_think_tokens
    if "response_format" in sampling_params:
        response_format = gc.response_format
        if response_format is not None and not isinstance(response_format, str):
            response_format = json.dumps(
                response_format, ensure_ascii=False, separators=(",", ":")
            )
        kwargs["response_format"] = response_format
    if "json_format" in sampling_params:
        kwargs["json_format"] = bool(gc.json_format)
    if "structural_tag" in inspect.signature(SamplingParams).parameters:
        kwargs["structural_tag"] = _jsonable_to_string(
            structural_tag if structural_tag is not None else gc.structural_tag
        )
    return SamplingParams(**kwargs)


def _jsonable_to_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value).strip()
    return text or None


def _build_model_infer_request_compat(**kwargs: Any) -> Any:
    params = inspect.signature(build_model_infer_request).parameters
    if "enable_thinking" not in params:
        kwargs.pop("enable_thinking", None)
    return build_model_infer_request(**kwargs)


def _append_int32_scalar_input(request: Any, name: str, value: int) -> None:
    inp = request.inputs.add()
    inp.name = name
    inp.datatype = "INT32"
    inp.shape.append(1)
    request.raw_input_contents.append(struct.pack("<i", int(value)))


def _resolve_dash_sc_grpc_port(server_manager) -> int:
    server_config = ServerConfig()
    server_config.start_port = int(server_manager.port)
    server_config.rank_id = 0
    return server_config.dash_sc_grpc_server_port


def _parse_infer_chunk(
    infer: Any,
) -> Tuple[List[int], Optional[int], Optional[int], Optional[int]]:
    generated: List[int] = []
    finish: Optional[int] = None
    prompt_token_num: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None
    for i, out in enumerate(infer.outputs):
        if i >= len(infer.raw_output_contents):
            break
        raw = infer.raw_output_contents[i]
        if out.name == "generated_ids" and out.datatype == "INT32":
            n = len(raw) // 4
            if n > 0 and list(out.shape)[-1:] != [0]:
                generated = list(struct.unpack("<%di" % n, raw))
        elif out.name == "finish_reason":
            finish = decode_finish_reason(out, raw)
        elif out.name == "prompt_token_num" and out.datatype == "INT32" and len(raw) >= 4:
            prompt_token_num = struct.unpack("<i", raw[:4])[0]
        elif (
            out.name == "prompt_cached_token_num"
            and out.datatype == "INT32"
            and len(raw) >= 4
        ):
            prompt_cached_token_num = struct.unpack("<i", raw[:4])[0]
    return generated, finish, prompt_token_num, prompt_cached_token_num


def _infer_param_value(param: Any) -> Any:
    choice = param.WhichOneof("parameter_choice")
    if choice is None:
        return None
    return getattr(param, choice)


def _parse_infer_parameters(infer: Any) -> dict[str, Any]:
    return {name: _infer_param_value(param) for name, param in infer.parameters.items()}


def _dashllm_message_header(parameters: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw = parameters.get("__messages__")
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    header = payload.get("header")
    return header if isinstance(header, dict) else None


def _find_subsequence(values: List[int], pattern: List[int]) -> int:
    if not pattern or len(values) < len(pattern):
        return -1
    n = len(pattern)
    for i in range(len(values) - n + 1):
        if values[i : i + n] == pattern:
            return i
    return -1


def _as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        result: set[int] = set()
        for item in value:
            result.update(_as_int_set(item))
        return result
    if isinstance(value, str):
        try:
            return _as_int_set(json.loads(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {int(value)}
    return {int(value)}


def _strip_trailing_token_ids(token_ids: List[int], strip_ids: set[int]) -> List[int]:
    if not strip_ids:
        return token_ids
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] in strip_ids:
        end -= 1
    return token_ids[:end]


class DashScGrpcComparer(NormalComparer):
    """Drive smoke traffic through DashSc gRPC instead of HTTP."""

    def format_query(self, query_json: dict[str, Any]) -> DashScGrpcQueryInfo:
        return DashScGrpcQueryInfo(**query_json)

    def format_result(self, result_json: dict[str, Any]) -> DashScGrpcExpected:
        return DashScGrpcExpected(**result_json)

    def _input_ids_from_query(self, query_info: DashScGrpcQueryInfo) -> List[int]:
        if query_info.input_ids is not None:
            if isinstance(query_info.input_ids, str):
                return [int(x) for x in json.loads(query_info.input_ids)]
            return [int(x) for x in query_info.input_ids]

        prompt = query_info.prompt
        messages_json = query_info.chat_messages_json
        if prompt is None and not messages_json.strip():
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: query must include prompt or input_ids",
            )
        if isinstance(prompt, list) and messages_json.strip():
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: provide either prompt messages or chat_messages_json, not both",
            )
        if isinstance(prompt, list):
            messages_json = json.dumps(prompt, ensure_ascii=False)
            prompt_text = ""
        else:
            prompt_text = prompt or ""
        ckpt = self.qr_info.get("_model_path") or self.qr_info.get("model_path")
        tok_path = ckpt
        model_type = self.qr_info.get("_model_type") or self.qr_info.get("model_type")
        tokenizer = TokenizerFactory.create(ckpt, tok_path or ckpt, model_type)
        return _encode_prompt_for_smoke(
            tokenizer=tokenizer,
            ckpt_path=ckpt,
            model_type=model_type,
            prompt=prompt_text,
            prompt_mode=query_info.prompt_mode,
            chat_messages_json=messages_json,
            tools=query_info.tools,
            tool_choice=query_info.tool_choice,
            enable_thinking=query_info.enable_thinking,
            dsv4_thinking_mode=query_info.dsv4_thinking_mode,
        )

    def _decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        ckpt = self.qr_info.get("_model_path") or self.qr_info.get("model_path")
        tok_path = ckpt
        model_type = self.qr_info.get("_model_type") or self.qr_info.get("model_type")
        tokenizer = TokenizerFactory.create(ckpt, tok_path or ckpt, model_type)
        return tokenizer.decode(token_ids)

    def _decode_dashscope_parts(
        self, generated_ids: List[int], parameters: dict[str, Any]
    ) -> Tuple[List[int], List[int], str, str]:
        stop_token_ids = _as_int_set(parameters.get("stop_token_id"))
        think_token_num = parameters.get("generate_think_token_num")
        if think_token_num is None:
            content_ids = _strip_trailing_token_ids(generated_ids, stop_token_ids)
            return [], content_ids, "", self._decode(content_ids)
        split_at = max(0, min(int(think_token_num), len(generated_ids)))
        reasoning_ids = generated_ids[:split_at]
        reasoning = self._decode(reasoning_ids)
        content_ids = _strip_trailing_token_ids(
            generated_ids[split_at:], stop_token_ids
        )
        content = self._decode(content_ids)
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>") :]
            if reasoning.startswith("\n"):
                reasoning = reasoning[1:]
        reasoning = reasoning.rstrip("\n")
        if reasoning.endswith("</think>"):
            reasoning = reasoning[: -len("</think>")]
            if reasoning.endswith("\n\n"):
                reasoning = reasoning[:-2]
            elif reasoning.endswith("\n"):
                reasoning = reasoning[:-1]
        return reasoning_ids, content_ids, reasoning, content

    @staticmethod
    def _requirements_met(
        expected: DashScGrpcExpected, generated_ids: List[int]
    ) -> bool:
        if expected.generated_ids is not None:
            expected_ids = [int(x) for x in expected.generated_ids]
            prefix_len = min(len(generated_ids), len(expected_ids))
            # Exact generated_ids checks may fail fast on mismatch, but must not
            # stop on a matching prefix; otherwise extra generated tokens would
            # be hidden by early-stop and the check would degrade to prefix match.
            return generated_ids[:prefix_len] != expected_ids[:prefix_len]
        if expected.min_generated_ids is not None and (
            len(generated_ids) < expected.min_generated_ids
        ):
            return False
        contains = expected.generated_ids_contains or []
        last_end = -1
        for pattern in contains:
            idx = _find_subsequence(generated_ids, [int(x) for x in pattern])
            if idx < 0:
                return False
            last_end = max(last_end, idx + len(pattern))
        min_after = expected.min_tokens_after_last_required_subsequence
        if min_after is not None and contains:
            if len(generated_ids) - last_end < min_after:
                return False
        return True

    def compare_result(
        self, expect: DashScGrpcExpected, actual: DashScGrpcResponse
    ) -> None:
        diffs: List[str] = []
        if expect.response is not None and expect.response != actual.response:
            diffs.append(
                f"response:\n    expect: {expect.response}\n    actual: {actual.response}"
            )
        if expect.generated_ids is not None:
            expected_ids = [int(x) for x in expect.generated_ids]
            if actual.generated_ids != expected_ids:
                first_mismatch = None
                for idx, (actual_id, expected_id) in enumerate(
                    zip(actual.generated_ids, expected_ids)
                ):
                    if actual_id != expected_id:
                        first_mismatch = (
                            f" first_mismatch_index={idx}"
                            f" expect={expected_id} actual={actual_id}"
                        )
                        break
                if first_mismatch is None and len(actual.generated_ids) != len(
                    expected_ids
                ):
                    first_mismatch = (
                        f" length_mismatch expect_len={len(expected_ids)}"
                        f" actual_len={len(actual.generated_ids)}"
                    )
                first_mismatch = first_mismatch or ""
                diffs.append(
                    "generated_ids:\n"
                    f"    expect_len={len(expected_ids)} actual_len={len(actual.generated_ids)}"
                    f"{first_mismatch}\n"
                    f"    expect: {expected_ids}\n"
                    f"    actual: {actual.generated_ids}"
                )
        if expect.content is not None and expect.content != actual.content:
            diffs.append(
                f"content:\n    expect: {expect.content}\n    actual: {actual.content}"
            )
        if (
            expect.reasoning_content is not None
            and expect.reasoning_content != actual.reasoning_content
        ):
            diffs.append(
                "reasoning_content:\n"
                f"    expect: {expect.reasoning_content}\n"
                f"    actual: {actual.reasoning_content}"
            )
        expected_output_ids = expect.generated_ids
        if expected_output_ids is None:
            expected_output_ids = expect.output_ids
        if expected_output_ids is not None and expected_output_ids != actual.generated_ids:
            diffs.append(
                f"output_ids:\n    expect: {expected_output_ids}\n    actual: {actual.generated_ids}"
            )
        if expect.reasoning_ids is not None and expect.reasoning_ids != actual.reasoning_ids:
            diffs.append(
                f"reasoning_ids:\n    expect: {expect.reasoning_ids}\n    actual: {actual.reasoning_ids}"
            )
        if expect.content_ids is not None and expect.content_ids != actual.content_ids:
            diffs.append(
                f"content_ids:\n    expect: {expect.content_ids}\n    actual: {actual.content_ids}"
            )
        if expect.content_startswith is not None and not actual.content.startswith(
            expect.content_startswith
        ):
            diffs.append(
                "content prefix:\n"
                f"    expect: {expect.content_startswith}\n"
                f"    actual: {actual.content[:len(expect.content_startswith)]}"
            )
        for text in expect.response_contains or []:
            if text not in actual.response:
                diffs.append(f"response missing substring {text!r}: {actual.response!r}")
        for text in expect.content_contains or []:
            if text not in actual.content:
                diffs.append(f"content missing substring {text!r}: {actual.content!r}")
        for text in expect.content_not_contains or []:
            if text in actual.content:
                diffs.append(
                    f"content contains forbidden substring {text!r}: {actual.content!r}"
                )
        for text in expect.reasoning_contains or []:
            if text not in actual.reasoning_content:
                diffs.append(
                    f"reasoning missing substring {text!r}: {actual.reasoning_content!r}"
                )
        for text in expect.reasoning_not_contains or []:
            if text in actual.reasoning_content:
                diffs.append(
                    f"reasoning contains forbidden substring {text!r}: {actual.reasoning_content!r}"
                )
        for name in expect.required_parameters or []:
            if name not in actual.parameters:
                diffs.append(
                    f"response parameter {name!r} missing; actual parameters: {actual.parameters}"
                )
        for name in expect.forbidden_parameters or []:
            if name in actual.parameters:
                diffs.append(
                    f"response parameter {name!r} should be absent; actual parameters: {actual.parameters}"
                )
        for name, value in (expect.parameters_equal or {}).items():
            if actual.parameters.get(name) != value:
                diffs.append(
                    f"response parameter {name!r}: expect {value!r}, actual {actual.parameters.get(name)!r}"
                )
        for token_id_text, max_count in (
            expect.max_generated_id_occurrences or {}
        ).items():
            token_id = int(token_id_text)
            actual_count = actual.generated_ids.count(token_id)
            if actual_count > int(max_count):
                diffs.append(
                    f"generated id {token_id} occurs {actual_count} times, expect <= {max_count}"
                )
        if expect.min_generated_ids is not None and (
            len(actual.generated_ids) < expect.min_generated_ids
        ):
            diffs.append(
                f"generated_ids length {len(actual.generated_ids)} < {expect.min_generated_ids}"
            )
        if expect.generated_ids_prefix is not None:
            prefix = [int(x) for x in expect.generated_ids_prefix]
            if actual.generated_ids[: len(prefix)] != prefix:
                diffs.append(
                    f"generated_ids prefix:\n    expect: {prefix}\n    actual: {actual.generated_ids[:len(prefix)]}"
                )
        contains = expect.generated_ids_contains or []
        last_end = -1
        for pattern in contains:
            pattern = [int(x) for x in pattern]
            idx = _find_subsequence(actual.generated_ids, pattern)
            if idx < 0:
                diffs.append(f"generated_ids missing subsequence {pattern}")
            else:
                last_end = max(last_end, idx + len(pattern))
        min_after = expect.min_tokens_after_last_required_subsequence
        if min_after is not None and contains and last_end >= 0:
            actual_after = len(actual.generated_ids) - last_end
            if actual_after < min_after:
                diffs.append(
                    "generated_ids tokens after last required subsequence: "
                    f"expect >= {min_after}, actual {actual_after}"
                )
        if expect.aux_info is not None:
            if actual.aux_info is None:
                diffs.append("aux_info missing")
            else:
                self._compare_aux_info(expect.aux_info, actual.aux_info, diffs)
        if expect.finish_reason is not None and actual.finish_reason != expect.finish_reason:
            diffs.append(
                f"finish_reason: expect {expect.finish_reason}, actual {actual.finish_reason}"
            )
        if expect.min_content_chars is not None and len(actual.content) < expect.min_content_chars:
            diffs.append(
                f"content length {len(actual.content)} < min_content_chars {expect.min_content_chars}"
            )
        if expect.expected_status_code is not None:
            header = _dashllm_message_header(actual.parameters)
            if header is None:
                diffs.append(
                    "DashLLM __messages__ header missing or invalid; "
                    f"actual parameters: {actual.parameters}"
                )
            else:
                if (
                    expect.expected_status_code is not None
                    and header.get("status_code") != expect.expected_status_code
                ):
                    diffs.append(
                        "status_code: "
                        f"expect {expect.expected_status_code}, "
                        f"actual {header.get('status_code')}"
                    )
        if (
            expect.json_content
            or expect.json_object
            or expect.required_json_keys
            or expect.expected_json is not None
        ):
            content = actual.content.strip()
            parsed_json: Any = None
            if not content:
                diffs.append("content JSON missing: actual content is empty")
            elif content.startswith("```"):
                diffs.append(f"content JSON must not be fenced: {actual.content!r}")
            else:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError as e:
                    diffs.append(
                        f"content is not valid JSON: {e}; actual content: {actual.content!r}"
                    )
            if parsed_json is not None:
                if expect.json_object and not isinstance(parsed_json, dict):
                    diffs.append(
                        f"content JSON is not an object: {type(parsed_json).__name__}"
                    )
                if expect.expected_json is not None and parsed_json != expect.expected_json:
                    diffs.append(
                        "content JSON not equal:\n"
                        f"    expect: {expect.expected_json}\n"
                        f"    actual: {parsed_json}"
                    )
                if isinstance(parsed_json, dict) and expect.required_json_keys:
                    missing_keys = [
                        key for key in expect.required_json_keys if key not in parsed_json
                    ]
                    if missing_keys:
                        diffs.append(
                            "content JSON missing required keys "
                            f"{missing_keys}; actual keys: {sorted(parsed_json.keys())}"
                        )
        if diffs:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                "\n".join(diffs)
                + f"\nactual generated tail: {actual.generated_ids[-32:]}",
            )

    def run(self) -> None:
        query_info = self.format_query(self.qr_info["query"])
        self.tracer.query = query_info

        ckpt = self.qr_info.get("_model_path") or self.qr_info.get("model_path")
        model_type = self.qr_info.get("_model_type") or self.qr_info.get("model_type")
        if not ckpt or not model_type:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                "dash_sc_grpc: missing _model_path / _model_type",
            )

        input_ids = self._input_ids_from_query(query_info)
        logging.info(
            "DashScGrpcComparer query_idx=%s input_len=%s",
            self.qr_info.get("_query_idx", 0),
            len(input_ids),
        )

        sampling = _generate_config_to_sampling(
            query_info.generate_config,
            query_info.max_new_think_tokens,
            (
                query_info.tool_call_structural_tag
                if query_info.tool_call_structural_tag is not None
                else query_info.structural_tag
            ),
        )
        return_input_ids = bool(
            query_info.return_input_ids
            or getattr(query_info.generate_config, "return_input_ids", False)
        )
        request = _build_model_infer_request_compat(
            request_id="smoke_dash_sc_grpc_%s" % self.qr_info.get("_query_idx", 0),
            model_name="default",
            input_ids=input_ids,
            sampling=sampling,
            return_input_ids=return_input_ids,
            enable_thinking=query_info.enable_thinking,
        )
        for name, value in (query_info.input_ints or {}).items():
            _append_int32_scalar_input(request, name, int(value))
        for name, value in (query_info.parameter_ints or {}).items():
            request.parameters[name].int64_param = int(value)
        for name, value in (query_info.parameter_strings or {}).items():
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            request.parameters[name].string_param = value
        if query_info.ds_header_attributes is not None:
            request.parameters["ds_header_attributes"].string_param = json.dumps(
                query_info.ds_header_attributes
            )

        grpc_port = _resolve_dash_sc_grpc_port(self.server_manager)
        grpc_addr = f"127.0.0.1:{grpc_port}"
        logging.info("DashScGrpcComparer grpc_addr=%s", grpc_addr)

        expected = self.format_result(self.qr_info["result"])
        accumulated: List[int] = []
        response_parameters: dict[str, Any] = {}
        last_prompt_token_num: Optional[int] = None
        last_prompt_cached: Optional[int] = None
        last_finish_reason: Optional[int] = None
        error_message: Optional[str] = None

        channel = grpc.insecure_channel(
            grpc_addr, options=dash_sc_grpc_client_channel_options()
        )
        try:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)

            def _req_iter():
                yield request

            response_iter = stub.ModelStreamInfer(
                _req_iter(), timeout=query_info.grpc_timeout_seconds
            )
            for resp in response_iter:
                if resp.error_message:
                    error_message = resp.error_message
                    break
                if not resp.HasField("infer_response"):
                    continue
                chunk_ids, chunk_finish, ptn, ptcn = _parse_infer_chunk(
                    resp.infer_response
                )
                response_parameters.update(_parse_infer_parameters(resp.infer_response))
                status_message = response_parameters.get("status_message")
                status_code = response_parameters.get("status_code")
                if status_message or (
                    status_code is not None and int(status_code) >= 400
                ):
                    error_message = str(status_message or status_code)
                    break
                if chunk_ids:
                    accumulated.extend(chunk_ids)
                if chunk_finish is not None:
                    last_finish_reason = chunk_finish
                if ptn is not None:
                    last_prompt_token_num = ptn
                if ptcn is not None:
                    last_prompt_cached = ptcn
                if (
                    query_info.stop_when_requirements_met
                    and self._requirements_met(expected, accumulated)
                ):
                    break
        finally:
            channel.close()

        if error_message:
            if expected.expected_error_message_contains is not None:
                if expected.expected_error_message_contains in error_message:
                    return
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"error_message mismatch: expect contains {expected.expected_error_message_contains!r}, "
                    f"actual: {error_message!r}",
                )
            raise SmokeException(
                QueryStatus.VISIT_FAILED,
                f"dash_sc_grpc error: {error_message}",
            )
        if expected.expected_error_message_contains is not None:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"expected error containing {expected.expected_error_message_contains!r} but request succeeded",
            )

        aux = None
        if last_prompt_token_num is not None or last_prompt_cached is not None:
            aux = AuxInfo(
                input_len=last_prompt_token_num,
                reuse_len=last_prompt_cached,
            )
        reasoning_ids, content_ids, reasoning_content, content = self._decode_dashscope_parts(
            accumulated, response_parameters
        )
        actual_result = DashScGrpcResponse(
            response=self._decode(accumulated),
            generated_ids=accumulated,
            output_ids=accumulated,
            reasoning_ids=reasoning_ids,
            content_ids=content_ids,
            parameters=response_parameters,
            reasoning_content=reasoning_content,
            content=content,
            aux_info=aux,
            finish_reason=last_finish_reason,
        )
        self.tracer.actual_result = actual_result
        self.tracer.expect_result = expected
        self._dump_actual_to_artifact(actual_result)

        test_with_sleep = bool(int(os.environ.get("TEST_WITH_SLEEP", 0)))
        if test_with_sleep:
            time.sleep(3600 * 100)
        if save_response():
            self.qr_info["result"] = actual_result.model_dump(exclude_defaults=True)
        if no_compare():
            return
        self.compare_result(expected, actual_result)

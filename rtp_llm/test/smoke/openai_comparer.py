import copy
import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import REL_PATH, QueryStatus, SmokeException
from smoke.concurrent_stress import _detect_repetition
from smoke.grammar_constraint_validator import validate_constraint
from smoke.utils import create_temporary_copy

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    UsageInfo,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo


class OpenaiComparer(BaseComparer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.is_stream = self.qr_info["query"].get("stream", False)

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        query_info = ChatCompletionRequest(**query_json)
        self._rewrite_query_info(query_info)
        return query_info

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        # Cases that only assert output health carry an empty expected result;
        # synthesize an empty response so the comparison can reach the health
        # checks instead of failing to parse the golden value.
        if not result_json and self.qr_info.get("compare_config", {}).get(
            "skip_choices", False
        ):
            usage = UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0)
            if self.is_stream:
                return ChatCompletionStreamResponse(choices=[], usage=usage)
            return ChatCompletionResponse(choices=[], usage=usage)
        if result_json.get("extra_outputs", None) is not None:
            path = result_json["extra_outputs"].get("all_hidden_states", None)
            if path is not None and isinstance(path, str):
                result_json["extra_outputs"]["all_hidden_states"] = (
                    torch.load(os.path.join(REL_PATH, path)).numpy().tolist()
                )
        if self.is_stream:
            return ChatCompletionStreamResponse(**result_json)
        else:
            return ChatCompletionResponse(**result_json)

    def curl_response_to_json(
        self, query_info: ChatCompletionRequest, curl_response: Any
    ) -> Dict[str, Any]:
        if self.is_stream:
            responses = list(filter(None, curl_response))
            choices = []
            usage = None
            for response in responses:
                response_json = json.loads(
                    response.decode("utf-8")[6:]
                )  # remove `data: `
                res_choices = response_json.get("choices")
                if choices == []:
                    choices = res_choices
                else:
                    assert len(choices) == len(res_choices)
                    for i in range(len(choices)):
                        # 同步修改了测试的逻辑
                        if choices[i]["delta"].get("content", None) == None:
                            choices[i]["delta"]["content"] = (
                                res_choices[i]["delta"].get("content", None) or None
                            )
                        else:
                            choices[i]["delta"]["content"] += (
                                res_choices[i]["delta"].get("content", None) or ""
                            )
                        if choices[i]["delta"].get("reasoning_content", None) == None:
                            choices[i]["delta"]["reasoning_content"] = (
                                res_choices[i]["delta"].get("reasoning_content", None)
                                or None
                            )
                        else:
                            choices[i]["delta"]["reasoning_content"] += (
                                res_choices[i]["delta"].get("reasoning_content", None)
                                or ""
                            )

                        choices[i]["delta"]["function_call"] = res_choices[i][
                            "delta"
                        ].get("function_call", None) or choices[i]["delta"].get(
                            "function_call", None
                        )
                        self._merge_tool_calls(choices, res_choices, i)

                        choices[i]["finish_reason"] = res_choices[i].get(
                            "finish_reason"
                        ) or choices[i].get("finish_reason")

                        if choices[i].get("logprobs", None) == None:
                            choices[i]["logprobs"] = res_choices[i].get(
                                "logprobs", None
                            )
                        else:
                            res_logprobs = res_choices[i].get("logprobs", None)
                            if res_logprobs:
                                choices[i]["logprobs"]["content"] += res_logprobs.get(
                                    "content", []
                                )

                usage = response_json.get("usage")
            return {"choices": choices, "usage": usage}
        else:
            # 确保非流式的情况下, choices中message中tool_calls的id的统一
            res = json.loads(curl_response)
            if res.get("choices", None) == None:
                return res
            for choice in res["choices"]:
                if choice.get("message", None) == None:
                    continue
                if choice["message"].get("tool_calls", None) == None:
                    continue
                for tool_call in choice["message"]["tool_calls"]:
                    tool_call["id"] = "call_" + "a" * 24
            return res

    def _merge_tool_calls(self, choices, res_choices, i):
        if not res_choices[i]["delta"].get("tool_calls"):
            return
        # 初始化当前choice的tool_calls
        if "tool_calls" not in choices[i]["delta"]:
            choices[i]["delta"]["tool_calls"] = []
        current_tool_calls = choices[i]["delta"]["tool_calls"]
        new_tool_calls = res_choices[i]["delta"]["tool_calls"]
        for new_tool_call in new_tool_calls:
            # 统一修改id格式
            if "id" in new_tool_call:
                new_tool_call["id"] = "call_" + "a" * 24
            tool_index = new_tool_call.get("index", 0)
            # 查找现有的tool_call
            existing_tool_call = None
            for existing in current_tool_calls:
                if existing.get("index") == tool_index:
                    existing_tool_call = existing
                    break
            if existing_tool_call is None:
                # 新的tool_call
                current_tool_calls.append(new_tool_call)
            else:
                # 合并arguments，保持name
                if "function" in new_tool_call and "function" in existing_tool_call:
                    if "arguments" in new_tool_call["function"]:
                        if "arguments" not in existing_tool_call["function"]:
                            existing_tool_call["function"]["arguments"] = ""
                        existing_tool_call["function"]["arguments"] += new_tool_call[
                            "function"
                        ]["arguments"]
                    # 确保name字段存在
                    if (
                        "name" not in existing_tool_call["function"]
                        and "name" in new_tool_call["function"]
                    ):
                        existing_tool_call["function"]["name"] = new_tool_call[
                            "function"
                        ]["name"]

    def extract_logprobs(self, choices):
        choices = copy.deepcopy(choices)
        logprobs = []
        for choice in choices:
            if choice.logprobs and choice.logprobs.content:
                for content in choice.logprobs.content:
                    logprobs.append(content.logprob)
                    content.logprob = 0
                    for logprob in content.top_logprobs:
                        logprobs.append(logprob.logprob)
                        logprob.logprob = 0
        return logprobs, choices

    def _to_json_safe(self, value: Any) -> Any:
        """Convert value to JSON-serializable form (e.g. BaseModel -> dict)."""
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [self._to_json_safe(x) for x in value]
        if isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items()}
        return value

    def _dump_value(self, value: Any) -> str:
        """Serialize value for diff output (JSON-serializable, handles BaseModel/list/dict)."""
        return json.dumps(self._to_json_safe(value), ensure_ascii=False, indent=2)

    def _format_expect_actual(self, title: str, expect: Any, actual: Any) -> str:
        """Format a single diff block with title and expect/actual for readability."""
        lines = [
            f"{title}:",
            "",
            "  expect:",
        ]
        for line in self._dump_value(expect).split("\n"):
            lines.append("    " + line)
        lines.append("")
        lines.append("  actual:")
        for line in self._dump_value(actual).split("\n"):
            lines.append("    " + line)
        return "\n".join(lines)

    def _format_all_diffs(self, diffs: List[str]) -> str:
        """Format collected diffs into a single error message (no early exit)."""
        if not diffs:
            return ""
        n = len(diffs)
        lines = [
            "",
            "=" * 60,
            f"  Compare failed: {n} difference(s) found",
            "=" * 60,
        ]
        for i, d in enumerate(diffs, 1):
            lines.append("")
            lines.append(f"  --- Diff {i}/{n} ---")
            lines.append("")
            for line in d.split("\n"):
                lines.append("    " + line if line.strip() else "")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _choice_content(
        self,
        result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        choice_index: int,
    ) -> Optional[str]:
        if choice_index < 0 or choice_index >= len(result.choices):
            return None
        choice = result.choices[choice_index]
        if isinstance(result, ChatCompletionStreamResponse):
            return choice.delta.content
        return choice.message.content

    def _compare_required_json_content(
        self,
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        compare_config: Dict[str, Any],
        diffs: List[str],
    ) -> None:
        should_check_json = any(
            [
                compare_config.get("require_json_object_content", False),
                compare_config.get("json_content", False),
                compare_config.get("json_object", False),
                compare_config.get("required_json_fields"),
                compare_config.get("required_json_keys"),
                "expected_json" in compare_config,
            ]
        )
        if not should_check_json:
            return

        choice_index = int(compare_config.get("json_choice_index", 0))
        content = self._choice_content(actual_result, choice_index)
        if not isinstance(content, str) or not content.strip():
            diffs.append(
                self._format_expect_actual(
                    "required JSON content missing",
                    "non-empty string",
                    content,
                )
            )
            return

        text = content.strip()
        if text.startswith("```"):
            diffs.append(
                self._format_expect_actual(
                    "JSON content must not be wrapped in Markdown fence",
                    "raw JSON object",
                    content,
                )
            )
            return

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            diffs.append(
                self._format_expect_actual(
                    f"content is not valid JSON: {e}",
                    "valid JSON object",
                    content,
                )
            )
            return

        if compare_config.get("json_object", True) and not isinstance(parsed, dict):
            diffs.append(
                self._format_expect_actual(
                    "content JSON is not an object",
                    "JSON object",
                    parsed,
                )
            )
            return

        if (
            "expected_json" in compare_config
            and parsed != compare_config["expected_json"]
        ):
            diffs.append(
                self._format_expect_actual(
                    "content JSON not equal",
                    compare_config["expected_json"],
                    parsed,
                )
            )

        required_fields = compare_config.get("required_json_fields", [])
        required_fields = required_fields or compare_config.get(
            "required_json_keys", []
        )
        if required_fields and not isinstance(parsed, dict):
            diffs.append(
                self._format_expect_actual(
                    "content JSON cannot check required fields on non-object",
                    required_fields,
                    parsed,
                )
            )
        elif required_fields:
            missing_fields = [field for field in required_fields if field not in parsed]
            if missing_fields:
                diffs.append(
                    self._format_expect_actual(
                        "content JSON missing required fields",
                        required_fields,
                        sorted(parsed.keys()),
                    )
                )

    def _validate_grammar_constraint(
        self, actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse]
    ) -> None:
        """grammar_constraint_only: each choice's content must satisfy response_format.

        Validates the constraint (regex / json_schema / structural_tag), not golden
        bytes. Raises SmokeException on any violation so the case fails loudly; the
        actual response is already dumped to smoke_actual/ before this runs.
        """
        if self.is_stream:
            return  # streaming smoke for grammar isn't used today; keep simple
        response_format = self.qr_info["query"].get("response_format")
        if not response_format:
            return
        try:
            for idx, choice in enumerate(actual_result.choices):
                validate_constraint(choice.message.content, response_format, idx)
        except Exception as e:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"[grammar_constraint_only] constraint check failed: {e}",
            ) from e

    def _choice_text(self, choice: Any) -> str:
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", "")
        else:
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", "") if delta is not None else ""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(self._to_json_safe(content), ensure_ascii=False)

    def _choice_generated_text(self, choice: Any) -> str:
        content = self._choice_text(choice)
        if content.strip():
            return content

        message = getattr(choice, "message", None)
        if message is not None:
            reasoning_content = getattr(message, "reasoning_content", "")
        else:
            delta = getattr(choice, "delta", None)
            reasoning_content = (
                getattr(delta, "reasoning_content", "") if delta is not None else ""
            )
        if reasoning_content is None:
            return ""
        if isinstance(reasoning_content, str):
            return reasoning_content
        return json.dumps(self._to_json_safe(reasoning_content), ensure_ascii=False)

    def _enum_value(self, value: Any) -> Any:
        return getattr(value, "value", value)

    def _validate_output_health(
        self,
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        compare_config: Dict[str, Any],
        diffs: List[str],
    ) -> None:
        health_config = compare_config.get("output_health_check")
        if not health_config:
            return
        if health_config is True:
            health_config = {}

        choices = getattr(actual_result, "choices", None) or []
        expected_choice_count = int(health_config.get("expected_choice_count", 1))
        if len(choices) != expected_choice_count:
            diffs.append(
                self._format_expect_actual(
                    "health choice count",
                    expected_choice_count,
                    len(choices),
                )
            )

        usage = getattr(actual_result, "usage", None)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            min_key = f"min_{field}"
            if min_key not in health_config:
                continue
            actual_value = getattr(usage, field, None) if usage is not None else None
            min_value = int(health_config[min_key])
            if actual_value is None or actual_value < min_value:
                diffs.append(
                    self._format_expect_actual(
                        f"health {field} >= {min_value}",
                        f">= {min_value}",
                        actual_value,
                    )
                )

        required_finish_reason = health_config.get("required_finish_reason")
        min_content_chars = int(health_config.get("min_content_chars", 1))
        min_unique_chars = int(health_config.get("min_unique_non_ws_chars", 0))
        repeat_window = int(health_config.get("repeat_window", 0))
        required_substrings = health_config.get("required_substrings", [])
        required_substrings_any = health_config.get("required_substrings_any", [])
        forbidden_substrings = health_config.get("forbidden_substrings", [])

        for idx, choice in enumerate(choices):
            if required_finish_reason is not None:
                actual_reason = self._enum_value(getattr(choice, "finish_reason", None))
                if actual_reason != required_finish_reason:
                    diffs.append(
                        self._format_expect_actual(
                            f"health choices[{idx}].finish_reason",
                            required_finish_reason,
                            actual_reason,
                        )
                    )

            content = self._choice_generated_text(choice)
            stripped = content.strip()
            if len(stripped) < min_content_chars:
                diffs.append(
                    self._format_expect_actual(
                        f"health choices[{idx}].content length",
                        f">= {min_content_chars}",
                        len(stripped),
                    )
                )

            control_chars = [
                c for c in content if ord(c) < 32 and c not in ("\n", "\r", "\t")
            ]
            if control_chars:
                diffs.append(
                    self._format_expect_actual(
                        f"health choices[{idx}].content control chars",
                        "none",
                        [ord(c) for c in control_chars[:20]],
                    )
                )

            if min_unique_chars > 0:
                unique_chars = {c for c in content if not c.isspace()}
                if len(unique_chars) < min_unique_chars:
                    diffs.append(
                        self._format_expect_actual(
                            f"health choices[{idx}].content unique chars",
                            f">= {min_unique_chars}",
                            len(unique_chars),
                        )
                    )

            if repeat_window > 0:
                repeated = _detect_repetition(content, repeat_window)
                if repeated is not None:
                    diffs.append(
                        self._format_expect_actual(
                            f"health choices[{idx}].content repetition",
                            "no repeated fragment",
                            repeated[:200],
                        )
                    )

            missing_required = [
                text for text in required_substrings if text not in content
            ]
            if missing_required:
                diffs.append(
                    self._format_expect_actual(
                        f"health choices[{idx}].content required substrings",
                        required_substrings,
                        {"missing": missing_required},
                    )
                )

            if required_substrings_any and not any(
                text in content for text in required_substrings_any
            ):
                diffs.append(
                    self._format_expect_actual(
                        f"health choices[{idx}].content required any substring",
                        required_substrings_any,
                        "none found",
                    )
                )

            present_forbidden = [
                text for text in forbidden_substrings if text and text in content
            ]
            if present_forbidden:
                diffs.append(
                    self._format_expect_actual(
                        f"health choices[{idx}].content forbidden substrings",
                        "none",
                        present_forbidden,
                    )
                )

    def _validate_aux_info_health(
        self,
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        compare_config: Dict[str, Any],
        diffs: List[str],
    ) -> None:
        aux_config = compare_config.get("aux_info_health_check")
        if not aux_config:
            return
        if aux_config is True:
            aux_config = {}

        aux_info = getattr(actual_result, "aux_info", None)
        if aux_info is None:
            diffs.append(self._format_expect_actual("aux_info health", aux_config, None))
            return

        for field in (
            "reuse_len",
            "local_reuse_len",
            "remote_reuse_len",
            "memory_reuse_len",
            "prefill_total_reuse_len",
            "prefill_local_reuse_len",
            "prefill_remote_reuse_len",
            "prefill_memory_reuse_len",
            "decode_total_reuse_len",
            "decode_local_reuse_len",
            "decode_remote_reuse_len",
            "decode_memory_reuse_len",
        ):
            actual_value = getattr(aux_info, field, None)
            min_key = f"min_{field}"
            if min_key in aux_config:
                min_value = int(aux_config[min_key])
                if actual_value is None or actual_value < min_value:
                    diffs.append(
                        self._format_expect_actual(
                            f"aux_info.{field} >= {min_value}",
                            f">= {min_value}",
                            actual_value,
                        )
                    )
            max_key = f"max_{field}"
            if max_key in aux_config:
                max_value = int(aux_config[max_key])
                if actual_value is None or actual_value > max_value:
                    diffs.append(
                        self._format_expect_actual(
                            f"aux_info.{field} <= {max_value}",
                            f"<= {max_value}",
                            actual_value,
                        )
                    )

        output_len = getattr(aux_info, "output_len", None)
        iter_count = getattr(aux_info, "iter_count", None)

        min_output_len = aux_config.get("min_output_len")
        if min_output_len is not None and (
            output_len is None or output_len < int(min_output_len)
        ):
            diffs.append(
                self._format_expect_actual(
                    f"aux_info.output_len >= {min_output_len}",
                    f">= {min_output_len}",
                    output_len,
                )
            )

        max_output_len = aux_config.get("max_output_len")
        if max_output_len is not None and (
            output_len is None or output_len > int(max_output_len)
        ):
            diffs.append(
                self._format_expect_actual(
                    f"aux_info.output_len <= {max_output_len}",
                    f"<= {max_output_len}",
                    output_len,
                )
            )

        max_iter_count = aux_config.get("max_iter_count")
        if max_iter_count is not None and (
            iter_count is None or iter_count > int(max_iter_count)
        ):
            diffs.append(
                self._format_expect_actual(
                    f"aux_info.iter_count <= {max_iter_count}",
                    f"<= {max_iter_count}",
                    iter_count,
                )
            )

        max_iter_count_ratio = aux_config.get("max_iter_count_ratio")
        if max_iter_count_ratio is not None:
            if output_len in (None, 0) or iter_count is None:
                diffs.append(
                    self._format_expect_actual(
                        "aux_info.iter_count/output_len",
                        f"<= {max_iter_count_ratio}",
                        {"iter_count": iter_count, "output_len": output_len},
                    )
                )
            else:
                ratio = float(iter_count) / float(output_len)
                if ratio > float(max_iter_count_ratio):
                    diffs.append(
                        self._format_expect_actual(
                            "aux_info.iter_count/output_len",
                            f"<= {max_iter_count_ratio}",
                            ratio,
                        )
                    )

        min_tokens_per_iter = aux_config.get("min_tokens_per_iter")
        if min_tokens_per_iter is not None:
            if iter_count in (None, 0) or output_len is None:
                diffs.append(
                    self._format_expect_actual(
                        "aux_info.output_len/iter_count",
                        f">= {min_tokens_per_iter}",
                        {"output_len": output_len, "iter_count": iter_count},
                    )
                )
            else:
                tokens_per_iter = float(output_len) / float(iter_count)
                if tokens_per_iter < float(min_tokens_per_iter):
                    diffs.append(
                        self._format_expect_actual(
                            "aux_info.output_len/iter_count",
                            f">= {min_tokens_per_iter}",
                            tokens_per_iter,
                        )
                    )

        max_tokens_per_iter = aux_config.get("max_tokens_per_iter")
        if max_tokens_per_iter is not None:
            if iter_count in (None, 0) or output_len is None:
                diffs.append(
                    self._format_expect_actual(
                        "aux_info.output_len/iter_count",
                        f"<= {max_tokens_per_iter}",
                        {"output_len": output_len, "iter_count": iter_count},
                    )
                )
            else:
                tokens_per_iter = float(output_len) / float(iter_count)
                if tokens_per_iter > float(max_tokens_per_iter):
                    diffs.append(
                        self._format_expect_actual(
                            "aux_info.output_len/iter_count",
                            f"<= {max_tokens_per_iter}",
                            tokens_per_iter,
                        )
                    )

        required_pd_sep = aux_config.get("required_pd_sep")
        if required_pd_sep is not None:
            actual_pd_sep = getattr(aux_info, "pd_sep", None)
            if actual_pd_sep != bool(required_pd_sep):
                diffs.append(
                    self._format_expect_actual(
                        "aux_info.pd_sep",
                        bool(required_pd_sep),
                        actual_pd_sep,
                    )
                )

    def compare_result(
        self,
        expect_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
    ) -> None:
        if self.qr_info.get("grammar_constraint_only"):
            self._validate_grammar_constraint(actual_result)
            return

        diffs: List[str] = []
        compare_config = self.qr_info.get("compare_config", {})
        skip_choices = compare_config.get("skip_choices", False)
        skip_usage = compare_config.get("skip_usage", False)

        if type(expect_result) != type(actual_result):
            diffs.append(
                "type not equal:\n  expect: "
                + type(expect_result).__name__
                + "\n  actual: "
                + type(actual_result).__name__
            )

        self._validate_output_health(actual_result, compare_config, diffs)
        self._validate_aux_info_health(actual_result, compare_config, diffs)

        if not skip_usage and expect_result.usage != actual_result.usage:
            diffs.append(
                self._format_expect_actual(
                    "usage not equal",
                    expect_result.usage.model_dump() if expect_result.usage else None,
                    actual_result.usage.model_dump() if actual_result.usage else None,
                )
            )

        # Skip aux_info comparison when expected auxinfo is null
        if expect_result.aux_info is not None:
            self._compare_aux_info(
                expect_result.aux_info,
                actual_result.aux_info,
                diffs,
            )

        required_aux_info = compare_config.get("required_aux_info", {})
        if required_aux_info:
            if actual_result.aux_info is None:
                diffs.append(
                    self._format_expect_actual(
                        "required aux_info missing",
                        required_aux_info,
                        None,
                    )
                )
            else:
                actual_required_aux = {
                    field: getattr(actual_result.aux_info, field, None)
                    for field in required_aux_info
                }
                if actual_required_aux != required_aux_info:
                    diffs.append(
                        self._format_expect_actual(
                            "required aux_info fields not equal",
                            required_aux_info,
                            actual_required_aux,
                        )
                    )

        expect_extra_outputs = copy.copy(expect_result.extra_outputs)
        actual_extra_outputs = copy.copy(actual_result.extra_outputs)
        if expect_extra_outputs is not None and actual_extra_outputs is not None:
            self._compare_extra_outputs(
                expect_extra_outputs,
                actual_extra_outputs,
                expect_result.extra_outputs,
                actual_result.extra_outputs,
                diffs,
            )
        elif expect_result.extra_outputs != actual_result.extra_outputs:
            diffs.append(
                self._format_expect_actual(
                    "extra_outputs not equal (one side None)",
                    expect_result.extra_outputs,
                    actual_result.extra_outputs,
                )
            )

        expect_logprobs, expect_choices = self.extract_logprobs(expect_result.choices)
        actual_logprobs, actual_choices = self.extract_logprobs(actual_result.choices)

        if not skip_choices and expect_choices != actual_choices:
            diffs.append(
                self._format_expect_actual(
                    "choices not equal (after normalizing logprobs)",
                    expect_choices,
                    actual_choices,
                )
            )

        rtol = atol = 1e-2
        if (
            not skip_choices
            and expect_logprobs is not None
            and actual_logprobs is not None
        ):
            if not all(
                torch.isclose(
                    torch.tensor(expect_logprobs),
                    torch.tensor(actual_logprobs),
                    rtol=rtol,
                    atol=atol,
                ).reshape(-1)
            ):
                diffs.append(
                    self._format_expect_actual(
                        "logprobs not close (rtol=atol=1e-2)",
                        expect_logprobs,
                        actual_logprobs,
                    )
                )

        self._compare_required_json_content(actual_result, compare_config, diffs)

        if diffs:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                self._format_all_diffs(diffs),
            )

    def _compare_extra_outputs(
        self,
        expect_extra: Any,
        actual_extra: Any,
        expect_orig: Any,
        actual_orig: Any,
        diffs: List[str],
    ) -> None:
        """Compare extra_outputs fields; append all diffs to diffs (no raise)."""
        rtol = atol = 1e-2

        def pop_attr(obj: Any, attr: str) -> Optional[Any]:
            val = getattr(obj, attr, None)
            if hasattr(obj, attr):
                setattr(obj, attr, None)
            return val

        for attr in ("hidden_states", "all_hidden_states", "loss", "logits"):
            expect_val = pop_attr(expect_extra, attr)
            actual_val = pop_attr(actual_extra, attr)
            if expect_val is None and actual_val is None:
                continue
            if expect_val is None or actual_val is None:
                diffs.append(
                    self._format_expect_actual(
                        f"extra_outputs.{attr}",
                        expect_val,
                        actual_val,
                    )
                )
                continue
            # logits values are inherently non-deterministic across GPU runs;
            # only verify shape matches (exact values are not meaningful).
            if attr == "logits":
                expect_t = torch.tensor(expect_val)
                actual_t = torch.tensor(actual_val)
                if expect_t.shape != actual_t.shape:
                    diffs.append(
                        self._format_expect_actual(
                            f"extra_outputs.{attr} shape mismatch "
                            f"(expect {expect_t.shape} vs actual {actual_t.shape})",
                            expect_val,
                            actual_val,
                        )
                    )
                continue
            res = torch.isclose(
                torch.tensor(expect_val),
                torch.tensor(actual_val),
                rtol=rtol,
                atol=atol,
            ).reshape(-1)
            if not all(res):
                diffs.append(
                    self._format_expect_actual(
                        f"extra_outputs.{attr} not close (rtol=atol=1e-2)",
                        expect_val,
                        actual_val,
                    )
                )

        if expect_extra != actual_extra:
            diffs.append(
                self._format_expect_actual(
                    "extra_outputs (remaining fields) not equal",
                    expect_orig,
                    actual_orig,
                )
            )

    def _rewrite_query_info(self, query_info: ChatCompletionRequest):
        for message in query_info.messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.image_url is not None:
                        part.image_url.url = create_temporary_copy(part.image_url.url)
                    if part.video_url is not None:
                        part.video_url.url = create_temporary_copy(part.video_url.url)

    def _compare_aux_info(
        self,
        expect_aux: Optional[AuxInfo],
        actual_aux: Optional[AuxInfo],
        diffs: List[str],
    ) -> None:
        """Compare aux_info and append any diff to diffs (no raise)."""
        if expect_aux is None and actual_aux is None:
            return
        if type(expect_aux) != type(actual_aux):
            diffs.append(
                self._format_expect_actual(
                    "aux_info type not equal",
                    expect_aux.model_dump() if expect_aux else None,
                    actual_aux.model_dump() if actual_aux else None,
                )
            )
            return
        if expect_aux is None or actual_aux is None:
            diffs.append(
                self._format_expect_actual(
                    "aux_info (one side None)",
                    expect_aux.model_dump() if expect_aux else None,
                    actual_aux.model_dump() if actual_aux else None,
                )
            )
            return

        obj1, obj2 = expect_aux, actual_aux
        ignore_fields = set(
            [
                "cost_time",
                "wait_time",
                "first_token_cost_time",
                "role_addrs.http_port",
                "role_addrs.grpc_port",
            ]
        )
        # Goldens recorded before the speculative counters existed have no such
        # keys in their raw JSON; only compare once a golden records the field.
        raw_result = self.qr_info.get("result")
        raw_aux = raw_result.get("aux_info") if isinstance(raw_result, dict) else None
        for spec_field in (
            "speculative_draft_rounds",
            "speculative_accepted_tokens_per_pos",
        ):
            if not isinstance(raw_aux, dict) or spec_field not in raw_aux:
                ignore_fields.add(spec_field)
        top_level_ignore = set()
        nested_ignore: Dict[str, set] = {}
        for field in ignore_fields:
            if "." in field:
                parts = field.split(".", 1)
                parent_field, child_field = parts[0], parts[1]
                if parent_field not in nested_ignore:
                    nested_ignore[parent_field] = set()
                nested_ignore[parent_field].add(child_field)
            else:
                top_level_ignore.add(field)

        all_fields = set(obj1.__annotations__.keys())
        fields_to_compare = all_fields - top_level_ignore

        for field in fields_to_compare:
            value1 = getattr(obj1, field)
            value2 = getattr(obj2, field)

            if field in nested_ignore and nested_ignore[field]:
                if isinstance(value1, list) and isinstance(value2, list):
                    if len(value1) != len(value2):
                        diffs.append(
                            self._format_expect_actual(
                                f"aux_info.{field} (length)",
                                value1,
                                value2,
                            )
                        )
                        continue
                    for idx, (item1, item2) in enumerate(zip(value1, value2)):
                        if hasattr(item1, "__dict__") and hasattr(item2, "__dict__"):
                            dict1 = item1.__dict__.copy()
                            dict2 = item2.__dict__.copy()
                            for ignore_field in nested_ignore[field]:
                                dict1.pop(ignore_field, None)
                                dict2.pop(ignore_field, None)
                            if dict1 != dict2:
                                diffs.append(
                                    self._format_expect_actual(
                                        f"aux_info.{field}[{idx}]",
                                        dict1,
                                        dict2,
                                    )
                                )
                        elif item1 != item2:
                            diffs.append(
                                self._format_expect_actual(
                                    f"aux_info.{field}[{idx}]",
                                    item1,
                                    item2,
                                )
                            )
                elif value1 != value2:
                    diffs.append(
                        self._format_expect_actual(f"aux_info.{field}", value1, value2)
                    )
            else:
                if value1 != value2:
                    diffs.append(
                        self._format_expect_actual(f"aux_info.{field}", value1, value2)
                    )

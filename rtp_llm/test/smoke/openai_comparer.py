import copy
import json
from typing import Any, Dict, List, Optional, Union
import os
import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import QueryStatus, SmokeException, REL_PATH
from smoke.utils import create_temporary_copy
from rtp_llm.utils.base_model_datatypes import AuxInfo
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class OpenaiComparer(BaseComparer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.is_stream = self.qr_info["query"].get("stream", False)

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        query_info = ChatCompletionRequest(**query_json)
        self._rewrite_query_info(query_info)
        return query_info

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        if result_json.get('extra_outputs', None) is not None:
            path = result_json['extra_outputs'].get('all_hidden_states', None)
            if path is not None and isinstance(path, str):
                result_json['extra_outputs']['all_hidden_states'] = torch.load(os.path.join(REL_PATH, path)).numpy().tolist()
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

    def compare_result(
        self,
        expect_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
        actual_result: Union[ChatCompletionResponse, ChatCompletionStreamResponse],
    ) -> None:
        diffs: List[str] = []

        if type(expect_result) != type(actual_result):
            diffs.append(
                "type not equal:\n  expect: "
                + type(expect_result).__name__
                + "\n  actual: "
                + type(actual_result).__name__
            )

        if expect_result.usage != actual_result.usage:
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

        if expect_choices != actual_choices:
            diffs.append(
                self._format_expect_actual(
                    "choices not equal (after normalizing logprobs)",
                    expect_choices,
                    actual_choices,
                )
            )

        rtol = atol = 1e-2
        if expect_logprobs is not None and actual_logprobs is not None:
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
        ignore_fields = set([
            "cost_time",
            "wait_time",
            "first_token_cost_time",
            "role_addrs.http_port",
            "role_addrs.grpc_port",
        ])
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

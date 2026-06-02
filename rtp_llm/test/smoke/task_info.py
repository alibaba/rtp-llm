from typing import Any, Dict, List, Optional, Union

import prettytable as pt
from pydantic import BaseModel
from smoke.common_def import QueryStatus


class LoraUpdateInfo(BaseModel):
    update_lora_action: Dict[str, Any]
    update_response: List[Union[bool, str]]


class TaskInfo(BaseModel):
    model_type: str
    model_path: str
    query_result: List[Dict[str, Any]]
    tokenizer_path: Optional[str] = None
    lora_infos: Optional[Dict[str, str]] = None
    endpoint: Optional[str] = "/"
    ptuning_path: Optional[str] = None
    update_lora_infos: Optional[List[LoraUpdateInfo]] = None
    taskinfo_rel_path: str = ""


class TaskStates(BaseModel):
    ret: bool = True
    total_count: int = 0
    query_status: List[Any] = []
    err_msg: str = ""

    def _error_status_json(self) -> List[Dict[str, Any]]:
        json_list: List[Dict[str, Any]] = []
        for idx, (status, error_msg, tracer) in enumerate(self.query_status):
            if status == QueryStatus.OK:
                continue
            else:
                json_list.append(
                    {
                        "Idx": idx,
                        "Query": (
                            None
                            if tracer.query is None
                            else tracer.query.model_dump_json(
                                indent=4, exclude_none=True
                            )
                        ),
                        "Status": status,
                        "Expect": (
                            None
                            if tracer.expect_result is None
                            else tracer.expect_result
                        ),
                        "Actual": (
                            None
                            if tracer.actual_result is None
                            else tracer.actual_result
                        ),
                        "Error": error_msg,
                    }
                )
        return json_list

    # 不知道为啥长度对齐有问题，先别用
    def _dump_pretty_status(self):
        error_status_json_list = self._error_status_json()
        for error_status in error_status_json_list:
            table = pt.PrettyTable(
                title="Error Info", align="l", padding_width=1, hrules=pt.ALL
            )
            table.field_names = ["Key", "Values"]
            table.max_width["Values"] = 150
            for k, v in error_status.items():
                table.add_row([k, v])
            print(table)

    def _pretty_error_status(self):
        rets: List[str] = []
        error_status_json_list = self._error_status_json()
        for error_status in error_status_json_list:
            idx = error_status["Idx"]
            rets.append(
                f"===============================Query Idx: {idx} ERROR================================="
            )
            for k, v in error_status.items():
                rets.append(f"{k}: {v}")
            rets.append(
                f"======================================================================================"
            )
            rets.append("\n")
        return "\n".join(rets)

    def __str__(self):
        suc_count = 0
        diff_count = 0
        visit_failed_count = 0
        other_count = 0
        for status, _, _ in self.query_status:
            if status == QueryStatus.OK:
                suc_count = suc_count + 1
            elif status == QueryStatus.COMPARE_FAILED:
                diff_count = diff_count + 1
            elif status == QueryStatus.VISIT_FAILED:
                visit_failed_count = visit_failed_count + 1
            elif status == QueryStatus.OTHERS:
                other_count = other_count + 1
        err_msg = f"total count:[{self.total_count}], suc count:[{suc_count}], compare diff count:[{diff_count}], visit_failed_count:[{visit_failed_count}], other_count: [{other_count}]"

        return (
            f"ret:[{self.ret}], err:[{self.err_msg}] curl_status:[{err_msg}]"
            + "\n"
            + self._pretty_error_status()
        )

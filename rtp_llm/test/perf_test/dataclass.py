import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from prettytable import PrettyTable


class ResponseInfo:
    success: bool = False
    input_len: int = 0
    output_len: int = 0
    wait_time: float = 0.0
    total_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    decode_time_per_token: float = 0.0
    """
    output example:
    {
        "response": [
            "aaa"
        ],
        "finished": true,
        "aux_info": [
            {
                "cost_time": 122667.0,
                "iter_count": 500,
                "prefix_len": 0,
                "input_len": 2049,
                "reuse_len": 0,
                "output_len": 500,
                "step_output_len": 500,
                "fallback_tokens": 0,
                "fallback_times": 0,
                "first_token_cost_time": 6129.027,
                "wait_time": 5021.9,
                "pd_sep": false,
                "cum_log_probs": [
                    1.401298464324817e-45
                ],
                "beam_responses": [],
                "softmax_probs": []
            }
        ]
    }
    """

    def __init__(self, response: dict, success: bool = True):
        if not success:
            return
        self.success = success
        aux_info = response.get("aux_info", {})
        self.input_len = aux_info.get("input_len", 0)
        self.output_len = aux_info.get("output_len", 0)
        self.wait_time = aux_info.get("wait_time", 0.0)
        self.total_time = aux_info.get("cost_time", 0.0) - self.wait_time
        self.prefill_time = aux_info.get("first_token_cost_time", 0.0) - self.wait_time
        self.decode_time = self.total_time - self.prefill_time
        self.decode_time_per_token = self.decode_time / (self.output_len - 1) if self.output_len > 1 else 0.0


@dataclass
class TestResultMetrics:
    total_requests: int
    success_requests: int
    fail_requests: int
    avg_input_len: float = 0.0
    avg_output_len: float = 0.0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    avg_total_time: float = 0.0
    max_total_time: float = 0.0
    avg_prefill_time: float = 0.0
    max_prefill_time: float = 0.0
    prefill_time_var: float = 0.0
    avg_decode_time: float = 0.0
    max_decode_time: float = 0.0
    decode_time_var: float = 0.0


def analyze_results(responses: List[ResponseInfo]) -> TestResultMetrics:
    total_request_count = len(responses)
    success_requests = [r for r in responses if r.success]
    success_count = len(success_requests)
    fail_count = total_request_count - success_count
    metrics = TestResultMetrics(
        total_requests=total_request_count,
        success_requests=success_count,
        fail_requests=fail_count,
    )
    if success_count:
        metrics.avg_input_len = (
            sum([r.input_len for r in success_requests]) / success_count
        )
        metrics.avg_output_len = (
            sum([r.output_len for r in success_requests]) / success_count
        )
        metrics.avg_wait_time = (
            sum([r.wait_time for r in success_requests]) / success_count
        )
        metrics.max_wait_time = max([r.wait_time for r in success_requests])
        metrics.avg_total_time = (
            sum([r.total_time for r in success_requests]) / success_count
        )
        metrics.max_total_time = max([r.total_time for r in success_requests])
        metrics.avg_prefill_time = (
            sum([r.prefill_time for r in success_requests]) / success_count
        )
        metrics.max_prefill_time = max([r.prefill_time for r in success_requests])
        metrics.prefill_time_var = (
            sum(
                [
                    (r.prefill_time - metrics.avg_prefill_time) ** 2
                    for r in success_requests
                ]
            )
            / success_count
        )
        metrics.avg_decode_time = (
            sum([r.decode_time_per_token for r in success_requests]) / success_count
        )
        metrics.max_decode_time = max(
            [r.decode_time_per_token for r in success_requests]
        )
        metrics.decode_time_var = (
            sum(
                [
                    (r.decode_time_per_token - metrics.avg_decode_time) ** 2
                    for r in success_requests
                ]
            )
            / success_count
        )
    return metrics


class MetricState(object):
    def __init__(self, input_len: int, batch_size: int, metrics: TestResultMetrics):
        self.input_len = input_len
        self.batch_size = batch_size
        self.metrics = metrics


class TableType(Enum):
    Prefill = "prefill"
    Decode = "decode"


def create_metrics_table(
    table_type: TableType,
    metrics_list: List[MetricState],
    dump_json_path: str,
    model_info: Dict[str, Any],
) -> str:
    title = "Prefill Result" if table_type == TableType.Prefill else "Decode Result"
    json_result: Dict[str, Any] = {
        "title": title,
        "metrics": [],
        "model_info": model_info,
    }
    main_table = PrettyTable()
    main_table.title = title
    main_table.field_names = [
        "Seq Len",
        "Batch Size",
        "Sucess/Total Req",
        "Input/Output",
        "Waiting Time(ms)",
    ] + (
        ["Prefill Time(ms)"] if table_type == TableType.Prefill else ["Decode Time(ms)"]
    )
    for metrics_item in metrics_list:
        metrics = metrics_item.metrics
        if metrics.success_requests > 0:
            main_table.add_row(
                [
                    metrics_item.input_len,
                    metrics_item.batch_size,
                    f"{metrics.success_requests}/{metrics.total_requests}",
                    f"{metrics.avg_input_len:.0f}/{metrics.avg_output_len:.0f}",
                    f"{metrics.avg_wait_time:.2f}",
                ]
                + (
                    [f"{metrics.avg_prefill_time:.2f}"]
                    if table_type == TableType.Prefill
                    else [f"{metrics.avg_decode_time:.2f}"]
                )
            )
            json_result["metrics"].append(
                {
                    "input_len": metrics_item.input_len,
                    "batch_size": metrics_item.batch_size,
                    "success_rate": metrics.success_requests / metrics.total_requests,
                    "avg_wait_time": metrics.avg_wait_time,
                    "avg_prefill_time": metrics.avg_prefill_time,
                    "avg_decode_time": metrics.avg_decode_time,
                }
            )
        else:
            main_table.add_row(
                [
                    metrics_item.input_len,
                    metrics_item.batch_size,
                    f"0/{metrics.total_requests}",
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )
    with open(f"{dump_json_path}/{title.replace(' ', '_')}.json", "w") as f:
        json.dump(json_result, f, indent=4)
    main_table.align = "l"
    return main_table.get_string()

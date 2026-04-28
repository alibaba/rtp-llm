import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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

    def __init__(self, response: Dict[str, Any], success: bool = True):
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
        self.decode_time_per_token = (
            self.decode_time / (self.output_len - 1) if self.output_len > 1 else 0.0
        )


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


class DistributionMetricState(object):
    def __init__(
        self, batch_size: int, seq_len_list: List[int], metrics: TestResultMetrics
    ):
        self.batch_size = batch_size
        self.seq_len_list = seq_len_list
        self.metrics = metrics


class TableType(Enum):
    Prefill = "prefill"
    Decode = "decode"


def create_metrics_table(
    table_type: TableType,
    metrics_list: List[MetricState],
    dump_json_path: str,
    model_info: Dict[str, Any],
    title: str,
    generate_config: Optional[Dict[str, Any]] = None,
) -> str:
    json_result: Dict[str, Any] = {
        "title": title,
        "mode": "grid",
        "metrics": [],
        "model_info": model_info,
        "generate_config": generate_config or {},
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
    os.makedirs(dump_json_path, exist_ok=True)
    with open(f"{dump_json_path}/{title.replace(' ', '_')}.json", "w") as f:
        json.dump(json_result, f, indent=4)
    main_table.align = "l"
    return main_table.get_string()


@dataclass
class TpsSearchStep:
    batch_size: int
    avg_decode_time: float  # ms
    success_rate: float
    satisfies_tpot: bool


@dataclass
class TpsResult:
    input_len: int = 0  # grid mode; 0 for distribution mode
    seq_len_list: List[int] = field(default_factory=list)  # distribution mode
    target_tpot: float = 0.0  # ms
    best_bs: int = 0  # max BS satisfying TPOT (0 = all failed)
    actual_tpot: float = 0.0  # ms
    tps: float = 0.0  # tokens/s = best_bs / (actual_tpot / 1000)
    search_steps: List[TpsSearchStep] = field(default_factory=list)


def create_tps_result_table(
    results: List[TpsResult],
    dump_json_path: str,
    title: str,
    generate_config: Optional[Dict[str, Any]] = None,
) -> str:
    json_result: Dict[str, Any] = {
        "title": title,
        "mode": "tps",
        "target_tpot": results[0].target_tpot if results else 0,
        "results": [],
        "generate_config": generate_config or {},
    }
    main_table = PrettyTable()
    main_table.title = title
    main_table.field_names = [
        "Input Len",
        "Target TPOT(ms)",
        "Best BS",
        "Actual TPOT(ms)",
        "TPS(tokens/s)",
    ]
    for r in results:
        label = str(r.input_len) if r.input_len > 0 else "distribution"
        main_table.add_row(
            [
                label,
                f"{r.target_tpot:.2f}",
                r.best_bs if r.best_bs > 0 else "N/A",
                f"{r.actual_tpot:.2f}" if r.best_bs > 0 else "N/A",
                f"{r.tps:.2f}" if r.best_bs > 0 else "N/A",
            ]
        )
        json_result["results"].append(
            {
                "input_len": r.input_len,
                "seq_len_list": r.seq_len_list,
                "target_tpot": r.target_tpot,
                "best_bs": r.best_bs,
                "actual_tpot": r.actual_tpot,
                "tps": r.tps,
                "search_steps": [
                    {
                        "batch_size": s.batch_size,
                        "avg_decode_time": s.avg_decode_time,
                        "success_rate": s.success_rate,
                        "satisfies_tpot": s.satisfies_tpot,
                    }
                    for s in sorted(r.search_steps, key=lambda x: x.batch_size)
                ],
            }
        )
    os.makedirs(dump_json_path, exist_ok=True)
    with open(f"{dump_json_path}/{title.replace(' ', '_')}.json", "w") as f:
        json.dump(json_result, f, indent=4)
    main_table.align = "l"
    logging.info("TPS result table:\n" + main_table.get_string())

    # Print binary search steps sorted by BS for each input_len
    steps_table = PrettyTable()
    steps_table.title = f"{title} — Binary Search Steps"
    steps_table.field_names = [
        "Input Len",
        "BS",
        "TPOT(ms)",
        "Success Rate",
        "Status",
    ]
    for r in results:
        label = str(r.input_len) if r.input_len > 0 else "distribution"
        for s in sorted(r.search_steps, key=lambda x: x.batch_size):
            steps_table.add_row(
                [
                    label,
                    s.batch_size,
                    f"{s.avg_decode_time:.2f}",
                    f"{s.success_rate:.2f}",
                    "PASS" if s.satisfies_tpot else "FAIL",
                ]
            )
    steps_table.align = "l"
    logging.info("Binary search steps:\n" + steps_table.get_string())

    return main_table.get_string()


@dataclass
class PerfTestConfig:
    """Unified test configuration, eliminates distribution/grid duplication in main()."""

    is_distribution: bool
    all_seq_lens: List[int]
    batch_size_list: List[int]
    input_len_list: List[int]
    max_seq_len: int
    max_concurrency: int
    test_config: Optional[Dict[str, Any]] = None


def create_distribution_metrics_table(
    metrics_list: List[DistributionMetricState],
    dump_json_path: str,
    distribution_source: str,
    title: str,
    generate_config: Optional[Dict[str, Any]] = None,
) -> str:
    json_result: Dict[str, Any] = {
        "title": title,
        "mode": "distribution",
        "distribution_source": distribution_source,
        "test_cases": [],
        "generate_config": generate_config or {},
    }
    main_table = PrettyTable()
    main_table.title = title
    main_table.field_names = [
        "Batch Size",
        "Seq Lens (min/max)",
        "Success/Total Req",
        "Avg Seq Len",
        "Avg Decode Time(ms)",
    ]
    for item in metrics_list:
        m = item.metrics
        if m.success_requests > 0:
            main_table.add_row(
                [
                    item.batch_size,
                    f"{min(item.seq_len_list)}/{max(item.seq_len_list)}",
                    f"{m.success_requests}/{m.total_requests}",
                    f"{m.avg_input_len:.0f}",
                    f"{m.avg_decode_time:.2f}",
                ]
            )
            json_result["test_cases"].append(
                {
                    "batch_size": item.batch_size,
                    "seq_len_list": item.seq_len_list,
                    "success_rate": m.success_requests / m.total_requests,
                    "avg_seq_len": m.avg_input_len,
                    "avg_decode_time_per_token": m.avg_decode_time,
                    "max_decode_time_per_token": m.max_decode_time,
                    "avg_wait_time": m.avg_wait_time,
                }
            )
        else:
            main_table.add_row(
                [
                    item.batch_size,
                    f"{min(item.seq_len_list)}/{max(item.seq_len_list)}",
                    f"0/{m.total_requests}",
                    "N/A",
                    "N/A",
                ]
            )
    os.makedirs(dump_json_path, exist_ok=True)
    with open(f"{dump_json_path}/{title.replace(' ', '_')}.json", "w") as f:
        json.dump(json_result, f, indent=4)
    main_table.align = "l"
    return main_table.get_string()

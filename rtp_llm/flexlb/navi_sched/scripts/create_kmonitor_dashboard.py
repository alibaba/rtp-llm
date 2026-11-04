#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


BASE_URL = "https://kmonitor2.alibaba-inc.com"
COOKIE_FILE = Path.home() / ".kmonitor-auth/cookies.txt"
OUTPUT_FILE = Path("/tmp/navi_sched_kmonitor_dashboard.json")
CAPACITY_SPLIT_PARAMETER_COUNT = 19
DEFAULT_HIPPO_APP = "whale_prod_Qwen3-235B-A22B-Instruct-2507_for_omega_64K"

parser = argparse.ArgumentParser()
parser.add_argument("--cookies", type=Path, default=COOKIE_FILE)
parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
parser.add_argument("--generate-only", action="store_true")
args = parser.parse_args()


def target(
    metric,
    alias,
    ref_id,
    role,
    aggregator=None,
    downsample_aggregator="avg",
    extra_tags=None,
):
    if aggregator is None:
        aggregator = "sum" if role == "master" else "global_avg"
    deployment_selector = (
        "[[master_deployment]]" if role == "master" else "[[deployment]]"
    )
    tags = {
        "hippo_app": "[[app]]",
        "deployment_name": deployment_selector,
        "role": role,
    }
    # Agent and Client run on the same inference hosts.
    if role in ("agent", "client"):
        tags["host_ip"] = "[[host]]"
    if extra_tags:
        tags.update(extra_tags)
    if role in ("agent", "client"):
        series_alias = alias + " / $tag_hippo_app / $tag_deployment_name / $tag_host_ip"
    else:
        series_alias = alias + " / $tag_hippo_app / $tag_deployment_name"
    if extra_tags:
        series_alias += " / " + " / ".join(
            f"$tag_{tag_name}" for tag_name in extra_tags
        )
    return {
        "aggregateMethod": "highest average",
        "aggregateNumber": 100,
        "aggregator": aggregator,
        "alias": series_alias,
        "downsampleAggregator": downsample_aggregator,
        "downsampleFillPolicy": "none",
        "granularity": "auto",
        "groupBy": "",
        "hide": False,
        "isFirst": ref_id == "A",
        "metric": metric,
        "refId": ref_id,
        "tags": tags,
        "tenant": "default",
    }


def graph(panel_id, title, x, y, metrics, unit="short", height=8):
    display_title = title + ("（ms）" if unit == "ms" else "")
    axis_format = "none" if unit == "ms" else unit
    return {
        "id": panel_id,
        "title": display_title,
        "type": "graph",
        "gridPos": {"h": height, "w": 12, "x": x, "y": y},
        "datasource": None,
        "lines": True,
        "linewidth": 1,
        "fill": 1,
        "points": False,
        "legend": {
            "alignAsTable": True,
            "avg": True,
            "current": True,
            "max": True,
            "min": True,
            "show": True,
            "values": True,
        },
        "tooltip": {"shared": True, "sort": 2, "value_type": "individual"},
        "yaxes": [
            {"format": axis_format, "logBase": 1, "show": True},
            {"format": "short", "logBase": 1, "show": True},
        ],
        "targets": [
            target(
                metric[0], metric[1], chr(ord("A") + index), metric[2],
                metric[3] if len(metric) > 3 else None,
                metric[4] if len(metric) > 4 else "avg",
                metric[5] if len(metric) > 5 else None,
            )
            for index, metric in enumerate(metrics)
        ],
    }


def row(panel_id, title, y):
    return {
        "id": panel_id,
        "title": title,
        "type": "row",
        "collapsed": True,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


panels = []
panel_id = 1
y = 0
column = 0
current_row = None


def add_row(title):
    global panel_id, y, column, current_row
    if column:
        y += 8
        column = 0
    current_row = row(panel_id, title, y)
    panels.append(current_row)
    panel_id += 1
    y += 1


def add_graph(title, metrics, unit="short"):
    global panel_id, y, column
    if current_row is None:
        raise ValueError(f"graph {title!r} does not belong to a dashboard row")
    current_row["panels"].append(
        graph(panel_id, title, column * 12, y, metrics, unit)
    )
    panel_id += 1
    column = 1 - column
    if column == 0:
        y += 8


add_row("Agent - 单机吞吐与执行状态")
add_graph("单机实际 Batch 大小", [
    ("navi_sched.agent.executed_batch_request_count_mean", "1 秒窗口平均请求数 / Batch", "agent"),
    ("navi_sched.agent.running_batch_request_count", "当前执行 Batch 请求数", "agent"),
])
add_graph("单机 Batch QPS", [
    ("navi_sched.agent.executed_batch_qps", "执行 Batch QPS", "agent", "sum"),
    ("navi_sched.agent.abnormal_finished_batch_qps", "异常 Batch QPS", "agent", "sum"),
])
add_graph("单机 Batch 聚合异常 QPS", [
    ("navi_sched.agent.batch_timeout_qps", "等待完整 Batch 超时", "agent", "sum"),
    (
        "navi_sched.agent.batch_timeout_missing_request_qps",
        "超时缺失 Request",
        "agent",
        "sum",
    ),
    ("navi_sched.agent.batch_late_request_qps", "迟到 Request", "agent", "sum"),
])
add_graph("单机 Batch 延迟", [
    ("navi_sched.agent.batch_formation_window_mean_ms", "凑 Batch 平均延迟", "agent"),
    ("navi_sched.agent.batch_formation_window_max_ms", "凑 Batch 最大延迟", "agent", "max", "max"),
    ("navi_sched.agent.validation_window_execution_actual_mean_ms", "实际执行延迟", "agent"),
], "ms")
add_graph("单机 Request QPS", [
    ("navi_sched.agent.request_received_qps", "请求接收 QPS", "agent", "sum"),
    ("navi_sched.agent.request_finished_qps", "请求完成 QPS", "agent", "sum"),
    ("navi_sched.agent.request_abnormal_finished_qps", "异常完成 QPS", "agent", "sum"),
])
add_graph("单机平均 Prompt 长度", [
    ("navi_sched.agent.request_prompt_tokens_mean", "平均 Prompt Tokens / Request", "agent"),
])
add_graph("单机 Request 延迟（Queue + Execution）", [
    ("navi_sched.agent.validation_window_total_actual_mean_ms", "实际延迟", "agent"),
    ("navi_sched.agent.validation_window_total_predicted_mean_ms", "预测延迟", "agent"),
], "ms")
add_graph("单机 Queue 延迟", [
    ("navi_sched.agent.validation_window_queue_actual_mean_ms", "C++ Batch Ready 到首次 Prefill", "agent"),
    ("navi_sched.agent.validation_window_queue_predicted_mean_ms", "Master 预测 Queue", "agent"),
    ("navi_sched.agent.queue_model_window_prediction_mean_ms", "Agent 拟合 Queue", "agent"),
], "ms")
add_graph("单机 vLLM 纯等待", [
    ("navi_sched.agent.vllm_admission_to_first_execution_window_mean_ms", "Admission 到首次执行均值", "agent"),
    ("navi_sched.agent.vllm_admission_to_first_execution_window_max_ms", "Admission 到首次执行最大值", "agent"),
], "ms")
add_graph("单机 vLLM 纯等待样本", [
    ("navi_sched.agent.vllm_admission_to_first_execution_window_sample_count", "完成样本数", "agent", "sum"),
])
add_graph("单机执行时间占比", [
    ("navi_sched.agent.prefill_card_time_utilization_pct", "NaviSched Prefill 区间", "agent"),
    ("navi_sched.agent.vllm_execution_time_utilization_pct", "vLLM execute_model（含 Prefill / Decode / Recompute）", "agent"),
], "percent")
add_graph("图片准备结果 QPS", [
    ("navi_sched.agent.image_prepare_completed_qps", "图片准备完成（不含取消）", "agent", "sum"),
    ("navi_sched.agent.image_prepare_success_qps", "图片准备成功", "agent", "sum"),
    ("navi_sched.agent.image_prepare_rate_limited_qps", "图片服务限流（503 RATE_LIMIT_EXCEEDED）", "agent", "sum"),
    ("navi_sched.agent.image_prepare_other_error_qps", "图片准备其他失败", "agent", "sum"),
    ("navi_sched.agent.image_prepare_cancelled_qps", "图片准备取消", "agent", "sum"),
])
add_graph("单机 vLLM Batch 执行分段", [
    ("navi_sched.agent.vllm_batch_wall_window_mean_ms", "首个 Prefill 开始到全批 Prefill 完成", "agent"),
    ("navi_sched.agent.vllm_batch_wall_window_max_ms", "最长全批 Prefill Wall Time", "agent"),
    ("navi_sched.agent.vllm_batch_execute_model_sum_window_mean_ms", "Batch 参与的 execute_model 累计", "agent"),
    ("navi_sched.agent.vllm_batch_inter_step_gap_window_mean_ms", "被其他 vLLM Step 穿插的间隙", "agent"),
    ("navi_sched.agent.vllm_batch_inter_step_gap_window_max_ms", "最大穿插间隙", "agent"),
], "ms")
add_graph("单机 vLLM Batch 穿插压力", [
    ("navi_sched.agent.vllm_batch_inter_step_gap_ratio_pct", "间隙占 Wall Time", "agent"),
], "percent")
add_graph("单机 vLLM EngineCore 单步耗时", [
    ("navi_sched.agent.vllm_core_step_window_mean_ms", "单步均值", "agent"),
    ("navi_sched.agent.vllm_core_step_window_max_ms", "单步最大值", "agent"),
    ("navi_sched.agent.vllm_future_wait_window_mean_ms", "Future 等待均值", "agent"),
    ("navi_sched.agent.vllm_future_wait_window_max_ms", "Future 等待最大值", "agent"),
], "ms")
add_graph("单机 vLLM EngineCore 单步观测数", [
    ("navi_sched.agent.vllm_core_step_window_count", "单步数", "agent", "sum"),
    ("navi_sched.agent.vllm_future_wait_window_count", "Future 等待数", "agent", "sum"),
])
add_graph("单机 vLLM Future 回调跨度（非单步）", [
    ("navi_sched.agent.vllm_execution_step_window_mean_ms", "回调跨度均值", "agent"),
    ("navi_sched.agent.vllm_execution_step_window_max_ms", "回调跨度最大值", "agent"),
    ("navi_sched.agent.vllm_pure_decode_step_window_mean_ms", "纯 Decode 回调跨度均值", "agent"),
    ("navi_sched.agent.vllm_pure_decode_step_window_max_ms", "纯 Decode 回调跨度最大值", "agent"),
], "ms")
add_graph("单机 vLLM Future 回调观测数", [
    ("navi_sched.agent.vllm_execution_step_window_count", "全部成功回调", "agent", "sum"),
    ("navi_sched.agent.vllm_pure_decode_step_window_count", "纯 Decode 回调", "agent", "sum"),
])
add_graph("单机 vLLM Batch Step 与 Decode", [
    ("navi_sched.agent.vllm_batch_execution_step_count_window_mean", "执行 Step / Batch", "agent"),
    ("navi_sched.agent.vllm_batch_prefill_tokens_window_mean", "Prefill Tokens / Batch", "agent"),
    ("navi_sched.agent.vllm_batch_decode_request_steps_window_mean", "穿插 Decode Request Steps / Batch", "agent"),
    ("navi_sched.agent.vllm_batch_decode_tokens_window_mean", "穿插 Decode Tokens / Batch", "agent"),
    ("navi_sched.agent.vllm_batch_observation_window_batch_count", "观测 Batch 数 / 窗口", "agent"),
], "short")
add_graph("单机当前 Batch 与请求数", [
    ("navi_sched.agent.pending_batches", "等待凑齐的 Batch", "agent"),
    ("navi_sched.agent.inflight_requests", "已接收未完成请求", "agent"),
    ("navi_sched.agent.requests_batch_waiting", "正在凑 Batch 的请求", "agent"),
    ("navi_sched.agent.requests_queued_for_execution", "等待 GPU 执行的请求", "agent"),
    ("navi_sched.agent.requests_running", "GPU 执行中请求", "agent"),
])
add_graph("单机在途 Request 预测延迟", [
    (
        "navi_sched.agent.inflight_predicted_latency_sum_ms",
        "预测延迟总和",
        "agent",
    ),
    (
        "navi_sched.agent.inflight_predicted_latency_average_ms",
        "平均预测延迟",
        "agent",
    ),
], "ms")
add_graph("单机在途 Request 一致性异常 QPS", [
    ("navi_sched.agent.inflight_invalid_add_qps", "非法加入", "agent", "sum"),
    ("navi_sched.agent.inflight_duplicate_add_qps", "重复加入", "agent", "sum"),
    ("navi_sched.agent.inflight_missing_remove_qps", "重复或缺失删除", "agent", "sum"),
])
add_graph("单机 Cache 容量与 Pending 压力", [
    ("navi_sched.agent.cache_total_blocks", "Cache 总容量（Blocks）", "agent"),
    ("navi_sched.agent.cached_block_count", "实际 Cache（Blocks）", "agent"),
    ("navi_sched.agent.pending_cache_unique_block_count", "Pending 唯一 Blocks", "agent"),
    ("navi_sched.agent.pending_cache_batch_count", "Pending Batch 数", "agent"),
])
add_graph("单机 Cache 容量占比", [
    ("navi_sched.agent.cache_occupancy_pct", "实际 Cache / 总容量", "agent"),
    (
        "navi_sched.agent.pending_cache_unique_block_pct",
        "Pending 唯一 Blocks / 总容量",
        "agent",
    ),
], "percent")
add_graph("单机凑 Batch 完成量", [
    ("navi_sched.agent.batch_formation_window_batch_count", "最近 1 秒完成 Batch 数", "agent", "sum"),
])
add_graph("单机 Ready 但未执行的空档", [
    ("navi_sched.agent.ready_without_running_current_ms", "当前空档", "agent"),
    ("navi_sched.agent.ready_without_running_mean_ms", "平均空档", "agent"),
    ("navi_sched.agent.ready_without_running_max_ms", "最大空档", "agent"),
], "ms")
add_graph("Execution 起点贯通延迟", [
    ("navi_sched.agent.batch_started_processing_lag_window_mean_ms", "最近 1 秒平均：Cache 准备 + 事件消费", "agent"),
    ("navi_sched.agent.batch_started_processing_lag_window_max_ms", "最近 1 秒最大：Cache 准备 + 事件消费", "agent"),
], "ms")
add_graph("单机存活与 Master 订阅", [
    ("navi_sched.role.alive", "Agent 存活", "agent"),
    ("navi_sched.agent.collector_count", "已订阅 Master 数", "agent"),
])

add_row("Agent - 单机预测精度")
add_graph("Validation 数据新鲜度", [
    ("navi_sched.agent.latency_validation_enabled", "Validation 已启用", "agent"),
    ("navi_sched.agent.validation_window_new_sample_count", "当前窗口新增样本", "agent", "sum"),
    ("navi_sched.agent.validation_last_update_time_s", "最后样本时间", "agent"),
    ("navi_sched.agent.validation_last_update_age_ms", "距最后样本", "agent"),
])
add_graph("Validation 分桶数据新鲜度", [
    ("navi_sched.agent.validation_execution_bucket_enabled", "Validation 已启用", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
    ("navi_sched.agent.validation_execution_bucket_new_sample_count", "当前窗口新增样本", "agent", "sum", "avg", {"bucket_id": "[[bucket]]"}),
    ("navi_sched.agent.validation_execution_bucket_last_update_time_s", "最后样本时间", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
    ("navi_sched.agent.validation_execution_bucket_last_update_age_ms", "距最后样本", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
])
add_graph("单机 Queue 预测值与真实排队", [
    ("navi_sched.agent.validation_window_queue_predicted_mean_ms", "Master 预测 Queue", "agent"),
    ("navi_sched.agent.validation_window_queue_actual_mean_ms", "实际 Queue", "agent"),
    ("navi_sched.agent.validation_window_queue_backend_mean_ms", "Backend 真实排队", "agent"),
], "ms")
add_graph("单机 Queue 预测误差", [
    ("navi_sched.agent.validation_queue_error_ms", "静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_queue_absolute_error_ms", "逐 Batch MAE", "agent"),
], "ms")
add_graph("Queue 在线模型：Raw / 拟合 / 真值", [
    ("navi_sched.agent.queue_model_window_raw_prediction_mean_ms", "Raw 前序时延总和", "agent"),
    ("navi_sched.agent.queue_model_window_prediction_mean_ms", "Agent 拟合预测", "agent"),
    ("navi_sched.agent.queue_model_window_actual_mean_ms", "实际 Queue", "agent"),
], "ms")
add_graph("Queue 在线模型：Raw / 拟合误差", [
    ("navi_sched.agent.queue_model_raw_error_ms", "Raw 静态偏差", "agent"),
    ("navi_sched.agent.queue_model_raw_absolute_error_ms", "Raw MAE", "agent"),
    ("navi_sched.agent.queue_model_error_ms", "拟合静态偏差", "agent"),
    ("navi_sched.agent.queue_model_absolute_error_ms", "拟合 MAE", "agent"),
], "ms")
add_graph("Queue 在线模型：样本特征与训练进度", [
    ("navi_sched.agent.queue_model_window_preceding_batch_count_mean", "前序 Batch 数", "agent"),
    ("navi_sched.agent.queue_model_fit_step", "Adam 更新步数", "agent"),
])
add_graph("Queue 在线模型：生效权重", [
    ("navi_sched.agent.queue_model_weight_0", "固定截距", "agent"),
    ("navi_sched.agent.queue_model_weight_1", "Raw Queue 校正", "agent"),
    ("navi_sched.agent.queue_model_weight_2", "每 Batch 间隙", "agent"),
])
add_graph("Execution 在线模型：版本与训练进度", [
    ("navi_sched.agent.latency_model_formula", "公式 ID", "agent"),
    ("navi_sched.agent.latency_model_binary_version", "二进制版本", "agent"),
    ("navi_sched.agent.latency_model_revision", "模型更新 Revision", "agent"),
    ("navi_sched.agent.latency_model_parameter_count", "参数个数", "agent"),
    ("navi_sched.agent.latency_model_trusted", "模型已可信", "agent"),
    ("navi_sched.agent.latency_model_warm_started", "已采用共享初值", "agent"),
])
add_graph("Execution 在线模型：零点对齐", [
    ("navi_sched.agent.latency_model_zero_point", "Agent 真实零点", "agent"),
    ("navi_sched.master.latency_model_global_zero_point_10", "Master 全局零点 / 10 参数", "master"),
    ("navi_sched.master.latency_model_global_zero_point_19", "Master 全局零点 / 19 参数", "master"),
], "ms")
add_graph("Execution 全局零点可用性", [
    ("navi_sched.master.latency_model_global_zero_point_available_10", "10 参数可用", "master"),
    ("navi_sched.master.latency_model_global_zero_point_available_19", "19 参数可用", "master"),
    ("navi_sched.master.latency_model_global_zero_point_invalid_reason_10", "10 参数不可用原因", "master"),
    ("navi_sched.master.latency_model_global_zero_point_invalid_reason_19", "19 参数不可用原因", "master"),
])
add_graph("Execution 全局零点匹配节点", [
    ("navi_sched.master.latency_model_global_zero_point_group_live_node_count_10", "10 参数在线节点", "master"),
    ("navi_sched.master.latency_model_global_zero_point_group_trusted_node_count_10", "10 参数可信节点", "master"),
    ("navi_sched.master.latency_model_global_zero_point_group_live_node_count_19", "19 参数在线节点", "master"),
    ("navi_sched.master.latency_model_global_zero_point_group_trusted_node_count_19", "19 参数可信节点", "master"),
])
add_graph("Execution 全局零点查询键", [
    ("navi_sched.master.latency_model_global_zero_point_query_formula_10", "10 参数 Formula", "master"),
    ("navi_sched.master.latency_model_global_zero_point_query_binary_version_10", "10 参数 Binary Version", "master"),
    ("navi_sched.master.latency_model_global_zero_point_query_parameter_count_10", "10 参数 Parameter Count", "master"),
    ("navi_sched.master.latency_model_global_zero_point_query_formula_19", "19 参数 Formula", "master"),
    ("navi_sched.master.latency_model_global_zero_point_query_binary_version_19", "19 参数 Binary Version", "master"),
    ("navi_sched.master.latency_model_global_zero_point_query_parameter_count_19", "19 参数 Parameter Count", "master"),
])
add_graph("Execution 在线模型：线性参数", [
    ("navi_sched.agent.latency_model_parameter_0", "P0 截距", "agent"),
    ("navi_sched.agent.latency_model_parameter_1", "P1 Batch 大小", "agent"),
    ("navi_sched.agent.latency_model_parameter_2", "P2 Cache Hit Tokens", "agent"),
    ("navi_sched.agent.latency_model_parameter_3", "P3 Compute Tokens", "agent"),
    ("navi_sched.agent.latency_model_parameter_4", "P4 Compute Tokens 平方", "agent"),
    ("navi_sched.agent.latency_model_parameter_5", "P5 Hit x Compute", "agent"),
])
add_graph("Execution 在线模型：Batch 与非线性参数", [
    ("navi_sched.agent.latency_model_parameter_6", "P6 Batch 数量", "agent"),
    ("navi_sched.agent.latency_model_parameter_7", "P7 零负载延迟", "agent"),
    ("navi_sched.agent.latency_model_parameter_8", "P8 非线性曲率", "agent"),
    ("navi_sched.agent.latency_model_parameter_9", "P9 总 Prompt Tokens", "agent"),
    ("navi_sched.agent.latency_model_parameter_10", "P10 Waiting 请求数", "agent"),
])
add_graph("Execution 在线模型：Capacity Split 参数", [
    ("navi_sched.agent.latency_model_parameter_11", "P11 Waiting Cache Hit", "agent"),
    ("navi_sched.agent.latency_model_parameter_12", "P12 Waiting Compute", "agent"),
    ("navi_sched.agent.latency_model_parameter_13", "P13 Waiting Compute 平方", "agent"),
    ("navi_sched.agent.latency_model_parameter_14", "P14 Waiting Hit x Compute", "agent"),
    ("navi_sched.agent.latency_model_parameter_15", "P15 前序请求数", "agent"),
    ("navi_sched.agent.latency_model_parameter_16", "P16 Running 已运行时间", "agent"),
    ("navi_sched.agent.latency_model_parameter_17", "P17 Running Decode 请求数", "agent"),
    ("navi_sched.agent.latency_model_parameter_18", "P18 Running Decode 已生成 Tokens", "agent"),
])
add_graph("Execution 在线模型：非线性工作区段 r", [
    ("navi_sched.agent.latency_nonlinear_r", "r = u / sqrt(曲率)", "agent"),
])
add_graph("Execution 在线模型：非线性输入与曲率", [
    ("navi_sched.agent.latency_nonlinear_u", "u", "agent"),
    ("navi_sched.agent.latency_nonlinear_curvature", "曲率", "agent"),
])

add_graph("Master 使用的 Residual 参数", [
    ("navi_sched.master.latency_residual_parameter_ms", "原始参数", "master", "global_avg", "avg", {"residual_bucket": "[[residual_bucket]]", "host_ip": "[[host]]"}),
    ("navi_sched.master.latency_residual_effective_parameter_ms", "衰减后参数", "master", "global_avg", "avg", {"residual_bucket": "[[residual_bucket]]", "host_ip": "[[host]]"}),
], "ms")
add_graph("Master 使用的 Residual 状态", [
    ("navi_sched.master.latency_residual_decay_factor", "衰减系数", "master", "global_avg", "avg", {"host_ip": "[[host]]"}),
    ("navi_sched.master.latency_residual_age_ms", "距最后更新", "master", "global_avg", "avg", {"host_ip": "[[host]]"}),
])
add_graph("单机 Execution 预测值与真值", [
    ("navi_sched.agent.validation_window_execution_master_predicted_mean_ms", "Master 调度前预测", "agent"),
    ("navi_sched.agent.validation_window_execution_agent_predicted_mean_ms", "Agent 更新前预测", "agent"),
    ("navi_sched.agent.validation_window_execution_actual_mean_ms", "实际执行", "agent"),
], "ms")
add_graph("单机 Execution 预测误差", [
    ("navi_sched.agent.validation_execution_master_error_ms", "Master 静态偏差", "agent"),
    ("navi_sched.agent.validation_execution_master_absolute_error_ms", "Master MAE", "agent"),
    ("navi_sched.agent.validation_execution_agent_error_ms", "Agent 静态偏差", "agent"),
    ("navi_sched.agent.validation_execution_agent_absolute_error_ms", "Agent MAE", "agent"),
], "ms")
add_graph("单机 Execution 相对误差", [
    ("navi_sched.agent.validation_execution_master_absolute_percentage_error_pct", "Master MAPE", "agent"),
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "Agent MAPE", "agent"),
], "percent")
add_graph("Execution 相对误差分布", [
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "P50", "agent", "global_avg", "p50"),
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "P95", "agent", "global_avg", "p95"),
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "P99", "agent", "global_avg", "p99"),
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "Max", "agent", "global_avg", "max"),
], "percent")
add_graph("Execution 分桶绝对误差", [
    ("navi_sched.agent.validation_execution_bucket_master_absolute_error_ms", "Master MAE", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
    ("navi_sched.agent.validation_execution_bucket_agent_absolute_error_ms", "Agent MAE", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
], "ms")
add_graph("Execution 分桶相对误差", [
    ("navi_sched.agent.validation_execution_bucket_master_absolute_percentage_error_pct", "Master MAPE", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
    ("navi_sched.agent.validation_execution_bucket_agent_absolute_percentage_error_pct", "Agent MAPE", "agent", "global_avg", "avg", {"bucket_id": "[[bucket]]"}),
], "percent")
add_graph("单机端到端预测值与真值", [
    ("navi_sched.agent.validation_window_total_predicted_mean_ms", "预测 Queue + Execution", "agent"),
    ("navi_sched.agent.validation_window_total_actual_mean_ms", "实际 Queue + Execution", "agent"),
], "ms")
add_graph("单机端到端预测误差", [
    ("navi_sched.agent.validation_total_error_ms", "静态偏差", "agent"),
    ("navi_sched.agent.validation_total_absolute_error_ms", "逐 Batch MAE", "agent"),
    ("navi_sched.agent.validation_total_absolute_percentage_error_pct", "MAPE", "agent"),
])
add_graph("单机 Cache Hit 与 Batch 特征", [
    ("navi_sched.agent.validation_window_cache_hit_tokens_predicted_mean", "预测命中 Tokens", "agent"),
    ("navi_sched.agent.validation_window_cache_hit_tokens_actual_mean", "实际命中 Tokens", "agent"),
    ("navi_sched.agent.validation_cache_hit_tokens_error", "命中偏差", "agent"),
    ("navi_sched.agent.validation_cache_hit_tokens_absolute_error", "命中 MAE", "agent"),
    ("navi_sched.agent.validation_window_batch_request_count_mean", "平均请求数 / Batch", "agent"),
])
add_graph("单机 Cache Hit 四段对比", [
    ("navi_sched.agent.cache_hit_path_master_predicted_tokens", "Master 调度预测", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_arrival_tokens", "Agent 到达时物理快照", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_pending_aware_tokens", "Agent 到达时 Pending-aware 预测", "agent"),
    ("navi_sched.agent.cache_hit_path_rtp_actual_tokens", "RTP 实际复用", "agent"),
])
add_graph("单机 Cache Hit 最终预测 Bias", [
    ("navi_sched.agent.cache_hit_path_master_to_agent_pending_aware_error_tokens", "Master 到 Agent Pending-aware Bias", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_pending_aware_to_rtp_error_tokens", "Agent Pending-aware 到 RTP Bias", "agent"),
    ("navi_sched.agent.cache_hit_path_master_to_rtp_error_tokens", "Master 到 RTP 最终 Bias", "agent"),
])
add_graph("单机 Cache Hit 最终预测 MAE", [
    ("navi_sched.agent.cache_hit_path_master_to_agent_pending_aware_absolute_error_tokens", "Master 到 Agent Pending-aware MAE", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_pending_aware_to_rtp_absolute_error_tokens", "Agent Pending-aware 到 RTP MAE", "agent"),
    ("navi_sched.agent.cache_hit_path_master_to_rtp_absolute_error_tokens", "Master 到 RTP 最终 MAE", "agent"),
])
add_graph("单机 Cache 物理化时序 Bias", [
    ("navi_sched.agent.cache_hit_path_master_to_agent_error_tokens", "Master 到到达时物理快照 Bias", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_to_rtp_error_tokens", "到达时物理快照到 RTP Bias", "agent"),
])
add_graph("单机 Cache 物理化时序 MAE", [
    ("navi_sched.agent.cache_hit_path_master_to_agent_absolute_error_tokens", "Master 到到达时物理快照 MAE", "agent"),
    ("navi_sched.agent.cache_hit_path_agent_to_rtp_absolute_error_tokens", "到达时物理快照到 RTP MAE", "agent"),
])
add_graph("Cache 物理快照差值分桶", [
    ("navi_sched.agent.cache_hit_path_aligned_agent_to_rtp_error_tokens", "Block-aligned 物理化 Bias", "agent"),
    ("navi_sched.agent.cache_hit_path_aligned_agent_to_rtp_absolute_error_tokens", "Block-aligned 物理化 MAE", "agent"),
    ("navi_sched.agent.cache_hit_path_unaligned_agent_to_rtp_error_tokens", "Unaligned 物理化 Bias", "agent"),
    ("navi_sched.agent.cache_hit_path_unaligned_agent_to_rtp_absolute_error_tokens", "Unaligned 物理化 MAE", "agent"),
])
add_graph("Hybrid Cache 逐段复用 Block", [
    ("navi_sched.agent.hybrid_reuse_window_agent_predicted_blocks_mean", "Agent 到达时 flat 命中", "agent"),
    ("navi_sched.agent.hybrid_reuse_window_candidate_blocks_mean", "RTP Candidate", "agent"),
    ("navi_sched.agent.hybrid_reuse_window_flat_matched_blocks_mean", "RTP 执行时 flat 命中", "agent"),
    ("navi_sched.agent.hybrid_reuse_window_min_full_blocks_mean", "RTP minimum FULL match", "agent"),
    ("navi_sched.agent.hybrid_reuse_window_selected_blocks_mean", "RTP 最终 selected", "agent"),
])
add_graph("Hybrid Cache 两段预测损失", [
    ("navi_sched.agent.hybrid_reuse_window_agent_to_full_overprediction_blocks_mean", "Agent 到 FULL 高估", "agent"),
    ("navi_sched.agent.hybrid_reuse_window_tail_rollback_blocks_mean", "FULL 到 LINEAR/SWA tail 回退", "agent"),
])
add_graph("RTP 与 Agent Cache 镜像", [
    ("navi_sched.agent.cached_block_count", "Agent 镜像 Block 数", "agent"),
    ("navi_sched.agent.full_ready_block_count", "FULL-ready Block 数", "agent"),
    ("navi_sched.agent.tail_ready_block_count", "Tail-ready Block 数", "agent"),
    ("navi_sched.agent.rtp_cache_mirror_available", "RTP 镜像计数是否可用", "agent"),
    ("navi_sched.agent.rtp_cached_block_count", "RTP SharedBlockCache 数", "agent"),
    ("navi_sched.agent.rtp_cache_mirror_block_count_delta", "Agent - RTP", "agent"),
])

add_row("Agent - Execution Adam 更新诊断")
parameter_tag = {"parameter_index": "*"}
add_graph("Adam 每步原始梯度", [
    ("navi_sched.agent.latency_model_raw_gradient", "Raw Gradient", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步一阶矩", [
    ("navi_sched.agent.latency_model_adam_moment_1", "Moment 1", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步二阶矩", [
    ("navi_sched.agent.latency_model_adam_moment_2", "Moment 2", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步归一化梯度", [
    ("navi_sched.agent.latency_model_adam_normalized_gradient", "M1 / sqrt(M2)", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步参数更新：投影前", [
    ("navi_sched.agent.latency_model_adam_candidate_update", "Candidate Update", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步参数更新：实际", [
    ("navi_sched.agent.latency_model_actual_update", "Actual Update", "agent", "global_avg", "avg", parameter_tag),
])
add_graph("Adam 每步单参数预测扰动", [
    ("navi_sched.agent.latency_model_parameter_prediction_shift_ms", "Prediction Shift", "agent", "global_avg", "avg", parameter_tag),
], "ms")
add_graph("Adam 每秒更新步数", [
    ("navi_sched.agent.latency_model_optimizer_step_qps", "Optimizer Steps/s", "agent"),
], "ops")
add_graph("Execution 模型安全恢复 QPS", [
    (
        "navi_sched.agent.latency_model_unsafe_waiting_candidate_recovery_qps",
        "拒绝负 Waiting 成本候选",
        "agent",
        "sum",
    ),
    (
        "navi_sched.agent.latency_model_unsafe_waiting_state_reset_qps",
        "重置旧负 Waiting 成本状态",
        "agent",
        "sum",
    ),
])

add_row("Agent - 非 PD 统一预测诊断")
add_graph("单机 PD 模式与 Workload 上报", [
    ("navi_sched.agent.pd_separated", "PD 分离标记", "agent"),
    ("navi_sched.agent.workload_revision", "Workload Revision", "agent"),
    ("navi_sched.agent.workload_update_qps", "Workload 更新 QPS", "agent", "sum"),
])
add_graph("单机 Workload 时效", [
    ("navi_sched.agent.workload_age_ms", "本地 Workload Age", "agent"),
    ("navi_sched.agent.workload_publish_window_mean_ms", "上报调用平均耗时", "agent"),
    ("navi_sched.agent.workload_publish_window_max_ms", "上报调用最大耗时", "agent"),
], "ms")
add_graph("单机 Workload 上报样本", [
    ("navi_sched.agent.workload_publish_window_count", "1 秒窗口上报次数", "agent", "sum"),
])
add_graph("单机 Workload 请求状态", [
    ("navi_sched.agent.workload_batch_count", "Batch 数", "agent"),
    ("navi_sched.agent.workload_request_count", "全部请求", "agent"),
    ("navi_sched.agent.workload_remaining_prefill_request_count", "未完成首次 Prefill", "agent"),
    ("navi_sched.agent.workload_running_request_count", "vLLM Running", "agent"),
    ("navi_sched.agent.workload_waiting_request_count", "vLLM Waiting", "agent"),
])
add_graph("单机 Running 已运行时间", [
    ("navi_sched.agent.workload_running_elapsed_time_ms", "Running 时间总和", "agent"),
], "ms")
add_graph("单机 Running Decode 进度", [
    ("navi_sched.agent.workload_running_decode_request_count", "Decode 请求数", "agent"),
    ("navi_sched.agent.workload_running_decode_generated_tokens", "已生成 Decode Tokens", "agent"),
    ("navi_sched.agent.workload_running_decode_remaining_token_budget", "Max Tokens 预算减已生成（非真实剩余）", "agent"),
])
add_graph("单机剩余 Prefill Workload", [
    ("navi_sched.agent.workload_remaining_prefill_context_tokens", "Prefill Context Tokens", "agent"),
    ("navi_sched.agent.workload_remaining_prefill_tokens", "剩余 Prefill Tokens", "agent"),
])
add_graph("单机 Running/Waiting 容量", [
    ("navi_sched.agent.workload_max_running_request_count", "Running 容量上限", "agent"),
    ("navi_sched.agent.workload_running_capacity_slot_count", "已占 Running 槽位", "agent"),
    (
        "navi_sched.agent.workload_running_capacity_prefill_request_count",
        "Running Prefill 请求",
        "agent",
    ),
    (
        "navi_sched.agent.workload_waiting_capacity_prefill_request_count",
        "Waiting Prefill 请求",
        "agent",
    ),
])
add_graph("单机 Running/Waiting Prefill Tokens", [
    (
        "navi_sched.agent.workload_running_capacity_prefill_context_tokens",
        "Running Hit Tokens",
        "agent",
    ),
    (
        "navi_sched.agent.workload_running_capacity_prefill_tokens",
        "Running Compute Tokens",
        "agent",
    ),
    (
        "navi_sched.agent.workload_waiting_capacity_prefill_context_tokens",
        "Waiting Hit Tokens",
        "agent",
    ),
    (
        "navi_sched.agent.workload_waiting_capacity_prefill_tokens",
        "Waiting Compute Tokens",
        "agent",
    ),
])
add_graph("单机 Running/Waiting 二次与交叉特征", [
    (
        "navi_sched.agent.workload_running_capacity_prefill_token_square_sum",
        "Running Compute 平方和",
        "agent",
    ),
    (
        "navi_sched.agent.workload_running_capacity_prefill_product_sum",
        "Running Hit x Compute",
        "agent",
    ),
    (
        "navi_sched.agent.workload_waiting_capacity_prefill_token_square_sum",
        "Waiting Compute 平方和",
        "agent",
    ),
    (
        "navi_sched.agent.workload_waiting_capacity_prefill_product_sum",
        "Waiting Hit x Compute",
        "agent",
    ),
])
add_graph("单机剩余 Total Workload", [
    ("navi_sched.agent.workload_remaining_total_context_tokens", "Total Context Tokens", "agent"),
    ("navi_sched.agent.workload_remaining_total_tokens", "剩余 Total Tokens", "agent"),
    ("navi_sched.agent.workload_total_prompt_tokens", "在途 Prompt Tokens", "agent"),
    ("navi_sched.agent.workload_inflight_scheduled_tokens", "已调度未结算 Tokens", "agent"),
])
add_graph("单机 Prefill 二次与交叉特征", [
    ("navi_sched.agent.workload_remaining_prefill_token_square_sum", "剩余 Prefill Token 平方和", "agent"),
    ("navi_sched.agent.workload_context_remaining_prefill_product_sum", "Context x 剩余 Prefill", "agent"),
])
add_graph("单机 Total 二次与交叉特征", [
    ("navi_sched.agent.workload_remaining_total_token_square_sum", "剩余 Total Token 平方和", "agent"),
    ("navi_sched.agent.workload_context_remaining_total_product_sum", "Context x 剩余 Total", "agent"),
])

add_row("Master - 集群状态与调度流量")
add_graph("Master 梯度下降实际轮数", [
    ("navi_sched.master.cost_optimizer_loop_count", "实际梯度迭代轮数", "master", "global_avg"),
    ("navi_sched.master.cost_optimizer_max_loop_count", "配置最大梯度迭代轮数", "master", "global_avg"),
])
add_graph("Master 梯度下降预算停止", [
    ("navi_sched.master.cost_optimizer_time_budget_exhausted", "因预算停止（1 是 / 0 否）", "master", "global_avg"),
    ("navi_sched.master.cost_optimizer_time_budget_stop_qps", "因预算停止轮次 QPS", "master", "sum"),
])
add_graph("Master 单轮梯度计算耗时（不含离散搜索与排队）", [
    ("navi_sched.master.cost_optimizer_gradient_iteration_ms", "单轮梯度计算平均耗时", "master", "global_avg"),
    ("navi_sched.master.cost_optimizer_gradient_iteration_max_ms", "单轮梯度计算最大耗时", "master", "global_avg"),
], "ms")
add_graph("Master Replay 采集累计耗时", [
    ("navi_sched.master.schedule_replay_collection_ms", "从优化预算扣除的采集耗时", "master", "global_avg"),
], "ms")
add_graph("Master 入口与调度 QPS", [
    ("navi_sched.master.client_request_received_qps", "入口请求 QPS", "master", "sum"),
    ("navi_sched.master.schedule_round_qps", "调度轮次 QPS", "master", "sum"),
    ("navi_sched.master.schedule_request_qps", "调度消费请求 QPS", "master", "sum"),
])
add_graph("Master Queue 过载限流 QPS", [
    (
        "navi_sched.master.queue_overload_evaluated_request_qps",
        "参与过载判断 Request QPS",
        "master",
        "sum",
    ),
    (
        "navi_sched.master.queue_overload_rejected_request_qps",
        "429 限流 Request QPS",
        "master",
        "sum",
    ),
])
add_graph("Master Queue 过载限流状态", [
    (
        "navi_sched.master.queue_overload_average_predicted_latency_ms",
        "可信节点在途 Request 平均预测延迟",
        "master",
    ),
], "ms")
add_graph("Master Queue 过载限流概率", [
    (
        "navi_sched.master.queue_overload_drop_probability",
        "当前拒绝概率",
        "master",
    ),
], "percentunit")
add_graph("Master 冷启动路由 QPS", [
    (
        "navi_sched.master.cold_start_trusted_request_qps",
        "进入可信模型 Cost 调度",
        "master",
        "sum",
    ),
    (
        "navi_sched.master.cold_start_untrusted_request_qps",
        "按 Inflight 投递冷节点",
        "master",
        "sum",
    ),
    (
        "navi_sched.master.cold_start_rejected_request_qps",
        "冷节点容量耗尽 429",
        "master",
        "sum",
    ),
    (
        "navi_sched.master.cold_start_unsafe_waiting_marginal_node_observation_qps",
        "负 Waiting 边际节点观测 QPS",
        "master",
        "sum",
    ),
])
add_graph("Master 冷启动节点与容量", [
    (
        "navi_sched.master.latency_model_trusted_node_count",
        "可信模型节点数",
        "master",
    ),
    (
        "navi_sched.master.latency_model_untrusted_node_count",
        "未可信模型节点数",
        "master",
    ),
    (
        "navi_sched.master.cold_start_latest_target_inflight",
        "上一轮可信节点平均 Inflight 目标",
        "master",
    ),
    (
        "navi_sched.master.cold_start_latest_untrusted_capacity",
        "上一轮冷节点可接收容量",
        "master",
    ),
    (
        "navi_sched.master.cold_start_latest_unsafe_waiting_marginal_node_count",
        "上一轮负 Waiting 边际节点数",
        "master",
    ),
])
add_graph("Master 模型公式共享状态", [
    (
        "navi_sched.master.latency_model_pool_warmup_subscriber_count",
        "等待候选的节点数",
        "master",
    ),
    (
        "navi_sched.master.latency_model_pool_fingerprint_count",
        "公式指纹数",
        "master",
    ),
    (
        "navi_sched.master.latency_model_pool_candidate_count",
        "持久化候选数",
        "master",
    ),
    (
        "navi_sched.master.latency_model_global_zero_point_live_node_count",
        "在线模型节点数",
        "master",
    ),
])
add_graph("Master 模型公式共享累计事件", [
    (
        "navi_sched.master.latency_model_pool_warmup_notification_total",
        "候选通知",
        "master",
    ),
    (
        "navi_sched.master.latency_model_pool_warmup_response_total",
        "候选下发",
        "master",
    ),
    (
        "navi_sched.master.latency_model_pool_warmup_notification_failure_total",
        "候选通知失败",
        "master",
    ),
    (
        "navi_sched.master.latency_model_pool_persist_failure_total",
        "公式池落盘失败",
        "master",
    ),
])
add_graph("Agent 反馈 QPS（Master 视角）", [
    ("navi_sched.master.agent_request_received_qps", "请求接收 QPS", "master", "sum"),
    ("navi_sched.master.agent_request_finished_qps", "请求完成 QPS", "master", "sum"),
    ("navi_sched.master.agent_batch_finished_qps", "Batch 完成 QPS", "master", "sum"),
    ("navi_sched.master.agent_request_abnormal_finished_qps", "异常完成 QPS", "master", "sum"),
])
add_graph("Master 当前请求数量", [
    ("navi_sched.master.requests_active", "Master 跟踪未结束请求总数", "master"),
    ("navi_sched.master.schedule_pending_requests", "等待调度", "master"),
    ("navi_sched.master.schedule_processing_requests", "正在计算调度", "master"),
    ("navi_sched.master.reservation_active_request_count", "已调度尚未结束", "master"),
    ("navi_sched.master.requests_inflight", "Agent 已接收", "master"),
    ("navi_sched.master.requests_queued_for_execution", "等待 GPU 执行", "master"),
    ("navi_sched.master.requests_running", "GPU 执行中", "master"),
])
add_graph("Master 调度图生命周期", [
    ("navi_sched.master.schedule_graph_running", "运行状态", "master"),
    ("navi_sched.master.schedule_graph_restart_total", "异常重启次数", "master"),
    (
        "navi_sched.master.schedule_request_pipe_generation",
        "Request Pipe 代数",
        "master",
    ),
])
add_graph("Master Reservation 生命周期", [
    (
        "navi_sched.master.reservation_batch_count",
        "未完成 Reservation Batch",
        "master",
    ),
    (
        "navi_sched.master.reservation_waiting_request_count",
        "等待 Agent 确认 Request",
        "master",
    ),
    (
        "navi_sched.master.reservation_dispatched_request_count",
        "已确认未完成 Request",
        "master",
    ),
    ("navi_sched.master.dispatch_waiter_count", "Dispatch Waiter", "master"),
    (
        "navi_sched.master.live_dispatch_waiter_count",
        "存活 Dispatch Waiter",
        "master",
    ),
    (
        "navi_sched.master.pending_cache_batch_count",
        "Pending Cache Batch",
        "master",
    ),
])
add_graph("Master 与 Agent 在线状态", [
    ("navi_sched.role.alive", "Master 存活", "master"),
    ("navi_sched.master.collector_count", "已订阅 Agent 数", "master"),
    ("navi_sched.master.online_node_count", "在线 Agent 数", "master"),
])
add_graph("Agent 事件到 Master 应用延迟", [
    ("navi_sched.master.agent_report_lag_mean_ms", "各 Agent 最新事件延迟均值", "master"),
    ("navi_sched.master.agent_report_lag_max_ms", "各 Agent 最新事件延迟最大值", "master"),
], "ms")
add_graph("Master Workload 节点状态", [
    ("navi_sched.master.workload_ready_node_count", "Workload Ready 节点", "master"),
    ("navi_sched.master.pd_separated_node_count", "PD 分离节点", "master"),
    ("navi_sched.master.agent_workload_update_received_qps", "Workload 更新接收 QPS", "master", "sum"),
    ("navi_sched.master.workload_stale_update_total", "拒绝旧 Revision 累计", "master"),
])
add_graph("Master Workload 时效", [
    ("navi_sched.master.workload_age_mean_ms", "Agent 事件 Age 均值", "master"),
    ("navi_sched.master.workload_age_min_ms", "Agent 事件 Age 最小值", "master"),
    ("navi_sched.master.workload_age_max_ms", "Agent 事件 Age 最大值", "master"),
    ("navi_sched.master.workload_transport_lag_mean_ms", "Agent 到 Master 传输均值", "master"),
    ("navi_sched.master.workload_transport_lag_min_ms", "Agent 到 Master 传输最小值", "master"),
    ("navi_sched.master.workload_transport_lag_max_ms", "Agent 到 Master 传输最大值", "master"),
], "ms")
add_graph("Master 聚合 Workload 请求状态", [
    ("navi_sched.master.workload_batch_count", "Batch 数", "master"),
    ("navi_sched.master.workload_request_count", "全部请求", "master"),
    ("navi_sched.master.workload_remaining_prefill_request_count", "未完成首次 Prefill", "master"),
    ("navi_sched.master.workload_running_request_count", "vLLM Running", "master"),
    ("navi_sched.master.workload_waiting_request_count", "vLLM Waiting", "master"),
])
add_graph("Master 聚合 Running 已运行时间", [
    ("navi_sched.master.workload_running_elapsed_time_ms", "Running 时间总和", "master"),
], "ms")
add_graph("Master 聚合 Running Decode 进度", [
    ("navi_sched.master.workload_running_decode_request_count", "Decode 请求数", "master"),
    ("navi_sched.master.workload_running_decode_generated_tokens", "已生成 Decode Tokens", "master"),
    ("navi_sched.master.workload_running_decode_remaining_token_budget", "Max Tokens 预算减已生成（非真实剩余）", "master"),
])
add_graph("Master 聚合剩余 Prefill Workload", [
    ("navi_sched.master.workload_remaining_prefill_context_tokens", "Prefill Context Tokens", "master"),
    ("navi_sched.master.workload_remaining_prefill_tokens", "剩余 Prefill Tokens", "master"),
    ("navi_sched.master.workload_remaining_prefill_token_square_sum", "剩余 Prefill Token 平方和", "master"),
    ("navi_sched.master.workload_context_remaining_prefill_product_sum", "Context x 剩余 Prefill", "master"),
])
add_graph("Master 聚合 Running/Waiting 容量", [
    ("navi_sched.master.workload_capacity_split_node_count", "启用容量分段节点", "master"),
    ("navi_sched.master.workload_max_running_request_count_sum", "Running 容量总和", "master"),
    ("navi_sched.master.workload_running_capacity_slot_count", "已占 Running 槽位", "master"),
    (
        "navi_sched.master.workload_running_capacity_prefill_request_count",
        "Running Prefill 请求",
        "master",
    ),
    (
        "navi_sched.master.workload_waiting_capacity_prefill_request_count",
        "Waiting Prefill 请求",
        "master",
    ),
])
add_graph("Master 聚合 Running/Waiting Prefill Tokens", [
    (
        "navi_sched.master.workload_running_capacity_prefill_context_tokens",
        "Running Hit Tokens",
        "master",
    ),
    (
        "navi_sched.master.workload_running_capacity_prefill_tokens",
        "Running Compute Tokens",
        "master",
    ),
    (
        "navi_sched.master.workload_waiting_capacity_prefill_context_tokens",
        "Waiting Hit Tokens",
        "master",
    ),
    (
        "navi_sched.master.workload_waiting_capacity_prefill_tokens",
        "Waiting Compute Tokens",
        "master",
    ),
])
add_graph("Master 聚合 Running/Waiting 二次与交叉特征", [
    (
        "navi_sched.master.workload_running_capacity_prefill_token_square_sum",
        "Running Compute 平方和",
        "master",
    ),
    (
        "navi_sched.master.workload_running_capacity_prefill_product_sum",
        "Running Hit x Compute",
        "master",
    ),
    (
        "navi_sched.master.workload_waiting_capacity_prefill_token_square_sum",
        "Waiting Compute 平方和",
        "master",
    ),
    (
        "navi_sched.master.workload_waiting_capacity_prefill_product_sum",
        "Waiting Hit x Compute",
        "master",
    ),
])
add_graph("Master 聚合剩余 Total Workload", [
    ("navi_sched.master.workload_remaining_total_context_tokens", "Total Context Tokens", "master"),
    ("navi_sched.master.workload_remaining_total_tokens", "剩余 Total Tokens", "master"),
    ("navi_sched.master.workload_total_prompt_tokens", "在途 Prompt Tokens", "master"),
    ("navi_sched.master.workload_inflight_scheduled_tokens", "已调度未结算 Tokens", "master"),
])
add_graph("Master 聚合 Total 二次与交叉特征", [
    ("navi_sched.master.workload_remaining_total_token_square_sum", "剩余 Total Token 平方和", "master"),
    ("navi_sched.master.workload_context_remaining_total_product_sum", "Context x 剩余 Total", "master"),
])
add_graph("Cache 更新链路一致性", [
    ("navi_sched.master.cache_sync_diverged_node_count", "Agent/Master 不一致节点", "master"),
    ("navi_sched.master.rtp_cache_mirror_diverged_node_count", "RTP/Agent 不一致节点", "master"),
    ("navi_sched.master.rtp_cache_mirror_mismatch_ratio", "RTP/Agent mismatch ratio", "master"),
])
add_graph("Pending Cache 重同步兜底", [
    ("navi_sched.master.pending_cache_sync_resync_trigger_total", "重同步触发累计", "master"),
    ("navi_sched.master.pending_cache_sync_revision_gap_total", "Revision Gap 累计", "master"),
    ("navi_sched.master.pending_cache_sync_state_mismatch_total", "状态校验失败累计", "master"),
    ("navi_sched.master.pending_cache_sync_not_ready_node_count", "等待快照节点", "master"),
])

add_row("Master - Batch 特征与预测精度")
add_graph("实际执行 Batch 大小", [
    ("navi_sched.master.validation_window_batch_request_count_mean", "最近 1 秒平均请求数 / Batch", "master"),
])
add_graph("Cache Hit Token 预测与实际", [
    ("navi_sched.master.validation_window_cache_hit_tokens_predicted_mean", "调度预测命中 Tokens", "master"),
    ("navi_sched.master.validation_window_cache_hit_tokens_actual_mean", "执行实际命中 Tokens", "master"),
])
add_graph("Queue 预测值与真值", [
    ("navi_sched.master.validation_window_queue_predicted_mean_ms", "调度预测 Queue", "master"),
    ("navi_sched.master.validation_window_queue_actual_mean_ms", "实际 Queue", "master"),
], "ms")
add_graph("Queue 预测误差", [
    ("navi_sched.agent.validation_queue_error_ms", "静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_queue_absolute_error_ms", "逐 Batch MAE", "agent"),
], "ms")
add_graph("Agent Queue 模型：Raw / 拟合预测", [
    ("navi_sched.master.queue_model_window_raw_prediction_mean_ms", "Raw 前序时延总和", "master"),
    ("navi_sched.master.queue_model_window_prediction_mean_ms", "Agent 拟合预测", "master"),
    ("navi_sched.master.validation_window_queue_actual_mean_ms", "实际 Queue", "master"),
], "ms")
add_graph("Agent Queue 模型：Raw / 拟合误差", [
    ("navi_sched.agent.queue_model_raw_error_ms", "Raw 静态偏差", "agent"),
    ("navi_sched.agent.queue_model_raw_absolute_error_ms", "Raw MAE", "agent"),
    ("navi_sched.agent.queue_model_error_ms", "拟合静态偏差", "agent"),
    ("navi_sched.agent.queue_model_absolute_error_ms", "拟合 MAE", "agent"),
], "ms")
add_graph("预测 Queue 为 0 的现场", [
    ("navi_sched.master.validation_window_queue_zero_actual_mean_ms", "实际 Queue", "master"),
    ("navi_sched.agent.validation_queue_zero_absolute_error_ms", "MAE", "agent"),
], "ms")
add_graph("预测 Queue 大于 0 的现场", [
    ("navi_sched.master.validation_window_queue_positive_predicted_mean_ms", "预测 Queue", "master"),
    ("navi_sched.master.validation_window_queue_positive_actual_mean_ms", "实际 Queue", "master"),
    ("navi_sched.agent.validation_queue_positive_error_ms", "静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_queue_positive_absolute_error_ms", "MAE", "agent"),
], "ms")
add_graph("Execution 预测值与真值", [
    ("navi_sched.master.validation_window_execution_predicted_mean_ms", "Master 调度前预测", "master"),
    ("navi_sched.master.validation_window_execution_agent_predicted_mean_ms", "Agent 更新前预测", "master"),
    ("navi_sched.master.validation_window_execution_actual_mean_ms", "实际执行", "master"),
], "ms")
add_graph("Execution 预测误差", [
    ("navi_sched.agent.validation_execution_master_error_ms", "Master 静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_execution_master_absolute_error_ms", "Master 逐 Batch MAE", "agent"),
    ("navi_sched.agent.validation_execution_agent_error_ms", "Agent 静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_execution_agent_absolute_error_ms", "Agent 逐 Batch MAE", "agent"),
], "ms")
add_graph("端到端预测值与真值", [
    ("navi_sched.master.validation_window_total_predicted_mean_ms", "预测 Queue + Execution", "master"),
    ("navi_sched.master.validation_window_total_actual_mean_ms", "实际 Queue + Execution", "master"),
], "ms")
add_graph("端到端预测误差", [
    ("navi_sched.agent.validation_total_error_ms", "静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_total_absolute_error_ms", "逐 Batch MAE", "agent"),
], "ms")
add_graph("Execution 与端到端相对误差", [
    ("navi_sched.agent.validation_execution_master_absolute_percentage_error_pct", "Execution MAPE", "agent"),
    ("navi_sched.agent.validation_execution_agent_absolute_percentage_error_pct", "Agent Execution MAPE", "agent"),
    ("navi_sched.agent.validation_total_absolute_percentage_error_pct", "端到端 MAPE", "agent"),
], "percent")
add_graph("Cache Hit 偏差与 MAE", [
    ("navi_sched.agent.validation_cache_hit_tokens_error", "静态偏差（实际-预测）", "agent"),
    ("navi_sched.agent.validation_cache_hit_tokens_absolute_error", "逐 Batch MAE", "agent"),
])

add_row("Client - 调度与 Agent 接收确认链路")
add_graph("Client 存活", [
    ("navi_sched.role.alive", "Client 存活", "client"),
])
add_graph("Client 入口与调度接收确认 QPS", [
    ("navi_sched.client.request_received_qps", "进入 Navi Client", "client", "sum"),
    ("navi_sched.client.master_request_qps", "提交 Master", "client", "sum"),
    ("navi_sched.client.master_success_qps", "Master 成功", "client", "sum"),
    ("navi_sched.client.master_error_qps", "Master 失败", "client", "sum"),
    ("navi_sched.client.master_rate_limited_qps", "收到 Master 限流拒绝", "client", "sum"),
])
add_graph("Client 本机回退 QPS", [
    ("navi_sched.client.master_error_qps", "Master 调用失败", "client", "sum"),
    ("navi_sched.client.local_fallback_qps", "回退本机 EngineCore", "client", "sum"),
    ("navi_sched.client.master_success_qps", "Master 调用恢复", "client", "sum"),
])
add_graph("Client 调度与 Agent 接收确认延迟", [
    ("navi_sched.client.master_request_latency_ms", "全部调用", "client"),
    ("navi_sched.client.master_success_latency_ms", "成功调用", "client"),
    ("navi_sched.client.master_error_latency_ms", "失败调用", "client"),
], "ms")
add_graph("Client 进入 Master 前失败 QPS", [
    ("navi_sched.client.invalid_request_qps", "参数无效", "client", "sum"),
    ("navi_sched.client.stopped_request_qps", "Client 已停止", "client", "sum"),
])
add_graph("Client 调度与 Agent 接收确认失败分类 QPS", [
    ("navi_sched.client.master_timeout_qps", "调度或 Agent 接收确认超时", "client", "sum"),
    ("navi_sched.client.master_transport_error_qps", "传输错误", "client", "sum"),
    ("navi_sched.client.master_routing_error_qps", "路由错误", "client", "sum"),
    ("navi_sched.client.master_no_decision_qps", "无调度结果", "client", "sum"),
    ("navi_sched.client.master_other_error_qps", "其他错误", "client", "sum"),
])
add_graph("Client 调度与 Agent 接收确认失败分类延迟", [
    ("navi_sched.client.master_timeout_latency_ms", "调度或 Agent 接收确认超时", "client"),
    (
        "navi_sched.client.master_transport_error_latency_ms",
        "传输错误",
        "client",
    ),
    (
        "navi_sched.client.master_routing_error_latency_ms",
        "路由错误",
        "client",
    ),
    (
        "navi_sched.client.master_no_decision_latency_ms",
        "无调度结果",
        "client",
    ),
    ("navi_sched.client.master_other_error_latency_ms", "其他错误", "client"),
], "ms")

dashboard = {
    "id": None,
    "uid": "navi-sched-online",
    "title": "NaviSched 线上调度与预测精度",
    "tags": ["NaviSched", "RTP-LLM", "vLLM"],
    "timezone": "browser",
    "schemaVersion": 30,
    "version": 0,
    "graphTooltip": 1,
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "templating": {
        "list": [{
            "name": "app",
            "label": "Whale App",
            "type": "query",
            "query": "suggest_tagv(navi_sched.role.alive,hippo_app)(1h-ago)(default)",
            "definition": "suggest_tagv(navi_sched.role.alive,hippo_app)(1h-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 0,
            "regex": "",
            "multi": False,
            "includeAll": True,
            "allValue": "None",
            "current": {
                "selected": True,
                "text": DEFAULT_HIPPO_APP,
                "value": DEFAULT_HIPPO_APP,
            },
            "options": [
                {"selected": False, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
                {"selected": True, "text": DEFAULT_HIPPO_APP, "value": DEFAULT_HIPPO_APP},
            ],
        }, {
            "name": "deployment",
            "label": "Agent Deployment",
            "type": "query",
            "query": "suggest_tagv(navi_sched.role.alive,deployment_name,role=agent,hippo_app=[[app]])(1h-ago)(default)",
            "definition": "suggest_tagv(navi_sched.role.alive,deployment_name,role=agent,hippo_app=[[app]])(1h-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 0,
            "regex": "",
            "multi": False,
            "includeAll": True,
            "allValue": "None",
            "current": {"selected": True, "text": "All", "value": "$__all"},
            "options": [
                {"selected": True, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
            ],
        }, {
            "name": "master_deployment",
            "label": "Master Deployment",
            "type": "query",
            "query": "suggest_tagv(navi_sched.role.alive,deployment_name,role=master,hippo_app=[[app]])(1h-ago)(default)",
            "definition": "suggest_tagv(navi_sched.role.alive,deployment_name,role=master,hippo_app=[[app]])(1h-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 0,
            "regex": "",
            "multi": False,
            "includeAll": True,
            "allValue": "None",
            "current": {"selected": True, "text": "All", "value": "$__all"},
            "options": [
                {"selected": True, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
            ],
        }, {
            "name": "host",
            "label": "Host",
            "type": "query",
            "query": "suggest_tagv(navi_sched.agent.latency_model_revision,host_ip,hippo_app=[[app]],deployment_name=[[deployment]])(5m-ago)(default)",
            "definition": "suggest_tagv(navi_sched.agent.latency_model_revision,host_ip,hippo_app=[[app]],deployment_name=[[deployment]])(5m-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 0,
            "regex": "",
            "multi": True,
            "includeAll": True,
            "allValue": "None",
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
            "options": [
                {"selected": True, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
            ],
        }, {
            "name": "bucket",
            "label": "Execution Bucket",
            "type": "query",
            "query": "suggest_tagv(navi_sched.agent.validation_execution_bucket_enabled,bucket_id,hippo_app=[[app]],deployment_name=[[deployment]],host_ip=[[host]])(5m-ago)(default)",
            "definition": "suggest_tagv(navi_sched.agent.validation_execution_bucket_enabled,bucket_id,hippo_app=[[app]],deployment_name=[[deployment]],host_ip=[[host]])(5m-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 3,
            "regex": "",
            "multi": True,
            "includeAll": True,
            "allValue": "None",
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
            "options": [
                {"selected": True, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
            ],
        }, {
            "name": "residual_bucket",
            "label": "Residual Bucket",
            "type": "query",
            "query": "suggest_tagv(navi_sched.master.latency_residual_parameter_ms,residual_bucket,hippo_app=[[app]],deployment_name=[[master_deployment]],host_ip=[[host]])(5m-ago)(default)",
            "definition": "suggest_tagv(navi_sched.master.latency_residual_parameter_ms,residual_bucket,hippo_app=[[app]],deployment_name=[[master_deployment]],host_ip=[[host]])(5m-ago)(default)",
            "datasource": None,
            "refresh": 1,
            "sort": 3,
            "regex": "",
            "multi": True,
            "includeAll": True,
            "allValue": "None",
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
            "options": [
                {"selected": True, "text": "All", "value": "$__all"},
                {"selected": False, "text": "*", "value": "*"},
            ],
        }]
    },
    "panels": panels,
}


def validate_dashboard(value):
    variables = value["templating"]["list"]
    if [variable["name"] for variable in variables] != [
        "app",
        "deployment",
        "master_deployment",
        "host",
        "bucket",
        "residual_bucket",
    ]:
        raise ValueError(
            "dashboard must expose agent/master deployment, host and bucket selectors"
        )
    app, deployment, master_deployment, host, bucket, residual_bucket = variables
    for variable in variables:
        if variable.get("allValue") != "None":
            raise ValueError(
                f"{variable['name']} All must aggregate without expanding tags"
            )
        if variable is not app and sum(
            option.get("text") == "All" and option.get("value") == "$__all"
            for option in variable.get("options", [])
        ) != 1:
            raise ValueError(f"{variable['name']} selector must expose one All option")
        wildcard_options = [
            option for option in variable.get("options", [])
            if option.get("text") == "*" and option.get("value") == "*"
        ]
        if len(wildcard_options) != 1:
            raise ValueError(
                f"{variable['name']} selector must expose one wildcard option"
            )
        current_text = variable.get("current", {}).get("text")
        if variable is app:
            if variable.get("current", {}).get("value") != DEFAULT_HIPPO_APP:
                raise ValueError("app selector must default to the current Whale app")
        elif current_text != "All" and current_text != ["All"]:
            raise ValueError(f"{variable['name']} selector must default to All")
    for variable in variables[1:]:
        if "hippo_app=[[app]]" not in variable.get("query", ""):
            raise ValueError(f"{variable['name']} selector must filter by app")
    if "deployment_name=[[deployment]]" not in host.get("query", ""):
        raise ValueError("host selector must follow agent deployment")
    if "deployment_name=[[deployment]]" not in bucket.get("query", "") or \
            "host_ip=[[host]]" not in bucket.get("query", ""):
        raise ValueError("execution bucket selector must follow deployment and host")
    if "deployment_name=[[master_deployment]]" not in residual_bucket.get("query", "") or \
            "host_ip=[[host]]" not in residual_bucket.get("query", ""):
        raise ValueError("residual bucket selector must follow master and host")

    graph_panels = []
    panel_ids = set()
    for section in value["panels"]:
        if section.get("type") != "row":
            raise ValueError(
                f"dashboard section must be a row, got {section.get('type')!r}"
            )
        if not section.get("collapsed"):
            raise ValueError(f"dashboard row must be collapsed: {section.get('title')!r}")
        section_panels = section.get("panels", [])
        if not section_panels:
            raise ValueError(f"dashboard row is empty: {section.get('title')!r}")
        for panel in section_panels:
            if panel.get("type") != "graph":
                raise ValueError(
                    f"dashboard row contains a non-graph panel: {panel.get('title')!r}"
                )
            panel_id = panel.get("id")
            if panel_id in panel_ids:
                raise ValueError(
                    f"dashboard graph belongs to multiple rows: {panel.get('title')!r}"
                )
            panel_ids.add(panel_id)
            graph_panels.append(panel)

    target_count = 0
    metrics = set()
    for panel in graph_panels:
        for item in panel.get("targets", []):
            target_count += 1
            metric = item.get("metric", "")
            if not metric.startswith("navi_sched."):
                raise ValueError(f"invalid NaviSched metric: {metric!r}")
            metrics.add(metric)
            tags = item.get("tags", {})
            if tags.get("hippo_app") != "[[app]]":
                raise ValueError(f"metric does not use app selector: {metric}")
            role = tags.get("role")
            if role not in ("agent", "master", "client"):
                raise ValueError(f"invalid metric role for {metric}: {role!r}")
            expected_deployment = (
                "[[master_deployment]]" if role == "master" else "[[deployment]]"
            )
            if tags.get("deployment_name") != expected_deployment:
                raise ValueError(
                    f"metric does not use its role deployment selector: {metric}"
                )
            if role in ("agent", "client"):
                if tags.get("host_ip") != "[[host]]":
                    raise ValueError(f"{role} metric does not use host selector: {metric}")
                if "$tag_host_ip" not in item.get("alias", ""):
                    raise ValueError(f"{role} alias does not identify host: {metric}")
            elif "host_ip" in tags:
                if not metric.startswith("navi_sched.master.latency_residual_"):
                    raise ValueError(
                        f"master metric is filtered by inference host: {metric}"
                    )
                if tags.get("host_ip") != "[[host]]":
                    raise ValueError(
                        f"master residual metric does not use host selector: {metric}"
                    )
            if "$tag_deployment_name" not in item.get("alias", ""):
                raise ValueError(f"metric alias does not identify deployment: {metric}")
            if item.get("alias", "").startswith("$tag_"):
                raise ValueError(f"metric alias must put its name before dimensions: {metric}")
            if "process_id" in tags:
                raise ValueError(f"metric still splits series by process id: {metric}")
    if target_count == 0 or not metrics:
        raise ValueError("dashboard contains no NaviSched metrics")
    required_model_parameters = {
        f"navi_sched.agent.latency_model_parameter_{index}"
        for index in range(CAPACITY_SPLIT_PARAMETER_COUNT)
    }
    missing_model_parameters = required_model_parameters - metrics
    if missing_model_parameters:
        raise ValueError(
            "dashboard is missing latency model parameters: "
            + ", ".join(sorted(missing_model_parameters))
        )
    return target_count, len(metrics)


target_count, metric_count = validate_dashboard(dashboard)
payload = {"dashboard": dashboard, "folderId": 0, "overwrite": True, "message": "Configure NaviSched role metrics"}
args.output.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

if args.generate_only:
    print(
        json.dumps(
            {
                "output": str(args.output),
                "panels": len(panels),
                "targets": target_count,
                "unique_metrics": metric_count,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0)

if not args.cookies.is_file():
    raise SystemExit(f"KMonitor cookie file is missing: {args.cookies}")

completed = subprocess.run(
    [
        "curl", "-fsS",
        "--connect-timeout", "5",
        "--max-time", "30",
        "-b", str(args.cookies),
        "-c", str(args.cookies),
        "-H", "Content-Type: application/json",
        "-H", "Accept: application/json",
        "-H", "Origin: " + BASE_URL,
        "--data-binary", "@" + str(args.output),
        "--write-out", "\n%{http_code}\n%{redirect_url}",
        BASE_URL + "/api/dashboards/db",
    ],
    check=True,
    stdout=subprocess.PIPE,
    universal_newlines=True,
)
response_body, status_code, redirect_url = completed.stdout.rsplit("\n", 2)
if status_code != "200":
    detail = f", redirect={redirect_url}" if redirect_url else ""
    raise SystemExit(f"KMonitor dashboard upload failed: HTTP {status_code}{detail}")
if not response_body:
    raise SystemExit("KMonitor dashboard upload returned an empty response")
result = json.loads(response_body)

print(json.dumps(result, ensure_ascii=False))
print(BASE_URL + result["url"] + "?orgId=1")

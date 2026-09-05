#!/usr/bin/env python3
"""Generate a factual BlockTreeCache benchmark report from one suite manifest."""

import argparse
import html
import json
import math
import os
import shlex
import statistics
import sys
from typing import Dict, Iterable, List, Tuple


def summarize(values: Iterable[float]) -> Dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {}
    median = statistics.median(samples)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in samples),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def human_number(value: float) -> str:
    """Format a number for humans: no scientific notation, thousands separators."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6:
        decimals = 0
    elif magnitude >= 1e3:
        decimals = 1
    elif magnitude >= 1:
        decimals = 3
    else:
        decimals = max(3, 2 - int(math.floor(math.log10(magnitude))))
    formatted = f"{value:,.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def format_duration(ns: float) -> str:
    """Format a nanosecond duration with an adaptive, human-readable unit."""
    if ns >= 1e9:
        return f"{human_number(ns / 1e9)} s"
    if ns >= 1e6:
        return f"{human_number(ns / 1e6)} ms"
    if ns >= 1e3:
        return f"{human_number(ns / 1e3)} us"
    return f"{human_number(ns)} ns"


def format_summary(
    summary: Dict[str, float],
    scale: float = 1.0,
    suffix: str = "",
    duration: bool = False,
) -> str:
    if not summary:
        return "-"
    render = format_duration if duration else lambda value: human_number(value * scale)
    if summary["n"] == 1:
        return f"{render(summary['median'])}{suffix}"
    return (
        f"{render(summary['median'])}{suffix} "
        f"(MAD {render(summary['mad'])}, "
        f"{render(summary['min'])}..{render(summary['max'])}, "
        f"n={summary['n']})"
    )


def load_json(path: str) -> Dict:
    with open(path) as source:
        return json.load(source)


def load_manifest_results(
    profile_dir: str, suite_manifest: Dict
) -> List[Tuple[Dict, List[Dict]]]:
    """Load only repetitions explicitly marked valid in this suite manifest."""
    cases = []
    environment = suite_manifest.get("environment", {})
    suite_provenance = {
        "binary_sha256": environment.get("binary_sha256"),
        "code_commit": environment.get("code_commit"),
    }
    for case_manifest in suite_manifest.get("cases", []):
        results = []
        for repetition in case_manifest.get("repetitions", []):
            if not repetition.get("valid") or repetition.get("status") != "completed":
                continue
            result_path = repetition.get("result_json")
            if not result_path:
                continue
            if not os.path.isabs(result_path):
                result_path = os.path.join(profile_dir, result_path)
            result = load_json(result_path)
            if result.get("status") == "completed":
                # Keep suite provenance and manifest identity alongside the
                # native result for task-pool unique-variable validation.
                result["_suite_provenance"] = suite_provenance
                result["_manifest_identity"] = {
                    "seed": repetition.get("seed"),
                    "repetition": repetition.get("repetition"),
                }
                results.append(result)
        cases.append((case_manifest, results))
    return cases


def metric_summary(results: List[Dict], key: str) -> Dict[str, float]:
    return summarize(
        result.get("metrics", {}).get(key)
        for result in results
        if key in result.get("metrics", {})
    )


def phase_summary(results: List[Dict], key: str) -> Dict[str, float]:
    return summarize(
        result.get("phases_ns", {}).get(key)
        for result in results
        if key in result.get("phases_ns", {})
    )


def section_summary(results: List[Dict], section: str, key: str) -> Dict[str, float]:
    return summarize(
        result.get(section, {}).get(key)
        for result in results
        if key in result.get(section, {})
    )


def read_vmstat(path: str) -> Dict[str, int]:
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as source:
        for line in source:
            key, value = line.split()
            values[key] = int(value)
    return values


def vmstat_deltas(case_manifest: Dict, key: str) -> Dict[str, float]:
    deltas = []
    for repetition in case_manifest.get("repetitions", []):
        if not repetition.get("valid"):
            continue
        result_path = repetition.get("result_json")
        if not result_path:
            continue
        rep_dir = os.path.dirname(result_path)
        before = read_vmstat(os.path.join(rep_dir, "vmstat_before.txt"))
        after = read_vmstat(os.path.join(rep_dir, "vmstat_after.txt"))
        if key in before and key in after:
            deltas.append(after[key] - before[key])
    return summarize(deltas)


def common_value(results: List[Dict], section: str, key: str) -> str:
    values = {str(result.get(section, {}).get(key, "-")) for result in results}
    return values.pop() if len(values) == 1 else "mixed"


def escape(value) -> str:
    return html.escape(str(value), quote=True)


def table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="muted">本次专项报告无此类 case</td></tr>'
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def status_badge(status: str) -> str:
    css_class = "ok" if status == "completed" else "fail"
    return f'<span class="pill {css_class}">{escape(status)}</span>'


def perf_artifact_link(case_manifest: Dict, artifact: str, label: str) -> str:
    relative_path = case_manifest.get("perf", {}).get("artifacts", {}).get(artifact)
    if not relative_path or os.path.isabs(relative_path):
        return "-"
    href = os.path.join("profile", case_manifest["case"], relative_path).replace(
        os.path.sep, "/"
    )
    return f'<a href="{escape(href)}">{escape(label)}</a>'


def offcpu_artifact_link(case_manifest: Dict, artifact: str, label: str) -> str:
    relative_path = case_manifest.get("offcpu", {}).get("artifacts", {}).get(artifact)
    if not relative_path or os.path.isabs(relative_path):
        return "-"
    href = os.path.join("profile", case_manifest["case"], relative_path).replace(
        os.path.sep, "/"
    )
    return f'<a href="{escape(href)}">{escape(label)}</a>'


def latency_pair(results: List[Dict], prefix: str) -> str:
    stats = []
    for label in ("p50", "p99", "max"):
        summary = metric_summary(results, f"{prefix}_latency_ns_{label}")
        stats.append(f"{label} {escape(format_summary(summary, duration=True))}")
    return "<br>".join(stats)


def human_bytes(value: float) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1e12:
        return f"{human_number(value / 1e12)} TB"
    if abs(value) >= 1e9:
        return f"{human_number(value / 1e9)} GB"
    if abs(value) >= 1e6:
        return f"{human_number(value / 1e6)} MB"
    return f"{human_number(value)} B"


def human_bytes_metric(results: List[Dict]) -> str:
    summary = metric_summary(results, "total_bytes_transferred")
    if not summary:
        return "-"
    if summary["n"] == 1:
        return human_bytes(summary["median"])
    return (
        f"{human_bytes(summary['median'])} "
        f"(MAD {human_bytes(summary['mad'])}, "
        f"{human_bytes(summary['min'])}..{human_bytes(summary['max'])}, "
        f"n={summary['n']})"
    )


def tree_section(cases: List[Tuple[Dict, List[Dict]]]) -> str:
    rows = []
    for case_manifest, results in cases:
        if case_manifest.get("subcommand") != "tree":
            continue
        name = f"<code>{escape(case_manifest['case'])}</code>"
        status = status_badge(case_manifest.get("status", "unknown"))
        if not results:
            rows.append([name, status] + ["-"] * 10)
            continue
        task_pool = escape(
            format_summary(metric_summary(results, "task_pool_size_resolved"))
        )
        rows.append(
            [
                name,
                status,
                task_pool,
                escape(
                    format_summary(phase_summary(results, "measured"), duration=True)
                ),
                escape(tree_completed_requests(results)),
                escape(tree_request_mix(results)),
                escape(tree_reuse(results)),
                latency_pair(results, "match"),
                latency_pair(results, "insert"),
                latency_pair(results, "match_to_ready"),
                escape(tree_batch(results)),
                tree_final_cleanup(results),
            ]
        )
    return table(
        [
            "Case",
            "状态",
            "后台 cache 任务线程",
            "测量窗口",
            "已完成请求生命周期（数量 / 每秒）",
            "请求组成（新会话 / 续写）",
            "命中深度（计划 / 实际，blocks/请求）",
            "cache 查找时延（match）",
            "路径发布时延（insert）",
            "查找到可 forward 时延",
            "READY batch（平均 / 最大）",
            "结束清理",
        ],
        rows,
    )


def tree_completed_requests(results: List[Dict]) -> str:
    completed = format_summary(
        metric_summary(results, "completed_request_transactions")
    )
    per_second = format_summary(
        metric_summary(results, "benchmark_request_transactions_per_second")
    )
    return f"{completed} / {per_second} req/s"


def tree_request_mix(results: List[Dict]) -> str:
    base = format_summary(metric_summary(results, "completed_base_transactions"))
    continuation = format_summary(
        metric_summary(results, "completed_continuation_transactions")
    )
    families = format_summary(
        metric_summary(results, "completed_continuation_family_count")
    )
    return f"新会话 {base} / 续写 {continuation}（覆盖 {families} 个会话族）"


def tree_final_cleanup(results: List[Dict]) -> str:
    keys = (
        "final.active_requests",
        "final.pending_load_tickets",
        "final.pending_tasks",
        "final.request_ref_blocks",
        "drain_timeouts",
    )
    values = [result.get("metrics", {}).get(key) for result in results for key in keys]
    if values and all(value == 0 for value in values):
        return '<span class="pill ok">通过：运行态与资源残留均为 0</span>'
    return '<span class="pill fail">异常：查看 final.* / drain_timeouts</span>'


def tree_reuse(results: List[Dict]) -> str:
    planned = format_summary(
        metric_summary(results, "planned_reuse_blocks_per_request")
    )
    actual = format_summary(metric_summary(results, "actual_matched_depth_per_request"))
    return f"{planned} / {actual}"


def tree_batch(results: List[Dict]) -> str:
    avg = format_summary(metric_summary(results, "ready_batch_size_avg"))
    mx = format_summary(metric_summary(results, "ready_batch_size_max"))
    return f"{avg} / {mx}"


def _parse_integer_list(value) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str) and value:
        return [int(item) for item in value.split(",")]
    return []


def tree_workload_description(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Explain the fixed Tree workload in user-facing terms."""
    tree_results = [
        result
        for case_manifest, results in cases
        if case_manifest.get("subcommand") == "tree"
        for result in results
    ]
    if not tree_results:
        return '<p class="muted">没有有效 Tree repetition，无法展示 workload 配置。</p>'

    config = tree_results[0].get("resolved_config", {})
    supplied = (report_metadata or {}).get("tree_workload", {})
    tokens_per_block = int(config.get("tokens_per_block", 0))
    lengths = _parse_integer_list(
        config.get("length_buckets_tokens", supplied.get("length_buckets_tokens"))
    )
    weights = _parse_integer_list(
        config.get("length_weights", supplied.get("length_weights"))
    )
    hit_rates = _parse_integer_list(
        config.get("hit_rates_percent", supplied.get("hit_rates_percent"))
    )

    payloads = []
    suffix = "_scaled_payload_bytes"
    for key, value in sorted(config.items()):
        if not key.startswith("group_set_") or not key.endswith(suffix):
            continue
        name = key[len("group_set_") : -len(suffix)]
        group_type = config.get(f"group_set_{name}_type", "-")
        payloads.append(f"{name} ({group_type}) {human_bytes(float(value))}")
    payload_text = "；".join(payloads) if payloads else "未记录"

    logical_concurrency = config.get("logical_concurrency", "-")
    scheduler_threads = config.get("foreground_scheduler_threads", "-")
    trace_count = config.get("operation_trace_count", "-")
    base_count = config.get("base_request_count", "-")
    continuation_count = config.get("continuation_request_count", "-")
    shared_nodes = config.get("shared_base_nodes", "-")
    background_nodes = config.get("background_tree_nodes", "-")
    device_blocks = config.get("device_pool_blocks", "-")
    host_blocks = config.get("host_pool_blocks", "-")
    forward_sleep = config.get("forward_sleep_ms", "-")
    warmup = config.get("warmup_seconds", "-")
    measured = config.get("measured_seconds", "-")

    bullets = [
        (
            f"并发模型：{escape(str(logical_concurrency))} 个逻辑会话 context（不是线程），"
            f"由 {escape(str(scheduler_threads))} 个前台 scheduler 线程推进；后台 cache task pool "
            "线程数见结果表。"
        ),
        (
            f"Block 语义：每个逻辑 path block 覆盖 {escape(str(tokens_per_block))} tokens。"
            f"Tree fixture 使用缩放后的每 block 有效 payload：{escape(payload_text)}；"
            "这是 benchmark 的 scaled payload，不等同于线上模型每 block 的实际显存占用。"
        ),
        (
            f"初始 cache：shared-base path {escape(str(shared_nodes))} blocks + "
            f"background tree {escape(str(background_nodes))} blocks；每个 GroupSet 的 device/host pool "
            f"分别为 {escape(str(device_blocks))}/{escape(str(host_blocks))} blocks。"
        ),
        (
            f"请求 trace：预生成 {escape(str(trace_count))} 条请求，包含两种生命周期类型："
            f"新会话 BASE {escape(str(base_count))} 条；续写 CONTINUATION {escape(str(continuation_count))} 条。"
            "BASE 从 shared base 选取命中前缀后追加唯一 suffix；CONTINUATION 继承同一会话的父 path，"
            "只追加更长的唯一 tail。每个会话首次请求或本次抽样长度不大于当前 leaf 时开始新的 BASE；"
            "只有抽样长度大于当前 leaf 时才生成 CONTINUATION。"
        ),
        (
            f"一次“完整请求生命周期”是 match → 必要的异步 load → READY batch forward → "
            f"insert 完整 path → 释放 request refs。每个 READY batch 固定 sleep "
            f"{escape(str(forward_sleep))} ms 模拟 forward；warmup {escape(str(warmup))} s，"
            f"measured {escape(str(measured))} s。结果表中的 req/s 是这个闭环数量除以 measured 墙钟时间，"
            "不是线上模型 TPS。"
        ),
    ]
    html_output = (
        "<h3>测试构造（如何理解这个 case）</h3>"
        '<ul class="findings">'
        + "".join(f"<li>{item}</li>" for item in bullets)
        + "</ul>"
    )

    if (
        lengths
        and len(lengths) == len(weights)
        and tokens_per_block > 0
        and sum(weights) > 0
    ):
        weight_total = sum(weights)
        min_tokens = min(lengths)
        max_tokens = max(lengths)
        min_blocks = (min_tokens + tokens_per_block - 1) // tokens_per_block
        max_blocks = (max_tokens + tokens_per_block - 1) // tokens_per_block
        dominant = sorted(zip(lengths, weights), key=lambda item: (-item[1], item[0]))[
            :4
        ]
        dominant_weight = sum(weight for _, weight in dominant)
        dominant_text = "、".join(
            f"{tokens:,} tokens（{100.0 * weight / weight_total:.1f}%）"
            for tokens, weight in dominant
        )
        remaining_probability = 100.0 * (weight_total - dominant_weight) / weight_total
        hit_text = ", ".join(f"{value}%" for value in hit_rates) or "未记录"
        html_output += (
            "<h3>请求长度与前缀命中分布</h3>"
            f"<p>每条 trace 请求按权重从 {len(lengths)} 个长度桶中抽样，而不是拆成 "
            f"{len(lengths)} 种不同 case。请求范围为 {min_tokens:,}–{max_tokens:,} tokens，"
            f"按每 block {tokens_per_block:,} tokens 换算约为 {min_blocks:,}–{max_blocks:,} 个逻辑 blocks。"
            f"权重最高的 {len(dominant)} 个桶为 {escape(dominant_text)}，合计约 "
            f"{100.0 * dominant_weight / weight_total:.1f}%；其余 {len(lengths) - len(dominant)} 个桶合计约 "
            f"{remaining_probability:.1f}%。"
            f"BASE 的计划前缀命中率从 {len(hit_rates)} 个等概率档位抽样："
            f"<code>{escape(hit_text)}</code>；"
            "CONTINUATION 的命中前缀等于同一会话已发布的父 path。</p>"
        )
    return html_output


def transfer_subsections(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> Tuple[str, str, str, str]:
    """Split transfer cases into sub-tables and return HTML for each subsection."""
    dh_cases = [(c, r) for c, r in cases if "device_host" in c["case"]]
    dd_cases = [(c, r) for c, r in cases if "device_disk" in c["case"]]
    hd_cases = [(c, r) for c, r in cases if "host_disk" in c["case"]]
    return (
        transfer_table(dh_cases, "Device↔Host 介质对"),
        transfer_table(dd_cases, "Device↔Disk 介质对"),
        transfer_table(hd_cases, "Host↔Disk 介质对"),
        bandwidth_judgment(cases, report_metadata or {}),
    )


def transfer_table(subset: List[Tuple[Dict, List[Dict]]], title: str) -> str:
    if not subset:
        return (
            f"<h3>{escape(title)}</h3>" '<p class="muted">本次专项报告无此类 case。</p>'
        )
    html = f"<h3>{escape(title)}</h3>\n"
    rows = []
    for case_manifest, results in subset:
        name = f"<code>{escape(case_manifest['case'])}</code>"
        status = status_badge(case_manifest.get("status", "unknown"))
        if not results:
            rows.append([name, status] + ["-"] * 9)
            continue
        direction_keys = sorted(
            {
                key
                for result in results
                for key in result.get("metrics", {})
                if key.startswith("direction.") and key.endswith(".throughput_bps")
            }
        )
        directions = "<br>".join(
            escape(key.split(".")[1])
            + " "
            + escape(format_summary(metric_summary(results, key), 1e-9, " GB/s"))
            for key in direction_keys
        )
        working_set = " / ".join(
            escape(common_value(results, "transfer_workload", key))
            for key in (
                "requested_working_set_blocks",
                "addressable_working_set_blocks",
                "visited_working_set_blocks",
            )
        )
        strategy = (
            escape(common_value(results, "resolved_config", "requested_copy_strategy"))
            + " → "
            + escape(common_value(results, "resolved_config", "actual_copy_strategy"))
        )
        mode = (
            common_value(results, "resolved_config", "disk_io_mode")
            if "disk" in case_manifest["case"]
            else "-"
        )
        requested_batch = common_value(
            results, "resolved_config", "requested_transfer_descriptor_batch_size"
        )
        if requested_batch == "0":
            requested_batch = "auto"
        resolved_batch = common_value(
            results, "resolved_config", "resolved_transfer_descriptor_batch_size"
        )
        batch_details = []
        for key in direction_keys:
            direction = key.split(".")[1]
            avg_key = f"direction.{direction}.descriptor_batch_size_avg"
            max_key = f"direction.{direction}.descriptor_batch_size_max"
            batch_details.append(
                f"{escape(direction)} "
                f"{escape(format_summary(metric_summary(results, avg_key)))} / "
                f"{escape(format_summary(metric_summary(results, max_key)))}"
            )
        descriptor_batch = (
            f"API {escape(requested_batch)} → {escape(resolved_batch)}"
            + ("<br>" + "<br>".join(batch_details) if batch_details else "")
        )
        rows.append(
            [
                name,
                status,
                escape(
                    format_summary(phase_summary(results, "measured"), duration=True)
                ),
                escape(mode if mode not in ("-", "mixed") else "-"),
                escape(
                    format_summary(
                        metric_summary(results, "logical_throughput_bytes_per_second"),
                        1e-9,
                        " GB/s",
                    )
                )
                + ("<br>" + directions if directions else ""),
                escape(
                    format_summary(metric_summary(results, "operations_per_second"))
                ),
                escape(human_bytes_metric(results)),
                strategy,
                descriptor_batch,
                working_set,
                escape(
                    format_summary(
                        section_summary(results, "workload", "failed_operations")
                    )
                ),
            ]
        )
    html += table(
        [
            "Case",
            "状态",
            "duration",
            "mode",
            "混合总吞吐（含各方向）",
            "ops/s",
            "总传输",
            "requested → actual strategy",
            "descriptor batch API requested → resolved；各方向 avg / max",
            "working set requested/addressable/visited",
            "failed",
        ],
        rows,
    )
    return html


def bandwidth_judgment(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Show measured bandwidth; only judge it when an explicit baseline is supplied."""
    supplied_rows = (report_metadata or {}).get("bandwidth_comparison", [])
    if supplied_rows:
        return table(
            ["路径", "硬件理论/实测极限", "benchmark 实测", "结论"],
            [
                [
                    escape(row.get("path", "-")),
                    escape(row.get("baseline", "-")),
                    escape(row.get("measured", "-")),
                    escape(row.get("conclusion", "-")),
                ]
                for row in supplied_rows
            ],
        )
    rows = [
        ["Device↔Host", "未提供环境实测基线", "-", "仅展示 benchmark 实测"],
        ["Host↔Disk (direct)", "未提供环境实测基线", "-", "仅展示 benchmark 实测"],
        ["Device↔Disk (direct)", "未提供环境实测基线", "-", "仅展示 benchmark 实测"],
        ["Host↔Disk (buffered)", "未提供环境实测基线", "-", "仅展示 benchmark 实测"],
        ["Device↔Disk (buffered)", "未提供环境实测基线", "-", "仅展示 benchmark 实测"],
    ]
    # Fill in measured values where we have data
    for case_manifest, results in cases:
        if not results:
            continue
        name = case_manifest["case"]
        thru = format_summary(
            metric_summary(results, "logical_throughput_bytes_per_second"),
            1e-9,
            " GB/s",
        )
        if "device_host" in name:
            if rows[0][2] == "-":
                rows[0][2] = thru
        elif "device_disk" in name:
            idx = 2 if "direct" in name else 4
            if rows[idx][2] == "-":
                rows[idx][2] = thru
        elif "host_disk" in name:
            idx = 1 if "direct" in name else 3
            if rows[idx][2] == "-":
                rows[idx][2] = thru

    return table(
        ["路径", "硬件理论/实测极限", "benchmark 实测", "结论"],
        rows,
    )


def io_section(cases: List[Tuple[Dict, List[Dict]]]) -> str:
    rows = []
    for case_manifest, _ in cases:
        if "disk" not in case_manifest["case"]:
            continue
        drain = summarize(
            repetition.get("drain_seconds", 0)
            for repetition in case_manifest.get("repetitions", [])
            if repetition.get("valid")
        )
        rows.append(
            [
                f"<code>{escape(case_manifest['case'])}</code>",
                escape(format_summary(vmstat_deltas(case_manifest, "pgpgin"))),
                escape(format_summary(vmstat_deltas(case_manifest, "pgpgout"))),
                escape(format_summary(vmstat_deltas(case_manifest, "nr_dirty"))),
                escape(format_summary(drain, 1.0, " s")),
            ]
        )
    return table(
        ["Case", "pgpgin delta", "pgpgout delta", "nr_dirty delta", "drain seconds"],
        rows,
    )


def environment_section(manifest: Dict) -> str:
    """Enhanced environment section with hardware details and execution command."""
    environment = dict(manifest.get("environment", {}))
    invocation = manifest.get("invocation", {})

    # Old manifests stored the entire overlay lowerdir/upperdir option list.
    # Compact them at render time so existing artifacts remain readable.
    legacy_disk = environment.pop("disk", None)
    if legacy_disk and "disk_mount" not in environment:
        disk_fields = str(legacy_disk).split()
        if len(disk_fields) >= 2:
            environment["disk_mount"] = (
                f"source={disk_fields[0]}, fstype={disk_fields[1]}"
            )
        else:
            environment["disk_mount"] = str(legacy_disk)
        environment.setdefault("disk_capacity", "not recorded by this legacy manifest")

    # Keep the raw manifest fields lossless, but render disk provenance as one
    # compact row. The invocation below already records --disk-root, so a
    # second explanatory notice and four separate environment rows add noise.
    disk_mount = environment.pop("disk_mount", None)
    environment.pop("disk_target", None)
    disk_capacity = environment.pop("disk_capacity", None)
    disk_scope = environment.pop("disk_scope", None)
    if disk_mount or disk_capacity or disk_scope:
        disk_parts = []
        if disk_mount:
            mount_text = str(disk_mount)
            if mount_text.startswith("target="):
                mount_text = "mount=" + mount_text[len("target=") :]
            disk_parts.append(mount_text)
        if disk_capacity:
            disk_parts.append(str(disk_capacity))
        if disk_scope:
            scope = (
                "container-visible mount namespace"
                if "container-visible" in str(disk_scope)
                else "benchmark mount namespace"
            )
            disk_parts.append(f"scope={scope}")
        environment["disk"] = "; ".join(disk_parts)

    rows = [
        [escape(key), f"<code>{escape(value)}</code>"]
        for key, value in environment.items()
    ]
    env_table = table(["项", "采集值"], rows)
    suite = invocation.get("suite", manifest.get("suite", "profile"))
    cmd_parts = ["block_tree_cache_benchmark_driver", "--suite", str(suite)]
    if invocation.get("case", "all") != "all":
        cmd_parts.extend(["--case", str(invocation["case"])])
    option_pairs = [
        ("--process-repetitions", invocation.get("process_repetitions")),
        ("--case-timeout-seconds", invocation.get("case_timeout_seconds")),
        ("--termination-grace-seconds", invocation.get("termination_grace_seconds")),
        ("--perf", invocation.get("perf")),
        ("--perf-frequency", invocation.get("perf_frequency")),
        ("--flamegraph-tools-dir", invocation.get("flamegraph_tools_dir")),
        ("--output-dir", invocation.get("output_dir")),
        ("--disk-root", invocation.get("disk_root")),
    ]
    for option, value in option_pairs:
        if value is not None:
            cmd_parts.extend([option, str(value)])
    if invocation.get("allow_incomplete"):
        cmd_parts.append("--allow-incomplete")
    if not invocation:
        cmd_parts.extend(["--perf", "record", "--output-dir", "<output-dir>"])
        if suite == "profile":
            cmd_parts.extend(["--disk-root", "<disk-root>"])
    cmd = " \\\n  ".join(shlex.quote(part) for part in cmd_parts)
    exec_html = (
        f"<h3>执行命令</h3>"
        f"<pre><code>export LD_LIBRARY_PATH=/opt/conda310/lib:$LD_LIBRARY_PATH\n{escape(cmd)}</code></pre>"
    )
    return env_table + exec_html


def conclusion_section(
    manifest: Dict,
    cases: List[Tuple[Dict, List[Dict]]],
    report_metadata: Dict = None,
) -> str:
    """Auto-generate a bullet-point executive summary."""
    completed = manifest.get("completed", 0)
    partial = manifest.get("partial", 0)
    failed = manifest.get("failed", 0)
    skipped = manifest.get("skipped", 0)
    total = manifest.get("total_cases", len(cases))
    canonical_total = manifest.get("canonical_total_cases", total)
    complete = completed == total and partial == failed == skipped == 0
    if total == canonical_total:
        scope = "完整 canonical suite" if complete else "不完整 canonical suite"
    else:
        scope = "完整专项复测" if complete else "不完整专项复测"

    bullets = [
        f"<li>{scope}：{completed}/{total} completed，{partial} partial，{failed} failed，{skipped} skipped。</li>",
    ]
    if not complete:
        bullets.append(
            "<li>Suite 不完整，不能发布正式性能结论，但可用于事实核对和问题定位。</li>"
        )

    # Count perf artifacts
    perf_count = sum(
        1 for c, _ in cases if c.get("perf", {}).get("status") in ("ok", "completed")
    )
    if perf_count > 0:
        bullets.append(f"<li>生成 {perf_count} 张代表性 CPU 火焰图。</li>")

    # Tree vs transfer breakdown
    tree_count = sum(
        1
        for c, _ in cases
        if c.get("subcommand") == "tree" and c.get("status") == "completed"
    )
    transfer_count = sum(
        1
        for c, _ in cases
        if c.get("subcommand") == "transfer" and c.get("status") == "completed"
    )
    if tree_count:
        bullets.append(f"<li>Tree 场景 {tree_count} 个 case completed。</li>")
    if transfer_count:
        bullets.append(f"<li>Transfer 场景 {transfer_count} 个 case completed。</li>")

    repetition_counts = [len(results) for _, results in cases]
    baseline_note = (
        "本次已提供同机硬件基线与分析元数据；性能归因及其边界见正文。"
        if (report_metadata or {}).get("bandwidth_comparison")
        else "未配置同机硬件基线与显式阈值时，性能归因待分析。"
    )
    if repetition_counts and set(repetition_counts) == {1}:
        bullets.append(
            "<li>本次每 case 仅 1 次有效 repetition，只能作为单次实测事实；"
            f"{baseline_note}</li>"
        )
    elif repetition_counts and len(set(repetition_counts)) == 1:
        bullets.append(
            f"<li>本次每 case 有 {repetition_counts[0]} 次有效 repetition；"
            f"波动范围见各表的 MAD/min/max。{baseline_note}</li>"
        )
    elif repetition_counts:
        bullets.append(
            "<li>各 case 的有效 repetition 数不一致；聚合值已逐项显示样本数。"
            f"{baseline_note}</li>"
        )

    cd = manifest.get("conclusion_details", [])
    for detail in cd:
        bullets.append(f"<li>{escape(detail)}</li>")

    return '<ul class="findings">' + "".join(bullets) + "</ul>"


def _tree_pairing_signature(results: List[Dict]) -> Dict:
    """Pair every repetition by seed/id and compare every non-pool identity field."""
    signatures = {}
    for index, result in enumerate(results):
        config = result.get("resolved_config", {})
        identity = json.dumps(
            [
                result.get("workload", {}).get("seed", f"missing-seed-{index}"),
                config.get("repetition_identity", index),
            ],
            separators=(",", ":"),
        )
        signature = {
            "resolved_config": {
                key: value
                for key, value in config.items()
                if key not in {"task_pool_size_resolved", "repetition_identity"}
            },
            "model_profile": result.get("model_profile", {}),
            "payload": result.get("payload", {}),
            "suite_provenance": result.get("_suite_provenance", {}),
            "manifest_identity": result.get("_manifest_identity", {}),
        }
        if identity in signatures:
            return {"duplicate_identity": identity}
        signatures[identity] = signature
    return signatures


def tree_observations(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Auto-generate Tree observations from lifecycle facts.

    There is exactly one active Tree workload shape; no multi/single-worker
    comparison is inferred. When several task-pool results are supplied they
    are grouped explicitly by task_pool_size_resolved, and comparison is
    refused unless every other config/trace field matches.
    """
    tree_data = [
        (case_manifest, results)
        for case_manifest, results in cases
        if case_manifest.get("subcommand") == "tree" and results
    ]

    obs = []
    groups: Dict[str, List[Tuple[Dict, List[Dict]]]] = {}
    for case_manifest, results in tree_data:
        pool_key = common_value(results, "resolved_config", "task_pool_size_resolved")
        groups.setdefault(pool_key, []).append((case_manifest, results))

    if len(groups) > 1:
        signatures = {
            pool_key: _tree_pairing_signature(
                [
                    result
                    for _, grouped_results in grouped_cases
                    for result in grouped_results
                ]
            )
            for pool_key, grouped_cases in groups.items()
        }
        signature_values = {
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
            for signature in signatures.values()
        }
        if len(signature_values) != 1:
            obs.append(
                "<li>task-pool 对照被拒绝：除 task pool size 外，各组的 resolved config / "
                "profile / seed-repetition / trace / binary SHA / code commit 不一致，"
                "不能作为唯一变量比较。</li>"
            )
        else:
            parts = []
            for pool_key in sorted(groups, key=int):
                pool_results = [
                    result
                    for _, group_results in groups[pool_key]
                    for result in group_results
                ]
                ready = format_summary(
                    metric_summary(pool_results, "match_to_ready_latency_ns_p50"),
                    duration=True,
                )
                wait = format_summary(
                    metric_summary(pool_results, "scheduler_no_ready_wait_ns"),
                    duration=True,
                )
                tps = format_summary(
                    metric_summary(
                        pool_results, "benchmark_request_transactions_per_second"
                    )
                )
                match_lat = format_summary(
                    metric_summary(pool_results, "match_latency_ns_p50"), duration=True
                )
                parts.append(
                    f"后台任务线程 <strong>{pool_key}</strong>: cache 查找 p50 {match_lat}, "
                    f"从查找到可 forward p50 {ready}, 无 READY 请求时累计等待 {wait}, "
                    f"完整请求生命周期 {tps} req/s"
                )
            obs.append(
                "<li>task-pool 唯一变量对照（逐 repetition workload 与执行身份一致）："
                + "；".join(parts)
                + "。这里的 req/s 是 benchmark 请求闭环口径，不是线上模型 TPS。</li>"
            )

    for case_manifest, results in tree_data:
        m = results[0].get("metrics", {})
        rc = results[0].get("resolved_config", {})
        ph = results[0].get("phases_ns", {})
        lifecycle = results[0].get("tree_lifecycle", {})
        measured_s = ph.get("measured", 0) / 1e9
        concurrency = rc.get("logical_concurrency", "-")
        scheduler_threads = rc.get("foreground_scheduler_threads", "-")
        task_pool = rc.get("task_pool_size_resolved", "-")
        trace_hash = rc.get("trace_hash", "-")
        forward_batches = m.get("forward_batches", 0)
        forward_requests = m.get("forward_requests", 0)
        sleep_ns = forward_batches * rc.get("forward_sleep_ms", 0) * 1_000_000
        obs.append(
            f"<li><code>{escape(case_manifest['case'])}</code>：{escape(str(concurrency))} 个逻辑 "
            f"会话 context，前台 scheduler <strong>{escape(str(scheduler_threads))}</strong> 线程，"
            f"后台 cache task pool {escape(str(task_pool))} 线程；trace hash <code>{escape(trace_hash)}</code>。"
            f"measured 窗口内完成 {forward_batches} 个 READY batch / {forward_requests} 个请求，"
            f"其中模拟 forward sleep 合计 {escape(format_duration(sleep_ns))}（每 batch 100ms），"
            f"measured {human_number(measured_s)}s。</li>"
        )
        held_peak = m.get("held_request_blocks_peak", 0)
        load_peak = m.get("load_tickets_pending_peak", 0)
        loading_peak = m.get("loading_requests_peak", 0)
        active_peak = m.get("active_requests_peak", 0)
        extra = m.get("unexpected_extra_match_count", 0)
        dep_skip = m.get("dependency_skip_count", 0)
        dep_failed = m.get("dependency_failed_descendants", 0)
        base_cnt = m.get("completed_base_transactions", "-")
        cont_cnt = m.get("completed_continuation_transactions", "-")
        cont_families = m.get("completed_continuation_family_count", "-")
        obs.append(
            f"<li>并发与资源峰值：同时在途请求最多 {human_number(active_peak)} 个"
            f"（上限 {escape(str(concurrency))}），其中正在 load 最多 {human_number(loading_peak)} 个，"
            f"待完成的异步 load ticket 最多 {human_number(load_peak)} 个；请求跨 forward 持有的 "
            f"block 峰值为 {human_number(held_peak)}。实际命中超过计划前缀的请求数为 "
            f"{human_number(extra)}。</li>"
        )
        obs.append(
            f"<li>请求组成：完成新会话请求 {escape(str(base_cnt))} 个、续写请求 "
            f"{escape(str(cont_cnt))} 个，32 个会话族中有 {escape(str(cont_families))} 个完成过续写。"
            f"因前序请求尚未发布而暂缓 admission 的扫描次数为 {human_number(dep_skip)} 次，"
            f"受前序失败影响的后继请求为 {human_number(dep_failed)} 个。</li>"
        )
        pressure = lifecycle.get("pressure_ready")
        if pressure is True:
            obs.append(
                "<li>Warmup 结束时所有预设水位观察条件均已达到。该水位快照只描述负载形态，"
                "不替代结束清理校验；结束时 active request、pending ticket、pending task 和 "
                "REQUEST ref 均无残留。</li>"
            )
        elif pressure is False:
            obs.append(
                "<li>Warmup 结束时没有同时达到全部预设水位观察条件（例如每个 device pool "
                "使用率至少 75%）。这是负载形态快照，不是失败条件；结束清理校验仍然通过。</li>"
            )

    obs_html = "<h3>主要观察</h3>" if obs else ""
    if obs:
        obs_html += '<ul class="findings">' + "".join(obs) + "</ul>"
    supplied = (report_metadata or {}).get("tree_findings", [])
    if supplied:
        obs_html += (
            '<h3>本次分析</h3><ul class="findings">'
            + "".join(f"<li>{escape(item)}</li>" for item in supplied)
            + "</ul>"
        )
    else:
        obs_html += (
            '<div class="notice">'
            "<strong>留白：</strong>报告作者应结合火焰图热点和本次测试数据补充详细分析。"
            "固定 100ms forward sleep 和 task-pool idle wait 是预期 off-CPU 时间，不能归因成 "
            "BlockTreeCache 退化；可关注方向：match/insert 锁竞争、eviction 淘汰路径的 CPU 占比、"
            "load 异步流水线效率等。"
            "</div>"
        )
    return obs_html


def supplied_findings(report_metadata: Dict, key: str, title: str = "") -> str:
    items = (report_metadata or {}).get(key, [])
    if not items:
        return ""
    heading = f"<h3>{escape(title)}</h3>" if title else ""
    return (
        heading
        + '<ul class="findings">'
        + "".join(f"<li>{escape(item)}</li>" for item in items)
        + "</ul>"
    )


def flamegraph_section(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Show only cases that actually produced profiling artifacts."""
    perf_rows = []
    offcpu_rows = []
    offcpu_skips = []
    for case_manifest, _ in cases:
        perf = case_manifest.get("perf", {})
        if perf.get("status") in ("ok", "completed") and perf.get("artifacts"):
            perf_rows.append(
                [
                    f"<code>{escape(case_manifest['case'])}</code>",
                    escape(perf.get("status", "-")),
                    escape(perf.get("mode", "-")),
                    perf_artifact_link(case_manifest, "flamegraph", "打开 SVG"),
                    perf_artifact_link(case_manifest, "perf_data", "下载 perf.data"),
                    perf_artifact_link(case_manifest, "summary", "查看摘要"),
                ]
            )
        offcpu = case_manifest.get("offcpu", {})
        if offcpu.get("status") in ("ok", "completed") and offcpu.get("artifacts"):
            offcpu_rows.append(
                [
                    f"<code>{escape(case_manifest['case'])}</code>",
                    escape(offcpu.get("status", "-")),
                    offcpu_artifact_link(case_manifest, "folded", "打开 folded"),
                    offcpu_artifact_link(case_manifest, "svg", "打开 SVG"),
                    offcpu_artifact_link(case_manifest, "manifest", "manifest"),
                    escape(offcpu.get("summary", "")[:60] or "-"),
                ]
            )
        elif offcpu:
            offcpu_skips.append(
                f"{case_manifest.get('case', '-')}: "
                f"{offcpu.get('status', 'skipped')} ({offcpu.get('reason', '未说明原因')})"
            )

    perf_table_html = table(
        ["Case", "状态", "模式", "CPU 火焰图", "原始数据", "文本摘要"],
        perf_rows,
    )
    offcpu_table_html = "<h3>Off-CPU 采集</h3>"
    if offcpu_rows:
        offcpu_table_html += "<p>独立进程采集，不计入 repetition 聚合。</p>" + table(
            ["Case", "状态", "raw folded", "Off-CPU SVG", "manifest", "质量摘要"],
            offcpu_rows,
        )
    else:
        supplied_status = (report_metadata or {}).get("offcpu_status", {})
        if supplied_status:
            status = supplied_status.get("status", "skipped")
            reason = supplied_status.get("reason", "未说明原因")
            message = f"{status}：{reason}"
        elif offcpu_skips:
            message = "；".join(offcpu_skips)
        else:
            message = "未采集：suite manifest 未记录 off-CPU 产物或跳过原因。"
        offcpu_table_html += (
            '<div class="notice"><p><strong>Off-CPU：</strong>'
            f"{escape(message)}</p></div>"
        )

    guidance = (
        '<div class="notice">'
        "<p>说明：</p>"
        "<ul>"
        "<li>火焰图只反映 on-CPU 时间；磁盘等待、CUDA 流同步等 off-CPU 段不会出现在 CPU 火焰图里。</li>"
        "<li><code>[unknown]</code> 可能来自闭源库、缺少 debug info 或 DWARF 展开失败；归因前必须检查 build config、build-id cache 和 perf.data 属主。</li>"
        "<li>采样质量、样本数、lost samples、<code>[unknown]</code> 占比见 perf_summary.txt。</li>"
        "<li>Tree 的固定 100ms forward sleep（sleep_for/nanosleep）和 task-pool idle wait 是预期 off-CPU 时间；"
        "off-CPU folded 保留原始栈，不将这两类等待归因成 cache 退化。</li>"
        "</ul>"
        "</div>"
    )
    return perf_table_html + offcpu_table_html + guidance


def artifact_section(manifest_href: str, report_metadata: Dict = None) -> str:
    """Render the report artifact inventory and reproducibility checklist."""
    supplied_rows = (report_metadata or {}).get("artifact_rows", [])
    if supplied_rows:
        rows = []
        for row in supplied_rows:
            location = escape(row.get("location", "-"))
            href = row.get("href")
            if href:
                location = f'<a href="{escape(href)}"><code>{location}</code></a>'
            else:
                location = f"<code>{location}</code>"
            rows.append(
                [
                    escape(row.get("name", "-")),
                    location,
                    escape(row.get("check", "-")),
                ]
            )
        return table(["产物", "位置", "完整性/校验"], rows)

    rows = [
        ["HTML 报告", "<code>index.html</code>", "本页；检查所有相对链接"],
        [
            "Suite manifest",
            f'<a href="{escape(manifest_href)}"><code>{escape(manifest_href)}</code></a>',
            "记录 case、valid repetitions 与环境指纹",
        ],
        [
            "原始 repetition 结果",
            f"<code>{escape('profile/<case>/rep_*/result.json')}</code>",
            "只使用 manifest 标记的 valid repetitions",
        ],
        [
            "stdout/stderr 与采样",
            f"<code>{escape('profile/<case>/rep_*/')}</code>",
            "stdout、stderr、vmstat/nvidia-smi before/after",
        ],
        [
            "perf 产物",
            f"<code>{escape('profile/<case>/perf/')}</code>",
            "perf.data、perf.folded、flamegraph.svg、perf_summary.txt",
        ],
        [
            "原始数据包",
            "<code>&lt;OSS URL / archive path&gt;</code>",
            "留白：填写 URL 与 SHA256",
        ],
    ]
    return table(["产物", "位置", "完整性/校验"], rows) + (
        '<div class="notice">'
        "<p><strong>留白。</strong>发布前填写上传地址、数据包 SHA256 和复现命令；"
        "代码 commit、binary SHA256、profile SHA256 必须与环境表及 manifest 一致。</p>"
        "</div>"
    )


def render_report(
    manifest: Dict, cases: List[Tuple[Dict, List[Dict]]], manifest_href: str
) -> str:
    completed = manifest.get("completed", 0)
    total = manifest.get("total_cases", len(cases))
    partial = manifest.get("partial", 0)
    failed = manifest.get("failed", 0)
    skipped = manifest.get("skipped", 0)
    complete = completed == total and partial == failed == skipped == 0
    canonical = total == manifest.get("canonical_total_cases", total)

    report_metadata = manifest.get("report_metadata", {})

    # Build subsections
    dh_html, dd_html, hd_html, bw_html = transfer_subsections(cases, report_metadata)

    # IO section
    supplied_io = supplied_findings(report_metadata, "io_findings", "本次 IO 分析")
    io_html = io_section(cases) + (
        supplied_io
        or (
            '<div class="notice">'
            "<p><strong>vmstat 数据解读指引（留白，由报告作者填写）：</strong></p>"
            "<ol>"
            "<li>buffered 场景的 dirty/writeback 变化和 drain 时间是否表明内核发生写回或节流；缺少阈值采样时明确证据不足。</li>"
            "<li>direct 场景 pgpgin/pgpgout 是否与逻辑传输量数量级一致；同时排除单位、系统其他 IO 与采样窗口影响。</li>"
            "<li>buffered 物理写回量占逻辑吞吐的比例（脏页被重写回收的程度，佐证工作集是否触发真实 writeback）。</li>"
            "</ol>"
            "</div>"
        )
    )

    executive_findings = supplied_findings(report_metadata, "key_findings")
    if not executive_findings:
        executive_findings = '<div class="notice"><strong>留白：</strong>报告作者应在发布前补充本次最重要的 Tree、Transfer、磁盘和 perf 事实，并明确结果能否发布。</div>'
    transfer_findings = supplied_findings(
        report_metadata, "transfer_findings", "本次 Transfer 分析"
    )
    if not transfer_findings:
        if report_metadata.get("bandwidth_comparison"):
            transfer_findings = '<div class="notice"><p><strong>留白：</strong>已提供同机带宽基线；报告作者仍需结合每方向吞吐、working-set、CPU 占用和 vmstat 补充 Transfer 归因及边界。</p></div>'
        else:
            transfer_findings = '<div class="notice"><p><strong>留白：</strong>结合同机 PCIe/磁盘基线、每方向吞吐、working-set、CPU 占用和 vmstat 分析瓶颈。没有同机基线与显式阈值时，结论保持“待分析”。</p></div>'
    perf_findings = supplied_findings(
        report_metadata, "perf_findings", "本次热点与采样质量分析"
    )
    limitations = supplied_findings(report_metadata, "limitations")
    if not limitations:
        limitations = """
    <div class="notice">
      <p><strong>留白。</strong>基于本次测试给出 3–5 条限制与后续建议。</p>
    </div>"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlockTreeCache Benchmark Report</title>
  <style>
    :root {{--bg:#f4f7fb;--surface:#fff;--text:#172033;--muted:#64748b;--line:#dbe3ee;--brand:#2563eb;--ok:#087a55;--ok-bg:#e5f7ef;--fail:#b42318;--fail-bg:#ffebe9;}}
    * {{box-sizing:border-box}} body {{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
    main {{width:min(1280px,calc(100% - 28px));margin:auto;padding:32px 0 64px}} h1 {{font-size:clamp(2rem,5vw,3.25rem);margin:0 0 8px}} h2 {{margin:2rem 0 0.5rem}} h3 {{margin:1rem 0 0.5rem}} p {{margin:.5rem 0}}
    .hero,.section {{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:26px;margin-bottom:18px}} .hero {{background:linear-gradient(135deg,#fff,#edf5ff)}}
    .eyebrow {{color:var(--brand);font-weight:750;letter-spacing:.08em;text-transform:uppercase}} .conclusion {{font-size:1.08rem;font-weight:650;margin-top:18px}} .muted {{color:var(--muted)}}
    .table-wrap {{overflow-x:auto}} table {{width:100%;border-collapse:collapse;min-width:760px}} th,td {{padding:11px 13px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th {{background:#f8fafc;color:var(--muted);font-size:.88rem}} code {{background:#eef2f7;border-radius:5px;padding:2px 5px;overflow-wrap:anywhere}}
    .pill {{display:inline-block;border-radius:999px;padding:3px 8px;font-size:.78rem;font-weight:750}} .pill.ok {{color:var(--ok);background:var(--ok-bg)}} .pill.fail {{color:var(--fail);background:var(--fail-bg)}}
    .notice {{border-left:4px solid var(--brand);background:#edf5ff;padding:13px 16px;border-radius:8px}} .findings {{margin:14px 0 0;padding-left:22px}} .findings li {{margin:5px 0}} a {{color:var(--brand);text-decoration:none}} a:hover {{text-decoration:underline}}
    pre {{background:#f2f4f7;padding:12px;border-radius:8px;overflow-x:auto;font-size:.88rem}}
  </style>
</head>
<body><main>
  <header class="hero">
    <div class="eyebrow">RTP-LLM · BlockTreeCache</div>
    <h1>Benchmark 报告</h1>
    <h2>0. 执行结论</h2>
    <p class="conclusion">
      {('完整 canonical suite：全部 case 有效完成。' if canonical else '专项复测：全部选定 case 有效完成。') if complete else f'Suite 不完整：{completed}/{total} completed，{partial} partial，{failed} failed，{skipped} skipped。'}
    </p>
    {conclusion_section(manifest, cases, report_metadata)}
    {executive_findings}
  </header>

  <section class="section"><h2>1. 测试环境</h2>{environment_section(manifest)}</section>

  <section class="section"><h2>2. Tree 在线生命周期场景</h2>
    {tree_workload_description(cases, report_metadata)}
    <h3>运行结果</h3>
    {tree_section(cases)}
    {tree_observations(cases, report_metadata)}
  </section>

  <section class="section"><h2>3. Transfer 场景</h2>
    <p>“混合总吞吐”是同一 measured window 内各方向成功字节之和除以墙钟时间，不是两个单方向峰值相加。</p>
    {dh_html}
    {dd_html}
    {hd_html}
    <h3>带宽判断</h3>
    {bw_html}
    {transfer_findings}
  </section>

  <section class="section"><h2>4. 火焰图与采样质量</h2>{flamegraph_section(cases, report_metadata)}{perf_findings}</section>

  <section class="section"><h2>5. 系统级 IO 旁证</h2>{io_html}</section>

  <section class="section"><h2>6. 限制与后续建议</h2>
    {limitations}
  </section>
  <section class="section"><h2>7. 产物与可复核性</h2>{artifact_section(manifest_href, report_metadata)}</section>
</main></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/btc_profile")
    parser.add_argument(
        "--output",
        help="HTML output path (default: <output-dir>/index.html)",
    )
    args = parser.parse_args()

    profile_dir = os.path.join(args.output_dir, "profile")
    manifest_path = os.path.join(profile_dir, "suite_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: suite manifest not found at {manifest_path}", file=sys.stderr)
        return 1
    manifest = load_json(manifest_path)
    metadata_path = os.path.join(args.output_dir, "report_metadata.json")
    if os.path.exists(metadata_path):
        manifest["report_metadata"] = load_json(metadata_path)
    cases = load_manifest_results(profile_dir, manifest)

    output_path = args.output or os.path.join(args.output_dir, "index.html")
    report = render_report(manifest, cases, "profile/suite_manifest.json")
    with open(output_path, "w") as output:
        output.write(report)
    print(f"HTML report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

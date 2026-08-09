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
                results.append(result)
        cases.append((case_manifest, results))
    return cases


def metric_summary(results: List[Dict], key: str) -> Dict[str, float]:
    return summarize(
        result.get("metrics", {}).get(key)
        for result in results
        if key in result.get("metrics", {})
    )


def metric_values(results: List[Dict], key: str) -> List[float]:
    return [
        result.get("metrics", {}).get(key)
        for result in results
        if key in result.get("metrics", {})
    ]


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


def latency_pair(results: List[Dict], prefix: str) -> str:
    stats = []
    for label in ("p50", "p99", "max"):
        summary = metric_summary(results, f"{prefix}_latency_ns_{label}")
        stats.append(f"{label} {escape(format_summary(summary, duration=True))}")
    return "<br>".join(stats)


def ops_per_second(results: List[Dict], calls_key: str) -> str:
    values = []
    for result in results:
        calls = result.get("metrics", {}).get(calls_key)
        measured_ns = result.get("phases_ns", {}).get("measured")
        if calls is not None and measured_ns:
            values.append(calls / (measured_ns / 1e9))
    return format_summary(summarize(values))


def request_shape(results: List[Dict]) -> str:
    insert_path = format_summary(metric_summary(results, "insert_path_keys_per_call"))
    new_nodes = format_summary(metric_summary(results, "insert_new_nodes_per_call"))
    match_keys = format_summary(metric_summary(results, "match_keys_per_call"))
    return f"{insert_path} / {new_nodes} / {match_keys}"


def scenario_matched_depths(results: List[Dict]) -> str:
    values = []
    for scenario in ("continuation", "fork", "cold"):
        values.append(
            format_summary(
                metric_summary(results, f"scenario.{scenario}.average_matched_depth")
            )
        )
    return " / ".join(values)


def matched_blocks_per_request(results: List[Dict]) -> str:
    device = format_summary(
        metric_summary(results, "match_device_matched_blocks_per_request")
    )
    host = format_summary(
        metric_summary(results, "match_host_matched_blocks_per_request")
    )
    return f"{device} / {host}"


def loads_summary(results: List[Dict]) -> str:
    committed = format_summary(metric_summary(results, "loads_committed"))
    succeeded = format_summary(metric_summary(results, "loads_succeeded"))
    labels = {
        "loads_failed": "failed",
        "loads_cancelled": "cancelled",
        "load_commit_failed": "commit failed",
        "loads_pending_at_measurement_end": "pending",
    }
    extra = []
    for key, label in labels.items():
        summary = metric_summary(results, key)
        if summary and summary.get("median", 0) != 0:
            extra.append(f"{label} {format_summary(summary)}")
    text = f"{committed} / {succeeded}"
    if extra:
        text += "<br>(" + ", ".join(extra) + ")"
    return text


def node_watermark(results: List[Dict]) -> str:
    avg = format_summary(
        summarize(
            round(value)
            for value in metric_values(results, "steady_state_node_count_avg")
        )
    )
    mn = format_summary(metric_summary(results, "steady_state_node_count_min"))
    mx = format_summary(metric_summary(results, "steady_state_node_count_max"))
    return f"{avg} [{mn}, {mx}]"


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
        rows.append(
            [
                name,
                status,
                escape(
                    format_summary(phase_summary(results, "measured"), duration=True)
                ),
                latency_pair(results, "insert"),
                latency_pair(results, "match"),
                escape(matched_blocks_per_request(results)),
                escape(request_shape(results)),
                escape(scenario_matched_depths(results)),
                escape(ops_per_second(results, "insert_calls")),
                escape(ops_per_second(results, "match_calls")),
                escape(loads_summary(results)),
                escape(node_watermark(results)),
            ]
        )
    return table(
        [
            "Case",
            "状态",
            "duration",
            "insert p50/p99/max",
            "match p50/p99/max",
            "avg matched blocks/request (device/host)",
            "request shape (insert path/new nodes/match keys)",
            "matched depth (continuation/fork/cold)",
            "insert ops/s",
            "match ops/s",
            "loads (committed/succeeded)",
            "节点水位 avg [min,max]",
        ],
        rows,
    )


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
            rows.append([name, status] + ["-"] * 8)
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
            "working set requested/addressable/visited",
            "failed",
        ],
        rows,
    )
    return html


def bandwidth_judgment(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Generate a bandwidth comparison table (placeholder for now)."""
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
        [
            "Device↔Host",
            "PCIe Gen4 x16 理论单向 ~32 GB/s、双向 ~64 GB/s",
            "-",
            "待分析",
        ],
        ["Host↔Disk (direct)", "云盘 O_DIRECT 实测带宽", "-", "待分析"],
        ["Device↔Disk (direct)", "云盘 O_DIRECT 实测带宽", "-", "待分析"],
        [
            "Host↔Disk (buffered)",
            "page cache 吸收后受 host 内存带宽约束",
            "-",
            "待分析",
        ],
        ["Device↔Disk (buffered)", "page cache + CUDA copy 组合路径", "-", "待分析"],
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
                rows[0] = [rows[0][0], rows[0][1], thru, "待分析"]
        elif "device_disk" in name:
            idx = 2 if "direct" in name else 4
            if rows[idx][2] == "-":
                rows[idx] = [rows[idx][0], rows[idx][1], thru, "待分析"]
        elif "host_disk" in name:
            idx = 2 if "direct" in name else 3
            if rows[idx][2] == "-":
                rows[idx] = [rows[idx][0], rows[idx][1], thru, "待分析"]

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
    rows = [
        [escape(key), f"<code>{escape(value)}</code>"]
        for key, value in manifest.get("environment", {}).items()
    ]
    env_table = table(["项", "采集值"], rows)
    invocation = manifest.get("invocation", {})
    suite = invocation.get("suite", manifest.get("suite", "profile"))
    cmd_parts = ["block_tree_cache_benchmark_driver", "--suite", str(suite)]
    if invocation.get("case", "all") != "all":
        cmd_parts.extend(["--case", str(invocation["case"])])
    option_pairs = [
        ("--process-repetitions", invocation.get("process_repetitions")),
        ("--min-measured-seconds", invocation.get("min_measured_seconds")),
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
    scope = "完整 canonical suite" if total == canonical_total else "专项复测"

    bullets = [
        f"<li>{'完整' if complete else '不完整'} {scope}：{completed}/{total} completed，{partial} partial，{failed} failed，{skipped} skipped。</li>",
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


def tree_observations(
    cases: List[Tuple[Dict, List[Dict]]], report_metadata: Dict = None
) -> str:
    """Auto-generate Tree observations from data, leaving blank for manual refinement."""
    # Find tree cases
    tree_data = []
    for case_manifest, results in cases:
        if case_manifest.get("subcommand") != "tree" or not results:
            continue
        name = case_manifest["case"]
        m = results[0].get("metrics", {})
        ph = results[0].get("phases_ns", {})
        measured_s = ph.get("measured", 0) / 1e9
        insert_calls = m.get("insert_calls", 0)
        match_calls = m.get("match_calls", 0)
        tree_data.append((name, m, measured_s, insert_calls, match_calls))

    obs = []
    if len(tree_data) == 2:
        # Compare multi vs single thread
        mt = tree_data[0]
        st = tree_data[1]
        mt_i_ops = mt[3] / mt[2] if mt[2] > 0 else 0
        st_i_ops = st[3] / st[2] if st[2] > 0 else 0
        mt_m_ops = mt[4] / mt[2] if mt[2] > 0 else 0
        st_m_ops = st[4] / st[2] if st[2] > 0 else 0
        mt_insert_p50 = mt[1].get("insert_latency_ns_p50", 0) / 1e6
        st_insert_p50 = st[1].get("insert_latency_ns_p50", 0) / 1e6
        mt_match_p50 = mt[1].get("match_latency_ns_p50", 0) / 1e6
        st_match_p50 = st[1].get("match_latency_ns_p50", 0) / 1e6

        obs.append(
            f"<li>多线程相对单线程的吞吐对比：insert ops/s <strong>{human_number(mt_i_ops)}</strong> vs "
            f"<strong>{human_number(st_i_ops)}</strong>（{'低于' if mt_i_ops < st_i_ops else '高于' if mt_i_ops > st_i_ops else '持平'}单线程）；"
            f"match ops/s <strong>{human_number(mt_m_ops)}</strong> vs <strong>{human_number(st_m_ops)}</strong>。"
            "单次 repetition 不能证明稳定收益，需结合多 repetition 波动判断。</li>"
        )
        ratio = mt_insert_p50 / st_insert_p50 if st_insert_p50 > 0 else 0
        obs.append(
            f"<li>多线程显著放大调用时延：insert p50 <strong>{human_number(mt_insert_p50)} ms</strong> vs "
            f"<strong>{human_number(st_insert_p50)} ms</strong>（约 {human_number(ratio)}×）；"
            f"match p50 <strong>{human_number(mt_match_p50)} ms</strong> vs "
            f"<strong>{human_number(st_match_p50)} ms</strong>。"
            f"原因需结合本次 perf 热点判断，不能仅凭时延差异归因于锁竞争。</li>"
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
            "可关注方向：锁竞争热点（mutex/malloc/红黑树）、eviction 淘汰路径的 CPU 占比、load 异步流水线效率等。"
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


def flamegraph_section(cases: List[Tuple[Dict, List[Dict]]]) -> str:
    """Enhanced perf section with sample quality table."""
    perf_rows = []
    sample_rows = []
    for case_manifest, _ in cases:
        perf = case_manifest.get("perf", {})
        perf_rows.append(
            [
                f"<code>{escape(case_manifest['case'])}</code>",
                escape(perf.get("status", "not collected")),
                escape(perf.get("mode", "off")),
                perf_artifact_link(case_manifest, "flamegraph", "打开 SVG"),
                perf_artifact_link(case_manifest, "perf_data", "下载 perf.data"),
                perf_artifact_link(case_manifest, "summary", "查看摘要"),
            ]
        )
        # Sample info from summary (placeholder)
        summary = perf.get("summary", "")
        sample_rows.append(
            [
                f"<code>{escape(case_manifest['case'])}</code>",
                escape(summary[:60] if summary else "待查看 perf_summary.txt"),
            ]
        )

    perf_table_html = table(
        ["Case", "状态", "模式", "CPU 火焰图", "原始数据", "文本摘要"],
        perf_rows,
    )

    guidance = (
        '<div class="notice">'
        "<p>说明：</p>"
        "<ul>"
        "<li>火焰图只反映 on-CPU 时间；磁盘等待、CUDA 流同步等 off-CPU 段不会出现在 CPU 火焰图里。</li>"
        "<li><code>[unknown]</code> 可能来自闭源库或缺少 frame pointer 的系统组件；归因前必须检查 build config、build-id cache 和 perf.data 属主。</li>"
        "<li>采样质量、样本数、lost samples、<code>[unknown]</code> 占比见 perf_summary.txt。</li>"
        "</ul>"
        "</div>"
    )
    return perf_table_html + guidance


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

  <section class="section"><h2>2. Tree 稳态场景</h2>
    <p>构造：100k 目标节点，scaled payload，8（或多/单线程）worker；build → warmup 10s → measured 30s；insert:match = 4:1。事件驱动淘汰，无后台线程。</p>
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

  <section class="section"><h2>4. 火焰图与采样质量</h2>{flamegraph_section(cases)}{perf_findings}</section>

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

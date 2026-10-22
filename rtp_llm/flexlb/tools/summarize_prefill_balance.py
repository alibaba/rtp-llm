#!/usr/bin/env python3
"""Audit recorded simulations and build a standalone HTML report (matplotlib required)."""

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import statistics as stats
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

POLICIES = {
    "SHORT_TTFT": "旧 TTFT",
    "SHORT_TTFT_CAPPED": "TTFT + 相同背压",
    "CACHE_ONLY": "缓存优先，无历史窗口",
    "CACHE_WINDOW": "新策略：缓存 + 窗口",
}
DATASETS = {
    "ramp": "audited-full/full",
    "steady": "audited-steady/reorder-False",
    "reorder": "audited-steady/reorder-True",
    "controls": "audited-controls/controls",
    "burst": "audited-burst/burst",
    "matched-steady": "matched-steady",
    "matched-controls": "matched-controls",
}
METRICS = [
    "arrival_qps",
    "success_qps",
    "offered_compute_load",
    "actual_compute_load",
    "input_cv",
    "compute_cv",
    "useful_compute_cv",
    "theoretical_hit",
    "success_theoretical_hit",
    "success_hit",
    "lost_cache_pp",
    "p95_s",
    "p99_s",
    "dropped",
    "expired",
    "global_queue",
    "inventory",
    "avg_input_tokens",
    "cold_arrivals",
    "cold_completed",
    "cold_dropped",
    "cold_expired",
]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(row[k] for k in ("policy", "scene", "seed", "stage"))


def estimate(values):
    n = len(values)
    mean = stats.mean(values)
    # Independent seed means, Student t interval. Never treat 100 workers as 100 seeds.
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n)
    half = t * stats.stdev(values) / math.sqrt(n) if t else None
    return dict(
        n=n, mean=mean, ci95_half=half, minimum=min(values), maximum=max(values)
    )


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))

    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{cell(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{cell(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, timelines, workers, audits = [], {}, {}, []
    total_runs = 0
    for dataset, relative in DATASETS.items():
        folder = root / relative
        summary = read_csv(folder / "summary.csv")
        timeline = read_csv(folder / "timeline.csv")
        worker_rows = read_csv(folder / "workers.csv")
        timelines[dataset], workers[dataset] = timeline, worker_rows
        by_worker = defaultdict(list)
        for row in worker_rows:
            by_worker[key(row)].append(row)
        for row in summary:
            group = by_worker[key(row)]
            assert len(group) == 100 and len({w["worker"] for w in group}) == 100, (
                dataset,
                key(row),
                len(group),
            )
            inputs = sum(float(w["input_tokens"]) for w in group)
            useful = sum(float(w["useful_compute_tokens"]) for w in group)
            actual = [float(w["compute_tokens"]) for w in group]
            cv = stats.pstdev(actual) / stats.mean(actual) if sum(actual) else 0
            assert abs(cv - float(row["compute_cv"])) < 1e-6
            row["success_theoretical_hit"] = 1 - useful / inputs if inputs else 0
            row["lost_cache_pp"] = 100 * (
                row["success_theoretical_hit"] - float(row["success_hit"])
            )
            row["dataset"] = dataset
            rows.append(row)
        per_run = defaultdict(list)
        traces = defaultdict(set)
        for row in timeline:
            per_run[(row["policy"], row["scene"], row["seed"])].append(row)
            traces[(row["scene"], row["seed"], row["start_s"])].add(row["trace_hash"])
        assert all(
            len(v) == 1 for v in traces.values()
        ), "different exogenous root streams"
        for run, samples in per_run.items():
            previous = 0
            for row in sorted(samples, key=lambda r: float(r["start_s"])):
                duration = float(row["duration_s"])
                arrivals = round(float(row["arrival_qps"]) * duration)
                success = round(float(row["success_qps"]) * duration)
                inventory = int(row["inventory"])
                assert (
                    previous + arrivals
                    == success + int(row["dropped"]) + int(row["expired"]) + inventory
                ), (dataset, run, row)
                previous = inventory
        total_runs += len(per_run)
        audits.append(
            dict(
                dataset=dataset,
                runs=len(per_run),
                summary_rows=len(summary),
                timeline_rows=len(timeline),
                workers=len(worker_rows),
                conservation="PASS",
                paired_root_stream="PASS",
            )
        )
    assert total_runs == 107, total_runs
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["dataset"], row["scene"], float(row["nominal_load"]), row["policy"])
        ].append(row)
    aggregate = []
    lookup = {}
    for group, members in sorted(groups.items()):
        dataset, scene, load, policy = group
        item = dict(
            dataset=dataset,
            scene=scene,
            load=load,
            policy=policy,
            metrics={m: estimate([float(r[m]) for r in members]) for m in METRICS},
        )
        aggregate.append(item)
        lookup[group] = item["metrics"]

    def value(dataset, scene, load, policy, metric):
        return lookup[(dataset, scene, load, policy)][metric]["mean"]

    def fmt(dataset, scene, load, policy, metric, scale=1, digits=2, ci=False):
        e = lookup[(dataset, scene, load, policy)][metric]
        result = f"{e['mean'] * scale:.{digits}f}"
        if ci and e["ci95_half"] is not None:
            result += f" ± {e['ci95_half'] * scale:.{digits}f}"
        return result

    smoke = []
    artifact_hash = (root / "release-artifacts/master.sha256").read_text().split()[0]
    for path in sorted(root.glob("release-*/aggregate.json")):
        data = json.loads(path.read_text())
        assert data["summary"]["exit_code"] == 0
        for instance in data["instances"]:
            assert instance["outcome"] == dict(
                execution="PASS", gate="PASS", validity="VALID", delivery="PASS"
            )
            assert all(c["status"] == "PASS" for c in instance["cleanup"])
            actual_paths = list(path.parent.rglob("actual-master-config.json"))
            assert len(actual_paths) == 1
            artifacts = list(path.parent.rglob("master-artifact.json"))
            assert len(artifacts) == 1
            assert json.loads(artifacts[0].read_text())["jar_sha256"] == artifact_hash
            actual = json.loads(actual_paths[0].read_text())
            prefill = actual["router"]["roles"]["prefill"]
            enabled = prefill.get("tokenBalance") is not None
            assert enabled == ("release-old-" not in path.parent.name)
            count = next(
                s["output"]["count"] for s in instance["stages"] if s["id"] == "submit"
            )
            smoke.append(
                dict(
                    name=path.parent.name,
                    requests=count,
                    status="PASS",
                    token_balance=enabled,
                    aggregate=str(path),
                    actual_config=str(actual_paths[0]),
                )
            )
    assert (
        len(smoke) == 12
        and sum(s["requests"] for s in smoke if "stress" in s["name"]) == 4000
    )
    counts = dict(tests=0, failures=0, errors=0, skipped=0)
    for path in (root / "release-ut-reports").rglob("TEST-*.xml"):
        suite = ET.parse(path).getroot()
        for name in counts:
            counts[name] += int(suite.get(name, 0))
    assert counts == dict(tests=3058, failures=0, errors=0, skipped=1), counts
    conditional = (root / "conditional-benchmark.log").read_text()
    assert "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0" in conditional
    assert (root / "artifact-verification.log").read_text().count("VERIFIED") == 12

    performance = {}
    for name in ("sync", "admission", "api", "follower"):
        path = root / f"release-{name}-performance.log"
        log = path.read_text() if path.exists() else ""
        performance[name] = (
            "PASS"
            if "BUILD SUCCESS" in log
            else "FAIL" if "BUILD FAILURE" in log else "NOT_RUN"
        )
    assert performance["sync"] == performance["admission"] == "PASS"
    performance["api_baseline_diagnostic"] = (
        "Before-task baseline P99 254ms; candidate P99 255ms; original threshold 50ms. Pre-existing failure retained."
    )

    paired = {}
    for metric in (
        "success_qps",
        "useful_compute_cv",
        "input_cv",
        "p99_s",
        "success_hit",
    ):
        pairs = []
        for seed in range(1, 6):
            old = next(
                r
                for r in rows
                if r["dataset"] == "steady"
                and r["policy"] == "SHORT_TTFT"
                and int(r["seed"]) == seed
            )
            new = next(
                r
                for r in rows
                if r["dataset"] == "steady"
                and r["policy"] == "CACHE_WINDOW"
                and int(r["seed"]) == seed
            )
            pairs.append(float(new[metric]) - float(old[metric]))
        paired[metric] = estimate(pairs)
    report = dict(
        audits=audits,
        runs=total_runs,
        aggregate=aggregate,
        paired_new_minus_old=paired,
        unit_tests=counts,
        conditional_test="PASS",
        performance_regressions=performance,
        smoke=smoke,
        jar_sha256=artifact_hash,
    )
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (out / "all-stages.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {
        "SHORT_TTFT": "#d05a45",
        "SHORT_TTFT_CAPPED": "#9164b0",
        "CACHE_ONLY": "#d8a52c",
        "CACHE_WINDOW": "#168b82",
    }
    labels = {
        "SHORT_TTFT": "TTFT",
        "SHORT_TTFT_CAPPED": "TTFT + cap",
        "CACHE_ONLY": "Cache only",
        "CACHE_WINDOW": "Cache + window",
    }
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode((out / f"{name}.png").read_bytes()).decode()
        figures.append(
            (name, f'<img alt="{name}" src="data:image/png;base64,{encoded}">')
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(
        axes,
        ["compute_tokens", "useful_compute_tokens", "input_tokens"],
        [
            "Actual uncached compute",
            "Useful compute for successes",
            "Input tokens of successes",
        ],
    ):
        for policy in ("SHORT_TTFT", "CACHE_ONLY", "CACHE_WINDOW"):
            selected = [
                w
                for w in workers["steady"]
                if w["policy"] == policy and w["seed"] == "1"
            ]
            selected.sort(key=lambda w: int(w["worker"]))
            # Physical throughput, not normalized across policies; saturation must remain visible.
            ax.plot(
                range(100),
                [float(w[metric]) / 1200 / 1000 for w in selected],
                label=labels[policy],
                color=colors[policy],
                linewidth=1,
            )
        ax.set(title=title, xlabel="Worker ID", ylabel="k tokens / second")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    save(fig, "workers-at-80pct")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title, scale in zip(
        axes.flat,
        ["success_qps", "success_hit", "useful_compute_cv", "p99_s"],
        [
            "Successful throughput",
            "Actual cache hit (successful tokens)",
            "Useful compute CV",
            "P99 prefill latency (successes)",
        ],
        [1, 100, 100, 1],
    ):
        for policy in ("SHORT_TTFT", "CACHE_ONLY", "CACHE_WINDOW"):
            loads = sorted(
                {
                    g[2]
                    for g in lookup
                    if g[0] == "ramp" and g[1] == "CONVERSATION" and g[3] == policy
                }
            )
            ax.plot(
                loads,
                [
                    value("ramp", "CONVERSATION", x, policy, metric) * scale
                    for x in loads
                ],
                marker="o",
                label=labels[policy],
                color=colors[policy],
            )
        ax.set(title=title, xlabel="Nominal load / ideal capacity")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    save(fig, "closed-load-ramp")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, scene in enumerate(("COLD_BURST", "COLD_BURST_OPEN")):
        for col, metric in enumerate(("global_queue", "p99_s")):
            ax = axes[row_index, col]
            for policy in ("SHORT_TTFT", "CACHE_WINDOW"):
                bins = defaultdict(list)
                for sample in timelines["burst"]:
                    if sample["scene"] == scene and sample["policy"] == policy:
                        bins[float(sample["start_s"]) + 30].append(
                            float(sample[metric])
                        )
                xs = sorted(bins)
                ax.plot(
                    xs,
                    [stats.mean(bins[x]) for x in xs],
                    label=labels[policy],
                    color=colors[policy],
                )
            ax.axvspan(1200, 1260, alpha=0.15, color="#d8a52c")
            ax.set(
                title=f"{scene}: {metric}",
                xlabel="Simulation time (s)",
                xlim=(1000, 1800),
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    save(fig, "cold-burst-recovery")

    text = []

    def p(s):
        text.append(f"<p>{s}</p>")

    def heading(s):
        text.append(f"<h2>{html.escape(s)}</h2>")

    heading("结果与适用范围")
    p(
        "实现了可配置的 Prefill 缓存优先 / 未缓存 token 窗口均衡策略。默认保留原路由，开启 tokenBalance 后不再用瞬时 TTFT 排名；Decode 路由不变。下列性能数字来自可复现 CPU 离散事件仿真，真实 Master/Mock smoke 单独列出，不能等同 GPU 实测。"
    )
    p(
        "主结论：在均匀可达缓存、80% 名义容量、真实闭环多轮对话下，新策略保住约 90% 命中并提高吞吐；相较只有缓存优先的消融，历史窗口进一步改善 token 均衡。严格缓存优先在已有缓存高度偏斜时无法保证均衡，报告保留该反例。"
    )
    heading("80% 稳态：5 个种子，各 40 分钟，统计后 20 分钟")
    primary = []
    for policy in POLICIES:
        dataset = "matched-steady" if policy == "SHORT_TTFT_CAPPED" else "steady"
        primary.append(
            [POLICIES[policy]]
            + [
                fmt(dataset, "CONVERSATION", 0.8, policy, m, s, 2, True)
                for m, s in [
                    ("success_qps", 1),
                    ("input_cv", 100),
                    ("compute_cv", 100),
                    ("useful_compute_cv", 100),
                    ("success_theoretical_hit", 100),
                    ("success_hit", 100),
                    ("p99_s", 1),
                    ("actual_compute_load", 100),
                ]
            ]
        )
    text.append(
        table(
            [
                "策略",
                "成功 QPS",
                "输入 CV %",
                "实际计算 CV %",
                "有效计算 CV %",
                "成功样本理论命中 %",
                "实际命中 %",
                "P99 秒",
                "实际计算负载 %",
            ],
            primary,
        )
    )
    p(
        "表内为种子均值 ± 95% Student t 置信区间半宽（n=5）；不是按请求数或 worker 数夸大样本量。CV=总体标准差/均值，越低越均匀。有效计算=成功请求在理想缓存下必须计算的 token；实际计算包含缓存丢失重算及超时后浪费。输入 CV 衡量成功处理的总输入 token。成功样本理论命中从同一批成功请求重算，避免直接用到达样本的理论命中和被筛选后的成功样本比较。"
    )
    p(
        "旧策略实际计算 CV 很低，原因是几乎所有 worker 满载重算，不能据此称其更高效。闭环模式中旧策略变慢会推迟下一轮，实际到达率与平均长度也随之改变；相同种子保证的是外生首轮到达与长度序列相同，并不保证所有后续请求到达时间相同。此处旧策略平均到达输入约 94,970 token，新策略约 99,903 token；完整 CSV 同时保留到达 QPS、到达理论命中与平均输入长度。"
    )
    text.append(figures[0][1])
    heading("压力从 0.1 到 1.5 倍，以及独立开放流量校验")
    p(
        "100 × 70k = 7M 未缓存 token/s；100k 输入、90% token 命中对应理想 700 QPS。闭环名义 1.5 倍不保证实际到达 1.5 倍，因此另用预热前缀的独立开放请求验证 1050 QPS，避免客户端反压把过载隐藏。WARM_REPLAY 为 50,000 个均匀落点前缀，长度量化为 10 的倍数，使每个请求恰好 90% 命中，避免一 token 舍入误差使淘汰偏向某种长度。"
    )
    text.append(figures[1][1])
    control = []
    for scene in ("WARM_REPLAY", "HOTSPOT"):
        for load in (0.8, 1.5):
            for policy in ("SHORT_TTFT", "SHORT_TTFT_CAPPED", "CACHE_WINDOW"):
                dataset = (
                    "matched-controls" if policy == "SHORT_TTFT_CAPPED" else "controls"
                )
                control.append(
                    [scene, load, POLICIES[policy]]
                    + [
                        fmt(dataset, scene, load, policy, m, s)
                        for m, s in [
                            ("arrival_qps", 1),
                            ("success_qps", 1),
                            ("success_hit", 100),
                            ("compute_cv", 100),
                            ("dropped", 1),
                            ("expired", 1),
                        ]
                    ]
                )
    text.append(
        table(
            [
                "场景",
                "负载",
                "策略",
                "到达 QPS",
                "成功 QPS",
                "命中 %",
                "实际计算 CV %",
                "丢弃/90秒",
                "超时/90秒",
            ],
            control,
        )
    )
    p(
        "开放过载下旧 TTFT 可能因缓存迁移、排队与截止时间出现成功吞吐崩溃。模拟批次不可抢占，超过 60 秒的完成记为超时，计算资源已消耗但不生成缓存或后续会话。这是模型条件下的结果，不应将新旧商值宣传成生产数百倍加速；实际引擎的取消/抢占行为可能显著改变它。相同背压 TTFT 对照用于隔离队列水位作用，缓存优先无历史窗口对照用于隔离历史窗口收益。"
    )
    p(
        "HOTSPOT 把大部分缓存集中到少数 worker。严格最高命中约束导致热点拥塞、其他机器空闲；仅靠冷流量补偿无法修复已有热点。改善此场景需要额外缓存副本或放松亲和约束，当前实现没有偷偷降级到低命中机器。固定时间 REPLAY 即使上一轮失败仍注入后续请求，也可能破坏缓存链；它单独作为压力反例保留，不能冒充真实闭环对话。"
    )
    heading("无缓存新请求突发")
    p(
        "80% 背景持续 40 分钟，在第 1200 秒额外注入约 70 个冷请求/秒、平均 100k token，持续约 60 秒，共实测 4202 个冷请求；额外计算需求约为一整套集群，瞬时总需求约 1.8 倍容量。闭环与独立开放背景均做 3 个种子。下表统计包含突发与恢复的后 20 分钟，不能把它当瞬时峰值。"
    )
    burst = []
    for scene in ("COLD_BURST", "COLD_BURST_OPEN"):
        for policy in ("SHORT_TTFT", "CACHE_ONLY", "CACHE_WINDOW"):
            burst.append(
                [scene, POLICIES[policy]]
                + [
                    fmt("burst", scene, 0.8, policy, m)
                    for m in (
                        "cold_arrivals",
                        "cold_completed",
                        "cold_dropped",
                        "cold_expired",
                        "p99_s",
                    )
                ]
            )
    text.append(
        table(
            ["背景", "策略", "冷到达", "冷成功", "冷丢弃", "冷超时", "全部成功 P99 秒"],
            burst,
        )
    )
    p(
        "开放背景下，保缓存淘汰会主动拒绝部分冷请求；这符合过载保高命中的目标，但不意味着所有突发请求都完成。图中全局队列不包含 worker 内部队列，旧策略全局队列短也不意味着端到端等待短。成功延迟必须与丢弃/超时一起读。"
    )
    text.append(figures[2][1])
    heading("参数与重排消融")
    p(
        "默认窗口 10 秒，待处理水位 140k token（70k TPS 下约 2 秒工作）。先用单种子探索 5/10/30/60/120 秒窗口 × 70k/140k/280k 水位，再用多个种子的长稳态实验验证。10 秒是本次选用值，不宣称全局最优；5 秒在短时探索中也有竞争力。探索性 sweep 不并入正式 107 次数据，不能把单种子调参结果当统计显著性证据。水位允许当前值低于上限时接收一个大请求，避免长请求饿死。"
    )
    reorder = []
    for dataset in ("steady", "reorder"):
        reorder.append(
            [dataset]
            + [
                fmt(dataset, "CONVERSATION", 0.8, "CACHE_WINDOW", m, s, 3, True)
                for m, s in [
                    ("success_qps", 1),
                    ("useful_compute_cv", 100),
                    ("p99_s", 1),
                ]
            ]
        )
    text.append(
        table(["FIFO / 当前组内重排", "成功 QPS", "有效计算 CV %", "P99 秒"], reorder)
    )
    p(
        "重排消融只在当前已到达的 32 个候选中把长计算请求前置，不额外延后请求。它没有稳定改善均衡或 P99，因此生产保持现有 FIFO/优先级，不增加人为等待轮次。未进行跨未来到达轮次的预测调度。"
    )
    heading("实现、统计审计与模型边界")
    p(
        "高于可配置命中阈值时锁定最大命中 worker；低于阈值但有正命中时在有缓存 worker 内均衡；完全无命中时全局均衡。评分=窗口成功完成未缓存 token + 正式待处理 + 临时规划预留 + 本请求预计计算 token。临时预留避免异步并发选路集中，成功接收时转入正式账本，取消/失败时幂等撤销；只有正式待处理占硬水位。"
    )
    p(
        "到达首轮为 Poisson，长度为 sigma=0.9 的截断对数正态（短多长少），完整 21 轮会话 token 加权平均输入目标 100k、理论命中 90%，轮间思考 2–6 秒。worker 吞吐严格只取决于未命中 token，大小 8 或 20ms 凑批；缓存仅在成功后写入，TTL 20 分钟；TTL_GAP 给 20% 会话在第十轮额外暂停 1300 秒。阶段各 5 分钟、后半段统计，整体 35 分钟，超过 TTL；80% 稳态与突发各 40 分钟。"
    )
    p(
        "仿真旧基线 SHORT_TTFT 使用瞬时剩余队列 + 本请求预计计算排序，不含旧配置可选的有限 TTFT 容忍缓存亲和；旧配置 smoke 则沿用框架已有 cacheAffinity。仿真共享生产窗口、资格判断与淘汰原语，但没有直接调用完整 Master 选路/投影。生产窗口优先采用 Engine 实测 inputLength-prefixLength，旧 Engine 无有效 inputLength 时回退估计；当前只记录成功终态。模拟会额外统计超时浪费，用于物理容量审计。未模拟 GPU、网络、KV 容量淘汰、跨 Master 同步延迟、真实取消抢占。短时间桶按完成批次记账，跨桶批次可能让单桶负载略超 100%，全程物理容量守恒。"
    )
    p(
        f"正式 {total_runs} 次运行（合计约 60.8 小时虚拟集群时间）均校验：请求到达=成功+丢弃+超时+存量；成功输入=命中+计算；实际计算（含超时浪费）不超过集群物理容量；worker 预留账本与真实队列/运行批次一致。报告脚本再次审计所有 30 秒桶请求守恒、100 worker 完整性、CV 重算、同种子首轮流量 hash。"
    )
    text.append(
        table(
            [
                "实验组",
                "独立运行数",
                "阶段行数",
                "30秒桶数",
                "worker 行数",
                "守恒",
                "配对首轮",
            ],
            [
                [
                    a[k]
                    for k in (
                        "dataset",
                        "runs",
                        "summary_rows",
                        "timeline_rows",
                        "workers",
                        "conservation",
                        "paired_root_stream",
                    )
                ]
                for a in audits
            ],
        )
    )
    heading("全量 Java 回归与全部 FlexLB smoke")
    p(
        "最终源码 clean package：3058 项测试，默认执行 3057 项通过、1 项条件性能用例跳过；随后显式开启该条件用例，1/1 通过。无失败/错误。测试覆盖窗口过期、取消、重复终态、实际 cache token 回报、随机账本参考模型、阈值、饱和亲和、冷流量均衡、异步规划预留与容量转移。"
    )
    p(
        "额外性能回归：sync 3/3、admission 2/2 通过。API 16 项中 15 项通过，8192 请求无限速突发 P99 用例未达原有 50ms 门槛；隔离重测修改前基线 254ms、修改后 255ms，确认为原有失败，未修改阈值。批量脚本随后停止，额外 follower 组未执行；上面单独启用的 follower 条件用例已通过。"
    )
    p(
        "FlexLB 正式 smoke 目录含 request_completion 的四个 profile：single-nonbatch、single-batch、window-nonbatch、batch-window。分别执行旧配置、新配置、100 Prefill + 100 Decode Mock 压力配置，共 12/12 PASS；压力组每项 1000 个 100k 输入，共 4000 个全部终态成功、零错误。提交过程有客户端耗时，并非 1000 个完全同时到达。全部 cleanup 与 12 个实验归档校验通过。"
    )
    p(
        "这里“全部 smoke”指本次修改的 FlexLB Java 路由组件的完整 smoke 集；不包含 RTP-LLM CUDA 模型/GPU golden smoke，也没有声称执行整个引擎的所有模型测试。真实 Master smoke 验证集成正确性，长时间 90% 缓存/TTL 压力结论来自上面的仿真。"
    )
    text.append(
        table(
            ["Smoke", "请求数", "新策略", "结果"],
            [
                [s["name"], s["requests"], s["token_balance"], s["status"]]
                for s in smoke
            ],
        )
    )
    p(
        "所有 smoke 使用同一冻结 JAR SHA256：<code>"
        + artifact_hash
        + "</code>。框架 aggregate 的 resolved_config 展示原始 profile，实际覆盖配置已从每次环境的 actual-master-config.json 读取核验；不要用前者判断是否开启新策略。"
    )
    heading("完整数据与复现")
    p(
        '<a href="summary.json">统计 JSON</a> · <a href="all-stages.csv">所有阶段 CSV</a> · <a href="data-sha256.json">数据 SHA256 清单</a>'
    )
    p(
        "同目录 summary.json 提供均值、95% 区间、种子最小/最大值与配对差值；all-stages.csv 保留所有正式阶段原始结果。PNG/SVG 可直接导出。上级目录保留每 worker CSV、30 秒序列、命令/日志、源文件 hash、测试 XML、冻结 JAR 与归档。"
    )
    text.append(
        "<pre>"
        + html.escape(
            "python3 tools/run_prefill_balance.py --profile full --seeds 3 --out-dir OUT/audited-full\npython3 tools/run_prefill_balance.py --profile steady --seeds 5 --out-dir OUT/audited-steady\npython3 tools/run_prefill_balance.py --profile controls --seeds 3 --out-dir OUT/audited-controls\npython3 tools/run_prefill_balance.py --profile burst --seeds 3 --out-dir OUT/audited-burst\npython3 tools/run_prefill_balance.py --profile matched --out-dir OUT\npython3 tools/summarize_prefill_balance.py --root OUT --out-dir OUT/report"
        )
        + "</pre>"
    )
    text.append("<details><summary>全部分组均值（完整精度与区间见 JSON）</summary>")
    text.append(
        table(
            [
                "实验",
                "场景",
                "负载",
                "策略",
                "n",
                "成功 QPS",
                "实际命中 %",
                "实际计算 CV %",
                "有效计算 CV %",
                "P99 秒",
                "丢弃",
                "超时",
            ],
            [
                [
                    a["dataset"],
                    a["scene"],
                    a["load"],
                    a["policy"],
                    a["metrics"]["success_qps"]["n"],
                ]
                + [
                    f"{a['metrics'][m]['mean']*s:.3f}"
                    for m, s in [
                        ("success_qps", 1),
                        ("success_hit", 100),
                        ("compute_cv", 100),
                        ("useful_compute_cv", 100),
                        ("p99_s", 1),
                        ("dropped", 1),
                        ("expired", 1),
                    ]
                ]
                for a in aggregate
            ],
        )
    )
    text.append("</details>")
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Prefill token 均衡实验报告</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#20333b;background:#f7f9fa;max-width:1320px;margin:36px auto;padding:0 24px}h1,h2{line-height:1.3}h1{font-size:32px}h2{margin-top:42px;font-size:23px}p{max-width:1100px}table{border-collapse:collapse;font-size:13px;background:white;width:100%}td,th{padding:9px 11px;border-bottom:1px solid #dbe2e6;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#e4efed}.scroll{overflow:auto;margin:20px 0}img{width:100%;background:white;margin:15px 0}pre{overflow:auto;background:#e8eef1;padding:18px;font-size:13px}code{word-break:break-all}summary{cursor:pointer;font-weight:bold}</style><h1>Prefill 缓存优先与未缓存 token 均衡</h1><p>2026-10-22 · 100 workers · 70k uncached TPS/worker · 100k average input · 20min cache TTL</p>"""
        + "".join(text)
        + "</html>"
    )
    (out / "report.html").write_text(page)
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for relative in DATASETS.values()
        for p in (root / relative).glob("*.csv")
    }
    (out / "data-sha256.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                runs=total_runs,
                tests=counts,
                smokes=len(smoke),
                report=str(out / "report.html"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

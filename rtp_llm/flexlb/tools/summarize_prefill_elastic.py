#!/usr/bin/env python3
"""Audit elastic Prefill experiments and emit a standalone HTML report plus raw CSV/JSON."""

import argparse
import base64
import csv
import hashlib
import html
import json
import re
import statistics as stats
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from summarize_prefill_balance import estimate, read_csv, table

POLICIES = {
    "CACHE_WINDOW": "Strict cache + window",
    "ELASTIC": "Idle-worker rescue",
    "ADAPTIVE": "Rescue + recovery",
}
METRICS = (
    "success_qps",
    "arrival_qps",
    "compute_cv",
    "input_cv",
    "useful_compute_cv",
    "success_hit",
    "p99_s",
    "dropped",
    "expired",
    "actual_compute_load",
    "cold_completed",
    "cold_dropped",
    "cold_expired",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((root / "full/runs.json").read_text())
    assert len(results) == 4 and all(r["returncode"] == 0 for r in results)
    rows, timelines, workers, audits, priorities = [], [], [], [], []
    hashes = {}
    for result in results:
        folder = root / "full" / result["name"]
        for path in folder.glob("*.csv"):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary, timeline, worker = [
            read_csv(folder / name)
            for name in ("summary.csv", "timeline.csv", "workers.csv")
        ]
        groups = defaultdict(list)
        for w in worker:
            groups[tuple(w[k] for k in ("policy", "scene", "seed", "stage"))].append(w)
        for r in summary:
            members = groups[tuple(r[k] for k in ("policy", "scene", "seed", "stage"))]
            count = int(r["active_workers"])
            assert len(members) == count and {int(w["worker"]) for w in members} == set(
                range(count)
            )
            compute = [float(w["compute_tokens"]) for w in members]
            cv = stats.pstdev(compute) / stats.mean(compute) if sum(compute) else 0
            assert abs(cv - float(r["compute_cv"])) < 1e-6
            total_input = sum(float(w["input_tokens"]) for w in members)
            useful = sum(float(w["useful_compute_tokens"]) for w in members)
            r["success_theoretical_hit"] = (
                1 - useful / total_input if total_input else 0
            )
        runs, traces = defaultdict(list), defaultdict(set)
        for r in timeline:
            runs[tuple(r[k] for k in ("policy", "scene", "seed"))].append(r)
            traces[tuple(r[k] for k in ("scene", "seed", "start_s"))].add(
                r["trace_hash"]
            )
        assert all(
            len(t) == 1 for t in traces.values()
        ), "unpaired external arrival streams"
        for key, samples in runs.items():
            previous = 0
            for r in sorted(samples, key=lambda r: float(r["start_s"])):
                seconds = float(r["duration_s"])
                arrived, completed = [
                    round(float(r[k]) * seconds) for k in ("arrival_qps", "success_qps")
                ]
                assert previous + arrived == completed + int(r["dropped"]) + int(
                    r["expired"]
                ) + int(r["inventory"]), (key, r)
                previous = int(r["inventory"])
        log = (root / "full" / f"{result['name']}.log").read_text()
        assert log.count("PASS ") == len(runs)
        for match in re.finditer(
            r"ELASTIC policy=(\w+) scene=(\w+) seed=(\d+).*?drops_priority=(\[[^\]]+\]) success_priority=(\[[^\]]+\])",
            log,
        ):
            priorities.append(
                dict(
                    policy=match[1],
                    scene=match[2],
                    seed=int(match[3]),
                    dropped=json.loads(match[4]),
                    completed=json.loads(match[5]),
                )
            )
        audits.append(
            dict(
                dataset=result["name"],
                runs=len(runs),
                conservation="PASS",
                paired_arrivals="PASS",
                worker_completeness="PASS",
            )
        )
        rows.extend(summary)
        timelines.extend(timeline)
        workers.extend(worker)
    assert sum(a["runs"] for a in audits) == 36
    aggregate = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], int(r["stage"]), r["policy"])].append(r)
    for (scene, stage, policy), members in sorted(groups.items()):
        assert len(members) == 3
        aggregate.append(
            dict(
                scene=scene,
                stage=stage,
                policy=policy,
                metrics={k: estimate([float(r[k]) for r in members]) for k in METRICS},
            )
        )
    counts = dict(tests=0, failures=0, errors=0, skipped=0)
    for path in (root / "release-ut-reports").rglob("TEST-*.xml"):
        suite = ET.parse(path).getroot()
        for k in counts:
            counts[k] += int(suite.get(k, 0))
    assert (
        counts["tests"] >= 3058
        and counts["failures"] == counts["errors"] == 0
        and counts["skipped"] == 1
    ), counts
    assert (
        "Tests run: 1, Failures: 0, Errors: 0, Skipped: 0"
        in (root / "conditional-benchmark.log").read_text()
    )
    assert (
        counts == json.loads((root / "ut-report-audit.json").read_text())["current_run"]
    )
    assert (root / "artifact-verification.log").read_text().count("VERIFIED ") == 16
    jar_hash = hashlib.sha256(
        (root / "release-artifacts/master.jar").read_bytes()
    ).hexdigest()
    smokes = []
    for path in sorted(root.glob("release-*/aggregate.json")):
        data = json.loads(path.read_text())
        assert data["summary"]["exit_code"] == 0
        for instance in data["instances"]:
            assert instance["outcome"] == dict(
                execution="PASS", gate="PASS", validity="VALID", delivery="PASS"
            )
            assert all(c["status"] == "PASS" for c in instance["cleanup"])
        actual = list(path.parent.rglob("actual-master-config.json"))
        assert len(actual) == 1
        config = json.loads(actual[0].read_text())
        balance = config["router"]["roles"]["prefill"].get("tokenBalance")
        kind = path.parent.name.split("-")[1]
        assert bool(balance is not None) == (kind != "old")
        if kind in ("new", "stress"):
            assert balance["hotspotRescue"]["maxInputTokens"] == 131072
            assert config["scheduler"]["prefillQueueShedding"]["recovery"] is not None
        artifact = list(path.parent.rglob("master-artifact.json"))
        assert artifact and jar_hash in artifact[0].read_text()
        smokes.append(
            dict(
                name=path.parent.name,
                status="PASS",
                aggregate=str(path),
                actual_config=str(actual[0]),
            )
        )
    assert len(smokes) == 16
    assert (root / "verification-completed.txt").exists()
    tuning = json.loads((root / "recovery-tuning.json").read_text())
    report = dict(
        simulator_fast_path=json.loads(
            (root / "simulation-fast-path.json").read_text()
        ),
        recovery_tuning=tuning,
        audits=audits,
        unit_tests=counts,
        conditional_benchmark="PASS",
        master_sha256=jar_hash,
        smokes=smokes,
        aggregate=aggregate,
        priority_outcomes=priorities,
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "data-sha256.json").write_text(json.dumps(hashes, indent=2))
    with (out / "all-stages.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.2})
    images = []
    for scene in ("SCALE_OUT", "HOTSPOT_SCALE_OUT", "RAMP", "RECOVERY_BURST"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for policy, label in POLICIES.items():
            series = defaultdict(list)
            for r in timelines:
                if r["scene"] == scene and r["policy"] == policy:
                    series[float(r["start_s"])].append(r)
            xs = sorted(series)
            for ax, metric in zip(axes, ("success_qps", "compute_cv", "success_hit")):
                ax.plot(
                    [x / 60 for x in xs],
                    [stats.mean(float(r[metric]) for r in series[x]) for x in xs],
                    label=label,
                )
                ax.set_ylabel(metric)
                if "SCALE_OUT" in scene:
                    ax.axvline(22.5, color="black", ls="--", alpha=0.5)
                if scene == "RECOVERY_BURST":
                    ax.axvline(15, color="black", ls="--", alpha=0.5)
        axes[0].legend(loc="best")
        axes[0].set_title(scene + " (3 paired seeds)")
        axes[-1].set_xlabel("Virtual time (minutes)")
        fig.tight_layout()
        name = scene.lower()
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)
        images.append(
            (scene, base64.b64encode((out / f"{name}.png").read_bytes()).decode())
        )
    for scene in ("SCALE_OUT", "HOTSPOT_SCALE_OUT"):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for policy, label in POLICIES.items():
            group = defaultdict(list)
            for row in workers:
                if (
                    row["scene"] == scene
                    and row["policy"] == policy
                    and row["stage"] == "5"
                ):
                    group[int(row["worker"])].append(row)
            assert set(group) == set(range(150))
            for ax, metric in zip(axes, ("compute_tokens", "input_tokens")):
                ax.plot(
                    range(150),
                    [
                        stats.mean(float(r[metric]) for r in group[i]) / 150 / 1000
                        for i in range(150)
                    ],
                    label=label,
                )
                ax.set_ylabel(metric + " (k/s)")
                ax.axvline(99.5, color="black", linestyle="--", alpha=0.6)
        axes[0].legend()
        axes[0].set_title(
            scene + " final 150 seconds; workers 100–149 added at 22.5 min"
        )
        axes[-1].set_xlabel("Worker ID (3-seed mean)")
        fig.tight_layout()
        name = scene.lower() + "-workers"
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)
        images.append(
            (
                scene + " per-worker",
                base64.b64encode((out / f"{name}.png").read_bytes()).decode(),
            )
        )
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Prefill 扩容与突发恢复实验</title>',
        "<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172132}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{width:100%}code{background:#eef;padding:2px 5px}</style>",
        "<h1>Prefill 扩容、热点迁移与突发恢复</h1>",
        "<p>本报告补充原始 TTFT / 缓存窗口均衡对比。这里比较严格缓存优先、空闲 worker 迁移、迁移加流量恢复三个策略。CPU 离散事件模拟使用生产的窗口账本和淘汰/恢复原语，不是完整 Master 或 GPU 性能实测；真实 Master/Mock smoke 单独验证。</p>",
        "<h2>试验条件与审计</h2>",
        "<p>初始 100 个 worker，每个仅按未命中 token 以恒定 70k TPS 处理；输入长度长尾随机、平均 100k、预热理论 token 命中 90%、cache TTL 20 分钟。开放流量逐级 0.2 / 0.5 / 0.8 / 1.2 / 1.5 / 1.5 倍，每级 300 秒；第 22.5 分钟扩至 150 个 worker。热点场景把 80% cache 放在 10 个 worker。突发场景在 80% 背景负载第 15 分钟一次注入 3000 个无 cache 请求。每组 3 个配对种子，共 36 次、18 小时虚拟集群运行。</p>",
        "<p>迁移门槛为等待 5 秒、总输入 ≤131072 token、目标未缓存 pending+planned 为零；原 worker 水位 140k。一次本地撤回后保留身份与截止时间，不迁移已派发请求。恢复采用 2 秒窗口、4 秒观察期、最多 10 秒恢复预算；软水位 30 秒，硬截止 60 秒。所有窗口均检查请求守恒、逐 worker 完整性与相同外部流量，完整 CSV 包含全部阶段。</p>",
        table(
            ["审计数据集", "运行数", "守恒", "配对流量", "worker"],
            [
                [
                    a["dataset"],
                    a["runs"],
                    a["conservation"],
                    a["paired_arrivals"],
                    a["worker_completeness"],
                ]
                for a in audits
            ],
        ),
        "<h2>阶段结果</h2><p>以下为每阶段后半段的 3 种子均值。CV 越低表示越均匀；必须同时观察吞吐、cache 和算力利用，避免把所有 worker 忙于重算误认为效率提高。扩容表选最后 150 秒，突发表选注入后 900 秒。P99 是各次实验分位数的均值；JSON 另含种子间 95% t 区间（3 种子不确定性较大）。</p>",
    ]
    for scene in ("SCALE_OUT", "HOTSPOT_SCALE_OUT", "RAMP", "RECOVERY_BURST"):
        selected = [
            a
            for a in aggregate
            if a["scene"] == scene
            and a["stage"] == (0 if scene == "RECOVERY_BURST" else 5)
        ]
        parts.append("<h3>" + html.escape(scene) + "</h3>")
        parts.append(
            table(
                [
                    "策略",
                    "成功 QPS",
                    "计算 CV %",
                    "输入 CV %",
                    "实际命中 %",
                    "P99 秒",
                    "淘汰",
                    "超时",
                    "算力利用 %",
                ],
                [
                    [
                        POLICIES[a["policy"]],
                        *[
                            f"{a['metrics'][k]['mean']*scale:.2f}"
                            for k, scale in [
                                ("success_qps", 1),
                                ("compute_cv", 100),
                                ("input_cv", 100),
                                ("success_hit", 100),
                                ("p99_s", 1),
                                ("dropped", 1),
                                ("expired", 1),
                                ("actual_compute_load", 100),
                            ]
                        ],
                    ]
                    for a in selected
                ],
            )
        )
    parts.append("<h2>突发冷请求与优先级</h2>")
    parts.append(
        table(
            ["策略", "冷请求成功", "冷请求淘汰", "冷请求超时"],
            [
                [
                    POLICIES[a["policy"]],
                    *[
                        f"{a['metrics'][k]['mean']:.1f}"
                        for k in ("cold_completed", "cold_dropped", "cold_expired")
                    ],
                ]
                for a in aggregate
                if a["scene"] == "RECOVERY_BURST"
            ],
        )
    )
    parts.append(
        "<p>优先级为 10/50/90，淘汰比较器首先淘汰低优先级，其次无 cache，再比较命中率；同分保留较早请求。全程各优先级成功/淘汰数量保存在 summary.json。跨时间的总数不能单独证明优先级顺序，真实队列和比较器回归另外覆盖该约束。</p>"
    )
    parts.append(
        "<p>模拟迭代优化：对没有空闲 worker 的迁移候选提前返回，并仅扫描实际有 cache 的候选；两轮交替测量从平均 101.46s 降至 60.10s（约 40.8%）。每轮九组配对场景的全部 CSV 逐字节一致。两个扩容数据集使用优化前模拟源码，持续上涨和突发数据集使用等价快路径，版本 SHA 与验证记录保存于 simulation-fast-path.json；生产调度代码未因此改变。此前排序容器调整没有显著运行时间收益，不计为加速。</p>"
    )
    parts.append(
        "<p>探索性参数反例（单种子，不作为统计结论）：把检测窗口/观察期/最大宽限从 2s/4s/10s 加长到 4s/8s/60s，冷请求成功从 2341 增至 2385，但总淘汰从 1268 增至 1538，P99 从 33.531s 增至 33.969s，故未采用。到达速率下降不等于积压已经持续下降；批次完成抖动与迁移重算会使恢复判断中断。保守默认恢复配置同样可能以暖请求延迟和总淘汰换取更多冷请求完成，保持独立、默认关闭。</p>"
    )
    for scene, image in images:
        parts.append(
            f'<h2>{scene}</h2><img src="data:image/png;base64,{image}" alt="{scene}">'
        )
    parts.extend(
        [
            "<h2>实现边界</h2>",
            "<p>此版本只把尚未发送的短请求移至空闲机器；长请求、无空闲容量和已进入 Engine 的请求仍受硬截止限制。新机器没有旧缓存，迁移必然可能增加重算；并不承诺在所有热点分布下达到 90% 命中与完全均匀两个目标。已经返回客户端的失败不可后台复活，补救发生在最终拒绝前，客户端新重试正常参与新集群调度。</p>",
            "<p>TTL 自然到期也会使严格缓存策略开始使用新增 worker，因此应以配对曲线比较迁移收益，而不是把所有新增 worker 工作归因于迁移。模拟使用不可抢占批次，超时完成产生的计算计入原始工作，但不会写入 cache；真实 Engine 的取消行为可能不同。此次扩容补充是预热独立请求，原始报告另含闭环多轮对话与冷突发，不混淆两种流量模型。</p>",
            "<h2>实际回归与交付</h2>",
            f'<p>Java 默认测试 {counts["tests"]} 项：失败/错误 0，条件跳过 1；显式启用该条件用例后 1/1 通过。全部 4 个 FlexLB smoke profile 分别在旧策略、严格新策略、迁移恢复策略、100-worker 压测配置执行，共 16/16 通过。这里的 smoke 范围是 FlexLB 调度器目录，不包含 RTP-LLM GPU 模型 golden 全量套件。</p>',
            f"<p>冻结 Master SHA256：<code>{jar_hash}</code>。每个 smoke 实际配置与同一个 jar 均核验，保留原始日志和 aggregate.json。额外 8192 无限速 API 性能断言仍有原有失败：原基线 P99 254ms、修改后 255ms，门槛 50ms，未篡改门槛；详情见原始报告。</p>",
            table(["smoke", "状态"], [[s["name"], s["status"]] for s in smokes]),
            "<p>实时工具：<code>python3 tools/watch_prefill_traffic.py --url http://MASTER:PORT --samples 30</code>；配置与行为说明见 docs/prefill-token-balance.md。所有新功能默认关闭，建议显式配置后按命中率、实际计算 TPS、积压及超时联合评估。</p>",
        ]
    )
    (out / "report.html").write_text("\n".join(parts))
    print(
        json.dumps(
            dict(
                runs=36,
                unit_tests=counts,
                smokes=len(smokes),
                report=str(out / "report.html"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

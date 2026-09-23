"""Offline comparison of two completed cache-scale-in runs."""

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from reporting import write_bundle, run_meta, compare_controls, details
from traffic.playback_config import comparison_notice
from workload.cache_gate import analyze, write_report
from workload.cache_comparison_config import validate_policy

REQUIRED = (
    "/criteria", "/client_environment", "/mock_jar_sha256",
    "/topology", "/capacity", "/performance", "/master_config",
    "/actual_master_config", "/configuration_sha256", "/trace_sha256",
)
CORE_METRICS = {"P Waiting / engine", "P engine count", "P cache hit ratio"}


def _load(path):
    path = Path(path)
    if path.is_dir():
        direct = path / "cache-gate-evidence.json"
        if direct.is_file():
            path = direct
        else:
            found = list(path.rglob("cache-gate-evidence.json"))
            if len(found) != 1:
                raise ValueError(f"expected one evidence file in {path}, found {len(found)}")
            path = found[0]
    return path, json.loads(path.read_text())


def _controls(e):
    p = e["provenance"]
    mock = [v for k, v in p.get("files", {}).items()
            if Path(k).name.startswith("flexlb-mock-engine-") and k.endswith(".jar")]
    return dict(
        criteria=e.get("criteria"),
        client_environment=p.get("client_environment"),
        topology=p.get("topology"),
        capacity=p.get("capacity"),
        performance=p.get("performance"),
        master_config=p.get("master_config"),
        actual_master_config=p.get("actual_master_config"),
        configuration_sha256=p.get("configuration_sha256"),
        mock_jar_sha256=mock[0] if len(mock) == 1 else None,
        trace_sha256=p.get("trace", {}).get("sha256"),
    )


def compare(a_path, b_path, output, *, alignment_event=None):
    validate_policy(dict(alignment_event=alignment_event))
    resolved = [_load(p) for p in (a_path, b_path)]
    paths, evidence = zip(*resolved)
    identities = [e["provenance"].get("master_artifact") or {} for e in evidence]
    controls = [_controls(e) for e in evidence]
    alignment = compare_controls(*controls, required=REQUIRED)
    notice = comparison_notice(
        evidence[0]["provenance"].get("trace", {}),
        evidence[1]["provenance"].get("trace", {}),
    )
    if notice:
        alignment["aligned"] = False
        alignment["status"] = "DIFFERENT"
        alignment["traffic_semantics"] = notice
    event_times = []
    for e in evidence:
        matches = [v.get("t") for v in e.get("events", [])
                   if alignment_event is not None and v.get("name") == alignment_event]
        event_times.append(matches[0] if len(matches) == 1
                           and type(matches[0]) in (int, float)
                           and math.isfinite(matches[0]) else None)
    event_aligned = alignment_event is not None and all(t is not None for t in event_times)
    time_alignment = dict(
        event=alignment_event,
        status="ALIGNED" if event_aligned else "UNAVAILABLE" if alignment_event else "NOT_REQUESTED",
        observed_times=dict(zip(("A", "B"), event_times)),
        missing_or_ambiguous_runs=[label for label, t in zip(("A", "B"), event_times)
                                   if alignment_event and t is None],
    )
    time_caption = (f"按 {alignment_event} 对齐到 0 秒。" if event_aligned else
                    f"对齐事件 {alignment_event} 缺失、重复或时间无效，保留各 run 原时间轴。"
                    if alignment_event else "保留各 run 原时间轴。")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels, combined, results = [], [], []
    start, end = 0, 1
    for label, path, e, shift, identity in zip(
        ("A", "B"), paths, evidence, event_times, identities
    ):
        directory = output / label
        directory.mkdir(exist_ok=True)
        for archive in path.parent.glob("telemetry/*/queries.json"):
            target = directory / archive.relative_to(path.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, target)
        result = analyze(e)
        panel = copy.deepcopy(write_report(directory, e, result)["panels"][0])
        results.append(result)
        panel["id"] = label
        panel["title"] = f"{label} · {e['provenance'].get('instance', 'run')} · {result['verdict']}"
        panel["caption"] += " " + time_caption
        for series in panel["series"]:
            for point in series["points"]:
                if event_aligned:
                    point["x"] -= shift
                start = min(start, point["x"])
                end = max(end, point["x"])
            paired = copy.deepcopy(series)
            paired["name"] = f"{label} · {series['name']}"
            paired["dash"] = [6, 4] if label == "A" else []
            paired["color"] = {
                "P cache hit ratio": ("#d4380d", "#1677ff"),
                "P Waiting / engine": ("#cf1322", "#2f54eb"),
                "P engine count": ("#fa8c16", "#722ed1"),
                "Client success QPS": ("#ad6800", "#13c2c2"),
            }.get(series["name"], (series["color"], series["color"]))[label == "B"]
            paired["hidden"] = series["name"] not in CORE_METRICS
            combined.append(paired)
        panels.append(panel)
    overlay = copy.deepcopy(panels[0])
    overlay.update(
        id="ab-overlay", title="A/B · 关键曲线同图", series=combined,
        caption=time_caption + "图例可选择和高亮。曲线仅来自归档监控。",
    )
    overlay["presets"] = {
        "核心": [s["name"] for s in combined if not s["hidden"]],
        "全部": [s["name"] for s in combined],
    }
    hashes = [i.get("jar_sha256") for i in identities]
    configs = [e["provenance"].get("actual_master_config") for e in evidence]
    identity = dict(
        master_artifact=("UNKNOWN" if not all(hashes) else
                         "SAME" if hashes[0] == hashes[1] else "DIFFERENT"),
        master_configuration=("UNKNOWN" if any(c is None for c in configs) else
                              "SAME" if configs[0] == configs[1] else "DIFFERENT"),
        runs={label: dict(instance=e["provenance"].get("instance"), master=artifact)
              for label, e, artifact in zip(("A", "B"), evidence, identities)},
    )
    summary = dict(
        purpose="EXPERIMENT_COMPARISON", verdicts=dict(zip(("A", "B"),
                                                          [r["verdict"] for r in results])),
        time_alignment=time_alignment, comparison_notice=notice,
        control_comparison=alignment, runs=dict(zip(("A", "B"), results)),
        identity=identity,
    )
    spec = dict(
        run_id="cache-scale-in-ab", title="缓存实验 · A/B",
        subtitle=f"A {results[0]['verdict']} / B {results[1]['verdict']} · controls {alignment['status']}",
        timeOriginLabel=time_caption,
        events=([dict(name=alignment_event, t=0)] if event_aligned else []),
        timeAxis=dict(min=start, max=end), kpis=[],
        panels=[overlay, *panels],
        sections=[
            details("运行身份与制品证据", identity),
            details("控制变量核对", alignment),
            details("绘图时间对齐", time_alignment),
        ],
    )
    write_bundle(
        output, "comparison", "cache-scale-in-ab", summary, spec,
        meta=run_meta(
            dict(id="cache-scale-in-ab"), evidence=[str(p) for p in paths],
            configuration=dict(alignment_event=alignment_event, controls=controls),
            runs={label: run_meta(
                dict(id=label, verdict=result["verdict"]),
                implementation=dict(master=identity, files=e["provenance"].get("files")),
                workload=e["provenance"].get("trace"),
                configuration={k: e["provenance"].get(k) for k in (
                    "topology", "capacity", "performance", "master_config",
                    "actual_master_config", "configuration_sha256",
                )},
                environment=e["provenance"].get("client_environment"),
                evidence=dict(path=str(path)),
            ) for label, path, e, result, identity in zip(
                ("A", "B"), paths, evidence, results, identities
            )},
        ),
        producer="cache-gate-ab",
    )
    return summary


def load_comparison_policy(path: Path) -> dict:
    import yaml

    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("comparison config must be a mapping")
    if "analysis" in config:
        from cases.config import validate_analysis

        policy = validate_analysis(config, str(path))
    else:
        policy = config
    return validate_policy(policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="optional experiment report policy YAML")
    args = parser.parse_args(argv)
    policy = load_comparison_policy(args.config) if args.config else {}
    result = compare(args.a, args.b, args.output, alignment_event=policy.get("alignment_event"))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))
    # Exit success means the report was produced, irrespective of run outcomes,
    # missing alignment events or control differences. Invalid input/I/O still fails.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

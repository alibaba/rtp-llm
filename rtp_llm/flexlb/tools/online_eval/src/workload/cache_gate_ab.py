"""Offline comparison of two completed cache-scale-in runs."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from reporting import write_bundle, run_meta, compare_controls, details
from traffic.playback import comparison_notice
from workload.cache_gate import analyze, write_report

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


def compare(old_path, new_path, output, *, mode="strong"):
    if mode not in {"strong", "weak", "none"}:
        raise ValueError(f"unknown comparison mode: {mode}")
    resolved = [_load(p) for p in (old_path, new_path)]
    paths, evidence = zip(*resolved)
    identities = [e["provenance"].get("master_artifact") for e in evidence]
    if any(not isinstance(i, dict) or not i.get("jar_sha256") for i in identities):
        raise ValueError("run evidence lacks observed master_artifact.jar_sha256")
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
    withdrawals = [
        next((v["t"] for v in e.get("events", []) if v["name"] == "withdraw_start"), None)
        for e in evidence
    ]
    event_aligned = all(t is not None for t in withdrawals)
    origin = max(withdrawals) if event_aligned else 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels, combined, results = [], [], []
    end = 1
    for label, path, e, shift, identity in zip(
        ("old", "new"), paths, evidence, withdrawals, identities
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
        version = identity.get("source_commit") or identity["jar_sha256"]
        panel["title"] = f"{label} · {version[:12]} · {result['verdict']}"
        panel["caption"] += (
            f" 原始缩容时刻 {shift:.2f}s；对齐到 {origin:.2f}s。"
            if event_aligned else " 缩容事件不完整，不能做事件对齐。"
        )
        for series in panel["series"]:
            for point in series["points"]:
                if event_aligned:
                    point["x"] += origin - shift
                end = max(end, point["x"])
            paired = copy.deepcopy(series)
            paired["name"] = f"{label} · {series['name']}"
            paired["dash"] = [6, 4] if label == "old" else []
            paired["color"] = {
                "P cache hit ratio": ("#d4380d", "#1677ff"),
                "P Waiting / engine": ("#cf1322", "#2f54eb"),
                "P engine count": ("#fa8c16", "#722ed1"),
                "Client success QPS": ("#ad6800", "#13c2c2"),
            }.get(series["name"], (series["color"], series["color"]))[label == "new"]
            paired["hidden"] = series["name"] not in CORE_METRICS
            combined.append(paired)
        panels.append(panel)
    overlay = copy.deepcopy(panels[0])
    overlay.update(
        id="ab-overlay", title="A/B · 关键曲线同图", series=combined,
        caption="按 withdraw_start 对齐；图例可选择和高亮。曲线仅来自归档监控。",
    )
    overlay["presets"] = {
        "核心": [s["name"] for s in combined if not s["hidden"]],
        "全部": [s["name"] for s in combined],
    }
    aligned = alignment["aligned"] and event_aligned
    different_versions = identities[0]["jar_sha256"] != identities[1]["jar_sha256"]
    observed = (aligned and different_versions and results[0]["verdict"] == "FAIL"
                and results[1]["verdict"] == "PASS")
    decision = (
        ("CONTROL_OBSERVED" if observed else "CONTROL_NOT_OBSERVED")
        if mode == "strong" else
        ("ALIGNED" if aligned else "UNALIGNED")
        if mode == "weak" else "REPORT_ONLY"
    )
    summary = dict(
        mode=mode, decision=decision, aligned=aligned,
        different_versions=different_versions,
        event_aligned=event_aligned, comparison_notice=notice,
        control_comparison=alignment, old=results[0], new=results[1],
        versions=dict(old=identities[0], new=identities[1]),
        expected_control_observed=observed,
    )
    spec = dict(
        run_id="cache-scale-in-ab", title="缩 P · Master A/B",
        subtitle=f"old {results[0]['verdict']} / new {results[1]['verdict']} · controls {alignment['status']} · {decision}",
        timeOriginLabel=(f"按缩容事件对齐，X={origin:.2f}s" if event_aligned else "缩容事件不完整，时间轴未对齐"),
        events=([dict(name="withdraw_start", t=origin)] if event_aligned else []),
        timeAxis=dict(min=0, max=end), kpis=[],
        panels=[overlay, *panels],
        sections=[
            details("Master 版本证据", summary["versions"]),
            details("控制变量核对", alignment),
            details("判定模式", dict(mode=mode, decision=decision)),
        ],
    )
    write_bundle(
        output, "comparison", "cache-scale-in-ab", summary, spec,
        meta=run_meta(
            dict(id="cache-scale-in-ab"), evidence=[str(p) for p in paths],
            configuration=dict(mode=mode, controls=controls),
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
                ("old", "new"), paths, evidence, results, identities
            )},
        ),
        producer="cache-gate-ab",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="optional downstream comparison policy YAML")
    parser.add_argument("--mode", choices=("strong", "weak", "none"))
    args = parser.parse_args()
    policy = {}
    if args.config:
        import yaml
        policy = yaml.safe_load(args.config.read_text())
        if policy.get("comparison") != "cache_scale_in" or policy.get("alignment_event") != "withdraw_start":
            raise ValueError("unsupported cache scale-in comparison policy")
        if policy.get("expected_verdicts") != {"old": "FAIL", "new": "PASS"}:
            raise ValueError("unsupported strong-control verdict pair")
    result = compare(args.old, args.new, args.output, mode=args.mode or policy.get("mode", "strong"))
    print(json.dumps({k: v for k, v in result.items() if k not in ("old", "new")}))
    raise SystemExit(
        0 if result["decision"] in {"CONTROL_OBSERVED", "ALIGNED", "REPORT_ONLY"} else 1
    )

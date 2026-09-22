"""Optional A/B observation. Absolute verdicts remain independent of the comparator."""

import argparse
import copy
import json
from pathlib import Path

from reporting import compare_controls, write_bundle, run_meta, details, table
from workload.performance_gate import analyze, report

REQUIRED = (
    "/criteria",
    "/mock_jar_sha256",
    "/performance",
    "/topology",
    "/capacity",
    "/workload_sha256",
    "/client_environment",
    "/actual_master_config",
    "/analyzer_sha256",
)


def controls(e):
    p = e.get("provenance", {})
    return dict(
        criteria=e.get("criteria"),
        mock_jar_sha256=p.get("mock_jar_sha256"),
        performance=p.get("performance"),
        topology=p.get("topology"),
        capacity=p.get("capacity"),
        workload_sha256=p.get("trace", {}).get("workload_sha256"),
        client_environment=p.get("client_environment"),
        actual_master_config=p.get("actual_master_config"),
        analyzer_sha256=p.get("analyzer_sha256"),
    )


def compare(left, right, output, allowed=()):
    # Explicit Master configuration fields only. Never exempt load, model or criteria.
    if any(
        not p.startswith("/actual_master_config/") or p.endswith("/") for p in allowed
    ):
        raise ValueError("only explicit Master configuration paths may vary")
    a, b = analyze(left), analyze(right)
    aligned = compare_controls(
        controls(left), controls(right), required=REQUIRED, allowed=allowed
    )
    declared = [
        dict(
            path=p,
            left_value=_at(controls(left), p),
            right_value=_at(controls(right), p),
        )
        for p in allowed
    ]
    result = dict(
        purpose="OBSERVATION_ONLY",
        controls=aligned,
        declared_changes=declared,
        left=a,
        right=b,
        verdicts=dict(left=a["verdict"], right=b["verdict"]),
    )
    output = Path(output)
    panels = []
    runs = {}
    for label, e, r in [("left", left, a), ("right", right, b)]:
        path = report(output / label, e, r)
        spec = json.loads((path / "report-spec.json").read_text())
        for index, source_panel in enumerate(spec["panels"]):
            panel = copy.deepcopy(source_panel)
            for series in panel["series"]:
                series["name"] = label + " · " + series["name"]
                series["dash"] = [6, 4] if label == "left" else []
            if label == "left":
                panel["caption"] = "left 虚线 / right 实线；时间按各自测量起点对齐"
                panels.append(panel)
            else:
                panels[index]["series"].extend(panel["series"])
        runs[label] = spec["run_meta"]
    rows = []
    for key in sorted(a["metrics"].keys() | b["metrics"].keys()):
        av, bv = a["metrics"].get(key), b["metrics"].get(key)
        rows.append(
            [key, av, bv, bv - av if av is not None and bv is not None else None]
        )
    write_bundle(
        output,
        "comparison",
        "master-performance",
        result,
        dict(
            title="Master 性能 A/B 观察",
            subtitle=f"left {a['verdict']} / right {b['verdict']} · controls {aligned['status']}",
            panels=panels,
            timeAxis=dict(
                min=0,
                max=max(left["criteria"]["measure_s"], right["criteria"]["measure_s"]),
            ),
            sections=[
                table(
                    "指标差异（不改变单 run 门禁）",
                    ["指标", "left", "right", "差值"],
                    rows,
                ),
                details("控制变量核对", aligned),
                details("声明配置变化", declared),
            ],
        ),
        meta=run_meta(dict(id="master-performance-ab"), runs=runs),
        producer="performance-compare",
    )
    return result


def _at(value, pointer):
    for part in pointer.strip("/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            raise ValueError("declared comparison path absent: " + pointer)
        value = value[key]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-master-change", action="append", default=[])
    a = p.parse_args()
    r = compare(
        json.loads(a.left.read_text()),
        json.loads(a.right.read_text()),
        a.output,
        a.allow_master_change,
    )
    print(
        json.dumps(
            dict(verdicts=r["verdicts"], controls=r["controls"]), allow_nan=False
        )
    )
    # This command validates comparability, not candidate eligibility.
    return (
        0
        if r["controls"]["aligned"]
        and all(v != "INVALID" for v in r["verdicts"].values())
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Optional A/B observation. Absolute verdicts remain independent of the comparator."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from reporting import compare_controls, write_bundle, run_meta, details, table
from reporting.pairing import paired_overlay
from reporting.view_config import view
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


def compare(left, right, output, allowed=(), left_directory=None, right_directory=None, json_only=False):
    comparison_view = view("master_performance.yaml")["comparison"]
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
    result["metric_deltas"] = {
        key: dict(left=a["metrics"].get(key), right=b["metrics"].get(key),
                  delta=b["metrics"][key] - a["metrics"][key])
        for key in a["metrics"].keys() & b["metrics"].keys()
        if a["metrics"][key] is not None and b["metrics"][key] is not None
    }
    if json_only:
        output.mkdir(parents=True, exist_ok=True)
        for label, evidence in (("left", left), ("right", right)):
            (output / (label + "-evidence.json")).write_text(json.dumps(evidence, allow_nan=False))
        (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    individual = []
    runs = {}
    run_sections = []
    for label, e, r, directory in [
        ("A", left, a, left_directory),
        ("B", right, b, right_directory),
    ]:
        destination = output / ("left" if label == "A" else "right")
        if directory:
            for source in Path(directory).glob("telemetry/*/queries.json"):
                target = destination / source.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
        path = report(destination, e, r)
        spec = json.loads((path / "report-spec.json").read_text())
        chart = copy.deepcopy(spec["panels"][0])
        config = e.get("provenance", {}).get("actual_master_config", {})
        mode = (
            config.get("scheduler", {}).get("decision", {}).get("type", "?")
            + " / "
            + config.get("dispatcher", {}).get("type", "?")
        )
        chart.update(id=label, title=label + " · " + mode + " · " + r["verdict"])
        individual.append(chart)
        runs[label] = spec["run_meta"]
        run_sections.append(details(label + " " + comparison_view["sections"]["run"], spec["sections"]))
    overlay = copy.deepcopy(individual[0])
    overlay.update(
        id="ab",
        title=comparison_view["overlay_title"],
        series=[],
        presets={},
        caption=comparison_view["overlay_caption"],
    )
    overlay["series"], overlay["presets"] = paired_overlay(individual)
    panels = [overlay, *individual]
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
            title=comparison_view["title"],
            subtitle=comparison_view["subtitle"].format(
                a_verdict=a["verdict"], b_verdict=b["verdict"],
                controls=aligned["status"],
                topology=left.get("provenance", {}).get("topology"),
                qps=left["criteria"]["qps"], benchmark_id=left["criteria"]["benchmark_id"]),
            kpis=[
                dict(
                    label=label + " " + comparison_view["kpi_label_suffix"],
                    value=(
                        f"{100 * (1 - r['metrics']['error_rate']):.6f}%"
                        if r["metrics"].get("error_rate") is not None
                        else "N/A"
                    ),
                )
                for label, r in [("A", a), ("B", b)]
            ],
            panels=panels,
            timeAxis=dict(
                min=0,
                max=max(left["criteria"]["measure_s"], right["criteria"]["measure_s"]),
            ),
            sections=[
                table(
                    comparison_view["sections"]["metrics"],
                    comparison_view["metrics_columns"],
                    rows,
                ),
                details(comparison_view["sections"]["controls"], aligned),
                details(comparison_view["sections"]["declared"], declared),
                *run_sections,
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
    p.add_argument("--json-only", action="store_true", help="compare controls and absolute gates without HTML")
    a = p.parse_args()
    r = compare(
        json.loads(a.left.read_text()),
        json.loads(a.right.read_text()),
        a.output,
        a.allow_master_change,
        a.left.parent,
        a.right.parent,
        json_only=a.json_only,
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

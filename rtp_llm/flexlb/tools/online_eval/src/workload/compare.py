"""Phase-aligned workload comparison. Changes rank attention, not regressions."""

import argparse
import hashlib
import statistics
from pathlib import Path

from reporting.statistics import select_window
from reporting import (
    write_bundle,
    load_analysis,
    run_meta,
    compare_controls,
)


def phase_windows(report):
    anchor = report["clock_anchor"]["monotonic_s"]
    return {
        s["id"]: (s["started_s"] - anchor, s["finished_s"] - anchor)
        for s in report["stages"]
        if s.get("finished_s") is not None and s["status"] != "BLOCKED"
    }


def compare(a, b):
    if a["id"] != b["id"]:
        raise ValueError("comparison requires the same workload instance")
    controls = [
        {
            "configuration_sha256": r.get("configuration_sha256"),
            "runtime": r["workload"].get("runtime_configuration"),
        }
        for r in (a, b)
    ]
    alignment = compare_controls(*controls, required=("/configuration_sha256",))
    if not alignment["aligned"]:
        raise ValueError(
            "comparison requires identical declared configuration and runtime configuration: "
            + str(alignment)
        )
    wa, wb = phase_windows(a), phase_windows(b)
    rows = []
    for phase in sorted(set(wa) | set(wb)):
        for metric in sorted(set(a["series"]) | set(b["series"])):
            samples = []
            for report, windows in ((a, wa), (b, wb)):
                window = windows.get(phase)
                samples.append(
                    []
                    if window is None
                    else [
                        [t - window[0], v]
                        for t, v in select_window(
                            report["series"].get(metric, []),
                            *window,
                            time=lambda p: p[0],
                        )
                    ]
                )
            left, right = samples
            row = dict(
                phase=phase,
                metric=metric,
                metric_kind=(
                    "derived_statistic"
                    if metric.startswith("statistics/")
                    else "raw_monitoring"
                ),
                statistic_sources=[
                    report.get("statistic_sources", {}).get(metric) for report in (a, b)
                ],
                baseline=left,
                candidate=right,
                verdict="DESCRIPTIVE_ONLY",
            )
            if not left or not right or any(v is None for _, v in left + right):
                row.update(
                    status="MISSING_DATA",
                    absolute_delta=None,
                    relative_delta=None,
                    rank_score=None,
                )
            elif any(
                report["workload"].get("runtime_validity") != "VALID"
                for report in (a, b)
            ):
                row.update(
                    status="INVALID_EVIDENCE",
                    absolute_delta=None,
                    relative_delta=None,
                    rank_score=None,
                )
            else:
                ma = statistics.fmean(p[1] for p in left)
                mb = statistics.fmean(p[1] for p in right)
                relative = (mb - ma) / abs(ma) if ma else None
                row.update(
                    status="COMPARED",
                    baseline_mean=ma,
                    candidate_mean=mb,
                    absolute_delta=mb - ma,
                    relative_delta=relative,
                    rank_score=abs(relative) if relative is not None else None,
                )
            rows.append(row)
    rows.sort(
        key=lambda r: (r["rank_score"] is not None, r["rank_score"] or 0), reverse=True
    )
    return dict(
        schema_version=1,
        id=a["id"],
        verdict="DESCRIPTIVE_ONLY",
        ranking="absolute relative change of raw or mature derived series means within matching stages; per-second percentiles are not pooled percentiles; zero baselines are unranked",
        baseline_validity=a["workload"]["runtime_validity"],
        candidate_validity=b["workload"]["runtime_validity"],
        changes=rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-panels",
        type=int,
        default=50,
        help="HTML panel limit; full comparison.json retains every metric",
    )
    args = parser.parse_args(argv)
    if args.max_panels < 1:
        parser.error("--max-panels must be positive")
    result = compare(load_analysis(args.baseline), load_analysis(args.candidate))
    args.out.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in result["changes"][: args.max_panels]:
        axis = sorted(
            set(t for field in ("baseline", "candidate") for t, v in row[field])
        )
        panels.append(
            dict(
                id=hashlib.sha256((row["phase"] + row["metric"]).encode()).hexdigest(),
                title=row["phase"] + " / " + row["metric"],
                caption=row["status"]
                + "; relative="
                + str(row["relative_delta"])
                + "; seconds since phase start",
                type="line",
                timeX=True,
                x=[str(t) for t in axis],
                xNums=axis,
                series=[
                    dict(
                        name=field,
                        points=[dict(x=t, y=v) for t, v in row[field]],
                        color=color,
                    )
                    for field, color in [
                        ("baseline", "#2563eb"),
                        ("candidate", "#dc2626"),
                    ]
                ],
            )
        )
    spec = dict(
        run_id=result["id"],
        title="Workload A/B: " + result["id"],
        timeOriginLabel="t=0 = 当前阶段开始",
        subtitle=result["ranking"]
        + f"; showing {len(panels)} of {len(result['changes'])} panels; all data in analysis.json; differences are not a product verdict",
        panels=panels,
        timeAxis=dict(
            min=0,
            max=max(
                (
                    t
                    for row in result["changes"]
                    for field in ("baseline", "candidate")
                    for t, _ in row[field]
                ),
                default=1,
            )
            or 1,
        ),
    )
    write_bundle(
        args.out,
        "comparison",
        result["id"],
        result,
        spec,
        meta=run_meta(
            dict(id=result["id"], kind="comparison"),
            evidence=[str(args.baseline), str(args.candidate)],
        ),
        producer="workload-compare",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

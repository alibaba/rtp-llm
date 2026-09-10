"""Phase-aligned workload comparison. Changes rank attention, not regressions."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from stress.canvas_report_render_html import render


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
    if not a.get("configuration_sha256") or a["configuration_sha256"] != b.get(
        "configuration_sha256"
    ):
        raise ValueError("comparison requires identical declared configuration")
    if a["workload"].get("runtime_configuration") != b["workload"].get(
        "runtime_configuration"
    ):
        raise ValueError("comparison requires identical runtime configuration")
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
                        for t, v in report["series"].get(metric, [])
                        if window[0] <= t < window[1]
                    ]
                )
            left, right = samples
            row = dict(
                phase=phase,
                metric=metric,
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
        ranking="absolute relative change of raw sample means within matching stages; zero baselines are unranked",
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
    result = compare(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
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
    (args.out / "comparison.html").write_text(
        render(
            dict(
                run_id=result["id"],
                title="Workload A/B: " + result["id"],
                timeOriginLabel="t=0 = 当前阶段开始",
                subtitle=result["ranking"]
                + f"; showing {len(panels)} of {len(result['changes'])} panels; all data in comparison.json; differences are not a product verdict",
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

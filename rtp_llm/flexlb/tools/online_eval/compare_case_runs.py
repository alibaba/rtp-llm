#!/usr/bin/env python3
"""Describe differences between two functional/scenario suite results.

Uses the same compiled instance IDs and keeps finding states distinct from
ordinary pass/fail. It does not rewrite case assertions or invent a verdict.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

from experiment_archive import create_archive

sys.path.insert(0, str(Path(__file__).with_name("stress")))
from canvas_report_render_html import render


class CaseComparisonError(ValueError):
    pass


def load_result(path: Path) -> tuple[dict, Path]:
    path = Path(path)
    if path.is_dir():
        found = [path / name for name in ("aggregate.json", "scenarios.json")
                 if (path / name).is_file()]
        if len(found) != 1:
            raise CaseComparisonError(f"{path}: need one aggregate.json or scenarios.json")
        path = found[0]
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or not isinstance(doc.get("instances"), list):
        raise CaseComparisonError(f"{path}: unsupported result schema")
    ids = [row.get("id") for row in doc["instances"] if isinstance(row, dict)]
    if len(ids) != len(doc["instances"]) or None in ids or len(ids) != len(set(ids)):
        raise CaseComparisonError(f"{path}: missing/duplicated instance identity")
    return doc, path


def _failed_checks(row: dict) -> list[str]:
    return sorted(f"{stage.get('id')}.{check.get('id')}"
                  for stage in row.get("stages", []) if isinstance(stage, dict)
                  for check in stage.get("checks", []) if isinstance(check, dict)
                  and check.get("status") == "FAIL")


def compare(a: dict, b: dict) -> dict:
    ia = {row["id"]: row for row in a["instances"]}
    ib = {row["id"]: row for row in b["instances"]}
    if set(ia) != set(ib):
        raise CaseComparisonError("instance sets differ; select matching suite/profile/grade")
    rows = []
    for identity in ia:
        left, right = ia[identity], ib[identity]
        for field in ("profile", "grade", "test_kind"):
            if left.get(field) != right.get(field):
                raise CaseComparisonError(f"{identity}: {field} differs")
        checks_a, checks_b = _failed_checks(left), _failed_checks(right)
        duration_a, duration_b = left.get("duration_ms"), right.get("duration_ms")
        changed = left.get("status") != right.get("status") or checks_a != checks_b
        rows.append({
            "id": identity, "test_kind": left.get("test_kind", "functional"),
            "profile": left.get("profile"),
            "status_a": left.get("status"), "status_b": right.get("status"),
            "failed_checks_a": checks_a, "failed_checks_b": checks_b,
            "duration_ms_a": duration_a, "duration_ms_b": duration_b,
            "duration_delta_ms": (duration_b - duration_a
                                  if type(duration_a) in (int, float)
                                  and type(duration_b) in (int, float) else None),
            "changed": changed,
        })
    return {"schema_version": 1, "classification": "descriptive_only",
            "summary": {"total": len(rows),
                        "changed": sum(row["changed"] for row in rows)},
            "instances": rows}


def chart_spec(report: dict) -> dict:
    rows = sorted((r for r in report["instances"]
                   if type(r["duration_ms_a"]) in (int, float)
                   and type(r["duration_ms_b"]) in (int, float)),
                  key=lambda r: abs(r["duration_delta_ms"]), reverse=True)[:30]
    return {
        "title": "Case / 场景 A/B 耗时差异",
        "subtitle": "按绝对差值显示前 30 项；结果状态与失败断言请看比较表",
        "panels": [{
            "id": "case_duration", "title": "实例耗时", "type": "bar",
            "caption": "相同实例 ID 成对比较；耗时不是正确性断言", "unit": "ms",
            "x": [row["id"] for row in rows],
            "series": [
                {"name": "A", "data": [row["duration_ms_a"] for row in rows],
                 "color": "#1677ff"},
                {"name": "B", "data": [row["duration_ms_b"] for row in rows],
                 "color": "#f5222d"},
            ],
        }] if rows else [],
    }


def render_table(report: dict) -> str:
    rows = []
    for row in report["instances"]:
        cells = (row["id"], row["status_a"], row["status_b"],
                 ", ".join(row["failed_checks_a"]), ", ".join(row["failed_checks_b"]),
                 row["duration_delta_ms"])
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
                    + "</tr>")
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
            "<title>Case A/B 结果对比</title><style>body{font:14px sans-serif;padding:24px}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;"
            "padding:6px;text-align:left}tr:nth-child(even){background:#f7f7f7}"
            "</style><h1>Case / 场景 A/B 结果对比</h1>"
            "<p>仅描述差异；FINDING-CONFIRMED 与 FAIL 保持不同语义。</p>"
            "<p><a href='case_ab_durations.html'>查看耗时图</a></p><table><tr>"
            "<th>ID</th><th>A</th><th>B</th><th>A 失败断言</th>"
            "<th>B 失败断言</th><th>耗时差 ms</th></tr>" + "".join(rows) +
            "</table></html>")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    try:
        a, path_a = load_result(args.run_a)
        b, path_b = load_result(args.run_b)
        report = compare(a, b)
    except (CaseComparisonError, OSError, ValueError) as exc:
        print(f"INCOMPARABLE: {exc}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "case_ab.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "case_ab_report.html").write_text(render_table(report), encoding="utf-8")
    (args.out_dir / "case_ab_durations.html").write_text(
        render(chart_spec(report)), encoding="utf-8")
    if args.archive:
        create_archive(args.archive,
                       {"run_a": path_a.parent, "run_b": path_b.parent,
                        "comparison": args.out_dir}, kind="ab",
                       metadata={"classification": "descriptive_only",
                                 "changed": report["summary"]["changed"]})
    print(f"case_ab={args.out_dir / 'case_ab.json'} changed={report['summary']['changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

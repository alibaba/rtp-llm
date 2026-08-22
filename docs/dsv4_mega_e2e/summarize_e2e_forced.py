#!/usr/bin/env python3
"""Build combined human- and machine-readable forced-decode reports."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


CANDIDATES = ("baseline2", "hca", "csa", "mega")


def _out_dir() -> Path:
    if len(sys.argv) == 2:
        return Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        raise SystemExit("usage: summarize_e2e_forced.py [logits-output-directory]")
    return Path(os.environ.get("E2E_OUT", "e2e_forced_logits_out")).resolve()


def _load_reports(out: Path) -> dict[str, dict[str, Any]]:
    reports = {}
    for candidate in CANDIDATES:
        path = out / f"compare.baseline_vs_{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text())
        if report["left"] != "baseline" or report["right"] != candidate:
            raise RuntimeError(f"unexpected labels in {path}")
        reports[candidate] = report
    record_ids = [record["record_id"] for record in reports[CANDIDATES[0]]["records"]]
    for candidate, report in reports.items():
        current = [record["record_id"] for record in report["records"]]
        if current != record_ids:
            raise RuntimeError(f"{candidate}: record order/set differs")
    return reports


def _write_full_csv(out: Path, reports: dict[str, dict[str, Any]]) -> Path:
    rows = []
    for candidate in CANDIDATES:
        for record in reports[candidate]["records"]:
            rows.append({"candidate": candidate, **record})
    fields = ["candidate"] + sorted({key for row in rows for key in row if key != "candidate"})
    path = out / "forced_logits_full_comparison.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _finite_max(records: list[dict[str, Any]], field: str) -> tuple[float, str]:
    values = [
        (float(record[field]), record["case_id"])
        for record in records
        if field in record and math.isfinite(float(record[field]))
    ]
    return max(values) if values else (math.nan, "-")


def _summary_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in CANDIDATES:
        records = reports[candidate]["records"]
        max_calc, calc_case = _finite_max(records, "calc_diff")
        max_js, js_case = _finite_max(records, "js_divergence")
        max_abs, abs_case = _finite_max(records, "max_abs")
        full = [record for record in records if record["full_vocab"]]
        rows.append(
            {
                "candidate": candidate,
                "records": len(records),
                "full_vocab_records": len(full),
                "top1_flips": sum(not record["top1_same"] for record in full),
                "max_calc_diff": max_calc,
                "max_calc_diff_case": calc_case,
                "max_js": max_js,
                "max_js_case": js_case,
                "max_abs": max_abs,
                "max_abs_case": abs_case,
            }
        )
    return rows


def _format_metric(record: dict[str, Any], field: str, digits: int = 3) -> str:
    value = record.get(field)
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}e}"


def _top_status(record: dict[str, Any]) -> str:
    if not record["full_vocab"]:
        return "probe"
    return "same" if record["top1_same"] else "FLIP"


def _write_markdown(
    out: Path, reports: dict[str, dict[str, Any]], summaries: list[dict[str, Any]]
) -> Path:
    indexed = {
        candidate: {record["record_id"]: record for record in reports[candidate]["records"]}
        for candidate in CANDIDATES
    }
    record_ids = [record["record_id"] for record in reports[CANDIDATES[0]]["records"]]
    path = out / "forced_logits_summary.md"
    with path.open("w") as handle:
        handle.write("# DSV4 forced-prefix logits comparison\n\n")
        handle.write(
            "All candidates consume an identical forced token prefix. `baseline2` is an "
            "independent source-path restart and therefore measures the run-to-run noise "
            "floor. Large-batch rows use fixed vocabulary probes; other rows compare the "
            "full vocabulary.\n\n"
        )
        handle.write("## Overall\n\n")
        handle.write(
            "| candidate | records | full vocab | top1 flips | max calc_diff | case | "
            "max JS | case | max_abs | case |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|---:|---|---:|---|\n")
        for row in summaries:
            handle.write(
                f"| {row['candidate']} | {row['records']} | {row['full_vocab_records']} | "
                f"{row['top1_flips']} | {row['max_calc_diff']:.6e} | "
                f"{row['max_calc_diff_case']} | {row['max_js']:.6e} | "
                f"{row['max_js_case']} | {row['max_abs']:.6f} | "
                f"{row['max_abs_case']} |\n"
            )

        handle.write("\n## Per Record\n\n")
        handle.write(
            "| record | B | noise calc/max | HCA calc/max/top1 | CSA calc/max/top1 | "
            "mega calc/max/top1 |\n"
        )
        handle.write("|---|---:|---|---|---|---|\n")
        for record_id in record_ids:
            noise = indexed["baseline2"][record_id]
            values = []
            for candidate in ("hca", "csa", "mega"):
                record = indexed[candidate][record_id]
                values.append(
                    f"{_format_metric(record, 'calc_diff')}/"
                    f"{float(record['max_abs']):.4f}/{_top_status(record)}"
                )
            handle.write(
                f"| {record_id} | {noise['batch']} | "
                f"{_format_metric(noise, 'calc_diff')}/{float(noise['max_abs']):.4f} | "
                f"{values[0]} | {values[1]} | {values[2]} |\n"
            )
    return path


def main() -> None:
    out = _out_dir()
    reports = _load_reports(out)
    csv_path = _write_full_csv(out, reports)
    summaries = _summary_rows(reports)
    summary_json = out / "forced_logits_summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2))
    markdown_path = _write_markdown(out, reports, summaries)
    print(f"full table: {csv_path}")
    print(f"summary: {markdown_path}")
    print(f"summary json: {summary_json}")


if __name__ == "__main__":
    main()

"""Fail CI unless collected AOne prefill artifacts prove a complete run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

from rtp_llm.test.perf_test.prefill.formula import normalize_observations
from rtp_llm.test.perf_test.prefill.report import _observation
from rtp_llm.test.perf_test.results import parse_result_envelope


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(root, name):
    try:
        value = (root / name).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{name}: missing or unreadable artifact") from error
    _require(value.strip(), f"{name}: empty artifact")
    return value


def _json(root, name):
    try:
        value = json.loads(_text(root, name))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name}: invalid JSON") from error
    _require(isinstance(value, dict), f"{name}: expected object")
    return value


def _svg(root, name):
    try:
        tree = ElementTree.fromstring(_text(root, name))
    except ElementTree.ParseError as error:
        raise ValueError(f"{name}: invalid SVG") from error
    _require(tree.tag.rsplit("}", 1)[-1] == "svg", f"{name}: expected SVG")


def verify_prefill_run(root: Path, *, require_interactive_charts: bool = True) -> dict:
    """Validate persisted measurements and deliverables; acceptance is advisory.

    The interactive-chart checks default to strict: the AOne CI gate runs on
    platform targets that ship Plotly and must reject degraded artifacts.
    Only test callers on hosts without Plotly may relax them explicitly.
    """
    root = Path(root)
    raw = _json(root, "results/result.json")
    parse_result_envelope(raw)
    _require(
        (raw["mode"], raw["backend"], raw["status"])
        == ("prefill", "aone", "completed"),
        "expected completed AOne prefill result",
    )
    _require(
        raw["model"]["topology"]["dp"] == 1,
        "concurrent grouped measurements require DP=1",
    )
    config = _json(root, "resolved_config.json")
    _require(
        config.get("workload", {}).get("transport") == "dashsc_input_ids",
        "expected persisted workload.transport=dashsc_input_ids",
    )
    cases = raw["cases"]
    _require(
        cases and len({c["case_id"] for c in cases}) == len(cases),
        "missing or duplicate cases",
    )
    _require(
        raw["summary"]
        == dict(planned_cases=len(cases), successful_cases=len(cases), failed_cases=0),
        "result summary is incomplete",
    )
    observations = []
    for case in cases:
        identity = case["identity"]
        _require(
            case["status"] == "ok" and not case["errors"], "all cases must succeed"
        )
        _require(
            identity.get("measurement_contract") == "concurrent_grouped_batch_v1"
            and identity.get("batch_size", 0) > 1,
            "expected concurrent grouped measurements",
        )
        row, rejection = _observation(
            SimpleNamespace(case_id=case["case_id"], identity=identity), case
        )
        _require(
            row is not None and rejection is None,
            f"{case['case_id']}: invalid measurement: {rejection}",
        )
        observations.append(row)
    _require(
        any(case["identity"].get("batch_size") == 4 for case in cases),
        "at least one successfully measured batch size 4 case is required",
    )
    _require(
        any(r["observed_reuse_length"] > 0 for r in observations),
        "no observed cache reuse",
    )
    probe = _json(root, "checkpoints/prefill_cache_probe.json")
    block = probe.get("block_size")
    responses = probe.get("responses", {})
    _require(
        probe.get("status") == "ok"
        and type(block) is int
        and block > 0
        and responses.get("seed", {}).get("reuse_len") == 0
        and responses.get("measure", {}).get("reuse_len") == block,
        "cache probe did not demonstrate one-block reuse",
    )
    report = _json(root, "reports/prefill_report.json")
    _require(
        report.get("mode") == "prefill"
        and report.get("report_kind") == "formal"
        and report.get("excluded") == [],
        "expected formal report without exclusions",
    )
    _require(
        report.get("counts") == dict(planned=len(cases), valid=len(cases), excluded=0),
        "report counts differ from result",
    )
    for key in ("config_fingerprint", "workload_fingerprint"):
        _require(report.get(key) == raw[key], "report fingerprint differs from result")
    _require(
        report.get("observations") == observations,
        "report observations differ from measured results",
    )
    _text(root, "reports/prefill_report.html")
    for chart in ("latency", "tpm"):
        name = f"reports/{chart}.interactive.html"
        html = _text(root, name)
        if require_interactive_charts:
            _require(
                "Plotly.newPlot(" in html
                and "updatemenus" in html
                and re.search(r"<script\b", html, re.I),
                f"{name}: missing interactive chart",
            )
            _require(
                not re.search(r"<(?:script|link)\b[^>]*(?:src|href)\s*=", html, re.I),
                f"{name}: chart must embed its dependencies",
            )
            scripts = re.findall(r"<script\b[^>]*>(.*?)</script\s*>", html, re.I | re.S)
            # The producer embeds the Plotly distribution, then calls it in a
            # separate script. Require its version header and exported plotting
            # APIs so an empty window.Plotly stub cannot stand in for that bundle.
            _require(
                any(
                    re.search(r"plotly\.js v\d+\.\d+\.\d+", script)
                    and re.search(r"window\.Plotly\s*=", script)
                    and all(
                        re.search(r"\." + api + r"\s*=\s*[A-Za-z_$]", script)
                        for api in ("newPlot", "react", "relayout")
                    )
                    for script in scripts
                ),
                f"{name}: missing embedded Plotly implementation",
            )
            _require(
                re.search(r"<div\b[^>]*\bid\s*=\s*[\'\"]chart[\'\"]", html, re.I),
                f"{name}: missing chart container",
            )
        _svg(root, f"reports/{chart}.svg")
    for name in ("prefill_3d.svg", "prefill_cold_miss.svg"):
        _svg(root, f"reports/{name}")
    rows, excluded = normalize_observations(observations)
    unique = len({row.geometry_key for row in rows})
    _require(
        not excluded and unique >= 30,
        "at least 30 valid measured geometries are required",
    )
    fit = _json(root, "reports/formula/fit_report.json")
    _require(
        fit.get("candidate_generated") is True
        and fit.get("complete") is True
        and fit.get("unique_geometry_count") == unique,
        "missing complete formula candidate with measured geometries",
    )
    model = _json(root, "reports/formula/model.json")
    formula = model.get("formula")
    _require(isinstance(formula, str) and formula.strip(), "missing FlexLB formula")
    _require(
        _text(root, "reports/formula/PREFILL_TIME_FORMULA.txt").strip()
        == "PREFILL_TIME_FORMULA=" + formula,
        "formula text differs from candidate",
    )
    _text(root, "reports/formula/predictions.csv")
    _svg(root, "reports/formula/fit_gap.svg")
    return dict(
        run_directory=str(root),
        successful_cases=len(cases),
        valid_geometries=unique,
        production_acceptance=fit.get("production_acceptance"),
    )


def find_fresh_run(results_root: Path, started_at: float) -> Path:
    """Require exactly one prefill directory created since the run began."""
    candidates = [
        p
        for p in Path(results_root).glob("prefill-*")
        if p.is_dir() and p.stat().st_mtime >= started_at
    ]
    _require(
        len(candidates) == 1, f"expected one fresh prefill run, found {len(candidates)}"
    )
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("perf_test_results"))
    parser.add_argument("--started-at", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        run = find_fresh_run(args.results_root, args.started_at)
        summary = verify_prefill_run(run)
        print(json.dumps(summary, sort_keys=True))
    except (ValueError, KeyError, TypeError) as error:
        print(f"prefill artifact gate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Regression coverage for confirmed PR review findings."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.consolidate import parse_env_file
from reporting.statistics import percentile_nr as report_percentile
from flexlb_cfg import ConfigOverride, _retype_ordering, render_env

ROOT = Path(__file__).resolve().parents[1]


def test_quoted_environment_preserves_json_and_empty_values(tmp_path):
    path = tmp_path / "env.txt"
    path.write_text("\"MODEL_SERVICE_CONFIG='{}'\" \\\n"
                    "FLEXLB_CONFIG='{\"name\":\"two words\"}'\n"
                    "\"EMPTY=''\"\nPLAIN=value\n")
    assert parse_env_file(path) == {
        "MODEL_SERVICE_CONFIG": "{}", "FLEXLB_CONFIG": '{"name":"two words"}',
        "EMPTY": "", "PLAIN": "value"}


def test_redundant_priority_override_preserves_existing_policy():
    ordering = {"type": "PRIORITY", "defaultPriority": 75,
                "preemption": {"allowedVictimStages": ["DECODE_ENGINE_OWNED"],
                               "timeoutMs": 900}}
    doc = {"scheduler": {"ordering": ordering.copy()}}
    _retype_ordering(doc, ConfigOverride(ordering="priority"))
    assert doc["scheduler"]["ordering"] == ordering
    _retype_ordering(doc, ConfigOverride(ordering="priority", default_priority=20,
                                       strip_preemption=True))
    assert doc["scheduler"]["ordering"] == {"type": "PRIORITY", "defaultPriority": 20}


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf"), True, "2"])
def test_invalid_decision_lifetime_rejected_before_rendering(value):
    with pytest.raises(ValueError, match="decision_lifetime"):
        render_env("stress-na130", ConfigOverride(decision_lifetime=value))


@pytest.mark.parametrize("values,p,expected", [([10, 20], .5, 10),
    (list(range(1, 101)), .95, 95), ([], .99, 0)])
def test_percentiles_agree_at_rank_boundaries(values, p, expected):
    tree = ast.parse((ROOT / "src/analysis/aggregate.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "percentile_nr")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "aggregate.py", "exec"), namespace)
    assert namespace["percentile_nr"](values, p) == expected
    assert report_percentile(values, p) == expected


def test_report_gini_uses_numeric_time_axis_and_missing_error_is_unknown(tmp_path):
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({"summary": {"total_requests": 10},
        "engine_dist": {"window_gini": {"t": [0, 5, 10],
                                        "prefill": [.1, .2, .3]}}}))
    report = tmp_path / "report.html"
    proc = subprocess.run([sys.executable, str(ROOT / "src/reporting/stress_report.py"),
        "--aggregate", str(aggregate), "--out", str(report)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    spec, _ = json.JSONDecoder().raw_decode(report.read_text().split("const SPEC = ", 1)[1])
    gini = [p for p in spec["panels"] if "Gini" in p["title"] and p.get("timeX")]
    assert gini and gini[0]["xNums"] == [0, 5, 10]
    error = next(k for k in spec["summary"]["kpis"] if k["label"] == "错误率")
    assert error["value"] == "—"

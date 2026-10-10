"""Behavioral boundaries: frozen admission, durable decisions and presentation IDs."""

import json
from unittest.mock import patch

import pytest

from analysis.checks import CheckResult
from analysis.gates import gate_checks
from cases.master_ha_failover.actions import _client_check
from reporting.run_context import canonical_spec
from runtime.deadline import Deadline
from runtime.outcome import RunOutcome, apply_outcome
from scenario.context import RuntimeContext
from scenario.execution_plan import freeze_plan, load_plan
from scenario.loader import ScenarioError
from scenario.stage_execution import _validate_output


def run(check="PASS", **changes):
    result = dict(status="PASS", execution_status="PASS", stages=[dict(
        id="gate", status="PASS" if check in {"PASS", "SKIP", "WARNING"} else check,
        error=None, checks=[dict(id="criterion", status=check)])], cleanup=[], finalization=[],
        workload=dict(runtime_validity="VALID"), error=None)
    result.update(changes)
    return result


def test_insufficient_ha_samples_survive_runtime_contract_without_a_fake_number(tmp_path):
    ctx = RuntimeContext({}, None, tmp_path, lambda: 0, lambda _: None)
    rows = ctx.register_resource("ha_rows", [])
    output = _client_check(ctx, dict(rows=rows, metric="ha_gate/success_rate", min_samples=100,
                                   expected=.95), Deadline(10, lambda: 0))
    _validate_output(ctx, output, {"actual": "nullable_number"}, {"criterion"})
    assert output.output["actual"] is None
    assert output.checks[0].detail == "insufficient request samples"
    assert output.checks[0].evidence["validity"] == "INVALID"


def test_ha_decision_needs_neither_a_metric_store_nor_a_monitor(tmp_path):
    ctx = RuntimeContext(dict(profile="batch-window"), None, tmp_path, lambda: 0, lambda _: None)
    rows = ctx.register_resource("ha_rows", [dict(status="ok", send_start_epoch_ms=1000)])
    with patch("monitoring.metric_store.publish", side_effect=AssertionError("publish during gate")):
        output = _client_check(ctx, dict(rows=rows, metric="ha_gate/success_rate", min_samples=1,
            expected=1, op="eq"), Deadline(10, lambda: 0))
    _validate_output(ctx, output, {"actual": "nullable_number"}, {"criterion"})
    assert output.checks[0].status == "PASS"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", ["PASS", "FAIL", "ERROR", "WARNING", "SKIP"])
def test_gate_and_run_share_effective_check_semantics(status):
    verdict, _ = gate_checks([CheckResult("criterion", status)], [])
    outcome = RunOutcome.from_result(run(status))
    assert outcome.gate == verdict


@pytest.mark.parametrize("gate", ["PASS", "FAIL"])
def test_delivery_failure_never_changes_gate(gate):
    result = run(gate, finalization=[dict(phase="report", status="ERROR")])
    outcome = apply_outcome(result)
    assert (outcome.execution, outcome.gate, outcome.validity, outcome.delivery) == (
        "PASS", gate, "VALID", "ERROR")
    assert result["status"] == "ERROR"


def test_plan_executes_compiled_values_after_scene_changes_and_rejects_code_drift(tmp_path):
    scene = tmp_path / "scene.yaml"
    scene.write_text("count: 1")
    compiled = [dict(id="example", stages=[dict(params=dict(count=1))])]
    path = tmp_path / "plan.json"
    with patch("scenario.execution_plan.dependency_files", return_value={"src/program.py": "a"}):
        digest = freeze_plan(path, compiled)
        scene.write_text("count: 999")
        assert load_plan(path, digest) == compiled
        with patch("scenario.execution_plan.dependency_files", return_value={"src/program.py": "b"}):
            with pytest.raises(ScenarioError, match="changed after planning"):
                load_plan(path, digest)
        payload = json.loads(path.read_text())
        payload["instances"][0]["stages"][0]["params"]["count"] = 999
        path.write_text(json.dumps(payload))
        with pytest.raises(ScenarioError, match="checksum"):
            load_plan(path, digest)


def test_report_structure_uses_ids_and_expands_checks_without_window_payloads():
    source = dict(id="case::default::profile", status="FAIL", workload=dict(runtime_validity="VALID"),
        checks=[dict(stage="gate", id="absolute", status="FAIL", actual="FAIL", expected="contract",
            evidence=dict(checks=[dict(id="ttft", actual=12, expected=10, status="FAIL")]))])
    spec = dict(panels=[], kpis=[dict(id="run.execution", label="改过的文案", value="old")], sections=[
        dict(id="run.checks", type="details", title="改过的标题", value="old"),
        dict(id="case.checks", type="details", title="任意标题", value="redundant"),
        dict(id="case.extra", type="details", title="门禁检查", value="keep")])
    first = canonical_spec(spec, source)
    assert first == canonical_spec(first, source)
    assert [section["id"] for section in first["sections"]] == ["run.checks", "run.validity", "case.extra"]
    assert first["sections"][0]["rows"] == [["gate/absolute/ttft", "FAIL", 12, 10]]
    assert sum(kpi.get("id") == "run.execution" for kpi in first["kpis"]) == 1
    assert "windows" not in json.dumps(first)


def test_child_uses_frozen_steps_without_loading_yaml(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from scenario import compile_scenarios
    from test_scenario_runtime import Backend, Clock, source
    from test_scenario_backend import lease_manifest
    module_path = Path(__file__).resolve().parents[1] / "scripts/commands/list_cases.py"
    spec = importlib.util.spec_from_file_location("frozen_child", module_path)
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)
    plans = compile_scenarios([("fixture", source())])
    plans[0]["test_kind"] = "functional"
    plan = tmp_path / "execution-plan.json"
    checksum = freeze_plan(plan, plans)
    lease = lease_manifest()
    for key, value in lease["child_env"].items():
        monkeypatch.setenv(key, value)
    lease_path = tmp_path / "lease.json"
    lease_path.write_text(json.dumps(lease))
    # The supplied YAML source does not exist. Neither loader nor compiler is allowed.
    with patch.object(child, "load_scenarios", side_effect=AssertionError("reload YAML")), \
         patch.object(child, "compile_scenarios", side_effect=AssertionError("recompile")), \
         patch("scenario.backend.JavaMockBackend", return_value=Backend(Clock())):
        assert child.main(["--source", str(tmp_path / "deleted-scenario"),
            "--plan", str(plan), "--plan-sha256", checksum,
            "--profile", "batch-window", "--grade", "normal",
            "--out-dir", str(tmp_path / "child"), "--lease-json", str(lease_path)]) == 0
    result = json.loads((tmp_path / "child/scenarios.json").read_text())["instances"][0]
    assert result["status"] == "PASS"
    assert result["outcome"] == dict(execution="PASS", gate="PASS", validity="VALID", delivery="PASS")

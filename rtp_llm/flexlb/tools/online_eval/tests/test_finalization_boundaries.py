"""Deadline, durable decision and extensible evidence failure boundaries."""

import json
import time
from pathlib import Path
from unittest import mock
import pytest
from scenario import compile_scenarios
from scenario.runtime import RuntimeContext, Deadline, execute_instance
from workload.runtime import execute_workload, WorkloadPolicy
from runtime.resource_evidence import ResourceEvidence
from test_scenario_runtime import Clock, Backend, source
from test_performance_gate import evidence as performance_evidence
from test_cache_scale_gate import CacheGateTest


def plan():
    instance = compile_scenarios([("fixture", source())])[0]
    instance.update(test_kind="workload", workload_runtime=dict(
        capture_metrics=False, sample_interval_s=1, collector_shutdown_s=2))
    instance["execution"]["monitoring"] = dict(instance["workload_runtime"])
    return instance


@pytest.mark.parametrize("kind", ["cache", "performance"])
def test_online_gate_never_calls_metric_or_html_delivery(tmp_path, kind):
    from workload.gate_result import load_gate
    from types import SimpleNamespace
    case = "cache_scale_in" if kind == "cache" else "master_performance"
    module = __import__("cases." + case + ".actions", fromlist=["actions"])
    e = CacheGateTest().evidence(.8) if kind == "cache" else performance_evidence()
    snapshot = e["flow"] if kind == "performance" else dict(
        issued=[], records=[], complete=True, errors=[], status=dict(submitted=0))
    flow = SimpleNamespace(directory=tmp_path, evidence_snapshot=lambda: snapshot)
    (tmp_path / "client_lifecycle.jsonl").write_text("")
    ctx = RuntimeContext({}, None, tmp_path, lambda: 0, lambda _: None)
    if kind == "cache":
        e["client_attribution"] = {}
    f = ctx.register_resource("java_flow", flow)
    h = ctx.register_resource("gate_evidence", e)
    ctx.monitor = SimpleNamespace(archive=lambda: None)
    # Cache attribution is unrelated to the presentation boundary exercised here.
    with mock.patch("cases." + case + ".metrics.produce", side_effect=AssertionError("metric delivery")), \
         mock.patch("cases." + case + ".report.write_report", side_effect=AssertionError("HTML delivery")):
        if kind == "cache":
            with mock.patch.object(module, "align_send_counters"), mock.patch.object(module, "attribute_client"):
                output = module.check(ctx, dict(flow=f, evidence=h), Deadline(100, lambda: 0))
        else:
            output = module.finish(ctx, dict(flow=f, evidence=h), Deadline(100, lambda: 0))
    _, frozen = load_gate(tmp_path, kind)
    assert output.checks[0].actual == frozen
    assert output.checks[0].status == {"PASS": "PASS", "FAIL": "FAIL", "INVALID": "ERROR"}[frozen["verdict"]]
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("completed", [True, False])
def test_report_failure_preserves_gate_and_runtime_validity(tmp_path, completed):
    clock = Clock()
    backend = Backend(clock)
    backend.completed = completed
    with mock.patch("workload.report.write_views", side_effect=RuntimeError("HTML failure")):
        result = execute_workload(plan(), backend, artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep)
    assert result["execution_status"] == ("PASS" if completed else "FAIL")
    assert result["workload"]["runtime_validity"] == "VALID"
    assert result["workload"]["report_status"] == "ERROR"
    assert result["stages"][-1]["checks"][0]["status"] == ("PASS" if completed else "FAIL")
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_virtual_deadline_includes_finalization_and_persists_base_result(tmp_path):
    clock = Clock()
    instance = plan()
    instance["execution"]["finalize_timeout_s"] = 2
    def slow(self, ctx, result, deadline):
        saved = json.loads((tmp_path / "result.json").read_text())
        assert saved["execution_status"] == "PASS"
        assert saved["status"] == "FINALIZING"
        assert saved["finalization"][0]["status"] == "RUNNING"
        deadline.sleep(10)
    with mock.patch.object(WorkloadPolicy, "finalize", slow):
        result = execute_workload(instance, Backend(clock), artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep)
    assert result["status"] == "TIMEOUT"
    assert result["duration_ms"] == 3000
    assert result["finalization"][1]["status"] == "BLOCKED"
    assert result["cleanup"] and all(row["status"] == "PASS" for row in result["cleanup"])
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_uncooperative_finalization_is_interrupted(tmp_path):
    instance = plan()
    instance["execution"]["finalize_timeout_s"] = .05
    # Real clock plus signal guard proves a blocking writer cannot hang the runner.
    class NoWaitBackend(Backend):
        def start_requests(self, ctx, params, deadline):
            return ctx.register_resource("requests", [])
        def teardown(self, ctx, deadline):
            pass
    started = time.monotonic()
    with mock.patch.object(WorkloadPolicy, "finalize", side_effect=lambda *args: time.sleep(5)):
        result = execute_workload(instance, NoWaitBackend(time.monotonic),
                                  artifact_dir=tmp_path, enforce_deadlines=True)
    assert time.monotonic() - started < 1
    assert result["finalization"][0]["status"] == "TIMEOUT"
    assert result["execution_status"] == "PASS"


def test_new_resource_evidence_needs_no_policy_branch(tmp_path):
    clock = Clock()
    instance = plan()
    ctx = RuntimeContext(instance, None, tmp_path, clock, clock.sleep)
    policy = WorkloadPolicy()
    policy.attach(ctx)
    ctx.env_epoch = 7
    ctx.register_resource("new_transport", object(), historical=True,
        evidence=lambda value, profile: ResourceEvidence(
            requests=dict(records=[dict(rid="new", status="ok")]), producer="new_worker",
            outages=(dict(source="7/new", started_epoch_s=10, ended_epoch_s=11, reason="test"),)))
    # Ignore duck-typed methods unless a resource explicitly declares an exporter.
    class NotEvidence:
        def evidence_snapshot(self):
            raise AssertionError("implicit resource evidence")
    ctx.register_resource("unexported", NotEvidence())
    result = dict(id="example", status="PASS", error=None, stages=[], cleanup=[])
    policy.finalize(ctx, result, Deadline(20, clock, clock.sleep))
    payload = json.loads((tmp_path / "workload-evidence.json").read_text())
    assert payload["request_resources"][0]["records"][0]["rid"] == "new"
    assert payload["expected_outages"][0]["source"] == "7/new"
    assert len(payload["request_resources"]) == 1


def test_report_budget_is_separate_and_included_in_elapsed_time(tmp_path):
    clock = Clock()
    instance = plan()
    instance["execution"]["report_timeout_s"] = .5
    def slow(self, ctx, result, analysis, deadline):
        deadline.sleep(10)
    with mock.patch.object(WorkloadPolicy, "render", slow):
        result = execute_workload(instance, Backend(clock), artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep)
    assert result["duration_ms"] == 1500
    assert result["execution_status"] == "PASS"
    assert result["workload"]["runtime_validity"] == "VALID"
    assert result["workload"]["report_status"] == "TIMEOUT"
    assert [row["status"] for row in result["finalization"]] == ["PASS", "TIMEOUT"]


@pytest.mark.parametrize("damage", ["modified", "missing"])
def test_committed_gate_corruption_never_falls_back_to_default(tmp_path, damage):
    from workload.gate_result import freeze_gate
    from cases.cache_scale_in.analysis import analyze
    from workload.report import write_views
    e = CacheGateTest().evidence(.8)
    result_path = freeze_gate(tmp_path, "cache", e, analyze(e))
    if damage == "missing":
        result_path.unlink()
    else:
        result_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_views(tmp_path, {"status": "ERROR"}, ["cache_scale_in.yaml"])


def test_atomic_result_writer_preserves_previous_checkpoint_on_serialization_failure(tmp_path):
    from artifacts.json_io import write_json
    target = tmp_path / "result.json"
    write_json(target, {"status": "FINALIZING", "execution_status": "PASS"})
    before = target.read_bytes()
    with pytest.raises(ValueError):
        write_json(target, {"value": float("nan")})
    assert target.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]

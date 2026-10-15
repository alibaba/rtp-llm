"""Reject invalid workloads and distinguish healthy survivors from planned victims."""

import copy
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cases.cache_sync_recovery.actions import atomic_write, collateral, converged, withdraw
from cases.cache_sync_recovery import actions
from document_io import load_document
from flexlb_cfg import ConfigOverride, render_env
from scenario.program_compiler import configure_program

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_document(ROOT / "config/scenarios/cache_sync_recovery.yaml")


def row(generation=1, age=20):
    return dict(generation=generation, status_age_ms=age, reported_alive=True, role="prefill",
                cache_indexed=True, cache_key_size=306000, cache_version=2, cache_indexed_version=2)


def test_registered_functional_case_compiles_with_owned_fixture_and_request_gates():
    plan = configure_program(config(), "cache_sync_recovery.yaml")
    actions = [stage["action"] for stage in plan["variants"][0]["stages"]]
    assert actions == ["setup", "cache_recovery_cycle", "request", "wait", "check", "check", "teardown"]


@pytest.mark.parametrize("section,key,value", [
    ("procedure", "delta_keys", 400000),
    ("procedure", "expired_workers", 96),
    ("procedure", "stable_s", 0),
    ("observation", "max_gap_s", 10),
    ("checks", "max_status_age_ms", 20000),
])
def test_invalid_parameters_fail_before_launch(section, key, value):
    data = config()
    data["parameters"][section][key] = value
    with pytest.raises(ValueError):
        configure_program(data, "case.yaml")


def test_stale_absent_or_replaced_healthy_worker_is_not_hidden_by_victim_exclusion():
    baseline = {"victim": row(), "healthy": row()}
    assert collateral({"healthy": row()}, baseline, {"victim"}, 10000) == []
    for current in ({}, {"healthy": row(age=10001)}, {"healthy": row(generation=2)}):
        assert collateral(current, baseline, {"victim"}, 10000) == ["healthy"]


def test_recovery_needs_exact_population_version_index_and_fresh_status():
    assert converged({"a": row()}, {"a"}, 306000, {"a": 2}, 10000)
    for field, value in (("cache_key_size", 305999), ("cache_indexed_version", 1),
                         ("cache_indexed", False), ("reported_alive", False), ("status_age_ms", -1)):
        current = row()
        current[field] = value
        assert not converged({"a": current}, {"a"}, 306000, {"a": 2}, 10000)
    assert not converged({}, {"a"}, 306000, {"a": 2}, 10000)


def test_withdrawal_preserves_other_roles_and_atomic_restore(tmp_path):
    original = json.dumps({"P": ["a", "b"], "D": ["c"]})
    hidden = withdraw(original, {"a"})
    assert json.loads(hidden) == {"P": ["b"], "D": ["c"]}
    with pytest.raises(ValueError):
        withdraw(original, {"missing"})
    path = tmp_path / "discovery.json"
    atomic_write(path, hidden)
    atomic_write(path, original)
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("violation", ["stale", "missing", "generation"])
def test_recorded_health_failure_stops_before_later_observation_errors(tmp_path, monkeypatch, violation):
    engines = [dict(name=f"prefill-{i}", role="prefill", grpc_addr=f"p{i}", http_addr=f"h{i}")
               for i in range(3)] + [dict(name="decode-0", role="decode", grpc_addr="d0", http_addr="hd0")]
    discovery = tmp_path / "discovery.json"
    original = json.dumps({"P": ["h0", "h1", "h2"], "D": ["hd0"]})
    discovery.write_text(original)
    params = config()["parameters"]
    params["procedure"].update(seed_interval_s=0, expired_workers=1)
    params["observation"]["converged_samples"] = 1
    calls = []
    stub = SimpleNamespace(GetWorkerStatus=lambda request, timeout: SimpleNamespace(alive=True))
    ops = SimpleNamespace(pb2=SimpleNamespace(StatusVersionPB=lambda: object()),
                          pb2_grpc=SimpleNamespace(RpcServiceStub=lambda channel: stub))
    ctx = SimpleNamespace(env=SimpleNamespace(discovery_file=discovery, spec=SimpleNamespace(n_prefill=3, n_decode=1)),
        ops=ops, artifact_dir=tmp_path, clock=lambda: 0, wall_clock=lambda: 10,
        add_cleanup=lambda name, callback: None, record_event=lambda name: {"id": name})

    def control(ops, endpoint, deadline, body=None):
        if endpoint.startswith("snapshot"):
            return {"engines": engines}
        return {"status": "ok", "engine": body["engine"], "fixture": {"version": 2}}

    def snapshot(ctx, deadline):
        calls.append(True)
        if len(calls) > 2:
            raise TimeoutError("later observation must not hide a recorded failure")
        rows = {entry["http_addr"]: row() for entry in engines}
        rows["hd0"]["role"] = "decode"
        if len(calls) == 2:
            if violation == "missing":
                rows.pop("h1")
            elif violation == "generation":
                rows["h1"]["generation"] = 2
            else:
                rows["h1"]["status_age_ms"] = 10001
        return rows

    monkeypatch.setattr(actions, "control_json", control)
    monkeypatch.setattr(actions, "master_rows", snapshot)
    monkeypatch.setattr(actions.grpc, "insecure_channel", lambda target: contextlib.nullcontext(target))
    result = actions.execute(ctx, params, SimpleNamespace(remaining=lambda: 100))
    from scenario.stage_execution import _validate_output
    from analysis.checks import check_verdict
    _validate_output(ctx, result, {}, actions.HANDLERS[0].checks)
    assert check_verdict(c.status for c in result.checks) == "FAIL"
    assert next(c for c in result.checks if c.id == "healthy_survivors").status == "FAIL"
    assert all(c.status == "SKIP" for c in result.checks if c.id != "healthy_survivors")
    summary = json.loads((tmp_path / "cache-recovery-summary.json").read_text())
    assert summary["affected_workers"] == ["h1"]
    assert summary["healthy_violation_probe"]["rpc_alive"] is True
    assert summary["aborted_after_health_violation"] is True
    assert summary["errors"] == []
    assert discovery.read_text() == original


def test_refresh_override_is_explicit_and_heap_parameters_are_frozen():
    default = json.loads(render_env("single-nonbatch"))
    assert "cacheStatus" not in default["workerRegistry"]
    rendered = json.loads(render_env("single-nonbatch", ConfigOverride(
        cache_min_refresh_ms=2400, cache_max_refresh_ms=2400)))
    assert rendered["workerRegistry"]["cacheStatus"] == {
        "minRefreshIntervalMs": 2400, "maxRefreshIntervalMs": 2400}
    with pytest.raises(ValueError):
        render_env("single-nonbatch", ConfigOverride(cache_min_refresh_ms=4000, cache_max_refresh_ms=2400))
    data = copy.deepcopy(config())
    data["environment"]["mock_heap"] = "8g; echo bad"
    with pytest.raises(ValueError):
        configure_program(data, "case.yaml")


def test_missing_health_evidence_cannot_pass_and_discovery_is_restored(tmp_path, monkeypatch):
    engines = [dict(name=f"prefill-{i}", role="prefill", grpc_addr=f"p{i}", http_addr=f"h{i}")
               for i in range(3)] + [dict(name="decode-0", role="decode", grpc_addr="d0", http_addr="hd0")]
    path = tmp_path / "discovery.json"
    original = json.dumps({"P": ["h0", "h1", "h2"], "D": ["hd0"]})
    path.write_text(original)
    cleanup = []
    ctx = SimpleNamespace(env=SimpleNamespace(discovery_file=path, spec=SimpleNamespace(n_prefill=3, n_decode=1)),
        ops=object(), artifact_dir=tmp_path, clock=lambda: 0, wall_clock=lambda: 10,
        add_cleanup=lambda name, callback: cleanup.append(callback), record_event=lambda name: {"id": name})

    def control(ops, endpoint, deadline, body=None):
        if endpoint.startswith("snapshot"):
            return {"engines": engines}
        return {"status": "ok", "engine": body["engine"], "fixture": {"version": 2}}

    monkeypatch.setattr(actions, "control_json", control)

    def missing_evidence(ctx, deadline):
        raise ValueError("Master lacks Worker poll-health evidence")

    monkeypatch.setattr(actions, "master_rows", missing_evidence)
    params = config()["parameters"]
    params["procedure"].update(seed_interval_s=0, expired_workers=1)
    with pytest.raises(ValueError, match="poll-health evidence"):
        actions.execute(ctx, params, object())
    assert path.read_text() == original
    assert "poll-health evidence" in json.loads((tmp_path / "cache-recovery-summary.json").read_text())["errors"][0]
    atomic_write(path, json.dumps({"P": [], "D": []}))
    cleanup[0](object())
    assert path.read_text() == original

"""Scale-in preserves ordering, partial evidence and drain outcomes on failure."""

import copy
from types import SimpleNamespace

import pytest

from cases.cache_scale_in.actions import observe
from file_io import read_json


@pytest.mark.parametrize("failure", [None, "baseline", "topology", "survivors", "removal"])
def test_scale_observation_keeps_evidence_and_joins_removals(tmp_path, monkeypatch, failure):
    now = [0]
    phase = [0]
    removals = []
    events = []
    criteria = dict(
        target_p=1,
        sample_s=1,
        baseline_s=4,
        warmup_min_s=4,
        warmup_timeout_s=8,
        baseline_min_hit=0.5,
        min_completed=1,
        qps=1,
        qps_tolerance=0.1,
        baseline_max_spread=0.1,
        max_gap_s=2,
        post_affinity_ms=0,
        removal_mode="immediate",
        drain_timeout_ms=1000,
        topology_timeout_s=3,
        observe_s=2,
    )

    def stamp(timestamp):
        return dict(elapsed_s=timestamp, epoch_s=100 + timestamp, monotonic_s=timestamp)

    def engines(timestamp):
        return {
            name: dict(
                hit_tokens_total=timestamp * 80,
                context_tokens_total=timestamp * 100,
                context_requests_total=timestamp,
                grpc_addr=name,
                engine_incarnation="1",
                admission_open=0 if phase[0] >= 2 and name == "p1" else 1,
                waiting=0,
                running=1,
            )
            for name in ("p0", "p1")
        }

    def record(name, *, timestamp=None):
        events.append(name)
        return dict(id=name, **(timestamp or dict(epoch_s=100 + now[0], monotonic_s=now[0])))

    def poll(deadline, interval, sample, **kwargs):
        phase[0] += 1
        label = ("baseline", "topology", "survivors")[phase[0] - 1]
        for timestamp in {1: [4], 2: [5], 3: [6, 7]}[phase[0]]:
            now[0] = timestamp
            if failure == label:
                raise ValueError(label + " interrupted")
            yield sample()

    def remove(ops, operation, deadline, params):
        removals.append(params["engine"])
        if failure == "removal":
            raise ValueError("drain interrupted")
        return dict(engine=params["engine"], drained=True)

    flow = SimpleNamespace(
        status=lambda: dict(observed_started=now[0], observed_terminal=now[0], process_returncode=None, state="SENDING"),
        issued_snapshot=lambda *args, **kwargs: [dict(send_start_epoch_ms=(100 + t) * 1000) for t in range(5)],
        stop_sending=lambda deadline: events.append("stop_command"),
    )
    ctx = SimpleNamespace(
        artifact_dir=tmp_path,
        instance={"id": "scale"},
        env_epoch=1,
        monitor=None,
        ops=None,
        resource=lambda *args: flow,
        register_resource=lambda kind, value, **kwargs: value,
        start_observation=lambda: SimpleNamespace(
            origin_epoch_s=100, origin_monotonic_s=0, stamp=stamp, to_dict=lambda: dict(origin_epoch_s=100, origin_monotonic_s=0)
        ),
        clock=lambda: now[0],
        record_event=record,
        services=SimpleNamespace(source_files=lambda: {}),
    )
    deadline = SimpleNamespace(check=lambda: None, remaining=lambda: 60)
    monkeypatch.setattr("cases.cache_scale_in.actions.gate_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr("cases.cache_scale_in.actions._master_count", lambda *args: 2 if phase[0] < 2 else 1)
    monkeypatch.setattr("cases.cache_scale_in.actions._affinity_transition", lambda *args: {})
    monkeypatch.setattr("cases.cache_scale_in.actions.poll_samples", poll)
    monkeypatch.setattr("cases.cache_scale_in.actions.mock_json", remove)
    monkeypatch.setattr("cases.cache_scale_in.observation.engine_metric_snapshot", lambda *args, **kwargs: engines(now[0]))
    monkeypatch.setattr(
        "cases.cache_scale_in.observation.measurement_samples",
        lambda *args, **kwargs: [dict(t=t, epoch_s=100 + t, started=0, terminal=0, engines=engines(t)) for t in range(5)],
    )
    params = dict(
        criteria=criteria,
        flow="flow",
        gate_input={"fields": {}},
        observation=dict(capture=dict(max_samples=100, max_bytes=100000), windows={}),
    )
    original = copy.deepcopy(params)
    observe(ctx, params, deadline)
    evidence = read_json(tmp_path / "cache-gate-evidence.json")
    assert params == original
    assert evidence["criteria"] == criteria
    assert removals == ([] if failure == "baseline" else ["p1"])
    assert evidence["errors"] == ([failure + " interrupted"] if failure in {"baseline", "topology", "survivors"} else [])
    if failure == "removal":
        assert evidence["removals"] == []
        assert evidence["removal_errors"] == [dict(index=0, error="drain interrupted")]
    elif failure != "baseline":
        assert evidence["removals"] == [dict(engine="p1", drained=True)]
    if failure in {None, "removal"}:
        assert events == [
            "warmup_start",
            "baseline_ready",
            "withdraw_start",
            "target_topology_observed",
            "observation_end",
            "stop_command",
            "sending_stopped",
        ]
        assert (evidence["baseline_start"], evidence["baseline_end"], evidence["post_start"], evidence["post_end"]) == (0, 4, 5, 7)
    else:
        assert "stop_command" not in events

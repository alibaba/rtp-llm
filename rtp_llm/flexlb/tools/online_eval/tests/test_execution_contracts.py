"""Input mutations, clock jumps and acquisition failures must remain visible."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from cases.config import configure_program
from input_contract import mapping_fields
from runtime.observation import ObservationClock, SampleBudget, evidence_origin, poll_samples
from scenario.loader import ScenarioError, load_document
from scenario.runtime import Deadline, RuntimeContext

ROOT = Path(__file__).resolve().parents[1]


def configuration(name):
    return load_document(ROOT / "config/scenarios" / (name + ".yaml"))


@pytest.mark.parametrize("name,window,side,offset,field,expected", [
    ("master_performance", "measurement", "from", 190, "warmup_s", 190),
    ("master_performance", "measurement", "until", 370, "measure_s", 190),
    ("cache_scale_in", "baseline", "from", -70, "baseline_s", 70),
    ("cache_scale_in", "post", "until", 70, "observe_s", 70),
])
def test_yaml_window_is_the_only_duration_authority(name, window, side, offset, field, expected):
    data = configuration(name)
    observation = data["parameters"]["observation"]
    observation["windows"][window][side]["offset_s"] = offset
    stages = configure_program(data, name + ".yaml")["variants"][0]["stages"]
    stage = next(row for row in stages if row["action"].endswith("observe"))
    assert stage["params"]["criteria"][field] == expected
    observation[field] = expected
    with pytest.raises(ScenarioError, match="unknown configuration fields"):
        configure_program(data, name + ".yaml")


def test_admission_dependency_is_required_before_any_live_poll():
    data = configuration("cache_scale_in")
    del data["parameters"]["observation"]["inputs"]["engine_counters"]["fields"]["admission_open"]
    with pytest.raises(ScenarioError, match="admission_open"):
        configure_program(data, "cache.yaml")


def test_cache_action_cannot_supply_a_hidden_removal_mode():
    from cases.cache_scale_in.actions import validate
    stages = configure_program(configuration("cache_scale_in"), "cache.yaml")["variants"][0]["stages"]
    params = next(row["params"] for row in stages if row["id"] == "scale_in")
    del params["criteria"]["removal_mode"]
    with pytest.raises(ValueError, match="removal_mode"):
        validate(params, NS(path="scale_in"))


@pytest.mark.parametrize("name", ["cache_scale_in", "master_performance"])
@pytest.mark.parametrize("field,value", [("unit", "wrong"), ("mode", "evaluated"), ("labels", [])])
def test_authoritative_metric_metadata_is_checked_against_consumer_dimensions(name, field, value):
    from monitoring.query_plan import load_plan
    plan = load_plan(name + ".yaml")
    metric = "running" if name == "cache_scale_in" else "rtp_llm_context_tps"
    plan["sources"]["mock"][metric][field] = value
    # Tamper at the plan binding boundary; loader validation alone is insufficient.
    with patch("monitoring.query_plan.load_plan", return_value=plan):
        with pytest.raises(ScenarioError, match="mismatch"):
            configure_program(configuration(name), name + ".yaml")


def test_extra_identity_labels_are_allowed_but_minimum_identity_is_mandatory():
    from monitoring.query_plan import load_plan
    plan = load_plan("cache_scale_in.yaml")
    plan["sources"]["mock"]["running"]["labels"].append("engine_ip")
    with patch("monitoring.query_plan.load_plan", return_value=plan):
        configure_program(configuration("cache_scale_in"), "cache.yaml")


@pytest.mark.parametrize("name", ["cache_scale_in", "master_performance", "master_ha_failover"])
def test_capture_budget_is_explicit_and_bounded(name):
    data = configuration(name)
    data["parameters"]["observation"]["capture"]["max_bytes"] = 0
    with pytest.raises(ScenarioError, match="capture.max_bytes"):
        configure_program(data, name + ".yaml")


def test_shared_field_guard_keeps_detail_with_adapter_exception_boundary():
    from cases.inputs import fields
    from scenario.parameters import validate_fields
    for read in (lambda data: mapping_fields(data, {"a"}, "input", required={"a"}),
                 lambda data: fields(data, {"a"}, "input"),
                 lambda data: validate_fields(data, NS(path="input"), {"a"}, {"a"})):
        with pytest.raises(ValueError, match="input.a"):
            read({})
        with pytest.raises(ValueError, match="unknown configuration fields.*typo"):
            read(dict(a=1, typo=2))


def test_monotonic_anchor_survives_wall_clock_jump_and_poll_uses_virtual_time(tmp_path):
    value = [0]
    ctx = RuntimeContext({}, None, tmp_path, lambda: value[0], lambda dt: value.__setitem__(0, value[0]+dt),
                         wall_clock=lambda: 1000 if value[0] == 0 else 500)
    clock = ObservationClock.start(ctx)
    deadline = Deadline(10, ctx.clock, ctx.sleeper)
    rows = list(poll_samples(deadline, 1, lambda: clock.stamp(ctx), until=3, immediate=True))
    assert [row["epoch_s"] for row in rows] == [1000, 1001, 1002, 1003]
    assert [row["elapsed_s"] for row in rows] == [0, 1, 2, 3]
    assert ctx.report_events[0]["epoch_s"] == clock.origin_epoch_s


def test_oversize_sample_is_not_appended_or_silently_truncated():
    budget = SampleBudget(dict(max_samples=1, max_bytes=100))
    budget.append(dict(state="SENDING"))
    with pytest.raises(ValueError, match="incomplete"):
        budget.append(dict(state="SENDING"))
    assert budget.count == 1
    with pytest.raises(ValueError, match="incomplete"):
        SampleBudget(dict(max_samples=10, max_bytes=3)).append(dict(state="SENDING"))


def test_performance_sampling_uses_framework_clock_and_freezes_failure(tmp_path):
    from cases.master_performance.actions import observe
    data = configuration("master_performance")
    params = next(stage for stage in configure_program(data, "performance.yaml")["variants"][0]["stages"]
                  if stage["id"] == "measure")["params"]
    params["observation"]["capture"]["max_samples"] = 1
    value = [0]
    ctx = RuntimeContext(dict(id="master_performance::default::batch-window", profile="batch-window"),
                         None, tmp_path, lambda: value[0], lambda dt: value.__setitem__(0, value[0]+dt),
                         wall_clock=lambda: 1000)
    flow = Mock()
    flow.control_status.return_value = dict(state="SENDING", process_returncode=None)
    params["flow"] = ctx.register_resource("java_flow", flow)
    with patch("cases.master_performance.actions.provenance", return_value={}):
        result = observe(ctx, params, Deadline(500, ctx.clock, ctx.sleeper))
    evidence = ctx.resource(result.output["evidence"], "gate_evidence")
    assert evidence["clock"]["origin_epoch_s"] == 1000
    assert len(evidence["samples"]) == 1
    assert "budget exceeded" in evidence["errors"][0]
    assert json.loads((tmp_path / "performance-gate-evidence.json").read_text()) == evidence
    with pytest.raises(ValueError, match="forged"):
        ctx.resource(result.output["evidence"], "snapshot")


def test_new_clock_is_authoritative_and_missing_historical_anchor_is_rejected():
    evidence = dict(clock=dict(origin_epoch_s=100), observation_origin_epoch_s=999,
                    samples=[dict(epoch_s=1000, t=5)])
    assert evidence_origin(evidence) == 100
    del evidence["clock"]
    assert evidence_origin(evidence) == 999
    del evidence["observation_origin_epoch_s"]
    assert evidence_origin(evidence) == 995
    evidence["samples"] = []
    with pytest.raises(ValueError, match="lacks a clock anchor"):
        evidence_origin(evidence)


def test_performance_stop_and_drain_are_explicit_program_stages():
    stages = configure_program(configuration("master_performance"), "performance.yaml")["variants"][0]["stages"]
    assert [row["action"] for row in stages] == ["setup", "java_flow_start", "performance_observe",
                                               "java_flow_stop", "java_flow_drain", "performance_finish", "teardown"]


def test_ha_sampler_budget_failure_is_reported_at_join(tmp_path):
    from cases.master_ha_failover.runtime import HaMasterStateSampler
    env = NS(master_specs=dict(A=NS(bind_ip="127.0.0.1", http_port=1)))
    sampler = HaMasterStateSampler(env, tmp_path/"states.jsonl", .001,
                                  limits=dict(max_samples=1, max_bytes=10000))
    with patch("cases.master_ha_failover.runtime.http_get_json", return_value=None):
        sampler.start()
        assert sampler._stop.wait(2)
        with pytest.raises(RuntimeError, match="budget exceeded"):
            sampler.stop()
    assert len(sampler.path.read_text().splitlines()) == 1


@pytest.mark.parametrize('name,check', [('cache_scale_in', 'baseline_hit'),
                                      ('master_performance', 'input_tps'),
                                      ('master_ha_failover', 'b_success')])
def test_semantic_program_errors_share_a_configuration_boundary(name, check):
    data = configuration(name)
    data['parameters']['checks'][check]['op'] = 'unknown'
    with pytest.raises(ScenarioError) as caught:
        configure_program(data, 'source.yaml')
    assert caught.value.__cause__ is not None
    assert type(caught.value.__cause__) is ValueError
    assert str(caught.value).startswith('source.yaml.parameters:')


def test_action_field_error_has_one_location_and_preserves_the_cause():
    from scenario.contracts import StageHandler
    from scenario.parameters import validate_fields
    from scenario.stage_compiler import stages

    handler = StageHandler(name='custom', outputs={}, checks=frozenset(),
        validate=lambda params, plan: validate_fields(params, plan, {'allowed'}),
        execute=lambda *args: None)
    with pytest.raises(ScenarioError) as caught:
        stages([{'id': 'setup', 'action': 'setup'},
                {'id': 'bad', 'action': 'custom', 'params': {'typo': True}}],
               'stages', 10, {'custom': handler})
    assert str(caught.value) == "stages[1].params: unknown configuration fields ['typo']"
    assert type(caught.value.__cause__) is ValueError

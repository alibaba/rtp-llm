"""Public program input groups, inactive criteria and window bindings are strict."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cases.config import configure_program
from scenario import compile_scenarios
from scenario.catalog import handlers
from scenario.loader import ScenarioError, load_document


ROOT = Path(__file__).resolve().parents[1]
CASES = ("request_completion", "master_performance", "cache_scale_in", "master_ha_failover")


def config(name):
    return load_document(ROOT / "config/scenarios" / (name + ".yaml"))


@pytest.mark.parametrize("name", CASES)
def test_base_parameters_and_each_read_group_reject_unknown_fields(name):
    data = config(name)
    assert set(data["parameters"]) <= {"traffic", "procedure", "observation", "analysis", "checks"}
    for group in [None, *data["parameters"]]:
        changed = config(name)
        target = changed["parameters"] if group is None else changed["parameters"][group]
        target["typo"] = 0
        with pytest.raises(ScenarioError, match="unknown configuration fields"):
            configure_program(changed, name + ".yaml")


def test_numeric_constraints_apply_to_nested_traffic_paths():
    data = config("request_completion")
    data["parameters"]["traffic"]["count"] = 10001
    with pytest.raises(ScenarioError, match="traffic.count"):
        configure_program(data, "case.yaml")
    data["parameter_schema"] = {"traffic.count": {"maximum": 10001}}
    with pytest.raises(ScenarioError, match="cannot weaken"):
        configure_program(data, "case.yaml")
    data["parameters"]["traffic"]["count"] = 10000
    del data["parameter_schema"]
    plan = configure_program(data, "case.yaml")
    assert plan["variants"][0]["stages"][1]["params"]["count"] == 10000


@pytest.mark.parametrize("name,stage", [
    ("master_performance", "measure"), ("cache_scale_in", "scale_in"),
])
def test_qps_has_one_yaml_source_and_is_frozen_into_observation(name, stage):
    data = config(name)
    qps = data["parameters"]["traffic"]["client"]["playback"]["qps"] * 1.01
    data["parameters"]["traffic"]["client"]["playback"]["qps"] = qps
    plan = configure_program(data, "case.yaml")
    params = next(s for s in plan["variants"][0]["stages"] if s["id"] == stage)["params"]
    assert params["criteria"]["qps"] == qps
    data["parameters"]["checks"]["qps"] = qps
    with pytest.raises(ScenarioError, match="unknown configuration fields"):
        configure_program(data, "case.yaml")


@pytest.mark.parametrize("patch", [dict(metric="ha_gate/typo"), dict(typo=1), dict(min_samples=0),
                                 dict(metric={}), dict(op=[]), dict(warning_profiles=[{}])])
def test_inactive_ha_checks_are_validated_without_running_their_program(patch):
    data = config("master_ha_failover")
    del data["variants"]
    del data["variant_axis"]
    data["parameters"]["checks"]["outage_failures"].update(patch)
    with pytest.raises(ValueError):
        configure_program(data, "ha.yaml")


def test_ha_windows_and_check_binding_are_yaml_data_and_compile_to_typed_outputs():
    data = config("master_ha_failover")
    data["parameters"]["observation"]["windows"]["baseline"]["until"]["offset_s"] = -3
    data["parameters"]["checks"]["b_success"]["windows"] = ["both"]
    document = configure_program(data, "ha.yaml")
    stages = {s["id"]: s for s in document["variants"][0]["stages"]}
    assert stages["baseline"]["params"]["until_offset_s"] == -3
    assert stages["b_success"]["params"]["rows"] == {"$ref": "stages.both.output.rows"}
    with patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
        assert len(compile_scenarios([("ha.yaml", document)], handlers=handlers())) == 4


@pytest.mark.parametrize("window", ["typo", "outage"])
def test_unknown_or_inactive_window_cannot_be_silently_substituted(window):
    data = config("master_ha_failover")
    data["parameters"]["checks"]["b_success"]["windows"] = [window]
    with pytest.raises(ScenarioError, match="window"):
        configure_program(data, "ha.yaml")


def test_bad_inactive_window_boundary_is_rejected_at_compile_time():
    data = config("master_ha_failover")
    data["parameters"]["observation"]["windows"]["outage"]["from"]["stage"] = "typo"
    with pytest.raises(ScenarioError, match="unknown timestamp output"):
        configure_program(data, "ha.yaml")


@pytest.mark.parametrize("wait", [{"wait_s": True}, {"wait_s": 181}, {"wait_s": 0, "typo": 1}])
def test_inactive_procedure_waits_still_have_a_strict_contract(wait):
    data = config("master_ha_failover")
    del data["variants"]
    del data["variant_axis"]
    data["parameters"]["procedure"]["outage_wait"] = wait
    with pytest.raises(ValueError):
        configure_program(data, "ha.yaml")


def test_ha_flow_waits_and_evidence_offsets_have_separate_effects():
    original = config("master_ha_failover")
    changed = config("master_ha_failover")
    changed["parameters"]["procedure"]["settle"]["wait_s"] = 7
    baseline = configure_program(original, "ha.yaml")
    compiled = configure_program(changed, "ha.yaml")
    for before, after in zip(baseline["variants"], compiled["variants"]):
        expected = [s.copy() for s in before["stages"]]
        for stage in expected:
            if stage["action"] == "master_mark" and stage["id"] in {
                "b_start", "a_start", "both_start", "outage_start",
            }:
                stage["params"] = {**stage["params"], "wait_s": 7}
        assert expected == after["stages"]
    changed = config("master_ha_failover")
    changed["parameters"]["observation"]["windows"]["baseline"]["until"]["offset_s"] = -3
    compiled = configure_program(changed, "ha.yaml")
    for before, after in zip(baseline["variants"], compiled["variants"]):
        expected = [s.copy() for s in before["stages"]]
        for stage in expected:
            if stage["id"] == "baseline":
                stage["params"] = {**stage["params"], "until_offset_s": -3}
        assert expected == after["stages"]


def test_ha_waits_cannot_be_redeclared_as_observation_inputs():
    data = config("master_ha_failover")
    data["parameters"]["observation"]["settle"] = {"wait_s": 5}
    with pytest.raises(ScenarioError, match="unknown configuration fields"):
        configure_program(data, "ha.yaml")

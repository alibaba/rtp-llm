"""Domain invariants at admission, delivery and persisted traffic boundaries."""

from dataclasses import FrozenInstanceError
import json
from unittest.mock import patch

import pytest

from scenario.compiler import ScenarioCompiler
from scenario.execution_plan import ExecutionPlanCodec, _sha
from scenario.errors import ScenarioError
from scenario.plan import ExecutionPlan, CompiledInstance
from scenario.runtime import ScenarioRunner
from scenario.stage_parameters import RequestParameters
from scenario.environment_config import environment
from scenario.environment_snapshot import EnvironmentSnapshot
from scenario.backend_spec import make_env_spec
from traffic import prefix_lineage, prefix_lineage_v3
from traffic.lineage_model import LineageExpander, LengthSemantics
from traffic.playback.intensity import RateCurve, LinearRetentionSchedule, SequenceRetentionSchedule
from test_scenario_runtime import source, Backend, Clock


def compiled():
    return ScenarioCompiler().compile([("fixture", source())])[0]


def test_compiler_and_runner_exchange_domain_objects(tmp_path):
    instance = compiled()
    assert isinstance(instance, CompiledInstance)
    assert isinstance(instance.stages[1].parameters, RequestParameters)
    with pytest.raises(FrozenInstanceError):
        instance.execution.timeout_s = 0
    projection = instance.to_dict()
    projection["stages"][1]["params"]["count"] = 999
    assert instance.stages[1].parameters.count == 1
    clock = Clock()
    result = ScenarioRunner().run(instance, Backend(clock), artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep)
    assert result.execution.status == "PASS"
    assert result.adjudicate().gate == "PASS"
    assert json.loads((tmp_path / "result.json").read_text()) == result.to_dict()


@pytest.mark.parametrize("damage", ["budget", "request", "snapshot", "identity", "reference"])
def test_valid_checksum_cannot_admit_semantically_invalid_plan(tmp_path, damage):
    codec = ExecutionPlanCodec(dependencies=lambda: {})
    path = tmp_path / "plan.json"
    digest = codec.write(path, ExecutionPlan((compiled(),), ()))
    assert codec.read(path, digest).instances[0] == compiled()
    document = json.loads(path.read_text())
    instance = document["instances"][0]
    if damage == "budget":
        instance["execution"]["timeout_s"] = 0
    elif damage == "request":
        instance["stages"][1]["params"]["count"] = True
    elif damage == "snapshot":
        instance["environment"]["resolved_config"]["dispatcher"]["type"] = "NON_BATCH"
    elif damage == "reference":
        instance["stages"][2]["params"]["requests"] = {"$ref": "stages.submit.output.invented"}
    else:
        instance["variant_id"] = "different"
    # Recompute the envelope digest to test semantic admission independently of hashing.
    document["sha256"] = _sha({key: value for key, value in document.items() if key != "sha256"})
    path.write_text(json.dumps(document))
    with pytest.raises((ScenarioError, ValueError)):
        codec.read(path, document["sha256"])


def test_final_effective_mock_settings_are_frozen_before_backend_projection():
    plan = environment(dict(prefill_perf=dict(fixed_ms=10, scale=1, max_batch_tokens=1024,
        max_batch_requests=2), prefill_max_waiting_batches=0,
        prefill_cache_policy=dict(device_tree=False, memory_tree=True, memory_blocks=42)),
        "env", "batch-window")
    snapshot = EnvironmentSnapshot.read(plan["rendered"])
    assert snapshot.performance["prefill"]["memory_cache"] == dict(
        enabled=True, capacity_blocks=42, enable_prefix_tree=True)
    assert snapshot.performance["prefill"]["max_waiting_batches"] == 0
    # Live declaration patches cannot silently reinterpret the frozen effective configuration.
    plan["prefill_perf"]["fixed_ms"] = 999
    plan["prefill_cache_policy"]["memory_blocks"] = 99
    with patch("adapters.perf_presets.load_preset_bundle", side_effect=AssertionError("late preset read")):
        spec = make_env_spec(plan, "batch-window", {"master_base": 28000})
    assert spec.perf == snapshot.performance


def test_each_environment_declaration_loads_preset_once_and_checks_excluded_profile():
    import adapters.perf_presets as presets
    with patch("scenario.environment_resolver.load_preset_bundle", wraps=presets.load_preset_bundle) as load:
        environment(dict(profile_overrides={"single-nonbatch": {"max_requests": 2},
                                            "single-batch": {"max_requests": 3}}), "env", "batch-window")
    assert load.call_count == 1
    with pytest.raises(ScenarioError):
        environment(dict(profile_overrides={"single-nonbatch": {"max_requests": "bad"}}), "env", "batch-window")


def test_lineage_length_units_and_reserved_partial_labels_stay_distinct():
    events = [[0, 1025, -1, 0], [10, 513, 0, 1], [20, 512, -1, 0]]
    aligned = prefix_lineage.BlockAlignedLineageCodec().decode(prefix_lineage.encode(events))
    exact = prefix_lineage_v3.ExactLengthLineageCodec().decode(prefix_lineage_v3.encode(events))
    assert aligned.metadata.tail == LengthSemantics.BLOCK_ALIGNED
    assert exact.metadata.tail == LengthSemantics.EXACT
    left, right = list(LineageExpander(aligned)), list(LineageExpander(exact))
    assert [row.input_tokens for row in left] == [1024, 512, 512]
    assert [row.input_tokens for row in right] == [1025, 513, 512]
    assert [row.block_labels for row in left] == [(1, 2), (1,), (5,)]
    assert [row.block_labels for row in right] == [(1, 2, 3), (1, 4), (5,)]
    assert [row.source_index for row in left] == [0, 1, 2]
    with pytest.raises(ValueError, match="budget"):
        list(LineageExpander(exact, label_limit=3))


def test_playback_objects_enforce_curve_and_lap_contracts():
    curve = RateCurve.read([[0, .5], [10, 3], [20, 1]])
    assert [curve.integral(t) for t in (10, 20, 30)] == [17.5, 37.5, 47.5]
    linear = LinearRetentionSchedule(0, 1, 3)
    sequence = SequenceRetentionSchedule((1, .5, 0))
    assert [linear.probability(lap) for lap in (1, 2, 3, 4)] == [0, .5, 1, 1]
    assert [sequence.probability(lap) for lap in (1, 2, 3, 4)] == [1, .5, 0, 0]
    with pytest.raises(ValueError):
        RateCurve.read([[0, 1], [0, 2]])
    with pytest.raises(ValueError):
        sequence.probability(0)


def test_freezing_preserves_collection_definition_order(tmp_path):
    from pathlib import Path
    from scenario.program_compiler import configure_program
    from scenario.catalog import handlers
    from scenario.document_io import load_document
    from monitoring.collection_plan import instance_plan
    path = Path(__file__).resolve().parents[1] / "config/scenarios/master_ha_failover.yaml"
    with patch("scenario.compiler.VICTIM_OFFSETS", (2048, 2049, 2050)), \
         patch("adapters.java_mock_resources.VICTIM_OFFSETS", (2048, 2049, 2050)):
        plan = ScenarioCompiler(handlers()).compile([(str(path), configure_program(load_document(path), str(path)))])
        codec = ExecutionPlanCodec(dependencies=lambda: {})
        frozen = tmp_path / "plan.json"
        digest = codec.write(frozen, ExecutionPlan(tuple(plan), ()))
        restored = codec.read(frozen, digest)
        assert restored == ExecutionPlan(tuple(plan), ())
        for instance in restored.instances:
            projected = instance.to_dict()
            assert instance_plan(projected) == projected["implementation"]["monitoring_query_plan"]["definition"]


@pytest.mark.parametrize("fails", [False, True])
def test_report_adapter_cannot_rewrite_frozen_gate_verdict(tmp_path, fails):
    document = source()
    if fails:
        document["stages"][-1]["params"]["expected"] = False
    instance = ScenarioCompiler().compile([("fixture", document)])[0]
    class Policy:
        def attach(self, ctx): pass
        def before_stage(self, ctx, stage): pass
        def after_stage(self, ctx, stage, row): pass
        def continue_after_failure(self, stage): return False
        def finalize(self, ctx, result, deadline):
            for stage in result["stages"]:
                for check in stage["checks"]:
                    check["status"] = "PASS" if fails else "FAIL"
            result["workload"] = {"runtime_validity": "VALID"}
            return {}
        def render(self, ctx, result, analysis, deadline):
            raise RuntimeError("report unavailable")
    clock = Clock()
    result = ScenarioRunner().run(instance, Backend(clock), artifact_dir=tmp_path,
                                  clock=clock, sleeper=clock.sleep, policy=Policy())
    outcome = result.adjudicate()
    assert outcome.gate == ("FAIL" if fails else "PASS")
    assert outcome.delivery == "ERROR"
    assert outcome.validity == "VALID"
    assert result.to_dict()["status"] == "ERROR"


def test_explicit_null_seed_is_not_treated_as_an_absent_seed():
    from traffic.playback.intensity import PlaybackIntensity
    with pytest.raises(ValueError, match="seed"):
        PlaybackIntensity.read({"arrival": "deterministic", "seed": None}, mode="uniform",
                               identity="structural-relabel", scalar_present=False, legacy_pacing=False)

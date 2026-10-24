"""真实构建与编译路径不经过内部文档或领域对象的字典读取器。"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from document_io import DocumentError
from scenario.builder import CaseBuilder
from scenario.catalog import handlers
from scenario.compiler import ScenarioCompiler
from scenario.contracts import StageHandler
from scenario.execution_plan import freeze_plan, load_plan
from scenario.loader import load_scenarios
from scenario.plan import CompiledInstance, EnvironmentPlan, StagePlan
from scenario.program import ProgramPlan, StageDefinition, VariantPlan

ROOT = Path(__file__).resolve().parents[1]


def test_custom_action_parameters_are_detached_without_json_decoding():
    from scenario.stage_parameters import decode_parameters

    raw = {"criteria": {"samples": [1, 2]}}
    parameters = decode_parameters("custom_action", raw)
    raw["criteria"]["samples"].append(3)
    with patch("scenario.stage_parameters.json.loads", side_effect=AssertionError("internal JSON decode")):
        snapshot = parameters.to_dict()
        assert snapshot == {"criteria": {"samples": [1, 2]}}
        snapshot["criteria"]["samples"].append(99)
        assert parameters.to_dict() == {"criteria": {"samples": [1, 2]}}


def test_case_config_is_parsed_once_before_building(monkeypatch):
    from document_io import load_document
    from scenario.inputs import InputSchema
    from scenario.program_compiler import configure_program

    calls = []
    parse = InputSchema.parse

    def traced(schema, parameters, rules=None):
        calls.append(parameters)
        return parse(schema, parameters, rules)

    monkeypatch.setattr(InputSchema, "parse", traced)
    configure_program(load_document(ROOT / "config/scenarios/request_completion.yaml"), "fixture")
    assert len(calls) == 1


def test_reference_only_actions_validate_the_same_resource_kinds():
    from scenario.contracts import PlanContext
    from scenario.parameters import reference_parameters

    context = PlanContext("finish", {"flow": {"flow": "java_flow"}, "read": {"evidence": "gate_evidence"}})
    params = dict(flow={"$ref": "stages.flow.output.flow"}, evidence={"$ref": "stages.read.output.evidence"})
    assert reference_parameters(params, context, resources={"flow": "java_flow", "evidence": "gate_evidence"}) == params
    with pytest.raises(ValueError):
        reference_parameters(params, context, resources={"flow": "ha_client", "evidence": "gate_evidence"})


def test_builder_returns_typed_isolated_steps_and_keeps_observation_purpose():
    builder = CaseBuilder({}, {}, number_rules={})
    params = {"nested": [1]}
    builder.step("setup", "setup")
    builder.observe("read", "observe", params=params, timeout_s=4)
    params["nested"].append(2)
    first = builder.finish()
    assert isinstance(first, tuple)
    assert all(isinstance(step, StageDefinition) for step in first)
    assert first[1].purpose == "observation"
    assert first[1].params == {"nested": [1]}
    first[1].params["nested"].append(3)
    assert builder.finish()[1].params == {"nested": [1]}


def test_real_case_builds_typed_program_and_compiles_without_dictionary_decoders(tmp_path):
    documents = load_scenarios(ROOT / "config/scenarios/request_completion.yaml")
    program = documents[0][1]
    assert isinstance(program, ProgramPlan)
    assert all(isinstance(variant, VariantPlan) for variant in program.variants)
    with (
        patch.object(CompiledInstance, "read", side_effect=AssertionError("instance roundtrip")),
        patch.object(StagePlan, "read", side_effect=AssertionError("stage roundtrip")),
        patch.object(EnvironmentPlan, "read", side_effect=AssertionError("environment roundtrip")),
    ):
        instances = ScenarioCompiler(handlers()).compile(documents)
    assert all(isinstance(stage, StagePlan) for instance in instances for stage in instance.stages)
    path = tmp_path / "execution-plan.json"
    projected = [instance.to_dict() for instance in instances]
    digest = freeze_plan(path, projected)
    assert load_plan(path, digest) == projected
    # The same program can be compiled again without changes or retained compiler state.
    assert [instance.to_dict() for instance in ScenarioCompiler(handlers()).compile(documents)] == projected


@pytest.mark.parametrize("reference", ["stages.future.output.count", "stages.setup.output.missing", "stages.adapter.output.count"])
def test_adapter_payload_references_are_checked_before_publishing_current_outputs(reference):
    program = load_scenarios(ROOT / "config/scenarios/request_completion.yaml")[0][1]
    variant = program.variants[0]
    adapter = StageDefinition("adapter", "custom", params={"nested": [{"$ref": reference}]})
    variant = replace(variant, stages=(variant.stages[0], adapter, *variant.stages[1:]))
    custom = StageHandler("custom", lambda params, context: params, lambda *_: None, {"count": "integer"})
    with pytest.raises(DocumentError, match="unknown or forward reference"):
        ScenarioCompiler({"custom": custom}).compile([("fixture", replace(program, variants=(variant,)))])


def test_raw_internal_documents_are_not_a_compiler_input():
    with pytest.raises(DocumentError, match="expected ProgramPlan"):
        ScenarioCompiler().compile([("fixture", {"id": "example", "stages": []})])


def test_selection_does_not_share_mutable_configuration_or_steps_with_original():
    from scenario.suites import preselect_documents

    documents = load_scenarios(ROOT / "config/scenarios/request_completion.yaml")
    selected = preselect_documents(documents, kind="functional")
    selected[0][1].environment["n_prefill"] = 123
    selected[0][1].variants[0].stages[1].params["count"] = 999
    assert documents[0][1].environment["n_prefill"] != 123
    assert documents[0][1].variants[0].stages[1].params["count"] != 999


@pytest.mark.parametrize("field,value", [("tags", None), ("requires", 1), ("findings", True)])
def test_invalid_metadata_sequences_remain_configuration_errors(field, value):
    from document_io import load_document
    from scenario.program_compiler import configure_program

    config = load_document(ROOT / "config/scenarios/request_completion.yaml")
    config["metadata"][field] = value
    with pytest.raises(DocumentError, match=field):
        configure_program(config, "invalid.yaml")

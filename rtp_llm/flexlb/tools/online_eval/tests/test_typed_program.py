"""真实构建与编译路径不经过内部文档或领域对象的字典读取器。"""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from case_api.builder import CaseBuilder
from case_api.gate_plugin import GatePlugin
from case_api.registry import CaseDefinition, MetricProducer, ProducerPhase, ReportView
from document_io import DocumentError
from scenario.catalog import handlers
from scenario.compiler import ScenarioCompiler
from scenario.execution_plan import freeze_plan, load_plan
from scenario.loader import load_scenarios
from scenario.plan import CompiledInstance, EnvironmentPlan, StagePlan
from scenario.program import ProgramPlan, StageDefinition, VariantPlan
from scenario.contracts import StageHandler

ROOT = Path(__file__).resolve().parents[1]


def gate(**kwargs):
    def report(directory, evidence, result, *, run=None, presentation=None):
        return directory
    return GatePlugin('example', lambda evidence: evidence, lambda a, b, c: None, report,
                      producer='example_metrics', describe=lambda a, b, c: {},
                      view='example.yaml', validate_view=lambda a, b, c: None, **kwargs)


def test_gate_registration_derives_producer_and_report_and_survives_definition_replacement():
    capability = gate(subtitle_fields=frozenset({'verdict'}))
    definition = CaseDefinition({'default': lambda case: None}, gates={'example': capability})
    assert definition.producers == {
        'example_metrics': MetricProducer(capability.describe, capability.project, ProducerPhase.GATE),
    }
    assert definition.report_views == {'example.yaml': ReportView(
        capability.validate_view, capability.render, frozenset({'verdict'}), capability.render_with_deadline,
    )}
    changed = replace(definition, builders={'default': lambda case: None})
    assert changed.producers == definition.producers
    assert changed.report_views == definition.report_views


@pytest.mark.parametrize('kind', ['producer', 'view', 'second_gate'])
def test_gate_registration_rejects_conflicting_capabilities(kind):
    capability = gate()
    kwargs = {'gates': {'example': capability}}
    if kind == 'producer':
        kwargs['producers'] = {'example_metrics': MetricProducer(lambda a, b, c: {}, lambda d: None, ProducerPhase.GATE)}
    elif kind == 'view':
        kwargs['report_views'] = {'example.yaml': ReportView(lambda a, b, c: None, lambda a, b, c: None)}
    else:
        kwargs['gates']['second'] = replace(capability, name='second')
    with pytest.raises(ValueError, match='conflicting gate'):
        CaseDefinition({'default': lambda case: None}, **kwargs)


@pytest.mark.parametrize('fields', [dict(producer=None), dict(describe=None), dict(view=None),
                                   dict(validate_view=None), dict(producer='invalid/name'),
                                   dict(view='default.yaml'), dict(describe=lambda a: None)])
def test_gate_registration_requires_complete_valid_bindings(fields):
    with pytest.raises(ValueError):
        replace(gate(), **fields)


def test_builder_returns_typed_isolated_steps_and_keeps_observation_purpose():
    builder = CaseBuilder({}, {}, number_rules={})
    params = {'nested': [1]}
    builder.step('setup', 'setup')
    builder.observe('read', 'observe', params=params, timeout_s=4)
    params['nested'].append(2)
    first = builder.finish()
    assert isinstance(first, tuple)
    assert all(isinstance(step, StageDefinition) for step in first)
    assert first[1].purpose == 'observation'
    assert first[1].params == {'nested': [1]}
    first[1].params['nested'].append(3)
    assert builder.finish()[1].params == {'nested': [1]}


def test_real_case_builds_typed_program_and_compiles_without_dictionary_decoders(tmp_path):
    documents = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')
    program = documents[0][1]
    assert isinstance(program, ProgramPlan)
    assert all(isinstance(variant, VariantPlan) for variant in program.variants)
    with patch.object(CompiledInstance, 'read', side_effect=AssertionError('instance roundtrip')), \
         patch.object(StagePlan, 'read', side_effect=AssertionError('stage roundtrip')), \
         patch.object(EnvironmentPlan, 'read', side_effect=AssertionError('environment roundtrip')):
        instances = ScenarioCompiler(handlers()).compile(documents)
    assert all(isinstance(stage, StagePlan) for instance in instances for stage in instance.stages)
    path = tmp_path / 'execution-plan.json'
    projected = [instance.to_dict() for instance in instances]
    digest = freeze_plan(path, projected)
    assert load_plan(path, digest) == projected
    # The same program can be compiled again without changes or retained compiler state.
    assert [instance.to_dict() for instance in ScenarioCompiler(handlers()).compile(documents)] == projected


@pytest.mark.parametrize('reference', ['stages.future.output.count', 'stages.setup.output.missing',
                                     'stages.adapter.output.count'])
def test_adapter_payload_references_are_checked_before_publishing_current_outputs(reference):
    program = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')[0][1]
    variant = program.variants[0]
    adapter = StageDefinition('adapter', 'custom', params={'nested': [{'$ref': reference}]})
    variant = replace(variant, stages=(variant.stages[0], adapter, *variant.stages[1:]))
    custom = StageHandler('custom', lambda params, context: params, lambda *_: None, {'count': 'integer'})
    with pytest.raises(DocumentError, match='unknown or forward reference'):
        ScenarioCompiler({'custom': custom}).compile([('fixture', replace(program, variants=(variant,)))])


def test_raw_internal_documents_are_not_a_compiler_input():
    with pytest.raises(DocumentError, match='expected ProgramPlan'):
        ScenarioCompiler().compile([('fixture', {'id': 'example', 'stages': []})])


def test_selection_does_not_share_mutable_configuration_or_steps_with_original():
    from scenario.suites import preselect_documents
    documents = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')
    selected = preselect_documents(documents, kind='functional')
    selected[0][1].environment['n_prefill'] = 123
    selected[0][1].variants[0].stages[1].params['count'] = 999
    assert documents[0][1].environment['n_prefill'] != 123
    assert documents[0][1].variants[0].stages[1].params['count'] != 999


@pytest.mark.parametrize('field,value', [('tags', None), ('requires', 1), ('findings', True)])
def test_invalid_metadata_sequences_remain_configuration_errors(field, value):
    from document_io import load_document
    from scenario.program_compiler import configure_program
    config = load_document(ROOT / 'config/scenarios/request_completion.yaml')
    config['metadata'][field] = value
    with pytest.raises(DocumentError, match=field):
        configure_program(config, 'invalid.yaml')

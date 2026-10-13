"""Compilation and selection preserve the same execution and provenance contract."""
import copy
import json
from pathlib import Path

import pytest

from input_contract import mapping_fields
from scenario.catalog import handlers
from scenario.compiler import compile_scenarios
from scenario.loader import ScenarioError, load_scenarios
from scenario.plan import CompiledInstance
from scenario.suites import classify
from scenario.validation import mapping

ROOT = Path(__file__).resolve().parents[1]


def plans(documents=None):
    if documents is None:
        documents = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')
    return compile_scenarios(documents, 'single-nonbatch', handlers())


def test_program_provenance_survives_plain_dict_and_json_transport():
    documents = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')
    source, document = documents[0]
    plain = json.loads(json.dumps(dict(document)))
    assert plain['implementation']['case_sources_sha256']
    assert plans([(source, plain)]) == plans(documents)


def test_selection_preserves_the_complete_execution_contract():
    compiled = plans()
    original = copy.deepcopy(compiled)
    selected = classify(compiled, kind='functional')
    assert selected == original
    assert selected[0]['test_kind'] == selected[0]['metadata']['kind']
    assert selected[0]['collection_profile'] == selected[0]['execution']['collection']
    selected[0]['execution']['monitoring']['capture_metrics'] = False
    assert compiled == original


@pytest.mark.parametrize('field,value', [
    ('test_kind', 'workload'), ('collection_profile', 'aggregate'),
    ('workload_runtime', {'capture_metrics': False}),
])
def test_frozen_policy_projection_cannot_override_canonical_policy(field, value):
    plan = plans()[0]
    plan[field] = value
    with pytest.raises(ScenarioError, match='policy projection disagrees'):
        CompiledInstance.read(plan)


def test_shared_mapping_rule_rejects_non_string_keys_at_both_boundaries():
    value = {1: 0, 'other': 0}
    with pytest.raises(ValueError, match='mapping keys must be strings'):
        mapping_fields(value, set(), 'input')
    with pytest.raises(ScenarioError, match='mapping keys must be strings'):
        mapping(value, 'input', set())


def test_program_document_requires_explicit_provenance():
    source, document = load_scenarios(ROOT / 'config/scenarios/request_completion.yaml')[0]
    del document['implementation']
    with pytest.raises(ScenarioError, match='implementation'):
        plans([(source, document)])

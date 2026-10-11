"""Analysis budgets are independent of request drain deadlines."""

import copy
from pathlib import Path

import pytest
import yaml

from cases.config import CaseBuilder
from cases.master_performance.program import default, NUMERIC_PARAMETERS
from cases.cache_scale_in.inputs import validate_criteria
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def build(parameters):
    case = CaseBuilder({}, parameters, number_rules=NUMERIC_PARAMETERS)
    case.validate_numbers()
    default(case)
    return {step['id']: step for step in case.steps}


def parameters():
    return yaml.safe_load((ROOT / 'config/scenarios/master_performance.yaml').read_text())['parameters']


def test_analysis_budget_is_not_derived_from_request_timeout():
    p = parameters()
    before = build(p)
    changed = copy.deepcopy(p)
    changed['traffic']['client']['TIMEOUT_MS'] = '10000'
    after = build(changed)
    assert before['gate']['timeout_s'] == after['gate']['timeout_s'] == 165
    assert before['drain']['timeout_s'] != after['drain']['timeout_s']
    changed['procedure']['analysis_timeout_s'] = 240
    extended = build(changed)
    assert extended['gate']['timeout_s'] == 240
    assert before['measure']['params']['criteria'] == extended['measure']['params']['criteria']


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan'), '165'])
def test_invalid_analysis_budget_is_rejected(value):
    p = parameters()
    p['procedure']['analysis_timeout_s'] = value
    with pytest.raises(ValueError, match='analysis_timeout_s'):
        build(p)


def test_missing_analysis_budget_is_rejected():
    p = parameters()
    del p['procedure']
    with pytest.raises(ValueError):
        build(p)


@pytest.mark.parametrize('field', ['intermediate_p', 'intermediate_hold_s'])
def test_cache_action_no_longer_accepts_hidden_staircase_inputs(field):
    with pytest.raises(ValueError, match='unknown'):
        validate_criteria({field: 60}, SimpleNamespace(path='scale_in'))

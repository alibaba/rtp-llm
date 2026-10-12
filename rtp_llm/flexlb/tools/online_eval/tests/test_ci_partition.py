"""Manual CI selection must run every requested instance exactly once."""
from unittest.mock import patch

import pytest

from scenario.loader import ScenarioError
from scripts.commands.plan_ci import build_selection, GROUPS


@pytest.fixture(autouse=True)
def worker_capacity():
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)), \
         patch('runtime.resource_plan.VICTIM_OFFSETS', (2048, 2049, 2050)):
        yield


def test_default_six_instances_partition_without_loss_or_duplicates():
    plan = build_selection('workload', ['single-nonbatch', 'batch-window'])
    groups = plan['groups']
    assert groups[GROUPS[0]] == ['cache_scale_in::default::single-nonbatch']
    assert groups[GROUPS[1]] == ['cache_scale_in::default::batch-window']
    assert set(groups[GROUPS[2]]) == {
        f'{case}::default::{profile}'
        for case in ('master_ha_failover', 'master_performance')
        for profile in ('single-nonbatch', 'batch-window')}
    assigned = [identity for ids in groups.values() for identity in ids]
    assert len(assigned) == len(set(assigned)) == 6
    assert set(assigned) == set(plan['instances'])


def test_explicit_controls_and_profiles_override_suite_selection():
    ids = ['cache_scale_in::affinity_disabled::batch-window',
           'master_ha_failover::non_rolling::single-nonbatch']
    plan = build_selection('unused', ['unused'], ','.join(ids))
    assert plan['groups'][GROUPS[0]] == []
    assert plan['groups'][GROUPS[1]] == ids[:1]
    assert plan['groups'][GROUPS[2]] == ids[1:]


def test_single_profile_omits_unrequested_cache_worker():
    plan = build_selection('workload', ['single-nonbatch'])
    assert len(plan['instances']) == 3
    assert plan['groups'][GROUPS[1]] == []


@pytest.mark.parametrize('ids', ['missing::default::batch-window',
                                'cache_scale_in::default::unknown',
                                'cache_scale_in::default',
                                ('cache_scale_in::default::batch-window,'
                                 'cache_scale_in::default::batch-window')])
def test_invalid_or_duplicate_ids_fail_before_any_worker(ids):
    with pytest.raises(ScenarioError):
        build_selection('workload', ['batch-window'], ids)


@pytest.mark.parametrize('profiles', [[], ['unknown'], ['batch-window', 'batch-window']])
def test_invalid_profile_selection_fails(profiles):
    with pytest.raises(ScenarioError):
        build_selection('workload', profiles)

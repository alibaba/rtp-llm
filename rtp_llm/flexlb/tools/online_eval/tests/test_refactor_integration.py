"""Preserve the validated workload controls across the execution refactor."""
from pathlib import Path
from unittest.mock import patch
import pytest
from document_io import load_document
from scenario.program_compiler import configure_program
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers
from scenario.suites import preselect_documents

ROOT = Path(__file__).resolve().parents[1]

def compile_case(data):
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)), patch('adapters.java_mock_resources.VICTIM_OFFSETS', (2048, 2049, 2050)):
        return compile_scenarios([('case.yaml', configure_program(data, 'case.yaml'))], handlers=handlers())

def test_manual_suite_preserves_exact_three_cases_per_profile():
    documents = preselect_documents(load_scenarios(ROOT/'config/scenarios'), 'workload')
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)), patch('adapters.java_mock_resources.VICTIM_OFFSETS', (2048, 2049, 2050)):
        plans = compile_scenarios(documents, handlers=handlers())
    assert {p['id'] for p in plans} == {f'{case}::default::{profile}' for case in ['cache_scale_in', 'master_ha_failover', 'master_performance'] for profile in ['single-nonbatch','batch-window']}

def test_cache_controls_share_inputs_and_keep_independent_gap_budgets():
    plans = compile_case(load_document(ROOT/'config/scenarios/cache_scale_in.yaml'))
    for profile in ['single-nonbatch','batch-window']:
        for archive in ('deepseek', 'glm'):
            pair = [p for p in plans if p['profile']==profile and
                    p['implementation']['variant_axes'][p['variant_id']]['data_archive']==archive]
            assert len(pair)==2
            assert pair[0]['environment']['rendered']==pair[1]['environment']['rendered']
            assert pair[0]['stages'][1]==pair[1]['stages'][1]
            for p in pair:
                criteria=next(s for s in p['stages'] if s['id']=='scale_in')['params']['criteria']
                expected=(900,1050,240,240,5) if archive=='glm' else (240,300,240,900,5)
                assert (criteria['warmup_min_s'],criteria['warmup_timeout_s'],criteria['observe_s'],criteria['qps'],criteria['max_gap_s'])==expected
                negative=p['variant_id'].endswith('affinity_disabled')
                assert criteria['expected_behavior']==('collapse' if negative else 'stable')
                assert criteria['post_affinity_ms']==(0 if negative else 1000000000)
                assert p['execution']['monitoring']['max_sample_gap_s']==10
                assert p['environment']['master_test_control'] is True

@pytest.mark.parametrize('mutation', ['missing_opt_in','wrong_initial','short_projection','bad_negative'])
def test_cache_intervention_rejects_incoherent_controls(mutation):
    data=load_document(ROOT/'config/scenarios/cache_scale_in.yaml')
    if mutation=='missing_opt_in':data['environment']['master_test_control']=False
    elif mutation=='wrong_initial':data['environment']['config_overrides']['cache_affinity_max_extra_ttft_ms']=0
    elif mutation=='short_projection':data['parameters']['traffic']['max_requests']=1000
    else:data['variants'][0]['parameters']['procedure']['post_affinity_ms']=100
    with pytest.raises(ValueError):compile_case(data)

def test_performance_fixed_window_patch_applies_only_to_batch_profile():
    plans=compile_case(load_document(ROOT/'config/scenarios/master_performance.yaml'))
    decisions={p['profile']:p['environment']['resolved_config']['scheduler']['decision'] for p in plans}
    assert decisions['single-nonbatch']=={'type':'SINGLE'}
    assert decisions['batch-window']['maxRequests']==64
    assert decisions['batch-window']['maxCollectionWaitMs']==350


def test_ci_cases_keep_generic_and_specific_report_views():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1] / "config/scenarios"
    for name in ("cache_scale_in", "master_ha_failover", "master_performance"):
        data = yaml.safe_load((root / (name + ".yaml")).read_text())
        assert data["reports"] == ["default.yaml", name + ".yaml"]


def test_cache_delivery_budgets_are_independent_of_observation():
    data = load_document(ROOT/'config/scenarios/cache_scale_in.yaml')
    for plan in compile_case(data):
        teardown = next(s for s in plan['stages'] if s['action'] == 'teardown')
        assert teardown['timeout_s'] == plan['execution']['cleanup_timeout_s'] == 660
        assert plan['execution']['finalize_timeout_s'] == 600
        assert plan['execution']['report_timeout_s'] == 300

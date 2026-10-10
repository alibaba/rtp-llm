"""Compilation selects consumers; archived producers cannot expand the catalog."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cases.config import configure_program
from monitoring.collection_plan import select_plan, instance_plan, physical_metrics
from monitoring.query_plan import load_plan, plan_hash
from reporting.view_config import view
from scenario.catalog import handlers
from scenario.compiler import compile_scenarios
from scenario.loader import load_document

ROOT = Path(__file__).resolve().parents[1]


def test_selection_is_union_and_default_view_does_not_enable_full_catalog():
    plan = load_plan('master_performance.yaml')
    selected = select_plan(plan, {'mock/rtp_llm_context_tps': {}}, [view('default.yaml')])
    assert list(selected['sources']['mock']) == ['rtp_llm_context_tps']
    assert selected['produced'] == {}
    assert physical_metrics(selected, 'mock') == ['rtp_llm_context_tps']
    assert plan == load_plan('master_performance.yaml')
    with pytest.raises(ValueError, match='undeclared collection demand'):
        select_plan(plan, {'mock/typo': {}}, [])


def test_case_compilation_freezes_variant_gate_chart_and_diagnostic_consumers():
    path = ROOT / 'config/scenarios/master_ha_failover.yaml'
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)):
        doc = configure_program(load_document(path), str(path))
        instances = compile_scenarios([(str(path), doc)], handlers=handlers())
    assert len(instances) == 4
    for instance in instances:
        frozen = instance['implementation']['monitoring_query_plan']
        assert frozen['sha256'] == plan_hash(frozen['definition'])
        assert instance_plan(instance) == frozen['definition']
        assert sum(map(len, frozen['definition']['sources'].values())) == 4
        assert len(frozen['definition']['produced']) == 19
        assert 'ha_gate/success_rate' in frozen['definition']['demand']['gate']
        assert 'ha/sent' in frozen['definition']['demand']['report']
    corrupted = copy.deepcopy(instances[0])
    corrupted['implementation']['monitoring_query_plan']['definition']['produced'].clear()
    with pytest.raises(ValueError, match='hash mismatch'):
        instance_plan(corrupted)


def test_producer_does_not_reload_full_catalog_or_publish_unselected_outputs(tmp_path):
    from monitoring.metric_store import export_metrics, MetricStore
    from cases.master_performance.metrics import produce
    from test_performance_gate import evidence
    from cases.master_performance.analysis import analyze
    selected = select_plan(load_plan('master_performance.yaml'),
                           {'performance_gate/goodput_rps': {}}, [])
    export_metrics(tmp_path, selected)
    e = evidence()
    with patch('monitoring.query_plan.load_plan', side_effect=AssertionError('runtime catalog reload')):
        produce(tmp_path, e, analyze(e))
    store = MetricStore.read(tmp_path)
    assert set(store.document['metrics']) == set(store.document['definitions'])
    assert 'request/ttft_p99_ms' not in store.document['definitions']
    assert store.select('performance_gate/goodput_rps')[0]['points'][0][1] == 10

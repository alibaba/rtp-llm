"""Canonical metric identities remain consistent across independently selected sets."""

from pathlib import Path

from monitoring.query_plan import DEFAULT_PLAN, definitions, load_plan, plan_hash
from monitoring.session import PrometheusSession


ROOT = Path(__file__).resolve().parents[1]


def meaning(definition):
    # Requiredness is a case-specific collection policy; it does not change the metric.
    return {key: definition.get(key) for key in (
        'source_type', 'source_kind', 'promql', 'producer', 'unit', 'value_kind', 'labels', 'measurement', 'calculation', 'exported_metrics',
    )} | {'mode': definition.get('mode', 'evaluated')}


def test_shared_metric_ids_have_one_meaning_across_shipped_sets():
    seen = {}
    for path in sorted((ROOT/'config/monitoring').glob('*.yaml')):
        for identity, definition in definitions(load_plan(path.name)).items():
            current = meaning(definition)
            if identity in seen:
                assert current == seen[identity], f'{path.name}: conflicting meaning for {identity}'
            seen[identity] = current


def test_default_session_uses_the_common_metric_set(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://127.0.0.1:1234/metrics'},
                                target_kinds={'mock': 'mock'})
    plan = load_plan(DEFAULT_PLAN)
    assert session.query_plan_name == 'default.yaml'
    assert session.query_plan == plan
    assert session.query_plan_sha256 == plan_hash(plan)
    assert definitions(plan)['mock/running_avg']['promql'] == 'avg by (role) (rtp_llm_running_stream_size${selector})'

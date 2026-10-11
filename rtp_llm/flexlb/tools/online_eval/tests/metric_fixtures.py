"""Synthetic archives explicitly freeze the metric definitions they exercise."""
import json
from pathlib import Path

from monitoring.metric_store import export_metrics
from monitoring.query_plan import load_plan


def freeze_metrics(directory, case):
    plan = load_plan(case + '.yaml')
    if case == 'master_ha_failover':
        from monitoring.collection_plan import select_plan
        from reporting.view_config import view
        gates = [identity for identity, spec in plan['produced'].items() if spec['value_kind'] == 'scalar']
        plan = select_plan(plan, gates, [view(case + '.yaml')])
    for path in Path(directory).glob('telemetry/*/queries.json'):
        data = json.loads(path.read_text())
        if 'metric_plan' not in data:
            data.update(metric_plan=plan, query_plan=case + '.yaml')
            path.write_text(json.dumps(data))
    return export_metrics(directory, plan)

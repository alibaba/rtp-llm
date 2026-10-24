"""Explicit case delivery helpers for frozen-evidence fixtures."""
from artifacts.gate import publish_gate
from runtime.deadline import Deadline
import time


def publish_cache(directory, evidence, result, **kwargs):
    from cases.cache_scale_in import metrics, report
    return publish_gate(directory, 'cache', evidence, result, metrics.produce, report.write_report, **kwargs)


def publish_performance(directory, evidence, result, **kwargs):
    from cases.master_performance import metrics, report
    return publish_gate(directory, 'performance', evidence, result, metrics.produce, report.write_report, **kwargs)


def finalize_metrics(directory, analysis):
    from scenario.registry import registry
    from monitoring.metric_store import MetricStore
    selected = {spec['producer'] for spec in MetricStore.read(directory).document['definitions'].values() if 'producer' in spec}
    analysis['metric_directory'] = str(directory)
    deadline = Deadline(time.monotonic()+30)
    for case in registry().cases.values():
        metrics = case.definition.metrics
        if metrics is not None and selected & metrics.CONTRACTS.keys() and hasattr(metrics, 'finalize'):
            metadata = metrics.finalize(directory, analysis, selected, deadline)
            analysis.setdefault('producer_results', {}).update(metadata)

"""Numeric producers are registered Python capabilities with explicit ownership."""

import json
from case_api.extension_contracts import FinalizationContext

from case_api.registry import registry
from case_api.registry import ProducerPhase
from monitoring.metric_store import MetricStore, MetricContractError


def produce(directory, context):
    store = MetricStore.read(directory)
    selected = {definition["producer"] for definition in store.document["definitions"].values()
                if "producer" in definition}
    if selected - set(registry().producers):
        raise MetricContractError("unknown metric producers: " + ", ".join(sorted(selected - set(registry().producers))))
    context["metric_directory"] = str(directory)
    for name in sorted(selected):
        capability = registry().producers[name]
        if capability.phase is ProducerPhase.FINALIZE:
            snapshot = FinalizationContext.snapshot(directory, context)
            metadata = capability.execute(directory, snapshot)
            if not isinstance(metadata, dict):
                raise MetricContractError(name + ': finalize producer must return metadata mapping')
            json.dumps(metadata, allow_nan=False)
            context.setdefault("producer_results", {})[name] = metadata

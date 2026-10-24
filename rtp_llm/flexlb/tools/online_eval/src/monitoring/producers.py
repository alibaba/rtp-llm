"""Numeric producers are registered Python capabilities with explicit ownership."""

import json
from case_api.extension_contracts import FinalizationContext

from case_api.registry import registry
from case_api.registry import ProducerPhase
from monitoring.metric_store import MetricStore
from monitoring.contracts import MetricContractError


def produce(directory, context, *, phase=ProducerPhase.FINALIZE, deadline=None):
    if not isinstance(phase, ProducerPhase):
        raise MetricContractError('producer phase must be ProducerPhase')
    store = MetricStore.read(directory)
    selected = {definition["producer"] for definition in store.document["definitions"].values()
                if "producer" in definition}
    capabilities = registry().producers
    if selected - set(capabilities):
        raise MetricContractError("unknown metric producers: " + ", ".join(sorted(selected - set(capabilities))))
    context["metric_directory"] = str(directory)
    for name in sorted(selected):
        capability = capabilities[name]
        if capability.phase is not phase:
            continue
        if deadline is not None:
            deadline.check()
        if phase is ProducerPhase.GATE:
            capability.execute(directory)
        else:
            snapshot = FinalizationContext.snapshot(directory, context, deadline)
            if deadline is not None:
                deadline.check()
            metadata = capability.execute(directory, snapshot)
            if deadline is not None:
                deadline.check()
            if not isinstance(metadata, dict):
                raise MetricContractError(name + ': finalize producer must return metadata mapping')
            json.dumps(metadata, allow_nan=False)
            context.setdefault("producer_results", {})[name] = metadata
        if deadline is not None:
            deadline.check()

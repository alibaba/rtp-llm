"""Numeric producers are registered Python capabilities with explicit ownership."""

from importlib import import_module
from monitoring.metric_store import MetricStore, MetricContractError

# Gate producers run with their frozen gate evidence before rendering. Finalize
# producers consume case evidence after cleanup; they never change gate verdicts.
PRODUCERS = {
    "ha_gates": ("cases.master_ha_failover.metrics", "gate", None),
    "ha_evidence": ("cases.master_ha_failover.metrics", "finalize", "ha_metric_metadata"),
    "performance_requests": ("cases.master_performance.metrics", "gate", None),
    "cache_windows": ("cases.cache_scale_in.metrics", "gate", None),
}


def produce(directory, context):
    store = MetricStore.read(directory)
    selected = {definition["producer"] for definition in store.document["definitions"].values()
                if "producer" in definition}
    if selected - set(PRODUCERS):
        raise MetricContractError("unknown metric producers: " + ", ".join(sorted(selected - set(PRODUCERS))))
    context["metric_directory"] = str(directory)
    for name in sorted(selected):
        module, phase, metadata_key = PRODUCERS[name]
        if phase == "finalize":
            context[metadata_key] = import_module(module).produce(directory, context)

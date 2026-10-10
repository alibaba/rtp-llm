"""Local field names bind explicit metric IDs and label projections."""

from monitoring.identity import METRIC_ID, NAME as FIELD


def metric_fields(spec):
    if (not isinstance(spec, dict) or set(spec) != {"source", "fields"}
            or spec["source"] != "metric_store" or not isinstance(spec["fields"], dict)
            or not spec["fields"]):
        raise ValueError("metric input requires metric_store and fields")
    identities = set()
    for name, binding in spec["fields"].items():
        if (type(name) is not str or not FIELD.fullmatch(name)
                or not isinstance(binding, dict) or set(binding) != {"metric", "labels"}
                or type(binding["metric"]) is not str or not METRIC_ID.fullmatch(binding["metric"])
                or not isinstance(binding["labels"], dict)
                or any(type(k) is not str or not FIELD.fullmatch(k)
                       or type(v) is not str or not v for k, v in binding["labels"].items())):
            raise ValueError("invalid metric field binding: " + str(name))
        identity = (binding["metric"], tuple(sorted(binding["labels"].items())))
        if identity in identities:
            raise ValueError("duplicate metric field projection: " + name)
        identities.add(identity)
    return spec["fields"]


ENGINE_IDENTITY_LABELS = ("role", "engine_name", "engine_incarnation")


def bind_engine_metrics(case, spec, units):
    """Assert consumer dimensions against the selected plan's authoritative metadata.

    Extra identity labels are valid; a consumer states the minimum it needs.
    The mock target is the registered TSDB target, not the metric-store backend.
    """
    bindings = metric_fields(spec)
    for field, binding in bindings.items():
        if field not in units or not binding["metric"].startswith("mock/"):
            raise ValueError("unknown raw engine metric binding: " + field)
        case.metric(binding["metric"], unit=units[field], labels=ENGINE_IDENTITY_LABELS, mode="scrape")

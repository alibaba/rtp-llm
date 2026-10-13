"""Local field names bind explicit metric IDs and label projections."""

from monitoring.identity import METRIC_ID, NAME as FIELD


def metric_fields(spec):
    if (not isinstance(spec, dict) or set(spec) != {"fields"}
            or not isinstance(spec["fields"], dict)
            or not spec["fields"]):
        raise ValueError("metric input requires fields; the program owns the reader")
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


def bind_metrics(case, spec, units, *, labels=(), mode=None, namespace=None):
    """Bind any registered metric namespace with explicit consumer dimensions."""
    bindings = metric_fields(spec)
    for field, binding in bindings.items():
        if field not in units or (namespace is not None and not binding["metric"].startswith(namespace + "/")):
            raise ValueError("unknown raw metric binding: " + field)
        case.metric(binding["metric"], unit=units[field], labels=labels, mode=mode)


def bind_mock_engine_metrics(case, spec, units):
    """Mock scrape contract; other clients use bind_metrics with their dimensions."""
    return bind_metrics(case, spec, units, labels=ENGINE_IDENTITY_LABELS, mode="scrape", namespace="mock")

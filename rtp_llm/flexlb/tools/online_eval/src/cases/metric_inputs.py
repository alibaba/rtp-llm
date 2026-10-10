"""Local field names bind explicit metric IDs and label projections."""

import re

METRIC_ID = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")
FIELD = re.compile(r"[a-z][a-z0-9_]*\Z")


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

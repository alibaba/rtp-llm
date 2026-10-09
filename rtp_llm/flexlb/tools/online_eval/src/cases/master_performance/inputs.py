"""Validate the case gate inputs and their declared metric bindings."""

import re


METRIC = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")


def engine_tps(spec, bounds=None):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"source", "metric_roles"}
        or spec["source"] != "metric_store"
    ):
        raise ValueError("engine TPS input requires metric_store and metric_roles")
    roles = spec["metric_roles"]
    if (
        not isinstance(roles, dict)
        or len(roles) != 3
        or any(
            not isinstance(k, str)
            or not METRIC.fullmatch(k)
            or v not in ("prefill", "decode")
            for k, v in roles.items()
        )
        or list(roles.values()).count("prefill") != 2
        or list(roles.values()).count("decode") != 1
        or (bounds is not None and set(bounds) != set(roles))
    ):
        raise ValueError("engine TPS floors must match YAML metric_roles")
    return spec

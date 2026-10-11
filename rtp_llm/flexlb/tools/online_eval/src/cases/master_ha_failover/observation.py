"""HA Master state sampling and ledger projection."""

import math

from runtime.network import master_url


STATE_FIELDS = {"http_up", "scheduler_inflight", "prefill_inflight_requests",
                "decode_master_queued", "decode_confirmed_running"}


def master_state_fields(data, fields=STATE_FIELDS - {"http_up"}):
    """Project a successful response; absent ledger fields are contract errors."""
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("HA state requires finite nonnegative ledger values")
        return value

    def total(endpoints, field):
        if not isinstance(endpoints, list):
            raise ValueError("HA state endpoint ledger must be a list")
        if any(not isinstance(endpoint, dict) or field not in endpoint for endpoint in endpoints):
            raise ValueError("HA state endpoint lacks " + field)
        return sum(number(endpoint[field]) for endpoint in endpoints)

    if not fields <= STATE_FIELDS - {"http_up"} or not isinstance(data, dict):
        raise ValueError("invalid Master state projection")
    result = {}
    for field in fields:
        if field == "scheduler_inflight":
            if field not in data:
                raise ValueError("HA state response lacks required ledger fields")
            result[field] = number(data[field])
        else:
            endpoints, key = {
                "prefill_inflight_requests": ("prefill_endpoints", "inflight_requests"),
                "decode_master_queued": ("decode_endpoints", "master_queued"),
                "decode_confirmed_running": ("decode_endpoints", "confirmed_running"),
            }[field]
            if endpoints not in data:
                raise ValueError("HA state response lacks required ledger fields")
            result[field] = total(data[endpoints], key)
    return result


def master_adapters(env, fields):
    """Case-owned protocol interpretation, with public lifecycle and field selection."""
    from monitoring.collectors import HttpJsonAdapter
    if not fields or not fields <= STATE_FIELDS:
        raise ValueError("unknown Master state fields")
    def adapter(name, spec):
        return HttpJsonAdapter(
            master_url(spec.bind_ip, spec.http_port, "inflight"),
            lambda data: dict(master=name, **({"http_up": 1} if "http_up" in fields else {}),
                              **master_state_fields(data, fields - {"http_up"})),
            lambda: dict(master=name, **({"http_up": 0} if "http_up" in fields else {})),
        )
    return {name: adapter(name, spec) for name, spec in env.master_specs.items()}

"""Decode Master owner ledgers without readiness, polling or case policy."""


def scheduler_inflight(data):
    value = data["scheduler_inflight"]
    if type(value) is not int or value < 0:
        raise ValueError("invalid scheduler inflight count")
    return value


def endpoint_loads(data):
    loads = {}
    for side in ("prefill", "decode"):
        rows = data[side + "_endpoints"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing endpoint ledger observations")
        values = []
        for row in rows:
            key = "inflight_batches" if side == "prefill" else "total_load"
            if side == "decode" and "total_load" not in row:
                key = "inflight_requests"
            value = row[key]
            if isinstance(value, list) and side == "prefill":
                value = len(value)
            if type(value) is not int or value < 0:
                raise ValueError("invalid endpoint ledger count")
            values.append(value)
        loads[side] = values
    return loads


def zero_owner_loads(data):
    """Keep both legacy Decode counters authoritative when both are present."""
    loads = endpoint_loads(data)
    decode = []
    for row in data["decode_endpoints"]:
        for field in ("inflight_requests", "total_load"):
            if field in row and (type(row[field]) is not int or row[field] < 0):
                raise ValueError("invalid decode inflight count")
        decode.append(row.get("inflight_requests", 0) or row.get("total_load", 0))
    loads["decode"] = decode
    return loads


def worker_counts(info, roles):
    """Missing role counts stay unknown and cannot satisfy topology readiness."""
    summary = info["worker_summary"]
    if summary is None:
        summary = {}
    if not isinstance(summary, dict):
        raise ValueError("invalid worker summary")
    counts = {}
    for role in roles:
        row = summary.get(role)
        if row is None:
            # The role may be absent between removal and rediscovery.
            # Preserve unknown; it cannot satisfy readiness.
            counts[role] = None
            continue
        if not isinstance(row, dict):
            raise ValueError("invalid worker role summary")
        values = (row["discovered"], row["alive"])
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("invalid topology counts")
        counts[role] = values
    return counts

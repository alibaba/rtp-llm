"""Deadline-bounded JSON requests and engine snapshots from the mock server."""

import copy
import json
import re
import urllib.request


ENGINE_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")


def _request(ops, path, deadline, body, timeout_s):
    deadline.check()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ops.mock_http_port}/{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=min(timeout_s, deadline.remaining()))


def mock_json(ops, path, deadline, body=None):
    """Mock workload operations may wait up to 65 seconds within the stage budget."""
    with _request(ops, path, deadline, body, 65.0) as response:
        return json.load(response)


def engine_snapshot(ctx, deadline):
    return {e["name"]: e for e in mock_json(ctx.ops, "snapshot", deadline)["engines"]}


def control_json(ops, endpoint, deadline, body=None):
    """Engine controls require a bounded JSON object response within 15 seconds."""
    with _request(ops, endpoint, deadline, body, 15) as response:
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("control response exceeds byte budget")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("control response must be a JSON object")
        return result


def select_engines(snapshot, targets):
    entries = snapshot.get("engines")
    if not isinstance(entries, list):
        raise ValueError("snapshot has no engine list")
    selected = {}
    for name in targets:
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"engine {name!r} missing or ambiguous")
        entry = matches[0]
        if (
            entry.get("role") not in {"prefill", "decode", "pdfusion"}
            or not isinstance(entry.get("grpc_addr"), str)
            or type(entry.get("stopped")) is not bool
        ):
            raise ValueError(
                f"engine {name!r} snapshot lacks role/address/stopped evidence"
            )
        selected[name] = copy.deepcopy(entry)
    return selected


def topology_pools(ctx, deadline, address_field="grpc_addr"):
    raw = control_json(ctx.ops, "snapshot?view=topology", deadline)
    pools = {
        role: sorted(r[address_field] for r in raw["engines"] if r["role"] == role)
        for role in ("prefill", "decode")
    }
    expected = {"prefill": ctx.env.spec.n_prefill, "decode": ctx.env.spec.n_decode}
    if any(
        len(pools[role]) != expected[role] or len(set(pools[role])) != expected[role]
        for role in pools
    ):
        raise ValueError("mock pool differs from configured topology")
    return pools

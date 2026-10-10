"""Driver identity and source-owned request priority at the execution boundary."""

import copy

from input_contract import NumberRule

PRIORITY = NumberRule(integer=True, minimum=1, maximum=100)
JAVA_LENGTH = NumberRule(integer=True, minimum=1, maximum=(1 << 31) - 1)


def source_priority(source):
    parameters = source.get("parameters") if isinstance(source, dict) else None
    priority = parameters.get("priority") if isinstance(parameters, dict) else None
    try:
        return PRIORITY.validate(priority, "traffic.source.parameters.priority")
    except ValueError as exc:
        raise ValueError(f"traffic.source.parameters.priority must be an integer in [{PRIORITY.minimum},{PRIORITY.maximum}]") from exc


def java_client_priority(source, client):
    priority = str(source_priority(source))
    if "PRIORITY" in client and client["PRIORITY"] != priority:
        raise ValueError("client PRIORITY conflicts with traffic source priority")
    return dict(client, PRIORITY=priority)


def uniform_gate_flow(flow, *, duration_s, trace_duration_s):
    """Validate the shared Java sender contract; case code declares its coverage budget."""
    from traffic.playback_config import normalize
    client, _ = normalize(flow["client"])
    if client.get("SEND_MODE") != "uniform" or client.get("LOOP") != "false":
        raise ValueError("gate requires a nonlooping uniform Java workload")
    qps = float(client["SEND_MODE_QPS"])
    if (int(client["DURATION_S"]) < duration_s
            or flow["source"]["parameters"]["count"] < trace_duration_s * qps):
        raise ValueError("traffic plan must cover worst-case observation duration")
    return client


def driver(spec, expected):
    value = copy.deepcopy(spec)
    if type(value.get("kind")) is not str or value.pop("kind") != expected:
        raise ValueError("traffic.kind must be " + expected)
    if expected == "java_flow":
        value["client"] = java_client_priority(value["source"], value["client"])
    elif expected == "ha_replay":
        source_priority(value["source"])
    return value

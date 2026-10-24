"""Driver identity and source-owned request priority at the execution boundary."""

import copy

from traffic.request_rules import PRIORITY
from adapters.client_configuration import ClientConfiguration
from traffic.playback.configuration import PacingMode


def source_priority(source):
    parameters = source.get("parameters") if isinstance(source, dict) else None
    priority = parameters.get("priority") if isinstance(parameters, dict) else None
    try:
        return PRIORITY.validate(priority, "traffic.source.parameters.priority")
    except ValueError as exc:
        raise ValueError(f"traffic.source.parameters.priority must be an integer in [{PRIORITY.minimum},{PRIORITY.maximum}]") from exc


def uniform_gate_flow(flow, *, duration_s, trace_duration_s):
    """Validate the shared Java sender contract; case code declares its coverage budget."""
    client = ClientConfiguration.read(flow["client"])
    if client.playback.mode != PacingMode.UNIFORM or client.playback.max_laps != 1:
        raise ValueError("gate requires a nonlooping uniform Java workload")
    qps = client.playback.qps
    if (client.playback.duration_s < duration_s
            or min(flow["source"]["parameters"]["count"], flow.get("max_requests", flow["source"]["parameters"]["count"])) < trace_duration_s * qps):
        raise ValueError("traffic plan must cover worst-case observation duration")
    return client


def driver(spec, expected):
    value = copy.deepcopy(spec)
    if type(value.get("kind")) is not str or value.pop("kind") != expected:
        raise ValueError("traffic.kind must be " + expected)
    if expected == "java_flow":
        source_priority(value["source"])
    elif expected == "ha_replay":
        source_priority(value["source"])
    return value

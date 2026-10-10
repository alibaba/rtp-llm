"""Driver identity and source-owned request priority at the execution boundary."""

import copy


def source_priority(source):
    parameters = source.get("parameters") if isinstance(source, dict) else None
    priority = parameters.get("priority") if isinstance(parameters, dict) else None
    if type(priority) is not int or not 1 <= priority <= 100:
        raise ValueError("traffic.source.parameters.priority must be an integer in [1,100]")
    return priority


def java_client_priority(source, client):
    priority = str(source_priority(source))
    if "PRIORITY" in client and client["PRIORITY"] != priority:
        raise ValueError("client PRIORITY conflicts with traffic source priority")
    return dict(client, PRIORITY=priority)


def driver(spec, expected):
    value = copy.deepcopy(spec)
    if type(value.get("kind")) is not str or value.pop("kind") != expected:
        raise ValueError("traffic.kind must be " + expected)
    if expected == "java_flow":
        value["client"] = java_client_priority(value["source"], value["client"])
    elif expected == "ha_replay":
        source_priority(value["source"])
    return value

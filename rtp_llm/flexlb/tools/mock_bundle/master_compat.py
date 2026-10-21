"""Test packaging bridge; the legacy master receives its configuration unchanged."""
import json


def current_discovery(endpoints, static=False, kvcm_json=None):
    """Use the selected Master's public topology contract; preserve engine ports."""
    env = dict(endpoints["env"])
    service = json.loads(env["MODEL_SERVICE_CONFIG"])
    if static:
        service.pop("discovery_file", None)
        service.pop("load_balance", None)
        for group in service["role_endpoints"]:
            for role in ("prefill", "decode"):
                endpoint = group.get(role + "_endpoint")
                if endpoint is None:
                    continue
                hosts = [e["http_addr"] for e in endpoints["engines"] if e["role"] == role]
                if not hosts:
                    raise ValueError("static bundle discovery requires both P and D")
                endpoint["discovery"] = {"type": "static-env", "hosts": hosts}
    if kvcm_json:
        kvcm = json.loads(kvcm_json)
        if not isinstance(kvcm, dict) or not kvcm.get("address") or not kvcm.get("discovery"):
            raise ValueError("MOCK_BUNDLE_KVCM_CONFIG_JSON requires a KVCM address and discovery")
        service["kvcm"] = kvcm
    env["MODEL_SERVICE_CONFIG"] = json.dumps(service)
    return env


def mock_formula_config(raw, legacy, mock_expression=None):
    config = json.loads(raw)
    if mock_expression is not None and (not isinstance(mock_expression, str) or not mock_expression.strip()):
        raise ValueError("MOCK_PREFILL_EXECUTION_FORMULA must be a nonempty string")
    if not legacy:
        if config.get("schemaVersion", 3) != 3:
            raise ValueError("legacy master config requires MOCK_BUNDLE_LEGACY_MASTER=1")
        if mock_expression is None:
            return raw
        config["router"]["roles"]["prefill"]["executionTimeEstimator"] = {
            "type": "FORMULA", "expression": mock_expression}
        return json.dumps(config)
    if config.get("schemaVersion") != 1:
        raise ValueError("legacy master requires schemaVersion 1")
    estimator = config["router"]["roles"]["prefill"]["executionTimeEstimator"]
    if estimator.get("type") != "FORMULA" or not estimator.get("expression"):
        raise ValueError("mixed-version mock requires an explicit P formula")
    if mock_expression is not None:
        estimator = {"type": "FORMULA", "expression": mock_expression}
    # Only the mock's timing parser consumes this projection. It is never sent
    # to master: capacity, ordering, lifecycle and selectors stay on schema 1.
    return json.dumps({"schemaVersion": 3,
        "requestLifecycle": {"request": {"timeoutMs":
            config["scheduler"]["lifecycle"]["staleInflightTimeoutMs"]}},
        "router": {"roles": {
        "prefill": {"executionTimeEstimator": estimator}}}})


def legacy_discovery(endpoints, file_discovery=False):
    env = dict(endpoints["env"])
    service = json.loads(env["MODEL_SERVICE_CONFIG"])
    discovery_file = service.pop("discovery_file", None)
    if file_discovery:
        if not discovery_file:
            raise ValueError("file discovery requires the mock discovery path")
        env["MOCK_DISCOVERY_FILE"] = discovery_file
    env["MODEL_SERVICE_CONFIG"] = json.dumps(service)
    for role in ("prefill", "decode"):
        addresses = [e["http_addr"] for e in endpoints["engines"] if e["role"] == role]
        if not addresses:
            raise ValueError("legacy bundle requires both P and D")
        env["DOMAIN_ADDRESS:" + endpoints[role + "_domain"]] = ",".join(addresses)
    return env

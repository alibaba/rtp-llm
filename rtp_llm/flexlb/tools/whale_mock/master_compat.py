"""Test packaging bridge; the legacy master receives its configuration unchanged."""
import json


def mock_formula_config(raw, legacy):
    config = json.loads(raw)
    if not legacy:
        if config.get("schemaVersion", 3) != 3:
            raise ValueError("legacy master config requires MOCK_BUNDLE_LEGACY_MASTER=1")
        return raw
    if config.get("schemaVersion") != 1:
        raise ValueError("legacy master requires schemaVersion 1")
    estimator = config["router"]["roles"]["prefill"]["executionTimeEstimator"]
    if estimator.get("type") != "FORMULA" or not estimator.get("expression"):
        raise ValueError("mixed-version mock requires an explicit P formula")
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

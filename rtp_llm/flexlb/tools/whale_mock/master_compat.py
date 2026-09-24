"""Test packaging bridge; the legacy master receives its configuration unchanged."""
import json

# Default prefill execution formula used when the legacy flat config has no
# explicit estimator. Matches the production DSv4 Flash fit.
_DEFAULT_PREFILL_FORMULA = (
    "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717"
    " + 2.30134977837751 * batchSize"
    " + 0.158123254797307 * sum(hitCacheTokens / 1024.)"
    " + 0.575522710053703 * sum(computeTokens / 1024.)"
    " + 0.0517623430739831 * sum(computeTokens / 1024. * computeTokens / 1024.)"
    " + 0.0395308136993267 * sum(hitCacheTokens / 1024. * computeTokens / 1024.)"
    " + 0.0104363634681015 * sum(hitCacheTokens / 1024. * hitCacheTokens / 1024.)"
    " + 0.575522710053703 * max(sum(computeTokens / 1024.) - 16, 0)"
    " + 2.82077211814514 * max(sum(computeTokens / 1024.) - 32, 0)"
    " - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0)"
    " + 2.15779213792494 * max(sum(computeTokens / 1024.) - 96, 0)"
    " + 0.247806025472364 * max(sum(hitCacheTokens / 1024.) - 32, 0)"
    " - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0)"
    " - 0.427317020061895 * max(sum(hitCacheTokens / 1024.) - 128, 0)"
    " + 0.347029077528455 * max(sum(hitCacheTokens / 1024.) - 256, 0)"
    " - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, 0)"
    " + 2.30134977837751 * max(batchSize - 8, 0)"
    " - 3.54884859699154 * max(batchSize - 16, 0)"
    " - 11.3438560779984 * max(batchSize - 24, 0)"
    " + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0))"
    " + 0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0))"
    " - 0.0513345988517118 * sum(max(computeTokens / 1024. - 8, 0))"
    " - 0.332584389129357 * sum(max(hitCacheTokens / 1024. - 2, 0))"
    " + 0.305819761192588 * sum(max(hitCacheTokens / 1024. - 4, 0))"
    " - 0.287610979974721 * sum(max(hitCacheTokens / 1024. - 8, 0))"
    " + 0.191310200712013 * sum(max(hitCacheTokens / 1024. - 12, 0))"
    " + 0.0130251644478961 * max(batchSize - 8, 0) * sum(hitCacheTokens / 1024.)"
    " + 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / 1024.)"
    " - 0.0299132587297009 * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.)"
    " + 0.0447455122487382 * max(batchSize - 8, 0) * sum(computeTokens / 1024.)"
    " + 0.0104635312001851 * max(batchSize - 16, 0) * sum(computeTokens / 1024.)"
    " + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens / 1024.))))"
)


def mock_formula_config(raw, legacy, mock_expression=None):
    config = json.loads(raw)
    if mock_expression is not None and (not isinstance(mock_expression, str) or not mock_expression.strip()):
        raise ValueError("MOCK_PREFILL_EXECUTION_FORMULA must be a nonempty string")
    if not legacy:
        # New master (0ca29bf6f+): accepts schemaVersion 1 or 3.
        schema = config.get("schemaVersion", 3)
        if schema not in (1, 3):
            raise ValueError(f"unsupported schemaVersion {schema} for non-legacy master")
        if mock_expression is None:
            return raw
        config["router"]["roles"]["prefill"]["executionTimeEstimator"] = {
            "type": "FORMULA", "expression": mock_expression}
        return json.dumps(config)
    # Legacy master (d9a0accf3 and earlier): flat config without schemaVersion,
    # or schemaVersion 1 with router.roles structure.
    schema = config.get("schemaVersion")
    if schema == 1 and "router" in config:
        # Schema-1 structured config for legacy master.
        estimator = config["router"]["roles"]["prefill"]["executionTimeEstimator"]
        if estimator.get("type") != "FORMULA" or not estimator.get("expression"):
            raise ValueError("mixed-version mock requires an explicit P formula")
        if mock_expression is not None:
            estimator = {"type": "FORMULA", "expression": mock_expression}
        return json.dumps({"schemaVersion": 3,
            "requestLifecycle": {"request": {"timeoutMs":
                config["scheduler"]["lifecycle"]["staleInflightTimeoutMs"]}},
            "router": {"roles": {
            "prefill": {"executionTimeEstimator": estimator}}}})
    elif schema is None and "router" not in config:
        # Flat config (pre-schema): no formula available in config.
        # Use MOCK_PREFILL_EXECUTION_FORMULA or the default production formula.
        expression = mock_expression or _DEFAULT_PREFILL_FORMULA
        timeout_ms = config.get("staleInflightTimeoutMs", 300000)
        return json.dumps({"schemaVersion": 3,
            "requestLifecycle": {"request": {"timeoutMs": timeout_ms}},
            "router": {"roles": {
            "prefill": {"executionTimeEstimator": {
                "type": "FORMULA", "expression": expression}}}}})
    else:
        raise ValueError(f"unsupported legacy config shape (schemaVersion={schema})")


def config_generation(raw):
    """Classify a FLEXLB_CONFIG by the master generation that consumes it.

    "flat" is the pre-schema master (d9a0accf3 and earlier): a schema-less
    document that serves only HTTP ports and never binds the master gRPC
    listener. "versioned" is any schema-bearing master (schema 1/3), which
    does open it. This is the single source of truth for the generation
    question; consumers translate it into their own consequence (readiness
    port, formula projection) instead of re-parsing the document themselves.
    """
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("FLEXLB_CONFIG must be a JSON object")
    return "versioned" if "schemaVersion" in config else "flat"


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

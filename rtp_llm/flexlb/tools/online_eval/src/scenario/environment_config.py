"""Decode authored environment settings before preset and profile resolution."""
import re
from adapters.perf_presets import preset_names
from adapters.zookeeper_config import ZooKeeperConfig
from scenario.validation import fail, mapping, number
from scenario.environment_declaration import EnvironmentDeclaration, environment_fields, optional_scalars, captured_fields
from scenario.environment_resolver import EnvironmentResolver

def read_environment(value, path):
    value = mapping(
        value,
        path,
        environment_fields(),
    )
    if value.get("backend", "java_mock") != "java_mock":
        fail(
            path + ".backend",
            "only java_mock is implemented; GPU/multi-host/TP/DP layouts need separate capabilities",
        )
    result = {"backend": "java_mock", "n_prefill": 2, "n_decode": 4}
    for key, default, allowed in (
        ("discovery", "file", ("file", "discovery_file")),
        ("perf_preset", "default", preset_names()),
        ("master_layout", "single", ("single", "dual_standalone", "dual_zookeeper")),
    ):
        val = value.get(key, default)
        if val not in allowed:
            fail(path + "." + key, f"invalid value {val!r}; expected one of {allowed}")
        result[key] = val
    if result['master_layout'] == 'dual_zookeeper':
        try:
            result['zookeeper'] = ZooKeeperConfig.read(value.get('zookeeper', {})).to_dict()
        except ValueError as exc:
            fail(path + '.zookeeper', str(exc))
    elif 'zookeeper' in value:
        fail(path + '.zookeeper', 'requires dual_zookeeper layout')
    resolver = EnvironmentResolver.read(value, path, result["perf_preset"])
    captured = resolver.captured
    for field, capture_field in captured_fields().items():
        if field not in value and capture_field in captured:
            result[field] = captured[capture_field]
    if "model_override" in value:
        override = mapping(value["model_override"], path + ".model_override",
            {"baseline", "reason"}, {"baseline", "reason"})
        if override["baseline"] != result["perf_preset"] or not isinstance(override["reason"], str) or not override["reason"].strip():
            fail(path + ".model_override", "baseline must name perf_preset and reason must be nonempty")
        result["model_override"] = dict(override)
    optional_scalars(value, path, result)
    if "metric_whitelist" in value:
        whitelist = value["metric_whitelist"]
        if not isinstance(whitelist, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*){0,15}", whitelist
        ):
            fail(
                path + ".metric_whitelist",
                "expected 1..16 comma-separated metric identifiers",
            )
        if len(whitelist) > 1024:
            fail(path + ".metric_whitelist", "metric whitelist exceeds byte budget")
        result["metric_whitelist"] = whitelist
    if result["master_sync_log"] and result["master_layout"] != "single":
        fail(
            path + ".master_sync_log",
            "isolated sync log currently requires single master",
        )
    result["master_stable_window_s"] = number(
        value.get("master_stable_window_s", 3), path + ".master_stable_window_s"
    )
    if "prefill_cache_policy" in value:
        field = path + ".prefill_cache_policy"
        keys = {"device_tree", "memory_tree", "memory_blocks"}
        cache = mapping(value["prefill_cache_policy"], field, keys, {"device_tree", "memory_tree"})
        for key in ("device_tree", "memory_tree"):
            if type(cache[key]) is not bool:
                fail(field + "." + key, "expected boolean")
        if "memory_blocks" in cache:
            number(cache["memory_blocks"], field + ".memory_blocks", integer=True)
        result["prefill_cache_policy"] = dict(cache)
    if "prefill_perf" in value:
        field = path + ".prefill_perf"
        required = {"fixed_ms", "scale", "max_batch_tokens", "max_batch_requests"}
        perf = mapping(value["prefill_perf"], field, required, required)
        result["prefill_perf"] = {
            key: number(val, field + "." + key, integer=key.startswith("max_batch_"))
            for key, val in perf.items()
        }
    for key in ("n_prefill", "n_decode"):
        if key in value:
            result[key] = number(value[key], path + "." + key, minimum=1, integer=True)
    unhandled = set(value) - set(result) - {"config_overrides", "profile_overrides"}
    if unhandled:
        fail(path, "declared environment fields have no decoder: " + str(sorted(unhandled)))
    return EnvironmentDeclaration.read_normalized(result), resolver


def environment(value, path, profile):
    declaration, resolver = read_environment(value, path)
    return resolver.resolve(declaration, path, profile).to_dict()

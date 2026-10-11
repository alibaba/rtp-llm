"""Validate and resolve profile, preset and case environment settings."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, fields, field, asdict
from flexlb_cfg import (
    OMIT, PROFILE_CAPS, ProfileIdentityError, PROFILES, VICTIM_STAGES,
    ConfigOverride, render_env, validate_profile_identity,
)
from runtime.perf_presets import load_preset_bundle, preset_names
from scenario.loader import ScenarioError
from scenario.validation import fail, mapping, number, names

MASTER_PATCH_FIELDS = {field.name for field in fields(ConfigOverride)} - {"strip_preemption"}

_ABSENT = object()



def environment_fields():
    """One vocabulary for both top-level fields and simple scalar declarations."""
    return {field.name for field in fields(EnvironmentDeclaration)} | {"config_overrides", "profile_overrides"}


def variant_environment_fields():
    # A variant may change declared data, but never switch its case's backend.
    return environment_fields() - {"backend"}


def optional_scalars(value, path, result, declarations=None):
    """Apply simple scalar rules; retain each field's old omission semantics."""
    if declarations is None:
        declarations = OPTIONAL_SCALARS
    for key, (kind, default, minimum) in declarations.items():
        if key not in value and default is _ABSENT:
            continue
        val = value.get(key, default)
        if kind is bool:
            if type(val) is not bool:
                fail(path + "." + key, "expected boolean")
            result[key] = val
        else:
            result[key] = number(val, path + "." + key, minimum=minimum, integer=True)

CAPABILITIES = set().union(*PROFILE_CAPS.values()) | {
    "priority",
    "preemption",
    "engine_cancellation",
}


def effective_capabilities(config):
    scheduler, dispatcher = config["scheduler"], config["dispatcher"]
    axes = dict(
        scheduler=scheduler["type"],
        ordering=scheduler["ordering"]["type"],
        decision=scheduler["decision"]["type"],
        dispatcher=dispatcher["type"],
    )
    caps = {"queue", axes["ordering"].lower(), axes["decision"].lower()}
    if dispatcher["type"] == "BATCH":
        caps.update({"batch_dispatch", "enqueue_batch", "fetch_response"})
    elif dispatcher["type"] == "NON_BATCH":
        caps.update({"non_batch_dispatch", "frontend_send", "generate_stream"})
    else:
        raise ValueError("unrecognized effective dispatcher")
    preemption = scheduler["ordering"].get("preemption")
    if preemption is None and scheduler["ordering"]["type"] == "PRIORITY":
        preemption = {"allowedVictimStages": list(VICTIM_STAGES)}
    if preemption:
        caps.add("preemption")
        if "DECODE_ENGINE_OWNED" in preemption.get("allowedVictimStages", []):
            caps.add("engine_cancellation")
    return axes, sorted(caps)


@dataclass(frozen=True)
class MasterPatch:
    """Validated configuration operations; absent/None inherits, OMIT removes."""

    configuration: ConfigOverride
    declared: dict

    @classmethod
    def read(cls, values, path):
        values = mapping(values, path, MASTER_PATCH_FIELDS)
        kwargs = {}
        for key, val in values.items():
            field = path + "." + key
            if val is None:
                continue
            if key in {"ordering", "decision", "dispatcher"}:
                choices = {
                    "ordering": ("fifo", "priority"),
                    "decision": ("single", "fixed_window"),
                    "dispatcher": ("batch", "non_batch"),
                }[key]
                if val not in choices:
                    fail(field, f"expected one of {choices}")
            elif key == "cache_affinity_max_extra_ttft_ms":
                number(val, field, minimum=0, integer=True)
            elif key == "cache_affinity_min_prefix_hit_percent":
                number(val, field, minimum=0)
            elif key == "prefill_expression":
                if not isinstance(val, str) or not val.strip() or len(val) > 4096:
                    fail(field, "expected a nonempty formula of at most 4096 characters")
            elif key == "preemption" and val != {"omit": True}:
                mapping(
                    val,
                    field,
                    {"allowed_victim_stages", "timeout_ms"},
                    {"allowed_victim_stages"},
                )
                victim_stages = names(
                    val["allowed_victim_stages"],
                    field + ".allowed_victim_stages",
                    VICTIM_STAGES,
                )
                if not victim_stages:
                    fail(field, "preemption victim stages cannot be empty")
                if "timeout_ms" in val:
                    number(
                        val["timeout_ms"], field + ".timeout_ms", minimum=1, integer=True
                    )
            elif key == "decision_lifetime":
                number(val, field, minimum=1)
            elif isinstance(val, dict):
                if val != {"omit": True} or type(val.get("omit")) is not bool:
                    fail(field, "expected exact {omit: true}")
                val = OMIT
            else:
                number(val, field, integer=True)
            kwargs[key] = val
        return cls(ConfigOverride(**kwargs), copy.deepcopy(values))

    def overlay(self, other):
        return MasterPatch.read(
            {**self.declared, **{key: value for key, value in other.declared.items() if value is not None}},
            "config_overrides",
        )


@dataclass(frozen=True)
class EnvironmentResolver:
    """One preset load and one patch parse per environment declaration."""

    performance: dict
    runtime: dict
    common: MasterPatch
    profiles: dict[str, MasterPatch]
    captured: dict[str, int]

    @classmethod
    def read(cls, value, path, preset):
        bundle = load_preset_bundle(preset)
        performance, runtime = bundle.performance, bundle.runtime
        common = MasterPatch.read(value.get("config_overrides", {}), path + ".config_overrides")
        patches = mapping(value.get("profile_overrides", {}), path + ".profile_overrides", PROFILES)
        profiles = {profile: MasterPatch.read(patch, path + ".profile_overrides." + profile)
                    for profile, patch in patches.items()}
        paired = runtime.get("paired_master")
        # Check excluded profiles too; this never recursively renders an environment.
        for profile, patch in profiles.items():
            cls._identity(profile, patch.declared, path)
            cls._identity(profile, common.declared, path)
            baseline = MasterPatch.read(paired.for_profile(profile) if paired else {}, path + ".preset")
            baseline.overlay(common).overlay(patch)
        return cls(performance, runtime, common, profiles, bundle.captured)

    @staticmethod
    def _identity(profile, patch, path):
        try:
            validate_profile_identity(profile, patch)
        except ProfileIdentityError as exc:
            raise ScenarioError(f"{path}: {exc}") from exc

    def master(self, profile, path):
        paired = self.runtime.get("paired_master")
        baseline = MasterPatch.read(paired.for_profile(profile) if paired else {}, path + ".preset")
        patch = self.common.overlay(self.profiles.get(profile, MasterPatch.read({}, path)))
        for candidate in (baseline, patch):
            self._identity(profile, candidate.declared, path)
        resolved = baseline.overlay(patch)
        try:
            config = json.loads(render_env(profile, resolved.configuration))
        except (TypeError, ValueError) as exc:
            fail(path + ".config_overrides", str(exc))
        return baseline, patch, resolved, config

    def resolve(self, declaration, path, profile):
        baseline, declared, resolved, config = self.master(profile, path)
        paired_master, declared_overrides = baseline.declared, declared.declared
        captured = self.captured
        if captured:
            changed = [field for field, source_field in (
                ("n_prefill", "n_prefill"), ("n_decode", "n_decode"),
                ("prefill_cache_blocks", "prefill_kv_pool_blocks"),
                ("decode_cache_blocks", "decode_kv_pool_blocks"),
            ) if getattr(declaration, field) is not None and getattr(declaration, field) != captured[source_field]]
            if ("prefill_expression" in declared_overrides and declared_overrides["prefill_expression"] != paired_master.get("prefill_expression")) or declaration.prefill_perf is not None:
                changed.append("prefill model")
            changed.extend("master." + key for key, val in declared_overrides.items()
                           if key in paired_master and val != paired_master[key])
            if declaration.prefill_cache_policy is not None:
                performance = self.performance
                policy = declaration.prefill_cache_policy
                original = performance.get("prefill", {})
                memory = original.get("memory_cache", {})
                if (policy.device_tree != original.get("enable_gpu_prefix_tree")
                    or policy.memory_tree != memory.get("enable_prefix_tree")
                    or (policy.memory_blocks is not None and policy.memory_blocks != memory.get("capacity_blocks"))):
                    changed.append("prefill_cache_policy")
            if changed and declaration.model_override is None:
                fail(path + ".model_override", "test deviation requires baseline and reason: " + ", ".join(changed))
        axes, capabilities = effective_capabilities(config)
        from scenario.environment_snapshot import EnvironmentSnapshot
        snapshot = EnvironmentSnapshot.freeze(
            config, self.effective_performance(declaration),
            {key: value for key, value in self.runtime.items() if key != "paired_master"},
        )
        return ResolvedEnvironment(declaration, resolved, snapshot, axes, tuple(capabilities))

    def effective_performance(self, declaration):
        return declaration.apply_performance(self.performance)


@dataclass(frozen=True)
class PrefillCachePolicy:
    device_tree: bool
    memory_tree: bool
    memory_blocks: int | None = None

    def __post_init__(self):
        if type(self.device_tree) is not bool or type(self.memory_tree) is not bool:
            raise ValueError("cache prefix-tree flags must be booleans")
        if self.memory_blocks is not None and (type(self.memory_blocks) is not int or self.memory_blocks < 0):
            raise ValueError("memory cache blocks must be a nonnegative integer")

    def apply(self, prefill):
        capacity = self.memory_blocks
        if capacity is None:
            capacity = prefill.get("memory_cache", {}).get("capacity_blocks")
        if type(capacity) is not int or capacity < 0:
            raise ValueError("prefill memory cache policy requires captured capacity or explicit memory_blocks")
        prefill["enable_gpu_prefix_tree"] = self.device_tree
        prefill["memory_cache"] = dict(enabled=capacity > 0, capacity_blocks=capacity,
                                     enable_prefix_tree=self.memory_tree)


@dataclass(frozen=True)
class ModelDeviation:
    baseline: str
    reason: str


    def __post_init__(self):
        if (type(self.baseline) is not str or not self.baseline.strip()
                or type(self.reason) is not str or not self.reason.strip()):
            raise ValueError("model deviation requires baseline and reason")


@dataclass(frozen=True)
class PrefillPerformance:
    fixed_ms: float
    scale: float
    max_batch_tokens: int
    max_batch_requests: int


    def __post_init__(self):
        for item in fields(self):
            number(getattr(self, item.name), "prefill_perf." + item.name,
                   integer=item.name.startswith("max_batch_"))


@dataclass(frozen=True)
class EnvironmentDeclaration:
    backend: str
    n_prefill: int
    n_decode: int
    discovery: str
    perf_preset: str
    master_layout: str
    master_stable_window_s: float
    debug_enabled: bool = field(metadata={"scalar": (bool, False, None)})
    master_sync_log: bool = field(metadata={"scalar": (bool, False, None)})
    prefill_cache_blocks: int | None = None
    decode_cache_blocks: int | None = None
    mock_auto_fetch: bool | None = field(default=None, metadata={"scalar": (bool, _ABSENT, None)})
    master_debug_log: bool | None = field(default=None, metadata={"scalar": (bool, _ABSENT, None)})
    mock_fetch_attach_timeout_ms: int | None = field(default=None, metadata={"scalar": (int, _ABSENT, 1)})
    prefill_max_waiting_batches: int | None = field(default=None, metadata={"scalar": (int, _ABSENT, 0)})
    metric_whitelist: str | None = None
    model_override: ModelDeviation | None = None
    prefill_perf: PrefillPerformance | None = None
    prefill_cache_policy: PrefillCachePolicy | None = None

    def __post_init__(self):
        if self.backend != "java_mock" or self.discovery not in {"file", "discovery_file"}:
            raise ValueError("invalid environment backend/discovery")
        if self.master_layout not in {"single", "dual_standalone"}:
            raise ValueError("invalid environment master layout")
        for name in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
            value = getattr(self, name)
            if value is not None:
                number(value, "environment." + name, minimum=1, integer=True)
        for item in fields(self):
            rule = item.metadata.get("scalar")
            value = getattr(self, item.name)
            if rule is None or value is None:
                continue
            kind, _, minimum = rule
            if kind is bool and type(value) is not bool:
                raise ValueError("environment." + item.name + " requires a boolean")
            if kind is int:
                number(value, "environment." + item.name, minimum=minimum, integer=True)
        number(self.master_stable_window_s, "environment.master_stable_window_s")
        if self.master_sync_log and self.master_layout != "single":
            raise ValueError("isolated sync log requires single master")
        if self.model_override is not None and self.model_override.baseline != self.perf_preset:
            raise ValueError("model deviation baseline must name perf_preset")

    @classmethod
    def read_normalized(cls, value):
        options = copy.deepcopy(value)
        policy = options.pop("prefill_cache_policy", None)
        performance = options.pop("prefill_perf", None)
        deviation = options.pop("model_override", None)
        return cls(**options,
                   prefill_cache_policy=PrefillCachePolicy(**policy) if policy is not None else None,
                   prefill_perf=PrefillPerformance(**performance) if performance is not None else None,
                   model_override=ModelDeviation(**deviation) if deviation is not None else None)


    def to_dict(self):
        result = {field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)
                  if getattr(self, field.name) is not None}
        for name in ("prefill_perf", "model_override"):
            value = getattr(self, name)
            if value is not None:
                result[name] = asdict(value)
        if self.prefill_cache_policy is not None:
            result["prefill_cache_policy"] = {
                field.name: getattr(self.prefill_cache_policy, field.name)
                for field in fields(self.prefill_cache_policy)
                if getattr(self.prefill_cache_policy, field.name) is not None}
        return result

    def apply_performance(self, baseline):
        performance = copy.deepcopy(baseline)
        if self.prefill_perf is not None:
            performance["prefill"] = asdict(self.prefill_perf)
        if self.prefill_max_waiting_batches is not None:
            performance.setdefault("prefill", {})["max_waiting_batches"] = self.prefill_max_waiting_batches
        if self.prefill_cache_policy is not None:
            self.prefill_cache_policy.apply(performance.setdefault("prefill", {}))
        return performance


# Simple field rules are metadata of their typed declaration, not a second schema.
OPTIONAL_SCALARS = {item.name: item.metadata["scalar"] for item in fields(EnvironmentDeclaration)
                    if "scalar" in item.metadata}


@dataclass(frozen=True)
class ResolvedEnvironment:
    declaration: EnvironmentDeclaration
    master_patch: MasterPatch
    snapshot: "EnvironmentSnapshot"
    effective_axes: dict[str, str]
    effective_capabilities: tuple[str, ...]

    def to_dict(self):
        return dict(self.declaration.to_dict(), config_overrides=copy.deepcopy(self.master_patch.declared),
                    resolved_config=json.loads(self.snapshot.master_json), rendered=self.snapshot.to_dict(),
                    effective_axes=dict(self.effective_axes), effective_capabilities=list(self.effective_capabilities))


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
        ("master_layout", "single", ("single", "dual_standalone")),
    ):
        val = value.get(key, default)
        if val not in allowed:
            fail(path + "." + key, f"invalid value {val!r}; expected one of {allowed}")
        result[key] = val
    resolver = EnvironmentResolver.read(value, path, result["perf_preset"])
    captured = resolver.captured
    for field, capture_field in (
        ("n_prefill", "n_prefill"), ("n_decode", "n_decode"),
        ("prefill_cache_blocks", "prefill_kv_pool_blocks"),
        ("decode_cache_blocks", "decode_kv_pool_blocks"),
    ):
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
    for key in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
        if key in value:
            result[key] = number(value[key], path + "." + key, minimum=1, integer=True)
    return EnvironmentDeclaration.read_normalized(result), resolver


def environment(value, path, profile):
    declaration, resolver = read_environment(value, path)
    return resolver.resolve(declaration, path, profile).to_dict()

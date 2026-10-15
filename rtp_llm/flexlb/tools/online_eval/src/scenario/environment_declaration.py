"""Typed authored environment data and scalar field contracts."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field, fields, asdict
from adapters.zookeeper_config import ZooKeeperConfig
from scenario.validation import fail, number

_ABSENT = object()

def environment_fields():
    """One vocabulary for both top-level fields and simple scalar declarations."""
    return {field.name for field in fields(EnvironmentDeclaration)} | {"config_overrides", "profile_overrides"}


def variant_environment_fields():
    # A variant may change declared data, but never switch its case's backend.
    return environment_fields() - {"backend"}


def scalar_field(kind, default=_ABSENT, minimum=None, *, startup=False, reconfigure=False, validate=None, capture=None):
    """Declare authored admission, startup forwarding and replacement in one place."""
    return field(default=None if default is _ABSENT else default, metadata={
        "scalar": (kind, default, minimum), "startup": startup, "reconfigure": reconfigure, "validate": validate, "capture": capture})


def validate_scalar(value, kind, minimum, path):
    if kind is bool:
        if type(value) is not bool:
            fail(path, "expected boolean")
        return value
    if kind is str:
        if type(value) is not str:
            fail(path, "expected string")
        return value
    return number(value, path, minimum=minimum, integer=kind is int)


def heap_size(value, path):
    import re
    if not re.fullmatch(r"[1-9][0-9]*[mMgG]", value):
        fail(path, "requires a positive JVM heap with m/g units")


def optional_scalars(value, path, result, declarations=None):
    """Decode scalar fields from their declaration, preserving absent defaults."""
    if declarations is None:
        declarations = OPTIONAL_SCALARS
    for key, (kind, default, minimum) in declarations.items():
        if key not in value and default is _ABSENT:
            continue
        result[key] = validate_scalar(value.get(key, default), kind, minimum, path + "." + key)


def startup_options(plan):
    """Only explicitly authored startup fields cross into the backend specification."""
    return {item.name: copy.deepcopy(plan[item.name]) for item in fields(EnvironmentDeclaration)
            if item.metadata.get("startup") and item.name in plan}


def captured_fields():
    return {item.name: item.metadata["capture"] for item in fields(EnvironmentDeclaration)
            if item.metadata.get("capture")}


def reconfigurable_fields():
    return {item.name for item in fields(EnvironmentDeclaration) if item.metadata.get("reconfigure")}


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
    n_prefill: int = field(metadata={"startup": True, "reconfigure": True, "capture": "n_prefill"})
    n_decode: int = field(metadata={"startup": True, "reconfigure": True, "capture": "n_decode"})
    discovery: str
    perf_preset: str
    master_layout: str
    master_stable_window_s: float
    debug_enabled: bool = field(metadata={"scalar": (bool, False, None)})
    master_sync_log: bool = field(metadata={"scalar": (bool, False, None)})
    prefill_cache_blocks: int | None = scalar_field(int, minimum=1, startup=True, capture="prefill_kv_pool_blocks")
    decode_cache_blocks: int | None = scalar_field(int, minimum=1, startup=True, capture="decode_kv_pool_blocks")
    mock_auto_fetch: bool | None = scalar_field(bool, minimum=None, startup=True, reconfigure=True)
    master_debug_log: bool | None = scalar_field(bool, minimum=None, startup=True, reconfigure=False)
    mock_fetch_attach_timeout_ms: int | None = scalar_field(int, minimum=1, startup=True, reconfigure=True)
    prefill_max_waiting_batches: int | None = scalar_field(int, minimum=0, startup=False, reconfigure=False)
    master_test_control: bool | None = scalar_field(bool, minimum=None, startup=False, reconfigure=False)
    master_jvm_heap: str | None = scalar_field(str, startup=True, validate=heap_size)
    mock_heap: str | None = scalar_field(str, startup=True, validate=heap_size)
    event_loop_threads: int | None = scalar_field(int, minimum=1, startup=True, reconfigure=True)
    rpc_threads: int | None = scalar_field(int, minimum=1, startup=True, reconfigure=True)
    rpc_max_calls: int | None = scalar_field(int, minimum=1, startup=True, reconfigure=True)
    metric_whitelist: str | None = field(default=None, metadata={"reconfigure": True})
    zookeeper: ZooKeeperConfig | None = None
    model_override: ModelDeviation | None = None
    prefill_perf: PrefillPerformance | None = None
    prefill_cache_policy: PrefillCachePolicy | None = None

    def __post_init__(self):
        if self.backend != "java_mock" or self.discovery not in {"file", "discovery_file"}:
            raise ValueError("invalid environment backend/discovery")
        if self.master_layout not in {"single", "dual_standalone", "dual_zookeeper"}:
            raise ValueError("invalid environment master layout")
        if (self.master_layout == 'dual_zookeeper') != (self.zookeeper is not None):
            raise ValueError('zookeeper configuration requires dual_zookeeper layout')
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
            validate_scalar(value, kind, minimum, "environment." + item.name)
            if item.metadata.get("validate") is not None:
                item.metadata["validate"](value, "environment." + item.name)
        number(self.master_stable_window_s, "environment.master_stable_window_s")
        if self.master_sync_log and self.master_layout != "single":
            raise ValueError("isolated sync log requires single master")
        if self.model_override is not None and self.model_override.baseline != self.perf_preset:
            raise ValueError("model deviation baseline must name perf_preset")

    @classmethod
    def read_normalized(cls, value):
        options = copy.deepcopy(value)
        zk = options.pop("zookeeper", None)
        policy = options.pop("prefill_cache_policy", None)
        performance = options.pop("prefill_perf", None)
        deviation = options.pop("model_override", None)
        return cls(**options, zookeeper=ZooKeeperConfig.read(zk) if zk is not None else None,
                   prefill_cache_policy=PrefillCachePolicy(**policy) if policy is not None else None,
                   prefill_perf=PrefillPerformance(**performance) if performance is not None else None,
                   model_override=ModelDeviation(**deviation) if deviation is not None else None)


    def to_dict(self):
        result = {field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)
                  if getattr(self, field.name) is not None}
        for name in ("prefill_perf", "model_override", "zookeeper"):
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
            if performance.get("prefill", {}).get("fifo") is not None:
                raise ValueError("prefill_perf legacy batch limits cannot override a FIFO preset")
            performance.setdefault("prefill", {}).update(asdict(self.prefill_perf))
        if self.prefill_max_waiting_batches is not None:
            performance.setdefault("prefill", {})["max_waiting_batches"] = self.prefill_max_waiting_batches
        if self.prefill_cache_policy is not None:
            self.prefill_cache_policy.apply(performance.setdefault("prefill", {}))
        return performance


# Simple field rules are metadata of their typed declaration, not a second schema.
OPTIONAL_SCALARS = {item.name: item.metadata["scalar"] for item in fields(EnvironmentDeclaration)
                    if "scalar" in item.metadata}

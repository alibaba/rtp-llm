"""Validated Master configuration operations and atomic overlay semantics."""
import copy
from dataclasses import dataclass, fields
from flexlb_cfg import ConfigOverride, OMIT, VICTIM_STAGES
from scenario.validation import fail, mapping, number, names

MASTER_PATCH_FIELDS = {field.name for field in fields(ConfigOverride)} - {"strip_preemption"}

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

"""Serializable selection configuration; forward paths never read environment."""

import json
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModuleDispatchConfig:
    mode: str = "legacy"
    platform: str = "auto"
    impl_overrides: Tuple[Tuple[str, str], ...] = ()
    path_overrides: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.mode not in ("legacy", "auto"):
            raise ValueError(f"Unknown module dispatch mode {self.mode!r}")
        from rtp_llm.device.runtime import validate_requested_device

        validate_requested_device(self.platform)
        for field in ("impl_overrides", "path_overrides"):
            entries = tuple(getattr(self, field))
            normalized = dict(entries)
            if len(entries) != len(normalized):
                raise ValueError(f"Duplicate keys in {field}")
            if any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in normalized.items()
            ):
                raise ValueError(f"{field} requires nonempty string keys and values")
            object.__setattr__(self, field, tuple(sorted(normalized.items())))
        if self.mode == "legacy" and (self.impl_overrides or self.path_overrides):
            raise ValueError(
                "Implementation overrides require module_dispatch.mode=auto"
            )

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("module_dispatch must be an object")
        allowed = {"mode", "platform", "impl_overrides", "path_overrides"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown module_dispatch fields: {sorted(unknown)}")
        values = dict(data)
        for field in ("impl_overrides", "path_overrides"):
            if field in values:
                if not isinstance(values[field], dict):
                    raise TypeError(f"{field} must be an object")
                values[field] = tuple(values[field].items())
        return cls(**values)

    def to_string(self):
        return json.dumps(
            {
                "mode": self.mode,
                "platform": self.platform,
                "impl_overrides": dict(self.impl_overrides),
                "path_overrides": dict(self.path_overrides),
            },
            sort_keys=True,
        )

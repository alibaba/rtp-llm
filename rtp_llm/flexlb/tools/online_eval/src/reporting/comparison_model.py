"""Typed frozen bundle inputs and stable measurement identities for comparison."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from reporting.report_model import ExperimentConditions
from reporting.spec import validate


@dataclass(frozen=True)
class FrozenReport:
    title: Optional[str]
    subtitle: Optional[str]
    time_origin_label: Optional[str]
    events: tuple[dict, ...]
    panels: tuple[dict, ...]
    sections: tuple[dict, ...]
    kpis: tuple[dict, ...]
    time_min: Optional[float]
    time_max: Optional[float]
    run_meta: dict
    controls: ExperimentConditions

    @classmethod
    def read(cls, spec):
        validate(spec)
        metadata = copy.deepcopy(spec["run_meta"])
        if not isinstance(metadata.get("comparison_controls"), dict):
            raise ValueError("frozen report lacks comparison controls")
        axis = spec.get("timeAxis") or {}
        return cls(
            spec.get("title"),
            spec.get("subtitle"),
            spec.get("timeOriginLabel"),
            tuple(copy.deepcopy(spec.get("events", []))),
            tuple(copy.deepcopy(spec.get("panels", []))),
            tuple(copy.deepcopy(spec.get("sections", []))),
            tuple(copy.deepcopy(spec.get("kpis", []))),
            axis.get("min"),
            axis.get("max"),
            metadata,
            ExperimentConditions.read(metadata["comparison_controls"]),
        )

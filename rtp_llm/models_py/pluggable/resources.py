"""Frozen allocation/weight preparation descriptions, without runtime imports."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .spec import canonical_json


@dataclass(frozen=True)
class ResourcePlan:
    """The model adapter describes resources; the engine and loader own them.

    Contract IDs remain compatibility checks. This plan carries the actual
    allocation geometry and the selected weight preparation entrypoint, and
    participates in the startup protocol before either resource is consumed.
    """

    allocation_json: str
    weight_preparation: str | None = None

    def __post_init__(self):
        allocation = json.loads(self.allocation_json)
        if not isinstance(allocation, dict):
            raise TypeError("Resource allocation plan must be an object")
        object.__setattr__(self, "allocation_json", canonical_json(allocation))

    def record(self):
        return {
            "allocation": json.loads(self.allocation_json),
            "weight_preparation": self.weight_preparation,
        }

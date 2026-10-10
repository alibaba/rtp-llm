"""Compile data-only observation boundaries into typed program references."""

import math
from dataclasses import dataclass

from cases.config import output
from cases.inputs import fields
from scenario.loader import ScenarioError


@dataclass(frozen=True)
class ObservationWindow:
    boundaries: dict

    @classmethod
    def read(cls, data, path, *, timestamp_stages):
        fields(data, (), path, optional={"from", "until"})
        params = {}
        for side, boundary in data.items():
            loc = f"{path}.{side}"
            fields(boundary, {"stage", "field"}, loc, optional={"offset_s"})
            if (type(boundary["stage"]) is not str or boundary["stage"] not in timestamp_stages
                    or boundary["field"] != "epoch_s"):
                raise ScenarioError(f"{loc}: unknown timestamp output")
            params[side] = output(boundary["stage"], boundary["field"])
            if "offset_s" in boundary:
                offset = boundary["offset_s"]
                if type(offset) not in (int, float) or not math.isfinite(offset):
                    raise ScenarioError(f"{loc}: offset_s must be finite")
                params[side + "_offset_s"] = offset
        return cls(params)


def anchored_window(data, path, *, anchor):
    """A bounded window around one runtime event, with explicit signed offsets."""
    fields(data, {"from", "until"}, path)
    offsets = {}
    for side, boundary in data.items():
        fields(boundary, {"event", "offset_s"}, path + "." + side)
        offset = boundary["offset_s"]
        if (boundary["event"] != anchor or type(offset) not in (int, float)
                or not math.isfinite(offset)):
            raise ScenarioError(f"{path}.{side}: requires {anchor} and finite offset_s")
        offsets[side] = offset
    if offsets["from"] >= offsets["until"]:
        raise ScenarioError(path + ": observation window is empty or inverted")
    return offsets



def anchored_windows(data, anchors, *, path="parameters.observation.windows"):
    """Validate named windows against their program-owned runtime event anchors."""
    fields(data, set(anchors), path)
    return {name: anchored_window(data[name], path + '.' + name, anchor=anchor)
            for name, anchor in anchors.items()}


def resolve_window(lower, upper, *, lower_offset_s=0, upper_offset_s=0):
    """Resolve seconds without changing cohort membership or boundary inclusion."""
    lo = None if lower is None else lower + lower_offset_s
    hi = None if upper is None else upper + upper_offset_s
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError("observation window is empty or inverted")
    return lo, hi

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

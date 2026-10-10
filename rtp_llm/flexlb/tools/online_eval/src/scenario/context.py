"""Execution context binds a compiled instance to owned resources and stage outputs."""

import time
from pathlib import Path
from scenario.resources import ResourceScope


class RuntimeContext(ResourceScope):
    def __init__(
        self, instance, backend, artifact_dir, clock, sleeper, enforce_deadlines=False,
        wall_clock=None,
    ):
        self.instance = instance
        self.backend = backend
        self.artifact_dir = Path(artifact_dir)
        self.wall_clock = wall_clock if wall_clock is not None else lambda: time.time()
        super().__init__(clock, sleeper, enforce_deadlines)
        self.env = self.ops = None
        self.outputs = {}
        self.report_events = []

    def record_event(self, identity, *, timestamp=None):
        """Record the actual case event time; presentation names belong to views."""
        import re
        if type(identity) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", identity):
            raise ValueError("invalid case event identity")
        if timestamp is None:
            timestamp = dict(epoch_s=self.wall_clock(), monotonic_s=self.clock())
        import math
        if (set(timestamp) != {"epoch_s", "monotonic_s"}
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in timestamp.values())):
            raise ValueError("event requires finite epoch_s and monotonic_s")
        event = dict(id=identity, **timestamp)
        self.report_events.append(event)
        return dict(event)

    def resolve(self, value):
        if isinstance(value, dict) and "$ref" in value:
            if set(value) != {"$ref"}:
                raise ValueError("reference cannot contain extra fields")
            parts = value["$ref"].split(".")
            if len(parts) != 4 or parts[0] != "stages" or parts[2] != "output":
                raise ValueError("invalid stage reference")
            return self.outputs[parts[1]][parts[3]]
        return value

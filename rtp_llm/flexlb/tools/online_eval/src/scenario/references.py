"""Output reference grammar and prior-stage type resolution."""

from dataclasses import dataclass
from scenario.validation import mapping, fail


@dataclass(frozen=True)
class OutputReference:
    stage_id: str
    field: str

    @classmethod
    def read(cls, value, path="reference"):
        mapping(value, path, {"$ref"}, {"$ref"})
        ref = value["$ref"]
        if not isinstance(ref, str):
            fail(path, "$ref must be a string")
        parts = ref.split(".")
        if len(parts) != 4 or parts[0] != "stages" or parts[2] != "output":
            fail(path, "expected stages.<earlier_stage>.output.<field>")
        return cls(parts[1], parts[3])

    def to_dict(self):
        return {"$ref": f"stages.{self.stage_id}.output.{self.field}"}


def reference(value, path, outputs, expected=None):
    ref = OutputReference.read(value, path)
    kind = outputs.get(ref.stage_id, {}).get(ref.field)
    if kind is None:
        fail(path, f"unknown or forward reference {value['$ref']!r}")
    if expected is not None and kind != expected:
        fail(path, f"reference is {kind}, expected {expected}")
    return kind

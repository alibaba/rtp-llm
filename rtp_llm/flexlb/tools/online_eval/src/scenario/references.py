"""Output reference grammar and prior-stage type resolution."""

from dataclasses import dataclass

from scenario.validation import fail, mapping


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


def validate_references(value, path, outputs):
    """Walk both compiled and persisted parameters with the same reference grammar."""
    if isinstance(value, dict):
        if "$ref" in value:
            ref = OutputReference.read(value, path)
            if ref.stage_id not in outputs:
                fail(path, "unknown or forward reference")
            if outputs[ref.stage_id] is not None:
                reference(value, path, outputs)
        else:
            for key, child in value.items():
                validate_references(child, path + "." + str(key), outputs)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_references(child, f"{path}[{index}]", outputs)


def enter_stage(action, index, setup_seen, torn_down, path):
    """Validate lifecycle at both compilation and frozen-file admission."""
    if torn_down:
        fail(path, "no stage may follow teardown")
    if action == "setup" and (setup_seen or index != 0):
        fail(path, "setup must occur exactly once as the first stage")
    if action != "setup" and not setup_seen:
        fail(path, "setup must precede actions")
    return setup_seen or action == "setup"

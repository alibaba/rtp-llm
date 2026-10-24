"""Build typed programs from compact, mutable unit-test inputs.

These helpers only assemble fixtures. Production builders and compilers never read
this shape, and there is no versioned program document or file reader.
"""

import copy
from dataclasses import asdict, fields

from document_io import DocumentError
from scenario.compiler import ScenarioCompiler as Compiler
from scenario.program import ProgramPlan, StageDefinition, VariantPlan


def stage(value):
    return value if isinstance(value, StageDefinition) else StageDefinition(**copy.deepcopy(value))


def program(value, source="fixture"):
    if isinstance(value, ProgramPlan):
        return value
    value = copy.deepcopy(value)
    allowed = {f.name for f in fields(ProgramPlan)} | {"stages"}
    if set(value) - allowed:
        raise DocumentError(source + ": unknown fixture fields")
    variants = value.pop("variants", [{"id": "default"}])
    base = value.pop("stages", None)
    compiled = []
    for variant in variants:
        variant = dict(variant)
        if "stages" in variant and "stage_overrides" in variant:
            raise DocumentError(source + ": explicit stages and stage_overrides are mutually exclusive")
        steps = copy.deepcopy(variant.pop("stages", base))
        if steps is None:
            raise DocumentError(source + ": expected stage list")
        overrides = variant.pop("stage_overrides", {})
        if set(overrides) - {s["id"] for s in steps}:
            raise DocumentError(source + ": unknown fixture stage override")
        for step in steps:
            if step["id"] in overrides:
                step["params"] = {**step.get("params", {}), **overrides[step["id"]]}
        for key in ("profiles", "requires", "findings"):
            if key in variant:
                variant[key] = tuple(variant[key])
        compiled.append(VariantPlan(stages=tuple(stage(s) for s in steps), **variant))
    for key in ("profiles", "requires", "tags", "findings"):
        if key in value and isinstance(value[key], list):
            value[key] = tuple(value[key])
    return ProgramPlan(variants=tuple(compiled), **value)


class ScenarioCompiler(Compiler):
    def __init__(self, handlers=None):
        if handlers is None:
            from scenario.catalog import handlers as catalog

            handlers = catalog()
        super().__init__(handlers)

    def compile(self, documents, profile=None, grade="normal"):
        return super().compile([(source, program(value, source)) for source, value in documents], profile, grade)


def compile_scenarios(documents, profile=None, handlers=None, grade="normal"):
    return [instance.to_dict() for instance in ScenarioCompiler(handlers).compile(documents, profile, grade)]


def stages(values, *args, **kwargs):
    from scenario.stage_compiler import stages as compile_stages

    return compile_stages(tuple(stage(value) for value in values), *args, **kwargs)


def mutable_program(value):
    """Editable input for tests that deliberately corrupt a built program."""
    result = {
        key: copy.deepcopy(getattr(value, key))
        for key in ("id", "description", "category", "environment", "implementation", "execution")
    }
    result["profiles"] = list(value.profiles)
    result["stages"] = [
        {key: child for key, child in asdict(step).items() if child is not None} for step in value.variants[0].stages
    ]
    return result

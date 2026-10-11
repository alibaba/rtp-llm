"""Freeze compiled execution and verify code/presentation before child admission."""

import hashlib
import json
from pathlib import Path

from artifacts.json_io import write_json
from scenario.loader import ScenarioError
from schema_contract import matches_schema
from scenario.plan import CompiledInstance, ExecutionPlan

ROOT = Path(__file__).resolve().parents[2]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def dependency_files():
    """Source code and runtime config are inputs; scene YAML is already compiled."""
    paths = list((ROOT / "src").rglob("*.py")) + list(ROOT.glob("*.py"))
    paths += list((ROOT / "scripts").rglob("*.py"))
    paths += [path for path in (ROOT / "src/reporting/assets").rglob("*") if path.is_file()]
    paths += [path for path in (ROOT / "config").rglob("*") if path.is_file()
              and "scenarios" not in path.relative_to(ROOT / "config").parts
              and path.suffix in {".yaml", ".json", ".bzl"}]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


class ExecutionPlanCodec:
    """Frozen artifact codec: validate envelope, dependencies and domain contracts."""

    def __init__(self, dependencies=None):
        self.dependencies = dependency_files if dependencies is None else dependencies

    def write(self, path, plan):
        if dict(plan.dependencies) != self.dependencies():
            raise ScenarioError("execution code/config changed after compilation")
        payload = dict(execution_plan_schema_version=1,
                       instances=[instance.to_dict() for instance in plan.instances],
                       dependencies=dict(plan.dependencies))
        payload["sha256"] = _sha(payload)
        write_json(path, payload)
        return payload["sha256"]

    def read(self, path, expected_sha):
        document = json.loads(Path(path).read_text())
        if not matches_schema(document, "execution_plan_schema_version", 1):
            raise ScenarioError("unsupported execution plan schema")
        if set(document) != {"execution_plan_schema_version", "instances", "dependencies", "sha256"}:
            raise ScenarioError("invalid execution plan fields")
        actual = _sha({key: value for key, value in document.items() if key != "sha256"})
        if document["sha256"] != actual or actual != expected_sha:
            raise ScenarioError("execution plan checksum mismatch")
        if document["dependencies"] != self.dependencies():
            raise ScenarioError("execution code/config changed after planning")
        instances = document["instances"]
        if not isinstance(instances, list):
            raise ScenarioError("execution plan requires compiled instances")
        return ExecutionPlan(tuple(CompiledInstance.read(instance) for instance in instances),
                             tuple(sorted(document["dependencies"].items())))


def freeze_plan(path, plans, *, dependencies=None):
    """Artifact-facing projection; internal callers can write ExecutionPlan directly."""
    dependencies = dependency_files() if dependencies is None else dependencies
    plan = ExecutionPlan(tuple(CompiledInstance.read(instance) for instance in plans),
                         tuple(sorted(dependencies.items())))
    return ExecutionPlanCodec().write(path, plan)


def load_plan(path, expected_sha):
    return [instance.to_dict() for instance in ExecutionPlanCodec().read(path, expected_sha).instances]

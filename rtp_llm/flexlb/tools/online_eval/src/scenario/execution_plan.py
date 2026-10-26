"""Freeze compiled execution and verify code/presentation before child admission."""

import hashlib
import json
from pathlib import Path

from document_io import DocumentError as ScenarioError
from file_io import read_json, write_json
from scenario.plan import CompiledInstance
from schema_contract import matches_schema

ROOT = Path(__file__).resolve().parents[2]


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected_cases(instances):
    names = set()
    for instance in instances:
        value = instance.to_dict() if isinstance(instance, CompiledInstance) else instance
        if not isinstance(value, dict) or not isinstance(value.get("implementation", {}), dict):
            raise ScenarioError("invalid frozen case declaration")
        configuration = value.get("implementation", {}).get("configuration", {})
        if not isinstance(configuration, dict):
            raise ScenarioError("invalid frozen case configuration")
        name = configuration.get("case")
        if name is None:
            return None  # Internal contract fixtures have no registered case.
        from monitoring.identity import NAME

        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ScenarioError("invalid frozen case identity")
        names.add(name)
    return frozenset(names)


def scope_dependencies(dependencies, instances):
    names = selected_cases(instances)
    if names is None:
        return dict(dependencies)
    views = set()
    for instance in instances:
        value = instance.to_dict() if isinstance(instance, CompiledInstance) else instance
        if value.get("metadata", {}).get("kind") == "workload":
            views.update({"default.yaml", *value.get("reports", [])})
    return {
        path: digest
        for path, digest in dependencies.items()
        if (not path.startswith("src/cases/") or path.split("/")[2] in names or path == "src/cases/__init__.py")
        and (not path.startswith("config/report_views/") or path.split("/")[-1] in views)
        and (not path.startswith("src/reporting/assets/") or views)
    }


def dependency_files(instances=None):
    """Source code and runtime config are inputs; scene YAML is already compiled."""
    paths = list((ROOT / "src").rglob("*.py")) + list(ROOT.glob("*.py"))
    paths += list((ROOT / "scripts").rglob("*.py"))
    paths += [path for path in (ROOT / "src/reporting/assets").rglob("*") if path.is_file()]
    paths += [
        path
        for path in (ROOT / "config").rglob("*")
        if path.is_file() and "scenarios" not in path.relative_to(ROOT / "config").parts and path.suffix in {".yaml", ".json", ".bzl"}
    ]
    selected = {str(path.relative_to(ROOT)): path for path in sorted(paths)}
    if instances is not None:
        selected = scope_dependencies(selected, instances)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in selected.items()}


def _instances(values):
    instances = tuple(CompiledInstance.read(value) for value in values)
    if not instances or len({instance.id for instance in instances}) != len(instances):
        raise ScenarioError("execution plan requires unique compiled instances")
    return instances


class ExecutionPlanCodec:
    """Frozen artifact codec: validate envelope, dependencies and domain contracts."""

    def __init__(self, dependencies=None):
        self.dependencies = dependencies

    def current_dependencies(self, instances):
        return dependency_files(instances) if self.dependencies is None else self.dependencies()

    def write(self, path, instances, dependencies):
        instances = _instances(instances)
        if dependencies != self.current_dependencies(instances):
            raise ScenarioError("execution code/config changed after compilation")
        payload = dict(
            execution_plan_schema_version=1,
            instances=[instance.to_dict() for instance in instances],
            dependencies=dict(sorted(dependencies.items())),
        )
        payload["sha256"] = _sha(payload)
        write_json(path, payload)
        return payload["sha256"]

    def read_document(self, path, expected_sha):
        document = read_json(Path(path))
        if not matches_schema(document, "execution_plan_schema_version", 1):
            raise ScenarioError("unsupported execution plan schema")
        if set(document) != {"execution_plan_schema_version", "instances", "dependencies", "sha256"}:
            raise ScenarioError("invalid execution plan fields")
        actual = _sha({key: value for key, value in document.items() if key != "sha256"})
        if document["sha256"] != actual or actual != expected_sha:
            raise ScenarioError("execution plan checksum mismatch")
        instances = document["instances"]
        if not isinstance(instances, list):
            raise ScenarioError("execution plan requires compiled instances")
        self.verify_dependencies(document)
        return document

    def verify_dependencies(self, document):
        if document["dependencies"] != self.current_dependencies(document["instances"]):
            raise ScenarioError("execution code/config changed after planning")

    def read(self, path, expected_sha):
        return self.decode_document(self.read_document(path, expected_sha))

    @staticmethod
    def decode_document(document):
        """Decode an already verified envelope after selected case registration."""
        return _instances(document["instances"])


def freeze_plan(path, plans, *, dependencies=None):
    """Write compiled instances and dependency checksums to the execution file."""
    plans = tuple(plans)
    dependencies = dependency_files(plans) if dependencies is None else scope_dependencies(dependencies, plans)
    return ExecutionPlanCodec().write(path, plans, dependencies)


def load_plan(path, expected_sha):
    return [instance.to_dict() for instance in ExecutionPlanCodec().read(path, expected_sha)]

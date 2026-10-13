"""Temporary scalar overrides applied before the existing case compiler."""
import copy
import json
from dataclasses import dataclass

from scenario.errors import ScenarioError
from scenario.document_io import strict_pairs, validate_data_tree


@dataclass(frozen=True)
class ExperimentParams:
    values: dict[str, dict[str, object]]

    @classmethod
    def read(cls, raw=None):
        try:
            values = json.loads(raw, object_pairs_hook=strict_pairs) if raw else {}
            validate_data_tree(values)
        except (ValueError, RecursionError) as exc:
            raise ScenarioError("experiment_params must be a JSON object: " + str(exc)) from exc
        if not isinstance(values, dict):
            raise ScenarioError("experiment_params must map case names to parameter paths")
        for case, patches in values.items():
            if not case or not isinstance(patches, dict) or not patches:
                raise ScenarioError("experiment_params requires nonempty case patches")
            paths = list(patches)
            for path, value in patches.items():
                parts = path.split('.')
                if len(parts) < 2 or parts[0] not in ('parameters', 'environment', 'execution') or not all(parts):
                    raise ScenarioError("unsupported experiment parameter path: " + path)
                if type(value) not in (str, int, float, bool) and value is not None:
                    raise ScenarioError("experiment parameters replace scalar fields only: " + path)
                if any(other.startswith(path + '.') for other in paths):
                    raise ScenarioError("overlapping experiment parameter paths: " + path)
        return cls(values)

    def apply(self, document):
        result = copy.deepcopy(document)
        for path, value in self.values.get(document.get('case'), {}).items():
            parts = path.split('.')
            parent = result
            for part in parts[:-1]:
                if not isinstance(parent, dict) or part not in parent:
                    raise ScenarioError("unknown experiment parameter: " + path)
                parent = parent[part]
            if not isinstance(parent, dict) or parts[-1] not in parent:
                raise ScenarioError("unknown experiment parameter: " + path)
            original = parent[parts[-1]]
            if isinstance(original, (dict, list)):
                raise ScenarioError("experiment parameters replace scalar fields only: " + path)
            numeric = type(original) in (int, float) and type(value) in (int, float)
            if not numeric and type(original) is not type(value):
                raise ScenarioError("experiment parameter type differs from YAML: " + path)
            parent[parts[-1]] = value
        return result

    def require_cases(self, cases):
        missing = set(self.values) - set(cases)
        if missing:
            raise ScenarioError("experiment parameters target unselected or unknown cases: " + ', '.join(sorted(missing)))

    def for_cases(self, cases):
        return ExperimentParams({case: patches for case, patches in self.values.items() if case in cases})

    def to_json(self):
        return json.dumps(self.values, sort_keys=True, allow_nan=False)

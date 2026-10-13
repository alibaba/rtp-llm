"""Load case data and invoke registered Python builders."""

from pathlib import Path
from scenario.document_io import load_document
from scenario.errors import ScenarioError


def load_scenarios(root, *, experiment_params=None):
    """Load data-only configurations and invoke their registered Python builders."""
    from scenario.program_compiler import configure_program

    from scenario.experiment_params import ExperimentParams
    overrides = ExperimentParams.read(experiment_params)
    root = Path(root).resolve()
    if root.is_file():
        raw = load_document(root)
        overrides.require_cases([raw.get('case')])
        return [(str(root), configure_program(overrides.apply(raw), str(root)))]
    if not root.is_dir():
        raise ScenarioError(f"scenario root does not exist: {root}")
    result = []
    cases = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".json", ".yaml", ".yml") or not path.is_file():
            continue
        if root not in path.resolve().parents:
            raise ScenarioError(f"{path}: scenario path escapes root")
        raw = load_document(path)
        cases.append(raw.get("case"))
        result.append((str(path), configure_program(overrides.apply(raw), str(path))))
    if not result:
        raise ScenarioError(f"no scenario definitions in {root}")
    overrides.require_cases(cases)
    return result

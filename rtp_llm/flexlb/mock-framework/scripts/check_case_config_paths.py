"""Check role configuration paths in the snapshot that Git will commit."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mock_framework.role_config_path import resolve_role_config_path

CASES = "rtp_llm/flexlb/mock-framework/cases"


def check_case(case_file):
    case_file = resolve_role_config_path(case_file.name, case_file.parent, "case.yaml")
    document = yaml.safe_load(case_file.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Case YAML must be a mapping")
    environment = document.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment: expected a mapping")
    for role, component in environment.items():
        if not isinstance(component, dict):
            raise ValueError(f"environment.{role}: expected a mapping")
        config = component.get("config")
        if isinstance(config, str):
            field = f"environment.{role}.config"
            source = resolve_role_config_path(config, case_file.parent, field)
            if not source.is_file():
                raise ValueError(f"{field}: {config!r} is not a staged file")


def main():
    repo = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    staged_files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", CASES], cwd=repo
    )
    errors = []
    # The snapshot also excludes untracked files that could mask a staged deletion.
    with TemporaryDirectory(prefix="mock-case-config-") as snapshot:
        root = Path(snapshot)
        subprocess.run(
            ["git", "checkout-index", "--stdin", "-z", f"--prefix={root}/"],
            input=staged_files,
            cwd=repo,
            check=True,
        )
        for case_file in sorted((root / CASES).rglob("case.yaml")):
            try:
                check_case(case_file)
            except (ValueError, OSError, yaml.YAMLError) as error:
                errors.append(f"{case_file.relative_to(root)}: {error}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Enforce syntax/name checks and architectural contracts before execution."""

import ast
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    # Import-linter's package graph excludes standalone modules. Inspect every
    # root module too, including imports nested in functions.
    packages = {path.name for path in (root / "src").iterdir() if path.is_dir()}
    for path in sorted(root.glob("*.py")) + sorted((root / "src").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(name.split(".")[0] in packages for name in names):
                print(f"{path}:{node.lineno}: shared leaf must not import a business package", file=sys.stderr)
                return 1
    commands = (
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy"],
        [sys.executable, "-c", "from importlinter.cli import lint_imports_command; lint_imports_command()", "--config", ".importlinter"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=root, env=environment)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

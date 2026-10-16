#!/usr/bin/env python3
"""Enforce syntax/name checks and architectural contracts before execution."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join((str(root / 'src'), str(root)))
    commands = (
        [sys.executable, '-m', 'ruff', 'check', 'src', 'scripts', 'tests'],
        [sys.executable, '-c', 'from importlinter.cli import lint_imports_command; lint_imports_command()',
         '--config', '.importlinter'],
    )
    for command in commands:
        result = subprocess.run(command, cwd=root, env=environment)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

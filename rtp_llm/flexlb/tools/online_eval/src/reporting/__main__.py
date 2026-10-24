"""Offline rendering: python -m reporting REPORT_SPEC --out PAGE."""
from file_io import read_json

import argparse
from pathlib import Path
from reporting import render
from reporting.spec import validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("spec", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
spec = read_json(a.spec)
try:
    validate(spec, versioned=True)
except ValueError as exc:
    p.error(str(exc))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(render(spec), encoding="utf-8")

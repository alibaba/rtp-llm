"""Offline rendering: python -m online_eval.reporting REPORT_SPEC --out PAGE."""

import argparse
import json
from pathlib import Path
from . import render

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("spec", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
spec = json.loads(a.spec.read_text())
if spec.get("schema_version") != 1:
    p.error("unsupported report spec version")
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(render(spec), encoding="utf-8")

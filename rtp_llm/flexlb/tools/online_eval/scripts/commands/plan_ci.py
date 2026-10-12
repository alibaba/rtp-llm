#!/usr/bin/env python3
"""Preview the same CI selection and worker assignments used for execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from scripts.pipeline.ci_selection import add_ci_selection, resolve_ci_selection
from scenario.loader import ScenarioError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_ci_selection(parser)
    parser.add_argument('--json', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plan = resolve_ci_selection(args)
    except ScenarioError as exc:
        parser.error(str(exc))
    payload = plan.to_dict(args.worker)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

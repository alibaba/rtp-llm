"""No-process scenario compiler: python -m flexlb_test_framework.scenario ROOT [--profile P]."""

import argparse
import json
import sys

from . import ScenarioError, compile_scenarios, load_scenarios
from .catalog import handlers
from .compiler import plan_counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and compile scenario instances; no services are started"
    )
    parser.add_argument("root")
    parser.add_argument("--profile")
    parser.add_argument(
        "--grade", choices=("strict", "normal", "loose"), default="normal"
    )
    args = parser.parse_args(argv)
    try:
        plans = compile_scenarios(
            load_scenarios(args.root), args.profile, handlers(), grade=args.grade
        )
        if not plans:
            raise ScenarioError("selection contains no scenario instances")
    except ScenarioError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "compile",
                "counts": plan_counts(plans),
                "instances": plans,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

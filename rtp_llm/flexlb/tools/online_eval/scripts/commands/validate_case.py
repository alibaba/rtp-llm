#!/usr/bin/env python3
"""Compile case contracts and explain inputs, checks, metrics and report bindings."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='scenario YAML file or directory')
    parser.add_argument('--profile', help='restrict the compiled instance list to one profile')
    parser.add_argument('--json', action='store_true', help='print expanded contracts as JSON')
    args = parser.parse_args(argv)
    try:
        from cases.validation import validate_cases
        report = validate_cases(args.source, profile=args.profile)
    except (ValueError, OSError) as exc:
        if args.json:
            print(json.dumps(dict(case_validation_schema_version=1, status='ERROR',
                scope='static_compilation', runtime_validation='NOT_RUN', error=str(exc)), ensure_ascii=False))
        else:
            print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        for instance in report['instances']:
            checks = sum(len(stage['check_ids']) for stage in instance['stages'])
            print(f"PASS {instance['id']} | stages={len(instance['stages'])} checks={checks} "
                  f"metrics={len(instance['metrics']['definitions'])} reports={','.join(instance['reports'])}")
        print('Static compilation passed; exporter coverage, execution and cleanup remain unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

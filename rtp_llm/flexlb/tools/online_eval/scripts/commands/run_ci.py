#!/usr/bin/env python3
"""Execute a declared CI job across its profiles, serially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from scripts.pipeline.ci_selection import add_ci_selection, resolve_ci_selection
from scenario.loader import ScenarioError
from scripts.pipeline.ci_progress import progress


def commands(plan, out_dir, *, worker=None, mock_stride=None, dry_run=False):
    identities = plan.identities(worker)
    for profile in plan.profiles:
        selected = [identity for identity in identities if identity.endswith('::' + profile)]
        if not selected:
            continue
        directory = out_dir / profile
        command = [sys.executable, str(ROOT / 'scripts/commands/run_cases.py'),
                   '--all', '--instances', ','.join(selected), '--profile', profile, '--parallel', str(plan.parallel),
                   '--grade', plan.grade, '--out-dir', str(directory), '--json', str(directory / 'aggregate.json')]
        if mock_stride is not None:
            command += ['--mock-stride', str(mock_stride)]
        if dry_run:
            command.append('--dry-run')
        yield profile, command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_ci_selection(parser)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--mock-stride', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    target = args.out_dir.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        parser.error('--out-dir must be a new or empty directory')
    try:
        plan = resolve_ci_selection(args)
    except ScenarioError as exc:
        parser.error(str(exc))
    target.mkdir(parents=True, exist_ok=True)
    (target / 'ci-plan.json').write_text(json.dumps(plan.to_dict(args.worker), indent=2) + '\n')
    results = []
    result_path = target / 'ci-results.json'
    planned = list(commands(plan, target, worker=args.worker, mock_stride=args.mock_stride, dry_run=args.dry_run))
    for profile, command in planned:
        print('[progress] starting profile=' + profile, flush=True)
        started = time.time()
        with progress(target, plan.identities(args.worker)):
            completed = subprocess.run(command, cwd=ROOT.parents[1])
        print('[progress] finished profile=' + profile + ' rc=' + str(completed.returncode), flush=True)
        results.append({'profile': profile, 'returncode': completed.returncode,
                        'started_at': started, 'ended_at': time.time(),
                        'aggregate': str(target / profile / 'aggregate.json') if not args.dry_run else None})
        payload = {'ci_results_schema_version': 1, 'job': args.job, 'suite': plan.suite,
                   'profiles': [profile for profile, _ in planned], 'parallel': plan.parallel,
                   'worker': args.worker, 'instances': list(plan.identities(args.worker)),
                   'dry_run': args.dry_run, 'results': results,
                   'complete': len(results) == len(planned)}
        temporary = result_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(result_path)
        if completed.returncode:
            return 2 if completed.returncode == 2 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

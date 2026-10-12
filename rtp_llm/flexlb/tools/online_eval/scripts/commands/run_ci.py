#!/usr/bin/env python3
"""Execute a declared CI job across its profiles, serially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from scenario.suites import SuiteCatalog


def commands(job, out_dir, *, mock_stride=None, grade='normal', dry_run=False):
    for profile in job.profiles:
        directory = out_dir / profile
        command = [sys.executable, str(ROOT / 'scripts/commands/run_cases.py'),
                   '--suite', job.suite, '--profile', profile, '--parallel', str(job.parallel),
                   '--grade', grade, '--out-dir', str(directory), '--json', str(directory / 'aggregate.json')]
        if mock_stride is not None:
            command += ['--mock-stride', str(mock_stride)]
        if dry_run:
            command.append('--dry-run')
        yield profile, command


def main(argv=None):
    catalog = SuiteCatalog.read()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=tuple(catalog.ci_jobs))
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--mock-stride', type=int)
    parser.add_argument('--grade', choices=('strict', 'normal', 'loose'), default='normal')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    target = args.out_dir.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        parser.error('--out-dir must be a new or empty directory')
    job = catalog.ci_jobs[args.job]
    # Compile every profile first so a bad declaration cannot run a partial job.
    from scenario import load_scenarios, compile_scenarios
    from scenario.catalog import handlers
    from scenario.suites import preselect_documents, classify
    from scenario.loader import ScenarioError
    try:
        documents = preselect_documents(load_scenarios(ROOT / 'config/scenarios'), job.suite)
        for profile in job.profiles:
            classify(compile_scenarios(documents, profile, handlers(), grade=args.grade), job.suite)
    except ScenarioError as exc:
        parser.error(str(exc))
    target.mkdir(parents=True, exist_ok=True)
    results = []
    result_path = target / 'ci-results.json'
    for profile, command in commands(job, target, mock_stride=args.mock_stride,
                                      grade=args.grade, dry_run=args.dry_run):
        completed = subprocess.run(command, cwd=ROOT.parents[1])
        results.append({'profile': profile, 'returncode': completed.returncode,
                        'aggregate': str(target / profile / 'aggregate.json') if not args.dry_run else None})
        payload = {'ci_results_schema_version': 1, 'job': args.job, 'suite': job.suite,
                   'profiles': list(job.profiles), 'parallel': job.parallel,
                   'dry_run': args.dry_run, 'results': results,
                   'complete': len(results) == len(job.profiles)}
        temporary = result_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(result_path)
        if completed.returncode:
            return 2 if completed.returncode == 2 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

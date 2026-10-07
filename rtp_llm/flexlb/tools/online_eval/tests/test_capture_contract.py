"""采集契约、运行位置与预算的行为验收。"""
import contextlib
import copy
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from traffic.capture_contract import BLOCK_SIZE, FIELDS, SCHEMA_VERSION, validate_row
from traffic.capture_frontend_prefix import capture, parser

ROOT = Path(__file__).resolve().parents[1]


class CaptureContractTest(unittest.TestCase):
    def fixture(self, root, count=2):
        logs = root / 'logs'
        logs.mkdir()
        row = dict(request_id='private-id', request_enter_ts_epoch_ms=2000,
                   ts_epoch_ms=2100, input_ids=[11] * 1025, status='OK')
        (logs / 'dash_sc_grpc_access_r0_s0.log').write_text(
            '\n'.join(json.dumps(dict(row, request_id=str(i))) for i in range(count)) + '\n')
        return logs

    def run_capture(self, root, logs, extra=(), clock=None):
        args = parser().parse_args(['--start', '1000', '--end', '3000', '--out', str(root/'pod-0'),
                                    '--log-dir', str(logs), *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return capture(args, **({'clock': clock} if clock else {}))

    def fit(self, root):
        return subprocess.run([sys.executable, str(ROOT/'src/traffic/fit_frontend_prefix.py'),
            '--source', str(root), '--out', str(root/'fit'), '--expected-shards', '1',
            '--output-tokens', '10'], capture_output=True, text=True)

    def test_fit_rejects_bad_rows_with_line_and_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_capture(root, self.fixture(root))
            path = root/'pod-0.jsonl.gz'
            with gzip.open(path, 'rt') as stream:
                original = [json.loads(line) for line in stream]
            self.assertEqual(set(FIELDS), set(original[0]))
            validate_row(original[0])
            cases = [('status', None, 'missing required field status'),
                     ('il', '1025', 'illegal type for il'),
                     ('il', True, 'illegal type for il'),
                     ('keys', [], 'keys/il block count mismatch'),
                     ('rid', 'private-id', 'invalid 128-bit digest'),
                     ('block_size', 1024, 'block_size must be 512')]
            for key, value, reason in cases:
                with self.subTest(key=key, value=value):
                    rows = copy.deepcopy(original)
                    if value is None:
                        del rows[1][key]
                    else:
                        rows[1][key] = value
                    with gzip.open(path, 'wt') as stream:
                        stream.write('\n'.join(map(json.dumps, rows)) + '\n')
                    summary_path = root/'pod-0.summary.json'
                    summary = json.loads(summary_path.read_text())
                    summary['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    summary_path.write_text(json.dumps(summary))
                    failed = self.fit(root)
                    self.assertNotEqual(0, failed.returncode)
                    self.assertIn('pod-0.jsonl.gz:2', failed.stderr)
                    self.assertIn(reason, failed.stderr)
                    failure_summary = json.loads((root/'fit/fit.summary.json').read_text())
                    self.assertFalse(failure_summary['complete'])
                    self.assertIn(reason, failure_summary['errors'][0])

    def test_local_directory_matches_pod_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = self.fixture(root)
            script = ROOT/'src/traffic/capture_frontend_prefix.py'
            subprocess.run([sys.executable, str(script), '--start', '1000', '--end', '3000',
                            '--out', 'pod-cwd'], cwd=root, check=True, capture_output=True)
            summary = self.run_capture(root, logs)
            with gzip.open(root/'pod-cwd.jsonl.gz', 'rt') as first, gzip.open(root/'pod-0.jsonl.gz', 'rt') as second:
                self.assertEqual(first.read(), second.read())
            other = json.loads((root/'pod-cwd.summary.json').read_text())
            for key in ('stats', 'schema_version', 'block_size', 'hash', 'complete', 'truncated', 'start', 'end'):
                self.assertEqual(other[key], summary[key])

    def test_budget_writes_incomplete_summary_and_fit_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = self.fixture(root, count=1000)
            for policy in ('truncate', 'error'):
                times = iter([0, 0, 2, 2])
                run = lambda: self.run_capture(root, logs, ['--time-budget-s', '1', '--on-budget', policy],
                                              clock=lambda: next(times))
                if policy == 'error':
                    with self.assertRaisesRegex(RuntimeError, 'capture incomplete'):
                        run()
                else:
                    run()
                summary = json.loads((root/'pod-0.summary.json').read_text())
                self.assertTrue(summary['truncated'])
                self.assertFalse(summary['complete'])
                self.assertEqual(1, summary['stats']['budget_exceeded'])
                self.assertEqual(999, summary['stats']['records'])
                self.assertIn('incomplete capture', self.fit(root).stderr)

    def test_single_block_declaration_controls_capture_fit_and_codec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root/'src/traffic'
            shutil.copytree(ROOT/'src/traffic', package, ignore=shutil.ignore_patterns('__pycache__'))
            contract = package/'capture_contract.py'
            contract.write_text(contract.read_text().replace('BLOCK_SIZE = 512', 'BLOCK_SIZE = 1024'))
            logs = self.fixture(root)
            subprocess.run([sys.executable, str(package/'capture_frontend_prefix.py'),
                '--start', '1000', '--end', '3000', '--out', str(root/'pod-0'),
                '--log-dir', str(logs)], check=True, capture_output=True)
            with gzip.open(root/'pod-0.jsonl.gz', 'rt') as stream:
                row = json.loads(next(stream))
            self.assertEqual(1, len(row['keys']))
            self.assertEqual(1024, row['block_size'])
            self.assertEqual(1024, json.loads((root/'pod-0.summary.json').read_text())['block_size'])
            subprocess.run([sys.executable, str(package/'fit_frontend_prefix.py'), '--source', str(root),
                '--out', str(root/'fit'), '--expected-shards', '1', '--output-tokens', '10'],
                check=True, capture_output=True)
            manifest = json.loads((root/'fit/lineage-model.manifest.json').read_text())
            self.assertEqual(1024, manifest['block_size'])
            from traffic.datasets import read_manifest
            with self.assertRaisesRegex(ValueError, 'header'):
                read_manifest(root/'fit/lineage-model.xz')

"""Exercise the capture -> fit -> scenario source boundary with branched prefixes."""

import gzip
import hashlib
import json
import lzma
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from traffic.traffic_source import materialize
from traffic.prefix_lineage_v3 import decode


class FrontendPrefixFitTest(unittest.TestCase):
    def test_xz_capture_uses_arrival_cutoff_and_keeps_long_inputs(self):
        script = Path(__file__).resolve().parents[1] / 'src/traffic/capture_frontend_prefix.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'logs').mkdir()
            rows = [dict(request_id=str(i), ts_epoch_ms=3000+i, input_ids=[1]*32769,
                         status='ERROR', **({'request_enter_ts_epoch_ms': arrival} if arrival is not None else {}))
                    for i, arrival in enumerate([1999, 2000, 2999, 3000, None])]
            (root/'logs/dash_sc_grpc_access_r0_s0.log').write_text('\n'.join(map(json.dumps, rows))+'\n')
            subprocess.run([sys.executable, str(script), '--start', '2000', '--end', '3000',
                            '--out', 'pod-0', '--format', 'xz'], cwd=root, check=True, capture_output=True)
            with lzma.open(root/'pod-0.jsonl.xz', 'rt') as source:
                captured = [json.loads(line) for line in source]
            self.assertEqual([r['ts'] for r in captured], [2000, 2999])
            self.assertEqual([r['il'] for r in captured], [32769, 32769])

    def test_capture_fit_roundtrip_and_checksum_guard(self):
        scripts = Path(__file__).resolve().parents[1] / "src/traffic"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            inputs = [
                [11] * 512 + [22] * 513,
                [11] * 512 + [33] * 513,
                [11] * 512 + [22] * 513,
            ]
            rows = [
                dict(
                    request_id=str(i),
                    upstream_request_id=str(i),
                    ts_epoch_ms=2001 + i * 1000,
                    request_enter_ts_epoch_ms=2000 + i * 1000,
                    input_ids=tokens,
                    status="ERROR",
                    output_token_len=0,
                )
                for i, tokens in enumerate(inputs)
            ]
            (root / "logs/dash_sc_grpc_access_r0_s0.log").write_text(
                "\n".join(json.dumps(row) for row in rows + rows[-1:]) + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(scripts / "capture_frontend_prefix.py"),
                    "--start",
                    "1000",
                    "--end",
                    "10000",
                    "--out",
                    "pod-0",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            capture = root / "pod-0.jsonl.gz"
            captured = [json.loads(line) for line in gzip.open(capture, "rt")]
            self.assertEqual(len(captured), 3)
            # Fit accepts opaque shard names, independent of capture location.
            renamed = root / "anonymous-source.jsonl.gz"
            capture.rename(renamed)
            (root / "pod-0.summary.json").rename(root / "anonymous-source.summary.json")
            capture = renamed
            self.assertTrue(all("input_ids" not in row for row in captured))
            command = [
                sys.executable,
                str(scripts / "fit_frontend_prefix.py"),
                "--source",
                str(root),
                "--out",
                str(root / "fit"),
                "--expected-shards",
                "2",
                "--namespace",
                "custom",
                "--output-tokens",
                "420",
            ]
            rejected = subprocess.run(command + ["--model-version", "2"], capture_output=True, text=True)
            self.assertNotEqual(0, rejected.returncode)
            self.assertIn("--v2-reason", rejected.stderr)
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads((root / "fit/fit-report.json").read_text())
            self.assertEqual(report["missing_shard_count"], 1)
            self.assertEqual(report["source_shards"], 1)
            self.assertAlmostEqual(report["source_qps"], 3 / 9)
            model = root / "fit/lineage-model.xz"
            from traffic.datasets import read_manifest
            manifest = read_manifest(model)
            self.assertEqual(3, manifest['statistics']['request_count'])
            self.assertEqual('real', manifest['data_kind'])
            self.assertEqual('unconfirmed', manifest['source']['attribution']['status'])
            _, fitted = decode(model.read_bytes())
            self.assertEqual(
                [list(e[2:4]) for e in fitted], [[-1, 0], [0, 1], [0, 2]]
            )
            plan = materialize(
                root / "scenario.jsonl",
                dict(
                    kind="trace",
                    model="prefix_lineage",
                    version="3",
                    parameters=dict(
                        path=str(model),
                        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                        count=3,
                        output_tokens=420,
                        priority=50,
                    ),
                ),
                "test:flow",
                root,
            )
            actual = [json.loads(line) for line in plan.read_text().splitlines()]
            expected = [
                json.loads(line)
                for line in (root / "fit/input-plan.jsonl").read_text().splitlines()
            ]
            self.assertTrue(expected[0]["rid"].startswith("custom:"))
            for a, e in zip(actual, expected):
                a.pop("rid")
                e.pop("rid")
                self.assertEqual(a, e)
                self.assertEqual(
                    a["ol"], 420
                )  # Error-censored zero outputs are excluded.
            subprocess.run(command + ["--model-version", "2", "--v2-reason", "historical regression", "--out", str(root / "fit2")],
                           check=True, capture_output=True)
            exact = manifest
            legacy = read_manifest(root / "fit2/lineage-model.xz")
            self.assertEqual(legacy['codec']['version'], 2)
            self.assertEqual(legacy['provenance']['v2_reason'], 'historical regression')
            self.assertEqual(exact['codec']['version'], 3)
            self.assertEqual(exact['statistics']['input_tokens']['resolution_tokens'], 1)
            self.assertEqual(exact['statistics']['input_tokens']['total'], 3075)
            self.assertEqual(exact['statistics']['input_tokens']['min'], 1025)
            capture.write_bytes(capture.read_bytes() + b"corrupted")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("checksum mismatch", failed.stderr)

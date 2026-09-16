import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from online_eval.traffic_source import materialize, validate_plan


class TrafficSourceTest(unittest.TestCase):
    def test_recorded_plan_preserves_joint_shape_and_declares_identity_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            rows = [
                dict(
                    rid=str(i),
                    ts=t,
                    il=2,
                    ol=o,
                    input_ids=[1, 2],
                    priority=priority,
                    cache_key_block_size=1024,
                )
                for i, (t, o, priority) in enumerate(
                    [(0, 4, 30), (0, 8, 90), (150, 4, 30)]
                )
            ]
            raw = "\n".join(json.dumps(r) for r in rows) + "\n"
            (p / "source.jsonl").write_text(raw)
            spec = dict(
                kind="trace",
                model="recorded",
                version="1",
                parameters=dict(
                    path="source.jsonl",
                    sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    identity="namespace",
                ),
            )
            target = materialize(p / "result.jsonl", spec, "run:formal", p)
            actual = [json.loads(l) for l in target.read_text().splitlines()]
            for original, record in zip(rows, actual):
                self.assertEqual(record.pop("original_rid"), original["rid"])
                self.assertEqual(record.pop("rid"), "run:formal:" + original.pop("rid"))
                self.assertEqual(record, original)
            self.assertEqual(
                json.loads(target.with_suffix(".manifest.json").read_text())["realism"],
                "NOT_VALIDATED",
            )
            spec["parameters"]["sha256"] = "bad"
            with self.assertRaises(ValueError):
                materialize(p / "bad.jsonl", spec, "run:formal", p)
            self.assertFalse((p / "bad.jsonl").exists())

    def test_bad_shape_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            raw = json.dumps(
                dict(
                    rid="x",
                    ts=0,
                    il=2,
                    input_ids=[1],
                    ol=1,
                    priority=50,
                    cache_key_block_size=1024,
                )
            )
            (p / "in.jsonl").write_text(raw)
            spec = dict(
                kind="trace",
                model="recorded",
                version="1",
                parameters=dict(
                    path="in.jsonl",
                    sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    identity="namespace",
                ),
            )
            with self.assertRaises(ValueError):
                materialize(p / "out.jsonl", spec, "g", p)
            self.assertFalse((p / "out.jsonl").exists())
            spec["version"] = "unknown"
            with self.assertRaises(ValueError):
                materialize(p / "out.jsonl", spec, "g", p)

    def test_alias_cannot_bypass_group_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.jsonl"
            row = dict(
                rid="group:x",
                request_id_int=7,
                ts=0,
                il=1,
                ol=1,
                input_ids=[1],
                priority=50,
                cache_key_block_size=1024,
            )
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "namespace"):
                validate_plan(path)

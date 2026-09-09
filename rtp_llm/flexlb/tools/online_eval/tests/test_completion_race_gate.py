import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_race_gate import INSTANCE, analyze_round, main


class CompletionRaceGateTest(unittest.TestCase):
    def fixture(self, root, zombie=False, count=1):
        stages = [dict(id="setup", status="PASS")]
        for i in range(1, 6):
            artifact = root / f"balance-master-clean-{i}.json"
            samples = (
                [
                    dict(time_s=100 + j * 0.5, raw=dict(scheduler_inflight=count))
                    for j in range(60)
                ]
                if zombie and i == 2
                else [dict(time_s=100, raw=dict(scheduler_inflight=0))]
            )
            artifact.write_text(json.dumps(samples))
            stages.append(
                dict(
                    id=f"wave{i}_master_clean",
                    status=(
                        "TIMEOUT"
                        if zombie and i == 2
                        else "BLOCKED" if zombie and i > 2 else "PASS"
                    ),
                    started_s=100,
                    finished_s=130.001,
                    artifacts=[str(artifact)],
                )
            )
        value = dict(
            id=INSTANCE,
            status="TIMEOUT" if zombie else "PASS",
            cleanup=[dict(status="PASS")],
            stages=stages,
        )
        path = root / "result.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_green_and_scheduler_zombie(self):
        for zombie in (False, True):
            with self.subTest(zombie=zombie), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, zombie)
                self.assertEqual(
                    analyze_round(root, int(zombie))["verdict"],
                    "ZOMBIE" if zombie else "PASS",
                )

    def test_other_owner_timeout_is_not_scheduler_zombie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, True, 0)
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")

    def test_missing_samples_and_setup_error_never_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")
            path, value = self.fixture(root)
            (root / "balance-master-clean-1.json").unlink()
            self.assertEqual(analyze_round(root, 0)["verdict"], "INVALID")
            self.fixture(root)
            value["stages"][0]["status"] = "ERROR"
            path.write_text(json.dumps(value))
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")

    def test_short_or_stale_observation_not_a_thirty_second_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, value = self.fixture(root, True)
            value["stages"][2]["finished_s"] = 105
            path.write_text(json.dumps(value))
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")
            self.fixture(root, True)
            p = root / "balance-master-clean-2.json"
            samples = json.loads(p.read_text())
            p.write_text(json.dumps(samples[:3] + samples[-3:]))
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")

    def test_timeout_samples_saved_without_stage_artifact_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, value = self.fixture(root, True)
            # Real runtime behavior: StageTimeout prevents returning StageOutput.
            value["stages"][2]["artifacts"] = []
            path.write_text(json.dumps(value))
            # Earlier PASS-stage samples precede this stage's time interval.
            for i in (1, 3, 4, 5):
                (root / f"balance-master-clean-{i}.json").write_text(
                    json.dumps([dict(time_s=50, raw=dict(scheduler_inflight=0))])
                )
            self.assertEqual(analyze_round(root, 1)["verdict"], "ZOMBIE")
            source = root / "balance-master-clean-2.json"
            (root / "balance-master-clean-ambiguous.json").write_text(
                source.read_text()
            )
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")

    def test_offline_archive_judgment_and_exit_codes(self):
        for verdict, expected_code in (("PASS", 0), ("ZOMBIE", 1), ("INVALID", 2)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "archive"
                rows = []
                for i in range(1, 21):
                    folder = root / f"round-{i:03d}" / "cases"
                    folder.mkdir(parents=True)
                    self.fixture(folder, zombie=i == 2 and verdict == "ZOMBIE")
                    if i == 2 and verdict == "INVALID":
                        (folder / "result.json").unlink()
                    rows.append(
                        dict(
                            round=i,
                            lane=0,
                            runner_exit_code=0 if i != 2 or verdict == "PASS" else 1,
                        )
                    )
                original = json.dumps(dict(results=rows, lanes=1))
                (root / "summary.json").write_text(original)
                out = Path(tmp) / "judgment"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "completion_race_gate.py",
                        "--analyze-dir",
                        str(root),
                        "--out-dir",
                        str(out),
                    ],
                ), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), expected_code)
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["exit_code"], expected_code)
                self.assertEqual(summary["rounds"], 20)
                self.assertEqual((root / "summary.json").read_text(), original)

    def test_boolean_count_and_exit_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")
            self.fixture(root, True, True)
            self.assertEqual(analyze_round(root, 1)["verdict"], "INVALID")


if __name__ == "__main__":
    unittest.main()

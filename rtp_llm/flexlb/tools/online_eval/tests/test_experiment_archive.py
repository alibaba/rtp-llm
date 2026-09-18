import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_archive import create_archive


class ArchiveTest(unittest.TestCase):
    def test_structured_full_raw_bounded_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "aggregate.json").write_text('{"success": 2}')
            (run / "engine.log").write_bytes(b"a" * (2 * 1024 * 1024 + 1))
            (run / "api_token.txt").write_text("do not archive")
            out = root / "run.zip"
            manifest = create_archive(out, {"run": run}, kind="stress")
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(json.loads(archive.read("run/aggregate.json"))["success"], 2)
                self.assertIn("run/engine.log.head", archive.namelist())
                self.assertIn("run/engine.log.tail", archive.namelist())
                self.assertNotIn("run/api_token.txt", archive.namelist())
                self.assertEqual(json.loads(archive.read("manifest.json")), manifest)
            self.assertEqual(manifest["files"][1]["completeness"], "head_tail")


if __name__ == "__main__":
    unittest.main()

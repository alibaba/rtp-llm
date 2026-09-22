import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts.archive import create_archive


class ArchiveTest(unittest.TestCase):
    def test_structured_full_raw_bounded_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            (run / "aggregate.json").write_text('{"success": 2}')
            (run / "input_token_length.json").write_text('{"tokens": 3}')
            (run / "engine.log").write_bytes(b"a" * (2 * 1024 * 1024 + 1))
            (run / "api_token.txt").write_text("do not archive")
            out = root / "run.zip"
            manifest = create_archive(out, {"run": run}, kind="stress")
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(json.loads(archive.read("run/aggregate.json"))["success"], 2)
                self.assertIn("run/engine.log.head", archive.namelist())
                self.assertIn("run/engine.log.tail", archive.namelist())
                self.assertNotIn("run/api_token.txt", archive.namelist())
                self.assertIn("run/input_token_length.json", archive.namelist())
                self.assertEqual(json.loads(archive.read("manifest.json")), manifest)
            self.assertEqual(next(e for e in manifest["files"] if e["path"] == "run/engine.log")
                             ["completeness"], "head_tail")

    def test_archive_inside_source_does_not_include_its_temporary_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aggregate.json").write_text("{}")
            output = root / "result.zip"
            manifest = create_archive(output, {"run": root}, kind="case",
                                      status="incomplete", metadata={"exit_code": 130})
            self.assertEqual([entry["path"] for entry in manifest["files"]],
                             ["run/aggregate.json"])
            self.assertEqual(manifest["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()

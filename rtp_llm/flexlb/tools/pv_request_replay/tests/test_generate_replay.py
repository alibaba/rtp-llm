from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
MODULE_SPEC = importlib.util.spec_from_file_location(
    "pv_request_replay_generate", TOOL_DIR / "generate_replay.py"
)
assert MODULE_SPEC and MODULE_SPEC.loader
GENERATE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(GENERATE)


class GenerateReplayTest(unittest.TestCase):
    def test_non_strict_build_records_partial_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "pv.log"
            source.write_text("", encoding="utf-8")
            output = root / "output"

            with (
                mock.patch.object(
                    GENERATE,
                    "snapshot_sources",
                    return_value=([source], None),
                ),
                mock.patch.object(
                    GENERATE,
                    "build_workbook",
                    return_value={
                        "request_count": 2,
                        "complete_request_count": 1,
                    },
                ),
                mock.patch.object(
                    GENERATE,
                    "build_html",
                    return_value={"request_count": 2},
                ),
            ):
                manifest = GENERATE.run_build(
                    input_path=source,
                    output_dir=output,
                    start=datetime.fromisoformat("2026-08-11T01:00:00+08:00"),
                    end=datetime.fromisoformat("2026-08-11T02:00:00+08:00"),
                    template=TOOL_DIR / "replay_template.html",
                )

            self.assertEqual("partial", manifest["status"])
            self.assertEqual(1, manifest["join"]["incomplete_request_count"])
            stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("partial", stored["status"])

    def test_strict_build_stops_before_html_for_partial_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "pv.log"
            source.write_text("", encoding="utf-8")
            output = root / "output"

            html_builder = mock.Mock()
            with (
                mock.patch.object(
                    GENERATE,
                    "snapshot_sources",
                    return_value=([source], None),
                ),
                mock.patch.object(
                    GENERATE,
                    "build_workbook",
                    return_value={
                        "request_count": 3,
                        "complete_request_count": 2,
                    },
                ),
                mock.patch.object(GENERATE, "build_html", html_builder),
                self.assertRaisesRegex(RuntimeError, "complete joins=2/3"),
            ):
                GENERATE.run_build(
                    input_path=source,
                    output_dir=output,
                    start=datetime.fromisoformat("2026-08-11T01:00:00+08:00"),
                    end=datetime.fromisoformat("2026-08-11T02:00:00+08:00"),
                    template=TOOL_DIR / "replay_template.html",
                    strict=True,
                )

            html_builder.assert_not_called()
            stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", stored["status"])


if __name__ == "__main__":
    unittest.main()

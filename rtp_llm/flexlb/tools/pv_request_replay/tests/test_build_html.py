from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook


TOOL_DIR = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "pv_request_replay_build_html", TOOL_DIR / "build_html.py"
)
assert MODULE_SPEC and MODULE_SPEC.loader
BUILD_HTML_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(BUILD_HTML_MODULE)
build_html = BUILD_HTML_MODULE.build_html


class BuildHtmlTest(unittest.TestCase):
    def test_builds_self_contained_replay_from_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workbook_path = root / "input.xlsx"
            output_path = root / "nested" / "replay.html"
            self._write_workbook(workbook_path)

            summary = build_html(
                workbook_path, TOOL_DIR / "replay_template.html", output_path
            )

            self.assertEqual(summary["request_count"], 2)
            self.assertEqual(summary["host_count"], 2)
            self.assertEqual(summary["candidate_count"], 2)
            self.assertTrue(output_path.is_file())

            html = output_path.read_text(encoding="utf-8")
            self.assertNotIn("__REPLAY_DATA__", html)
            self.assertNotIn("<script src=", html)
            self.assertNotIn("https://", html)
            self.assertIn('id="speed-select"', html)
            self.assertIn("所选机器的预测首 Token 前工作量（Token）", html)
            self.assertIn("引擎首 Token 分位", html)
            self.assertIn("空格</kbd> 播放 / 暂停", html)
            self.assertNotIn("recorded decision candidates", html)
            self.assertNotIn("目标机器：", html)
            self.assertIn("event.code === 'ArrowLeft'", html)
            self.assertIn("event.code === 'ArrowRight'", html)

            replay = self._embedded_replay(html)
            self.assertEqual(replay["meta"]["source"], "input.xlsx")
            self.assertEqual(replay["hosts"], ["10.0.0.1", "10.0.0.2"])
            self.assertEqual(
                [request["id"] for request in replay["requests"]],
                ["instance-b::request-earlier", "instance-a::request-later"],
            )
            self.assertEqual(replay["requests"][0]["requestId"], "request-earlier")
            self.assertEqual(
                replay["candidates"]["instance-b::request-earlier"][0]["host"],
                "10.0.0.2",
            )

            node = shutil.which("node")
            if node:
                script = re.search(r"<script>([\s\S]*?)</script>", html)
                self.assertIsNotNone(script)
                script_path = root / "replay.js"
                script_path.write_text(script.group(1), encoding="utf-8")
                result = subprocess.run(
                    [node, "--check", str(script_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_builds_replay_from_legacy_requests_header_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workbook_path = root / "legacy.xlsx"
            output_path = root / "replay.html"
            self._write_workbook(workbook_path, legacy_requests_header=True)

            summary = build_html(
                workbook_path, TOOL_DIR / "replay_template.html", output_path
            )

            self.assertEqual(summary["request_count"], 2)
            self.assertTrue(output_path.is_file())

    @staticmethod
    def _embedded_replay(html: str) -> dict:
        prefix = "const REPLAY = "
        start = html.index(prefix) + len(prefix)
        end = html.index(";\n  (() =>", start)
        return json.loads(html[start:end])

    @staticmethod
    def _write_workbook(path: Path, legacy_requests_header: bool = False) -> None:
        workbook = Workbook()
        requests = workbook.active
        requests.title = "Requests"
        if legacy_requests_header:
            for _ in range(3):
                requests.append([])
        requests.append(
            [
                "request_id",
                "flexlb_instance",
                "prefill_host",
                "route_log_time (decision)",
                "input_queue_enqueue_time",
                "input_queue_drain_time",
                "first_token_time",
                "selection_reason",
                "prefill_engine_ttft_ms",
                "prefill_ttft_percentile",
                "running_to_first_token_ms",
                "prefill_step_count",
                "first_prefill_step_id",
                "last_prefill_step_id",
                "input_tokens",
                "uncache_tokens",
                "actual_hit_rate_pct",
                "predicted_hit_rate_pct",
                "selected snapshot request uncache",
            ]
        )
        requests.append(
            [
                "request-later",
                "instance-a",
                "10.0.0.1",
                "2026-08-11 02:00:10.000",
                "2026-08-11 02:00:10.010",
                "2026-08-11 02:00:10.020",
                "2026-08-11 02:00:12.000",
                "CACHE_LEADER",
                2000,
                "P95-P99",
                1900,
                2,
                100,
                101,
                100000,
                1000,
                99,
                98.5,
                1500,
            ]
        )
        requests.append(
            [
                "request-earlier",
                "instance-b",
                "10.0.0.2",
                "2026-08-11 02:00:00.000",
                "2026-08-11 02:00:00.010",
                "2026-08-11 02:00:00.020",
                "2026-08-11 02:00:03.000",
                "SHORTEST_TTFT",
                3000,
                "P99-P100",
                2800,
                3,
                90,
                92,
                200000,
                200000,
                "0%",
                "0%",
                200000,
            ]
        )

        candidates = workbook.create_sheet("Decision Snapshot Top5")
        for _ in range(3):
            candidates.append([])
        candidates.append(
            [
                "request_id",
                "flexlb_instance",
                "candidate_rank_by_estimated_TTFT",
                "candidate_ip",
                "candidate_port",
                "selected",
                "cache_leader",
                "shortest_TTFT",
                "outstanding_guard_eligible",
                "request_hit_rate_pct",
                "request_uncache_tokens",
                "queue_work_before_route",
                "estimated_TTFT_work",
            ]
        )
        candidates.append(
            [
                "request-earlier",
                "instance-b",
                2,
                "10.0.0.1",
                8001,
                False,
                True,
                False,
                True,
                90,
                20000,
                50000,
                70000,
            ]
        )
        candidates.append(
            [
                "request-earlier",
                "instance-b",
                1,
                "10.0.0.2",
                8001,
                True,
                False,
                True,
                True,
                0,
                200000,
                0,
                200000,
            ]
        )
        workbook.save(path)


if __name__ == "__main__":
    unittest.main()

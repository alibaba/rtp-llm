#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("audit_kmonitor_dashboard.py")
SPEC = importlib.util.spec_from_file_location("audit_kmonitor_dashboard", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditKmonitorDashboardTest(unittest.TestCase):
    def test_query_deduplicates_after_resolving_variables(self):
        targets = [
            {"metric": "navi_sched.test", "tags": {"host": "$agent_host"}},
            {"metric": "navi_sched.test", "tags": {"host": "$host"}},
            {"metric": "navi_sched.other", "tags": {"deployment": "$deployment"}},
        ]
        payload = AUDIT.query_payload(targets, "deployment-a", 10, 20)
        self.assertEqual(2, len(payload["queries"]))
        self.assertEqual("*", payload["queries"][0]["tags"]["host"])
        self.assertEqual(
            "deployment-a", payload["queries"][1]["tags"]["deployment"]
        )

    def test_main_queries_unique_targets_in_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dashboard = directory / "dashboard.json"
            output = directory / "audit.json"
            dashboard.write_text(
                json.dumps(
                    {
                        "dashboard": {
                            "panels": [
                                {
                                    "targets": [
                                        {"metric": "navi_sched.a", "tags": {}},
                                        {"metric": "navi_sched.a", "tags": {}},
                                        {"metric": "navi_sched.b", "tags": {}},
                                        {"metric": "navi_sched.c", "tags": {}},
                                    ]
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            query_sizes = []

            def request_json(_endpoint, _cookies, payload, _attempts):
                query_sizes.append(len(payload["queries"]))
                return [
                    {
                        "metric": query["metric"],
                        "tags": query["tags"],
                        "dps": {"100": 1.0},
                    }
                    for query in payload["queries"]
                ]

            argv = [
                str(SCRIPT),
                "--dashboard",
                str(dashboard),
                "--endpoint",
                "http://kmonitor.invalid/query",
                "--output",
                str(output),
                "--deployment",
                "deployment-a",
                "--query-chunk-size",
                "2",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                AUDIT, "request_json", side_effect=request_json
            ), mock.patch.object(AUDIT.time, "time", return_value=200):
                self.assertEqual(0, AUDIT.main())

            self.assertEqual([2, 1], query_sizes)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(3, result["series_count"])
            self.assertEqual(2, result["query_chunk_size"])
            self.assertEqual([], result["failures"])


if __name__ == "__main__":
    unittest.main()

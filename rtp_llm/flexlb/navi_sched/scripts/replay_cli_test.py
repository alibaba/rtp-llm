#!/usr/bin/env python3

import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from navi_sched import replay_cli


class _Handler(BaseHTTPRequestHandler):
    requests = []
    response = {"accepted": True, "generation": 1}
    raw_response = None
    status = 200

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        self.__class__.requests.append(json.loads(self.rfile.read(size)))
        if self.__class__.raw_response is None:
            body = json.dumps(self.__class__.response).encode("utf-8")
        else:
            body = self.__class__.raw_response
        self.send_response(self.__class__.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


class ReplayCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_args = [
            "--url",
            "http://127.0.0.1:{}/replay-control".format(cls.server.server_port),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def setUp(self):
        _Handler.requests = []
        _Handler.response = {"accepted": True, "generation": 1}
        _Handler.raw_response = None
        _Handler.status = 200

    def run_cli(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = replay_cli.main(self.base_args + args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_query_and_stop_payloads(self):
        self.assertEqual(0, self.run_cli(["query"])[0])
        self.assertEqual(0, self.run_cli(["stop"])[0])
        self.assertEqual([{"action": "query"}, {"action": "stop"}], _Handler.requests)

    def test_fixed_qps_start_payload(self):
        code, _, error = self.run_cli(
            [
                "start",
                "--mode",
                "fixed_qps",
                "--start-time-us",
                "1000000",
                "--end-time-us",
                "2000000",
                "--total-qps",
                "24.5",
                "--loop-count",
                "3",
            ]
        )
        self.assertEqual(0, code, error)
        self.assertEqual(
            {
                "action": "start",
                "mode": "fixed_qps",
                "historyMode": "time_range",
                "startTimeUs": 1000000,
                "endTimeUs": 2000000,
                "durationMs": 0,
                "totalQps": 24.5,
                "loopCount": 3,
                "recentQueryCount": 0,
            },
            _Handler.requests[0],
        )

    def test_recent_count_start_payload(self):
        code, _, error = self.run_cli(
            [
                "start",
                "--mode",
                "fixed_qps",
                "--history-mode",
                "recent_count",
                "--recent-query-count",
                "50",
                "--total-qps",
                "20",
                "--loop-count",
                "2",
            ]
        )
        self.assertEqual(0, code, error)
        self.assertEqual("recent_count", _Handler.requests[0]["historyMode"])
        self.assertEqual(50, _Handler.requests[0]["recentQueryCount"])
        self.assertEqual(0, _Handler.requests[0]["startTimeUs"])
        self.assertEqual(0, _Handler.requests[0]["endTimeUs"])

    def test_realtime_update_payload(self):
        code, _, error = self.run_cli(
            [
                "update",
                "--mode",
                "realtime",
                "--start-time-us",
                "3000000",
                "--duration-ms",
                "5000",
                "--loop-count",
                "2",
            ]
        )
        self.assertEqual(0, code, error)
        self.assertEqual(0.0, _Handler.requests[0]["totalQps"])
        self.assertEqual(0, _Handler.requests[0]["endTimeUs"])
        self.assertEqual(5000, _Handler.requests[0]["durationMs"])

    def test_qps_only_update_payload(self):
        code, _, error = self.run_cli(["update", "--total-qps", "37.5"])
        self.assertEqual(0, code, error)
        self.assertEqual(
            {"action": "update", "totalQps": 37.5}, _Handler.requests[0]
        )

    def test_qps_only_update_rejects_mixed_arguments(self):
        code, _, error = self.run_cli(
            ["update", "--total-qps", "20", "--loop-count", "2"]
        )
        self.assertEqual(1, code)
        self.assertIn("不能同时修改", error)
        self.assertEqual([], _Handler.requests)

    def test_mode_specific_and_range_validation(self):
        cases = [
            (["start", "--mode", "fixed_qps", "--start-time-us", "1", "--end-time-us", "2", "--loop-count", "1"], "--total-qps"),
            (["start", "--mode", "realtime", "--start-time-us", "1", "--end-time-us", "2", "--loop-count", "1", "--total-qps", "2"], "不能指定"),
            (["start", "--mode", "realtime", "--start-time-us", "2", "--end-time-us", "1", "--loop-count", "1"], "必须大于"),
            (["start", "--mode", "realtime", "--start-time-us", "1", "--end-time-us", "2"], "--loop-count"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                code, _, error = self.run_cli(args)
                self.assertEqual(1, code)
                self.assertIn(message, error)
        self.assertEqual([], _Handler.requests)

    def test_rejected_response_and_http_error_are_nonzero(self):
        _Handler.response = {"accepted": False, "error": "no eligible client"}
        code, output, _ = self.run_cli(["query"])
        self.assertEqual(2, code)
        self.assertIn("no eligible client", output)

        _Handler.status = 503
        code, _, error = self.run_cli(["query"])
        self.assertEqual(1, code)
        self.assertIn("HTTP 503", error)

    def test_invalid_json_response_is_nonzero(self):
        _Handler.raw_response = b"not-json"
        code, _, error = self.run_cli(["query"])
        self.assertEqual(1, code)
        self.assertIn("不是合法 JSON", error)

    def test_nonfinite_qps_is_rejected_before_request(self):
        with self.assertRaises(SystemExit):
            self.run_cli(
                [
                    "start",
                    "--mode",
                    "fixed_qps",
                    "--start-time-us",
                    "1",
                    "--end-time-us",
                    "2",
                    "--loop-count",
                    "1",
                    "--total-qps",
                    "nan",
                ]
            )
        self.assertEqual([], _Handler.requests)

    def test_help_is_chinese(self):
        help_text = replay_cli._parser().format_help()
        self.assertIn("查询和控制", help_text)
        self.assertIn("立即停止回放", help_text)

    def test_port_defaults_from_master_environment(self):
        with mock.patch.dict(
            "os.environ", {"NAVI_SCHED_MASTER_HTTP_PORT": "12233"}, clear=False
        ):
            parser = replay_cli._parser()
            args = parser.parse_args(["query"])
        self.assertEqual(12233, args.port)


if __name__ == "__main__":
    unittest.main()

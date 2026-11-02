#!/usr/bin/env python3

import importlib.util
import json
import re
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_SCRIPT = SCRIPT_DIR / "navi_sched_config.py"
UPDATE_SCRIPT = SCRIPT_DIR / "update_scheduler_config.sh"
EXAMPLE_SCRIPT = SCRIPT_DIR / "update_scheduler_config_example.sh"
SCHEDULER_CONFIG_CPP = SCRIPT_DIR.parent / "resource" / "SchedulerConfigR.cpp"

SPEC = importlib.util.spec_from_file_location("navi_sched_config", CONFIG_SCRIPT)
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def _cpp_fields(function_name):
    source = SCHEDULER_CONFIG_CPP.read_text(encoding="utf-8")
    if function_name == "applyPatch":
        pattern = (
            r"bool SchedulerConfigR::applyPatch\(.*?supportedFields = \{"
            r"(.*?)\n    \};"
        )
    else:
        pattern = (
            r"const std::set<std::string> &agentConfigFields\(\).*?fields = \{"
            r"(.*?)\n    \};"
        )
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        raise AssertionError("cannot find scheduler config field set: " + function_name)
    return set(re.findall(r'"([a-z][a-z0-9_]+)"', match.group(1)))


class _Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        self.__class__.requests.append(json.loads(self.rfile.read(size)))
        body = json.dumps(
            {"accepted": True, "generation": len(self.__class__.requests)}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


class SchedulerConfigToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Handler.requests = []
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        cls.url = "http://127.0.0.1:{}/scheduler-config".format(
            cls.server.server_port
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def test_python_field_lists_match_production(self):
        self.assertEqual(_cpp_fields("applyPatch"), set(CONFIG.MASTER_FIELDS))
        self.assertEqual(_cpp_fields("agentConfigFields"), set(CONFIG.AGENT_FIELDS))

    def test_python_update_accepts_every_supported_field(self):
        patch = {
            name: (
                True
                if value_type is bool
                else ({} if value_type is dict else [] if value_type is list else value_type(1))
            )
            for name, value_type in CONFIG.MASTER_FIELDS.items()
        }
        with mock.patch.object(
            CONFIG,
            "_post",
            return_value={"accepted": True, "generation": 2},
        ) as post:
            self.assertEqual(
                0,
                CONFIG.main(
                    ["--url", self.url, "update", "--patch", json.dumps(patch)]
                ),
            )
        payload = post.call_args[0][1]
        self.assertTrue(payload["update"])
        self.assertEqual(patch, json.loads(payload["configJson"]))

    def test_shell_get_and_typed_update(self):
        before = len(_Handler.requests)
        result = subprocess.run(
            [str(UPDATE_SCRIPT), "--url", self.url, "--get"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual({}, _Handler.requests[before])

        result = subprocess.run(
            [
                str(UPDATE_SCRIPT),
                "--url",
                self.url,
                "--set",
                "batch_timeout_ms=17",
                "--set",
                "latency_learning_rate=0.02",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        payload = _Handler.requests[before + 1]
        self.assertTrue(payload["update"])
        self.assertEqual(
            {"batch_timeout_ms": 17, "latency_learning_rate": 0.02},
            json.loads(payload["configJson"]),
        )

    def test_shell_hardware_selector_is_separate_from_patch(self):
        before = len(_Handler.requests)
        hardware = "AMD_Instinct_MI308X"
        result = subprocess.run(
            [
                str(UPDATE_SCRIPT),
                "--url",
                self.url,
                "--hardware",
                hardware,
                "--set",
                "residual_learning_rate=1.0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        payload = _Handler.requests[before]
        self.assertEqual(hardware, payload["hardware"])
        self.assertEqual(
            {"residual_learning_rate": 1.0},
            json.loads(payload["configJson"]),
        )

    def test_python_hardware_selector_is_separate_from_patch(self):
        hardware = "NVIDIA RTX PRO 5000 72GB Blackwell"
        with mock.patch.object(
            CONFIG,
            "_post",
            return_value={"accepted": True, "generation": 2},
        ) as post:
            self.assertEqual(
                0,
                CONFIG.main(
                    [
                        "--url",
                        self.url,
                        "update",
                        "--hardware",
                        hardware,
                        "--set",
                        "residual_learning_rate=0.055",
                    ]
                ),
            )
        payload = post.call_args[0][1]
        self.assertEqual(hardware, payload["hardware"])
        self.assertEqual(
            {"residual_learning_rate": 0.055},
            json.loads(payload["configJson"]),
        )

    def test_example_uses_local_master_defaults(self):
        before = len(_Handler.requests)
        env = dict(**__import__("os").environ)
        env["NAVI_MASTER_HTTP_PORT"] = str(self.server.server_port)
        result = subprocess.run(
            [str(EXAMPLE_SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        payload = _Handler.requests[before]
        self.assertEqual(
            {"latency_current_sample_weight": 4.0},
            json.loads(payload["configJson"]),
        )


if __name__ == "__main__":
    unittest.main()

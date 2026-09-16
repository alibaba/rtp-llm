"""Shared contracts: no scenario imports, consistent launcher isolation."""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from online_eval.load_client import LOAD_CLIENT_ENV_VARS
from online_eval.metrics import parse_prometheus_samples
from online_eval.requests import ClientRecords, request_success
from flexlb_test_framework.harness import ClientOps, _parse_per_engine_lines

ROOT = Path(__file__).resolve().parents[1]


class SharedRuntimeTests(unittest.TestCase):
    def test_requests_can_load_without_any_case_framework(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import online_eval.requests; "
                'assert not any(k.startswith("flexlb_test_framework") for k in sys.modules)',
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unfinished_output_is_not_business_success(self):
        rows = ClientRecords("run1")
        record = rows.issue(42, lambda: 1)
        rows.update(
            record,
            schedule={"status": "OK"},
            stream={"status": "OK", "first_output_s": 2},
            consumer_exit_s=3,
        )
        self.assertFalse(request_success(record))
        rows.update(record, business_finished=True)
        self.assertTrue(request_success(record))
        rows.update(record, business_error_code=8511)
        self.assertFalse(request_success(record))

    def test_python_blanks_ambient_priority_and_preserves_explicit_override(self):
        with patch.dict(
            os.environ,
            {
                "FORCE_PRIORITY": "99",
                "RAMP_UP_SECONDS": "500",
                "REPLAY_UNIQUE_PREFIX": "false",
            },
        ):
            env = ClientOps(None)._base_env({"PRIORITY": 70})
        self.assertEqual(env["PRIORITY"], "70")
        for name in ("FORCE_PRIORITY", "RAMP_UP_SECONDS", "REPLAY_UNIQUE_PREFIX"):
            self.assertEqual(env[name], "")

    def test_shell_and_python_use_identical_contract_through_both_entries(self):
        env = dict(os.environ, FLEXLB_DIR=str(ROOT.parents[1]))
        for entry in ("lib_load_client.sh", "stress/lib_load_client.sh"):
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1" || exit; printf "%s\\n" "${JAVA_LOAD_CLIENT_ENV_VARS[@]}"',
                    "test",
                    str(ROOT / entry),
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), LOAD_CLIENT_ENV_VARS)

    def test_java_environment_reads_are_all_isolated(self):
        source = (
            ROOT.parents[1]
            / "flexlb-mock-engine/src/main/java/org/flexlb/mockengine/JavaLoadClient.java"
        ).read_text()
        names = set(
            re.findall(
                r'(?:System\.getenv|env(?:Int|Long|Double|Bool)?)\("([A-Z_]+)"', source
            )
        )
        self.assertTrue(names)
        self.assertFalse(names - set(LOAD_CLIENT_ENV_VARS))
        self.assertEqual(len(LOAD_CLIENT_ENV_VARS), len(set(LOAD_CLIENT_ENV_VARS)))

    def test_balance_filter_retains_parser_skip_and_timestamp_semantics(self):
        body = "\n".join(
            [
                "# HELP mock_engine_running running",
                'mock_engine_running{engine_name="P0",role="PREFILL",} 3 12345',
                'mock_engine_running{engine_name="P1"} invalid',
                "unrelated 4",
            ]
        )
        sample = ("mock_engine_running", {"engine_name": "P0", "role": "PREFILL"}, 3.0)
        self.assertEqual(
            parse_prometheus_samples(body, "mock_engine_", {"role": "PREFILL"}),
            [sample],
        )
        self.assertEqual(_parse_per_engine_lines(body), [sample])


if __name__ == "__main__":
    unittest.main()

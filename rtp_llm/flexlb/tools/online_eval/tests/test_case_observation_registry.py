"""New evidence cases stay registered and cannot hide unavailable sources as findings."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from flexlb_ft.cases.debug_snapshot_readonly import master_debug_snapshot
from flexlb_ft.cases.no_fetch_observation import normal_no_fetch_observation
from flexlb_ft.context import CaseExecutionError
from flexlb_ft.debug_client import DebugUnavailable
from flexlb_ft.harness import PROFILE_CAPS, PROFILES, EnvSpec
from flexlb_functional_tests import ALL_CASES, classify_outcome


class ObservationRegistryTest(unittest.TestCase):
    def test_two_additions_have_exact_profiles_and_no_expected_fail(self):
        new = {
            c.name: c
            for c in ALL_CASES
            if c.name in ("master_debug_snapshot", "normal_no_fetch_observation")
        }
        self.assertEqual(len(ALL_CASES), 139)
        self.assertEqual(len(new), 2)
        for case in new.values():
            self.assertEqual(case.category, "status")
            self.assertEqual(case.profiles, ["batch-window"])
            self.assertEqual(case.requires, ["enqueue_batch"])
            self.assertFalse(case.expected_fail)
        counts = {
            p: sum(
                (c.profiles is None or p in c.profiles)
                and set(c.requires or []).issubset(PROFILE_CAPS[p])
                for c in ALL_CASES
            )
            for p in PROFILES
        }
        self.assertEqual(
            counts,
            {
                "batch-window": 120,
                "single-nonbatch": 80,
                "single-batch": 73,
                "window-nonbatch": 65,
            },
        )

    def test_execution_errors_are_not_findings_or_contract_failures(self):
        for expected_fail in (False, True):
            self.assertEqual(classify_outcome(expected_fail, False, True), "ERROR")
        self.assertEqual(classify_outcome(True, False), "FINDING-CONFIRMED")
        self.assertEqual(classify_outcome(False, False), "FAIL")

    def test_missing_sources_raise_typed_error_and_debug_is_explicit(self):
        from unittest.mock import Mock

        for module, fn in [
            ("debug_snapshot_readonly", master_debug_snapshot),
            ("no_fetch_observation", normal_no_fetch_observation),
        ]:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                ensure = Mock(return_value=NS(master_http_port=28000))
                ops = NS(
                    snapshot=Mock(side_effect=DebugUnavailable("missing mock source"))
                )
                ctx = NS(
                    profile="batch-window",
                    smoke_spec=lambda: EnvSpec(),
                    env_manager=NS(ensure=ensure),
                    engine_ops=lambda env: ops,
                    case_dir=lambda name: Path(tmp),
                )
                client = NS(
                    snapshot=Mock(side_effect=DebugUnavailable("missing debug source"))
                )
                with patch(
                    "flexlb_ft.cases." + module + ".DebugClient", return_value=client
                ):
                    with self.assertRaises(CaseExecutionError):
                        fn(ctx)
                self.assertEqual(
                    ensure.call_args.args[0].master_env,
                    {"FLEXLB_DEBUG_ENABLED": "true"},
                )


if __name__ == "__main__":
    unittest.main()

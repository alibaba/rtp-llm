import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke import v41_startup_memory as memory


class StartupEventsTest(unittest.TestCase):
    def test_v2_delegated_scope_and_nonidentifying_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "cg" / "child"
            group.mkdir(parents=True)
            (root / "cgroup").write_text("0::/private-parent/child\n")
            (root / "mountinfo").write_text(
                f"8 1 0:2 /private-parent {root}/cg rw - cgroup2 cgroup rw\n"
            )
            events = group / "memory.events"
            events.write_text("high 7\nmax 0\noom 2\noom_kill 8\n")
            before = memory._snapshot(root)
            events.write_text("high 88\nmax 4\noom 3\noom_kill 9\n")
            expected = {
                "status": "observed",
                "version": 2,
                "delta": {"max": 4, "oom": 1, "oom_kill": 1},
            }
            self.assertEqual(
                memory._event_delta(before, memory._snapshot(root)), expected
            )
            (root / "cgroup").write_text("0::/unrelated/child\n")
            self.assertIsNone(memory._snapshot(root))

    def test_v1_namespace_root_requires_self_membership_and_real_kill_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cgroup").write_text("8:memory:/\n1:cpu:/\n")
            (root / "mountinfo").write_text(
                f"1 2 0:3 /delegated {root} rw - cgroup cgroup rw,memory\n"
            )
            events = root / "memory.oom_control"
            events.write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 14\n")
            self.assertIsNone(memory._snapshot(root))
            (root / "cgroup.procs").write_text(f"{os.getpid()}\n")
            before = memory._snapshot(root)
            self.assertEqual(
                memory._event_delta(before, before)["delta"], {"oom_kill": 0}
            )
            events.write_text("oom_kill_disable 0\nunder_oom 0\n")
            self.assertIsNone(memory._snapshot(root))

    def test_cgroup_changes_or_counter_reset_are_not_zero(self):
        before = ((2, 1, 9), {"oom_kill": 5})
        for after in (
            None,
            ((2, 1, 10), {"oom_kill": 5}),
            ((2, 1, 9), {"oom_kill": 0}),
        ):
            self.assertEqual(
                memory._event_delta(before, after), {"status": "unavailable"}
            )

    def test_startup_exception_survives_and_artifact_does_not_leak_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counter = ((2, 12345, 998877), {"oom_kill": 42})
            with patch.object(memory, "_snapshot", return_value=counter):
                with self.assertRaisesRegex(RuntimeError, "backend died"):
                    with memory.record_v41_startup_events(root):
                        raise RuntimeError("backend died")
                report = json.loads((root / "v41_startup_memory.json").read_text())
                self.assertEqual(set(report), {"scope", "version", "delta", "status"})
                self.assertEqual(report["delta"], {"oom_kill": 0})
                with self.assertRaisesRegex(RuntimeError, "backend died"):
                    with memory.record_v41_startup_events(root / "no-directory"):
                        raise RuntimeError("backend died")


if __name__ == "__main__":
    unittest.main()

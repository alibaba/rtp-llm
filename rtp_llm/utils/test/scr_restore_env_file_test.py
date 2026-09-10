"""Verify the optional platform file through the real SCR release boundary."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from rtp_llm.utils import scr_runtime_fixup as runtime
from rtp_llm.utils import scr_template_utils as scr
from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle


class ScrRestoreEnvFileTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "envs.json"
        for context in (
            patch.object(runtime, "RESTORE_ENV_FILE", str(self.path)),
            patch.object(runtime, "_restore_env_provider", None),
            patch.object(runtime, "_runtime_identity", None),
            patch.dict(
                os.environ,
                {"RequestedIP": "192.0.2.10", "HIPPO_SLAVE_IP": "192.0.2.100"},
            ),
            patch.dict(
                sys.modules,
                {
                    "libth_transformer": None,
                    "rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper": None,
                },
            ),
        ):
            context.start()
            self.addCleanup(context.stop)

    def publish(self, values):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values), encoding="utf-8")
        temporary.replace(self.path)

    def barrier(self, events, after_barrier=None):
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "component",
            CallbackHook(
                prepare=lambda g: events.append("prepare"),
                fixup=lambda g: events.append(("fixup", os.environ["RequestedIP"])),
                release=lambda g: events.append("release"),
                abort=lambda g: events.append("abort"),
            ),
        )

        def arrive(**kwargs):
            events.append("barrier")
            if after_barrier:
                after_barrier()
            return 0

        with patch.object(
            scr, "is_scr_template_phase_active", return_value=True
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", side_effect=arrive
        ):
            return scr.arrive_scr_template_barrier(
                worker_id=0, worker_num=1, generation="same-seed"
            )

    def test_missing_file_falls_back_and_releases(self):
        events = []
        with patch.object(runtime, "current_pod_ip", return_value="192.0.2.20"):
            self.assertEqual(self.barrier(events), 0)
        self.assertEqual(
            events, ["prepare", "barrier", ("fixup", "192.0.2.20"), "release"]
        )
        self.assertEqual(os.environ["HIPPO_SLAVE_IP"], "192.0.2.100")
        self.assertEqual(runtime.get_restore_runtime_identity().environment_keys, ())

    def test_file_is_read_after_barrier_and_before_components(self):
        # The seed-side bytes are invalid; only the newly published input may be read.
        self.path.write_text("not-json", encoding="utf-8")
        events = []
        with patch.object(
            runtime, "current_pod_ip", side_effect=AssertionError("file supplies IP")
        ):
            self.barrier(
                events,
                lambda: self.publish(
                    {"RequestedIP": "192.0.2.20", "HIPPO_SLAVE_IP": "192.0.2.200"}
                ),
            )
        self.assertEqual(events[-2:], [("fixup", "192.0.2.20"), "release"])
        self.assertEqual(os.environ["HIPPO_SLAVE_IP"], "192.0.2.200")

    def test_repeated_restore_rereads_atomically_replaced_file(self):
        for address in ("192.0.2.20", "192.0.2.30"):
            self.publish({"RequestedIP": address})
            self.barrier([])
            self.assertEqual(runtime.get_restore_runtime_identity().pod_ip, address)

    def test_file_removed_after_previous_restore_uses_fallback(self):
        self.publish({"RequestedIP": "192.0.2.20", "HIPPO_SLAVE_IP": "192.0.2.200"})
        self.barrier([])
        self.path.unlink()
        with patch.object(runtime, "current_pod_ip", return_value="192.0.2.30"):
            self.barrier([])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.30")
        self.assertEqual(os.environ["HIPPO_SLAVE_IP"], "192.0.2.200")
        self.assertEqual(runtime.get_restore_runtime_identity().environment_keys, ())

    def test_full_environment_only_updates_supported_fields(self):
        self.publish(
            {"RequestedIP": "192.0.2.20", "kmonitorPort": "4141", "TP_SIZE": "99"}
        )
        with patch.dict(os.environ, {"TP_SIZE": "2"}):
            self.barrier([])
            self.assertEqual(os.environ["TP_SIZE"], "2")
            self.assertEqual(os.environ["kmonitorPort"], "4141")
        self.assertEqual(
            runtime.get_restore_runtime_identity().environment_keys,
            ("RequestedIP", "kmonitorPort"),
        )

    def test_null_removes_seed_value_and_rediscovers_pod_ip(self):
        self.publish({"RequestedIP": None, "HIPPO_ROLE_SHORT_NAME": None})
        with patch.dict(
            os.environ, {"HIPPO_ROLE_SHORT_NAME": "seed-role"}
        ), patch.object(runtime, "current_pod_ip", return_value="192.0.2.30"):
            self.barrier([])
            self.assertNotIn("HIPPO_ROLE_SHORT_NAME", os.environ)
            self.assertEqual(os.environ["RequestedIP"], "192.0.2.30")

    def test_empty_object_preserves_discovery_fallback(self):
        self.publish({})
        with patch.object(runtime, "current_pod_ip", return_value="192.0.2.30"):
            self.barrier([])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.30")

    def test_invalid_files_abort_without_mutating_environment(self):
        for data in (
            b"",
            b"{",
            b"\xff",
            b"[]",
            b"null",
            b'"string"',
            b'{"RequestedIP":"192.0.2.20","HIPPO_ROLE":42}',
            b'{"RequestedIP":"127.0.0.1"}',
        ):
            with self.subTest(data=data):
                self.path.write_bytes(data)
                events = []
                with self.assertRaises(ValueError):
                    self.barrier(events)
                self.assertEqual(events, ["prepare", "barrier", "abort"])
                self.assertEqual(os.environ["RequestedIP"], "192.0.2.10")
                self.assertIsNone(runtime.get_restore_runtime_identity())

    def test_permission_error_is_not_treated_as_absent(self):
        events = []
        with patch(
            "builtins.open", side_effect=PermissionError("denied")
        ), self.assertRaises(PermissionError):
            self.barrier(events)
        self.assertEqual(events, ["prepare", "barrier", "abort"])

    def test_custom_provider_overrides_file_and_none_restores_default(self):
        self.publish({"RequestedIP": "192.0.2.30"})
        provider = Mock(return_value={"RequestedIP": "192.0.2.20"})
        runtime.register_restore_env_provider(provider)
        self.barrier([])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.20")
        provider.assert_called_once_with("same-seed")
        runtime.register_restore_env_provider(None)
        self.barrier([])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.30")

    def test_explicit_input_overrides_provider_and_file(self):
        self.path.write_text("not-json", encoding="utf-8")
        provider = Mock(side_effect=AssertionError("explicit input wins"))
        runtime.register_restore_env_provider(provider)
        runtime.fixup_runtime_after_restore(
            "same-seed", Mock(), restore_env={"RequestedIP": "192.0.2.40"}
        )
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.40")
        provider.assert_not_called()

    def test_normal_startup_does_not_read_file(self):
        self.path.write_text("not-json", encoding="utf-8")
        with patch.object(
            scr, "is_scr_template_phase_active", return_value=False
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", return_value=None
        ), patch.object(
            runtime, "_read_restore_env_file"
        ) as reader:
            scr.arrive_scr_template_barrier(worker_id=0, worker_num=1)
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

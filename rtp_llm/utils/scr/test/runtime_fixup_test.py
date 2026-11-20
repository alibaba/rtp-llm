"""Exercise restore input, native bridge and lifecycle order without a GPU."""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from rtp_llm.utils.scr import runtime_fixup as runtime
from rtp_llm.utils.scr import template_utils as scr
from rtp_llm.utils.scr.network import cache_store_advertise_ip
from rtp_llm.utils.scr.lifecycle import CallbackHook, TemplateLifecycle


class ScrRuntimeFixupTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        for context in (
            patch.object(runtime, "RESTORE_ENV_FILE", temp.name + "/absent-envs.json"),
            patch.object(runtime, "_runtime_identity", None),
            patch.object(runtime.socket, "socket"),
            patch.dict(
                os.environ,
                {
                    "RequestedIP": "192.0.2.10",
                    "POD_IP": "192.0.2.10",
                    "RTPLLM_ENABLE_SCR": "1",
                    "SCR_PHASE": "restore",
                },
            ),
            patch.dict(sys.modules, {"libth_transformer": None}),
        ):
            context.start()
            self.addCleanup(context.stop)
        # The helper is an import-time cache; isolate it too when another
        # selected test module has already imported the metrics stack.
        module = sys.modules.get(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper"
        )
        if module is not None:
            for name in (
                "host_ip",
                "container_ip",
                "role",
                "app",
                "group",
                "app_workdir",
            ):
                context = patch.object(
                    module.HippoHelper, name, getattr(module.HippoHelper, name)
                )
                context.start()
                self.addCleanup(context.stop)

    def mock_restore_reader(self, *args, **kwargs):
        reader = patch.object(runtime, "_read_restore_env_file", *args, **kwargs)
        reader.start()
        self.addCleanup(reader.stop)

    def restore_with_environment(self, lifecycle, values):
        with patch.object(runtime, "_read_restore_env_file", return_value=values):
            return runtime.fixup_runtime_after_restore(lifecycle)

    def lifecycle(self, events):
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "component",
            CallbackHook(
                prepare=lambda: events.append("prepare"),
                fixup=lambda g: events.append(("component", os.environ["RequestedIP"])),
                release=lambda: events.append("release"),
                abort=lambda: events.append("abort"),
            ),
        )
        return lifecycle

    def test_real_barrier_wrapper_orders_identity_before_components_and_release(self):
        events = []
        lifecycle = self.lifecycle(events)
        native = NS(refresh_logger_after_scr=lambda ip: events.append(("logger", ip)))
        self.mock_restore_reader(
            side_effect=lambda: events.append("read-env")
            or {"RequestedIP": "192.0.2.20"},
        )
        with patch.dict(sys.modules, {"libth_transformer": native}), patch.object(
            scr, "is_scr_template_phase_active", return_value=True
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr,
            "arrive_scr_checkpoint_barrier",
            side_effect=lambda **kw: events.append("barrier") or 0,
        ):
            self.assertEqual(
                scr.arrive_scr_template_barrier(
                    worker_id=0,
                    worker_num=1,
                ),
                0,
            )
        self.assertEqual(
            events,
            [
                "prepare",
                "barrier",
                "read-env",
                ("logger", "192.0.2.20"),
                ("component", "192.0.2.20"),
                "release",
            ],
        )
        self.assertEqual(os.environ["POD_IP"], "192.0.2.20")

    def test_repeated_restore_rereads_input(self):
        provider = Mock(
            side_effect=[{"RequestedIP": "192.0.2.20"}, {"RequestedIP": "192.0.2.30"}]
        )
        self.mock_restore_reader(provider)
        events = []
        lifecycle = self.lifecycle(events)
        for ip in ("192.0.2.20", "192.0.2.30"):
            lifecycle.prepare_for_template("restore")
            identity = runtime.fixup_runtime_after_restore(lifecycle)
            self.assertEqual(identity.pod_ip, ip)
            lifecycle.release_template()
        self.assertEqual(provider.call_count, 2)

    def test_without_restore_file_ignores_seed_environment_and_previous_snapshot(self):
        lifecycle = self.lifecycle([])
        lifecycle.prepare_for_template("checkpoint")
        with patch.object(
            runtime, "current_pod_ip", side_effect=["192.0.2.20", "192.0.2.30"]
        ):
            for expected in ("192.0.2.20", "192.0.2.30"):
                self.assertEqual(
                    runtime.fixup_runtime_after_restore(lifecycle).pod_ip,
                    expected,
                )

    def test_env_validation_and_discovery_fail_before_mutation_or_component_fixup(self):
        lifecycle = Mock()
        invalid = [
            {"TP_SIZE": "4"},
            {"NCCL_SOCKET_IFNAME": "lo"},
            {"LD_PRELOAD": "/scr/nccl.so"},
            {"SCR_PHASE": "restore"},
            {"RequestedIP": "127.0.0.1"},
            {"RequestedIP": "bad"},
            {"HIPPO_ROLE": "x\ny"},
            {"HIPPO_ROLE": 3},
            {"RequestedIP": "0.0.0.0"},
            {"RequestedIP": "224.0.0.1"},
            {"RequestedIP": "::1"},
            {"RequestedIP": "192.0.2.1\0suffix"},
            {"RequestedIP": ""},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(
                (ValueError, TypeError)
            ):
                self.restore_with_environment(lifecycle, values)
            self.assertEqual(os.environ["RequestedIP"], "192.0.2.10")
            self.assertEqual(os.environ["POD_IP"], "192.0.2.10")
        with patch.object(
            runtime, "current_pod_ip", side_effect=OSError("no route")
        ), self.assertRaises(OSError):
            runtime.fixup_runtime_after_restore(lifecycle)
        lifecycle.restore_fixup.assert_not_called()
        self.assertIsNone(runtime.get_restore_runtime_identity())

    def test_old_loaded_native_library_blocks_release(self):
        events = []
        lifecycle = self.lifecycle(events)
        with patch.dict(sys.modules, {"libth_transformer": NS()}), patch.object(
            runtime, "current_pod_ip", return_value="192.0.2.20"
        ), patch.object(
            scr, "is_scr_template_phase_active", return_value=True
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", return_value=0
        ):
            with self.assertRaisesRegex(RuntimeError, "lacks SCR Logger fixup"):
                scr.arrive_scr_template_barrier(
                    worker_id=0,
                    worker_num=1,
                )
        self.assertEqual(events, ["prepare", "abort"])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.10")

    def test_failed_restore_read_does_not_release(self):
        events = []
        lifecycle = self.lifecycle(events)
        self.mock_restore_reader(side_effect=ValueError("invalid restore data"))
        with patch.object(
            scr, "is_scr_template_phase_active", return_value=True
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", return_value=0
        ):
            with self.assertRaisesRegex(ValueError, "invalid restore data"):
                scr.arrive_scr_template_barrier(
                    worker_id=0,
                    worker_num=1,
                )
        self.assertEqual(events, ["prepare", "abort"])

    def test_failed_barrier_does_not_read_identity_or_release_components(self):
        for error in (
            TimeoutError("barrier timed out"),
            RuntimeError("barrier failed"),
        ):
            with self.subTest(error=type(error).__name__):
                events = []
                lifecycle = self.lifecycle(events)
                provider = Mock(side_effect=AssertionError("must not read identity"))
                self.mock_restore_reader(provider)
                with patch.object(
                    scr, "get_template_lifecycle", return_value=lifecycle
                ), patch.object(
                    scr, "arrive_scr_checkpoint_barrier", side_effect=error
                ):
                    with self.assertRaises(type(error)):
                        scr.arrive_scr_template_barrier(
                            worker_id=0,
                            worker_num=1,
                        )
                self.assertEqual(events, ["prepare", "abort"])
                provider.assert_not_called()
                self.assertIsNone(runtime.get_restore_runtime_identity())

    def test_normal_startup_does_not_read_restore_input(self):
        provider = Mock(side_effect=AssertionError("unexpected read"))
        self.mock_restore_reader(provider)
        with patch.object(
            scr, "is_scr_template_phase_active", return_value=False
        ), patch.object(scr, "arrive_scr_checkpoint_barrier", return_value=None):
            scr.arrive_scr_template_barrier(worker_id=0, worker_num=1)
        provider.assert_not_called()

    def test_real_arrival_failures_abort_even_with_default_fail_closed_argument(self):
        for epsilon in (
            None,
            NS(
                is_snapstart_enable=lambda: True,
                snapstart_checkpoint=Mock(return_value=7),
            ),
            NS(
                is_snapstart_enable=lambda: True,
                snapstart_checkpoint=Mock(side_effect=TimeoutError("expired")),
            ),
        ):
            with self.subTest(epsilon=epsilon):
                events = []
                provider = Mock(return_value={"RequestedIP": "192.0.2.20"})
                self.mock_restore_reader(provider)
                with patch.dict(
                    os.environ,
                    {
                        "RTPLLM_ENABLE_SCR": "1",
                        "SCR_PHASE": "checkpoint",
                    },
                    clear=True,
                ), patch.object(
                    scr, "get_template_lifecycle", return_value=self.lifecycle(events)
                ), patch.object(
                    scr, "_load_epsilon", return_value=epsilon
                ), patch.object(
                    scr, "_capture_cuda_device", return_value=None
                ), patch.object(
                    scr, "_prepare_cuda_for_arrival"
                ):
                    with self.assertRaises(scr.ScrArrivalError):
                        scr.arrive_scr_template_barrier(
                            worker_id=0,
                            worker_num=1,
                        )
                self.assertEqual(events, ["prepare", "abort"])
                provider.assert_not_called()

    def test_real_arrival_accepts_legacy_none_and_zero_success(self):
        for result in (None, 0):
            with self.subTest(result=result):
                events = []
                checkpoint = Mock(return_value=result)
                epsilon = NS(
                    is_snapstart_enable=lambda: True, snapstart_checkpoint=checkpoint
                )
                self.mock_restore_reader(return_value={"RequestedIP": "192.0.2.20"})
                with patch.dict(
                    os.environ,
                    {
                        "RTPLLM_ENABLE_SCR": "1",
                        "SCR_PHASE": "checkpoint",
                    },
                    clear=True,
                ), patch.object(
                    scr, "get_template_lifecycle", return_value=self.lifecycle(events)
                ), patch.object(
                    scr, "_load_epsilon", return_value=epsilon
                ), patch.object(
                    scr, "_capture_cuda_device", return_value=None
                ), patch.object(
                    scr, "_prepare_cuda_for_arrival"
                ):
                    self.assertEqual(
                        scr.arrive_scr_template_barrier(
                            worker_id=0,
                            worker_num=1,
                        ),
                        result,
                    )
                checkpoint.assert_called_once()
                self.assertEqual(
                    events, ["prepare", ("component", "192.0.2.20"), "release"]
                )

    def test_server_config_ip_follows_restored_pod_identity(self):
        configs = NS(server_config=NS(ip="192.0.2.10"))
        observed = []
        lifecycle = TemplateLifecycle()
        lifecycle.register_fixup(
            "server-config", scr._ServerConfigTemplateHook(configs)
        )
        # A hook registered afterwards must observe the refreshed value.
        lifecycle.register(
            "consumer",
            CallbackHook(fixup=lambda _g: observed.append(configs.server_config.ip)),
        )
        lifecycle.prepare_for_template("restore")
        self.restore_with_environment(lifecycle, {"RequestedIP": "192.0.2.20"})
        self.assertEqual(configs.server_config.ip, "192.0.2.20")
        self.assertEqual(observed, ["192.0.2.20"])
        # Repeated restores of one checkpoint must re-read the new Pod IP.
        lifecycle.prepare_for_template("restore")
        self.restore_with_environment(lifecycle, {"RequestedIP": "192.0.2.30"})
        self.assertEqual(configs.server_config.ip, "192.0.2.30")

    def test_native_or_component_fixup_failure_blocks_normal_release(self):
        for fail_native in (True, False):
            with self.subTest(fail_native=fail_native):
                events = []
                lifecycle = self.lifecycle(events)
                refresh = Mock(
                    side_effect=RuntimeError("native failed") if fail_native else None
                )
                if not fail_native:
                    lifecycle.register(
                        "broken",
                        CallbackHook(
                            fixup=Mock(side_effect=RuntimeError("component failed"))
                        ),
                    )
                with patch.dict(
                    sys.modules,
                    {"libth_transformer": NS(refresh_logger_after_scr=refresh)},
                ), patch.object(
                    runtime, "current_pod_ip", return_value="192.0.2.20"
                ), patch.object(
                    scr, "is_scr_template_phase_active", return_value=True
                ), patch.object(
                    scr, "get_template_lifecycle", return_value=lifecycle
                ), patch.object(
                    scr, "arrive_scr_checkpoint_barrier", return_value=0
                ):
                    with self.assertRaises(RuntimeError):
                        scr.arrive_scr_template_barrier(
                            worker_id=0,
                            worker_num=1,
                        )
                self.assertNotIn("release", events)
                self.assertIn("abort", events)

    def test_metrics_and_kv_advertisement_share_explicit_fresh_identity(self):
        from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import (
            HippoHelper,
        )

        cached_names = (
            "host_ip",
            "container_ip",
            "role",
            "app",
            "group",
            "app_workdir",
        )
        for name in cached_names:
            context = patch.object(HippoHelper, name, getattr(HippoHelper, name))
            context.start()
            self.addCleanup(context.stop)
        lifecycle = self.lifecycle([])
        lifecycle.prepare_for_template("restore")
        with patch.dict(
            os.environ,
            {"HIPPO_ROLE_SHORT_NAME": "seed-role"},
        ):
            self.restore_with_environment(
                lifecycle,
                {
                    "RequestedIP": "192.0.2.20",
                    "HIPPO_SLAVE_IP": "192.0.2.200",
                    "HIPPO_ROLE": "restored-role",
                    "HIPPO_ROLE_SHORT_NAME": None,
                },
            )
            with patch(
                "socket.gethostbyname",
                side_effect=AssertionError("must reuse fresh identity"),
            ):
                tags = HippoHelper.get_hippo_tags()
                native = NS(resume_kmonitor_after_scr=Mock(return_value=True))
                with patch.dict(sys.modules, {"libth_transformer": native}):
                    scr._start_native_kmonitor()
                world = NS(
                    num_nodes=1,
                    members=[
                        NS(ip="127.0.0.1", local_rank=i, world_rank=i) for i in range(2)
                    ],
                )
                self.assertEqual(
                    cache_store_advertise_ip(
                        world, NS(world_size=2, local_world_size=2)
                    ),
                    "192.0.2.20",
                )
            self.assertEqual(tags["container_ip"], "192.0.2.20")
            self.assertEqual(tags["host_ip"], "192.0.2.200")
            self.assertEqual(tags["hippo_role"], "restored-role")
            self.assertEqual([m.ip for m in world.members], ["127.0.0.1"] * 2)


if __name__ == "__main__":
    unittest.main()

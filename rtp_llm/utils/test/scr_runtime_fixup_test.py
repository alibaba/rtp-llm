"""Exercise restore input, native bridge and lifecycle order without a GPU."""

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from rtp_llm.utils import scr_runtime_fixup as runtime
from rtp_llm.utils import scr_template_utils as scr
from rtp_llm.utils.scr_local_comm import cache_store_advertise_ip
from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle


class ScrRuntimeFixupTest(unittest.TestCase):
    def setUp(self):
        for context in (
            patch.object(runtime, "_runtime_identity", None),
            patch.object(runtime, "_restore_env_provider", None),
            patch.dict(os.environ, {"RequestedIP": "192.0.2.10"}),
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

    def lifecycle(self, events):
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "component",
            CallbackHook(
                prepare=lambda g: events.append("prepare"),
                fixup=lambda g: events.append(("component", os.environ["RequestedIP"])),
                release=lambda g: events.append("release"),
                abort=lambda g: events.append("abort"),
            ),
        )
        return lifecycle

    def test_real_barrier_wrapper_orders_identity_before_components_and_release(self):
        events = []
        lifecycle = self.lifecycle(events)
        native = NS(refresh_logger_after_scr=lambda ip: events.append(("logger", ip)))
        runtime.register_restore_env_provider(
            lambda g: events.append(("read-env", g)) or {"RequestedIP": "192.0.2.20"}
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
                    worker_id=0, worker_num=1, generation="g1"
                ),
                0,
            )
        self.assertEqual(
            events,
            [
                "prepare",
                "barrier",
                ("read-env", "g1"),
                ("logger", "192.0.2.20"),
                ("component", "192.0.2.20"),
                "release",
            ],
        )

    def test_repeated_restore_rereads_provider_even_with_same_checkpoint_generation(
        self,
    ):
        provider = Mock(
            side_effect=[{"RequestedIP": "192.0.2.20"}, {"RequestedIP": "192.0.2.30"}]
        )
        runtime.register_restore_env_provider(provider)
        events = []
        lifecycle = self.lifecycle(events)
        for ip in ("192.0.2.20", "192.0.2.30"):
            lifecycle.prepare_for_template("same-seed", "restore")
            identity = runtime.fixup_runtime_after_restore("same-seed", lifecycle)
            self.assertEqual(identity.pod_ip, ip)
            lifecycle.release_template("same-seed")
        self.assertEqual(provider.call_count, 2)

    def test_without_provider_ignores_seed_environment_and_previous_snapshot(self):
        lifecycle = self.lifecycle([])
        lifecycle.prepare_for_template("g1", "checkpoint")
        with patch.object(
            runtime, "current_pod_ip", side_effect=["192.0.2.20", "192.0.2.30"]
        ):
            for expected in ("192.0.2.20", "192.0.2.30"):
                self.assertEqual(
                    runtime.fixup_runtime_after_restore("g1", lifecycle).pod_ip,
                    expected,
                )

    def test_env_validation_and_discovery_fail_before_mutation_or_component_fixup(self):
        lifecycle = Mock()
        invalid = [
            {"TP_SIZE": "4"},
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
                runtime.fixup_runtime_after_restore("g1", lifecycle, restore_env=values)
            self.assertEqual(os.environ["RequestedIP"], "192.0.2.10")
        with patch.object(
            runtime, "current_pod_ip", side_effect=OSError("no route")
        ), self.assertRaises(OSError):
            runtime.fixup_runtime_after_restore("g1", lifecycle)
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
                    worker_id=0, worker_num=1, generation="g1"
                )
        self.assertEqual(events, ["prepare", "abort"])
        self.assertEqual(os.environ["RequestedIP"], "192.0.2.10")

    def test_failed_provider_does_not_release(self):
        events = []
        lifecycle = self.lifecycle(events)
        runtime.register_restore_env_provider(
            Mock(side_effect=ValueError("stale generation"))
        )
        with patch.object(
            scr, "is_scr_template_phase_active", return_value=True
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", return_value=0
        ):
            with self.assertRaisesRegex(ValueError, "stale generation"):
                scr.arrive_scr_template_barrier(
                    worker_id=0, worker_num=1, generation="g1"
                )
        self.assertEqual(events, ["prepare", "abort"])

    def test_normal_startup_does_not_read_restore_input(self):
        provider = Mock(side_effect=AssertionError("unexpected read"))
        runtime.register_restore_env_provider(provider)
        with patch.object(
            scr, "is_scr_template_phase_active", return_value=False
        ), patch.object(scr, "arrive_scr_checkpoint_barrier", return_value=None):
            scr.arrive_scr_template_barrier(worker_id=0, worker_num=1)
        provider.assert_not_called()

    def test_frontend_identity_updates_outside_loopback_mode(self):
        visitor = Mock(source_ip="192.0.2.10")
        configs = NS(
            server_config=NS(),
            distribute_config=NS(),
            parallelism_config=NS(),
            role_config=NS(role_type="PREFILL"),
        )
        endpoint_module = NS(resolve_world_info=Mock(return_value=NS()))
        distributed_module = NS(
            get_world_info=Mock(return_value=NS(num_nodes=1)),
            get_dp_addrs_from_world_info=Mock(return_value=["192.0.2.40:9000"]),
        )
        lifecycle = TemplateLifecycle()
        lifecycle.register("visitor", scr._BackendVisitorTemplateHook(visitor, configs))
        lifecycle.prepare_for_template("g1", "restore")
        with patch.dict(
            os.environ, {"RTP_LLM_SCR_LOCAL_COMM": "0", "SCR_PHASE": "restore"}
        ), patch.dict(
            sys.modules,
            {
                "rtp_llm.distribute.distributed_server": distributed_module,
                "rtp_llm.utils.scr_endpoint_provider": endpoint_module,
            },
        ):
            runtime.fixup_runtime_after_restore(
                "g1", lifecycle, restore_env={"RequestedIP": "192.0.2.20"}
            )
        self.assertEqual(visitor.source_ip, "192.0.2.20")
        visitor.update_addresses.assert_called_once_with(["192.0.2.40:9000"])
        self.assertTrue(
            endpoint_module.resolve_world_info.call_args.kwargs["require_manifest"]
        )

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
                            worker_id=0, worker_num=1, generation="g1"
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
        lifecycle.prepare_for_template("g1", "restore")
        with patch.dict(
            os.environ,
            {"HIPPO_ROLE_SHORT_NAME": "seed-role", "RTP_LLM_SCR_LOCAL_COMM": "1"},
        ):
            runtime.fixup_runtime_after_restore(
                "g1",
                lifecycle,
                restore_env={
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
                tags = HippoHelper.refresh_runtime_identity()
                native = NS(resume_kmonitor_after_scr=Mock(return_value=True))
                with patch.dict(sys.modules, {"libth_transformer": native}):
                    hook = scr._NativeKmonitorTemplateHook()
                    hook._paused = True
                    hook.release_template("g1")
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

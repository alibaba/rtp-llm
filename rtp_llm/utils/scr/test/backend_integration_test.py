"""Unit tests for rank-local Epsilon registration and barrier arrival.

The RTP-LLM process never invokes the SCR controller. Dump and restore are
initiated by the external control plane; a backend rank only registers state
and announces its Epsilon barrier arrival.
"""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import rtp_llm.start_backend_server as backend
import rtp_llm.start_server as launcher
from rtp_llm.ops import RoleType, VitSeparation


class BackendScrIntegrationTest(unittest.TestCase):
    def test_manager_arrives_before_waiting_for_rank_readiness(self):
        config = self._launcher_config(world_size=2)
        config.distribute_config = SimpleNamespace(fake_gang_env=False)
        config.server_config.shutdown_timeout = 1
        config.server_config.monitor_interval = 1
        order = mock.Mock()
        manifest = self._manifest()
        with mock.patch.object(backend.multiprocessing, "get_context"), mock.patch.object(
            backend, "_create_rank_processes", order.spawn
        ), mock.patch.object(
            backend, "_start_scr_manager_arrival", order.arrive
        ), mock.patch.object(
            backend, "_wait_for_ranks_startup", order.wait
        ), mock.patch.object(
            backend, "_send_pipe_status", order.status
        ), mock.patch.object(
            backend, "ProcessManager"
        ):
            backend.multi_rank_start(None, config, scr_manifest=manifest)
        self.assertEqual([call[0] for call in order.mock_calls[:4]], ["spawn", "arrive", "wait", "status"])
        order.arrive.assert_called_once_with(manifest)

    def test_rank_starts_services_only_after_template_release(self):
        from rtp_llm.utils.scr import template_utils as scr
        from rtp_llm.utils.scr.lifecycle import CallbackHook, TemplateLifecycle
        from rtp_llm.utils.scr.restore_context import RestoreContext

        for phase, failing_step in (
            ("normal", None),
            ("checkpoint", None),
            ("restore", None),
            ("restore", "checkpoint"),
            ("restore", "fixup"),
        ):
            with self.subTest(
                phase=phase, failing_step=failing_step
            ), ExitStack() as stack:
                order = []
                config = self._config(local_rank=0, world_rank=0)
                config.parallelism_config.world_size = 1
                config.prefill_cp_config = None
                config.moe_config = object()
                config.deep_ep_config = object()
                config.server_config = mock.Mock()
                config.distribute_config = mock.Mock()
                manager = mock.Mock()
                manager.start.side_effect = lambda: order.append("prepare-engine")
                manager.start_service.side_effect = lambda: order.append(
                    "start-services"
                )
                manager.serve_forever.side_effect = lambda: order.append("serve")
                pipe = mock.Mock()
                pipe.send.side_effect = lambda status: order.append(status["status"])

                lifecycle = TemplateLifecycle()
                lifecycle.register(
                    "test-services",
                    CallbackHook(
                        prepare=lambda: order.append("prepare-template"),
                        release=lambda: order.append("release"),
                        abort=lambda: order.append("abort"),
                    ),
                )

                def checkpoint(**kwargs):
                    manager.start_service.assert_not_called()
                    pipe.send.assert_not_called()
                    order.append("checkpoint")
                    if failing_step == "checkpoint":
                        raise TimeoutError("checkpoint failed")
                    return 0

                def fixup(lifecycle):
                    order.append("fixup")
                    if failing_step == "fixup":
                        raise ValueError("fixup failed")
                    lifecycle.restore_fixup(RestoreContext("192.0.2.20"))

                stack.enter_context(
                    mock.patch.dict(
                        os.environ,
                        {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": phase},
                        clear=True,
                    )
                )
                for name in (
                    "_install_hot_hook_runtime",
                    "copy_gemm_config",
                    "set_parallelism_config",
                    "finalize_rank_local_deepep_internode",
                    "configure_kv_cache_event_host_ip_port",
                    "setup_cuda_device_and_accl_env",
                    "configure_kernel_tuning",
                    "set_global_controller",
                    "install_oom_dump",
                    "ignore_sigpipe",
                ):
                    stack.enter_context(mock.patch.object(backend, name))
                stack.enter_context(mock.patch.object(backend.signal, "signal"))
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        {
                            "rtp_llm.server.backend_manager": SimpleNamespace(
                                BackendManager=mock.Mock(return_value=manager)
                            )
                        },
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        backend,
                        "register_for_scr",
                        side_effect=lambda *args, **kwargs: order.append("register-kv")
                        or True,
                    )
                )
                stack.enter_context(mock.patch.object(scr, "ignore_sigpipe"))
                stack.enter_context(
                    mock.patch.object(
                        scr, "get_template_lifecycle", return_value=lifecycle
                    )
                )
                arrival = mock.Mock(side_effect=checkpoint)
                epsilon = SimpleNamespace(
                    is_snapstart_enable=lambda: True, snapstart_checkpoint=arrival
                )
                load_epsilon = stack.enter_context(
                    mock.patch.object(scr, "_load_epsilon", return_value=epsilon)
                )
                stack.enter_context(mock.patch.object(scr, "_capture_cuda_device"))
                stack.enter_context(mock.patch.object(scr, "_prepare_cuda_for_arrival"))
                stack.enter_context(
                    mock.patch.object(
                        scr, "fixup_runtime_after_restore", side_effect=fixup
                    )
                )

                if failing_step is not None:
                    error_type = (
                        scr.ScrArrivalError
                        if failing_step == "checkpoint"
                        else ValueError
                    )
                    with self.assertRaisesRegex(error_type, f"{failing_step} failed"):
                        backend.local_rank_start(
                            None,
                            config,
                            pipe_writer=pipe,
                            scr_manifest=self._manifest(),
                        )
                    self.assertEqual(
                        order,
                        [
                            "prepare-engine",
                            "register-kv",
                            "prepare-template",
                            "checkpoint",
                            *(["fixup"] if failing_step == "fixup" else []),
                            "abort",
                            "failed",
                        ],
                    )
                    manager.start_service.assert_not_called()
                    manager.serve_forever.assert_not_called()
                else:
                    backend.local_rank_start(
                        None,
                        config,
                        pipe_writer=pipe,
                        scr_manifest=self._manifest() if phase != "normal" else None,
                    )
                    expected = ["prepare-engine"]
                    if phase != "normal":
                        expected.extend(
                            [
                                "register-kv",
                                "prepare-template",
                                "checkpoint",
                                "fixup",
                                "release",
                            ]
                        )
                    self.assertEqual(
                        order, [*expected, "start-services", "success", "serve"]
                    )
                if phase == "normal":
                    arrival.assert_not_called()
                    load_epsilon.assert_not_called()
                else:
                    arrival.assert_called_once()
                    self.assertEqual(arrival.call_args.kwargs["wait_mode"], 1)
                    self.assertEqual(arrival.call_args.kwargs["worker_id"], 0)
                    self.assertEqual(arrival.call_args.kwargs["worker_num"], 4)
                self.assertFalse(lifecycle.active)

    def test_parent_arrival_runs_template_lifecycle_hooks(self):
        from rtp_llm.utils.scr import template_utils as scr
        from rtp_llm.utils.scr.lifecycle import CallbackHook, TemplateLifecycle

        order = []
        lifecycle = TemplateLifecycle()
        lifecycle.register(
            "parent-reporter",
            CallbackHook(
                prepare=lambda: order.append("prepare"),
                fixup=lambda context: order.append("fixup"),
                release=lambda: order.append("release"),
            ),
        )
        manifest = SimpleNamespace(
            worker_id=lambda role, instance: 0,
            worker_num=1,
        )
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
        ), mock.patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), mock.patch.object(
            scr,
            "arrive_scr_checkpoint_barrier",
            side_effect=lambda **kwargs: order.append("barrier") or 0,
        ):
            self.assertEqual(launcher._start_parent_scr_arrival(manifest), 0)
        self.assertEqual(order, ["prepare", "barrier", "fixup", "release"])
        self.assertFalse(lifecycle.active)

    def _config(self, local_rank=1, world_rank=5):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(
                local_rank=local_rank,
                world_rank=world_rank,
                local_world_size=4,
            ),
            kernel_tuning_config=SimpleNamespace(
                aiter_fmoe_config_paths="",
                rocm_fp8_moe_deterministic_reduce=False,
            ),
        )

    def _launcher_config(
        self,
        *,
        role_type=RoleType.PREFILL,
        vit_separation=VitSeparation.VIT_SEPARATION_LOCAL,
        world_size=1,
        world_rank=0,
        tp_size=1,
        frontend_server_count=1,
    ):
        return SimpleNamespace(
            role_config=SimpleNamespace(role_type=role_type),
            vit_config=SimpleNamespace(vit_separation=vit_separation),
            parallelism_config=SimpleNamespace(
                world_size=world_size,
                world_rank=world_rank,
                tp_size=tp_size,
                pp_size=1,
                dp_size=world_size,
                local_world_size=1,
            ),
            server_config=SimpleNamespace(
                frontend_server_count=frontend_server_count,
            ),
        )

    @staticmethod
    def _manifest(worker_num=4):
        return SimpleNamespace(
            worker_id=lambda role, instance: int(instance),
            worker_num=worker_num,
            participant_ids={f"backend_rank:{i}": i for i in range(worker_num)},
        )

    def test_disabled_path_is_inert(self):
        manager = SimpleNamespace(engine=object())
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            backend, "register_for_scr"
        ) as register:
            self.assertIsNone(backend._register_scr_resources(manager, self._config()))
        register.assert_not_called()

    def test_registration_does_not_start_checkpoint_arrival(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch.object(
            backend, "register_for_scr", return_value=True
        ) as register:
            self.assertIs(backend._register_scr_resources(manager, config), manager.engine)

        register.assert_called_once_with(manager.engine, local_rank=2)

    def test_rank_arrival_uses_world_rank_from_the_manifest(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        manifest = self._manifest()
        manifest.worker_id = mock.Mock(return_value=2)
        manifest.participant_ids["backend_rank:7"] = 2
        result = 0
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "LOCAL_WORLD_SIZE": "4",
            },
            clear=True,
        ), mock.patch.object(
            backend, "arrive_scr_checkpoint_barrier", return_value=result
        ) as arrive:
            self.assertEqual(
                backend._start_scr_rank_arrival(manager, config, scr_manifest=manifest),
                result,
            )

        manifest.worker_id.assert_called_once_with("backend_rank", "7")
        arrive.assert_called_once_with(
            worker_id=2,
            worker_num=4,
            fail_closed=True,
        )

    def test_active_arrival_failure_is_not_silent(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=0, world_rank=0)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch.object(
            backend,
            "arrive_scr_checkpoint_barrier",
            side_effect=RuntimeError("arrival timeout"),
        ):
            with self.assertRaisesRegex(RuntimeError, "arrival timeout"):
                backend._start_scr_rank_arrival(
                    manager, config, scr_manifest=self._manifest()
                )

    def test_active_arrival_requests_fail_closed_mode(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=0, world_rank=0)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch.object(
            backend, "arrive_scr_checkpoint_barrier", return_value=0
        ) as arrive:
            self.assertEqual(
                backend._start_scr_rank_arrival(
                    manager, config, scr_manifest=self._manifest()
                ),
                0,
            )

        arrive.assert_called_once_with(
            worker_id=0,
            worker_num=4,
            fail_closed=True,
        )

    def test_registration_failure_remains_fail_open(self):
        manager = SimpleNamespace(engine=object())
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch.object(
            backend, "register_for_scr", return_value=False
        ):
            self.assertIs(backend._register_scr_resources(manager, self._config()), manager.engine)

    def test_manifest_is_inert_when_external_phase_is_inactive(self):
        config = self._launcher_config()
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "normal"},
            clear=True,
        ):
            self.assertIsNone(launcher._build_scr_participant_manifest(config))

    def test_frontend_role_does_not_add_phantom_vit_participant(self):
        config = self._launcher_config(
            role_type=RoleType.FRONTEND,
            vit_separation=VitSeparation.VIT_SEPARATION_ROLE,
        )
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ):
            manifest = launcher._build_scr_participant_manifest(config)

        self.assertIsNotNone(manifest)
        self.assertNotIn("backend_vit:0", manifest.participant_ids)
        self.assertNotIn("backend_rank:0", manifest.participant_ids)

    def test_vit_role_stays_outside_the_template_quorum(self):
        config = self._launcher_config(
            role_type=RoleType.VIT,
            vit_separation=VitSeparation.VIT_SEPARATION_ROLE,
        )
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ):
            manifest = launcher._build_scr_participant_manifest(config)

        self.assertIsNone(manifest)

    def test_pd_rank_with_separate_vit_uses_its_nonzero_world_rank(self):
        for role in (RoleType.PREFILL, RoleType.DECODE):
            config = self._launcher_config(
                role_type=role, world_size=4, world_rank=2,
                vit_separation=VitSeparation.VIT_SEPARATION_ROLE,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "RTPLLM_ENABLE_SCR": "1",
                    "SCR_PHASE": "checkpoint",
                    "LOCAL_WORLD_SIZE": "1",
                },
                clear=True,
            ):
                manifest = launcher._build_scr_participant_manifest(config)
            self.assertEqual(
                set(manifest.participant_ids),
                {"start_server:0", "backend_rank:2", "frontend:2:0", "dash_sc:2:0"},
            )

    def test_vit_children_do_not_participate_in_scr(self):
        scr_env = {
            "RTPLLM_ENABLE_SCR": "1",
            "SCR_PHASE": "checkpoint",
            "UNRELATED": "keep",
        }
        seen = {}

        def fake_vit_entry(*args):
            seen["env"] = dict(os.environ)
            return "started"

        with mock.patch.dict(os.environ, scr_env, clear=True):
            with launcher._without_scr():
                result = fake_vit_entry(0, "cfg")
            self.assertEqual(dict(os.environ), scr_env)

        self.assertEqual(result, "started")
        expected = dict(scr_env)
        del expected["RTPLLM_ENABLE_SCR"]
        self.assertEqual(seen["env"], expected)

    def test_manifest_uses_the_resolved_pod_rank_range(self):
        config = self._launcher_config(world_size=8, world_rank=4)
        config.parallelism_config.local_world_size = 4
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "CUDA_VISIBLE_DEVICES": "0,1,2",
            },
            clear=True,
        ):
            manifest = launcher._build_scr_participant_manifest(config)
            self.assertEqual(
                list(launcher._iter_serving_world_ranks(config)), [4, 5, 6, 7]
            )
        self.assertEqual(
            {
                key
                for key in manifest.participant_ids
                if key.startswith("backend_rank:")
            },
            {"backend_rank:4", "backend_rank:5", "backend_rank:6", "backend_rank:7"},
        )

    def test_manifest_rejects_misaligned_backend_ranks(self):
        config = self._launcher_config(world_size=12, world_rank=4)
        config.parallelism_config.local_world_size = 3
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "aligned Pod topology"):
            launcher._build_scr_participant_manifest(config)

    def test_template_phase_does_not_arm_in_place_restart(self):
        from rtp_llm.config.py_config_modules import PyEnvConfigs

        configs = PyEnvConfigs()
        configs.role_config.role_type = RoleType.PREFILL
        configs.server_config.engine_restart_limit = 2
        for phase, expected in (
            ("checkpoint", False),
            ("restore", False),
            ("normal", True),
        ):
            with self.subTest(phase=phase), mock.patch.dict(
                os.environ,
                {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": phase},
                clear=True,
            ):
                self.assertIs(launcher._engine_restart_enabled(configs), expected)

    def test_restarted_backend_generation_starts_outside_the_template(self):
        from rtp_llm.utils.backend_restart import RestartOutcome
        from rtp_llm.utils.drain_flag import DrainFlag

        config = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_size=1, local_world_size=1, world_rank=0
            ),
            server_config=SimpleNamespace(
                engine_restart_limit=1,
                engine_restart_heartbeat_timeout_s=10,
                startup_health_check_timeout=30,
            ),
        )
        scr_env = {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "restore"}
        seen = []

        def spawn_backend(*args, **kwargs):
            seen.append(
                (
                    os.environ.get("RTPLLM_ENABLE_SCR"),
                    kwargs.get("scr_manifest", args[3] if len(args) > 3 else None),
                )
            )
            return mock.Mock(), (lambda: True), {"max_seq_len": 4096}

        with mock.patch.dict(os.environ, scr_env, clear=True), mock.patch.object(
            launcher, "_spawn_backend_process", side_effect=spawn_backend
        ), mock.patch.object(launcher, "_maybe_run_startup_real_warmup"):
            restarter = launcher._create_backend_restarter(
                None, config, mock.Mock(shutdown_requested=False), None, DrainFlag()
            )
            self.addCleanup(restarter.coordinator.close)
            outcome, _ = restarter._start_generation()
            self.assertEqual(outcome, RestartOutcome.RESTARTED)
            # The launcher keeps its own phase; only the child starts outside it.
            self.assertEqual(dict(os.environ), scr_env)

        self.assertEqual(seen, [(None, None)])


if __name__ == "__main__":
    unittest.main()

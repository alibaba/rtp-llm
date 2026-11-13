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
    def test_rank_starts_services_only_after_template_release(self):
        from rtp_llm.utils.scr import template_utils as scr
        from rtp_llm.utils.scr.lifecycle import CallbackHook, TemplateLifecycle

        for phase, failing_step in (
            ("normal", None),
            ("checkpoint", None),
            ("restore", None),
            ("restore", "checkpoint"),
            ("restore", "fixup"),
        ):
            with self.subTest(phase=phase, failing_step=failing_step), ExitStack() as stack:
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
                        prepare=lambda generation: order.append("prepare-template"),
                        release=lambda generation: order.append("release"),
                        abort=lambda generation: order.append("abort"),
                    ),
                )

                def checkpoint(**kwargs):
                    manager.start_service.assert_not_called()
                    pipe.send.assert_not_called()
                    order.append("checkpoint")
                    if failing_step == "checkpoint":
                        raise TimeoutError("checkpoint failed")
                    return 0

                def fixup(generation, lifecycle):
                    order.append("fixup")
                    if failing_step == "fixup":
                        raise ValueError("fixup failed")
                    lifecycle.restore_fixup(SimpleNamespace(generation=generation))

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
                    mock.patch.object(
                        scr, "_load_epsilon", return_value=epsilon
                    )
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
                prepare=lambda generation: order.append("prepare"),
                fixup=lambda generation: order.append("fixup"),
                release=lambda generation: order.append("release"),
            ),
        )
        manifest = SimpleNamespace(
            worker_id=lambda role, instance: 0,
            worker_num=1,
            generation="parent-generation",
        )
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
        ), mock.patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), mock.patch.object(
            scr, "arrive_scr_checkpoint_barrier",
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
            )
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
            generation="",
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
        self.assertFalse(hasattr(manager, "_scr_checkpoint_arrival"))

    def test_rank_arrival_uses_local_rank_and_local_world_size(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
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
                backend._start_scr_rank_arrival(
                    manager, config, scr_manifest=self._manifest()
                ),
                result,
            )

        arrive.assert_called_once_with(
            worker_id=2,
            worker_num=4,
            generation=None,
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
            generation=None,
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
        ), mock.patch.object(launcher, "_visible_device_count", return_value=1):
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
        ), mock.patch.object(launcher, "_visible_device_count", return_value=1):
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
        ), mock.patch.object(launcher, "_visible_device_count", return_value=1):
            manifest = launcher._build_scr_participant_manifest(config)

        self.assertIsNotNone(manifest)
        # VIT does not arrive at the barrier and does not restore runtime state,
        # so it must not appear as a participant that the quorum waits for.
        self.assertNotIn("backend_vit:0", manifest.participant_ids)
        self.assertNotIn("backend_rank:0", manifest.participant_ids)
        self.assertNotIn("backend_manager:0", manifest.participant_ids)

    def test_vit_children_run_without_the_scr_environment(self):
        scr_env = {
            "RTPLLM_ENABLE_SCR": "1",
            "SCR_PHASE": "checkpoint",
            "SCR_GENERATION": "3",
            "RTP_LLM_SCR_GENERATION": "3",
            "RTP_LLM_SCR_TIMEOUT": "900",
            "RTP_LLM_SCR_INACTIVITY_TIMEOUT": "10",
            "RTP_LLM_SCR_RESTORE_START_EPOCH_MS": "1",
            "UNRELATED": "keep",
        }
        seen = {}

        def fake_vit_entry(*args):
            seen["env"] = dict(os.environ)
            return "started"

        with mock.patch.dict(os.environ, scr_env, clear=True):
            result = launcher._run_vit_without_scr(fake_vit_entry, 0, "cfg")

        self.assertEqual(result, "started")
        self.assertEqual(seen["env"], {"UNRELATED": "keep"})

    def test_manifest_uses_visible_device_count_when_local_world_size_is_unset(self):
        config = self._launcher_config(world_size=4)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch.object(launcher, "_visible_device_count", return_value=1):
            manifest = launcher._build_scr_participant_manifest(config)

        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.worker_num, 4)
        self.assertIn("backend_rank:0", manifest.participant_ids)
        self.assertNotIn("backend_rank:1", manifest.participant_ids)
        self.assertNotIn("frontend:1:0", manifest.participant_ids)
        self.assertNotIn("dash_sc:1:0", manifest.participant_ids)

    def test_manifest_keeps_explicit_local_world_size_in_sync_with_rank_dispatch(self):
        config = self._launcher_config(world_size=4)
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "LOCAL_WORLD_SIZE": "4",
            },
            clear=True,
        ), mock.patch.object(launcher, "_visible_device_count", return_value=1):
            self.assertEqual(launcher._scr_local_world_size(config), 4)


if __name__ == "__main__":
    unittest.main()

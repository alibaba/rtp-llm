"""Unit tests for rank-local Epsilon registration and barrier arrival.

The RTP-LLM process never invokes the SCR controller. Dump and restore are
initiated by the external control plane; a backend rank only registers state
and announces its Epsilon barrier arrival.
"""

from types import SimpleNamespace
import os
import unittest
from unittest import mock

import rtp_llm.start_backend_server as backend
import rtp_llm.start_server as launcher
from rtp_llm.ops import RoleType, VitSeparation


class BackendScrIntegrationTest(unittest.TestCase):
    def test_parent_arrival_runs_template_lifecycle_hooks(self):
        from rtp_llm.utils import scr_template_utils as scr
        from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle

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
                local_rank=local_rank, world_rank=world_rank
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
            ),
            server_config=SimpleNamespace(
                frontend_server_count=frontend_server_count,
            ),
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

        register.assert_called_once_with(manager.engine, rank=7, local_rank=2)
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
            self.assertEqual(backend._start_scr_rank_arrival(manager, config), result)

        arrive.assert_called_once_with(
            worker_id=2,
            worker_num=4,
            generation=None,
            fail_closed=True,
        )

    def test_rank_arrival_honors_shared_scheduler_mapping(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        result = 0
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "RTP_LLM_SCR_WORKER_OFFSET": "4",
                "RTP_LLM_SCR_WORKER_NUM": "8",
            },
            clear=True,
        ), mock.patch.object(
            backend, "arrive_scr_checkpoint_barrier", return_value=result
        ) as arrive:
            self.assertEqual(backend._start_scr_rank_arrival(manager, config), result)

        arrive.assert_called_once_with(
            worker_id=6,
            worker_num=8,
            generation=None,
            fail_closed=True,
        )

    def test_rank_arrival_skips_invalid_shared_scheduler_mapping(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "RTP_LLM_SCR_WORKER_OFFSET": "4",
                "RTP_LLM_SCR_WORKER_NUM": "4",
            },
            clear=True,
        ), mock.patch.object(
            backend, "arrive_scr_checkpoint_barrier"
        ) as start:
            with self.assertRaisesRegex(ValueError, "worker_id must be in"):
                backend._start_scr_rank_arrival(manager, config)

        start.assert_not_called()

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
                backend._start_scr_rank_arrival(manager, config)

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
            self.assertEqual(backend._start_scr_rank_arrival(manager, config), 0)

        arrive.assert_called_once_with(
            worker_id=0,
            worker_num=1,
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
        ), mock.patch("torch.cuda.device_count", return_value=1):
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
        ), mock.patch("torch.cuda.device_count", return_value=1):
            manifest = launcher._build_scr_participant_manifest(config)

        self.assertIsNotNone(manifest)
        self.assertNotIn("backend_vit:0", manifest.participant_ids)
        self.assertNotIn("backend_rank:0", manifest.participant_ids)

    def test_manifest_uses_visible_device_count_when_local_world_size_is_unset(self):
        config = self._launcher_config(world_size=4)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
            clear=True,
        ), mock.patch("torch.cuda.device_count", return_value=1):
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
        ), mock.patch("torch.cuda.device_count", return_value=1):
            self.assertEqual(launcher._scr_local_world_size(config), 4)


if __name__ == "__main__":
    unittest.main()

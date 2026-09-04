"""Unit tests for the rank-local Epsilon startup wiring."""

from types import SimpleNamespace
import os
import unittest
from unittest import mock

import rtp_llm.start_backend_server as backend


class BackendScrIntegrationTest(unittest.TestCase):
    def _config(self, local_rank=1, world_rank=5):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(
                local_rank=local_rank, world_rank=world_rank
            )
        )

    def test_disabled_path_is_inert(self):
        manager = SimpleNamespace(engine=object())
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            backend, "register_for_scr"
        ) as register, mock.patch.object(
            backend, "start_scr_checkpoint_thread"
        ) as start:
            self.assertIsNone(backend._setup_scr_worker(manager, self._config()))
            self.assertIsNone(backend._start_scr_worker_waiter(manager))
        register.assert_not_called()
        start.assert_not_called()

    def test_registration_precedes_nonjoined_waiter_and_honors_scope_offset(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        with mock.patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "RTP_LLM_SCR_WORKER_OFFSET": "4"},
            clear=True,
        ), mock.patch.object(
            backend, "register_for_scr", return_value=True
        ) as register, mock.patch.object(
            backend, "start_scr_checkpoint_thread", return_value="waiter"
        ) as start:
            self.assertIs(backend._setup_scr_worker(manager, config), manager.engine)
            self.assertEqual(getattr(manager, "_scr_worker_id"), 6)
            self.assertEqual(backend._start_scr_worker_waiter(manager), "waiter")

        register.assert_called_once_with(manager.engine, rank=7, local_rank=2)
        start.assert_called_once_with(
            manager=manager, engine=manager.engine, worker_id=6
        )
        self.assertEqual(manager._scr_checkpoint_waiter, "waiter")


if __name__ == "__main__":
    unittest.main()

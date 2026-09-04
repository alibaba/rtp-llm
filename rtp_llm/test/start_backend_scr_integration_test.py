"""Unit tests for rank-local Epsilon registration.

The RTP-LLM process does not start a checkpoint waiter or invoke the SCR
controller.  Dump and restore are initiated by the external control plane.
"""

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
        ) as register:
            self.assertIsNone(backend._register_scr_resources(manager, self._config()))
        register.assert_not_called()

    def test_registration_does_not_start_a_checkpoint_waiter(self):
        manager = SimpleNamespace(engine=object())
        config = self._config(local_rank=2, world_rank=7)
        with mock.patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "1"}, clear=True
        ), mock.patch.object(
            backend, "register_for_scr", return_value=True
        ) as register:
            self.assertIs(backend._register_scr_resources(manager, config), manager.engine)

        register.assert_called_once_with(manager.engine, rank=7, local_rank=2)
        self.assertFalse(hasattr(manager, "_scr_checkpoint_waiter"))
        self.assertFalse(hasattr(backend, "_start_scr_worker_waiter"))

    def test_registration_failure_remains_fail_open(self):
        manager = SimpleNamespace(engine=object())
        with mock.patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "1"}, clear=True
        ), mock.patch.object(
            backend, "register_for_scr", return_value=False
        ):
            self.assertIs(backend._register_scr_resources(manager, self._config()), manager.engine)


if __name__ == "__main__":
    unittest.main()

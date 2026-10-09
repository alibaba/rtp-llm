"""Counterexamples for offline batch evidence and owned master recovery."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from runtime.engine_ops import EngineOps
from scenario.actions.master import (
    decode_residual_bound,
    owner_clean,
)


class ExecutionEvidenceTests(unittest.TestCase):


    def test_bound_tracks_client_configuration_and_owner_stays_strict(self):
        for n in [1, 4, 16]:
            client = NS(flow=NS(_overrides={"MAX_CONCURRENCY": str(n)}))
            b = decode_residual_bound(client)
            self.assertEqual(n, b)
            self.assertTrue(owner_clean(0, {"prefill": [1, 0], "decode": [b, 0]}, 1, b))
            self.assertFalse(
                owner_clean(0, {"prefill": [1, 0], "decode": [b + 1, 0]}, 1, b)
            )
            self.assertFalse(
                owner_clean(1, {"prefill": [0, 0], "decode": [0, 0]}, 1, b)
            )

    def test_channel_invalidation_is_targeted_idempotent_and_recreates(self):
        ops = EngineOps.__new__(EngineOps)
        old, engine, new = Mock(), Mock(), Mock()
        ops._channels = {"master": old, "engine": engine}
        ops.invalidate_channel("master")
        ops.invalidate_channel("master")
        old.close.assert_called_once()
        engine.close.assert_not_called()
        with patch(
            "runtime.engine_ops.grpc.insecure_channel", return_value=new
        ):
            self.assertIs(new, ops._channel("master"))
        self.assertIs(engine, ops._channel("engine"))

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    def test_speculative_reserve_matches_engine(self):
        mtp = self._configs(SpeculativeType.MTP, 3)
        dspark = self._configs(SpeculativeType.DSPARK, 3)

        self.assertEqual(
            start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
            4,
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_speculative_reserve_step(
                dspark
            ),
            8,
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048568,
        )

    def test_warmup_failure_propagates_to_health_gate_owner(self):
        configs = self._configs(SpeculativeType.DSPARK, 3)
        with (
            patch.object(
                start_server, "_should_run_startup_real_warmup", return_value=True
            ),
            patch.object(
                start_server,
                "_run_startup_real_warmup_grpc",
                new=lambda _configs: object(),
            ),
            patch.object(
                start_server,
                "_run_startup_real_warmup_async",
                side_effect=RuntimeError("warmup failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                start_server._maybe_run_startup_real_warmup(configs)


if __name__ == "__main__":
    unittest.main()

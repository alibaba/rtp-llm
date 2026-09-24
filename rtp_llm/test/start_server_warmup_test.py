import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                7,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048567,
        )

    def test_request_leaves_output_space_after_speculative_reserve(self):
        for reserve in (0, 4, 7, 9):
            with self.subTest(reserve=reserve):
                length = start_server._get_startup_real_warmup_request_token_len(
                    1048576, 1048576, reserve
                )
                self.assertEqual(1048576 - reserve - length, 1)
                self.assertEqual(
                    start_server._get_startup_real_warmup_request_token_len(
                        8, 32, reserve
                    ),
                    8,
                )
        with self.assertRaises(ValueError):
            start_server._get_startup_real_warmup_request_token_len(5, 5, 4)

    def test_empty_warmup_cannot_mark_service_ready(self):
        async def empty_outputs(_):
            if False:
                yield

        client = Mock(enqueue=empty_outputs, close=AsyncMock())
        configs = SimpleNamespace(grpc_config=None)
        with (
            patch.object(
                start_server, "_get_startup_real_warmup_token_lens", return_value=[8]
            ),
            patch.object(
                start_server, "_get_startup_real_warmup_max_len", return_value=32
            ),
            patch.object(
                start_server,
                "_get_startup_real_warmup_speculative_reserve_step",
                return_value=4,
            ),
            patch.object(
                start_server,
                "_get_startup_real_warmup_grpc_addresses",
                return_value=["localhost:1"],
            ),
            patch(
                "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient",
                return_value=client,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no generated tokens"):
                asyncio.run(start_server._run_startup_real_warmup_grpc(configs))
        client.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

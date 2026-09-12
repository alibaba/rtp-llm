import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from rtp_llm import start_server
from rtp_llm.ops import SpeculativeType
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


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
            1048568,
        )


class StartupRealWarmupResponseTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _output(**changes):
        values = {
            "finished": True,
            "output_ids": torch.tensor([[42]], dtype=torch.int32),
            "aux_info": AuxInfo(input_len=1048567, output_len=1),
        }
        values.update(changes)
        return GenerateOutputs(generate_outputs=[GenerateOutput(**values)])

    async def _run(self, chunks):
        async def enqueue(request):
            self.assertEqual(request.token_ids.numel(), 1048567)
            self.assertEqual(request.generate_config.max_new_tokens, 1)
            for chunk in chunks:
                yield chunk

        client = SimpleNamespace(enqueue=enqueue, close=AsyncMock())
        helpers = {
            "_get_startup_real_warmup_token_lens": [1048576],
            "_get_startup_real_warmup_max_len": 1048576,
            "_get_startup_real_warmup_speculative_reserve_step": 9,
            "_get_startup_real_warmup_grpc_addresses": ["127.0.0.1:62001"],
        }
        with ExitStack() as patches:
            for name, value in helpers.items():
                patches.enter_context(
                    patch.object(start_server, name, return_value=value)
                )
            patches.enter_context(
                patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient",
                    return_value=client,
                )
            )
            try:
                await start_server._run_startup_real_warmup_grpc(
                    SimpleNamespace(grpc_config=None)
                )
            finally:
                client.close.assert_awaited_once()

    async def test_completed_output_with_exact_server_counts_passes(self):
        await self._run([self._output()])

    async def test_empty_stream_is_rejected(self):
        with self.assertRaises(RuntimeError):
            await self._run([])

    async def test_empty_response_is_rejected(self):
        with self.assertRaises(RuntimeError):
            await self._run([GenerateOutputs()])

    async def test_unfinished_output_is_rejected(self):
        with self.assertRaises(RuntimeError):
            await self._run([self._output(finished=False)])

    async def test_missing_tokens_or_server_counts_are_rejected(self):
        for changes in (
            {"output_ids": torch.empty((1, 0), dtype=torch.int32)},
            {"aux_info": None},
            {"aux_info": AuxInfo(input_len=524288, output_len=1)},
            {"aux_info": AuxInfo(input_len=1048567, output_len=0)},
        ):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                await self._run([self._output(**changes)])


if __name__ == "__main__":
    unittest.main()

"""DSv4 instance options remain frozen across the generic chunker migration."""

import os
import unittest
from types import MappingProxyType
from unittest import mock

from rtp_llm.models_py.modules.dsv4.chunk_env import (
    chunked_moe_enabled,
    moe_chunk_tokens_from_env,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    resolve_moe_max_tokens_per_rank,
)


class FrozenChunkOptionsTest(unittest.TestCase):
    def test_explicit_moe_chunk_policy_survives_environment_changes(self):
        options = MappingProxyType({"DSV4_MOE_CHUNK_PREFILL": "1"})
        for enabled in ("0", "1"):
            with mock.patch.dict(
                os.environ,
                {"DSV4_MOE_CHUNK_PREFILL": enabled, "DSV4_CHUNK_TOKENS": "0"},
            ):
                self.assertTrue(chunked_moe_enabled(options))
                self.assertEqual(moe_chunk_tokens_from_env(options=options), 16384)

    def test_explicit_budget_does_not_reread_process_chunk_options(self):
        first = MappingProxyType({"DSV4_CHUNK_TOKENS": "3"})
        second = MappingProxyType({"DSV4_MOE_CHUNK_PREFILL": "0"})
        kwargs = dict(
            max_seq_len=64,
            current_max_tokens_per_rank=64,
            cp_size=1,
            max_generate_batch_size=1,
        )
        with mock.patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "1"}):

            def budget(options=None):
                return resolve_moe_max_tokens_per_rank(
                    **kwargs,
                    chunking_enabled=chunked_moe_enabled(options),
                    chunk_tokens=moe_chunk_tokens_from_env(options=options)
                )

            self.assertEqual(budget(first), 3)
            self.assertEqual(budget(second), 64)
            self.assertEqual(budget(), 1)


if __name__ == "__main__":
    unittest.main()

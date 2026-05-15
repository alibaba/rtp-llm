from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe"),
)

from rtp_llm.models_py.modules.dsv4.moe.mega_jit_warmup import (
    clamp_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)


class MegaMoEJitWarmupTest(unittest.TestCase):
    def test_env_switch_defaults_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(mega_moe_jit_warmup_enabled())
        with mock.patch.dict(os.environ, {"DSV4_MEGA_MOE_JIT_WARMUP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_default_dsv4_ep4_chunk_16k_representatives(self):
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "1"}):
            tokens = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=256,
                num_experts_per_rank=64,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=16384,
            )
        self.assertEqual(
            tokens,
            [1, 11, 91, 177, 347, 689, 1030, 2049, 4097, 16384],
        )

    def test_representatives_share_bucket_with_cp4_even_values(self):
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "1"}):
            tokens = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=256,
                num_experts_per_rank=64,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=65536,
            )
        cp4_even_tokens = [
            2,
            12,
            92,
            178,
            348,
            690,
            1030,
            2050,
            4098,
            8194,
            65536,
        ]
        for warmup_t, cp4_t in zip(tokens, cp4_even_tokens):
            self.assertEqual(
                mega_moe_config_signature(
                    num_ranks=4,
                    num_experts=256,
                    num_experts_per_rank=64,
                    num_tokens=warmup_t,
                    num_topk=6,
                    intermediate_hidden=2048,
                    num_sms=148,
                ),
                mega_moe_config_signature(
                    num_ranks=4,
                    num_experts=256,
                    num_experts_per_rank=64,
                    num_tokens=cp4_t,
                    num_topk=6,
                    intermediate_hidden=2048,
                    num_sms=148,
                ),
            )

    def test_chunk_disabled_keeps_long_bucket_start(self):
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "0"}):
            tokens = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=256,
                num_experts_per_rank=64,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=250000,
            )
        self.assertEqual(
            tokens,
            [1, 11, 91, 177, 347, 689, 1030, 2049, 4097, 8193, 18433],
        )

    def test_global_chunk_zero_keeps_long_bucket_start(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_CHUNK_TOKENS": "0", "DSV4_MOE_CHUNK_PREFILL": "1"},
            clear=True,
        ):
            tokens = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=256,
                num_experts_per_rank=64,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=250000,
            )
        self.assertEqual(
            tokens,
            [1, 11, 91, 177, 347, 689, 1030, 2049, 4097, 8193, 18433],
        )

    def test_ep_size_changes_representatives(self):
        with mock.patch.dict(os.environ, {"DSV4_MOE_CHUNK_PREFILL": "1"}):
            ep4 = generate_mega_moe_jit_token_counts(
                num_ranks=4,
                num_experts=256,
                num_experts_per_rank=64,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=65536,
            )
            ep8 = generate_mega_moe_jit_token_counts(
                num_ranks=8,
                num_experts=256,
                num_experts_per_rank=32,
                num_topk=6,
                intermediate_hidden=2048,
                num_sms=148,
                max_tokens_per_rank=65536,
            )
        self.assertNotEqual(ep4, ep8)
        self.assertEqual(ep8[-1], 65536)

    def test_override_tokens_are_sorted_unique_and_clamped(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_MEGA_MOE_JIT_WARMUP_TOKENS": "4098, 2, 2, 999999, bad"},
        ):
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        with mock.patch.dict(
            os.environ,
            {"DSV4_MEGA_MOE_JIT_WARMUP_TOKENS": "4098, 2, 2, 999999, 0, -1"},
        ):
            tokens = parse_mega_moe_jit_warmup_tokens_override()
        self.assertEqual(tokens, [2, 4098, 999999])
        self.assertEqual(clamp_token_counts(tokens or [], 65536), [2, 4098, 65536])


if __name__ == "__main__":
    unittest.main()

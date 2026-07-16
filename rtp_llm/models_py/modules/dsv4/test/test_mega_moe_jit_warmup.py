from __future__ import annotations

import os
import sys
import tempfile
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
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe.strategies",
    os.path.join(
        _REPO,
        "rtp_llm",
        "models_py",
        "modules",
        "dsv4",
        "moe",
        "strategies",
    ),
)

from rtp_llm.models_py.modules.dsv4.moe.mega_jit_warmup import (
    clamp_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import (
    MegaMoEStrategy,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _MEGA_MOE_JIT_WARMED_KEYS,
    _mega_moe_rank_nvcc_tmpdir,
    _restore_tmpdir,
)


class MegaMoEJitWarmupTest(unittest.TestCase):
    def test_env_switch_defaults_on(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(mega_moe_jit_warmup_enabled())
        with mock.patch.dict(os.environ, {"DSV4_MEGA_MOE_JIT_WARMUP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_rank_local_nvcc_tmpdir_uses_deepgemm_cache_and_rank(self):
        with mock.patch.dict(
            os.environ,
            {"DG_JIT_CACHE_DIR": "/tmp/dg-cache"},
            clear=True,
        ):
            self.assertEqual(
                _mega_moe_rank_nvcc_tmpdir(7),
                "/tmp/dg-cache/rtp_llm_dsv4_mega_moe_nvcc/rank_7",
            )

    def test_activate_rank_local_nvcc_tmpdir_restores_previous_tmpdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {"DSV4_MEGA_MOE_NVCC_TMPDIR": tmpdir, "TMPDIR": "/old/tmp"},
                clear=True,
            ):
                active, previous = _activate_mega_moe_rank_nvcc_tmpdir(3)
                self.assertEqual(previous, "/old/tmp")
                self.assertEqual(os.environ["TMPDIR"], active)
                self.assertTrue(os.path.isdir(active))
                self.assertTrue(active.endswith("rank_3"))
                _restore_tmpdir(previous)
                self.assertEqual(os.environ["TMPDIR"], "/old/tmp")

    def test_maybe_warmup_restores_tmpdir_when_compile_fails(self):
        fake_deep_gemm = types.SimpleNamespace(get_num_sms=lambda: 148)
        cfg = MoeCfg(
            layer_id=123,
            dim=7168,
            moe_inter_dim=2048,
            n_routed_experts=256,
            n_activated_experts=6,
            swiglu_limit=7.0,
            ep_size=8,
            ep_rank=5,
            n_local_experts=32,
            local_expert_start=160,
            local_expert_end=192,
            max_tokens_per_rank=4096,
        )
        strategy = MegaMoEStrategy(cfg)
        strategy.warmup_jit = mock.Mock(side_effect=RuntimeError("compile failed"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {
                    "DSV4_MEGA_MOE_NVCC_TMPDIR": tmpdir,
                    "DSV4_MOE_CHUNK_PREFILL": "1",
                    "TMPDIR": "/old/tmp",
                },
                clear=True,
            ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
                "torch.cuda.is_current_stream_capturing", return_value=False
            ), mock.patch(
                "torch.distributed.is_initialized", return_value=True
            ), mock.patch(
                "torch.distributed.get_rank", return_value=5
            ):
                _MEGA_MOE_JIT_WARMED_KEYS.clear()
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    strategy._maybe_warmup_jit_once()

                self.assertEqual(os.environ["TMPDIR"], "/old/tmp")
                self.assertTrue(
                    os.path.isdir(
                        os.path.join(
                            tmpdir,
                            "rtp_llm_dsv4_mega_moe_nvcc",
                            "rank_5",
                        )
                    )
                )
                strategy.warmup_jit.assert_called_once()
                self.assertEqual(len(_MEGA_MOE_JIT_WARMED_KEYS), 0)

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

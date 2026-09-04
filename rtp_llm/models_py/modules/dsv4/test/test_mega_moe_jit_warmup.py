import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    _MEGA_MOE_JIT_WARMED_KEYS,
    MegaMoeExecutor,
    _activate_mega_moe_rank_nvcc_tmpdir,
    _mega_moe_rank_nvcc_tmpdir,
    _restore_tmpdir,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_config_signature,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)


class MegaMoeJitWarmupTest(unittest.TestCase):
    def test_model_warmup_switch(self):
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "0"}):
            self.assertFalse(mega_moe_jit_warmup_enabled())

    def test_rank_local_nvcc_directory(self):
        with mock.patch.dict(
            os.environ, {"DG_JIT_CACHE_DIR": "/tmp/dg-cache"}, clear=True
        ):
            self.assertEqual(
                _mega_moe_rank_nvcc_tmpdir(7),
                "/tmp/dg-cache/rtp_llm_mega_moe_nvcc/rank_7",
            )

    def test_tmpdir_is_restored_after_warmup_failure(self):
        executor = MegaMoeExecutor.__new__(MegaMoeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=1,
            hidden_size=7168,
            moe_inter_dim=2048,
            expert_num=256,
            moe_k=6,
            n_shared_experts=1,
            swiglu_limit=7.0,
            ep_size=8,
            ep_rank=5,
            max_tokens_per_rank=4096,
            moe_strategy="mega_moe",
        )
        executor.warmup_jit = mock.Mock(side_effect=RuntimeError("compile failed"))
        fake_deep_gemm = types.SimpleNamespace(get_num_sms=lambda: 148)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "MEGA_MOE_NVCC_TMPDIR": tmpdir,
                "MODEL_WARM_UP": "1",
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
                executor._maybe_warmup_jit_once()
            self.assertEqual(os.environ["TMPDIR"], "/old/tmp")
            self.assertTrue(
                os.path.isdir(os.path.join(tmpdir, "rtp_llm_mega_moe_nvcc", "rank_5"))
            )

    def test_unchunked_generated_counts_use_bucket_representative(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
        )
        self.assertLess(tokens[-1], 16384)

    def test_chunked_generated_counts_cover_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=16384,
            include_cap=True,
        )
        self.assertEqual(tokens[-1], 16384)

    def test_generated_counts_cover_every_unique_bucket_for_ep_sizes(self):
        for ep_size in (2, 4, 8):
            with self.subTest(ep_size=ep_size):
                params = dict(
                    num_ranks=ep_size,
                    num_experts=256,
                    num_experts_per_rank=256 // ep_size,
                    num_topk=6,
                    intermediate_hidden=2048,
                    num_sms=148,
                )
                tokens = generate_mega_moe_jit_token_counts(
                    **params,
                    max_tokens_per_rank=4096,
                )
                self.assertTrue(any(1 < token < 4096 for token in tokens))
                signatures = [
                    mega_moe_config_signature(**params, num_tokens=token)
                    for token in tokens
                ]
                expected_signatures = []
                previous = None
                for token in range(1, 4097):
                    signature = mega_moe_config_signature(**params, num_tokens=token)
                    if signature != previous:
                        expected_signatures.append(signature)
                        previous = signature
                self.assertEqual(signatures, expected_signatures)

    def test_generated_counts_never_exceed_runtime_cap(self):
        tokens = generate_mega_moe_jit_token_counts(
            num_ranks=8,
            num_experts=256,
            num_experts_per_rank=32,
            num_topk=6,
            intermediate_hidden=2048,
            num_sms=148,
            max_tokens_per_rank=257,
        )
        self.assertTrue(tokens)
        self.assertLessEqual(max(tokens), 257)

    def test_override_tokens_are_generic_sorted_and_clamped(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "4098,2,2,999999,0,-1"},
        ):
            tokens = parse_mega_moe_jit_warmup_tokens_override()
        self.assertEqual(tokens, [2, 4098, 999999])
        self.assertEqual(clamp_token_counts(tokens or [], 65536), [2, 4098, 65536])

    def test_invalid_override_falls_back_to_generated_counts(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "2,not-a-token"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("invalid MEGA_MOE_JIT_WARMUP_TOKENS", "\n".join(logs.output))

    def test_override_without_positive_tokens_falls_back(self):
        with mock.patch.dict(
            os.environ,
            {"MEGA_MOE_JIT_WARMUP_TOKENS": "0,-1,-20"},
            clear=True,
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(parse_mega_moe_jit_warmup_tokens_override())
        self.assertIn("contains no positive token counts", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()

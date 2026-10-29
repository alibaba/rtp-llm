"""Fused main attention must preserve the native first-call RoPE policy."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


def check_policy(mode):
    # The native policy uses std::call_once, so every case needs a fresh process.
    from rtp_llm.ops import RopeStyle, get_rope_cache_once

    from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
        SparseGqaFmhaImpl,
    )
    from rtp_llm.models_py.modules.qwen4_exp.test.main_qk_cache_test import (
        MainQkCacheTest,
    )

    tester = MainQkCacheTest()
    tester.setUp()
    case = tester.case(12, 1, 4, [127])
    cfg, norm, source, _, params, original, _, _, cache = case
    assert cfg.rope_config.style == RopeStyle.Mrope
    base = tester.tester.setup_case(12, 4)[0]
    base.rope_config.style = RopeStyle.Base
    if mode == "base_first":
        assert get_rope_cache_once(base.rope_config, base.max_seq_len).used
    if mode == "native":
        original.forward(norm(source), cache, params)
    elif mode != "omitted_main":
        owner = tester.mutation_owner(case)

        class WriterReached(Exception):
            pass

        def launch():
            assert owner._qsa_main_cache_mutation_started
            # Inspect the policy at the launch boundary, before any MTP layer.
            policy = get_rope_cache_once(base.rope_config, base.max_seq_len)
            assert policy.used == (mode == "base_first")
            raise WriterReached

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_cache_triton.prepare_main_qk_cache",
            return_value=SimpleNamespace(run=Mock(side_effect=launch)),
        ):
            try:
                SparseGqaFmhaImpl.forward(
                    owner,
                    source,
                    cache,
                    3,
                    selected_indices=torch.empty(
                        4, 2051, device="cuda", dtype=torch.int32
                    ),
                    qk_norm=norm,
                )
            except WriterReached:
                pass
            else:
                raise AssertionError("fused launch was not reached")
        owner.rope_kvcache_impl.forward.assert_not_called()
    policy = get_rope_cache_once(base.rope_config, base.max_seq_len)
    assert policy.used == (mode in ("base_first", "omitted_main"))
    torch.cuda.synchronize()
    print("ROPE_POLICY_PASS", mode, policy.used)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RopeCachePolicyTest(unittest.TestCase):
    def fresh_process(self, mode):
        code = (
            "from rtp_llm.models_py.modules.qwen4_exp.test."
            "rope_cache_policy_test import check_policy; "
            f"check_policy({mode!r})"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ROPE_POLICY_PASS", result.stdout)

    def test_native_mrope_first_keeps_mtp_uncached(self):
        self.fresh_process("native")

    def test_fused_mrope_first_keeps_mtp_uncached(self):
        self.fresh_process("fused")

    def test_preexisting_base_cache_is_preserved(self):
        self.fresh_process("base_first")

    def test_omitted_main_call_changes_native_policy(self):
        self.fresh_process("omitted_main")


if __name__ == "__main__":
    unittest.main()

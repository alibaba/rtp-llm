"""
Unit tests for PCPAllGatherAttnOp (all-gather without overlap).

Tests two scenarios:
  1. Normal context-parallel attention (no prefix cache)
  2. Context-parallel attention with prefix cache
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl import (
    PCPAllGatherAttnOp,
    _build_cp_sharded_params_block_table,
    _use_fa4_cp_paged,
)
from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    CPAttnTestBase,
    build_cp_attn_inputs,
    make_configs,
)
from rtp_llm.ops.compute_ops import fill_mla_params

_AG_MODULE = (
    "rtp_llm.models_py.modules.factory.attention."
    "cuda_cp_impl.prefill_mha.allgather_cp_impl"
)


class TestPCPAllGatherAttnOp(CPAttnTestBase):
    OP_CLASS = PCPAllGatherAttnOp
    AG_MODULE = _AG_MODULE

    def setUp(self):
        super().setUp()
        # CPAttnTestBase's functional mock models separate K and V all-gathers.
        # Keep those legacy numerical tests on that path; the packed-KV forward
        # optimization uses a single collective and is covered by CP E2E tests.
        self._forward_opt_env = patch.dict(
            "os.environ", {"RTP_LLM_CP_PREFILL_FORWARD_OPT": "0"}
        )
        self._forward_opt_env.start()

    def tearDown(self):
        self._forward_opt_env.stop()
        super().tearDown()

    def test_sharded_params_table_covers_all_logical_pages(self):
        table = _build_cp_sharded_params_block_table(
            prefix_lengths=torch.tensor([79_872, 512], dtype=torch.int32),
            input_lengths=torch.tensor([627, 1], dtype=torch.int32),
            page_size=128,
        )
        self.assertEqual(table.shape, (2, 629))
        self.assertEqual(table[0, 0].item(), 0)
        self.assertEqual(table[0, 628].item(), 628)
        self.assertEqual(table[1, 0].item(), 629)
        self.assertEqual(table[1, 4].item(), 633)

    def test_sharded_prepare_does_not_pass_compact_physical_table_to_params(self):
        attn_cfg, par_cfg = make_configs(cp_size=4, cp_rank=2, tokens_per_block=128)
        par_cfg.prefill_cp_config.kv_cache_sharded = True
        attn_inputs = build_cp_attn_inputs(
            sequence_lengths=[1028],
            cp_chunk_lengths=[4],
            cp_size=4,
            tokens_per_block=128,
            prefix_lengths=[1024],
            device=self.device,
        )
        # One rank owns only ceil(9 / 4) physical pages, but fill_mla_params
        # requires all nine logical page entries.
        attn_inputs.kv_cache_kernel_block_id_host = torch.tensor(
            [[11, 12, 13]], dtype=torch.int32
        )
        captured = {}
        real_fill = fill_mla_params

        def capture_fill(prefix, sequence, input_lengths, block_table, page_size):
            captured["block_table"] = block_table
            return real_fill(prefix, sequence, input_lengths, block_table, page_size)

        op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
        with patch(f"{_AG_MODULE}.fill_mla_params", side_effect=capture_fill), patch(
            f"{_AG_MODULE}.plan_prefix_paged_attention"
        ):
            op.prepare(attn_inputs)
        self.assertEqual(captured["block_table"].shape, (1, 9))
        self.assertEqual(captured["block_table"].tolist(), [list(range(9))])

    def test_fa4_is_kept_for_no_prefix_only(self):
        with patch(f"{_AG_MODULE}._HAS_FA4", True), patch.dict(
            "os.environ", {"RTP_LLM_CP_PREFILL_FA4": "1"}
        ):
            self.assertTrue(_use_fa4_cp_paged(has_prefix=False))
            self.assertFalse(_use_fa4_cp_paged(has_prefix=True))

    def test_fa4_is_disabled_for_fp8_cache_serving_process(self):
        with patch(f"{_AG_MODULE}._HAS_FA4", True), patch.dict(
            "os.environ", {"RTP_LLM_CP_PREFILL_FA4": "1", "REUSE_CACHE": "1"}
        ):
            self.assertFalse(_use_fa4_cp_paged(fp8_kv_cache=True))
            self.assertTrue(_use_fa4_cp_paged(fp8_kv_cache=False))

    def test_fa4_env_disable_still_applies_without_prefix(self):
        with patch(f"{_AG_MODULE}._HAS_FA4", True), patch.dict(
            "os.environ", {"RTP_LLM_CP_PREFILL_FA4": "0"}
        ):
            self.assertFalse(_use_fa4_cp_paged(has_prefix=False))

    def test_sharded_prefix_hit_plans_contiguous_gathered_pages(self):
        attn_cfg, par_cfg = make_configs(cp_size=2, cp_rank=0)
        par_cfg.prefill_cp_config.kv_cache_sharded = True
        attn_inputs = build_cp_attn_inputs(
            sequence_lengths=[96],
            cp_chunk_lengths=[16],
            cp_size=2,
            tokens_per_block=16,
            prefix_lengths=[64],
            device=self.device,
        )

        op = self.OP_CLASS(attn_cfg, attn_inputs, par_cfg)
        with patch(f"{_AG_MODULE}.plan_prefix_paged_attention") as plan_prefix:
            op.prepare(attn_inputs)
        self.assertTrue(op.has_prefix)
        self.assertTrue(op._kv_sharded)
        self.assertTrue(plan_prefix.call_args.kwargs["contiguous_page_indices"])

    # ==================================================================
    # Case 1: Normal CP attention (no prefix cache)
    # ==================================================================

    def test_no_prefix_single_seq_rank0(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=0)

    def test_no_prefix_single_seq_rank1(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[32], cp_size=2, cp_rank=1)

    def test_no_prefix_multi_batch(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[32, 64], cp_size=2, cp_rank=0
        )

    def test_no_prefix_larger(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_no_prefix_cp4(self):
        self.run_no_prefix(batch_size=1, sequence_lengths=[64], cp_size=4, cp_rank=2)

    def test_no_prefix_gqa4(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[32],
            cp_size=2,
            cp_rank=0,
            head_num=16,
            kv_head_num=4,
            head_dim=64,
        )

    def test_no_prefix_multi_batch_cp4(self):
        self.run_no_prefix(
            batch_size=2, sequence_lengths=[64, 64], cp_size=4, cp_rank=1
        )

    # ==================================================================
    # Case 2: CP attention with prefix cache
    # ==================================================================

    def test_prefix_single_seq_rank0(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_single_seq_rank1(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[32],
            prefix_lengths=[64],
            cp_size=2,
            cp_rank=1,
            tokens_per_block=16,
        )

    def test_prefix_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[32, 64],
            prefix_lengths=[64, 128],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=32,
        )

    def test_prefix_larger(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[128],
            cp_size=2,
            cp_rank=0,
            head_num=32,
            kv_head_num=8,
            head_dim=128,
            tokens_per_block=64,
        )

    def test_prefix_cp4(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[64],
            prefix_lengths=[64],
            cp_size=4,
            cp_rank=2,
            tokens_per_block=16,
        )

    # ==================================================================
    # Case 3: Irregular seq_len (non-power-of-2, partial pages)
    # ==================================================================

    def test_no_prefix_irregular_seqlen(self):
        self.run_no_prefix(
            batch_size=1,
            sequence_lengths=[20],
            cp_size=2,
            cp_rank=0,
        )

    def test_no_prefix_irregular_multi_batch(self):
        self.run_no_prefix(
            batch_size=2,
            sequence_lengths=[20, 36],
            cp_size=2,
            cp_rank=1,
        )

    def test_prefix_irregular_seqlen(self):
        self.run_with_prefix(
            batch_size=1,
            new_lengths=[20],
            prefix_lengths=[48],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )

    def test_prefix_irregular_multi_batch(self):
        self.run_with_prefix(
            batch_size=2,
            new_lengths=[20, 36],
            prefix_lengths=[48, 32],
            cp_size=2,
            cp_rank=0,
            tokens_per_block=16,
        )


if __name__ == "__main__":
    unittest.main()

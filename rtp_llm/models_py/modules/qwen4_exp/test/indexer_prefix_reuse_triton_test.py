"""Fused prefix-reuse preparation: bit-exactness, declines and transactions.

The fused writer must be a drop-in for the staged vectorized writer on the
geometry it specializes on (page-aligned prefix reuse, ratio 4, 128-token
KV/state pages, 128-wide heads): pools match bit for bit, the padded scorer
query matches the eager norm + RoPE + padding chain bit for bit, declines
leave every pool untouched, and launch failures restore both snapshots.
"""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    clear_step_plan_cache,
)
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton import (
    prefix_reuse_fused_stats,
    write_prefix_reuse_prefill,
)
from rtp_llm.ops import RopeStyle


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IndexerPrefixReuseTritonTest(unittest.TestCase):
    HEADS = 4
    D = 128
    RATIO = 4
    PAGE = 128
    EPS = 1e-6

    def _table(self, batch, width, blocks):
        ids = (
            torch.randperm(blocks - 1, device="cuda", dtype=torch.int64)[
                : batch * width
            ]
            + 1
        )
        return ids.reshape(batch, width).to(torch.int32).contiguous()

    def inputs(self, prefixes, lengths, seed=20261102):
        torch.manual_seed(seed)
        batch = len(lengths)
        maximum = max(lengths)
        tokens = sum(lengths)
        width = max(
            (p + l + self.PAGE - 1) // self.PAGE
            for p, l in zip(prefixes, lengths)
        )
        blocks = batch * width + 2
        raw = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(
            tokens, self.HEADS, 128, device="cuda", dtype=torch.bfloat16
        )
        cu = torch.tensor(
            [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
        )
        prefixes_i32 = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
        prefixes_i64 = torch.tensor(prefixes, device="cuda", dtype=torch.int64)
        kv_table = self._table(batch, width, blocks)
        state_table = self._table(batch, width, blocks)
        q_gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
        k_gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(blocks, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(blocks, 8, 128, device="cuda")
        positions = torch.cat(
            [
                torch.arange(p, p + l, device="cuda", dtype=torch.int64)
                for p, l in zip(prefixes, lengths)
            ]
        )
        frequency = 10000.0 ** (
            -torch.arange(32, device="cuda", dtype=torch.float32) / 32
        )
        angles = positions[:, None].float() * frequency[None, :]
        current_cos = (
            torch.cat((angles.cos(), angles.cos()), -1).bfloat16().contiguous()
        )
        current_sin = (
            torch.cat((angles.sin(), angles.sin()), -1).bfloat16().contiguous()
        )
        block_starts = positions - positions.remainder(self.RATIO)
        block_angles = block_starts[:, None].float() * frequency[None, :]
        rope_cos = (
            torch.cat((block_angles.cos(), block_angles.cos()), -1)
            .bfloat16()
            .contiguous()
        )
        rope_sin = (
            torch.cat((block_angles.sin(), block_angles.sin()), -1)
            .bfloat16()
            .contiguous()
        )
        return dict(
            raw=raw,
            q=q,
            cu=cu,
            prefixes_i32=prefixes_i32,
            prefixes_i64=prefixes_i64,
            lengths=list(lengths),
            prefix_values=list(prefixes),
            current_cos=current_cos,
            current_sin=current_sin,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            q_gamma=q_gamma,
            k_gamma=k_gamma,
            kv=kv,
            state=state,
            kv_table=kv_table,
            state_table=state_table,
            blocks=blocks,
            maximum=maximum,
        )

    def write(self, args, kv, state):
        return write_prefix_reuse_prefill(
            args["q"],
            args["raw"],
            args["cu"],
            args["prefixes_i32"],
            args["lengths"],
            args["prefix_values"],
            args["current_cos"],
            args["current_sin"],
            args["rope_cos"],
            args["rope_sin"],
            args["q_gamma"],
            args["k_gamma"],
            kv,
            args["kv_table"],
            state,
            args["state_table"],
            kv_tokens_per_block=self.PAGE,
            state_tokens_per_block=self.PAGE,
            norm_eps=self.EPS,
        )

    def expected_padded_q(self, args):
        lengths = args["lengths"]
        maximum = max(lengths)
        normed = exact_head_rms_norm(args["q"], args["q_gamma"], self.EPS)
        rotated = apply_partial_rope(
            normed,
            args["current_cos"].unsqueeze(1),
            args["current_sin"].unsqueeze(1),
        )
        row_map = torch.cat(
            [
                args["cu"][r]
                + torch.clamp(
                    torch.arange(maximum, device="cuda"), max=lengths[r] - 1
                ).to(torch.int64)
                for r in range(len(lengths))
            ]
        )
        return rotated.index_select(0, row_map).reshape(
            len(lengths), maximum, self.HEADS, 128
        )

    def test_fused_matches_vectorized_writer_bit_for_bit(self):
        cases = (
            ([0, 0], [128, 4]),  # aligned starts; divisible + one-group tail
            ([128, 128], [129, 7]),  # prefix reuse crossing pages, non-divisible
            ([256, 0], [4, 132]),  # mixed hot/cold rows, divisible, multi-page
            ([0], [3]),  # every row inside one incomplete group
            ([384, 128, 0], [260, 1, 131]),  # batch > 1, mixed prefixes/lengths
            ([0], [512]),  # four pages in one request
        )
        for prefixes, lengths in cases:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                args = self.inputs(prefixes, lengths)
                kv_ref, state_ref = args["kv"].clone(), args["state"].clone()
                kv_fused, state_fused = args["kv"].clone(), args["state"].clone()
                with patch.dict(
                    os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
                ):
                    reference = write_indexer_cache(
                        args["raw"],
                        args["cu"],
                        args["prefixes_i64"],
                        args["rope_cos"],
                        args["rope_sin"],
                        args["k_gamma"],
                        kv_ref,
                        args["kv_table"],
                        state_ref,
                        args["state_table"],
                        ratio=self.RATIO,
                        kv_tokens_per_block=self.PAGE,
                        state_tokens_per_block=self.PAGE,
                        norm_eps=self.EPS,
                        capture_undo=True,
                        rope_is_token_aligned=True,
                    )
                    fused = self.write(args, kv_fused, state_fused)
                self.assertIsNotNone(fused)
                torch.testing.assert_close(state_fused, state_ref, rtol=0, atol=0)
                torch.testing.assert_close(kv_fused, kv_ref, rtol=0, atol=0)
                torch.testing.assert_close(
                    fused["padded_q"], self.expected_padded_q(args), rtol=0, atol=0
                )
                restore_indexer_cache(fused["undo"])
                restore_indexer_cache(reference["undo"])
                torch.testing.assert_close(kv_fused, args["kv"], rtol=0, atol=0)
                torch.testing.assert_close(state_fused, args["state"], rtol=0, atol=0)

    def test_fused_declines_unsupported_geometry_without_mutation(self):
        args = self.inputs([128], [132])
        before = args["kv"].clone(), args["state"].clone()
        projection = torch.randn(
            sum(args["lengths"]) + 16, 256, device="cuda", dtype=torch.bfloat16
        )
        strided = projection[8:, -128:]
        self.assertFalse(strided.is_contiguous())
        declined = write_prefix_reuse_prefill(
            args["q"],
            strided,
            args["cu"],
            args["prefixes_i32"],
            args["lengths"],
            args["prefix_values"],
            args["current_cos"],
            args["current_sin"],
            args["rope_cos"],
            args["rope_sin"],
            args["q_gamma"],
            args["k_gamma"],
            args["kv"],
            args["kv_table"],
            args["state"],
            args["state_table"],
            kv_tokens_per_block=self.PAGE,
            state_tokens_per_block=self.PAGE,
            norm_eps=self.EPS,
        )
        self.assertIsNone(declined)
        # A prefix that is not page aligned must decline before any write.
        misaligned = dict(args)
        misaligned["prefix_values"] = [4]
        self.assertIsNone(self.write(misaligned, args["kv"], args["state"]))
        # A non-bf16 query and mismatched pool geometry decline as well.
        float_q = dict(args)
        float_q["q"] = args["q"].float()
        self.assertIsNone(self.write(float_q, args["kv"], args["state"]))
        narrow_kv = dict(args)
        narrow_kv["kv"] = args["kv"][:, :16].contiguous()
        self.assertIsNone(self.write(narrow_kv, narrow_kv["kv"], args["state"]))
        torch.testing.assert_close(args["kv"], before[0], rtol=0, atol=0)
        torch.testing.assert_close(args["state"], before[1], rtol=0, atol=0)

    def test_fused_failure_restores_both_snapshots(self):
        args = self.inputs([128], [132])
        before = args["kv"].clone(), args["state"].clone()
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton."
            "_prefix_reuse_k_compress"
        ) as kernel:
            kernel.__getitem__.return_value.side_effect = RuntimeError(
                "injected KV launch failure"
            )
            with self.assertRaisesRegex(RuntimeError, "injected KV launch failure"):
                self.write(args, args["kv"], args["state"])
        torch.testing.assert_close(args["kv"], before[0], rtol=0, atol=0)
        torch.testing.assert_close(args["state"], before[1], rtol=0, atol=0)

    def runtime_inputs(self, prefixes, lengths):
        args = self.inputs(prefixes, lengths)
        batch = len(lengths)
        cu_kv = torch.tensor(
            [0, *itertools.accumulate(p + l for p, l in zip(prefixes, lengths))],
            device="cuda",
            dtype=torch.int32,
        )
        positions = torch.cat(
            [
                torch.arange(p, p + l, device="cuda", dtype=torch.int32)
                for p, l in zip(prefixes, lengths)
            ]
        )
        metadata = dict(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor(lengths, dtype=torch.int32).pin_memory(),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32).pin_memory(),
            sequence_lengths=torch.empty(0, dtype=torch.int32).pin_memory(),
            cu_seqlens_device=args["cu"],
            cu_kv_seqlens_device=cu_kv,
            combo_position_ids=positions,
        )
        context = Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=SimpleNamespace(**metadata),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=args["kv"].view(torch.uint8).reshape(
                    args["kv"].shape[0], -1
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_kv_inputs=SimpleNamespace(
                **metadata, kv_cache_kernel_block_id_device=args["kv_table"]
            ),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=args["state"].reshape(args["state"].shape[0], -1),
                seq_size_per_block=self.PAGE,
            ),
            indexer_state_inputs=SimpleNamespace(
                **metadata, kv_cache_block_id_device=args["state_table"]
            ),
        )
        indexer = SimpleNamespace(
            head_dim=128,
            n_heads=self.HEADS,
            compress_ratio=4,
            token_budget=64,
            q_norm_gamma=args["q_gamma"],
            k_norm_gamma=args["k_gamma"],
            norm_eps=self.EPS,
        )
        rope = SimpleNamespace(
            style=RopeStyle.Base,
            index_factor=1,
            dim=64,
            base=10000,
            scale=1,
            indexer_is_neox_style=True,
        )
        return context, indexer, rope, args

    def test_runtime_prefix_reuse_fused_selection_matches_staged_writer(self):
        for prefixes, lengths in (
            ([0], [132]),
            ([128, 128], [129, 7]),
            ([256, 0, 384], [4, 132, 1]),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                clear_step_plan_cache()
                fused_context, fused_indexer, rope, fused_args = self.runtime_inputs(
                    prefixes, lengths
                )
                staged_context, staged_indexer, _, staged_args = self.runtime_inputs(
                    prefixes, lengths
                )
                q, raw = fused_args["q"], fused_args["raw"]
                normed = exact_head_rms_norm(q, fused_args["q_gamma"], self.EPS)
                hits = prefix_reuse_fused_stats()["hits"]
                with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
                    selected_fused = fused_context.select_prefix_reuse_prefill_tokens(
                        q,
                        raw,
                        indexer=fused_indexer,
                        rope_config=rope,
                        raw_query=True,
                    )
                self.assertGreater(prefix_reuse_fused_stats()["hits"], hits)
                clear_step_plan_cache()
                with patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}):
                    selected_staged = staged_context.select_prefix_reuse_prefill_tokens(
                        normed,
                        staged_args["raw"],
                        indexer=staged_indexer,
                        rope_config=rope,
                    )
                self.assertTrue(torch.equal(selected_fused, selected_staged))

    def test_strict_validation_bypasses_the_fused_writer(self):
        prefixes, lengths = [128], [132]
        clear_step_plan_cache()
        context, indexer, rope, args = self.runtime_inputs(prefixes, lengths)
        q, raw = args["q"], args["raw"]
        with patch.dict(
            os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}
        ), patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton."
            "write_prefix_reuse_prefill"
        ) as fused_writer:
            selected = context.select_prefix_reuse_prefill_tokens(
                q,
                raw,
                indexer=indexer,
                rope_config=rope,
                raw_query=True,
            )
        fused_writer.assert_not_called()
        self.assertEqual(tuple(selected.shape), (132, 64 + self.RATIO - 1))


if __name__ == "__main__":
    unittest.main()

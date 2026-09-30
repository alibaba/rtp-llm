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
from rtp_llm.models_py.modules.qwen4_exp.indexer_compressor import (
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer_prefill_triton import (
    write_zero_prefix_prefill,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa_runtime import Qwen4ExpQSARuntimeContext
from rtp_llm.ops import RopeStyle


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IndexerPrefillTritonTest(unittest.TestCase):
    def runtime_inputs(self, lengths):
        args = self.inputs(lengths)
        raw, cu, _, _, _, gamma, kv, table, state, state_table = args
        metadata = dict(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.zeros(len(lengths), dtype=torch.int32),
            sequence_lengths=torch.zeros(len(lengths), dtype=torch.int32),
            cu_seqlens_device=cu,
            cu_kv_seqlens_device=cu,
            combo_position_ids=torch.cat(
                [torch.arange(n, dtype=torch.int32) for n in lengths]
            ),
        )
        context = Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=SimpleNamespace(**metadata),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=kv.view(torch.uint8).reshape(kv.shape[0], -1),
                seq_size_per_block=128,
            ),
            indexer_kv_inputs=SimpleNamespace(
                **metadata, kv_cache_kernel_block_id_device=table
            ),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=state.reshape(state.shape[0], -1),
                seq_size_per_block=128,
            ),
            indexer_state_inputs=SimpleNamespace(
                **metadata, kv_cache_block_id_device=state_table
            ),
        )
        indexer = SimpleNamespace(
            head_dim=128, compress_ratio=4, k_norm_gamma=gamma, norm_eps=1e-6
        )
        rope = SimpleNamespace(
            style=RopeStyle.Base,
            index_factor=1,
            dim=64,
            base=10_000,
            scale=1,
            indexer_is_neox_style=True,
        )
        return context, raw, indexer, rope, kv, state

    def test_runtime_metadata_fusion_preserves_cache_and_exact_undo(self):
        for lengths in ([129, 7], [4096, 131]):
            results = []
            for enabled in ("0", "1"):
                context, raw, indexer, rope, kv, state = self.runtime_inputs(lengths)
                before = kv.clone(), state.clone()
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_QWEN4_FUSED_INDEXER_PREFILL": "1",
                        "RTP_LLM_QWEN4_PREFILL_METADATA_FUSION": enabled,
                    },
                ):
                    result = context.write_prefill_indexer_cache(
                        raw, indexer=indexer, rope_config=rope
                    )
                results.append((kv.clone(), state.clone(), result))
                self.assertIsNotNone(context._side_cache_undo)
                restore_indexer_cache(context._side_cache_undo)
                torch.testing.assert_close(kv, before[0], atol=0, rtol=0)
                torch.testing.assert_close(state, before[1], atol=0, rtol=0)
            for i in (0, 1):
                torch.testing.assert_close(results[0][i], results[1][i], atol=0, rtol=0)
            for i in (1, 2):
                torch.testing.assert_close(
                    results[0][2][i], results[1][2][i], atol=0, rtol=0
                )
            for key in ("state_slots", "kv_slots", "completed"):
                torch.testing.assert_close(
                    results[0][2][3][key], results[1][2][3][key], atol=0, rtol=0
                )

    def test_runtime_metadata_fusion_keeps_all_write_preflights(self):
        for enabled, case in itertools.product(
            ("0", "1"),
            ("cu", "prefix", "position", "state_missing", "kv_oob", "alias_bad_cu"),
        ):
            with self.subTest(enabled=enabled, case=case):
                context, raw, indexer, rope, kv, state = self.runtime_inputs([129, 7])
                before = kv.clone(), state.clone()
                if case in ("cu", "alias_bad_cu"):
                    context.main_inputs.cu_seqlens_device[1] = 128
                    if case == "alias_bad_cu":
                        table = (
                            context.indexer_kv_inputs.kv_cache_kernel_block_id_device
                        )
                        table[1, 0] = table[0, 0]
                elif case == "prefix":
                    context.main_inputs.prefix_lengths[0] = 128
                elif case == "position":
                    context.main_inputs.combo_position_ids[0] = 7
                elif case == "state_missing":
                    context.indexer_state_inputs.kv_cache_block_id_device[0, 0] = 0
                else:
                    context.indexer_kv_inputs.kv_cache_kernel_block_id_device[0, 0] = (
                        kv.shape[0]
                    )
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_QWEN4_FUSED_INDEXER_PREFILL": "1",
                        "RTP_LLM_QWEN4_PREFILL_METADATA_FUSION": enabled,
                    },
                ):
                    with self.assertRaises((RuntimeError, ValueError)):
                        context.write_prefill_indexer_cache(
                            raw, indexer=indexer, rope_config=rope
                        )
                self.assertIsNone(context._side_cache_undo)
                torch.testing.assert_close(kv, before[0], atol=0, rtol=0)
                torch.testing.assert_close(state, before[1], atol=0, rtol=0)

    def test_runtime_metadata_fallback_checks_cu_before_any_write(self):
        context, raw, indexer, rope, kv, state = self.runtime_inputs([129, 7])
        before = kv.clone(), state.clone()
        context.main_inputs.cu_seqlens_device[1] = 128
        with (
            patch.dict(
                os.environ,
                {
                    "RTP_LLM_QWEN4_FUSED_INDEXER_PREFILL": "1",
                    "RTP_LLM_QWEN4_PREFILL_METADATA_FUSION": "1",
                },
            ),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.indexer_prefill_triton.write_zero_prefix_prefill",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cu_seqlens do not match"):
                context.write_prefill_indexer_cache(
                    raw, indexer=indexer, rope_config=rope
                )
        torch.testing.assert_close(kv, before[0], atol=0, rtol=0)
        torch.testing.assert_close(state, before[1], atol=0, rtol=0)

    def inputs(self, lengths):
        torch.manual_seed(412)
        b, maximum = len(lengths), max(lengths)
        pages = (maximum + 127) // 128
        blocks = b * pages + 1
        raw = torch.randn(sum(lengths), 128, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor(
            [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
        )
        table = (
            torch.randperm(blocks - 1, device="cuda", dtype=torch.int32).reshape(
                b, pages
            )
            + 1
        )
        state_table = table.flip(0).contiguous()
        positions = torch.arange(maximum, device="cuda", dtype=torch.float32)
        freqs = torch.arange(32, device="cuda", dtype=torch.float32) / 32
        angles = positions[:, None] * freqs[None, :]
        cos = (
            torch.cat((angles.cos(), angles.cos()), -1)
            .bfloat16()
            .expand(b, -1, -1)
            .contiguous()
        )
        sin = (
            torch.cat((angles.sin(), angles.sin()), -1)
            .bfloat16()
            .expand(b, -1, -1)
            .contiguous()
        )
        gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(blocks, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(blocks, 8, 128, device="cuda")
        return raw, cu, lengths, cos, sin, gamma, kv, table, state, state_table

    def reference(self, args):
        raw, cu, lengths, cos, sin, gamma, kv, table, state, state_table = args
        return write_indexer_cache(
            raw,
            cu,
            torch.zeros(len(lengths), device="cuda", dtype=torch.int32),
            cos,
            sin,
            gamma,
            kv,
            table,
            state,
            state_table,
            ratio=4,
            kv_tokens_per_block=128,
            state_tokens_per_block=128,
            capture_undo=True,
        )

    def fused(self, args):
        return write_zero_prefix_prefill(*args, page_size=128, norm_eps=1e-6)

    def test_ragged_long_prefill_cache_metadata_and_rollback(self):
        for lengths in ([1, 3], [4, 7, 127, 128, 129, 257], [4096, 131]):
            with self.subTest(lengths=lengths):
                args = self.inputs(lengths)
                before_kv, before_state = args[6].clone(), args[8].clone()
                ref_args = list(args)
                ref_args[6], ref_args[8] = before_kv.clone(), before_state.clone()
                reference = self.reference(ref_args)
                result = self.fused(args)
                self.assertIsNotNone(result)
                torch.testing.assert_close(args[8], ref_args[8], atol=0, rtol=0)
                torch.testing.assert_close(
                    args[6], ref_args[6], atol=0.015625, rtol=0.008
                )
                for key in ("state_slots", "kv_slots", "completed"):
                    torch.testing.assert_close(
                        result[key], reference[key], atol=0, rtol=0
                    )
                for key in ("num_state_writes", "num_kv_writes"):
                    self.assertEqual(result[key], reference[key])
                restore_indexer_cache(result["undo"])
                torch.testing.assert_close(args[6], before_kv, atol=0, rtol=0)
                torch.testing.assert_close(args[8], before_state, atol=0, rtol=0)

    def test_invalid_cu_or_required_page_fails_before_either_pool_mutates(self):
        for case in ("cu", "state_missing", "state_oob", "kv_missing", "kv_oob"):
            with self.subTest(case=case):
                args = list(self.inputs([129, 7]))
                before_kv, before_state = args[6].clone(), args[8].clone()
                if case == "cu":
                    args[1][1] = 128
                else:
                    table = args[9] if case.startswith("state") else args[7]
                    pool = args[8] if case.startswith("state") else args[6]
                    table[0, 0] = pool.shape[0] if case.endswith("oob") else 0
                with self.assertRaises(ValueError):
                    self.fused(args)
                torch.testing.assert_close(args[6], before_kv, atol=0, rtol=0)
                torch.testing.assert_close(args[8], before_state, atol=0, rtol=0)

    def test_joint_qk_projection_strided_keys_and_float_norm_weight(self):
        args = list(self.inputs([257, 7]))
        projection = torch.randn(
            sum(args[2]), 5 * 128, device="cuda", dtype=torch.bfloat16
        )
        args[0] = projection[:, -128:]
        args[5] = args[5].float()
        self.assertFalse(args[0].is_contiguous())
        ref_args = list(args)
        ref_args[6], ref_args[8] = args[6].clone(), args[8].clone()
        reference = self.reference(ref_args)
        result = self.fused(args)
        self.assertIsNotNone(result)
        torch.testing.assert_close(args[8], ref_args[8], atol=0, rtol=0)
        torch.testing.assert_close(args[6], ref_args[6], atol=0.015625, rtol=0.008)
        for key in ("state_slots", "kv_slots", "completed"):
            torch.testing.assert_close(result[key], reference[key], atol=0, rtol=0)

    def test_unused_tail_pages_are_ignored_and_shared_pages_fall_back(self):
        args = list(self.inputs([257, 7]))
        args[7][1, 1:] = -1
        args[9][1, 1:] = args[8].shape[0] + 123
        self.assertIsNotNone(self.fused(args))
        for table_idx in (7, 9):
            args = list(self.inputs([128, 128]))
            args[table_idx][1, 0] = args[table_idx][0, 0]
            before_kv, before_state = args[6].clone(), args[8].clone()
            self.assertIsNone(self.fused(args))
            torch.testing.assert_close(args[6], before_kv, atol=0, rtol=0)
            torch.testing.assert_close(args[8], before_state, atol=0, rtol=0)

    def test_write_failure_restores_both_snapshots(self):
        args = self.inputs([129, 7])
        before_kv, before_state = args[6].clone(), args[8].clone()
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_prefill_triton._kv_write"
        ) as kernel:
            kernel.__getitem__.return_value.side_effect = RuntimeError(
                "injected KV launch failure"
            )
            with self.assertRaisesRegex(RuntimeError, "injected KV launch failure"):
                self.fused(args)
        torch.testing.assert_close(args[6], before_kv, atol=0, rtol=0)
        torch.testing.assert_close(args[8], before_state, atol=0, rtol=0)

    def test_partial_state_seals_correctly_on_next_decode(self):
        args = list(self.inputs([7]))
        ref_args = list(args)
        ref_args[6], ref_args[8] = args[6].clone(), args[8].clone()
        self.reference(ref_args)
        self.fused(args)
        tail = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        start = torch.tensor([7], device="cuda", dtype=torch.int32)
        for x in (args, ref_args):
            x[3] = torch.cat((x[3], x[3][:, -1:]), dim=1)
            x[4] = torch.cat((x[4], x[4][:, -1:]), dim=1)
            write_indexer_cache(
                tail,
                cu,
                start,
                x[3],
                x[4],
                x[5],
                x[6],
                x[7],
                x[8],
                x[9],
                ratio=4,
                kv_tokens_per_block=128,
                state_tokens_per_block=128,
            )
        torch.testing.assert_close(args[8], ref_args[8], atol=0, rtol=0)
        torch.testing.assert_close(args[6], ref_args[6], atol=0.015625, rtol=0.008)


if __name__ == "__main__":
    unittest.main()

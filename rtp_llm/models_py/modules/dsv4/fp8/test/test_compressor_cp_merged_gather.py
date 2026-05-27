"""UT: CompressorFP8 CP prefill gathers fused kv/score once and stays flat."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.cp import CPContext
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8, CompressorMeta


def _build_cpu_compressor(
    *, dim: int, head_dim: int, compress_ratio: int
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.zeros(compress_ratio, coff * head_dim, dtype=torch.float32),
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16),
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16),
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    cmp = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=0,
        compress_ratio=compress_ratio,
        max_batch_size=1,
        compressor_weights=weights,
    )

    state_eb = 4
    state_view = torch.zeros(4, 2 * coff * head_dim, dtype=torch.float32)
    state_block_table = torch.ones(1, 1, dtype=torch.int32)
    kv_pool = torch.zeros(1, 1, INDEXER_ENTRY_BYTES, dtype=torch.uint8)
    kv_block_table = torch.ones(1, 1, dtype=torch.int32)
    cmp.set_pool_context(
        kv_pool_view=kv_pool,
        kv_block_table=kv_block_table,
        kv_eb=1,
        state_pool_view=state_view,
        state_block_table=state_block_table,
        state_eb=state_eb,
        state_tokens_per_block=state_eb,
        kv_tokens_per_block=compress_ratio,
    )
    return cmp


def _make_cp_ctx() -> CPContext:
    return CPContext(
        cp_size=2,
        cp_rank=0,
        chunk_length=2,
        padded_seq_len=4,
        seq_len_full=3,
        relative_positions=torch.tensor([0, 1], dtype=torch.long),
        prefix_length=0,
        global_positions=torch.tensor([0, 1], dtype=torch.long),
        local_is_real=torch.tensor([True, True]),
        unpad_restore=torch.tensor([0, 2, 3], dtype=torch.long),
        seq_len_total=3,
        cp_info=object(),
        input_lengths_global=torch.tensor([3], dtype=torch.int32),
        cu_seqlens_global=torch.tensor([0, 3], dtype=torch.int32),
        unpad_restore_is_prefix=False,
    )


class CompressorFP8CPMergedGatherTest(unittest.TestCase):
    def test_cp_prefill_gathers_fused_projection_once_and_launches_flat(self) -> None:
        torch.manual_seed(0)
        dim = 8
        head_dim = INDEXER_HEAD_DIM
        compress_ratio = 4
        coff = 1 + (compress_ratio == 4)
        out_dim = coff * head_dim
        cmp = _build_cpu_compressor(
            dim=dim, head_dim=head_dim, compress_ratio=compress_ratio
        )
        cp_ctx = _make_cp_ctx()
        cmp.set_cp_ctx(cp_ctx)

        x = torch.randn(cp_ctx.chunk_length, dim, dtype=torch.bfloat16)
        gathered_fused = torch.arange(
            cp_ctx.seq_len_full * 2 * out_dim, dtype=torch.bfloat16
        ).reshape(cp_ctx.seq_len_full, 2 * out_dim)
        meta = CompressorMeta(
            positions=torch.arange(cp_ctx.seq_len_full, dtype=torch.long),
            b_idx=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long),
            state_slots=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long),
            kv_slots=torch.zeros(cp_ctx.seq_len_full, dtype=torch.long),
            token_to_req=torch.zeros(cp_ctx.seq_len_full, dtype=torch.int32),
            is_batched=True,
            seq_start_per_req=torch.tensor([0], dtype=torch.int32),
            cu_seq_per_req=torch.tensor([0, cp_ctx.seq_len_full], dtype=torch.int32),
        )

        gather_inputs = []
        handle = object()
        launch_args = {}

        def fake_start(local_2d, ctx, stream=None):
            del stream
            gather_inputs.append((local_2d, ctx))
            return handle

        def fake_wait(actual_handle):
            self.assertIs(actual_handle, handle)
            return gathered_fused

        def fake_launch(kv_flat, score_flat, actual_meta, seq_start=None):
            launch_args["kv_flat"] = kv_flat
            launch_args["score_flat"] = score_flat
            launch_args["meta"] = actual_meta
            launch_args["seq_start"] = seq_start

        def fake_linear(local_2d, weight):
            del weight
            return torch.zeros(local_2d.shape[0], 2 * out_dim, dtype=torch.bfloat16)

        with (
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor._linear_bf16_bf16_fp32",
                side_effect=fake_linear,
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_all_gather_full_async",
                side_effect=fake_start,
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.compressor.cp_wait_gather_full",
                side_effect=fake_wait,
            ),
            patch.object(cmp, "_launch", side_effect=fake_launch),
        ):
            cmp.forward(x, 0, meta=meta)

        self.assertEqual(len(gather_inputs), 1)
        gathered_local, actual_ctx = gather_inputs[0]
        self.assertIs(actual_ctx, cp_ctx)
        self.assertEqual(
            tuple(gathered_local.shape), (cp_ctx.chunk_length, 2 * out_dim)
        )
        self.assertEqual(
            tuple(launch_args["kv_flat"].shape), (cp_ctx.seq_len_full, out_dim)
        )
        self.assertEqual(
            tuple(launch_args["score_flat"].shape), (cp_ctx.seq_len_full, out_dim)
        )
        self.assertTrue(
            torch.equal(launch_args["kv_flat"], gathered_fused[:, :out_dim])
        )
        self.assertTrue(
            torch.equal(launch_args["score_flat"], gathered_fused[:, out_dim:])
        )
        self.assertIs(launch_args["meta"], meta)
        self.assertIsNone(launch_args["seq_start"])


if __name__ == "__main__":
    unittest.main()

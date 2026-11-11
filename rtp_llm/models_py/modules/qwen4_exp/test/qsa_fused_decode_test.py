import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_decode_positions_triton import (
    try_prepare_decode_positions,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_interleaved_mrope,
    cached_inv_freq,
)
from rtp_llm.ops import RopeStyle


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QSAFusedDecodeTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 decode geometry")
        self.config = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=64,
            mrope_dim1=11,
            mrope_dim2=11,
            mrope_dim3=10,
            mrope_interleaved=True,
            indexer_is_neox_style=True,
            base=10000000.0,
            scale=1.0,
        )
        self.frequency = cached_inv_freq(self.config, "cuda")

    def reference(self, seq, positions):
        kwargs = dict(token_count=seq.numel(), dtype=torch.bfloat16, device=seq.device)
        cc, cs = build_interleaved_mrope(positions, self.config, **kwargs)
        block = positions.long().view(-1, 3) - seq.remainder(4)[:, None]
        bc, bs = build_interleaved_mrope(block.flatten(), self.config, **kwargs)
        visible = seq + 1
        return cc, cs, bc, bs, visible, visible // 4

    def run_fused(self, seq, positions):
        result = try_prepare_decode_positions(
            seq, positions, self.frequency, self.config, target_mrope=True
        )
        self.assertIsNotNone(result)
        return result

    def assert_outputs(self, actual, expected):
        for name, a, e in zip(
            (
                "current_cos",
                "current_sin",
                "block_cos",
                "block_sin",
                "visible",
                "compressed",
            ),
            actual,
            expected,
        ):
            self.assertEqual(a.dtype, e.dtype, name)
            self.assertEqual(a.shape, e.shape, name)
            self.assertTrue(torch.equal(a, e), name)

    def test_mrope_offsets_residues_and_graph_replay(self):
        for batch in (1, 16, 32, 64):
            seq = torch.full((batch,), 2048, dtype=torch.int32, device="cuda")
            pos = torch.full((batch * 3,), 2048, dtype=torch.int32, device="cuda")
            self.run_fused(seq, pos)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.run_fused(seq, pos)
            try:
                # Includes cache page changes and the 512 -> 513 pooled-key edge.
                for start in (0, 124, 2044, 2048, 2052, 32764):
                    seq.copy_(torch.arange(batch, device="cuda") + start)
                    # Three unequal axes also test the full interleaved builder;
                    # runtime continuation ordinarily carries three equal axes.
                    pos.copy_(
                        (
                            seq[:, None] + torch.tensor([101, 203, 307], device="cuda")
                        ).flatten()
                    )
                    expected = self.reference(seq, pos)
                    graph.replay()
                    self.assert_outputs(actual, expected)
            finally:
                graph.reset()

    def test_fallback_and_position_dtypes(self):
        seq = torch.arange(16, dtype=torch.int32, device="cuda") + 2047
        pos = (seq[:, None] - 21).expand(-1, 3).contiguous().flatten()
        for dtype in (torch.int32, torch.int64):
            self.assert_outputs(
                self.run_fused(seq, pos.to(dtype)), self.reference(seq, pos)
            )
        self.config.mrope_interleaved = False
        self.assertIsNone(
            try_prepare_decode_positions(
                seq, pos, self.frequency, self.config, target_mrope=True
            )
        )

    def test_combined_prepare_snapshot_writer_and_rollback_graph(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.decode_prepare_triton import (
            try_prepare_decode,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
            restore_indexer_cache,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
            write_decode_key_,
            write_decode_key_with_undo_,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_norm_rope import (
            try_indexer_norm_rope,
        )
        from rtp_llm.models_py.modules.qwen4_exp.test.target_writer_fusion_test import (
            TargetWriterFusionTest,
        )

        for batch in (16, 32, 64):
            args = TargetWriterFusionTest.case(self, [128] * batch, 1, strided=True)
            seq, kv, kt, state, st = args[1], *args[5:]
            pos = seq[:, None].expand(-1, 3).contiguous().flatten()
            saved_kv, saved_state = kv.clone(), state.clone()
            projection = torch.randn(batch, 640, device="cuda", dtype=torch.bfloat16)
            query = projection[:, :512].view(batch, 4, 128)
            query_gamma = torch.randn_like(args[4]) * 0.07

            def prepare_and_write():
                result = try_prepare_decode(
                    seq,
                    pos,
                    self.frequency,
                    self.config,
                    kv,
                    kt,
                    state,
                    st,
                    query=query,
                    query_gamma=query_gamma,
                )
                self.assertIsNotNone(result)
                metadata, undo, rotated_q = result
                write_decode_key_(
                    args[0],
                    seq,
                    metadata[2],
                    metadata[3],
                    args[4],
                    kv,
                    kt,
                    state,
                    st,
                    norm_eps=1e-6,
                )
                return metadata, undo, rotated_q

            prepare_and_write()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual_metadata, actual_undo, actual_q = prepare_and_write()
            try:
                for generation in range(12):
                    seq.copy_(
                        124 + (torch.arange(batch, device="cuda") + generation) % 12
                    )
                    pos.copy_(
                        (
                            seq[:, None] + torch.tensor([101, 203, 307], device="cuda")
                        ).flatten()
                    )
                    args[0].normal_()
                    query.normal_()
                    kv.copy_(saved_kv)
                    state.copy_(saved_state)
                    expected_metadata = self.reference(seq, pos)
                    args[2:4] = expected_metadata[2:4]
                    expected_undo = write_decode_key_with_undo_(*args, norm_eps=1e-6)
                    after_kv, after_state = kv.clone(), state.clone()
                    kv.copy_(saved_kv)
                    state.copy_(saved_state)
                    graph.replay()
                    self.assert_outputs(actual_metadata, expected_metadata)
                    expected_q = try_indexer_norm_rope(
                        query,
                        expected_metadata[0],
                        expected_metadata[1],
                        query_gamma,
                        1e-6,
                    )
                    # Grouped FP32 reductions may cross a BF16 rounding boundary.
                    torch.testing.assert_close(
                        actual_q, expected_q, rtol=0.016, atol=0.016
                    )
                    self.assertTrue(torch.equal(kv, after_kv))
                    self.assertTrue(torch.equal(state, after_state))
                    for name in (
                        "state_slots",
                        "kv_slots",
                        "original_state",
                        "original_kv",
                    ):
                        self.assertTrue(
                            torch.equal(
                                getattr(actual_undo, name), getattr(expected_undo, name)
                            ),
                            name,
                        )
                    restore_indexer_cache(actual_undo)
                    self.assertTrue(torch.equal(kv, saved_kv))
                    self.assertTrue(torch.equal(state, saved_state))
                self.assertIsNone(
                    try_prepare_decode(
                        seq, pos, self.frequency, self.config, kv, kt[:, :0], state, st
                    )
                )
            finally:
                graph.reset()

    def test_raw_query_norm_rope_graph_is_exact(self):
        from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
            exact_head_rms_norm,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_norm_rope import (
            try_indexer_norm_rope,
        )

        for batch in (16, 32, 64):
            qk = torch.randn(batch, 640, dtype=torch.bfloat16, device="cuda")
            q = qk[:, :512].view(batch, 4, 128)
            gain = torch.randn(128, dtype=torch.bfloat16, device="cuda") * 0.07
            cos = torch.empty(batch, 64, dtype=torch.bfloat16, device="cuda")
            sin = torch.empty_like(cos)
            cos.uniform_(-1, 1)
            sin.uniform_(-1, 1)
            self.assertIsNotNone(try_indexer_norm_rope(q, cos, sin, gain, 1e-6))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = try_indexer_norm_rope(q, cos, sin, gain, 1e-6)
            try:
                for _ in range(8):
                    q.normal_()
                    gain.normal_(0, 0.07)
                    cos.uniform_(-1, 1)
                    sin.uniform_(-1, 1)
                    expected = apply_partial_rope(
                        exact_head_rms_norm(q, gain, 1e-6),
                        cos[:, None],
                        sin[:, None],
                    )
                    graph.replay()
                    self.assertTrue(torch.equal(actual, expected))
            finally:
                graph.reset()

    def test_score_graph_skips_only_empty_tiles(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
            qsa_paged_indexer_score,
        )

        for batch in (16, 32, 64):
            columns = 1024
            q = torch.randn(batch, 1, 4, 128, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(batch, 4, device="cuda")
            pool = torch.randn(
                (batch * 32 + 1) * 32, 128, device="cuda", dtype=torch.bfloat16
            )
            table = torch.arange(
                1, batch * 32 + 1, device="cuda", dtype=torch.int32
            ).view(batch, 32)
            table[0, 0] = 0
            table[1, 1] = -1
            table[2, 2] = batch * 32 + 9
            lengths = torch.full((batch, 1), columns, device="cuda", dtype=torch.int32)

            def score():
                return qsa_paged_indexer_score(
                    q,
                    weight,
                    pool,
                    table,
                    lengths,
                    block_size=32,
                    max_ctx_len=columns,
                    validate_block_table=False,
                )

            score()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = score()
            try:
                for length in (1024, 0, 31, 32, 511, 512, 513, 1024):
                    lengths.fill_(length)
                    q.normal_()
                    # Independent dense reference for gathered valid pages.
                    valid_page = (table > 0) & (table <= batch * 32)
                    page = table.clamp(0, batch * 32).long()
                    keys = pool.view(-1, 32, 128)[page].reshape(batch, columns, 128)
                    expected = (
                        torch.einsum("bhd,bnd->bhn", q[:, 0].float(), keys.float())
                        .relu()
                        .mul(weight[:, :, None])
                        .sum(1)
                    )
                    valid = valid_page.repeat_interleave(32, 1) & (
                        torch.arange(columns, device="cuda")[None, :] < length
                    )
                    expected.masked_fill_(~valid, -torch.inf)
                    graph.replay()
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            finally:
                graph.reset()

    def test_topk_order_tail_and_graph_replay(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
            select_qsa_paged_tokens,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.cub_topk import try_cub_qsa_tokens

        native_path = "rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_topk_triton.try_select_qsa_tokens"
        for batch in (16, 32, 64):
            for columns in (1023, 1024):
                scores = torch.randn(batch, columns, device="cuda")
                lengths = torch.full((batch,), 2048, device="cuda", dtype=torch.int32)
                self.assertIsNotNone(try_cub_qsa_tokens(scores, lengths))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = try_cub_qsa_tokens(scores, lengths)
                try:
                    for mode in range(4):
                        scores.normal_()
                        if mode == 1:
                            scores.floor_()
                        elif mode == 2:
                            scores.zero_()
                            scores[:, ::2] = -0.0
                        elif mode == 3:
                            scores[:, :6] = torch.tensor(
                                [
                                    float("nan"),
                                    -float("nan"),
                                    float("inf"),
                                    -float("inf"),
                                    0.0,
                                    -0.0,
                                ],
                                device="cuda",
                            )
                        for length in (0, 127, 2047, 2048, 2051, 2052, columns * 4):
                            lengths.fill_(length)
                            with patch(native_path, return_value=None):
                                expected = select_qsa_paged_tokens(
                                    scores,
                                    lengths,
                                    compress_ratio=4,
                                    token_budget=2048,
                                )
                            graph.replay()
                            self.assertTrue(
                                torch.equal(actual, expected),
                                (batch, columns, mode, length),
                            )
                finally:
                    graph.reset()


if __name__ == "__main__":
    unittest.main()

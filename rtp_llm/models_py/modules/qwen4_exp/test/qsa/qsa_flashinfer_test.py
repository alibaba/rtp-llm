import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
    qsa_paged_indexer_score,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_indexer import (
    select_flashinfer_prefix_blocks,
    try_flashinfer_decode_selection,
    try_flashinfer_prefill_selection,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_prefill import (
    try_flashinfer_cold_prefill,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
    build_qsa_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode import (
    try_prepare_fused_decode,
    write_decode_key_,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill import (
    try_prepare_fused_prefill,
    write_zero_prefix_prefill,
)


class QsaFlashInferTest(unittest.TestCase):
    def _prep_fixture(self, rows):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("fused QSA preparation requires SM103")
        torch.manual_seed(1234)
        rope = SimpleNamespace(
            style="Mrope",
            dim=64,
            base=1e6,
            scale=1.0,
            index_factor=3,
            mrope_interleaved=True,
            mrope_dim1=12,
            mrope_dim2=10,
            mrope_dim3=10,
            indexer_is_neox_style=True,
        )
        inv = rope.base ** (-2.0 * torch.arange(32, device="cuda") / 64)
        joint = torch.randn((rows, 9, 128), device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1
        return rope, inv, joint, gamma

    def _prep_rope(self, positions, rope):
        return build_qsa_rope(
            positions.flatten(),
            rope,
            token_count=positions.shape[0],
            dtype=torch.bfloat16,
            device=positions.device,
        )

    def test_model_routes_prefill_to_fused_indexer(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpAttention
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
            Qwen4ExpQSARuntimeContext,
        )

        rope, _, joint, gamma = self._prep_fixture(4)
        hidden = joint.new_empty((4, 32))
        indexer = Qwen4ExpQSAIndexer(
            joint.new_empty((1152, 32)),
            gamma,
            gamma,
            n_heads=8,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            rope_config=rope,
        )
        attention = Qwen4ExpAttention.__new__(Qwen4ExpAttention)
        torch.nn.Module.__init__(attention)
        attention.qsa_indexer = indexer
        attention.qsa_rope_config = rope
        attention.is_mtp_draft = False
        inputs = SimpleNamespace(is_prefill=True, is_target_verify=False)
        ctx = Qwen4ExpQSARuntimeContext(None, inputs, None, inputs, None, inputs)
        q, raw = joint[:, :8], joint[:, 8]
        selected = torch.empty(4, 2051, device="cuda", dtype=torch.int32)
        with (
            patch.dict(
                "os.environ",
                {
                    "QWEN4_QSA_FUSED_INDEXER_PREP": "1",
                    "QWEN4_QSA_INDEXER_BACKEND": "flashinfer",
                },
            ),
            patch(
                "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_prefill.try_fused_prefill_selection",
                autospec=True,
                return_value=selected,
            ) as prepare,
        ):
            self.assertIs(
                attention._qsa_selected_indices(hidden, ctx, projected=(q, raw)),
                selected,
            )
        prepare.assert_called_once_with(ctx, q, raw, indexer, rope)

    def test_fused_cold_prefill_matches_dense_keys_and_cache(self):
        rows = 129
        rope, inv, joint, gamma = self._prep_fixture(rows)
        q, raw = joint[:, :8], joint[:, 8]
        cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        prefixes = cu[:1]
        positions = (
            torch.arange(rows, device="cuda")[:, None] + torch.arange(3, device="cuda")
        ).int()
        table = torch.tensor([[2, 1]], device="cuda", dtype=torch.int32)
        kv = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(3, 8, 128, device="cuda")
        coords = torch.full((3, 8, 3), -1, device="cuda", dtype=torch.int32)
        expected_kv, expected_state = kv.clone(), state.clone()
        cos, sin = self._prep_rope(positions, rope)
        expected = write_zero_prefix_prefill(
            raw,
            cu,
            [rows],
            cos[None],
            sin[None],
            gamma,
            expected_kv,
            table,
            expected_state,
            table,
            page_size=128,
            norm_eps=1e-6,
            emit_dense_keys=True,
        )
        rotated, dense, visible, compressed = try_prepare_fused_prefill(
            q,
            raw,
            cu,
            prefixes,
            positions,
            gamma,
            gamma,
            inv,
            state,
            table,
            kv,
            table,
            coords,
            table,
            max_length=rows,
            norm_eps=1e-6,
            rope_config=rope,
            emit_dense=True,
        )
        self.assertIsNone(visible)
        self.assertIsNone(compressed)
        expected_q = apply_partial_rope(
            exact_head_rms_norm(q, gamma, 1e-6),
            cos[:, None],
            sin[:, None],
        )
        for actual, reference in (
            (rotated, expected_q),
            (dense, expected["block_keys"]),
            (kv, expected_kv),
            (state, expected_state),
            (coords[2], positions[120:128]),
            (coords[1, 0], positions[128]),
        ):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_fused_prefill_prepares_mixed_prefix_and_cold_rows(self):
        rows = 136
        rope, inv, joint, gamma = self._prep_fixture(rows)
        q, raw = joint[:, :8], joint[:, 8]
        cu = torch.tensor([0, 129, rows], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
        logical = torch.cat(
            (torch.arange(129, device="cuda"), torch.arange(128, 135, device="cuda"))
        )
        positions = (logical[:, None] + torch.arange(3, device="cuda")).int()
        kv = torch.randn(6, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(6, 8, 128, device="cuda")
        coords = torch.full((6, 8, 3), -1, device="cuda", dtype=torch.int32)
        before_k, before_state, before_coords = (
            kv.clone(),
            state.clone(),
            coords.clone(),
        )
        kt = torch.tensor([[4, 3], [1, 2]], device="cuda", dtype=torch.int32)
        st = torch.tensor([[3, 2], [4, 1]], device="cuda", dtype=torch.int32)
        pt = torch.tensor([[2, 1], [4, 3]], device="cuda", dtype=torch.int32)
        prepared = try_prepare_fused_prefill(
            q,
            raw,
            cu,
            prefixes,
            positions,
            gamma,
            gamma,
            inv,
            state,
            st,
            kv,
            kt,
            coords,
            pt,
            max_length=129,
            norm_eps=1e-6,
            rope_config=rope,
            emit_dense=False,
        )
        self.assertIsNotNone(prepared)
        rotated, dense, visible, compressed = prepared
        self.assertIsNone(dense)
        cos, sin = self._prep_rope(positions, rope)
        expected_q = apply_partial_rope(
            exact_head_rms_norm(q, gamma, 1e-6),
            cos[:, None],
            sin[:, None],
        )
        torch.testing.assert_close(rotated, expected_q, atol=0, rtol=0)
        torch.testing.assert_close(visible, logical.int() + 1)
        torch.testing.assert_close(compressed, prefixes.new_tensor([32, 33]))
        for source, count, page in ((0, 32, 4), (129, 1, 2)):
            groups = raw[source : source + 4 * count].unflatten(0, (-1, 4))
            pooled = exact_head_rms_norm(groups.float().mean(1).bfloat16(), gamma, 1e-6)
            expected = apply_partial_rope(
                pooled,
                cos[source : source + 4 * count : 4],
                sin[source : source + 4 * count : 4],
            )
            before_k[page, :count].copy_(expected)
        for start, end, state_id, pos_id in (
            (120, 128, 3, 2),
            (128, 129, 2, 1),
            (129, 136, 1, 3),
        ):
            before_state[state_id, : end - start].copy_(raw[start:end])
            before_coords[pos_id, : end - start].copy_(positions[start:end])
        for actual, expected in (
            (kv, before_k),
            (state, before_state),
            (coords, before_coords),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        block_ids = select_flashinfer_prefix_blocks(
            rotated,
            kv,
            kt,
            cu,
            prefixes,
            compressed,
            129,
            33,
        )
        for row in (0, 3, 128, 129, 132, 135):
            ids = block_ids[row]
            wanted = torch.arange(
                (logical[row].item() + 1) // 4, device="cuda", dtype=torch.int32
            )
            torch.testing.assert_close(ids[ids >= 0].sort().values, wanted)

    def test_fused_decode_replays_with_live_lengths_and_positions(self):
        rope, inv, joint, gamma = self._prep_fixture(4)
        q, raw = joint[:, :8], joint[:, 8]
        lengths = torch.tensor([128, 129, 132, 1], device="cuda", dtype=torch.int32)
        xyz = (lengths[:, None] + torch.arange(3, device="cuda")).int()
        table = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [0, 0]], device="cuda", dtype=torch.int32
        )
        kv = torch.randn(7, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(7, 8, 128, device="cuda")
        coords = torch.randint(0, 512, (7, 8, 3), device="cuda", dtype=torch.int32)
        initial = tuple(x.clone() for x in (kv, state, coords))

        def prepare():
            return try_prepare_fused_decode(
                q,
                raw,
                lengths,
                xyz,
                gamma,
                gamma,
                inv,
                state,
                table,
                kv,
                table,
                coords,
                table,
                norm_eps=1e-6,
                rope_config=rope,
                padded=True,
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.assertIsNotNone(prepare())
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            rotated, visible = prepare()
        for actual, before in zip((kv, state, coords), initial):
            actual.copy_(before)
        lengths.copy_(lengths.new_tensor([132, 128, 129, 2]))
        xyz.add_(7)
        table.copy_(table.new_tensor([[5, 6], [1, 2], [3, 4], [0, 0]]))
        joint.normal_()
        graph.replay()
        cos, sin = self._prep_rope(xyz, rope)
        q_expected = apply_partial_rope(
            exact_head_rms_norm(q, gamma, 1e-6),
            cos[:, None],
            sin[:, None],
        )
        torch.testing.assert_close(rotated, q_expected, atol=0, rtol=0)
        torch.testing.assert_close(visible, lengths.new_tensor([132, 128, 129, 0]))
        expected_kv, expected_state, expected_coords = initial
        source = torch.stack(
            (expected_coords[6, 0], expected_coords[1, 4], xyz[2], xyz[3])
        )
        kcos, ksin = self._prep_rope(source, rope)
        write_decode_key_(
            raw.contiguous(),
            lengths - 1,
            kcos,
            ksin,
            gamma,
            expected_kv,
            table,
            expected_state,
            table,
            norm_eps=1e-6,
        )
        for row, block, slot in ((0, 6, 3), (1, 1, 7), (2, 4, 0)):
            expected_coords[block, slot].copy_(xyz[row])
        for actual, expected in zip((kv, state, coords), initial):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_indexer_selects_segment_local_causal_blocks_and_tail(self):
        tokens = 2055  # 513 complete blocks plus a partial tail.
        q = torch.zeros((1, tokens, 8, 128), device="cuda", dtype=torch.bfloat16)
        q[..., 0], q[..., 1] = 128, 1
        keys = q.new_zeros((1, tokens // 4, 128))
        blocks = torch.arange(tokens // 4, device=q.device)
        keys[0, :, 0], keys[0, :, 1] = blocks // 128, blocks % 128

        with patch.dict("os.environ", {"QWEN4_QSA_INDEXER_CHUNK_SIZE": "1024"}):
            actual = try_flashinfer_prefill_selection(q, keys)[0]
        with patch.dict("os.environ", {"QWEN4_QSA_INDEXER_CHUNK_SIZE": "0"}):
            unchunked = try_flashinfer_prefill_selection(q, keys)[0]
        torch.testing.assert_close(
            actual.sort(dim=-1).values, unchunked.sort(dim=-1).values
        )
        for length in (1, 3, 4, 2049, 2055):
            complete, tail = divmod(length, 4)
            first = 4 * max(0, complete - 512)
            expected = torch.arange(first, length, device=q.device, dtype=torch.int32)
            row = actual[length - 1]
            torch.testing.assert_close(row[row >= 0].sort().values, expected)
            torch.testing.assert_close(
                row[2048 : 2048 + tail], expected[expected.numel() - tail :]
            )

    def test_paged_indexer_replays_with_live_lengths_and_missing_pages(self):
        rows, pages = 4, 257
        q = torch.zeros((rows, 8, 128), device="cuda", dtype=torch.bfloat16)
        q[..., 0], q[..., 1] = 128, 1
        cache = q.new_zeros((pages + 1, 32, 128))
        blocks = torch.arange(pages * 32, device=q.device).view(pages, 32)
        cache[1:, :, 0], cache[1:, :, 1] = blocks // 128, blocks % 128
        cache[0].fill_(10000)  # Reserved page must never win TopK.
        table = torch.arange(1, pages + 1, device=q.device, dtype=torch.int32)
        table = table.repeat(rows, 1)
        lengths = torch.tensor(
            [32768, 32769, 2055, 0], device=q.device, dtype=torch.int32
        )
        weights = torch.full((rows, 8), 128**-0.5, device=q.device)

        def reference():
            scores = qsa_paged_indexer_score(
                q.unsqueeze(1),
                weights,
                cache.flatten(0, 1),
                table,
                lengths[:, None] // 4,
                block_size=32,
                max_ctx_len=pages * 32,
                validate_block_table=False,
            )
            return (
                select_qsa_paged_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    validate_lengths=False,
                    scores_are_masked=True,
                )
                .sort(dim=-1)
                .values
            )

        eager = try_flashinfer_decode_selection(q, cache, table, lengths)
        torch.testing.assert_close(eager.sort(dim=-1).values, reference())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = try_flashinfer_decode_selection(q, cache, table, lengths)
        lengths.copy_(torch.tensor([32761, 2055, 131, 3], dtype=torch.int32))
        table[0, 255] = 0
        table[1, 0] = -1
        table[2, 0] = cache.shape[0]
        graph.replay()
        torch.testing.assert_close(captured.sort(dim=-1).values, reference())

    def test_cold_prefill_matches_ragged_grouped_attention(self):
        torch.manual_seed(42)
        lengths = [3, 5]
        heads, kv_heads, dim = 8, 2, 256
        packed = torch.randn(
            (sum(lengths), (heads + 2 * kv_heads) * dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q, k, v = packed.split((heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
        selected = torch.tensor(
            [[0, -1], [1, 0], [0, 2], [0, -1], [0, 1], [2, -1], [1, 3], [0, 4]],
            device="cuda",
            dtype=torch.int32,
        )
        impl = SimpleNamespace(
            _is_mtp_draft=False,
            attn_inputs=SimpleNamespace(is_cuda_graph=False),
            head_dim=dim,
            head_num=heads,
            kv_head_num=kv_heads,
            q_width=heads * dim,
        )
        actual = try_flashinfer_cold_prefill(
            impl, q, k, v, selected, lengths, trusted_selected=True
        )
        self.assertIsNotNone(actual)
        expected = []
        offset = 0
        group = torch.arange(heads, device="cuda") // (heads // kv_heads)
        for length in lengths:
            end = offset + length
            query = q[offset:end].reshape(length, heads, dim).float()
            keys = k[offset:end].reshape(length, kv_heads, dim).float()[:, group]
            values = v[offset:end].reshape(length, kv_heads, dim).float()[:, group]
            ids = selected[offset:end]
            mask = (ids[:, :, None] == torch.arange(length, device="cuda")).any(dim=1)
            logits = torch.einsum("qhd,khd->qhk", query, keys) / dim**0.5
            weights = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
            expected.append(torch.einsum("qhk,khd->qhd", weights, values))
            offset = end
        expected = torch.cat(expected).reshape_as(actual).to(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)


if __name__ == "__main__":
    unittest.main()

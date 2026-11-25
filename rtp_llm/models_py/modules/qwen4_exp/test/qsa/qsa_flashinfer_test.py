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
        self._check_fused_cold_prefill()

    def test_fused_cold_draft_prefill_matches_dense_keys_and_cache(self):
        for factor in (1, 3):
            with self.subTest(position_factor=factor):
                self._check_fused_cold_prefill(draft_factor=factor)

    def _check_fused_cold_prefill(self, *, draft_factor=None):
        rows = 129
        rope, inv, joint, gamma = self._prep_fixture(rows)
        q, raw = joint[:, :8], joint[:, 8]
        cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        prefixes = cu[:1]
        if draft_factor is None:
            positions = (
                torch.arange(rows, device="cuda")[:, None]
                + torch.arange(3, device="cuda")
            ).int()
        else:
            rope.style = "Base"
            rope.index_factor = draft_factor
            positions = torch.arange(13, rows + 13, device="cuda", dtype=torch.int32)[
                :, None
            ].repeat(1, draft_factor)
        table = torch.tensor([[2, 1]], device="cuda", dtype=torch.int32)
        kv = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(3, 8, 128, device="cuda")
        coords = (
            torch.full((3, 8, 3), -1, device="cuda", dtype=torch.int32)
            if draft_factor is None
            else None
        )
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
            table if coords is not None else None,
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
        ):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        if coords is not None:
            torch.testing.assert_close(coords[2], positions[120:128], atol=0, rtol=0)
            torch.testing.assert_close(coords[1, 0], positions[128], atol=0, rtol=0)

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

    def test_fused_target_decode_matches_verify_and_replay(self):
        from rtp_llm.models_py.kernels.cuda.qsa.indexer_norm_rope import _norm_rope
        from rtp_llm.models_py.kernels.cuda.qsa.position_state import (
            store_positions,
            target_position_slots,
        )
        from rtp_llm.models_py.kernels.cuda.qsa.target_writer import (
            _snapshot_target_window,
            _write_target_window,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
            IndexerCacheUndo,
            restore_indexer_cache,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode import (
            try_prepare_fused_target_decode,
        )

        for width in (1, 2, 3, 4):
            rope, inv, joint, gamma = self._prep_fixture(width)
            rope.scale = 1.75
            q, raw = joint[:, :8], joint[:, 8]
            prefix = torch.tensor([127], device="cuda", dtype=torch.int32)
            cu = prefix.new_tensor([0, width])
            order = torch.arange(width, device="cuda", dtype=torch.int32)
            positions = (
                prefix[:, None] + 17 * order[:, None] + prefix.new_tensor([8, 37, 69])
            ).contiguous()
            table = prefix.new_tensor([[1, 2, 3]])
            kv = q.new_empty((4, 32, 128)).normal_()
            state = q.new_empty((4, 8, 128)).normal_().float()
            coords = torch.randint(
                100, 200_000, (4, 8, 3), device="cuda", dtype=torch.int32
            )
            visible = prefix.new_empty((1, width))
            compressed = torch.empty_like(visible)
            empty_table = torch.zeros_like(table)
            padding_tables = (table, empty_table) if width % 2 else (empty_table, table)
            pools = kv, state, coords

            def prepare():
                slots, history, current = target_position_slots(
                    prefix,
                    cu,
                    table,
                    width,
                    coords.shape[0],
                    lengths=(visible, compressed),
                    padding_tables=padding_tables,
                )
                result = try_prepare_fused_target_decode(
                    q,
                    raw,
                    prefix,
                    positions,
                    gamma,
                    gamma,
                    inv,
                    state,
                    table,
                    kv,
                    table,
                    coords,
                    visible,
                    slots,
                    history,
                    current,
                    norm_eps=1e-6,
                    rope_config=rope,
                )
                self.assertIsNotNone(result)
                return slots, result

            def reference(initial):
                ref_kv, ref_state, ref_coords = (x.clone() for x in initial)
                lookup = target_position_slots(
                    prefix, cu, table, width, coords.shape[0]
                )
                write_slots, history_slots, current_rows = lookup
                history = ref_coords.flatten(0, 1)[history_slots]
                current = positions[current_rows.clamp_min(0)]
                group_positions = torch.where(
                    current_rows[:, None] >= 0, current, history
                )
                old_positions = torch.where(
                    write_slots[:, None] >= 0,
                    ref_coords.flatten(0, 1)[write_slots.clamp_min(0)],
                    0,
                )
                axes = torch.arange(32, device=q.device)
                axis = torch.zeros_like(axes)
                axis[1 : 3 * rope.mrope_dim2 : 3] = 1
                axis[2 : 3 * rope.mrope_dim3 : 3] = 2

                def cos_sin(xyz):
                    angle = (xyz[:, axis].float() * (1 / rope.scale)) * inv
                    cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
                    return cos.repeat(1, 2), sin.repeat(1, 2)

                kcos, ksin = cos_sin(group_positions)
                qcos, qsin = cos_sin(positions)
                store_positions(ref_coords, write_slots, positions)
                undo = IndexerCacheUndo(
                    state_pool=ref_state,
                    state_slots=torch.empty_like(write_slots),
                    original_state=torch.empty_like(raw, dtype=ref_state.dtype),
                    kv_pool=ref_kv,
                    kv_slots=torch.empty_like(prefix, dtype=torch.int64),
                    original_kv=q.new_empty((1, 128)),
                    padded_slots=True,
                )
                layout = dict(
                    Q=width,
                    KTW=table.shape[1],
                    STW=table.shape[1],
                    KB=ref_kv.shape[0],
                    SB=ref_state.shape[0],
                    BQ=1 << (width - 1).bit_length(),
                    num_warps=4,
                )
                _snapshot_target_window[(1,)](
                    prefix,
                    ref_kv,
                    table,
                    ref_state,
                    table,
                    undo.state_slots,
                    undo.original_state,
                    undo.kv_slots,
                    undo.original_kv,
                    **layout,
                )
                _write_target_window[(1,)](
                    raw,
                    prefix,
                    kcos,
                    ksin,
                    gamma,
                    ref_kv,
                    table,
                    ref_state,
                    table,
                    RAW_STRIDE=raw.stride(0),
                    EPS=1e-6,
                    **layout,
                )
                rotated = torch.empty(q.shape, device=q.device, dtype=q.dtype)
                _norm_rope[(width, 8)](
                    q,
                    gamma,
                    qcos,
                    qsin,
                    rotated,
                    HEADS=8,
                    STRIDE=q.stride(0),
                    EPS=1e-6,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
                return rotated, undo, old_positions, (ref_kv, ref_state, ref_coords)

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                prepare()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                slots, (rotated, lengths, undo, old_positions) = prepare()

            for page in (124, 125, 126, 127, 128, 255, 256, 127):
                padded = page == 256
                prefix.fill_(page)
                positions.copy_(
                    prefix[:, None]
                    + 17 * order[:, None]
                    + prefix.new_tensor([81, 37, 69])
                )
                if padded:
                    table.zero_()
                else:
                    table.copy_(
                        table.new_tensor(
                            [[1 + page % 3, 1 + (page + 1) % 3, 1 + (page + 2) % 3]]
                        )
                    )
                joint.normal_()
                gamma.normal_(0, 0.1)
                inv.mul_(0.97)
                kv.normal_()
                state.copy_(state.to(torch.bfloat16).normal_())
                coords.random_(100, 200_000)
                initial = tuple(x.clone() for x in pools)
                expected_q, expected_undo, expected_positions, expected_pools = (
                    reference(initial)
                )
                graph.replay()
                with self.subTest(width=width, prefix=page, padded=padded):
                    expected_visible = (
                        torch.zeros_like(visible)
                        if padded
                        else prefix[:, None] + order + 1
                    )
                    torch.testing.assert_close(
                        visible, expected_visible, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        compressed, expected_visible // 4, atol=0, rtol=0
                    )
                    self.assertTrue(
                        torch.equal(
                            rotated.view(torch.uint8), expected_q.view(torch.uint8)
                        )
                    )
                    torch.testing.assert_close(
                        lengths, expected_visible[:, 0] + width - 1
                    )
                    torch.testing.assert_close(old_positions, expected_positions)
                    for field in (
                        "state_slots",
                        "original_state",
                        "kv_slots",
                        "original_kv",
                    ):
                        torch.testing.assert_close(
                            getattr(undo, field),
                            getattr(expected_undo, field),
                            atol=0,
                            rtol=0,
                        )
                    torch.testing.assert_close(pools, expected_pools, atol=0, rtol=0)
                    restore_indexer_cache(undo)
                    store_positions(coords, slots, old_positions)
                    torch.testing.assert_close(pools, initial, atol=0, rtol=0)
            graph.reset()

        with self.subTest("two requests cannot share one writer's ownership"):
            before = tuple(x.clone() for x in pools)
            prefixes = prefix.repeat(2)
            repeated_table = table.repeat(2, 1)
            slots, history, current = target_position_slots(
                prefixes,
                cu.new_tensor([0, width, 2 * width]),
                repeated_table,
                2 * width,
                coords.shape[0],
            )
            self.assertIsNone(
                try_prepare_fused_target_decode(
                    q.repeat(2, 1, 1),
                    raw.repeat(2, 1),
                    prefixes,
                    positions.repeat(2, 1),
                    gamma,
                    gamma,
                    inv,
                    state,
                    repeated_table,
                    kv,
                    repeated_table,
                    coords,
                    visible.repeat(2, 1),
                    slots,
                    history,
                    current,
                    norm_eps=1e-6,
                    rope_config=rope,
                )
            )
            torch.testing.assert_close(pools, before, atol=0, rtol=0)

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

    def test_qtoken_g8_paged_gate_and_live_decode_metadata(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_qtoken import (
            PagedQTokenG8,
        )

        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Qwen4 QToken G8 requires SM100 or SM103")
        torch.manual_seed(47)
        pool = torch.randn((5, 262272), device="cuda", dtype=torch.bfloat16) * 0.2
        table = torch.tensor([[1], [3]], device="cuda", dtype=torch.int32)
        keys, values = PagedQTokenG8._cache_views(pool)
        slots = torch.arange(2051, device="cuda", dtype=torch.int32)
        for is_prefill, lengths in ((True, [9, 3]), (False, [1, 1])):
            with self.subTest(prefill=is_prefill):
                cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
                prefixes = torch.tensor([11, 69], dtype=torch.int32)
                visible = torch.cat(
                    [torch.arange(p + 1, p + n + 1) for p, n in zip(prefixes, lengths)]
                ).cuda()
                selected = torch.where(slots < visible[:, None], slots, -1).int()
                inputs = SimpleNamespace(
                    is_prefill=is_prefill,
                    is_target_verify=False,
                    is_cuda_graph=not is_prefill,
                    cu_seqlens_device=cu.cuda(),
                    decode_cu_seqlens_device=cu.cuda(),
                    input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    prefix_lengths=prefixes,
                    prefix_lengths_device=prefixes.cuda(),
                    sequence_lengths_plus_1_device=(prefixes + 1).cuda(),
                    kv_cache_kernel_block_id_device=table,
                )
                query = (
                    torch.randn(
                        (sum(lengths), 64, 256), device="cuda", dtype=pool.dtype
                    )
                    * 0.2
                )
                gate = torch.randn_like(query)
                backend = PagedQTokenG8(
                    inputs, query.shape, pool, split_kv=not is_prefill
                )

                def run():
                    backend.prepare(query.flatten(1), selected, pool)
                    return backend.attend(pool, gate).view_as(query)

                def reference():
                    result = []
                    row = 0
                    for page, count in zip(table.flatten().tolist(), lengths):
                        for _ in range(count):
                            end = visible[row].item()
                            q = query[row].float().view(4, 16, 256)
                            k, v = (
                                keys[page, :, :end].float(),
                                values[page, :, :end].float(),
                            )
                            scores = torch.einsum("hgd,hsd->hgs", q, k) * 0.0625
                            result.append(
                                torch.einsum("hgs,hsd->hgd", scores.softmax(-1), v)
                            )
                            row += 1
                    rounded = torch.stack(result).view_as(query).bfloat16()
                    return (rounded.float() * gate.float().sigmoid()).bfloat16()

                torch.testing.assert_close(run(), reference(), rtol=0.02, atol=0.003)
                if not is_prefill:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = run()
                    table.copy_(table.flip(0))
                    visible.add_(3)
                    inputs.sequence_lengths_plus_1_device.copy_(visible)
                    selected.copy_(torch.where(slots < visible[:, None], slots, -1))
                    query.neg_()
                    gate.add_(0.5)
                    graph.replay()
                    torch.testing.assert_close(output, run(), rtol=0, atol=0)
                    torch.testing.assert_close(
                        output, reference(), rtol=0.02, atol=0.003
                    )

    def test_qtoken_draft_base_writer_matches_native(self):
        from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
            ZeroCenteredFusedQKRMSNorm,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_qtoken import (
            PagedQTokenG8,
        )
        from rtp_llm.ops import AttentionConfigs, RopeCache, RopeStyle
        from rtp_llm.ops.fused_rope_kvcache_op import (
            FusedRopeKVCacheDecodeOp,
            FusedRopeKVCachePrefillOpQKVOut,
            _get_fused_rope_kvcache,
        )

        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Qwen4 QToken G8 requires SM100 or SM103")
        torch.manual_seed(73)
        config = AttentionConfigs()
        config.head_num, config.kv_head_num, config.size_per_head = 64, 4, 256
        config.kernel_tokens_per_block, config.max_seq_len = 128, 16464
        rope = config.rope_config
        rope.style = RopeStyle.Base
        rope.dim, rope.index_factor, rope.base = 64, 3, 10000000.0
        rope.scale, rope.offset = 1.0, 0
        norm = ZeroCenteredFusedQKRMSNorm(
            torch.randn(256, device="cuda") * 0.1,
            torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1,
            64,
            4,
            256,
        )
        with patch(
            "rtp_llm.ops.fused_rope_kvcache_op.get_rope_cache_once",
            return_value=RopeCache(),
        ):
            for prefill, graph, rows, live, prefix in (
                (True, False, 129, 129, 0),
                (False, True, 1, 1, 127),
                (True, True, 4, 3, 127),
            ):
                with self.subTest(prefill=prefill, graph=graph):
                    length = torch.tensor([live], dtype=torch.int32)
                    prefixes = torch.tensor([prefix], dtype=torch.int32)
                    cu = torch.tensor([0, live], device="cuda", dtype=torch.int32)
                    visible = torch.arange(1, rows + 1, device="cuda") + prefix
                    slots = torch.arange(2051, device="cuda", dtype=torch.int32)
                    selected = torch.where(slots < visible[:, None], slots, -1).int()
                    # Padded replay reserves the production page-table capacity
                    # even when only the first two logical pages are live.
                    table_columns = 129 if prefill and graph else 2
                    table = torch.zeros(
                        (1, table_columns), device="cuda", dtype=torch.int32
                    )
                    table[0, :2] = torch.tensor(
                        [1, 3], device="cuda", dtype=torch.int32
                    )
                    positions = torch.full(
                        (rows, 3), 333, device="cuda", dtype=torch.int32
                    )
                    positions[:, 0] = torch.arange(rows, device="cuda") * 3 + 9
                    if rows > 1:
                        positions[1, 0] = -1
                    qkv = torch.randn(
                        rows, 72 * 256, device="cuda", dtype=torch.bfloat16
                    )
                    initial = torch.randn(
                        5, 2, 4, 128, 256, device="cuda", dtype=qkv.dtype
                    )
                    actual_pool = initial.clone()
                    reference_pool = initial.clone()
                    metadata = SimpleNamespace(
                        is_prefill=prefill,
                        is_target_verify=False,
                        is_cuda_graph=graph,
                        cu_seqlens_device=cu,
                        decode_cu_seqlens_device=cu,
                        input_lengths=length,
                        prefix_lengths=prefixes,
                        prefix_lengths_device=prefixes.cuda(),
                        sequence_lengths_plus_1_device=(prefixes + 1).cuda(),
                        kv_cache_kernel_block_id_device=table,
                    )
                    actual = PagedQTokenG8(
                        metadata, qkv.shape, actual_pool, split_kv=graph
                    )
                    reference = PagedQTokenG8(
                        metadata, qkv.shape, reference_pool, split_kv=graph
                    )
                    if prefill:
                        native = FusedRopeKVCachePrefillOpQKVOut(config)
                    else:
                        native = FusedRopeKVCacheDecodeOp(config)
                        native._sequence_lengths_device = prefixes.cuda()

                    def run():
                        actual.prepare(
                            qkv,
                            selected,
                            actual_pool,
                            positions,
                            norm,
                            base_rope=True,
                            zero_tail=True,
                        )

                    def compare():
                        params = SimpleNamespace(
                            kv_cache_offset=_get_fused_rope_kvcache().convert_offset_to_block_array(
                                table
                            ),
                            kv_cache_offset_h=None,
                            padding_offset=None,
                            position_ids=positions[:live].flatten(),
                            cu_seqlens=cu,
                            sequence_lengths=(
                                native._sequence_lengths_device
                                if not prefill
                                else length
                            ),
                            prefix_lengths=prefixes,
                            max_seq_len=live,
                            max_prefix_length=int(prefixes[0]),
                        )
                        cache = SimpleNamespace(
                            kv_cache_base=reference_pool, kv_scale_base=None
                        )
                        expected_qkv = native.forward(
                            norm(qkv.clone())[:live], cache, params
                        )
                        reference.prepare(
                            expected_qkv, selected, reference_pool, zero_tail=True
                        )
                        torch.testing.assert_close(
                            actual.query, reference.query, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual_pool, reference_pool, rtol=0, atol=0
                        )

                    run()
                    compare()
                    if graph:
                        replay = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(replay):
                            run()
                        actual_pool.copy_(initial)
                        reference_pool.copy_(initial)
                        qkv.neg_()
                        positions[:, 0].add_(17)
                        table[:, :2].copy_(table[:, :2].flip(1))
                        if prefill:
                            live -= 1
                            length.fill_(live)
                            cu[1] = live
                        replay.replay()
                        compare()

    def test_qtoken_gate_preserves_fp32_sigmoid_until_store(self):
        from rtp_llm.models_py.kernels.cuda.qsa.qtoken_gate import apply_qtoken_gate

        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("Qwen4 QToken G8 requires SM100 or SM103")
        output = torch.full((1, 64, 256), 1.5, device="cuda", dtype=torch.bfloat16)
        gate = torch.full_like(output, 0.5)
        # Rounding sigmoid(0.5) before multiplying would store 0.9296875.
        expected = torch.full_like(output, 0.93359375)
        torch.testing.assert_close(
            apply_qtoken_gate(output, gate), expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    cache_layout,
    gather_index_rows,
    gather_main_rows,
    quantize_main_index_rows,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    q8kv4_sparse_decode_attention,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.q8kv4_decode import (
    _logical_decode_block_table,
    q8kv4_paged_sparse_decode,
)


class TestLogicalDecodeBlockTable(unittest.TestCase):
    def test_graph_reserve_is_a_zero_copy_strided_view(self):
        table = torch.arange(2 * 8199, dtype=torch.int32).reshape(2, 8199)
        view = _logical_decode_block_table(table, 128, 1048576)
        self.assertEqual(view.shape, (2, 8192))
        self.assertEqual(view.data_ptr(), table.data_ptr())
        self.assertEqual(view.stride(), (8199, 1))
        self.assertTrue(torch.equal(view[1], table[1, :8192]))

    def test_no_implicit_crop_or_change_to_existing_geometry(self):
        table = torch.zeros(2, 8199, dtype=torch.int32)
        self.assertIs(_logical_decode_block_table(table, 128, None), table)
        small = table[:, :1031]
        self.assertIs(_logical_decode_block_table(small, 128, 131072), small)
        with self.assertRaisesRegex(ValueError, "8192 logical"):
            _logical_decode_block_table(table, 128, 1048577)
        with self.assertRaisesRegex(ValueError, "positive"):
            _logical_decode_block_table(table, 128, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestQ8KV4Decode(unittest.TestCase):
    def test_one_million_graph_reserve_matches_logical_page_table(self):
        # Real RTP writer -> native index/TopK/attention, with poisoned reserve
        # columns. Distinct random index pages avoid ambiguous all-zero TopK
        # ties. E2E separately validates full-model context quality.
        device = torch.device("cuda")
        torch.manual_seed(20261002)
        main = torch.zeros(8192, 65536, dtype=torch.uint8, device=device)
        side = torch.zeros(8192, 17408, dtype=torch.uint8, device=device)
        layout = cache_layout(main, side, 4, 128, 128)
        slots = torch.arange(1048576, device=device)
        k = torch.zeros(1048576, 4, 128, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(
            k,
            torch.full_like(k, 6),
            torch.randn(1048576, 1, 128, dtype=torch.bfloat16, device=device),
            slots,
            layout,
            mma_scale_layout=True,
        )
        rows = 8
        q = torch.zeros(rows, 64, 128, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
        # Grouped verification uses the same index query per request here.
        idx_q[1:].copy_(idx_q[:1].expand(rows - 1, -1, -1))
        padded = torch.full((rows, 8199), 2000000000, dtype=torch.int32, device=device)
        padded[:, :8192].copy_(torch.arange(8192, device=device, dtype=torch.int32))
        baseline = padded[:, :8192].contiguous()
        lens = torch.empty(rows, dtype=torch.int32, device=device)

        def run(table, bound, width):
            return q8kv4_paged_sparse_decode(
                q,
                idx_q,
                layout,
                table,
                lens,
                indexer_dim=128,
                block_size=128,
                topk=16,
                init_blocks=1,
                local_blocks=2,
                score_type="max",
                mma_scale_layout=True,
                query_width=width,
                max_seq_len=bound,
            )

        for width in (1, 8):
            lens.copy_(
                1048576
                - torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)
            )
            run(padded, 1048576, width)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run(padded, 1048576, width)
            for length in (1048576, 1000000, 4097 * 128, 128):
                lens.copy_(
                    length
                    - torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)
                )
                reference = run(baseline, None, width)
                expected = [
                    x.clone()
                    for x in (
                        reference.index_scores,
                        reference.topk_indices,
                        reference.output,
                    )
                ]
                graph.replay()
                torch.cuda.synchronize()
                for name, left, right in zip(
                    ("scores", "topk", "output"),
                    (actual.index_scores, actual.topk_indices, actual.output),
                    expected,
                ):
                    if name == "topk":
                        # Mandatory local pages share sentinel scores; their
                        # ordering is unspecified, but the selected set isn't.
                        left = left.sort(dim=-1).values
                        right = right.sort(dim=-1).values
                    self.assertTrue(
                        torch.equal(left, right),
                        f"{name}: width={width} length={length} "
                        f"mismatches={torch.count_nonzero(left != right).item()}",
                    )
                self.assertTrue(
                    torch.equal(actual.output, torch.full_like(actual.output, 6))
                )

    def test_four_page_dispatch_matches_single_page_dynamic_graph(self):
        """Test production dispatch against retained one-page math at the boundary."""
        from unittest.mock import patch

        from rtp_llm.models_py.triton_kernels.sparse_msa.decode import (
            nvfp4_q8_attention as attention,
        )

        class SinglePageLauncher:
            def __getitem__(self, grid):
                return attention._q8kv4_selected_page_partial[(grid[0], grid[1], 16)]

        torch.manual_seed(20261002)
        device = torch.device("cuda")
        for mma in (False, True):
            main = torch.zeros(9, 65536, dtype=torch.uint8, device=device)
            side = torch.full((9, 17408), 127, dtype=torch.uint8, device=device)
            layout = cache_layout(main, side, 4, 128, 128)
            # Seed through the real writer; leave the last page tail poisoned.
            slots = torch.arange(8 * 128 - 7, device=device)
            k = torch.randn(slots.numel(), 4, 128, dtype=torch.bfloat16, device=device)
            quantize_main_index_rows(
                k,
                torch.randn_like(k),
                torch.randn(slots.numel(), 1, 128, dtype=torch.bfloat16, device=device),
                slots,
                layout,
                mma_scale_layout=mma,
            )
            views = layout.logical_views(128)
            for rows in (16, 32, 63, 64, 80, 112, 128):
                with self.subTest(mma=mma, rows=rows):
                    q = torch.randn(rows, 64, 128, device=device).to(
                        torch.float8_e4m3fn
                    )
                    table = (
                        torch.arange(8, device=device, dtype=torch.int32)
                        .expand(rows, -1)
                        .contiguous()
                    )
                    topk = torch.full(
                        (4, rows, 16), -1, dtype=torch.int32, device=device
                    )
                    lens = torch.full(
                        (rows,), 8 * 128 - 7, dtype=torch.int32, device=device
                    )
                    valid = torch.ones(rows, dtype=torch.bool, device=device)

                    def buffers():
                        return dict(
                            out=torch.empty(
                                rows, 64, 128, dtype=torch.bfloat16, device=device
                            ),
                            partial_out=torch.empty(
                                rows,
                                4,
                                16,
                                16,
                                128,
                                dtype=torch.bfloat16,
                                device=device,
                            ),
                            partial_lse=torch.empty(rows, 4, 16, 16, device=device),
                            counts=torch.empty(
                                rows, 4, dtype=torch.int32, device=device
                            ),
                        )

                    baseline, actual = buffers(), buffers()

                    def run(workspace):
                        attention.q8kv4_sparse_decode_attention(
                            q,
                            views.main_k_fp4,
                            views.main_v_fp4,
                            views.main_k_scale,
                            views.main_v_scale,
                            table,
                            topk,
                            lens,
                            sm_scale=128**-0.5,
                            mma_scale_layout=mma,
                            valid_token_mask=valid,
                            **workspace,
                        )

                    topk[:, :, :8].copy_(
                        torch.arange(8, device=device, dtype=torch.int32)
                    )
                    run(actual)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(actual)
                    for phase in range(3):
                        topk[:, :, :8].copy_(
                            torch.arange(8, device=device, dtype=torch.int32).roll(
                                phase
                            )
                        )
                        topk[:, :, 0] = -1 if phase == 1 else 7
                        table[0, 7] = 9 if phase == 1 else 7
                        lens.copy_(
                            torch.full_like(lens, 8 * 128 - 7)
                            - torch.arange(rows, device=device) % 7
                        )
                        valid[-1] = phase != 1
                        if phase == 1:
                            q[-1].fill_(float("nan"))
                        else:
                            q[-1].zero_()
                        with patch.object(
                            attention, "_q8kv4_four_page_partial", SinglePageLauncher()
                        ):
                            run(baseline)
                        graph.replay()
                        torch.cuda.synchronize()
                        for name in actual:
                            carrier = {
                                torch.bfloat16: torch.int16,
                                torch.float32: torch.int32,
                            }.get(actual[name].dtype)
                            left = (
                                actual[name]
                                if carrier is None
                                else actual[name].view(carrier)
                            )
                            right = (
                                baseline[name]
                                if carrier is None
                                else baseline[name].view(carrier)
                            )
                            self.assertTrue(torch.equal(left, right), name)
                        self.assertTrue(
                            torch.isfinite(actual["out"][valid]).all().item()
                        )
                        if not valid[-1]:
                            self.assertEqual(
                                torch.count_nonzero(actual["out"][-1]).item(), 0
                            )

    def test_short_ragged_verify_request_permutation_is_bitwise_invariant(self):
        """Freeze post-RoPE inputs; change only compact width/request position.

        Short contexts select all pages in production TopK, so matching visible
        cache bytes and Q8 carriers must produce identical attention outputs.
        This does not establish projection/MoE or full-model equivalence.
        """
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
            build_decode_physical_slots,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        torch.manual_seed(20261001)
        device = torch.device("cuda")
        heads, dim, page = 4, 128, 128
        tables = torch.tensor(
            [[3, 5, 1, 7], [8, 2, 6, 4]], device=device, dtype=torch.int32
        )
        main = torch.zeros(9, 65536, device=device, dtype=torch.uint8)
        # Unwritten future scales deliberately contain E4M3 NaNs.
        side = torch.full((9, 17408), 127, device=device, dtype=torch.uint8)
        layout = cache_layout(main, side, heads, page, dim)
        histories = []
        queries = []
        for request in range(2):
            k = torch.randn(400, heads, dim, device=device, dtype=torch.bfloat16)
            histories.append(
                (
                    k,
                    torch.randn_like(k),
                    torch.randn(400, 1, dim, device=device, dtype=torch.bfloat16),
                )
            )
            queries.append(
                (
                    torch.randn(8, 64, dim, device=device, dtype=torch.bfloat16),
                    torch.randn(8, heads, dim, device=device, dtype=torch.bfloat16),
                )
            )

        def inputs(prefix, order, widths):
            request_prefixes = [prefix if request == 0 else 313 for request in order]
            cu = [0]
            for width in widths:
                cu.append(cu[-1] + width)
            return (
                torch.cat(
                    [
                        queries[request][0][:width]
                        for request, width in zip(order, widths)
                    ]
                ),
                torch.cat(
                    [
                        queries[request][1][:width]
                        for request, width in zip(order, widths)
                    ]
                ),
                tables[list(order)].contiguous(),
                torch.tensor(request_prefixes, device=device, dtype=torch.int32),
                torch.tensor(cu, device=device, dtype=torch.int32),
            )

        def run(data):
            q, idx_q, table, prefix, cu = data
            expanded, positions, lens, valid = (
                rtp_llm_ops.mtp_msa_target_verify_ragged_addressing_prepare(
                    table, prefix, cu, q.shape[0]
                )
            )
            result = q8kv4_paged_sparse_decode(
                q,
                idx_q,
                layout,
                expanded,
                lens,
                indexer_dim=dim,
                block_size=page,
                topk=16,
                init_blocks=1,
                local_blocks=2,
                score_type="max",
                mma_scale_layout=True,
                fuse_bf16_query_rounding=True,
                query_cu_seqlens=cu,
                max_query_width=8,
                valid_token_mask=valid,
            )
            return result, expanded, positions, lens

        for prefix in (223, 227, 253, 255, 256):
            with self.subTest(prefix=prefix):
                main.zero_()
                side.fill_(127)
                # Seed all seven candidate rows identically before every geometry;
                # the reader must not expose future rows to an earlier query.
                for request, length in enumerate((prefix + 7, 316)):
                    logical = torch.arange(length, device=device)
                    slots = (
                        tables[request, logical // page].long() * page + logical % page
                    )
                    k, v, idx = histories[request]
                    quantize_main_index_rows(
                        k[:length],
                        v[:length],
                        idx[:length],
                        slots,
                        layout,
                        mma_scale_layout=True,
                    )
                frozen_main, frozen_side = main.clone(), side.clone()
                baseline, _, baseline_positions, baseline_lens = run(
                    inputs(prefix, (0,), (5,))
                )
                scores, topk, output = (
                    baseline.index_scores.clone(),
                    baseline.topk_indices.clone(),
                    baseline.output.clone(),
                )
                cases = (((1, 0), (3, 7), 3), ((0, 1), (7, 3), 0), ((0,), (5,), 0))
                for order, widths, start in cases:
                    data = inputs(prefix, order, widths)
                    result, expanded, positions, lens = run(data)
                    torch.testing.assert_close(
                        positions[start : start + 5], baseline_positions, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        lens[start : start + 5], baseline_lens, rtol=0, atol=0
                    )
                    actual_slots = build_decode_physical_slots(lens, expanded)
                    expected_slots = (
                        tables[0, baseline_positions.long() // page].long() * page
                        + baseline_positions.long() % page
                    )
                    torch.testing.assert_close(
                        actual_slots[start : start + 5], expected_slots, rtol=0, atol=0
                    )
                    # Reapply the actual row writer through the CUDA-produced slots.
                    ks, vs, indices = [], [], []
                    for request, width in zip(order, widths):
                        begin = prefix if request == 0 else 313
                        k, v, idx = histories[request]
                        ks.append(k[begin : begin + width])
                        vs.append(v[begin : begin + width])
                        indices.append(idx[begin : begin + width])
                    quantize_main_index_rows(
                        torch.cat(ks),
                        torch.cat(vs),
                        torch.cat(indices),
                        actual_slots,
                        layout,
                        mma_scale_layout=True,
                    )
                    torch.testing.assert_close(main, frozen_main, rtol=0, atol=0)
                    torch.testing.assert_close(side, frozen_side, rtol=0, atol=0)
                    result, _, _, _ = run(data)
                    torch.testing.assert_close(
                        result.index_scores[:, start : start + 5],
                        scores,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        result.topk_indices[:, start : start + 5], topk, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        result.output[start : start + 5], output, rtol=0, atol=0
                    )

                data = inputs(prefix, (1, 0), (3, 7))
                run(data)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, _, _, _ = run(data)
                # Same captured shape, mutable cu_seqlens/table/query ownership.
                for order, widths, start in cases[:2] + cases[:1]:
                    updated = inputs(prefix, order, widths)
                    for destination, source in zip(data, updated):
                        destination.copy_(source)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        captured.index_scores[:, start : start + 5],
                        scores,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        captured.topk_indices[:, start : start + 5],
                        topk,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        captured.output[start : start + 5], output, rtol=0, atol=0
                    )

    def test_native_attention_constant_value_fake_live_graph(self):
        """Shared writer pages, short last page and fake/live graph transitions.

        Constant V=6 is exactly represented by E2M1 with scale=1. Q/K=0
        gives uniform probabilities exactly representable in FP8, so every
        nonempty attention row must return 6.
        This oracle does not call a second copy of the attention kernel.
        """
        device = torch.device("cuda")
        page, heads, dim, pages, length = 128, 4, 128, 7, 269
        table_row = torch.tensor([4, 1, 5], dtype=torch.int32, device=device)
        logical_rows = torch.arange(length, device=device)
        slots = table_row[logical_rows // page].long() * page + logical_rows % page
        k = torch.zeros(length, heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.full_like(k, 6)
        idx_k = torch.zeros(length, 1, dim, dtype=torch.bfloat16, device=device)
        for mma in (False, True):
            main = torch.zeros(pages, 65536, dtype=torch.uint8, device=device)
            # Writer touches only committed rows. Unread page tails contain
            # E4M3 NaNs, which must not enter PV via zero * NaN.
            side = torch.full((pages, 17408), 127, dtype=torch.uint8, device=device)
            layout = cache_layout(main, side, heads, page, dim)
            quantize_main_index_rows(k, v, idx_k, slots, layout, mma_scale_layout=mma)
            views = layout.logical_views(dim)
            for rows in (1, 8, 16, 120):
                with self.subTest(mma=mma, query_rows=rows):
                    q = torch.zeros(rows, 64, dim, device=device).to(
                        torch.float8_e4m3fn
                    )
                    table = table_row.expand(rows, -1).contiguous()
                    topk = torch.full(
                        (4, rows, 16), -1, dtype=torch.int32, device=device
                    )
                    lens = torch.zeros(rows, dtype=torch.int32, device=device)
                    out = torch.empty(
                        rows, 64, dim, dtype=torch.bfloat16, device=device
                    )
                    partial = torch.empty(
                        rows, 4, 16, 16, dim, dtype=torch.bfloat16, device=device
                    )
                    lse = torch.empty(rows, 4, 16, 16, device=device)
                    counts = torch.empty(rows, 4, dtype=torch.int32, device=device)

                    def run():
                        q8kv4_sparse_decode_attention(
                            q,
                            views.main_k_fp4,
                            views.main_v_fp4,
                            views.main_k_scale,
                            views.main_v_scale,
                            table,
                            topk,
                            lens,
                            sm_scale=dim**-0.5,
                            out=out,
                            partial_out=partial,
                            partial_lse=lse,
                            counts=counts,
                            mma_scale_layout=mma,
                        )

                    # Compile the nonempty path before capture as well as fake rows.
                    lens.fill_(length)
                    selected = torch.arange(3, dtype=torch.int32, device=device)
                    topk[:, :, :3].copy_(selected)
                    run()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    for live in (False, True, False):
                        lens.fill_(length if live else 0)
                        topk.fill_(-1)
                        if live:
                            topk[:, :, :3].copy_(selected)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            out, torch.full_like(out, 6 if live else 0), rtol=0, atol=0
                        )
                        self.assertTrue((counts == 16).all().item())
                        if not live:
                            self.assertTrue((partial == 0).all().item())
                            self.assertTrue(torch.isneginf(lse).all().item())

    def test_grouped_verify_matches_expanded_decode(self):
        """Real writer -> grouped score -> production TopK -> native attention."""
        torch.manual_seed(20260926)
        device = torch.device("cuda")
        batch, heads, dim, page, pages = 2, 4, 128, 128, 24
        blocks = batch * pages
        main_bytes = 2 * heads * page * dim // 2
        side_bytes = (
            2 * heads * page * (dim // 16) + page * dim // 2 + page * (dim // 16)
        )
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, heads, page, dim)
        slots = torch.arange(blocks * page, dtype=torch.int64, device=device)
        k = torch.randn(blocks * page, heads, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(
            k,
            torch.randn_like(k),
            torch.randn(blocks * page, 1, dim, dtype=torch.bfloat16, device=device),
            slots,
            layout,
        )
        options = dict(
            indexer_dim=dim,
            block_size=page,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )
        # Widths5/6/7/8 cover verified draft budgets4/5/6/7 plus the
        # anchor; width17 exercises the unchanged generic scorer fallback.
        for width in (3, 5, 6, 7, 8, 17):
            with self.subTest(width=width):
                q = torch.randn(
                    batch * width, 64, dim, dtype=torch.bfloat16, device=device
                )
                idx_q = torch.randn(
                    batch * width, heads, dim, dtype=torch.bfloat16, device=device
                )
                table = torch.randperm(blocks, device=device).to(torch.int32)
                table = table.view(batch, pages)
                table = table.repeat_interleave(width, dim=0)
                lens = (
                    (
                        torch.tensor([2300, 0], device=device)[:, None]
                        + torch.arange(width, device=device)[None, :]
                    )
                    .reshape(-1)
                    .to(torch.int32)
                )
                # Include inactive graph-bucket rows and per-query causal lengths.
                lens[width:] = 0

                def run(grouped, valid_token_mask=None):
                    return q8kv4_paged_sparse_decode(
                        q,
                        idx_q,
                        layout,
                        table,
                        lens,
                        query_width=width if grouped else 1,
                        valid_token_mask=valid_token_mask,
                        **options,
                    )

                def compare():
                    ref = run(False)
                    scores = ref.index_scores.clone()
                    indices = ref.topk_indices.clone()
                    output = ref.output.clone()
                    actual = run(True)
                    torch.testing.assert_close(
                        actual.index_scores, scores, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual.topk_indices.sort(-1).values,
                        indices.sort(-1).values,
                        rtol=0,
                        atol=0,
                    )
                    # The production TopK output is unordered; partial combine
                    # may therefore differ by BF16 rounding despite equal sets.
                    torch.testing.assert_close(
                        actual.output, output, rtol=0.02, atol=0.002
                    )
                    return actual.output.clone()

                compare()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run(True)
                q.normal_()
                idx_q.normal_()
                lens[:width].add_(page)
                # Revive previously padded rows without recapturing the graph.
                lens[width:] = torch.arange(
                    1, width + 1, device=device, dtype=torch.int32
                )
                table.copy_(table.flip(1))
                expected = compare()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    captured.output, expected, rtol=0.02, atol=0.002
                )

                rows = batch * width
                valid = torch.ones(rows, dtype=torch.bool, device=device)
                run(True, valid)
                masked_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(masked_graph):
                    masked = run(True, valid)
                output_ptr = masked.output.data_ptr()
                for live in (0, max(1, rows // 2), rows):
                    valid.copy_(torch.arange(rows, device=device) < live)
                    reference = run(True).output.clone()
                    masked_graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(masked.output.data_ptr(), output_ptr)
                    torch.testing.assert_close(
                        masked.output,
                        torch.where(
                            valid[:, None, None], reference, torch.zeros_like(reference)
                        ),
                        rtol=0.02,
                        atol=0.002,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            masked.output[~valid].view(torch.int16)
                        ).item(),
                        0,
                    )

    def test_ragged_grouped_verify_matches_expanded_decode_and_graph_replay(self):
        """Compact request offsets preserve the ordinary per-row Q8KV4 result."""
        torch.manual_seed(20260930)
        device = torch.device("cuda")
        widths = [1, 3, 5, 7]
        batch, tokens, heads, dim, page, pages = (
            len(widths),
            sum(widths),
            4,
            128,
            128,
            24,
        )
        blocks = batch * pages
        main_bytes = 2 * heads * page * dim // 2
        side_bytes = (
            2 * heads * page * (dim // 16) + page * dim // 2 + page * (dim // 16)
        )
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, heads, page, dim)
        slots = torch.arange(blocks * page, dtype=torch.int64, device=device)
        k = torch.randn(blocks * page, heads, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(
            k,
            torch.randn_like(k),
            torch.randn(blocks * page, 1, dim, dtype=torch.bfloat16, device=device),
            slots,
            layout,
        )
        q = torch.randn(tokens, 64, dim, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device)
        request_table = (
            torch.randperm(blocks, device=device).to(torch.int32).view(batch, pages)
        )
        cu_seqlens = torch.tensor([0, 1, 4, 9, 16], dtype=torch.int32, device=device)
        table = torch.repeat_interleave(
            request_table, torch.tensor(widths, device=device), dim=0
        ).contiguous()
        lens = torch.cat(
            [
                torch.arange(2301 - width, 2301, dtype=torch.int32, device=device)
                for width in widths
            ]
        ).contiguous()
        options = dict(
            indexer_dim=dim,
            block_size=page,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )

        def run(ragged):
            return q8kv4_paged_sparse_decode(
                q,
                idx_q,
                layout,
                table,
                lens,
                query_cu_seqlens=cu_seqlens if ragged else None,
                max_query_width=8 if ragged else 1,
                **options,
            )

        def compare():
            reference = run(False)
            scores = reference.index_scores.clone()
            indices = reference.topk_indices.clone()
            output = reference.output.clone()
            actual = run(True)
            torch.testing.assert_close(actual.index_scores, scores, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.topk_indices.sort(-1).values,
                indices.sort(-1).values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(actual.output, output, rtol=0.02, atol=0.002)
            return actual.output.clone()

        compare()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(True)

        # Same B/M graph bucket, different per-request compact widths and pages.
        replay_widths = [7, 5, 3, 1]
        cu_seqlens.copy_(torch.tensor([0, 7, 12, 15, 16], device=device))
        table.copy_(
            torch.repeat_interleave(
                request_table.flip(1),
                torch.tensor(replay_widths, device=device),
                dim=0,
            )
        )
        lens.copy_(
            torch.cat(
                [
                    torch.arange(2433 - width, 2433, dtype=torch.int32, device=device)
                    for width in replay_widths
                ]
            )
        )
        q.normal_()
        idx_q.normal_()
        expected = compare()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.output, expected, rtol=0.02, atol=0.002)

    def test_short_fake_stream_block_table_smaller_than_topk(self):
        """DP idle fake streams reserve only 1 + propose_step cache blocks."""
        torch.manual_seed(20260924)
        device = torch.device("cuda")
        batch, q_heads, kv_heads, dim = 1, 64, 4, 128
        page, blocks, topk = 128, 4, 16
        groups = dim // 16
        main_bytes = 2 * kv_heads * page * dim // 2
        side_bytes = 2 * kv_heads * page * groups + page * dim // 2 + page * groups
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, kv_heads, page, dim)
        slots = torch.arange(blocks * page, dtype=torch.int64, device=device)
        k = torch.randn(
            blocks * page, kv_heads, dim, dtype=torch.bfloat16, device=device
        )
        v = torch.randn_like(k)
        idx_k = torch.randn(blocks * page, 1, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(k, v, idx_k, slots, layout)

        result = q8kv4_paged_sparse_decode(
            torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(batch, kv_heads, dim, dtype=torch.bfloat16, device=device),
            layout,
            torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks),
            torch.tensor([2], dtype=torch.int32, device=device),
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )
        self.assertTrue(torch.isfinite(result.output).all().item())
        self.assertTrue(
            torch.equal(
                result.topk_indices[:, :, 0],
                torch.zeros_like(result.topk_indices[:, :, 0]),
            )
        )
        self.assertTrue((result.topk_indices[:, :, 1:] == -1).all().item())

        # Single-request verify uses expand/reshape and preserves stride zero.
        # Exercise that layout through the public wrapper, not just combine.
        width = 5
        q = torch.randn(width, q_heads, dim, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(width, kv_heads, dim, dtype=torch.bfloat16, device=device)
        table = torch.arange(blocks, dtype=torch.int32, device=device).expand(width, -1)
        lens = torch.arange(2, 2 + width, dtype=torch.int32, device=device)
        for stride in (0, 2):
            storage = torch.ones(
                1 if stride == 0 else width * 2, dtype=torch.bool, device=device
            )
            valid = (
                storage[:, None].expand(1, width).reshape(-1)
                if stride == 0
                else storage[::2]
            )
            self.assertEqual(valid.stride(0), stride)

            def run_mask(mask):
                return q8kv4_paged_sparse_decode(
                    q,
                    idx_q,
                    layout,
                    table,
                    lens,
                    indexer_dim=dim,
                    block_size=page,
                    topk=topk,
                    init_blocks=1,
                    local_blocks=2,
                    score_type="max",
                    query_width=width,
                    valid_token_mask=mask,
                )

            reference = run_mask(None).output.clone()
            run_mask(valid)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                masked = run_mask(valid)
            for enabled in (False, True, False):
                storage.fill_(enabled)
                if stride == 2 and enabled:
                    storage[0] = False
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    masked.output,
                    torch.where(
                        valid[:, None, None], reference, torch.zeros_like(reference)
                    ),
                    rtol=0.02,
                    atol=0.002,
                )
                self.assertEqual(
                    torch.count_nonzero(masked.output[~valid].view(torch.int16)).item(),
                    0,
                )

    def test_rtp_two_region_layout_runs_without_working_pages(self):
        torch.manual_seed(20260923)
        device = torch.device("cuda")
        # MiniMax-M3.1 decode geometry: 64 Q heads / 4 KV and index heads.
        batch, q_heads, kv_heads, dim = 2, 64, 4, 128
        page, blocks_per_request, topk = 128, 20, 16
        blocks = batch * blocks_per_request
        groups = dim // 16
        main_bytes = 2 * kv_heads * page * dim // 2
        side_bytes = 2 * kv_heads * page * groups + page * dim // 2 + page * groups
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, kv_heads, page, dim)

        rows = blocks * page
        slots = torch.arange(rows, dtype=torch.int64, device=device)
        k = torch.randn(rows, kv_heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        idx_k = torch.randn(rows, 1, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(k, v, idx_k, slots, layout)

        q = torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(batch, kv_heads, dim, dtype=torch.bfloat16, device=device)
        block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(
            batch, blocks_per_request
        )
        seq_lens = torch.tensor(
            [blocks_per_request * page - 17, 17 * page + 31],
            dtype=torch.int32,
            device=device,
        )
        result = q8kv4_paged_sparse_decode(
            q,
            idx_q,
            layout,
            block_table,
            seq_lens,
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )
        self.assertEqual(result.output.dtype, torch.bfloat16)
        self.assertEqual(tuple(result.output.shape), (batch, q_heads, dim))
        self.assertEqual(tuple(result.topk_indices.shape), (kv_heads, batch, topk))
        self.assertTrue(torch.isfinite(result.output).all().item())

        # Reference the same persistent bytes through the established BF16
        # reader, then apply the demo's scale-1 E4M3 compute contract.  This
        # checks RTP byte offsets independently of the native readers.
        k_pages = torch.empty(
            blocks, kv_heads, page, dim, dtype=torch.bfloat16, device=device
        )
        v_pages = torch.empty_like(k_pages)
        idx_pages = torch.empty(
            blocks * page, 1, dim, dtype=torch.bfloat16, device=device
        )
        gather_main_rows(layout, slots, slots, k_pages, v_pages, out_hnd=True)
        gather_index_rows(layout, dim, slots, slots, idx_pages)
        idx_pages = idx_pages.view(blocks, page, dim).to(torch.float8_e4m3fn)
        idx_q8 = idx_q.to(torch.float8_e4m3fn)
        ref_scores = torch.full_like(result.index_scores, float("-inf"))
        for b in range(batch):
            visible = (int(seq_lens[b]) + page - 1) // page
            local_start = max(0, visible - 2)
            for logical_block in range(visible):
                physical = int(block_table[b, logical_block])
                valid = min(page, int(seq_lens[b]) - logical_block * page)
                token_score = torch.einsum(
                    "hd,td->ht",
                    idx_q8[b].float(),
                    idx_pages[physical, :valid].float(),
                )
                ref_scores[:, b, logical_block] = token_score.max(dim=-1).values * (
                    dim**-0.5
                )
                if logical_block >= local_start:
                    ref_scores[:, b, logical_block] = 1.0e29
                elif logical_block < 1:
                    ref_scores[:, b, logical_block] = 1.0e30
        # Equal 1e30/1e29 sentinel scores intentionally leave tie order to the
        # production bitonic TopK.  Selection is the contract; torch.topk's
        # unrelated tie order is not.
        reference_topk = torch.topk(ref_scores, topk, dim=-1).indices.to(torch.int32)
        torch.testing.assert_close(
            result.topk_indices.sort(dim=-1).values,
            reference_topk.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )

        q8 = q.to(torch.float8_e4m3fn)
        k8 = k_pages.to(torch.float8_e4m3fn)
        v8 = v_pages.to(torch.float8_e4m3fn)
        reference = torch.empty_like(result.output)
        for b in range(batch):
            for h in range(q_heads):
                kv_head = h // (q_heads // kv_heads)
                keys, values = [], []
                for logical_block in result.topk_indices[kv_head, b].tolist():
                    physical = int(block_table[b, logical_block])
                    valid = min(
                        page,
                        max(0, int(seq_lens[b]) - logical_block * page),
                    )
                    if valid:
                        keys.append(k8[physical, kv_head, :valid].float())
                        values.append(v8[physical, kv_head, :valid].float())
                key = torch.cat(keys)
                value = torch.cat(values)
                probability = torch.softmax(
                    q8[b, h].float() @ key.T / (dim**0.5), dim=-1
                )
                reference[b, h] = (probability[:, None] * value).sum(0)
        torch.testing.assert_close(
            result.output.float(), reference.float(), rtol=0.03, atol=0.04
        )

        # All temporary tensors are bucket-owned and reused, so the complete
        # Q cast -> score -> top-k -> sparse attention chain is replayable.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = q8kv4_paged_sparse_decode(
                q,
                idx_q,
                layout,
                block_table,
                seq_lens,
                indexer_dim=dim,
                block_size=page,
                topk=topk,
                init_blocks=1,
                local_blocks=2,
                score_type="max",
            )
        graph.replay()
        torch.cuda.synchronize()
        first = captured.output.clone()
        q.normal_()
        idx_q.normal_()
        seq_lens.copy_(torch.tensor([19 * page + 9, 16 * page + 3], device=device))
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.output.clone()
        expected = q8kv4_paged_sparse_decode(
            q,
            idx_q,
            layout,
            block_table,
            seq_lens,
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        ).output.clone()
        self.assertFalse(torch.equal(first, replayed))
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

"""CPU adapter contracts and GPU metadata/scale replay checks."""

import ast
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


SOURCE = Path(__file__).resolve().parents[1] / "decode" / "cake_attention.py"


class CakeSourceContractTest(unittest.TestCase):
    def test_scale_layout_is_bijective(self):
        tree = ast.parse(SOURCE.read_text())
        refresh = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_refresh_selected_scales")
        assignments = {n.targets[0].id: n.value for n in ast.walk(refresh)
                       if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name)}
        # Evaluate the production expressions, rather than a duplicated mapping.
        expressions = {name: compile(ast.Expression(assignments[name]), SOURCE, "eval")
                       for name in ("swz_token", "swz_group")}
        mma_branch = next(n for n in ast.walk(refresh) if isinstance(n, ast.If)
                          and isinstance(n.test, ast.Name) and n.test.id == "MMA")
        mma = compile(ast.Expression(mma_branch.body[0].value), SOURCE, "eval")
        mma_offsets, swizzle_offsets = set(), set()
        for token in range(128):
            for group in range(8):
                scope = {"token": token, "group": group}
                mma_offsets.add(eval(mma, {}, scope))
                st = eval(expressions["swz_token"], {}, scope)
                sg = eval(expressions["swz_group"], {}, scope)
                swizzle_offsets.add(st * 8 + sg)
        self.assertEqual(mma_offsets, set(range(1024)))
        self.assertEqual(swizzle_offsets, set(range(1024)))

    def test_shared_scales_and_no_host_readback(self):
        text = SOURCE.read_text()
        tree = ast.parse(text)
        bucket = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "CakeAttentionWorkspace")
        create = next(n for n in bucket.body if isinstance(n, ast.FunctionDef)
                      and n.name == "create")
        self.assertIn("scale_workspace", [n.arg for n in create.args.kwonlyargs])
        self.assertEqual(create.args.kw_defaults, [None])
        for forbidden in ("torch.unique", ".item()", ".cpu()", "nan_to_num",
                          "masked_fill_", "convert_rtp_scales"):
            self.assertNotIn(forbidden, text)
        self.assertIn("ws.scales.seen.zero_()", text)
        self.assertIn("tl.atomic_max", text)
        self.assertNotIn("tl.atomic_cas", text)
        self.assertIn("owner == selection.to(tl.int64) + 1", text)
        self.assertIn("page = tl.load(TABLE", text)
        self.assertIn(".to(tl.int64)", text)

    def test_elected_owner_is_immutable_in_refresh(self):
        tree = ast.parse(SOURCE.read_text())
        refresh = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_refresh_selected_scales")
        # Scalar loads are replicated across warps. Mutating election state in
        # this CTA could make its warps disagree and leave a partial scale page.
        refresh_text = ast.get_source_segment(SOURCE.read_text(), refresh)
        self.assertNotIn("tl.atomic_", refresh_text)
        stores = [n for n in ast.walk(refresh) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "store"]
        self.assertTrue(stores)
        self.assertTrue(all("SEEN" not in ast.unparse(n.args[0]) for n in stores))
        candidates = [(3 << 32) | 640, (128 << 32) | 1, (128 << 32) | 639]
        elected = max(candidates)
        self.assertEqual((elected >> 32, elected & 0xFFFFFFFF), (128, 639))


@unittest.skipUnless(torch is not None, "PyTorch required")
class CakeCacheAbiCpuTest(unittest.TestCase):
    def test_packed_cache_planes_are_zero_copy(self):
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import cache_layout
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode.cake_attention import CakeScaleWorkspace
        main = torch.empty((7, 65536 + 64), dtype=torch.uint8)
        side = torch.empty((7, 17408 + 64), dtype=torch.uint8)
        layout = cache_layout(main, side, 4, 128, 128)
        for plane in (0, 1):
            packed, scales = layout.main_plane(plane)
            cake = packed.view(7, 4, 128, 64)
            self.assertEqual(cake.data_ptr(), main.data_ptr() + plane * 32768)
            self.assertEqual(cake.stride(), (65600, 8192, 64, 1))
            self.assertEqual(scales.view(torch.uint8).data_ptr(),
                             side.data_ptr() + plane * 4096)
            self.assertEqual(scales.stride(0), side.stride(0))
        scratch = CakeScaleWorkspace(torch.empty((7, 4, 128, 8), dtype=torch.uint8),
                                     torch.empty((7, 4, 128, 8), dtype=torch.uint8),
                                     torch.empty((7, 4), dtype=torch.int64), 0)
        self.assertEqual(scratch.nbytes, 7 * (8192 + 32))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
class CakeMetadataGpuTest(unittest.TestCase):
    def setUp(self):
        import torch
        from rtp_llm.models_py.triton_kernels.sparse_msa.decode import cake_attention
        self.torch, self.adapter = torch, cake_attention

    def test_duplicate_cta_immutable_owner_graph_layers(self):
        torch, adapter = self.torch, self.adapter
        rows, cols, pages = 40, 711, 16
        topk = torch.arange(16, device="cuda", dtype=torch.int32).expand(4, rows, 16).contiguous()
        table = torch.zeros((rows, cols), device="cuda", dtype=torch.int32)
        table[:, :16] = torch.arange(16, device="cuda", dtype=torch.int32)
        lens = torch.full((rows,), 2048, device="cuda", dtype=torch.int32)
        scratch = adapter.CakeScaleWorkspace.create(table.device, pages)
        token = torch.arange(128, device="cuda")[:, None]
        group = torch.arange(8, device="cuda")[None, :]
        mma = (group // 4) * 512 + (token % 32) * 16 + (token // 32) * 4 + group % 4
        swizzle = ((token // 4) * 4 + group // 2) * 8 + (group % 2) * 4 + token % 4
        sources, tags = [], []
        for layer in range(2):
            tag = ((torch.arange(pages, device="cuda")[:, None, None, None] * 13
                    + torch.arange(4, device="cuda")[None, :, None, None] * 19
                    + token[None, None] * 3 + group[None, None] * 7 + layer * 41) % 120).to(torch.uint8)
            source = torch.empty((pages, 4, 1024), dtype=torch.uint8, device="cuda")
            source[:, :, mma] = tag
            sources.append(source)
            tags.append(tag)
        failures = torch.zeros((), device="cuda", dtype=torch.int64)

        def run():
            for source, tag in zip(sources, tags):
                scratch.k.fill_(0x7f)
                scratch.v.fill_(0x7f)
                scratch.seen.zero_()
                adapter._selected_valid_prefix[(rows, 4)](
                    topk, table, lens, scratch.seen, ROWS=rows, COLS=cols, num_warps=4)
                elected = scratch.seen.clone()
                adapter._refresh_selected_scales[(rows * 16, 4)](
                    topk, table, scratch.seen, source, source, scratch.k, scratch.v,
                    source.stride(0), source.stride(0), ROWS=rows, COLS=cols,
                    MMA=True, num_warps=4)
                prefix = (lens.max().to(torch.int64)
                          - torch.arange(pages, device="cuda") * 128).clamp(0, 128)
                # Longest-prefix ties choose the greatest selection ID. Row39
                # has the max length and ID for every selected page/head.
                owners = (rows - 1) * 16 + torch.arange(pages, device="cuda") + 1
                wanted_election = ((prefix << 32) | owners)[:, None].expand(pages, 4)
                wanted = torch.where(token[None, None] < prefix[:, None, None, None], tag, 0)
                failures.add_(torch.count_nonzero(elected != wanted_election))
                failures.add_(torch.count_nonzero(scratch.seen != elected))
                failures.add_(torch.count_nonzero(scratch.k != wanted))
                failures.add_(torch.count_nonzero(scratch.v.reshape(pages, 4, 1024)[:, :, swizzle] != wanted))

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for iteration in range(100):
            lens.fill_(1921 + (iteration * 17) % 128)
            lens[::2] -= 1
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(failures), 0)

    def test_metadata_selected_scales_and_graph_mutation(self):
        torch, adapter = self.torch, self.adapter
        rows, pages, cols = 3, 9, 8192
        q = torch.randn(rows, 64, 128, dtype=torch.bfloat16, device="cuda")
        # Large logical history, duplicate/reordered physical pages, padded table.
        table_storage = torch.full((rows, cols + 7), -1, dtype=torch.int64, device="cuda")
        table = table_storage[:, :cols]
        table[:, :5] = torch.tensor([4, 2, 4, 0, pages], device="cuda")
        table[:, 5] = 2**40
        table[:, -1] = 7
        lens = torch.tensor([1048576, 3, 0], dtype=torch.int64, device="cuda")
        topk = torch.full((4, rows, 16), -1, dtype=torch.int32, device="cuda")
        topk[:, :, :10] = torch.tensor([8191, 2, 0, 2, 4, -1, 8192, 1, 3, 5], device="cuda")
        mask_storage = torch.tensor([True, False, True, False, False, False], device="cuda")
        mask = mask_storage[::2]
        qo, ko = torch.empty_like(q), torch.empty_like(topk)
        to = torch.empty((rows, cols), dtype=torch.int32, device="cuda")
        lo = torch.empty((rows,), dtype=torch.int32, device="cuda")
        # Padded physical scale stride, including exact NaN byte preservation.
        src = torch.randint(0, 256, (pages, 4 * 1024 + 37), dtype=torch.uint8, device="cuda")
        src[4, 0] = 0x7f
        src[0, 2 * 1024] = 0xff
        scales = adapter.CakeScaleWorkspace.create(q.device, pages)
        scales.k.fill_(0xa5)
        scales.v.fill_(0xa5)

        def run():
            adapter._prepare_metadata[(rows, 4)](
                q, topk, table, lens, mask, qo, ko, to, lo,
                q.stride(0), q.stride(1), table.stride(0), table.stride(1),
                topk.stride(0), topk.stride(1), topk.stride(2), lens.stride(0), mask.stride(0),
                ROWS=rows, COLS=cols, PHYSICAL=pages, HAS_MASK=True,
                COPY_Q=True, BLOCK=cols, num_warps=4)
            scales.seen.zero_()
            adapter._selected_valid_prefix[(rows, 4)](
                ko, to, lo, scales.seen, ROWS=rows, COLS=cols, num_warps=4)
            adapter._refresh_selected_scales[(rows * 16, 4)](
                ko, to, scales.seen, src, src, scales.k, scales.v,
                src.stride(0), src.stride(0), ROWS=rows, COLS=cols, MMA=True, num_warps=4)

        def check():
            length = torch.where(mask, lens.clamp(0, cols * 128), 0)
            expected = topk.to(torch.int64).clone()
            valid = (expected >= 0) & (expected < cols)
            valid &= expected < ((length + 127) // 128)[None, :, None]
            physical = torch.gather(table[None, :, :].expand(4, -1, -1),
                                    2, expected.clamp(0, cols - 1))
            valid &= (physical >= 0) & (physical < pages)
            expected = torch.where(valid, expected, 2147483647).sort(-1).values
            expected[expected == 2147483647] = -1
            self.assertTrue(torch.equal(ko, expected.to(torch.int32)))
            self.assertTrue(torch.equal(lo, length.to(torch.int32)))
            self.assertTrue(torch.equal(qo[~mask].view(torch.int16),
                                        torch.zeros_like(qo[~mask]).view(torch.int16)))
            selected = set(physical[valid].cpu().tolist())
            for page in selected:
                token = torch.arange(128, device="cuda")[:, None]
                group = torch.arange(8, device="cuda")[None, :]
                offset = (group // 4) * 512 + (token % 32) * 16 + (token // 32) * 4 + group % 4
                expected_k = src[page, :4096].reshape(4, 1024)[:, offset].clone()
                for head in range(4):
                    prefix = 0
                    for row in range(rows):
                        blocks = ko[head, row]
                        for logical in blocks[blocks >= 0].tolist():
                            if int(to[row, logical]) == page:
                                prefix = max(prefix, min(128, int(lo[row]) - logical * 128))
                    expected_k[head, prefix:] = 0
                self.assertTrue(torch.equal(scales.k[page], expected_k))
                st, sg = (token // 4) * 4 + group // 2, (group % 2) * 4 + token % 4
                self.assertTrue(torch.equal(scales.v[page][:, st, sg], expected_k))
            self.assertTrue(torch.equal(scales.k[6], torch.full_like(scales.k[6], 0xa5)))

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        mask[1] = False
        src.random_(0, 256)
        lens[0] = 129
        graph.replay()
        check()

    def test_masked_direct_bf16_query_alias(self):
        from unittest.mock import patch
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
            cache_layout, quantize_main_index_rows)
        torch, adapter = self.torch, self.adapter
        main = torch.zeros((6, 65536), dtype=torch.uint8, device="cuda")
        side = torch.zeros((6, 17408), dtype=torch.uint8, device="cuda")
        layout = cache_layout(main, side, 4, 128, 128)
        k = torch.randn(768, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.4
        v = torch.randn_like(k)
        idx = torch.zeros(768, 1, 128, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(768, dtype=torch.int64, device="cuda")
        quantize_main_index_rows(k, v, idx, slots, layout, mma_scale_layout=True)
        q = torch.randn(3, 64, 128, dtype=torch.bfloat16, device="cuda")
        q_reference = torch.empty(3, 64, 129, dtype=torch.bfloat16, device="cuda")[:, :, :128]
        q_reference.copy_(q)
        table = torch.arange(1, 6, dtype=torch.int32, device="cuda").repeat(3, 1)
        topk = torch.full((4, 3, 16), -1, dtype=torch.int32, device="cuda")
        topk[:, :, :5] = torch.tensor([4, 1, 3, 0, 2], device="cuda")
        lens = torch.tensor([513, 257, 513], dtype=torch.int32, device="cuda")
        mask = torch.tensor([True, True, False], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            shared = adapter.CakeScaleWorkspace.create(q.device, 6)
            reference = adapter.CakeAttentionWorkspace.create(q, layout, table, topk, lens, scale_workspace=shared)
            alias = adapter.CakeAttentionWorkspace.create(q, layout, table, topk, lens, scale_workspace=shared)
            alias.q = q  # Prototype original-Q binding; disable only the Q copy.
        def run(ws):
            query = q if ws is alias else q_reference
            return adapter.cake_paged_sparse_decode(query, layout, table, topk, lens,
                                                    workspace=ws, valid_token_mask=mask)
        original_metadata = adapter._prepare_metadata
        class NoQueryCopy:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    kwargs["COPY_Q"] = False
                    return original_metadata[grid](*args, **kwargs)
                return launch
        with torch.cuda.stream(stream):
            run(reference)
            with patch.object(adapter, "_prepare_metadata", NoQueryCopy()):
                run(alias)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run(alias)
        for live, lengths in ((2, [513, 257, 513]), (1, [129, 513, 513]),
                              (3, [513, 3, 257]), (0, [513, 513, 513])):
            with torch.cuda.stream(stream):
                mask.copy_(torch.arange(3, device="cuda") < live)
                lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
                q.normal_()
                q[~mask] = float("nan")
                q_reference.copy_(q)
                table.copy_(torch.roll(table, 1, dims=1))
                expected = run(reference).clone()
                graph.replay()
                self.assertTrue(torch.equal(expected, alias.output))
                self.assertEqual(torch.count_nonzero(alias.output[~mask].view(torch.int16)).item(), 0)
                self.assertTrue(torch.isnan(q[~mask]).all())
        # A NaN scale on a VALID token must remain observable, not be coerced.
        with torch.cuda.stream(stream):
            mask.fill_(True)
            lens.fill_(513)
            q.normal_()
            _, scales = layout.main_plane(0)
            page = int(table[0, 0].item())
            scales.view(torch.uint8)[page, 0] = 0x7f
            graph.replay()
            self.assertEqual(int(shared.k[page, 0, 0, 0]), 0x7f)
            self.assertTrue(torch.isnan(alias.output[0, :16]).any())
        torch.cuda.current_stream().wait_stream(stream)

    def test_eager_fresh_queries_do_not_accumulate_runners(self):
        import gc
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import cache_layout

        torch, adapter = self.torch, self.adapter
        main = torch.zeros((2, 65536), dtype=torch.uint8, device="cuda")
        side = torch.zeros((2, 17408), dtype=torch.uint8, device="cuda")
        layout = cache_layout(main, side, 4, 128, 128)
        table = torch.ones((1, 1), dtype=torch.int32, device="cuda")
        topk = torch.full((4, 1, 16), -1, dtype=torch.int32, device="cuda")
        topk[:, :, 0] = 0
        lens = torch.tensor([1], dtype=torch.int32, device="cuda")
        q = torch.ones((1, 64, 128), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            scales = adapter.CakeScaleWorkspace.create(q.device, 2)
            ws = adapter.CakeAttentionWorkspace.create(
                q, layout, table, topk, lens, scale_workspace=scales)

        def run_fresh_query():
            with torch.cuda.stream(stream):
                fresh_q = q.clone()
                adapter.cake_paged_sparse_decode(
                    fresh_q, layout, table, topk, lens, workspace=ws)

        for _ in range(10):
            run_fresh_query()
        torch.cuda.synchronize()
        gc.collect()
        allocated = torch.cuda.memory_allocated()
        for _ in range(1000):
            run_fresh_query()
        torch.cuda.synchronize()
        gc.collect()
        self.assertEqual(len(ws.runners), 0)
        self.assertLessEqual(torch.cuda.memory_allocated(), allocated + 1024 * 1024)

        # Captured Q bindings remain owned for the lifetime of replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            adapter.cake_paged_sparse_decode(q, layout, table, topk, lens, workspace=ws)
        self.assertEqual(len(ws.runners), 1)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(len(ws.runners), 1)

    def test_physical_scale_offset_exceeds_signed_int32(self):
        torch, adapter = self.torch, self.adapter
        # Allocate only one large-stride backing buffer; touch its two 4KiB
        # scale rows. This tests a REAL >2GiB physical address, not a reused
        # logical-history page or a compilation-only integer-expression check.
        stride = 2**31 + 16
        storage = torch.empty(stride + 4096, dtype=torch.uint8, device="cuda")
        source = storage.as_strided((2, 4096), (stride, 1))
        source[0].fill_(0x28)
        source[1].fill_(0x38)
        ordered = torch.full((4, 1, 16), -1, dtype=torch.int32, device="cuda")
        ordered[:, 0, 0] = 0
        table = torch.ones((1, 1), dtype=torch.int32, device="cuda")
        lens = torch.tensor([128], dtype=torch.int32, device="cuda")
        scratch = adapter.CakeScaleWorkspace.create(source.device, 2)
        scratch.seen.zero_()
        adapter._selected_valid_prefix[(1, 4)](
            ordered, table, lens, scratch.seen, ROWS=1, COLS=1, num_warps=4)
        adapter._refresh_selected_scales[(16, 4)](
            ordered, table, scratch.seen, source, source, scratch.k, scratch.v,
            stride, stride, ROWS=1, COLS=1, MMA=True, num_warps=4)
        self.assertTrue(torch.equal(scratch.k[1], torch.full_like(scratch.k[1], 0x38)))
        self.assertTrue(torch.equal(scratch.v[1], torch.full_like(scratch.v[1], 0x38)))


if __name__ == "__main__":
    unittest.main()

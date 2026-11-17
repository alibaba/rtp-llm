import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import resolve_read_blocks


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PleReadBlocksTest(unittest.TestCase):
    def test_dummy_page_and_pool_boundaries_with_strided_table(self):
        table = torch.tensor(
            [[0, 3, 9, 2], [3, 0, 8, 10], [1, 2, 3, 4], [4, 3, 2, 1]],
            device="cuda",
            dtype=torch.int32,
        ).t()
        for dtype in (torch.int32, torch.int64):
            typed_table = table.to(dtype)
            for values in ((-1, 0, 1, 4), (1, 2, 3, 0), (-(2**30), 2**30, 0, 2)):
                pages = torch.tensor(values, device="cuda", dtype=torch.int64)
                actual = resolve_read_blocks(typed_table, pages, 10)
                expected = []
                for row, page in enumerate(values):
                    block = int(typed_table[row, page]) if 0 <= page < 4 else 0
                    expected.append(block if 0 < block < 10 else 0)
                torch.testing.assert_close(
                    actual,
                    torch.tensor(expected, device="cuda", dtype=torch.int64),
                    rtol=0,
                    atol=0,
                )

    def test_graph_refreshes_table_and_request_order(self):
        table = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda", dtype=torch.int32
        )
        pages = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
        for _ in range(3):
            resolve_read_blocks(table, pages, 10)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = resolve_read_blocks(table, pages, 10)
        cases = [
            ([2, 0, 1], [[9, 8, 7], [6, 5, 4], [3, 2, 1]], [7, 6, 2]),
            ([-(2**40), 2**40, 1], [[9, 8, 7], [6, 5, 4], [3, 0, 1]], [0, 0, 0]),
            ([0, 1, 2], [[-1, 8, 7], [6, 10, 4], [3, 2, 9]], [0, 0, 9]),
        ]
        for logical, physical, expected in cases:
            pages.copy_(torch.tensor(logical, device="cuda"))
            table.copy_(torch.tensor(physical, device="cuda", dtype=torch.int32))
            graph.replay()
            torch.testing.assert_close(
                actual, torch.tensor(expected, device="cuda"), rtol=0, atol=0
            )

    def test_graph_gather_keeps_tag_local_pools_and_history(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import (
            gather_target_read_state,
        )

        torch.manual_seed(9021)
        state_pool = torch.randn(12, 9, 10240, device="cuda", dtype=torch.bfloat16)
        ctx_pool = torch.arange(24, device="cuda", dtype=torch.int64).reshape(12, 2)
        st = torch.tensor([[1, 3, 5], [2, 4, 6]], device="cuda", dtype=torch.int32)
        ct = torch.tensor([[8, 7, 6], [9, 10, 11]], device="cuda", dtype=torch.int32)
        prefix = torch.tensor([128, 129], device="cuda", dtype=torch.int64)
        ids = torch.arange(8, device="cuda", dtype=torch.int64)
        for _ in range(3):
            gather_target_read_state(prefix, ids, st, ct, state_pool, ctx_pool, 128)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather_target_read_state(
                prefix, ids, st, ct, state_pool, ctx_pool, 128
            )
        for step, lengths in enumerate(([128, 129], [384, 0], [-100, 2**40], [1, 128])):
            prefix.copy_(torch.tensor(lengths, device="cuda"))
            ids.add_(7)
            if step == 3:
                st[0, 0] = 0
                ct[1, 0] = 12
            expected_state, expected_ctx = [], []
            for row, length in enumerate(lengths):
                page = (length - 1) // 128
                sb = int(st[row, page]) if 0 <= page < 3 else 0
                cb = int(ct[row, page]) if 0 <= page < 3 else 0
                expected_state.append(state_pool[sb if 0 < sb < 12 else 0])
                expected_ctx.append(ctx_pool[cb if 0 < cb < 12 else 0])
            graph.replay()
            expected_state = torch.stack(expected_state)
            expected_ctx = torch.stack(expected_ctx)
            for a, b in zip(
                actual,
                (
                    expected_state,
                    expected_ctx,
                    torch.cat((expected_ctx, ids.reshape(2, 4)), 1),
                ),
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_decode_graph_gathers_and_commits_tag_local_strided_pools(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import (
            prepare_decode_graph,
            try_store_decode_state,
        )

        state = torch.arange(5 * 24, device="cuda", dtype=torch.bfloat16)
        state = state.view(5, 24)[:, :18].view(5, 3, 6)
        context = torch.arange(20, device="cuda").view(5, 4)[:, :2]
        state_table = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
        ctx_table = torch.tensor([[4, 3], [2, 1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([129, 1], device="cuda", dtype=torch.int32)
        ids = torch.tensor([21, 22], device="cuda")
        new_state = torch.full((2, 3, 6), 42, device="cuda", dtype=torch.bfloat16)

        def step():
            buf, history, state_dest, ctx_dest = prepare_decode_graph(
                lengths, ids, state_table, ctx_table, state, context, 128, 99
            )
            self.assertTrue(
                try_store_decode_state(
                    state, context, state_dest, ctx_dest, new_state, history
                )
            )
            return buf, history, state_dest, ctx_dest

        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = step()
        for seq, st, ct, current in (
            ([129, 1], [[1, 2], [3, 4]], [[4, 3], [2, 1]], [21, 22]),
            ([2, 385], [[-1, 3], [4, 3]], [[1, 5], [2, 3]], [31, 32]),
            ([1, 129], [[2, 4], [3, 4]], [[4, 1], [2, 1]], [41, 42]),
        ):
            lengths.copy_(torch.tensor(seq, device="cuda"))
            ids.copy_(torch.tensor(current, device="cuda"))
            state_table.copy_(torch.tensor(st, device="cuda"))
            ctx_table.copy_(torch.tensor(ct, device="cuda"))
            before_state, before_context = state.clone(), context.clone()
            expected_buffer, expected_history, expected_sd, expected_cd = [], [], [], []
            for i, length in enumerate(seq):
                read, write = (length - 2) // 128, (length - 1) // 128

                def physical(table, page):
                    block = table[i][page] if length > 1 and 0 <= page < 2 else 0
                    return block if 0 < block < 5 else 0

                sb, cb = physical(st, read), physical(ct, read)
                sw, cw = physical(st, write), physical(ct, write)
                expected_sd.append(sw)
                expected_cd.append(cw)
                expected_buffer.append(
                    before_state[sb] if length > 1 else torch.zeros_like(state[0])
                )
                hist = before_context[cb].tolist() if length > 1 else [99, 99]
                expected_history.append([*hist, current[i]])
            graph.replay()
            torch.testing.assert_close(
                actual[0], torch.stack(expected_buffer), atol=0, rtol=0
            )
            for tensor, expected in zip(
                actual[1:], (expected_history, expected_sd, expected_cd)
            ):
                self.assertEqual(tensor.tolist(), expected)
            for i, (sw, cw) in enumerate(zip(expected_sd, expected_cd)):
                if sw:
                    before_state[sw] = new_state[i]
                if cw:
                    before_context[cw] = torch.tensor(
                        expected_history[i][-2:], device="cuda"
                    )
            torch.testing.assert_close(state, before_state, atol=0, rtol=0)
            torch.testing.assert_close(context, before_context, atol=0, rtol=0)

    def test_prefill_commits_paired_pages_and_partial_tail(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import (
            try_store_prefill_state,
        )

        normed = torch.arange(108, device="cuda", dtype=torch.bfloat16).view(1, 18, 6)
        history = torch.arange(20, device="cuda").view(1, 20)
        state = torch.zeros((5, 24), device="cuda", dtype=torch.bfloat16)
        state = state[:, :18].view(5, 3, 6)
        ctx = torch.zeros((5, 4), device="cuda", dtype=torch.long)[:, :2]
        state_ids = torch.tensor([3, 2, 1], device="cuda", dtype=torch.int32)
        ctx_ids = torch.tensor([2, 0, 4], device="cuda", dtype=torch.long)

        self.assertTrue(
            try_store_prefill_state(state, ctx, normed, history, state_ids, ctx_ids, 8)
        )
        expected_state = torch.zeros_like(state)
        expected_ctx = torch.zeros_like(ctx)
        expected_state[3] = normed[0, 5:8]
        expected_state[1] = normed[0, 15:18]
        expected_ctx[2] = history[0, 8:10]
        expected_ctx[4] = history[0, 18:20]
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(ctx, expected_ctx, atol=0, rtol=0)

    def test_cpu_and_empty_inputs_keep_fallback(self):
        self.assertIsNone(
            resolve_read_blocks(
                torch.ones(1, 1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int64),
                2,
            )
        )
        table = torch.ones(1, 0, device="cuda", dtype=torch.int32)
        pages = torch.zeros(1, device="cuda", dtype=torch.int64)
        self.assertIsNone(resolve_read_blocks(table, pages, 2))
        self.assertIsNone(
            resolve_read_blocks(
                torch.ones(1, 1, device="cuda", dtype=torch.int32), pages, 0
            )
        )


if __name__ == "__main__":
    unittest.main()

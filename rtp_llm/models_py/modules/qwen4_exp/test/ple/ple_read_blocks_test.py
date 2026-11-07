import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
    resolve_read_blocks,
)


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
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
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

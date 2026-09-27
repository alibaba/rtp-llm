import unittest

import torch

from rtp_llm.models_py.triton_kernels.dspark_swa import commit_paged_gqa_kv


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkKVCommitTest(unittest.TestCase):
    def test_strided_rows_and_graph_replay(self):
        torch.manual_seed(926)
        tokens, heads, dim, page = 17, 4, 128, 128
        # Different row strides and nonzero view offsets for K and V.
        k = torch.randn(tokens, 3, heads, dim, device="cuda", dtype=torch.bfloat16)[
            :, 1
        ]
        v = torch.randn(tokens, 2, heads, dim, device="cuda", dtype=torch.bfloat16)[
            :, 1
        ]
        storage = torch.full(
            (3, 2 * heads * page * dim + 256), -7.0, device="cuda", dtype=torch.bfloat16
        )
        cache = storage[:, :-256].view(3, 2, heads, page, dim)
        slots = torch.arange(tokens, device="cuda", dtype=torch.int64) + 120
        slots[0], slots[1] = -1, 3 * page
        valid = torch.arange(tokens, device="cuda") % 3 != 0

        def reference(initial):
            expected = initial.clone()
            view = expected[:, :-256].view(3, 2, heads, page, dim)
            for row, slot in enumerate(slots.tolist()):
                if valid[row].item() and 0 <= slot < 3 * page:
                    block, pos = divmod(slot, page)
                    view[block, 0, :, pos] = k[row]
                    view[block, 1, :, pos] = v[row]
            return expected

        expected = reference(storage)
        commit_paged_gqa_kv(k, v, cache, slots, valid)
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit_paged_gqa_kv(k, v, cache, slots, valid)
        k.normal_()
        v.normal_()
        slots.copy_(slots.flip(0))
        valid.logical_not_()
        expected = reference(storage)
        graph.replay()
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "contiguous within"):
            commit_paged_gqa_kv(
                k.transpose(1, 2),
                v.transpose(1, 2),
                cache.view(3, 2, dim, page, heads),
                slots,
                valid,
            )


if __name__ == "__main__":
    unittest.main()

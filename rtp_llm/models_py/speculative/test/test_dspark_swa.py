import unittest

import torch

from rtp_llm.models_py.triton_kernels.dspark_swa import (
    _swa_launch_config,
    commit_paged_gqa_kv,
    paged_gqa_swa,
)


class DSparkSWALaunchConfigTest(unittest.TestCase):
    def test_measured_shape_and_fallbacks(self):
        measured = [16, 7, 64, 4, 128, 4095, True, 10]
        self.assertEqual(_swa_launch_config(*measured), (64, 128, 8))
        alternatives = {
            0: (1, 3, 8, 15, 17, 32),
            1: (1, 4, 5, 6, 8, 16),
            2: (32, 128),
            3: (2, 8),
            4: (64, 256),
            5: (0, 127, 4096),
            6: (False,),
            7: (8, 9, 12),
        }
        for index, values in alternatives.items():
            for value in values:
                args = measured.copy()
                args[index] = value
                with self.subTest(args=args):
                    rows = args[1] * args[2] // args[3]
                    tile = min(32, max(16, 1 << (rows - 1).bit_length()))
                    self.assertEqual(_swa_launch_config(*args), (tile, 64, 4))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkSWATest(unittest.TestCase):
    def test_row_tiles_graph_padding_and_slot_overwrite(self):
        torch.manual_seed(927)
        batch, hq, hk, dim, page = 3, 64, 4, 128, 128
        payload = 2 * hk * page * dim
        for width in (1, 4, 7, 16):
            for causal in (True, False):
                with self.subTest(width=width, causal=causal):
                    storage = torch.full(
                        (9, payload + 256), 37, device="cuda", dtype=torch.bfloat16
                    )
                    cache = storage[:, :payload].view(9, 2, hk, page, dim)
                    cache.normal_()
                    table = torch.randperm(9, device="cuda").reshape(batch, 3).int()
                    q = torch.randn(
                        batch, width, hq, dim, device="cuda", dtype=torch.bfloat16
                    )
                    k = torch.randn(
                        batch, width, hk, dim, device="cuda", dtype=torch.bfloat16
                    )
                    v = torch.randn_like(k)
                    lengths = torch.tensor(
                        [0, 129, 257], device="cuda", dtype=torch.int32
                    )
                    live = torch.tensor(
                        [width, max(1, width - 1), 0], device="cuda", dtype=torch.int32
                    )
                    out = torch.empty_like(q)

                    def forward():
                        return paged_gqa_swa(
                            q,
                            k,
                            v,
                            cache,
                            table,
                            lengths,
                            live,
                            window_left=127,
                            causal=causal,
                            out=out,
                        )

                    def check():
                        expected = torch.zeros_like(q)
                        for row, count in enumerate(live.tolist()):
                            if count == 0:
                                continue
                            length = int(lengths[row])
                            pages = cache[table[row].long()]
                            past_k = (
                                pages[:, 0]
                                .permute(0, 2, 1, 3)
                                .reshape(-1, hk, dim)[:length]
                            )
                            past_v = (
                                pages[:, 1]
                                .permute(0, 2, 1, 3)
                                .reshape(-1, hk, dim)[:length]
                            )
                            keys = torch.cat(
                                (past_k, k[row, :count])
                            ).repeat_interleave(hq // hk, dim=1)
                            values = torch.cat(
                                (past_v, v[row, :count])
                            ).repeat_interleave(hq // hk, dim=1)
                            positions = torch.arange(length + count, device="cuda")
                            query_positions = length + torch.arange(
                                count, device="cuda"
                            )
                            mask = positions[None, :] >= query_positions[:, None] - 127
                            if causal:
                                mask &= positions[None, :] <= query_positions[:, None]
                            scores = (
                                torch.einsum(
                                    "qhd,khd->hqk", q[row, :count].float(), keys.float()
                                )
                                * dim**-0.5
                            )
                            probabilities = scores.masked_fill(
                                ~mask, -float("inf")
                            ).softmax(-1)
                            expected[row, :count] = torch.einsum(
                                "hqk,khd->qhd", probabilities, values.float()
                            ).to(q.dtype)
                        torch.testing.assert_close(
                            out, expected, atol=0.012, rtol=0.025
                        )
                        for row, count in enumerate(live.tolist()):
                            self.assertEqual(int(out[row, count:].count_nonzero()), 0)
                        self.assertTrue(bool((storage[:, payload:] == 37).all()))

                    for _ in range(3):
                        forward()
                    check()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        forward()
                    pointers = [
                        t.data_ptr()
                        for t in (q, k, v, cache, table, lengths, live, out)
                    ]
                    q.normal_()
                    k.normal_()
                    v.normal_()
                    table.copy_(table.roll(1, dims=1))
                    lengths.copy_(
                        torch.tensor([257, 0, 129], device="cuda", dtype=torch.int32)
                    )
                    live.copy_(
                        torch.tensor([width, 0, 1], device="cuda", dtype=torch.int32)
                    )
                    slot = int(table[0, 2]) * page
                    slots = torch.tensor([slot], device="cuda", dtype=torch.int64)
                    valid = torch.ones(1, device="cuda", dtype=torch.bool)
                    wk = torch.full(
                        (1, hk, dim), 13, device="cuda", dtype=torch.bfloat16
                    )
                    wv = torch.full_like(wk, 17)
                    commit_paged_gqa_kv(wk, wv, cache, slots, valid)
                    wk.fill_(-3)
                    wv.fill_(5)
                    commit_paged_gqa_kv(wk, wv, cache, slots, valid)
                    expected_cache = cache.clone()
                    graph.replay()
                    check()
                    replay = out.clone()
                    forward()
                    torch.testing.assert_close(out, replay, atol=0, rtol=0)
                    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
                    self.assertEqual(
                        pointers,
                        [
                            t.data_ptr()
                            for t in (q, k, v, cache, table, lengths, live, out)
                        ],
                    )
                    del graph


if __name__ == "__main__":
    unittest.main()

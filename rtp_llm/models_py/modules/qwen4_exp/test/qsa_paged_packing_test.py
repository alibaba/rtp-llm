"""Check packed paged QSA against the legacy path and independent FP32 attention."""

import itertools
import unittest

import torch
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    sparse_paged_gqa_attn,
    try_sm90_packed_paged_decode,
    try_sm90_packed_paged_prefill,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90",
)
class PagedPackingTest(unittest.TestCase):
    def inputs(self, page, lengths, prefixes):
        torch.manual_seed(41)
        rows = sum(lengths)
        width = max(p + n for p, n in zip(prefixes, lengths))
        pages = (width + page - 1) // page
        batch = len(lengths)
        table = (
            (torch.randperm(batch * pages, device="cuda") + 1)
            .reshape(batch, pages)
            .int()
        )
        cache = torch.randn(
            batch * pages + 1, 2, 1, page, 256, device="cuda", dtype=torch.bfloat16
        )
        # A padded physical row models the shared HybridPool allocation.
        backing = torch.empty(
            cache.shape[0], 2 * page * 256 + 256, device="cuda", dtype=cache.dtype
        )
        backing[:, : 2 * page * 256].copy_(cache.flatten(1))
        cache = backing[:, : 2 * page * 256].view_as(cache)
        q = torch.randn(rows, 12, 256, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(q)
        cu = torch.tensor(
            [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
        )
        prefix = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
        visible = torch.cat(
            [
                torch.arange(p + 1, p + n + 1, device="cuda")
                for p, n in zip(prefixes, lengths)
            ]
        )
        # Preserve non-sorted selection order, with holes and masked future ids.
        selected = torch.randperm(width, device="cuda")[:2051].repeat(rows, 1).int()
        if selected.shape[1] < 2051:
            selected = torch.nn.functional.pad(
                selected, (0, 2051 - selected.shape[1]), value=-1
            )
        selected[:, 7::17] = -1
        selected[0].fill_(-1)
        return q, cache, table, cu, prefix, selected, gate, visible

    def assert_reference(self, output, args, page, sample_rows):
        q, cache, table, cu, _, selected, gate, visible = args
        boundaries = cu.tolist()
        for row in sample_rows:
            request = next(i for i, end in enumerate(boundaries[1:]) if row < end)
            ids = selected[row].long()
            ids = ids[(ids >= 0) & (ids < visible[row])]
            if ids.numel():
                physical = table[request, ids // page].long()
                k = cache[physical, 0, 0, ids % page].float()
                v = cache[physical, 1, 0, ids % page].float()
                weights = (q[row].float() @ k.T / 16).softmax(-1)
                expected = (weights @ v) * gate[row].float().sigmoid()
            else:
                expected = torch.zeros_like(q[row]).float()
            torch.testing.assert_close(
                output[row].float(), expected, atol=0.004, rtol=0.025
            )

    def test_cached_ragged_prefill(self):
        for page, lengths, prefixes in (
            (64, [4800], [3200]),
            (64, [5200], [4800]),
            (64, [4093, 7], [64, 128]),
            (128, [4093, 7], [128, 256]),
        ):
            with self.subTest(page=page, lengths=lengths, prefixes=prefixes):
                args = self.inputs(page, lengths, prefixes)
                q, cache, table, cu, prefix, selected, gate, visible = args
                actual = try_sm90_packed_paged_prefill(
                    q,
                    cache,
                    table,
                    cu,
                    prefix,
                    selected,
                    page_size=page,
                    kv_head_num=1,
                    output_gate=gate,
                )
                self.assertIsNotNone(actual)
                for request, (start, end) in enumerate(
                    zip(cu.tolist(), cu.tolist()[1:])
                ):
                    legacy = sparse_paged_gqa_attn(
                        q[start:end].unsqueeze(0).transpose(1, 2),
                        cache,
                        table[request : request + 1],
                        visible[start:end].int().unsqueeze(0),
                        selected[start:end].unsqueeze(0),
                        page_size=page,
                        kv_head_num=1,
                        output_gate=gate[start:end].unsqueeze(0),
                        bthd_output=True,
                        validate_selected=False,
                    )[0]
                    torch.testing.assert_close(
                        actual[start:end], legacy, atol=0.004, rtol=0.025
                    )
                self.assert_reference(
                    actual, args, page, [0, 1, lengths[0] - 1, len(q) - 1]
                )

    def test_decode_graph_replay(self):
        for page in (64, 128):
            with self.subTest(page=page):
                args = self.inputs(page, [1] * 16, [2048 + page * i for i in range(16)])
                q, cache, table, _, _, selected, gate, visible = args

                def run():
                    return try_sm90_packed_paged_decode(
                        q,
                        cache,
                        table,
                        visible.int()[:, None],
                        selected[:, None],
                        page_size=page,
                        kv_head_num=1,
                        output_gate=gate.flatten(1),
                    )

                for _ in range(3):
                    output = run()
                self.assertIsNotNone(output)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()
                for _ in range(2):
                    graph.replay()
                    self.assert_reference(output.view_as(q), args, page, [0, 1, 15])
                    selected[:, 9:100] = -1


if __name__ == "__main__":
    unittest.main()

"""SM120 joint attention, planar cache strides and graph replay qualification."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def cpu_oracle(q, sinks, pools, valid):
    result = torch.zeros_like(q, device="cpu")
    q, sinks, valid = q.cpu().double(), sinks.cpu().double(), valid.cpu()
    decoded = []
    for cache, slots in pools:
        cpu = V41PagedCache(cache.raw.cpu(), cache.tag, cache.gen_num_per_cycle)
        decoded.append((cpu, slots.cpu()))
    for row in range(q.shape[0]):
        if not valid[row]:
            continue
        kv = torch.cat(
            [
                cache.read(
                    indices[row][indices[row] >= 0].contiguous(), backend="eager"
                )
                for cache, indices in decoded
            ]
        ).double()
        logits = q[row] @ kv.T * (512**-0.5)
        logits = torch.cat((logits, sinks[:, None]), -1)
        result[row] = (logits.softmax(-1)[:, :-1] @ kv).bfloat16()
    return result


class V41AttentionHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "V4.1 hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(414)

    def cache(self, tag):
        layout = dsv41_page_layouts(3)[tag]
        storage = torch.full(
            (3, layout.stride_bytes + 128), 173, dtype=torch.uint8, device="cuda"
        )
        cache = V41PagedCache(storage[:, : layout.stride_bytes], tag, 3)
        count = 3 * layout.entries
        values = torch.randn(count, 512, dtype=torch.bfloat16, device="cuda")
        cache.write(values, torch.arange(count, device="cuda"))
        return cache

    def fixture(self, rows, heads, swa_width, global_width, tag):
        swa, glob = self.cache("swa_kv"), self.cache(tag)
        q = torch.randn(rows, heads, 512, dtype=torch.bfloat16, device="cuda")
        sinks = torch.linspace(-2, 3, heads, device="cuda")
        # Exercise both slot widths, holes and page-boundary relocation.
        swa_slots = torch.randint(
            0,
            3 * swa.layout.entries,
            (rows, swa_width),
            dtype=torch.int32,
            device="cuda",
        )
        global_slots = torch.randint(
            0,
            3 * glob.layout.entries,
            (rows, global_width),
            dtype=torch.int64,
            device="cuda",
        )
        swa_slots[:, 1::5] = -1
        global_slots[:, 2::7] = -1
        valid = torch.ones(rows, dtype=torch.bool, device="cuda")
        if rows > 1:
            valid[-1] = False
            q[-1] = float("nan")
            swa_slots[-1] = 2**30
            global_slots[-1] = -99
        return q, sinks, swa, swa_slots, valid, glob, global_slots

    def test_joint_attention_matches_double_oracle(self):
        op = V41SparseAttention()
        for rows, heads, sw, gw in ((1, 1, 1, 1), (4, 16, 31, 33), (17, 64, 128, 512)):
            for tag in ("csa_kv", "hca_kv"):
                with self.subTest(rows=rows, heads=heads, tag=tag):
                    q, sink, swa, slots, valid, glob, selected = self.fixture(
                        rows, heads, sw, gw, tag
                    )
                    originals = [swa.raw.clone(), glob.raw.clone()]
                    out = op(
                        q,
                        sink,
                        swa,
                        slots,
                        valid_rows=valid,
                        global_cache=glob,
                        global_slots=selected,
                    )
                    expected = cpu_oracle(
                        q, sink, [(swa, slots), (glob, selected)], valid
                    )
                    torch.testing.assert_close(
                        out.cpu(), expected, atol=0.004, rtol=0.01
                    )
                    for cache, saved in zip((swa, glob), originals):
                        torch.testing.assert_close(cache.raw, saved, atol=0, rtol=0)

    def test_swa_only_empty_slots_and_extreme_sinks(self):
        q, sink, swa, slots, valid, _, _ = self.fixture(4, 64, 128, 0, "hca_kv")
        sink[0], sink[1] = -1e38, 1e38
        op = V41SparseAttention()
        for indices in (slots, torch.full_like(slots, -1), slots[:, :0]):
            out = op(q, sink, swa, indices, valid_rows=valid)
            expected = cpu_oracle(q, sink, [(swa, indices)], valid)
            torch.testing.assert_close(out.cpu(), expected, atol=0.004, rtol=0.01)
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_graph_reads_changed_queries_sinks_slots_masks_and_cache(self):
        q, sink, swa, slots, valid, glob, selected = self.fixture(
            4, 64, 128, 512, "hca_kv"
        )
        op = V41SparseAttention()
        args = dict(valid_rows=valid, global_cache=glob, global_slots=selected)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm"):
                op(q, sink, swa, slots, **args)
        out = torch.empty_like(q)
        for _ in range(2):
            op(q, sink, swa, slots, out=out, **args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(q, sink, swa, slots, out=out, **args)
        for all_dead in (False, True, False):
            q.normal_()
            sink.uniform_(-3, 2)
            valid.fill_(not all_dead)
            slots.random_(0, 3 * swa.layout.entries)
            selected.random_(0, 3 * glob.layout.entries)
            slots[:, 1::3] = -1
            selected[:, 2::5] = -1
            for cache in (swa, glob):
                count = 3 * cache.layout.entries
                cache.write(
                    torch.randn(count, 512, dtype=torch.bfloat16, device="cuda"),
                    torch.arange(count, device="cuda"),
                )
            graph.replay()
            torch.cuda.synchronize()
            expected = cpu_oracle(q, sink, [(swa, slots), (glob, selected)], valid)
            torch.testing.assert_close(out.cpu(), expected, atol=0.004, rtol=0.01)


if __name__ == "__main__":
    unittest.main()

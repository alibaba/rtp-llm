"""Real original-table FA4 gate: tail writes, padding, replay and commit."""

import importlib.util
from pathlib import Path
import unittest

import torch


def load_kernel(name):
    path = Path(__file__).resolve().parents[2] / "triton_kernels" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkFA4Test(unittest.TestCase):
    def test_original_table_tail_and_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("Blackwell FA4 required")
        baseline = load_kernel("dspark_swa")
        fa4 = load_kernel("dspark_swa_fa4")
        torch.manual_seed(100818)
        # Long logical histories reuse physical history pages, but each request
        # owns two private tail pages. This tests 1M addressing, not 1M capacity.
        for batch, width, causal, i64 in (
            (3, 8, False, True), (4, 7, True, False),
            (8, 7, False, False), (8, 7, True, False),
            (12, 7, False, False), (12, 7, True, False),
            (20, 7, False, False), (24, 7, False, False),
            (1, 7, False, True),
        ):
            with self.subTest(batch=batch, width=width, causal=causal):
                lens = torch.full((batch,), 81920, device="cuda", dtype=torch.int64 if i64 else torch.int32)
                if batch == 3:
                    lens[:] = torch.tensor([127, 129, 1000000], device="cuda")
                if batch == 1:
                    lens.zero_()
                cols = (int(lens.max()) + width + 127) // 128
                table = torch.arange(cols, device="cuda").remainder(32)[None].repeat(batch, 1)
                table += torch.arange(batch, device="cuda")[:, None] * 34
                for b in range(batch):
                    tail = int(lens[b]) // 128
                    table[b, tail:] = torch.arange(cols - tail, device="cuda") + b * 34 + 32
                table = table.to(lens.dtype)
                payload = 2 * 4 * 128 * 128
                storage = torch.randn(batch * 34, payload + 256, device="cuda", dtype=torch.bfloat16)
                cache = storage[:, :payload].view(batch * 34, 2, 4, 128, 128)
                q = torch.randn(batch, width, 64, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(batch, width, 4, 128, device="cuda", dtype=torch.bfloat16)
                v = torch.randn_like(k)
                if batch == 20:
                    packed = torch.randn(batch, width, 8, 128, device="cuda", dtype=torch.bfloat16)
                    k, v = packed[:, :, :4], packed[:, :, 4:]
                live = torch.full_like(lens, width)
                if batch > 1:
                    if batch > 3:
                        live[-1] = 0
                    live[0] = 1
                out = torch.empty_like(q)
                def call():
                    fa4.paged_gqa_swa_fa4(q, k, v, cache, table, lens, live,
                                         causal=causal, window_left=4095, out=out)
                def check():
                    before = cache.clone()
                    expected = baseline.paged_gqa_swa(q, k.contiguous(), v.contiguous(), cache, table, lens, live,
                                                      causal=causal, window_left=4095)
                    call()
                    torch.testing.assert_close(out, expected, atol=0.012, rtol=0.025)
                    for b in range(batch):
                        for row in range(int(live[b])):
                            pos = int(lens[b]) + row
                            page = int(table[b, pos // 128])
                            before[page, 0, :, pos % 128] = k[b, row]
                            before[page, 1, :, pos % 128] = v[b, row]
                        self.assertEqual(int(out[b, int(live[b]):].count_nonzero()), 0)
                    self.assertTrue(torch.equal(cache, before), "only live tail rows may change")
                check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                for n in (0, 1, width):
                    k.normal_(); v.normal_(); q.normal_()
                    live[0] = n
                    if int(lens[0]) >= 4096:
                        table[0, :32] = table[0, :32].roll(1)
                    expected = baseline.paged_gqa_swa(q, k.contiguous(), v.contiguous(), cache, table, lens, live,
                                                      causal=causal, window_left=4095)
                    graph.replay()
                    torch.testing.assert_close(out, expected, atol=0.012, rtol=0.025)
                # Simulate feature commit of a=1 accepted row, then next query.
                if int(lens[0]) > 0:
                    pos = int(lens[0])
                    slot = int(table[0, pos // 128]) * 128 + pos % 128
                    ck, cv = torch.randn_like(k[0, :1]), torch.randn_like(v[0, :1])
                    baseline.commit_paged_gqa_kv(ck, cv, cache,
                        torch.tensor([slot], device="cuda"),
                        torch.tensor([True], device="cuda"))
                    lens[0] += 1
                    live[0] = width
                    graph.replay()
                    torch.testing.assert_close(cache[slot // 128, 0, :, slot % 128], ck[0], atol=0, rtol=0)
                    torch.testing.assert_close(cache[slot // 128, 1, :, slot % 128], cv[0], atol=0, rtol=0)
                    # Full verify commit includes the target bonus row. The
                    # next proposal must preserve all newly accepted features.
                    start = int(lens[0])
                    count = width + 1
                    # Extend this short synthetic table's private tail mapping
                    # only when it already has the required logical columns.
                    if (start + count + width + 127) // 128 <= cols:
                        positions = torch.arange(start, start + count, device="cuda")
                        slots = table[0, positions // 128].long() * 128 + positions % 128
                        ck = torch.randn(count, 4, 128, device="cuda", dtype=torch.bfloat16)
                        cv = torch.randn_like(ck)
                        baseline.commit_paged_gqa_kv(ck, cv, cache, slots,
                            torch.ones(count, device="cuda", dtype=torch.bool))
                        lens[0] += count
                        graph.replay()
                        for j, physical in enumerate(slots.tolist()):
                            torch.testing.assert_close(cache[physical // 128, 0, :, physical % 128], ck[j], atol=0, rtol=0)
                            torch.testing.assert_close(cache[physical // 128, 1, :, physical % 128], cv[j], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

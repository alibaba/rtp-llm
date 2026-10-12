"""Qualify upstream SWA on SM120's unsharded tagged ring geometry."""

import unittest

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8.test import test_v41_swa_codec
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


class TaggedSwaCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(410528)

    def make_cache(self, gen_num_per_cycle, device="cuda"):
        layout = dsv41_page_layouts(gen_num_per_cycle)["swa_kv"]
        stride = layout.stride_bytes + 64
        storage = torch.full((3 * stride + 32,), 0x5A, device=device, dtype=torch.uint8)
        raw = storage[16:-16].view(3, stride)[:, : layout.stride_bytes]
        cache = V41PagedCache(raw, "swa_kv", gen_num_per_cycle)
        pool = raw.as_strided((3, layout.entries, 528), (stride, 528, 1))
        return storage, cache, pool

    def test_tagged_ring_bytes_padding_and_generation_slack(self):
        for gen in (0, 1, 7):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(gen=gen, dtype=dtype):
                    storage, cache, pool = self.make_cache(gen)
                    expected, reference, _ = self.make_cache(gen, "cpu")
                    entries = cache.layout.entries
                    slots = torch.tensor(
                        [entries, 2 * entries - 1, 2 * entries, 3 * entries - 1, -1],
                        device="cuda",
                    )
                    values = torch.randn(5, 528, device="cuda", dtype=dtype)[:, :512]
                    values[0].zero_()
                    values[1] = torch.tensor(
                        [
                            448.0,
                            1.0625,
                            1.9375,
                            2.0**-10,
                            3 * 2.0**-10,
                            -0.0,
                            -448.0,
                            2.0,
                        ]
                        * 64,
                        device="cuda",
                    )
                    reference.write(
                        values.cpu().contiguous(), slots.cpu(), backend="eager"
                    )
                    codec.quantize_and_insert_swa_k_cache(values, pool, slots)
                    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
                    out = torch.full((5, 544), -73, device="cuda", dtype=dtype)
                    result = out[:, :512]
                    self.assertIs(
                        codec.dequantize_swa_k_cache(pool, slots, out=result), result
                    )
                    torch.testing.assert_close(
                        result.cpu(),
                        reference.read(slots.cpu(), out_dtype=dtype, backend="eager"),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(out[:, 512:].eq(-73).all())
                    torch.testing.assert_close(
                        cache.read(slots, out_dtype=dtype), result, rtol=0, atol=0
                    )
                    values.normal_()
                    cache.write(values.contiguous(), slots)
                    torch.testing.assert_close(
                        codec.dequantize_swa_k_cache(pool, slots, out_dtype=dtype),
                        cache.read(slots, out_dtype=dtype),
                        rtol=0,
                        atol=0,
                    )

    def test_tagged_ring_graph_changes_slots_lengths_and_keys(self):
        for gen in (0, 1, 7):
            with self.subTest(gen=gen):
                storage, cache, pool = self.make_cache(gen)
                entries = cache.layout.entries
                values = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
                slots = torch.tensor(
                    [entries - 1, entries, 2 * entries - 1, -1], device="cuda"
                )
                lengths = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
                out = torch.empty(2, 4, 528, device="cuda", dtype=torch.bfloat16)

                def run():
                    codec.quantize_and_insert_swa_k_cache(values, pool, slots)
                    out.fill_(-73)
                    codec.dequantize_and_gather_k_cache_slots(
                        out[:, :, :512], pool, slots.view(2, 2), lengths, 1
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for iteration in range(3):
                    values.normal_()
                    slots[:3].add_(1)
                    slots[-1] = 2 * entries + 11 if iteration == 1 else -1
                    lengths.copy_(torch.tensor([iteration, 2], device="cuda"))
                    reference = V41PagedCache(cache.raw.cpu(), "swa_kv", gen)
                    reference.write(values.cpu(), slots.cpu(), backend="eager")
                    expected = torch.full_like(out.cpu(), -73)
                    rows = reference.read(slots.cpu(), backend="eager").view(2, 2, 512)
                    for batch, length in enumerate(lengths.cpu().tolist()):
                        expected[batch, 1 : 1 + length, :512] = rows[batch, :length]
                    graph.replay()
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        cache.raw.cpu(), reference.raw, rtol=0, atol=0
                    )
                    self.assertTrue(storage[:16].eq(0x5A).all())
                    self.assertTrue(storage[-16:].eq(0x5A).all())


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Upstream SWA qualification requires a real SM120 GPU")
    for case in (
        test_v41_swa_codec.V41SwaCodecContractTest,
        test_v41_swa_codec.V41SwaCodecGpuTest,
        test_v41_swa_codec.V41SwaGatherOutputTest,
    ):
        tests.addTests(loader.loadTestsFromTestCase(case))
    # BatchedBoundedSWA needs the attention owner and CP metadata integration.
    return tests


if __name__ == "__main__":
    unittest.main()

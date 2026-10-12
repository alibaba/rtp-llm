"""Qualify upstream FP4 codecs against SM120 tagged-cache page geometry."""

import unittest

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_fp4_codec,
    test_v41_fp4_transport,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def _views(storage, tag):
    layout = dsv41_page_layouts(0)[tag]
    stride = layout.stride_bytes + 64
    raw = storage[16:-16].view(4, stride)[:, : layout.stride_bytes]
    cache = V41PagedCache(raw, tag, gen_num_per_cycle=0)
    row_bytes = layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
    pool = raw.as_strided((4, layout.entries, row_bytes), (stride, row_bytes, 1))
    return cache, pool


def _storage(tag):
    stride = dsv41_page_layouts(0)[tag].stride_bytes + 64
    return torch.full((4 * stride + 32,), 0x5A, device="cuda", dtype=torch.uint8)


def _write(tag, values, pool, slots):
    if tag == "indexer_kv":
        codec.quantize_indexer_k_fp4(values, slots, pool)
    else:
        codec.quantize_and_insert_k_cache_fp4(values, pool, slots)


def _read(tag, pool, slots, out):
    if tag == "indexer_kv":
        return codec.dequantize_indexer_k_fp4(pool, slots, out=out)
    return codec.dequantize_k_cache_slots_fp4(pool, slots, out=out)


class TaggedCacheCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(410288)

    def test_tagged_planes_full_index_pages_and_padding(self):
        for tag in ("csa_kv", "hca_kv", "indexer_kv"):
            with self.subTest(tag=tag):
                storage = _storage(tag)
                cache, pool = _views(storage, tag)
                expected = storage.cpu()
                reference, _ = _views(expected, tag)
                entries = cache.layout.entries
                slots = torch.tensor(
                    [entries, 2 * entries - 1, 3 * entries, entries + 1, -1],
                    dtype=torch.int64,
                    device="cuda",
                )
                dim = 128 if tag == "indexer_kv" else 512
                values = torch.randn(5, dim, device="cuda", dtype=torch.bfloat16)
                values[0].zero_()
                reference.write(values.cpu(), slots.cpu(), backend="eager")
                _write(tag, values, pool, slots)
                torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
                # Read donor bytes through the existing tagged-cache provider.
                torch.testing.assert_close(
                    cache.read(slots, out_dtype=torch.float32).cpu(),
                    reference.read(
                        slots.cpu(), out_dtype=torch.float32, backend="eager"
                    ),
                    rtol=0,
                    atol=0,
                )
                if tag == "indexer_kv":
                    payload, scales = codec.gather_indexer_k_fp4(pool, slots)
                    packed = torch.cat(
                        (
                            payload.view(torch.uint8),
                            scales.view(torch.uint8).view(-1, 4),
                        ),
                        dim=1,
                    )
                    # Ratio-2 GLOBAL pages have 128 entries; index pages retain 256.
                    self.assertEqual(entries, 256)
                else:
                    packed = codec.gather_k_cache_bytes_fp4(pool, slots)
                torch.testing.assert_close(
                    packed.cpu(),
                    reference.gather(slots.cpu(), backend="eager"),
                    rtol=0,
                    atol=0,
                )
                # Read independently encoded tagged-cache bytes with the donor.
                values.normal_()
                cache.write(values, slots)
                out = torch.empty(5, dim, device="cuda", dtype=torch.float32)
                self.assertIs(_read(tag, pool, slots, out), out)
                torch.testing.assert_close(
                    out, cache.read(slots, out_dtype=torch.float32), rtol=0, atol=0
                )

    def test_graph_replay_mutates_writes_slots_and_output(self):
        for tag in ("csa_kv", "hca_kv", "indexer_kv"):
            with self.subTest(tag=tag):
                storage = _storage(tag)
                cache, pool = _views(storage, tag)
                entries = cache.layout.entries
                dim = 128 if tag == "indexer_kv" else 512
                values = torch.randn(5, dim, device="cuda", dtype=torch.bfloat16)
                slots = torch.tensor(
                    [entries - 1, entries, 2 * entries - 1, 3 * entries, -1],
                    dtype=torch.int64,
                    device="cuda",
                )
                out = torch.empty(5, dim, device="cuda", dtype=torch.float32)

                def run():
                    _write(tag, values, pool, slots)
                    _read(tag, pool, slots, out)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for iteration in range(3):
                    expected = storage.cpu()
                    reference, _ = _views(expected, tag)
                    values.normal_()
                    slots[:4].add_(1)
                    slots[-1] = entries + 17 if iteration == 1 else -1
                    reference.write(values.cpu(), slots.cpu(), backend="eager")
                    graph.replay()
                    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        out.cpu(),
                        reference.read(
                            slots.cpu(), out_dtype=torch.float32, backend="eager"
                        ),
                        rtol=0,
                        atol=0,
                    )


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Upstream FP4 qualification requires a real SM120 GPU")
    tests.addTests(loader.loadTestsFromModule(test_v41_fp4_codec))
    tests.addTests(loader.loadTestsFromModule(test_v41_fp4_transport))
    return tests


if __name__ == "__main__":
    unittest.main()

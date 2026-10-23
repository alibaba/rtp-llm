"""FP4 gather geometry and compilation checks on SM120."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp4.score import gather_cache


def write_reference_pages(packed, scales, entries=64):
    blocks = (packed.shape[0] + entries - 1) // entries
    pool = torch.full(
        (blocks + 1, entries, 68), 0xAB, dtype=torch.uint8, device=packed.device
    )
    for block in range(blocks):
        lo, hi = block * entries, min((block + 1) * entries, packed.shape[0])
        page = pool[block + 1].flatten()
        page[: entries * 64].view(entries, 64)[: hi - lo] = packed[lo:hi]
        page[entries * 64 :].view(entries, 4)[: hi - lo] = scales[lo:hi]
    return pool


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Fp4IndexerContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)
        self.device = torch.device("cuda")

    def test_gather_request_shapes_use_bounded_compilation(self):
        from rtp_llm.models_py.modules.dsv4.fp4.score import _gather_cache

        _gather_cache.device_caches.clear()
        for batch, length in (
            (1, 17),
            (2, 31),
            (3, 64),
            (4, 65),
            (7, 129),
            (2, 257),
            (11, 73),
            (17, 65),
            (3, 4097),
            (1, 8191),
        ):
            with self.subTest(batch=batch, length=length):
                packed = torch.randint(
                    0, 256, (length, 64), dtype=torch.uint8, device=self.device
                )
                scales = torch.randint(
                    0, 256, (length, 4), dtype=torch.uint8, device=self.device
                )
                pool = write_reference_pages(packed, scales)
                table = torch.arange(
                    1, pool.shape[0], dtype=torch.int32, device=self.device
                ).repeat(batch, 1)
                lengths = (
                    torch.arange(batch + 1, dtype=torch.int32, device=self.device)
                    * length
                )
                out = torch.empty(
                    (batch * length, 64), dtype=torch.uint8, device=self.device
                )
                out_scales = torch.empty(
                    (batch * length, 4), dtype=torch.uint8, device=self.device
                )
                gather_cache(pool, out, out_scales, table, lengths)
                torch.testing.assert_close(out, packed.repeat(batch, 1), rtol=0, atol=0)
                torch.testing.assert_close(
                    out_scales, scales.repeat(batch, 1), rtol=0, atol=0
                )
                self.assertLessEqual(
                    sum(
                        len(cache) for cache, *_ in _gather_cache.device_caches.values()
                    ),
                    2,
                )
        self.assertEqual(
            sum(len(cache) for cache, *_ in _gather_cache.device_caches.values()), 2
        )

    def test_gather_empty_requests_and_multiple_boundary_tiles(self):
        lengths = [0, 1, 17, 0, 65, 2, 64, 0, 3, 129, 5]
        capacity = 192
        packed = torch.randint(
            0, 256, (len(lengths) * capacity, 64), dtype=torch.uint8, device=self.device
        )
        scales = torch.randint(
            0, 256, (len(lengths) * capacity, 4), dtype=torch.uint8, device=self.device
        )
        pool = write_reference_pages(packed, scales)
        table = torch.arange(
            1, pool.shape[0], dtype=torch.int32, device=self.device
        ).reshape(len(lengths), -1)
        cu = torch.tensor([0] + lengths, dtype=torch.int32, device=self.device).cumsum(
            0, dtype=torch.int32
        )
        out = torch.empty((sum(lengths), 64), dtype=torch.uint8, device=self.device)
        out_scales = torch.empty(
            (sum(lengths), 4), dtype=torch.uint8, device=self.device
        )
        gather_cache(pool, out, out_scales, table, cu)
        for actual, source in ((out, packed), (out_scales, scales)):
            expected = torch.cat(
                [
                    source[i * capacity : i * capacity + length]
                    for i, length in enumerate(lengths)
                ]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

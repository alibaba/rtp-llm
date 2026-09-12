"""Reject overlapping compact writer buffers before any GPU write."""

import unittest

import test_compact_writer as fixture
import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import ENCODINGS, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact, write_compact


class CompactWriterAliasGpuTest(unittest.TestCase):
    def reject_unchanged(self, operation, tensors):
        before = [tensor.clone() for tensor in tensors]
        with self.assertRaisesRegex(ValueError, "alias"):
            operation()
        for actual, expected in zip(tensors, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_encode_aliases_are_rejected(self):
        for region in ENCODINGS:
            rows, dim, width = (
                4,
                ENCODINGS[region].head_dim,
                ENCODINGS[region].entry_bytes,
            )
            for alias in ("output_input", "status_input", "status_output"):
                with self.subTest(region=region.value, alias=alias):
                    values = torch.ones(
                        (rows, dim), dtype=torch.bfloat16, device="cuda"
                    )
                    output = torch.full(
                        (rows, width), 91, dtype=torch.uint8, device="cuda"
                    )
                    status = torch.full((rows,), 11, dtype=torch.int32, device="cuda")
                    if alias == "output_input":
                        output = (
                            values.view(torch.uint8)
                            .flatten()[: rows * width]
                            .view(rows, width)
                        )
                    elif alias == "status_input":
                        status = values.view(torch.int32).flatten()[:rows]
                    else:
                        status = output.view(torch.int32).flatten()[:rows]
                    self.reject_unchanged(
                        lambda: encode_compact(
                            values, region, output=output, status=status
                        ),
                        (values, output, status),
                    )

    @torch.inference_mode()
    def test_scatter_aliases_are_rejected(self):
        for region in ENCODINGS:
            rows, dim = 4, ENCODINGS[region].head_dim
            for alias in (
                "pages_input",
                "pages_slots",
                "status_input",
                "status_pages",
                "status_slots",
            ):
                with self.subTest(region=region.value, alias=alias):
                    pages, storage = fixture._pages(region, rows)
                    values = torch.ones(
                        (rows, dim), dtype=torch.bfloat16, device="cuda"
                    )
                    slots = torch.arange(
                        128, 128 + rows, dtype=torch.int32, device="cuda"
                    )
                    status = torch.full((rows,), 13, dtype=torch.int32, device="cuda")
                    if alias == "pages_input":
                        values = (
                            pages.data[1, : rows * dim * 2]
                            .view(torch.bfloat16)
                            .view(rows, dim)
                        )
                        values.fill_(1)
                    elif alias == "pages_slots":
                        slots = pages.data[1, : rows * 4].view(torch.int32)
                        slots.copy_(torch.arange(128, 128 + rows, device="cuda"))
                    elif alias == "status_input":
                        status = values.view(torch.int32).flatten()[:rows]
                    elif alias == "status_pages":
                        status = pages.data[1, : rows * 4].view(torch.int32)
                    else:
                        status = slots
                    self.reject_unchanged(
                        lambda: write_compact(values, pages, slots, status=status),
                        (values, storage, slots, status),
                    )

    @torch.inference_mode()
    def test_disjoint_views_share_allocation_and_replay_with_changed_inputs(self):
        region, rows, dim = CacheRegion.SWA, 4, 512
        width = ENCODINGS[region].entry_bytes
        storage = torch.empty(
            rows * (dim * 2 + width + 4), dtype=torch.uint8, device="cuda"
        )
        split = rows * dim * 2
        values = storage[:split].view(torch.bfloat16).view(rows, dim)
        output = storage[split : split + rows * width].view(rows, width)
        status = storage[split + rows * width :].view(torch.int32)
        values.fill_(1)
        encode_compact(values, region, output=output, status=status).check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = encode_compact(values, region, output=output, status=status)
        for value in (2, -3):
            values.fill_(value)
            graph.replay()
            result.check()
            expected = encode_compact(values.clone(), region)
            expected.check()
            torch.testing.assert_close(output, expected.output, rtol=0, atol=0)
            torch.testing.assert_close(
                values, torch.full_like(values, value), rtol=0, atol=0
            )
        graph.reset()


if __name__ == "__main__":
    unittest.main()

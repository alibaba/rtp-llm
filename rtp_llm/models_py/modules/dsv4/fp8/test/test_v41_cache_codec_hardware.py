"""SM120 codec and graph gate; missing hardware fails instead of skipping."""

import unittest

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.v41_cache import (
    V41PagedCache,
    dequantize_rows_eager,
    quantize_rows_eager,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
    V41CacheIdentity,
    V41L20Contract,
    V41L20ReplicaReceiver,
    V41L20Span,
    export_l20_delta,
)

TAGS = ("csa_kv", "hca_kv", "indexer_kv", "swa_kv")


class V41CacheCodecHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "V4.1 hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(412)

    def test_l20_delta_import_commits_both_regions_after_relocation(self):
        contract = V41L20Contract("a" * 64, "b" * 64, "c" * 64, 4096)
        identity = V41CacheIdentity(101, 1, 12)
        source, target = {}, {}
        for tag in ("hca_kv", "indexer_kv"):
            stride = dsv41_page_layouts(0)[tag].stride_bytes
            source[tag] = V41PagedCache(
                torch.randint(0, 256, (3, stride), dtype=torch.uint8, device="cuda"),
                tag,
            )
            target[tag] = V41PagedCache(torch.zeros_like(source[tag].raw), tag)
        producer = {"hca_kv": [2, 0], "indexer_kv": [0, 2]}
        replica = {"hca_kv": [0, 1], "indexer_kv": [2, 1]}
        receiver = V41L20ReplicaReceiver(contract, 0, target)
        receiver.begin_request(identity, replica)
        for chunk, start, count in ((0, 0, 257), (1, 257, 31)):
            spans = [V41L20Span(identity, chunk, start, count)]
            message = export_l20_delta(
                contract, 0, chunk, spans, source, {identity: producer}
            )
            receiver.apply(message, spans)
            receiver.require_ready(identity, start + count, chunk)
            for tag, key in (
                ("hca_kv", "v41_l20_global_delta"),
                ("indexer_kv", "v41_l20_index_delta"),
            ):
                local_slots = torch.tensor(
                    [
                        replica[tag][i // 256] * 256 + i % 256
                        for i in range(start, start + count)
                    ],
                    device="cuda",
                )
                torch.testing.assert_close(
                    target[tag].gather(local_slots), message[key], rtol=0, atol=0
                )

    def test_codec_bytes_and_reads(self):
        for tag in TAGS:
            layout = dsv41_page_layouts(1)[tag]
            dim = 128 if tag == "indexer_kv" else 512
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(tag=tag, dtype=dtype):
                    raw = torch.full(
                        (3, layout.stride_bytes + 64),
                        173,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    cache = V41PagedCache(raw[:, : layout.stride_bytes], tag)
                    expected_raw = raw.cpu()
                    expected = V41PagedCache(
                        expected_raw[:, : layout.stride_bytes], tag
                    )
                    values = torch.randn(7, dim, dtype=dtype)
                    values[0] = torch.tensor(
                        [
                            0.25,
                            0.75,
                            1.25,
                            1.75,
                            2.5,
                            3.5,
                            5,
                            6,
                            -0.25,
                            -0.75,
                            -1.25,
                            -1.75,
                            -2.5,
                            -3.5,
                            -5,
                            -6,
                        ]
                        * (dim // 16)
                    )
                    values[1].zero_()
                    values[-1].fill_(float("nan"))
                    entries = layout.entries
                    slots = torch.tensor(
                        [
                            0,
                            entries - 1,
                            entries,
                            2 * entries - 1,
                            2 * entries,
                            3 * entries - 1,
                            -1,
                        ]
                    )
                    expected.write(values, slots, backend="eager")
                    cache.write(values.cuda(), slots.cuda())
                    torch.testing.assert_close(raw.cpu(), expected_raw, rtol=0, atol=0)
                    torch.testing.assert_close(
                        cache.gather(slots.cuda()).cpu(),
                        expected.gather(slots, backend="eager"),
                        rtol=0,
                        atol=0,
                    )
                    actual = cache.read(slots.cuda(), out_dtype=dtype).cpu()
                    torch.testing.assert_close(
                        actual,
                        expected.read(slots, out_dtype=dtype, backend="eager"),
                        rtol=0,
                        atol=0,
                    )
                    if tag != "swa_kv":
                        self.assertTrue(bool(torch.signbit(actual[0, 8])))

    def test_fp8_rounding_boundaries_and_scale_extremes(self):
        for tag in TAGS:
            dim = 128 if tag == "indexer_kv" else 512
            layout = dsv41_page_layouts(1)[tag]
            maxima = torch.tensor(
                [0.0, 1e-38, 6 * 2.0**-9, 6.375, 11.625, 2688.0, 3000.0, 3e38]
            )
            values = maxima[:, None].repeat(1, dim)
            if tag == "swa_kv":
                # Fix scale=1 and isolate ties, exponent carry and subnormals.
                values[3] = torch.tensor(
                    [
                        448.0,
                        1.0625,
                        1.9375,
                        2.0**-10,
                        3 * 2.0**-10,
                        -0.0,
                        -1.0625,
                        -1.9375,
                    ]
                    * (dim // 8)
                )
            slots = torch.arange(values.shape[0], dtype=torch.int32, device="cuda")
            cache = V41PagedCache(
                torch.zeros(1, layout.stride_bytes, dtype=torch.uint8, device="cuda"),
                tag,
            )
            cache.write(values.cuda(), slots)
            expected = quantize_rows_eager(values, tag)
            torch.testing.assert_close(
                cache.gather(slots).cpu(), expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                cache.read(slots, out_dtype=torch.float32).cpu(),
                dequantize_rows_eager(expected, tag, torch.float32),
                rtol=0,
                atol=0,
            )

    def test_arbitrary_packed_bytes_transfer_without_requantization(self):
        for tag in TAGS:
            layout = dsv41_page_layouts(1)[tag]
            width = layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
            raw = torch.full(
                (3, layout.stride_bytes + 64), 173, dtype=torch.uint8, device="cuda"
            )
            target = V41PagedCache(raw[:, : layout.stride_bytes], tag)
            expected_raw = raw.cpu()
            expected = V41PagedCache(expected_raw[:, : layout.stride_bytes], tag)
            rows = torch.arange(5 * width).remainder(256).to(torch.uint8).view(5, width)
            slots = torch.tensor(
                [0, layout.entries - 1, layout.entries, 3 * layout.entries - 1, -1],
                dtype=torch.int32,
            )
            target.scatter(rows.cuda(), slots.cuda())
            expected.scatter(rows, slots, backend="eager")
            torch.testing.assert_close(raw.cpu(), expected_raw, rtol=0, atol=0)
            rows[-1].zero_()
            torch.testing.assert_close(
                target.gather(slots.cuda()).cpu(), rows, rtol=0, atol=0
            )

    def test_graph_replay_with_updated_rows_and_slots(self):
        for tag in TAGS:
            layout = dsv41_page_layouts(1)[tag]
            dim = 128 if tag == "indexer_kv" else 512
            width = layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
            source = V41PagedCache(
                torch.zeros(3, layout.stride_bytes, dtype=torch.uint8, device="cuda"),
                tag,
            )
            target = V41PagedCache(torch.zeros_like(source.raw), tag)
            values = torch.randn(5, dim, dtype=torch.bfloat16, device="cuda")
            src_slots = torch.tensor(
                [0, 1, layout.entries, 2 * layout.entries, -1], device="cuda"
            )
            dst_slots = torch.tensor(
                [layout.entries - 1, 2, 2 * layout.entries + 1, layout.entries + 3, -1],
                device="cuda",
            )
            packed = torch.empty(5, width, dtype=torch.uint8, device="cuda")
            result = torch.empty_like(values)

            def run():
                source.write(values, src_slots)
                source.gather(src_slots, out=packed)
                target.scatter(packed, dst_slots)
                target.read(dst_slots, out=result)

            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                run()
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(3):
                values.copy_(torch.randn_like(values))
                src_slots[:4].add_(1)
                dst_slots[:4].add_(1)
                graph.replay()
                expected = quantize_rows_eager(values.cpu(), tag)
                expected[-1].zero_()
                torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    result.cpu(), dequantize_rows_eager(expected, tag), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()

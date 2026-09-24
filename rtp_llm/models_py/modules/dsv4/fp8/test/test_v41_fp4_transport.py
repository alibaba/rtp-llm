"""Byte transport coverage, identity decoding, and graph replay on Blackwell."""

import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec


class V41Fp4TransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("FP4 transport requires Blackwell")

    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )

    def make_pool(self, entries, padding):
        stride = entries * 288 + padding
        storage = torch.randint(
            0, 256, (4 * stride + 2,), device="cuda", dtype=torch.uint8
        )
        pool = storage[1:-1].as_strided((4, entries, 288), (stride, 288, 1))
        pages = storage[1:-1].view(4, stride)
        rows = torch.cat(
            (
                pages[:, : entries * 256].reshape(4 * entries, 256),
                pages[:, entries * 256 : entries * 288].reshape(4 * entries, 32),
            ),
            dim=1,
        )
        return storage, pool, rows

    def test_gather_poison_canaries_padding_and_arbitrary_slots(self):
        for entries in (1, 64, 128):
            for padding in (0, 7, 512):
                with self.subTest(entries=entries, padding=padding):
                    storage, pool, rows = self.make_pool(entries, padding)
                    before = storage.clone()
                    # Non-contiguous int32 input also exercises the normalization path.
                    ids = (
                        torch.tensor(
                            [
                                3 * entries,
                                -1,
                                entries - 1,
                                -2147483648,
                                0,
                                0,
                                -2,
                                4 * entries - 1,
                            ],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        .view(2, 4)
                        .t()
                    )
                    flat = ids.flatten().long()
                    expected = rows[flat.clamp_min(0)]
                    expected[flat < 0] = 0
                    guard = torch.full(
                        (ids.numel() * 288 + 2,), 0xA5, device="cuda", dtype=torch.uint8
                    )
                    target = guard[1:-1].view(-1, 288)
                    real_empty = torch.empty

                    def poisoned_empty(shape, **kwargs):
                        if (
                            tuple(shape) == tuple(target.shape)
                            and kwargs.get("dtype") == torch.uint8
                        ):
                            return target
                        return real_empty(shape, **kwargs)

                    with mock.patch.object(
                        codec.torch, "empty", side_effect=poisoned_empty
                    ):
                        actual = codec.gather_k_cache_bytes_fp4(pool, ids)
                    self.assertEqual(actual.data_ptr(), target.data_ptr())
                    self.assert_bits(actual, expected)
                    self.assertEqual(guard[0].item(), 0xA5)
                    self.assertEqual(guard[-1].item(), 0xA5)
                    self.assert_bits(storage, before)
                    self.assert_bits(
                        codec.dequantize_k_cache_bytes_fp4(actual),
                        codec.dequantize_k_cache_slots_fp4(pool, ids),
                    )

    def test_all_invalid_and_empty(self):
        _, pool, _ = self.make_pool(128, 9)
        for n in (0, 1, 7, 513):
            with self.subTest(n=n):
                slots = torch.full((n,), -9, device="cuda", dtype=torch.int64)
                raw = codec.gather_k_cache_bytes_fp4(pool, slots)
                self.assertEqual(tuple(raw.shape), (n, 288))
                self.assertEqual(torch.count_nonzero(raw).item(), 0)
                out = codec.dequantize_k_cache_bytes_fp4(raw)
                self.assertEqual(tuple(out.shape), (n, 512))
                self.assertEqual(torch.count_nonzero(out).item(), 0)

    def test_identity_all_codes_scales_strides_and_output_dtype(self):
        # Each payload byte meets every scale byte, including signed zero and NaN.
        for padding in (0, 7, 512):
            backing = torch.full(
                (256, 288 + padding), 0xA5, device="cuda", dtype=torch.uint8
            )
            raw = backing[:, :288]
            raw[:, :256] = torch.arange(256, device="cuda", dtype=torch.uint8)[None, :]
            raw[:, 256:] = torch.arange(256, device="cuda", dtype=torch.uint8)[:, None]
            before = backing.clone()
            ids = torch.arange(256, device="cuda", dtype=torch.int64)
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(padding=padding, dtype=dtype):
                    expected = codec.dequantize_k_cache_slots_fp4(
                        raw.view(256, 1, 288), ids, out_dtype=dtype
                    )
                    guard = torch.full((256 * 512 + 2,), 19, device="cuda", dtype=dtype)
                    out = guard[1:-1].view(256, 512)
                    with mock.patch.object(
                        codec.torch,
                        "arange",
                        side_effect=AssertionError("identity arange"),
                    ):
                        # Supplied output dtype takes precedence over the default.
                        actual = codec.dequantize_k_cache_bytes_fp4(raw, out=out)
                    self.assertIs(actual, out)
                    self.assert_bits(actual, expected)
                    self.assertEqual(guard[0].item(), 19)
                    self.assertEqual(guard[-1].item(), 19)
            self.assert_bits(backing, before)

    def test_four_owner_bytes_reassemble(self):
        _, pool, _ = self.make_pool(128, 17)
        ids = torch.arange(513, device="cuda", dtype=torch.int64) % 512
        ids[-1] = -1
        expected = codec.gather_k_cache_bytes_fp4(pool, ids)
        result = torch.zeros_like(expected)
        for rank in range(4):
            owned = torch.where((ids >= 0) & ((ids // 128) == rank), ids, -1)
            result += codec.gather_k_cache_bytes_fp4(pool, owned)
        self.assert_bits(result, expected)
        self.assert_bits(
            codec.dequantize_k_cache_bytes_fp4(result),
            codec.dequantize_k_cache_slots_fp4(pool, ids),
        )

    def test_graph_replay_mutated_slots_and_pool(self):
        storage, pool, _ = self.make_pool(128, 7)
        ids = torch.arange(257, device="cuda", dtype=torch.int64)
        for dtype in (torch.bfloat16, torch.float32):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    raw = codec.gather_k_cache_bytes_fp4(pool, ids)
                    codec.dequantize_k_cache_bytes_fp4(raw, out_dtype=dtype)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                raw = codec.gather_k_cache_bytes_fp4(pool, ids)
                actual = codec.dequantize_k_cache_bytes_fp4(raw, out_dtype=dtype)
            for invalid in (True, False, True):
                storage.random_(0, 256)
                ids.copy_(torch.arange(257, device="cuda", dtype=torch.int64).flip(0))
                if invalid:
                    ids[::3] = -1
                graph.replay()
                expected = codec.dequantize_k_cache_slots_fp4(
                    pool, ids, out_dtype=dtype
                )
                self.assert_bits(actual, expected)
                self.assert_bits(raw, codec.gather_k_cache_bytes_fp4(pool, ids))


if __name__ == "__main__":
    unittest.main()

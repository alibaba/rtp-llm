"""Bit-exact fused KV patch coverage, including untouched output storage."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.cp_kv_patch import restore_patch_kv


class PatchTest(unittest.TestCase):
    def check(self, device):
        for width in (1, 17, 576):
            for rows in (0, 1, 11, 257):
                with self.subTest(device=device, width=width, rows=rows):
                    src = torch.randint(
                        -32768,
                        32767,
                        (rows + 3, width),
                        dtype=torch.int16,
                        device=device,
                    )
                    initial = torch.randint(
                        -32768,
                        32767,
                        (rows * 2 + 7, width),
                        dtype=torch.int16,
                        device=device,
                    )
                    restore = torch.arange(
                        rows, device=device, dtype=torch.int64
                    ).remainder(3)
                    dest = torch.arange(rows, device=device, dtype=torch.int64) * 2 + 1
                    ref = initial.clone()
                    ref.index_copy_(0, dest, src.index_select(0, restore))
                    out = initial.clone()
                    restore_patch_kv(
                        src.view(torch.bfloat16),
                        restore,
                        dest,
                        out.view(torch.bfloat16),
                    )
                    self.assertTrue(torch.equal(out, ref))

    def test_cpu_reference(self):
        self.check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_bits(self):
        self.check("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_side_stream_dependency(self):
        producer = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        src = torch.arange(576 * 7, device="cuda", dtype=torch.int16).view(7, 576)
        dst = torch.full_like(src, -77)
        indices = torch.tensor([6, 2, 0], device="cuda")
        dest = torch.tensor([1, 3, 5], device="cuda")
        side.wait_stream(producer)
        with torch.cuda.stream(side):
            restore_patch_kv(
                src.view(torch.bfloat16), indices, dest, dst.view(torch.bfloat16)
            )
        producer.wait_stream(side)
        ref = torch.full_like(src, -77)
        ref.index_copy_(0, dest, src.index_select(0, indices))
        self.assertTrue(torch.equal(dst, ref))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_padded_k_scale_bytes(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _pack_padded_k_scale,
        )

        for actual_lens, padded_lens in [
            ([0], [0]),
            ([0], [5]),
            ([7], [7]),
            ([3], [8]),
            ([0, 3, 0, 2], [4, 4, 4, 4]),
            ([513, 1, 0], [768, 256, 256]),
        ]:
            for width, scale_width in [(128, 4), (256, 8), (17, 3)]:
                with self.subTest(actual=actual_lens, padded=padded_lens, width=width):
                    n, rows = sum(actual_lens), sum(padded_lens)
                    k = torch.randint(
                        0, 256, (n, width), dtype=torch.uint8, device="cuda"
                    )
                    scales = torch.randint(
                        0, 256, (n, scale_width), dtype=torch.uint8, device="cuda"
                    )
                    mapping = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
                    ref = torch.zeros(
                        (rows, width + scale_width), dtype=torch.uint8, device="cuda"
                    )
                    src = dst = 0
                    for actual, padded in zip(actual_lens, padded_lens):
                        mapping[dst : dst + actual] = torch.arange(
                            src, src + actual, device="cuda"
                        )
                        ref[dst : dst + actual, :width] = k[src : src + actual]
                        ref[dst : dst + actual, width:] = scales[src : src + actual]
                        src += actual
                        dst += padded
                    result = _pack_padded_k_scale(
                        k.view(torch.float8_e4m3fn), scales, rows, mapping
                    )
                    self.assertTrue(torch.equal(result, ref))
                    if n == rows or n == 0:
                        self.assertTrue(
                            torch.equal(_pack_padded_k_scale(k, scales, rows), ref)
                        )


if __name__ == "__main__":
    unittest.main()

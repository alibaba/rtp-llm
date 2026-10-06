"""Live rows/capacities must not trigger a new packed-cache/scale binary."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common import nvfp4_kv_cache as cache_op
from rtp_llm.models_py.triton_kernels.moe import mxfp8_kernels as scale_op


class KernelRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.hashes = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            compiled = self.kernel[grid](*args, **kwargs)
            self.hashes.append(compiled.hash)
            return compiled

        return launch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DynamicKernelGeometryTest(unittest.TestCase):
    def test_bounds_and_scale_stride_are_runtime_unspecialized(self):
        for kernel, names in (
            (
                cache_op._quantize_main_index_rows_d128_kernel,
                ("N", "NUM_BLOCKS", "PERSIST_NUM_BLOCKS"),
            ),
            (scale_op._pack_flashinfer_mxfp8_scale_kernel, ("M", "ALIGNED_MN")),
        ):
            params = {param.name: param for param in kernel.params}
            for name in names:
                with self.subTest(kernel=kernel.__name__, parameter=name):
                    self.assertFalse(params[name].is_constexpr)
                    self.assertTrue(params[name].do_not_specialize)

    def test_scale_rows_and_aligned_stride_reuse_binary(self):
        recorder = KernelRecorder(scale_op._pack_flashinfer_mxfp8_scale_kernel)
        with patch.object(scale_op, "_pack_flashinfer_mxfp8_scale_kernel", recorder):
            for rows in (129, 143, 161, 129):
                source = torch.randint(
                    0, 255, (rows, 192), device="cuda", dtype=torch.uint8
                )
                output = scale_op.pack_flashinfer_mxfp8_scale_triton(source, rows, 6144)
                # Independent byte/shift oracle for the unchanged TMA layout.
                groups = source.long().reshape(rows, 48, 4)
                shifts = torch.tensor([0, 8, 16, 24], device="cuda")
                expected = (groups << shifts).sum(-1).to(torch.int32)
                self.assertTrue(torch.equal(output, expected))
        self.assertEqual(len(set(recorder.hashes)), 1)

    def test_cp_writer_live_rows_and_both_capacities_reuse_binary(self):
        recorder = KernelRecorder(cache_op._quantize_main_index_rows_d128_kernel)

        def planes(pages):
            return tuple(
                torch.full(
                    (pages, h, 128, d), 0xA5, device="cuda", dtype=torch.uint8
                ).view(dtype)
                for h, d, dtype in (
                    (4, 64, torch.uint8), (4, 8, torch.float8_e4m3fn),
                    (4, 64, torch.uint8), (4, 8, torch.float8_e4m3fn),
                    (1, 64, torch.uint8), (1, 8, torch.float8_e4m3fn),
                )
            )

        with patch.object(cache_op, "_quantize_main_index_rows_d128_kernel", recorder):
            for rows, pages in ((19, 3), (23, 5), (32, 7), (19, 3)):
                source = torch.ones(rows, 1152, device="cuda", dtype=torch.bfloat16)
                indices = torch.arange(rows, device="cuda", dtype=torch.int64)
                slots = indices + 128
                slots[-3:] = torch.tensor(
                    [-1, pages * 128, pages * 128 + 3], device="cuda"
                )
                persistent_slots = indices + 128
                persistent_slots[-3:] = torch.tensor(
                    [pages * 128, -1, (pages + 1) * 128], device="cuda"
                )
                working, persistent = planes(pages), planes(pages + 1)
                cache_op.quantize_cp_main_index_rows_to_planes(
                    source,
                    indices,
                    slots,
                    *working,
                    persistent_slots=persistent_slots,
                    persistent_planes=persistent,
                )
                torch.cuda.synchronize()
                scale = (
                    torch.tensor(1 / 6, dtype=torch.float32)
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                    .item()
                )
                for outputs, row_slots in ((working, slots), (persistent, persistent_slots)):
                    expected = [
                        torch.full_like(p.view(torch.uint8), 0xA5) for p in outputs
                    ]
                    for slot in row_slots.cpu().tolist():
                        page, token = divmod(slot, 128)
                        if not 0 <= page < outputs[0].shape[0]:
                            continue
                        for plane in (0, 2, 4):
                            expected[plane][page, :, token].fill_(0x77)
                        for plane in (1, 3, 5):
                            heads = outputs[plane].shape[1]
                            mma = expected[plane].view(-1, heads, 2, 32, 4, 4)
                            mma[page, :, :, token % 32, token // 32, :].fill_(scale)
                    for actual, oracle in zip(outputs, expected):
                        self.assertTrue(torch.equal(actual.view(torch.uint8), oracle))
        self.assertEqual(len(set(recorder.hashes)), 1)


if __name__ == "__main__":
    unittest.main()

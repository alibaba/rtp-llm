"""Bitwise output masking and in-place metadata replay for Q8KV4 combine."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    LOG2E_F32,
    _q8kv4_combine,
    q8kv4_sparse_decode_attention,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CombineMaskTest(unittest.TestCase):
    def check_case(self, rows, mask_stride=1):
        torch.manual_seed(1234 + rows)
        partial = torch.randn(rows, 4, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(rows, 4, 16, 16, device="cuda") * 8
        counts = torch.randint(0, 17, (rows, 4), device="cuda", dtype=torch.int32)
        counts[0] = torch.tensor([0, 1, 7, 16], device="cuda", dtype=torch.int32)
        mask_storage = torch.ones(
            max(1, rows * mask_stride), device="cuda", dtype=torch.bool
        )
        valid = mask_storage.as_strided((rows,), (mask_stride,))
        old = torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
        new = torch.empty_like(old)

        def launch(out, masked):
            extra = (
                {
                    "valid_token_mask": valid,
                    "HAS_VALID_TOKEN_MASK": True,
                    "valid_token_mask_stride": valid.stride(0),
                }
                if masked
                else {}
            )
            _q8kv4_combine[(rows, 4)](
                partial,
                lse,
                counts,
                out,
                out.stride(0),
                out.stride(1),
                LOG2E=LOG2E_F32,
                num_warps=4,
                **extra,
            )

        def check():
            launch(old, False)
            expected = torch.where(valid[:, None, None], old, torch.zeros_like(old))
            self.assertTrue(
                torch.equal(expected.view(torch.int16), new.view(torch.int16))
            )
            self.assertEqual(
                torch.count_nonzero(new[~valid].view(torch.int16)).item(), 0
            )

        launch(new, True)
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch(new, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(new, True)
        pointers = (
            valid.data_ptr(),
            new.data_ptr(),
            partial.data_ptr(),
            counts.data_ptr(),
        )
        for live in (0, max(1, rows // 2), rows, max(0, rows - 1)):
            if mask_stride == 0:
                mask_storage.fill_(live != 0)
            else:
                valid.copy_(torch.arange(rows, device="cuda") < live)
            partial.normal_()
            partial[~valid] = float("nan")
            counts.random_(0, 17)
            lse.normal_()
            lse.mul_(64)
            graph.replay()
            check()
            self.assertEqual(
                pointers,
                (
                    valid.data_ptr(),
                    new.data_ptr(),
                    partial.data_ptr(),
                    counts.data_ptr(),
                ),
            )

    def test_eager_and_graph_mask_changes(self):
        for rows in (1, 3, 5, 16, 80, 96, 128):
            with self.subTest(rows=rows):
                self.check_case(rows)

    def test_reject_invalid_mask_metadata(self):
        q = torch.empty(2, 64, 128, device="cuda", dtype=torch.float8_e4m3fn)
        for mask in (
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, device="cuda", dtype=torch.float32),
            torch.ones(2, 1, device="cuda", dtype=torch.bool),
        ):
            with self.assertRaisesRegex(ValueError, "valid_token_mask"):
                q8kv4_sparse_decode_attention(
                    q,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    sm_scale=1,
                    out=None,
                    partial_out=None,
                    partial_lse=None,
                    counts=None,
                    valid_token_mask=mask,
                )

    def test_strided_masks(self):
        for rows in (5, 80):
            for stride in (0, 2):
                with self.subTest(rows=rows, stride=stride):
                    self.check_case(rows, stride)


if __name__ == "__main__":
    unittest.main()

"""GLM53 CUDA cache/topk addresses beyond the 32-bit element boundary."""

import unittest

import torch
from rtp_llm.models_py.modules.glm5_mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
    stage_shared_fp8_input_scales,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class Int64CacheOpsTest(unittest.TestCase):
    def test_shared_fp8_scale_large_source_stride(self):
        # The SE variants reuse the same packed int32 scale staging kernel.
        stride, tokens = (1 << 25) + 128, 65
        storage = torch.empty(
            (tokens - 1) * stride + 1, dtype=torch.int32, device="cuda"
        )
        source = storage.as_strided((tokens, 1), (stride, 1))
        values = (
            torch.arange(tokens, device="cuda", dtype=torch.int32).view(tokens, 1)
            + 1024
        )
        source.copy_(values)
        destination = torch.full((384, 1), -1, device="cuda", dtype=torch.int32)
        stage_shared_fp8_input_scales(source, destination, tokens, block_m=64)
        rows = shared_fp8_scale_row_indices(tokens, 64, source.device)
        expected = torch.zeros_like(destination)
        expected[rows] = values
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
        print("shared FP8 scale source offset:", (tokens - 1) * stride, "passed")

    def test_persistent_topk_large_row_offsets_and_stride(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertTrue(hasattr(rtp_llm_ops, "dsv4_persistent_topk"))
        # Exercise both persistent/filtered dispatch and >UINT32_MAX physical stride.
        for rows, stride in (
            (3, (1 << 31) + 128),
            (33, (1 << 27) + 128),
            (2, (1 << 32) + 128),
        ):
            with self.subTest(rows=rows, stride=stride):
                width, k = 1024, 512
                storage = torch.empty((rows - 1) * stride + width, device="cuda")
                logits = storage.as_strided((rows, width), (stride, 1))
                values = (
                    torch.stack(
                        [
                            torch.arange(width, device="cuda").roll(17 * r)
                            for r in range(rows)
                        ]
                    ).float()
                    / width
                )
                logits.copy_(values)
                lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
                out = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
                workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
                rtp_llm_ops.dsv4_persistent_topk(
                    logits, lengths, out, workspace, k, width
                )
                expected = values.topk(k, dim=-1).indices.sort(-1).values
                actual = out.long().sort(-1).values
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                print("topk rows/stride:", rows, stride, "passed")
                del storage, logits, values, out, workspace
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

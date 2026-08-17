from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe._nccl_ep_combine_triton import (
    mxfp8_dequant_peer_sum,
)


class Sm120NcclEpCombineTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dequantizes_and_sums_fixed_peer_blocks(self):
        world_size = 4
        n_rows = 3
        hidden_size = 64
        device = torch.device("cuda")

        values = (
            torch.arange(world_size * n_rows * hidden_size, device=device)
            .reshape(world_size, n_rows, hidden_size)
            .remainder(9)
            .sub(4)
            .to(torch.float8_e4m3fn)
        )
        encoded_scales = torch.tensor(
            [127, 128, 126, 129], dtype=torch.uint8, device=device
        )
        scales = torch.exp2(encoded_scales.float() - 127.0)
        scale_bytes = encoded_scales[:, None, None].expand(
            world_size, n_rows, hidden_size // 32
        )
        payload = torch.cat(
            [values.view(torch.uint8), scale_bytes], dim=-1
        ).reshape(world_size * n_rows, hidden_size + hidden_size // 32)

        actual = mxfp8_dequant_peer_sum(
            payload.contiguous(), n_rows, hidden_size, world_size
        )
        expected = (values.float() * scales[:, None, None]).sum(dim=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_wrong_payload_shape(self):
        payload = torch.empty((2, 4), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "unexpected payload shape"):
            mxfp8_dequant_peer_sum(payload, 1, 32, 2)


if __name__ == "__main__":
    unittest.main()

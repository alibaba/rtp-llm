"""Native group-32/128 quantization and V4.1 block-32 dense GEMM contracts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv41.utils import V41Embedding, V41MXFP8Linear


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MXFP8DenseTest(unittest.TestCase):
    def test_main_tp_embedding_preserves_decode_and_draft_token_shapes(self):
        weight = torch.arange(16 * 32, device="cuda", dtype=torch.bfloat16).reshape(
            16, 32
        )
        module = V41Embedding(
            None,
            SimpleNamespace(get_attn_tp_size=lambda: 4),
            weight[:, 8:16].contiguous(),
        )
        for shape in ((4,), (2, 3), (2, 1, 3)):
            ids = torch.arange(
                torch.tensor(shape).prod().item(), device="cuda"
            ).reshape(shape)
            flat = ids.flatten()
            shards = torch.cat(
                [
                    torch.nn.functional.embedding(flat, part)
                    for part in weight.chunk(4, dim=-1)
                ]
            )
            with patch(
                "rtp_llm.models_py.modules.base.common.embedding.all_gather",
                return_value=shards,
            ) as gather:
                actual = module(ids)
                torch.testing.assert_close(
                    gather.call_args.args[0],
                    torch.nn.functional.embedding(flat, weight[:, 8:16]),
                )
            torch.testing.assert_close(
                actual, torch.nn.functional.embedding(ids, weight), rtol=0, atol=0
            )

    def test_native_group_quant_payload_and_all_scale_groups(self):
        torch.manual_seed(104)
        for kernel in ("legacy", "v2"):
            for group in (32, 128):
                for rows in (3, 127):
                    with self.subTest(kernel=kernel, group=group, rows=rows):
                        x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
                        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": kernel}):
                            q, scales = sgl_per_token_group_quant_fp8(
                                x,
                                group_size=group,
                                eps=torch.finfo(torch.float32).tiny,
                                column_major_scales=True,
                                scale_tma_aligned=True,
                                scale_ue8m0=True,
                            )
                        self.assertEqual(scales.shape, (rows, (640 // group + 3) // 4))
                        expected_scale = torch.pow(
                            2.0,
                            torch.ceil(
                                torch.log2(
                                    x.float().reshape(rows, -1, group).abs().amax(-1)
                                    / 448
                                )
                            ),
                        )
                        packed = scales.contiguous().view(torch.uint8)[
                            :, : 640 // group
                        ]
                        actual_scale = torch.pow(2.0, packed.float() - 127)
                        torch.testing.assert_close(
                            actual_scale, expected_scale, rtol=0, atol=0
                        )
                        expected_q = (
                            x.float().reshape(rows, -1, group)
                            / expected_scale.unsqueeze(-1)
                        ).to(torch.float8_e4m3fn)
                        torch.testing.assert_close(
                            q.float(), expected_q.reshape_as(x).float(), rtol=0, atol=0
                        )

    def test_checkpoint_block32_dense_matches_quantized_reference(self):
        torch.manual_seed(105)
        x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 512, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(
            2.0, torch.randint(-3, 3, (8, 16), device="cuda").float()
        ).to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(w, scales)
        y = linear(x)
        x_blocks = x.float().reshape(7, -1, 32)
        x_scales = torch.pow(2.0, torch.ceil(torch.log2(x_blocks.abs().amax(-1) / 448)))
        quant_x = (x_blocks / x_scales.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        ).float() * x_scales.unsqueeze(-1)
        quant_w = w.float() * scales.float().repeat_interleave(32, 0).repeat_interleave(
            32, 1
        )
        expected = (quant_x.reshape_as(x) @ quant_w.T).to(torch.bfloat16)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y, expected, rtol=0.015, atol=0.02)


if __name__ == "__main__":
    unittest.main()

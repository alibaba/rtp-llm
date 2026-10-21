"""Check shared packing math and the two recipes' non-finite policies."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_fp8_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_input_packer import (
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_input_packer import (
    get_mega_moe_se_input_packer,
)


class WeightScaleTest(unittest.TestCase):
    def test_fp4_raw_and_float_scales_use_same_layout(self):
        scales = torch.ones((2, 128, 8), dtype=torch.float32)
        with patch("deep_gemm.transform_sf_into_required_layout") as transform:
            prepare_fp4_weight_scale_for_deepgemm(scales, 128, 256, 2)
            actual, *layout = transform.call_args.args
            torch.testing.assert_close(actual, scales)
            self.assertEqual(layout, [128, 256, (1, 32), 2])
            prepare_fp4_weight_scale_for_deepgemm(
                scales.to(torch.float8_e8m0fnu), 128, 256, 2
            )
            torch.testing.assert_close(transform.call_args.args[0], scales)

    def test_fp8_shape_validation_and_packed_passthrough(self):
        packed = torch.zeros((128, 4), dtype=torch.int32)
        for prepare in (
            prepare_fp4_weight_scale_for_deepgemm,
            prepare_fp8_weight_scale_for_deepgemm,
        ):
            self.assertIs(prepare(packed, 128, 256), packed)
        with self.assertRaises(ValueError):
            prepare_fp8_weight_scale_for_deepgemm(torch.ones(128, 8), 128, 256)
        with patch("deep_gemm.transform_sf_into_required_layout") as transform:
            prepare_fp8_weight_scale_for_deepgemm(torch.ones(1, 2), 128, 256)
            self.assertEqual(transform.call_args.args[1:], (128, 256, (128, 128)))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class InputPackPolicyTest(unittest.TestCase):
    def buffer(self, tokens, dim=256, topk=8):
        # Non-contiguous row strides also occur in symmetric-memory views.
        return SimpleNamespace(
            x=torch.empty(tokens, dim * 2, dtype=torch.float8_e4m3fn, device="cuda")[
                :, :dim
            ],
            x_sf=torch.empty(tokens, dim // 64, dtype=torch.int32, device="cuda")[
                :, : dim // 128
            ],
            topk_idx=torch.empty(tokens, topk, dtype=torch.int64, device="cuda"),
            topk_weights=torch.empty(tokens, topk, dtype=torch.float32, device="cuda"),
            shared_l1_acts_sf=torch.full(
                (4096, dim // 128), -1, dtype=torch.int32, device="cuda"
            ),
        )

    def test_finite_inputs_match_for_decode_prefill_and_empty_rank(self):
        for tokens in (0, 1, 17, 2049):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 512, dtype=torch.bfloat16, device="cuda")[
                    :, :256
                ]
                weights = torch.rand(tokens, 8, device="cuda")
                ids = torch.randint(0, 256, (tokens, 8), device="cuda")
                ordinary, prepared = self.buffer(tokens), self.buffer(tokens)
                FusedMegaMoEInputPacker().pack(x, weights, ids, ordinary, tokens)
                get_mega_moe_input_packer().pack(
                    x, weights, ids.int(), prepared, tokens
                )
                for name in ("x", "x_sf", "topk_idx", "topk_weights"):
                    a, b = getattr(ordinary, name), getattr(prepared, name)
                    if name == "x":
                        a, b = a.view(torch.uint8), b.view(torch.uint8)
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(prepared.topk_idx, ids)
                torch.testing.assert_close(prepared.topk_weights, weights)

    def test_non_finite_policy_is_preserved(self):
        x = torch.ones(1, 256, dtype=torch.bfloat16, device="cuda")
        x[0, :3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")], device="cuda"
        )
        weights = torch.ones(1, 8, device="cuda")
        ids = torch.zeros(1, 8, dtype=torch.int64, device="cuda")
        ordinary, prepared, reference = self.buffer(1), self.buffer(1), self.buffer(1)
        FusedMegaMoEInputPacker().pack(x, weights, ids, ordinary, 1)
        FusedMegaMoEInputPacker().pack(
            torch.nan_to_num(x, nan=0, posinf=0, neginf=0), weights, ids, reference, 1
        )
        get_mega_moe_input_packer().pack(x, weights, ids, prepared, 1)
        torch.testing.assert_close(
            ordinary.x.view(torch.uint8), reference.x.view(torch.uint8)
        )
        torch.testing.assert_close(
            prepared.x.float()[0, :3],
            torch.tensor([448.0, 448.0, -448.0], device="cuda"),
        )

    def test_shared_scale_padding_after_batch_shrink_and_graph_replay(self):
        packer = get_mega_moe_se_input_packer()
        buf = self.buffer(193)
        for tokens, block_m in ((193, 192), (17, 16), (0, 16)):
            x = torch.ones(tokens, 256, dtype=torch.bfloat16, device="cuda")
            weights = torch.ones(tokens, 8, device="cuda")
            ids = torch.zeros(tokens, 8, dtype=torch.int64, device="cuda")
            # Prewarm the specialization before graph capture.
            packer.pack(x, weights, ids, buf, tokens, block_m)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packer.pack(x, weights, ids, buf, tokens, block_m)
            buf.shared_l1_acts_sf.fill_(-1)
            graph.replay()
            active = (
                ((tokens + block_m - 1) // block_m) * ((block_m + 127) // 128) * 128
            )
            if tokens == 0:
                self.assertEqual(torch.count_nonzero(buf.shared_l1_acts_sf).item(), 0)
            else:
                expected = torch.zeros_like(buf.shared_l1_acts_sf[:active])
                rows = shared_fp8_scale_row_indices(tokens, block_m, x.device)
                expected.index_copy_(0, rows, buf.x_sf[:tokens])
                torch.testing.assert_close(buf.shared_l1_acts_sf[:active], expected)


if __name__ == "__main__":
    unittest.main()

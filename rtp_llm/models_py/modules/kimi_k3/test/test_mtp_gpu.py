"""GPU regression checks for K3 MTP integration."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpLayer
from rtp_llm.models_py.triton_kernels.moe.output_add import add_moe_output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KimiK3MtpGpuTest(unittest.TestCase):
    def test_mtp_reuses_moe_residual_fusion_in_both_phases(self):
        # Real MTP forward and real fused add; expensive attention/expert GEMMs
        # are replaced at their boundaries with exact BF16 fixture values.
        class Attention:
            def tp_input_projection_weights(self):
                return []

            def output_projection_weight(self):
                return None

            def __call__(self, x, *args, **kwargs):
                return torch.full_like(x, 0.125)

        def moe(x, *, residual=None, **kwargs):
            return add_moe_output(
                torch.full_like(x, 1),
                torch.full_like(x, 0.25),
                residual,
            )

        for prefill in (False, True):
            with self.subTest(prefill=prefill):
                layer = SimpleNamespace(
                    enorm=lambda x: x,
                    hnorm=lambda x: x,
                    eh_proj=lambda x: x[:, :128].contiguous(),
                    input_norm=lambda x: x,
                    attention=Attention(),
                    attn_tp_size=1,
                    _local_projection=lambda x, w: x,
                    post_norm=lambda x: x,
                    moe=moe,
                )
                x = torch.ones(3, 128, device="cuda", dtype=torch.bfloat16)
                positions = torch.tensor([0, 1, 2], device="cuda")
                layout = SimpleNamespace(
                    tokens=SimpleNamespace(
                        local_valid_tokens=2, local_tokens=3, physical_tokens=3
                    )
                )
                actual = KimiK3MtpLayer.forward(
                    layer,
                    x,
                    x,
                    positions,
                    SimpleNamespace(release_forward_workspace=lambda: None),
                    None,
                    SimpleNamespace(is_prefill=prefill),
                    sp_layout=layout,
                )
                expected = torch.full_like(x, 2.375)
                expected[0].fill_(1.375)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

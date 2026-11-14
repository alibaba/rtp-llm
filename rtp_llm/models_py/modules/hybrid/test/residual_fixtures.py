import torch

from rtp_llm.models_py.modules.hybrid.gated_residual import GatedResidual


class ResidualFp8Fixture:
    def module(self, inject=True):
        def weight(shape):
            return (torch.randn(shape, device="cuda") * 0.1).to(torch.float8_e4m3fn)

        return GatedResidual(
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07,
            weight((320, 10240)),
            weight((10240, 320)),
            weight((4, 10240)) if inject else None,
            hc_mult=4,
            norm_eps=1.0e-6,
            weight_scales=[torch.tensor([0.03125], device="cuda")] * 3,
            role="PREFILL",
        )

    def assert_outputs(self, actual, expected):
        for value, reference in zip(actual, expected):
            if reference is None:
                assert value is None
            else:
                torch.testing.assert_close(value, reference, rtol=0, atol=0)

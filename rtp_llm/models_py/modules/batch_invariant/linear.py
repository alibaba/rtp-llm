import torch

from rtp_llm.models_py.modules.factory.linear.linear_base import LinearBase


class BatchInvariantLinear(LinearBase):
    @classmethod
    def can_handle(
        cls,
        quant_config,
        weight,
        weight_scales,
        hw_kernel_config=None,
        weight_scale_2=None,
        input_scale=None,
    ):
        return (
            weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight_scales is None
            and weight_scale_2 is None
            and input_scale is None
            and (quant_config is None or not quant_config.is_quanted())
        )

    def __init__(self, weight, bias=None, **kwargs):
        super().__init__(weight, bias=bias)
        self.weight = weight.T
        self.bias = bias

    def forward(self, input):
        from .kernels import linear

        return linear(input, self.weight, self.bias)

"""TP shared expert consuming the loader-owned local FP8/FP32 tensors."""

from torch import nn

from ...modules.linear.fp8_linear import PpuFp8Linear
from .ppu_shared_expert import PpuSharedExpert


class PpuTPSharedExpert(PpuSharedExpert):
    def __init__(
        self,
        dim,
        inter_dim,
        expert_weights,
        *,
        tp_size,
        tp_rank,
        swiglu_limit=0.0,
        platform_provider,
    ):
        nn.Module.__init__(self)
        if tp_size != 4 or not 0 <= tp_rank < tp_size:
            raise ValueError("PPU shared expert requires TP4 and rank in [0, 4)")
        if inter_dim <= 0 or inter_dim % (128 * tp_size) or dim <= 0 or dim % 128:
            raise ValueError("Shared expert dimensions must preserve block-128 scales")
        local = inter_dim // tp_size
        for name, shape in (("w13", (2 * local, dim)), ("w2", (dim, local))):
            weight, scale = expert_weights[name + "_w"], expert_weights[name + "_s"]
            if tuple(weight.shape) != shape:
                raise ValueError(
                    f"Shared {name} requires loader-prepared shape {shape}"
                )
            setattr(
                self,
                name,
                PpuFp8Linear(
                    weight,
                    scale,
                    scale_is_prepared=True,
                    share_input_quantization=platform_provider._bool(
                        "DSV4_PPU_SHARED_QKV_QUANT", False
                    ),
                ),
            )
        self.swiglu_limit = swiglu_limit
        self.preserve_output_dtype = True
        self.tp_size, self.tp_rank = tp_size, tp_rank

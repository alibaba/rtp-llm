"""Qwen4 shared-expert fusion, isolated from generic MLP forward."""

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType


class Qwen4DenseMLP(DenseMLP):
    def forward(self, x: torch.Tensor, skip_allreduce: bool = False) -> torch.Tensor:
        fused_down = None
        if not self.is_gated and self.activation_type == ActivationType.Gelu:
            activated = self.up_proj.forward_with_bias_gelu(x)
        else:
            up = self.up_proj(x)
            if self.is_gated and type(self.act_fn) is FusedSiluAndMul:
                from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear

                if (
                    type(self.down_proj) is Fp8Linear
                    and "forward" not in self.down_proj.__dict__
                    and "forward" not in self.act_fn.__dict__
                    and not self.down_proj._forward_pre_hooks
                    and not self.down_proj._forward_hooks
                    and not self.act_fn._forward_pre_hooks
                    and not self.act_fn._forward_hooks
                    and not torch.nn.modules.module._global_forward_pre_hooks
                    and not torch.nn.modules.module._global_forward_hooks
                ):
                    fused_down = self.down_proj.forward_silu_and_mul(up)
            if fused_down is None:
                activated = self.act_fn(up)

        ffn_tp_size = self.parallelism_config.get_ffn_tp_size()
        output = fused_down if fused_down is not None else self.down_proj(activated)
        if not skip_allreduce and ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

import aiter
import torch
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
class FusedSiluActDenseMLP(FusedSiluActDenseMLP):
    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        aiter.silu_and_mul(output, gate_up)
        down_proj = self.down_proj(output)
        if self.config.tp_size > 1:
            down_proj = all_reduce(down_proj, group=Group.TP)
        return down_proj

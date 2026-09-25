"""Native BF16 RMSNorm for K3's latent expert branch."""

import torch

from rtp_llm.models_py.modules.base.common.norm import BaseNorm


class KimiK3LatentRMSNorm(BaseNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda:
            x = hidden_states.float()
            inv_rms = torch.rsqrt(
                x.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
            )
            return (x * inv_rms * self.weight.float()).to(hidden_states.dtype)
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        op = getattr(rtp_llm_ops, "kimi_k3_rms_norm", None)
        if op is None:
            raise RuntimeError("K3 latent RMSNorm requires the CUDA13 native binding")
        return op(hidden_states, self.weight, self.variance_epsilon)

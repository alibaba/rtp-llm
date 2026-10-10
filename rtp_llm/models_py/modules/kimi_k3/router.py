"""K3 MoE gate projection and grouped expert selection."""

import torch
from torch import nn


class KimiK3Router(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        top_k: int,
        groups: int,
        top_groups: int,
        renormalize: bool,
        scale: float,
    ):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("K3 router weight must have shape [hidden, experts]")
        fp32 = weight.float()
        self._bf16_exact = weight.dtype == torch.bfloat16 or torch.equal(
            fp32, fp32.bfloat16().float()
        )
        # Preserve higher-precision values present in the loaded weight.
        dtype = torch.bfloat16 if self._bf16_exact else torch.float32
        # Prepare the cuBLAS [experts, hidden] view at load, never in forward.
        prepared = weight.to(dtype=dtype).t().contiguous().t()
        self.register_buffer("weight", prepared, persistent=False)
        self.correction = bias.float()
        self.top_k = top_k
        self.groups = groups
        self.top_groups = top_groups
        self.renormalize = renormalize
        self.scale = scale
        self.group_topk = None
        if prepared.is_cuda:
            from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK

            self.group_topk = GroupTopK()

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.is_cuda and hidden.dtype == torch.bfloat16 and self._bf16_exact:
            # Write FP32 logits directly, without an intermediate BF16 output.
            return torch.mm(hidden, self.weight, out_dtype=torch.float32)
        return hidden.float() @ self.weight.float()

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._project_logits(hidden)
        bias = self.correction
        top_k, groups, top_groups = self.top_k, self.groups, self.top_groups
        renormalize, scale = self.renormalize, self.scale
        group_topk = self.group_topk
        if logits.is_cuda:
            from rtp_llm.models_py.modules.base.cuda.select_topk import GroupTopK

            if logits.dtype != torch.float32 or bias.dtype != torch.float32:
                raise RuntimeError("K3 routing requires FP32 logits and correction bias")
            if GroupTopK.fused_sigmoid_supported(
                logits, bias, groups, top_groups, top_k
            ):
                if group_topk is None:
                    group_topk = GroupTopK()
                shape = (logits.shape[0], top_k)
                routing = torch.empty(shape, dtype=torch.float32, device=logits.device)
                ids = torch.empty(shape, dtype=torch.int32, device=logits.device)
                group_topk.forward_fused_sigmoid(
                    routing, ids, logits, bias, top_k, renormalize, scale
                )
                return routing, ids

        raise ValueError(
            "K3 fused routing requires nonempty contiguous CUDA FP32 logits "
            "and correction bias, 896 experts, top_k=16 and groups=top_groups=1"
        )

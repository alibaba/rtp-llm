import torch
from torch import nn


def exact_head_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm over the last dim, matching upstream bit-for-bit.

    Upstream ``Qwen3_5RMSNorm.forward`` is ``_norm(x.float()) * (1.0 + w.float())``:
    the ``+1`` happens in fp32, after the cast. ``gamma`` is therefore the raw
    checkpoint tensor and is broadcast across heads.
    """
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + gamma.float())).type_as(x)


class Qwen4ExpFusedQKRMSNorm(nn.Module):
    """Exact-parity replacement for ``FusedQKRMSNorm`` on the fused qkv tensor.

    Interface and in-place contract match the fused version so it can be dropped
    straight onto ``CausalAttention.qk_fuse_norm``, but the fused path cannot be
    used here: it hands a bf16 gamma to ``flashinfer.norm.rmsnorm``, and upstream's
    ``1 + gamma`` is not representable in bf16 -- the spacing near 1.0 is 2**-8, so
    folding the one in costs up to ~4e-3 relative on the gain. Measured on the real
    checkpoint's ``layers.11.self_attn.k_norm``: 195 of 256 elements shift.

    Consequence: q/k norm runs in torch rather than one fused kernel, on the 12
    full-attention layers. Correctness over speed for now; a fused kernel that
    takes the one into account is left to the performance milestone.
    """

    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.eps = eps
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = head_num * size_per_head
        self.kv_size = kv_head_num * size_per_head

    @classmethod
    def replacing(cls, fused: nn.Module) -> "Qwen4ExpFusedQKRMSNorm":
        """Build from an already-constructed ``FusedQKRMSNorm``.

        Both the CUDA and ROCm variants expose the same fields, so this stays
        device-agnostic and cannot drift from their constructor signature.
        """
        return cls(
            fused.q_weight,
            fused.k_weight,
            head_num=fused.head_num,
            kv_head_num=fused.kv_head_num,
            size_per_head=fused.size_per_head,
            eps=fused.eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() == 2
        rows, width = hidden_states.shape
        qkv = hidden_states.reshape(
            rows, self.head_num + self.kv_head_num * 2, self.size_per_head
        )
        q = qkv[:, : self.head_num, :]
        k = qkv[:, self.head_num : self.head_num + self.kv_head_num, :]
        # Materialized before copy_, so reading q/k while writing them is safe.
        q.copy_(exact_head_rms_norm(q, self.q_weight, self.eps))
        k.copy_(exact_head_rms_norm(k, self.k_weight, self.eps))
        return qkv.reshape(rows, width)

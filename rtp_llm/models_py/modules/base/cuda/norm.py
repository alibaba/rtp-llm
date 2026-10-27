from typing import Optional, Tuple

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseNorm,
    BaseResNorm,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


# zero_centered=True aligns these CUDA norms with vLLM's Qwen RMSNorm:
# consume raw gamma and add 1 in FP32 inside the kernel to avoid an
# intermediate BF16 rounding of gamma + 1.
class RMSNorm(BaseNorm):
    def __init__(
        self,
        weight: torch.Tensor,
        eps: float = 1e-6,
        *,
        zero_centered: bool = False,
        batch_invariant_after_add: bool = False,
    ):
        super().__init__(weight, eps)
        self.zero_centered = zero_centered
        from rtp_llm.models_py.modules.batch_invariant import construction_policy

        self.batch_invariant_policy = construction_policy()
        self.batch_invariant_after_add = batch_invariant_after_add
        if self.batch_invariant_policy.enabled and zero_centered:
            raise ValueError(
                "batch invariant dense-gqa-v1 does not support zero-centered RMSNorm"
            )

    def forward(
        self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.batch_invariant_policy.enabled:
            from rtp_llm.models_py.modules.batch_invariant.kernels import rmsnorm

            if self.batch_invariant_after_add:
                shape = hidden_states.shape
                result = rtp_llm_ops.batch_invariant_post_residual_rmsnorm(
                    hidden_states.reshape(-1, shape[-1]).contiguous(),
                    self.weight.contiguous(),
                    self.variance_epsilon,
                ).view(shape)
            else:
                result = rmsnorm(hidden_states, self.weight, self.variance_epsilon)
            if output is not None:
                output.copy_(result)
                return output
            return result
        if output is None:
            output = torch.empty_like(hidden_states)
        if self.zero_centered:
            if hidden_states.numel() == 0:
                return output
            if hidden_states.dim() != 2 or hidden_states.shape[-1] != 256:
                # Keep FlashInfer reduction/rsqrt ordering; results may differ
                # from vLLM while raw gamma + 1 is still computed in FP32.
                return flashinfer.norm.gemma_rmsnorm(
                    hidden_states,
                    self.weight,
                    eps=self.variance_epsilon,
                    out=output,
                )
            from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fused_qk_rmsnorm import (
                vllm_style_zero_centered_rmsnorm,
            )

            return vllm_style_zero_centered_rmsnorm(
                hidden_states,
                self.weight,
                eps=self.variance_epsilon,
                output=output,
            )
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.rmsnorm(
            output,
            hidden_states,
            self.weight.data,
            self.variance_epsilon,
            stream_id,
        )
        return output


class RMSResNorm(BaseResNorm):
    def __init__(
        self, weight: torch.Tensor, eps: float = 1e-6, *, zero_centered: bool = False
    ):
        super().__init__(weight, eps)
        self.zero_centered = zero_centered

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.zero_centered:
            if hidden_states.numel() == 0:
                return hidden_states, residual
            # Keep FlashInfer reduction/rsqrt ordering; results may differ
            # from vLLM while raw gamma + 1 is still computed in FP32.
            flashinfer.norm.gemma_fused_add_rmsnorm(
                hidden_states,
                residual,
                self.weight,
                eps=self.variance_epsilon,
            )
            return hidden_states, residual
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.fused_add_rmsnorm(
            hidden_states,
            residual,
            self.weight.data,
            self.variance_epsilon,
            stream_id,
        )
        return hidden_states, residual


class QKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
        *,
        zero_centered: bool = False,
    ):
        super().__init__()
        self.q_norm = RMSNorm(q_weight, eps, zero_centered=zero_centered)
        self.k_norm = RMSNorm(k_weight, eps, zero_centered=zero_centered)
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head
        self.variance_epsilon = eps

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.size_per_head)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.size_per_head)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(self, hidden_states):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        output = torch.cat([q, k, v], dim=-1)
        return output


class FusedQKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
        enable_pdl: bool = False,
        *,
        zero_centered: bool = False,
    ):
        super().__init__()
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.eps = eps
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head
        self.enable_pdl = enable_pdl
        self.zero_centered = zero_centered
        from rtp_llm.models_py.modules.batch_invariant import construction_policy

        self.batch_invariant_policy = construction_policy()
        if self.batch_invariant_policy.enabled and zero_centered:
            raise ValueError(
                "batch invariant dense-gqa-v1 does not support zero-centered QK RMSNorm"
            )

    def forward(self, hidden_states: torch.Tensor):
        if self.batch_invariant_policy.enabled:
            from rtp_llm.models_py.modules.batch_invariant.kernels import rmsnorm

            qkv = hidden_states.view(
                -1, self.head_num + 2 * self.kv_head_num, self.size_per_head
            )
            q, k = (
                qkv[:, : self.head_num],
                qkv[:, self.head_num : self.head_num + self.kv_head_num],
            )
            q.copy_(rmsnorm(q, self.q_weight, self.eps))
            k.copy_(rmsnorm(k, self.k_weight, self.eps))
            return hidden_states
        if self.zero_centered and hidden_states.numel() == 0:
            return hidden_states
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fused_qk_rmsnorm import (
            maybe_fused_qk_rmsnorm,
        )

        fused = maybe_fused_qk_rmsnorm(
            hidden_states,
            self.q_weight,
            self.k_weight,
            self.head_num,
            self.kv_head_num,
            self.size_per_head,
            self.eps,
            zero_centered=self.zero_centered,
        )
        if fused is not None:
            return fused
        assert hidden_states.dim() == 2
        m, n = hidden_states.shape
        qkv = hidden_states.reshape(
            m, (self.head_num + self.kv_head_num * 2), self.size_per_head
        )
        q = qkv[:, : self.head_num, :]
        k = qkv[:, self.head_num : self.head_num + self.kv_head_num, :]
        if self.zero_centered:
            # CUDA Gemma accepts 2D only. Packed Q/K views are not contiguous,
            # so flatten each separately and write back; V is never modified.
            for values, weight in ((q, self.q_weight), (k, self.k_weight)):
                flat = values.reshape(-1, self.size_per_head)
                if self.size_per_head != 256:
                    normed = flashinfer.norm.gemma_rmsnorm(
                        flat,
                        weight,
                        eps=self.eps,
                        enable_pdl=self.enable_pdl,
                    )
                    values.copy_(normed.reshape(values.shape))
                    continue
                from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fused_qk_rmsnorm import (
                    vllm_style_zero_centered_rmsnorm,
                )

                normed = vllm_style_zero_centered_rmsnorm(
                    flat,
                    weight,
                    eps=self.eps,
                )
                values.copy_(normed.reshape(values.shape))
            return qkv.reshape(m, n)
        flashinfer.norm.rmsnorm(
            q,
            self.q_weight,
            eps=self.eps,
            out=q,
            enable_pdl=self.enable_pdl,
        )
        flashinfer.norm.rmsnorm(
            k,
            self.k_weight,
            eps=self.eps,
            out=k,
            enable_pdl=self.enable_pdl,
        )
        return qkv.reshape(m, n)


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        rtp_llm_ops.fused_add_layernorm(
            hidden_states,
            residual,
            bias,
            self.weight.data,
            self.beta,
            self.variance_epsilon,
        )
        return hidden_states

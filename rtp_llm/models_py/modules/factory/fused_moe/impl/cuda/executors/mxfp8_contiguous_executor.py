"""MXFP8 contiguous grouped MoE executor for MiniMax-M3 (optimized).

Optimized version that uses fused Triton kernels from DeepEP executor:
- ep_expand: replaces ~24 PyTorch kernel launches for token permutation
- ep_fill_m_indices: fused m_indices fill
- ep_gather: fused weighted gather with router weights

This significantly reduces kernel launch overhead compared to the original
Python loop + PyTorch ops approach.
"""

from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
    mxfp8_grouped_gemm,
    pack_mxfp8_scale,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.common.swiglu_oai import (
    is_swiglu_oai,
    swiglu_oai_alpha_limit,
    swiglu_oai_torch,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_expand,
    ep_fill_m_indices,
    ep_gather,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.utils.model_weight import W


def _contiguous_alignment() -> int:
    import deep_gemm

    return deep_gemm.get_m_alignment_for_contiguous_layout()


class Mxfp8ContiguousExecutor(FusedMoeExpertExecutor):
    """Optimized MXFP8 contiguous executor using fused Triton kernels.

    Unlike the original version that uses Python loops + PyTorch ops for token
    permutation, this version uses:

    1. **ep_expand**: fused triton kernel for token expansion into per-expert
       contiguous layout (replaces ~24 PyTorch kernel launches)
    2. **ep_fill_m_indices**: fused m_indices fill for each expert's padded block
    3. **ep_gather**: fused triton kernel for weighted gather with router weights

    This matches the DeepEP executor's approach but adapted for pure-TP mode
    where we have all tokens on all ranks (no DeepEP dispatch).
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 10)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.w1 = weights[W.moe_w1]  # [E, 2*inter, hidden] (up|gate) e4m3
        self.w2 = weights[W.moe_w2]  # [E, hidden, inter] (down) e4m3
        self.w1_scale = weights[W.moe_s1]  # fp32 (1,32) power-of-two scale
        self.w2_scale = weights[W.moe_s2]
        self.E = self.w1.size(0)
        self.K = self.w1.size(2)  # hidden_size
        self.N = self.w1.size(1)  # 2 * moe_inter (gate+up)
        self.top_k = config.moe_k
        self._w1_sp = None
        self._w2_sp = None
        self._align = _contiguous_alignment()

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def _packed_scales(self):
        # Pack the (1,32) fp32 expert scales into DeepGEMM int32 layout on first
        # forward (deferred from load; see Mxfp8Weight._postprocess) and cache.
        if self._w1_sp is None:
            e, ngu, k1 = self.w1.shape
            self._w1_sp = pack_mxfp8_scale(self.w1_scale, mn=ngu, k=k1, num_groups=e)
            e2, hid, k2 = self.w2.shape
            self._w2_sp = pack_mxfp8_scale(self.w2_scale, mn=hid, k=k2, num_groups=e2)
        return self._w1_sp, self._w2_sp

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden = payload.expert_x  # [M, K] BF16, all tokens on this rank (pure TP)
        topk_ids = payload.expert_topk_ids  # [M, top_k], global expert IDs
        topk_weights = payload.expert_topk_weights  # [M, top_k]
        assert topk_ids is not None and topk_weights is not None

        M = hidden.size(0)
        device = hidden.device
        top_k = topk_ids.size(1)

        # Early exit: no tokens
        if M == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    0, self.K, device=device, dtype=torch.bfloat16
                )
            )

        # --- Compute per-expert padded counts ---
        # For pure TP, we have all tokens on this rank, so we need to compute
        # the padded counts ourselves (DeepEP dispatch provides this)
        flat_ids = topk_ids.reshape(-1).to(torch.long)

        # Filter out -1 (non-local experts, shouldn't exist in pure TP but be safe)
        valid = flat_ids >= 0
        if not bool(valid.all()):
            flat_ids = flat_ids[valid]
            if flat_ids.numel() == 0:
                return CombineForwardPayload(
                    fused_expert_output=torch.zeros(
                        M, self.K, device=device, dtype=torch.bfloat16
                    )
                )

        counts = torch.bincount(flat_ids, minlength=self.E)
        padded = ((counts + self._align - 1) // self._align * self._align).to(
            torch.long
        )
        padded_counts = padded.tolist()

        # Compute all_tokens (expanded size)
        all_tokens = sum(padded_counts)
        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    M, self.K, device=device, dtype=torch.bfloat16
                )
            )

        # --- Expand tokens into per-expert contiguous layout ---
        # Build padded offsets on CPU to avoid ~5 GPU kernel launches
        # (zeros + slice + cast + cumsum + copy) for a tiny E-element vector.
        cpu_cumsum = [0] * (self.E + 1)
        for i in range(self.E):
            cpu_cumsum[i + 1] = cpu_cumsum[i] + padded_counts[i]
        assert cpu_cumsum[self.E] == all_tokens

        padded_counts_t = torch.tensor(padded_counts, dtype=torch.int32, device=device)
        padded_offsets = torch.tensor(
            cpu_cumsum[: self.E], dtype=torch.long, device=device
        )

        expanded_hidden = torch.empty(
            all_tokens, self.K, device=device, dtype=torch.bfloat16
        )
        output_index = torch.full((M, top_k), -1, device=device, dtype=torch.int32)
        m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)

        # ep_fill_m_indices only reads padded_offsets, while ep_expand mutates
        # them via atomic_add. Running fill first on the same CUDA stream
        ep_fill_m_indices(padded_counts_t, padded_offsets, m_indices)
        ep_expand(
            hidden,
            topk_ids,
            padded_offsets,
            expanded_hidden,
            output_index,
            self.E,
        )

        # --- Activation function parameters ---
        if is_swiglu_oai(activation):
            alpha, limit = swiglu_oai_alpha_limit(extra_expert_args)
        else:
            alpha, limit = None, None

        # --- Gate+Up grouped MXFP8 GEMM ---
        w1_sp, w2_sp = self._packed_scales()
        upgate = mxfp8_grouped_gemm(expanded_hidden, self.w1, w1_sp, m_indices)
        del expanded_hidden
        # upgate: [all_tokens, 2*inter] BF16

        # --- SwiGLU activation ---
        inter = self.N // 2
        if alpha is not None and limit is not None:
            act = swiglu_oai_torch(upgate, alpha, limit, gate_first=False)
        else:
            act = torch.empty(all_tokens, inter, device=device, dtype=torch.bfloat16)
            silu_and_mul(act, upgate)
        del upgate

        # --- Down grouped MXFP8 GEMM ---
        down = mxfp8_grouped_gemm(act.contiguous(), self.w2, w2_sp, m_indices)
        del act
        # down: [all_tokens, K] BF16

        # --- Apply router weights + gather back to original token order ---
        # Using fused ep_gather triton kernel instead of multi-launch PyTorch path
        gather_out = torch.empty(M, self.K, device=device, dtype=torch.bfloat16)
        ep_gather(down, topk_ids, topk_weights, output_index, gather_out)

        return CombineForwardPayload(fused_expert_output=gather_out)

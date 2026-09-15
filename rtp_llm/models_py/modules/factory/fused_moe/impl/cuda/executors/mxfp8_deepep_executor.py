"""MXFP8 DeepEP normal executor for MiniMax-M3.

Optimized executor for the DeepEP normal dispatch path with MXFP8 (1x32)
quantization. Unlike :class:`Mxfp8ContiguousExecutor` (designed for pure-TP),
this executor is tailored for the DeepEP normal dispatch output format:

- DeepEP non-expand dispatch produces ``[N_recv, K]`` of **unique** received
  tokens (no per-expert duplication) with ``[N_recv, top_k]`` routing metadata
- The executor expands tokens into per-expert contiguous layout (duplicating
  tokens with multiple local expert assignments), runs MXFP8 grouped GEMMs,
  then gathers results back with router weight application via ``ep_gather``

DeepEP normal dispatch output (non-expand mode)::

    expert_x:        [N_recv, K] BF16, unique received tokens (no duplicates)
    expert_topk_ids: [N_recv, top_k] local expert IDs in [0, E_local) + -1
    expert_num_tokens: [E_local] padded to expert_alignment (128)

Executor output (fed to normal combine)::

    fused_expert_output: [N_recv, K] BF16
        = Σ_{k assigned to local experts} expert_ffn(token, expert_k) * weight_k

Normal combine then sums partial results across ranks (no weight application).
"""

import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
    mxfp8_grouped_gemm,
    mxfp8_grouped_gemm_masked,
    mxfp8_grouped_gemm_masked_prequantized,
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


class Mxfp8DeepepExecutor(FusedMoeExpertExecutor):
    """MXFP8 executor optimized for DeepEP normal dispatch.

    DeepEP non-expand dispatch outputs ``[N_recv, K]`` unique tokens with
    ``[N_recv, top_k]`` routing metadata. This executor:

    1. **Expands** unique tokens into per-expert contiguous layout
       (``[all_tokens, K]``, ``all_tokens = Σ padded_counts``), duplicating
       tokens that have multiple local expert assignments.
    2. Builds ``m_indices`` and ``output_index`` (mapping each (token, slot)
       pair to its row in the expanded tensor).
    3. Runs MXFP8 grouped GEMMs on the expanded tensor.
    4. **Gathers** results back using ``ep_gather`` triton kernel, which
       applies router weights and accumulates into ``[N_recv, K]`` output.

    Compared to :class:`DeepGemmHybridExecutor`:
    - No ``ep_scatter`` needed (MXFP8 dispatches BF16, no FP8 scale reformatting)
    - No FP8 activation quant between SiLU and down GEMM (MXFP8 quantizes
      inside ``mxfp8_grouped_gemm`` with recipe=(1,32))
    - Expand uses ``ep_expand`` triton kernel (atomic_add scatter, 1 launch)
      instead of sort-based PyTorch expand (~24 launches)
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
        # Weights: [E_local, ...] e4m3 with prepacked int32 UE8M0 [1, 32] scales
        self.w1 = weights[W.moe_w1]  # [E, 2*inter, hidden] (up|gate)
        self.w2 = weights[W.moe_w2]  # [E, hidden, inter] (down)
        self.w1_scale = weights[W.moe_s1]  # fp32 (1, 32) power-of-two scale
        self.w2_scale = weights[W.moe_s2]
        self.E = self.w1.size(0)  # E_local
        self.K = self.w1.size(2)  # hidden_size
        self.N = self.w1.size(1)  # 2 * moe_inter (gate+up concatenated)
        self._w1_sp = None
        self._w2_sp = None
        self._align = _contiguous_alignment()
        if config.enable_cuda_graph:
            self._packed_scales()

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        # DeepEP normal dispatch consumes global expert ids as int64. Asking
        # SelectTopk to produce that dtype avoids a per-layer int32->int64 copy
        # in the router; DeepEP still returns local ids for this executor.
        return torch.int64

    def _packed_scales(self):
        """Pack (1, 32) fp32 weight scales into DeepGEMM int32 layout (cached)."""
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
        hidden = payload.expert_x  # [N_recv, K] BF16 unique recv tokens
        topk_ids = payload.expert_topk_ids  # [N_recv, top_k], local IDs + -1
        topk_weights = payload.expert_topk_weights  # [N_recv, top_k]
        assert topk_ids is not None and topk_weights is not None
        assert (
            payload.expert_tokens_meta is not None
        ), "DeepEP dispatch must provide expert_tokens_meta"

        # --- Per-expert padded counts from DeepEP dispatch ---
        # DeepEP dispatch with expert_alignment=self._align guarantees each
        # expert's slot count is padded to a multiple of self._align.
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            padded_counts: List[int] = list(
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
        elif payload.expert_tokens_meta.expert_num_tokens is not None:
            padded_counts = payload.expert_tokens_meta.expert_num_tokens.tolist()
        else:
            raise ValueError(
                "expert_tokens_meta must provide expert_num_tokens or "
                "expert_num_tokens_cpu (from DeepEP dispatch)"
            )

        N_recv = hidden.size(0)
        device = hidden.device
        top_k = topk_ids.size(1)

        # Early exit: no tokens received
        if N_recv == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    0, self.K, device=device, dtype=torch.bfloat16
                )
            )

        # --- Compute all_tokens (expanded size) ---
        # all_tokens = Σ padded_counts ≥ N_recv because tokens with multiple
        # local expert assignments are duplicated, and each expert's count is
        # padded to self._align.
        all_tokens = sum(padded_counts)
        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    N_recv, self.K, device=device, dtype=torch.bfloat16
                )
            )

        # --- Expand unique tokens into per-expert contiguous layout ---
        # Two fused triton kernels (ep_expand + ep_fill_m_indices) replace
        # ~24 PyTorch kernel launches for token duplication + m_indices fill.
        #
        # Build per-expert padded offsets on CPU: padded_counts is a Python
        # list of E ints, so cumsum on CPU avoids ~5 GPU kernel launches
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
        output_index = torch.full((N_recv, top_k), -1, device=device, dtype=torch.int32)
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
        # mxfp8_grouped_gemm quantizes activations internally via
        # _mxfp8_quant_act_v2 (per-row, per-32-col UE8M0 scale)
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
        # Normal combine is weightless (pure sum), so the executor must
        # produce the weighted partial result for each unique received token.
        #
        # ep_gather is a fused triton kernel that does:
        #   gather_out[token_i] = Σ_k down[output_index[i,k]] * topk_weights[i,k]
        # in a single kernel launch, replacing the multi-launch PyTorch path
        # (flatten → filter → index_select → mul → index_add_).
        gather_out = torch.empty(N_recv, self.K, device=device, dtype=torch.bfloat16)
        ep_gather(down, topk_ids, topk_weights, output_index, gather_out)

        return CombineForwardPayload(fused_expert_output=gather_out)


class Mxfp8LowLatencyExecutor(Mxfp8DeepepExecutor):
    """MXFP8 executor for DeepEP low-latency masked dispatch.

    Low-latency dispatch returns per-expert padded tensors of shape
    [E_local, M, K]. MXFP8 grouped GEMM consumes contiguous grouped layout, so
    flatten the already grouped expert dimension, run the two MXFP8 GEMMs, and
    reshape back for low_latency_combine().
    """

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        expert_x = payload.expert_x
        assert expert_x.dim() == 3, "low-latency MXFP8 expects [E, M, K] expert_x"
        E, M, K = expert_x.shape
        assert E == self.E and K == self.K
        assert (
            expert_x.dtype == torch.bfloat16
        ), "MXFP8 executor quantizes BF16 activations internally"
        assert payload.expert_tokens_meta is not None
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        assert masked_m is not None and len(masked_m) == self.E
        expected_m = (
            min(M, payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )

        if is_swiglu_oai(activation):
            alpha, limit = swiglu_oai_alpha_limit(extra_expert_args)
        else:
            alpha, limit = None, None

        w1_sp, w2_sp = self._packed_scales()
        # DeepEP LL valid rows are exactly masked_m[expert].  For CUDA Graph we
        # cannot launch from a GPU max(masked_m), so the correctness bound must
        # come from the fixed LL receive buffer, not from this rank's dynamic
        # token count.  M is num_max_dispatch_tokens_per_rank * ep_size, i.e.
        # the per-expert slot capacity returned by low_latency_dispatch.
        top_k = (
            int(payload.expert_topk_ids.shape[1])
            if payload.expert_topk_ids is not None
            else 1
        )
        quant_max_m = M
        expected_m = max(expected_m, quant_max_m)
        # If the fixed LL buffer already covers every routed assignment from
        # this graph bucket, M is a safe total active-row capacity. This is the
        # RTP-LLM decode path because server_config sizes ll_num_max_token by
        # moe_k. Otherwise fall back to the DeepEP API upper bound M * top_k.
        source_rows = (
            int(payload.expert_topk_ids.shape[0])
            if payload.expert_topk_ids is not None
            else M
        )
        ep_size = max(1, int(getattr(self.config, "ep_size", 1)))
        bucket_assignments = max(1, source_rows * ep_size * top_k)
        active_row_capacity = M if bucket_assignments <= M else M * top_k
        quant_max_active_rows = min(self.E * M, active_row_capacity)

        from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
            mxfp8_build_active_rows,
        )

        active_row_experts, active_row_tokens, active_row_count = (
            mxfp8_build_active_rows(
                masked_m, self.E, quant_max_m, quant_max_active_rows
            )
        )
        upgate = mxfp8_grouped_gemm_masked(
            expert_x,
            self.w1,
            w1_sp,
            masked_m,
            expected_m,
            quant_max_m=quant_max_m,
            active_row_experts=active_row_experts,
            active_row_tokens=active_row_tokens,
            active_row_count=active_row_count,
            quant_max_active_rows=quant_max_active_rows,
        )
        inter = self.N // 2
        if alpha is not None and limit is not None:
            from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
                mxfp8_swiglu_oai_quant_active_row_packed_triton,
            )

            act_q, act_s_packed = mxfp8_swiglu_oai_quant_active_row_packed_triton(
                upgate,
                masked_m,
                alpha,
                limit,
                active_row_experts,
                active_row_tokens,
                active_row_count,
                max_active_rows=quant_max_active_rows,
            )
            del upgate
            down = mxfp8_grouped_gemm_masked_prequantized(
                act_q,
                act_s_packed,
                self.w2,
                w2_sp,
                masked_m,
                expected_m,
            )
            del act_q, act_s_packed
        else:
            act = torch.empty(E * M, inter, device=upgate.device, dtype=torch.bfloat16)
            silu_and_mul(act, upgate.reshape(E * M, self.N).contiguous())
            act = act.reshape(E, M, inter)
            del upgate
            down = mxfp8_grouped_gemm_masked(
                act.contiguous(),
                self.w2,
                w2_sp,
                masked_m,
                expected_m,
                quant_max_m=quant_max_m,
                active_row_experts=active_row_experts,
                active_row_tokens=active_row_tokens,
                active_row_count=active_row_count,
                quant_max_active_rows=quant_max_active_rows,
            )
            del act
        # DeepEP low_latency_combine consumes only routed slots from the dispatch
        # handle. Keep padded rows untouched; zeroing the full [E, M, H] buffer
        # costs hundreds of microseconds in decode profiles.
        return CombineForwardPayload(fused_expert_output=down)

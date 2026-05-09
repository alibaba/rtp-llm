"""MXFP8 MoE executor using FlashInfer's CuteDSL masked grouped GEMM.

Mirrors `CutedslFp4Executor` but operates on FP8 (E4M3FN) weights with UE8M0
block scales (sf_vec_size=32). Supports two weight-loading paths:

* "Direct" — model already provides FP8 weights together with an int32
  scale tensor in the swizzled layout that flashinfer expects (W.moe_s1/s2).
* "Online" — model provides BF16 weights; the executor performs the MXFP8
  quantization at construction time using ``quant_mxfp8_per_expert``.
"""

import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    flashinfer_cutedsl_moe_masked_fp8,
    quant_mxfp8_per_expert,
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
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class CutedslFp8Executor(FusedMoeExpertExecutor):
    """MXFP8 MoE executor using FlashInfer's CuteDSL kernels."""

    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTEDSL_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )
        from rtp_llm.models_py.utils.arch import get_sm

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(get_sm()[0] >= 10)
        checker.check(
            config.moe_strategy
            in (
                "fp8_cutedsl_ep_low_latency",
                "fp8_cutedsl_ep_normal",
                "fp8_cutedsl_no_dp",
            )
        )

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        w1 = weights.get(W.moe_w1, None)
        w2 = weights.get(W.moe_w2, None)
        assert w1 is not None and w2 is not None, "moe_w1/moe_w2 must be provided"

        w1_scale = weights.get(W.moe_s1, None)
        w2_scale = weights.get(W.moe_s2, None)

        if w1.dtype == torch.float8_e4m3fn and w1_scale is not None:
            self._w1 = w1.contiguous()
            self._w2 = w2.contiguous()
            self._w1_scale = w1_scale
            self._w2_scale = w2_scale
        else:
            assert w1.dtype in (
                torch.bfloat16,
                torch.float16,
            ), f"online MXFP8 quant requires bf16/fp16 weights, got {w1.dtype}"
            self._w1, self._w1_scale = quant_mxfp8_per_expert(w1.contiguous())
            self._w2, self._w2_scale = quant_mxfp8_per_expert(w2.contiguous())

        self._w1_alpha = weights.get(W.moe_w1_s2, None)
        self._w2_alpha = weights.get(W.moe_w2_s2, None)
        if self._w1_alpha is not None and self._w1_alpha.dtype != torch.float32:
            self._w1_alpha = self._w1_alpha.to(torch.float32)
        if self._w2_alpha is not None and self._w2_alpha.dtype != torch.float32:
            self._w2_alpha = self._w2_alpha.to(torch.float32)

        self._E, self._N2, self._K = self._w1.size()
        assert self._N2 % 2 == 0, f"w1 row dim must be 2*N, got {self._N2}"
        self._N = self._N2 // 2
        assert self._w2.size(0) == self._E
        assert self._w2.size(1) == self._K
        assert self._w2.size(2) == self._N

    @property
    def local_num_experts(self) -> int:
        return self._E

    def _scatter_flat_to_masked(
        self,
        expert_x: torch.Tensor,
        expert_topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter a 2D flat ``(total_recv, K)`` dispatch output into the 3D
        masked layout ``(E, padded_M, K)`` using per-token expert routing.

        For each received row i, ``expert_topk_ids[i, k]`` gives the local
        topk experts the row is routed to (``-1`` for non-local).
        """
        device = expert_x.device
        total_recv = expert_x.shape[0]
        top_k = expert_topk_ids.shape[1]
        E = self._E

        topk_cpu = expert_topk_ids.detach().cpu().tolist()

        counter = [0] * E
        slot_assign: list[list[int]] = [[-1] * top_k for _ in range(total_recv)]
        for i in range(total_recv):
            row = topk_cpu[i]
            for k in range(top_k):
                eid = int(row[k])
                if 0 <= eid < E:
                    slot_assign[i][k] = counter[eid]
                    counter[eid] += 1

        max_m = max(counter) if counter else 0
        padded_m = max((max_m + 127) // 128 * 128, 128)

        scattered_x = torch.zeros(
            (E, padded_m, self._K),
            dtype=expert_x.dtype,
            device=device,
        )

        eid_list: list[int] = []
        slot_list: list[int] = []
        src_list: list[int] = []
        for i in range(total_recv):
            row = topk_cpu[i]
            for k in range(top_k):
                eid = int(row[k])
                if 0 <= eid < E:
                    eid_list.append(eid)
                    slot_list.append(slot_assign[i][k])
                    src_list.append(i)
        if eid_list:
            eid_t = torch.tensor(eid_list, device=device, dtype=torch.long)
            slot_t = torch.tensor(slot_list, device=device, dtype=torch.long)
            src_t = torch.tensor(src_list, device=device, dtype=torch.long)
            scattered_x[eid_t, slot_t] = expert_x[src_t]

        masked_m = torch.tensor(counter, device=device, dtype=torch.int32)
        output_index = torch.tensor(slot_assign, device=device, dtype=torch.int64)
        return scattered_x, masked_m, output_index

    def _gather_masked_to_flat(
        self,
        masked_out: torch.Tensor,
        expert_topk_ids: torch.Tensor,
        output_index: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse of :meth:`_scatter_flat_to_masked`. Per-rank weighted sum
        of expert outputs back to the flat (total_recv, K) layout.
        """
        E, padded_m, K = masked_out.shape
        total_recv, top_k = expert_topk_ids.shape
        device = masked_out.device
        out = torch.zeros((total_recv, K), dtype=masked_out.dtype, device=device)

        topk_cpu = expert_topk_ids.detach().cpu().tolist()
        idx_cpu = output_index.detach().cpu().tolist()
        weights_cpu = topk_weights.detach().to(torch.float32).cpu().tolist()
        for i in range(total_recv):
            for k in range(top_k):
                eid = int(topk_cpu[i][k])
                slot = int(idx_cpu[i][k])
                if 0 <= eid < E and slot >= 0:
                    out[i] += masked_out[eid, slot].to(out.dtype) * float(
                        weights_cpu[i][k]
                    )
        return out

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        assert expert_num_tokens is not None

        expert_x = payload.expert_x
        is_flat_input = expert_x.ndim == 2
        scatter_index: Optional[torch.Tensor] = None

        if is_flat_input:
            assert expert_x.dtype in (
                torch.bfloat16,
                torch.float16,
            ), f"flat expert_x must be bf16/fp16 for online quant, got {expert_x.dtype}"
            assert payload.expert_topk_ids is not None
            expert_x, expert_num_tokens, scatter_index = self._scatter_flat_to_masked(
                expert_x, payload.expert_topk_ids
            )

        assert expert_x.ndim == 3, f"expected 3D expert_x, got {expert_x.shape}"
        E, M, K = expert_x.size()
        assert E == self._E and K == self._K, (E, M, K, self._E, self._K)

        if expert_x.dtype in (torch.bfloat16, torch.float16):
            hidden_states = (expert_x, None)
        else:
            assert (
                expert_x.dtype == torch.float8_e4m3fn
            ), f"unexpected expert_x dtype {expert_x.dtype}"
            assert payload.expert_x_scale is not None
            hidden_states = (expert_x, payload.expert_x_scale)

        output = flashinfer_cutedsl_moe_masked_fp8(
            hidden_states=hidden_states,
            w1=self._w1,
            w1_blockscale=self._w1_scale,
            w2=self._w2,
            w2_blockscale=self._w2_scale,
            masked_m=expert_num_tokens,
            w1_alpha=self._w1_alpha,
            w2_alpha=self._w2_alpha,
        )

        if is_flat_input and scatter_index is not None:
            assert payload.expert_topk_ids is not None
            assert payload.expert_topk_weights is not None
            output = self._gather_masked_to_flat(
                output,
                payload.expert_topk_ids,
                scatter_index,
                payload.expert_topk_weights,
            )
        return CombineForwardPayload(fused_expert_output=output)

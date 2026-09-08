"""PPU DeepGEMM hybrid executors for DeepEP Normal mode."""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
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
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
    silu_mul_masked_bf16_no_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_warmup import (
    WarmupMode,
    get_deep_gemm_warmup_mode,
    resolve_deep_gemm_warmup_max_tokens,
    warmup_grouped_bf16_gemm,
    warmup_grouped_int8_gemm,
)
from rtp_llm.platforms.ppu.kernels.int8.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    deep_gemm_default_num_sms,
    has_deep_gemm_bf16_grouped,
    has_deep_gemm_bf16_grouped_nopad,
    has_deep_gemm_int8_grouped,
    has_deep_gemm_int8_grouped_nopad,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_bf16_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_contiguous,
    m_grouped_int8_gemm_nt_masked,
    m_grouped_int8_gemm_nt_nopad,
)
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.fused_silu_mul_int8_quant import (
    silu_and_mul_masked_per_token_quant_int8_fwd,
)
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.int8_quant import (
    per_token_quant_int8,
)
from rtp_llm.platforms.ppu.modules.fused_moe.kernels.ep_kernels import (
    ep_scatter_bf16,
    ep_scatter_int8,
    ep_scatter_v2_bf16,
    ep_scatter_v2_int8,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class DeepGemmBf16HybridExecutor(FusedMoeExpertExecutor):
    """Executor for DeepEP Normal bf16 mode using deepgemm grouped GEMM.

    Dispatches between two paths at runtime based on token count:
    - Masked  (token_num <= masked_max_token_num): 3D layout, efficient for decode.
    - Contiguous (token_num > masked_max_token_num): flat layout, efficient for prefill.
    """

    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]
    # Upper bound for prefill warmup tokens when not overridden via env.
    DEFAULT_WARMUP_MAX_TOKENS = 8192
    WARMUP_MASKED_LAYOUT = True

    @classmethod
    def executor_type(cls) -> ExecutorType:
        # Returns DEEPGEMM_MASKED as the nominal type; consumers use this for
        # logging/registration only — actual dispatch (masked vs contiguous) is
        # done at runtime inside execute() based on token count.
        return ExecutorType.DEEPGEMM_MASKED

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )
        from rtp_llm.platforms.ppu.kernels.int8.deepgemm_wrapper import (
            has_deep_gemm_bf16_grouped,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm_bf16_grouped())
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num

        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.activation = config.activation_type
        self.masked_max_token_num = config.masked_max_token_num

        # Weight initialization (bf16, no quantization)
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]

        self.num_local_experts, self.intermediate_size, self.hidden_size = (
            self.w1.size()
        )
        assert self.intermediate_size % 2 == 0
        assert self.w2.size(0) == self.num_local_experts
        assert self.w2.size(1) == self.hidden_size
        assert self.w2.size(2) == self.intermediate_size // 2

        self.num_gemm_sms = deep_gemm_default_num_sms()

        # Prefer the nopad grouped GEMM when the deep_gemm build ships it: the
        # contiguous path pads each expert to EXPERT_ALIGNMENT (128) and launches
        # GEMM tiles over the padding rows, whereas nopad packs the actual tokens
        # and drives the ragged grouped scheduler via per-expert counts (m_rows).
        # Only affects execute_contiguous (prefill). Set MOE_BF16_NOPAD=0 to force
        # the contiguous path (for A/B perf comparison).
        self.use_nopad = (
            has_deep_gemm_bf16_grouped_nopad()
            and os.environ.get("MOE_BF16_NOPAD", "1") == "1"
        )
        logger.info(
            f"[DeepGemmBf16HybridExecutor] contiguous(prefill) GEMM layout: "
            f"{'nopad' if self.use_nopad else 'contiguous(128-padded)'}"
        )
        if self.w1.dtype == torch.bfloat16:
            self._warmup_bf16_kernels()

    def _get_warmup_limits(self) -> tuple[WarmupMode, int, int]:
        warmup_mode = get_deep_gemm_warmup_mode()
        if warmup_mode == "skip":
            return warmup_mode, 0, 0
        max_masked_m = min(
            align(self.masked_max_token_num, self.EXPERT_ALIGNMENT),
            ceil_div(
                self.masked_max_token_num * self.top_k,
                self.num_experts_per_partition,
            ),
        )
        if not self.WARMUP_MASKED_LAYOUT:
            max_masked_m = 0
        max_prefill_tokens = resolve_deep_gemm_warmup_max_tokens(
            int(self.config.model_config.max_seq_len),
            default=self.DEFAULT_WARMUP_MAX_TOKENS,
        )
        max_packed_m = max_prefill_tokens * self.top_k
        if not self.use_nopad:
            max_packed_m += self.num_experts_per_partition * (self.EXPERT_ALIGNMENT - 1)
        return warmup_mode, max_masked_m, max_packed_m

    def _warmup_bf16_kernels(self) -> None:
        try:
            warmup_mode, max_masked_m, max_prefill_m = self._get_warmup_limits()
            if warmup_mode == "skip":
                return
            for weight in (self.w1, self.w2):
                if max_masked_m:
                    warmup_grouped_bf16_gemm(
                        weight,
                        max_m=max_masked_m,
                        layout="masked",
                        mode=warmup_mode,
                        num_sms=self.num_gemm_sms,
                    )
                warmup_grouped_bf16_gemm(
                    weight,
                    max_m=max_prefill_m,
                    layout="nopad" if self.use_nopad else "contiguous",
                    mode=warmup_mode,
                    num_sms=self.num_gemm_sms,
                )
        except Exception:
            logger.exception("DeepGEMM BF16 warmup failed; execution will use lazy JIT")

    def _to_local_expert_ids(self, topk_idx: torch.Tensor) -> torch.Tensor:
        """Convert global expert IDs to partition-local IDs (0-based), -1 for out-of-partition."""
        local = topk_idx - self.start_expert_id
        return torch.where(
            (local >= 0) & (local < self.num_experts_per_partition),
            local,
            torch.tensor(-1, device=local.device, dtype=local.dtype),
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        assert payload.expert_tokens_meta.expert_num_tokens is not None
        # Router weight is always applied at the gather stage (ep_gather). DeepEP Normal
        # callers must pass apply_router_weight_on_input=False.
        assert not apply_router_weight_on_input, (
            "DeepGemmBf16HybridExecutor applies router weight at gather; "
            "apply_router_weight_on_input=True is not supported."
        )

        token_num = payload.expert_x.shape[0]
        # Empty rank: DeepEP small-batch / skewed routing can leave this rank with
        # zero tokens. Return an empty same-shape output before dispatch — otherwise
        # the masked path (token_num <= masked_max_token_num) would run with
        # alignment == 0 and launch 0-grid Triton scatter / 0-size DeepGEMM.
        if token_num == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.empty_like(payload.expert_x)
            )
        if token_num <= self.masked_max_token_num:
            return self.execute_masked(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        else:
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            hidden_states = payload.expert_x
            topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
            topk_weights = payload.expert_topk_weights
            num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

            token_num = hidden_states.shape[0]
            num_experts = num_recv_tokens_per_expert.shape[0]
            max_token_num = token_num * self.top_k
            token_num_mean_per_expert = ceil_div(max_token_num, num_experts)
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, token_num_mean_per_expert)

            device = hidden_states.device
            hidden_states_shape = hidden_states.shape

            # Step 1: Scatter flat [M, K] → 3D [E, alignment, K]
            input_tensor = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)

            ep_scatter_v2_bf16(
                hidden_states,
                topk_idx,
                alignment,
                expert_start_loc,
                input_tensor.view(
                    self.num_experts_per_partition * alignment, self.hidden_size
                ),
                output_index,
            )
            dispose_tensor(hidden_states)

            # Step 2: Gate and Up GEMM (deepgemm bf16 masked)
            upgate_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                input_tensor,
                self.w1,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(input_tensor)

            # Step 3: SiLU Activation (masked bf16)
            down_input = torch.empty(
                (
                    self.num_experts_per_partition,
                    alignment,
                    self.intermediate_size // 2,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
            silu_mul_masked_bf16_no_post_quant_fwd(
                input=upgate_output,
                output=down_input,
                masked_m=num_recv_tokens_per_expert,
                expected_m=expected_m,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
            )
            dispose_tensor(upgate_output)

            # Step 4: Down GEMM (deepgemm bf16 masked)
            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_bf16_gemm_nt_masked(
                down_input,
                self.w2,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
            )
            dispose_tensor(down_input)

            # Step 5: Gather 3D → flat, with router weight multiplication
            gather_out = torch.empty(
                hidden_states_shape, device=device, dtype=torch.bfloat16
            )
            ep_gather(
                down_output.view(
                    self.num_experts_per_partition * alignment, self.hidden_size
                ),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
            dispose_tensor(down_output)

            return CombineForwardPayload(fused_expert_output=gather_out)

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """Large-token-count path: flat [all_tokens, K] layout, efficient for prefill.

        Two layouts, chosen by self.use_nopad:
          - nopad (preferred when deep_gemm ships it): each expert's tokens packed
            with NO per-expert padding (all_tokens = Σ actual), the ragged grouped
            scheduler driven by per-expert counts (m_rows). No wasted tiles/memory.
          - contiguous (fallback): each expert padded to EXPERT_ALIGNMENT (128), so
            GEMM tiles run over padding rows.
        """
        hidden_states = payload.expert_x
        topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
        topk_weights = payload.expert_topk_weights

        # Get per-expert token counts as a Python list (needed for alignment arithmetic)
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            tokens_per_expert_list = payload.expert_tokens_meta.expert_num_tokens_cpu
        else:
            tokens_per_expert_list = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
        if isinstance(tokens_per_expert_list, torch.Tensor):
            tokens_per_expert_list = tokens_per_expert_list.tolist()

        # nopad: no per-expert padding (align to 1); contiguous: pad each to 128.
        expert_alignment = 1 if self.use_nopad else self.EXPERT_ALIGNMENT
        aligned_tokens = [align(x, expert_alignment) for x in tokens_per_expert_list]
        all_tokens = sum(aligned_tokens)

        device = hidden_states.device
        hidden_states_shape = hidden_states.shape

        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_shape, device=device, dtype=torch.bfloat16
                )
            )

        num_recv_tokens_per_expert_gpu = torch.tensor(
            aligned_tokens, dtype=torch.int32, pin_memory=True, device="cpu"
        ).to(device=device, non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
        m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
        output_index = torch.empty_like(topk_idx)

        # Step 1: Scatter flat [M, K] → expert-sorted flat [all_tokens, K]
        input_tensor = torch.empty(
            (all_tokens, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        ep_scatter_bf16(
            hidden_states,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor,
            m_indices,
            output_index,
        )
        # ep_scatter_bf16 fills m_indices for occupied slots and leaves padding slots at 0
        # (from torch.empty initialization). clamp_ guards against any stale values in
        # unoccupied trailing slots that deepgemm uses as the expert-index array.
        m_indices.clamp_(min=0, max=self.num_experts_per_partition - 1)
        dispose_tensor(hidden_states)

        # For nopad, m_rows = actual per-expert token counts (== num_recv_tokens_per_expert_gpu
        # here, since expert_alignment=1). Unused by the contiguous path.
        m_rows = num_recv_tokens_per_expert_gpu

        # Step 2: Gate and Up GEMM (deepgemm bf16)
        gateup_output = torch.empty(
            (all_tokens, self.intermediate_size), device=device, dtype=torch.bfloat16
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            if self.use_nopad:
                m_grouped_bf16_gemm_nt_nopad(
                    input_tensor, self.w1, gateup_output, m_indices, m_rows
                )
            else:
                m_grouped_bf16_gemm_nt_contiguous(
                    input_tensor, self.w1, gateup_output, m_indices
                )
        dispose_tensor(input_tensor)

        # Step 3: SiLU activation (flat, no mask needed)
        down_input = torch.empty(
            (all_tokens, self.intermediate_size // 2),
            device=device,
            dtype=torch.bfloat16,
        )
        ppu_silu_and_mul_bf16(down_input, gateup_output)
        dispose_tensor(gateup_output)

        # Step 4: Down GEMM (deepgemm bf16)
        down_output = torch.empty(
            (all_tokens, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            if self.use_nopad:
                m_grouped_bf16_gemm_nt_nopad(
                    down_input, self.w2, down_output, m_indices, m_rows
                )
            else:
                m_grouped_bf16_gemm_nt_contiguous(
                    down_input, self.w2, down_output, m_indices
                )
        dispose_tensor(down_input)

        # Step 5: Gather flat → [M, K], apply router weights
        gather_out = torch.empty(
            hidden_states_shape, device=device, dtype=torch.bfloat16
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        dispose_tensor(down_output)

        return CombineForwardPayload(fused_expert_output=gather_out)


class DeepGemmInt8HybridExecutor(DeepGemmBf16HybridExecutor):
    """INT8 DeepEP dispatch and expert GEMMs with BF16 GEMM outputs."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm_int8_grouped())
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.w1_scale = weights[W.moe_s1]
        self.w2_scale = weights[W.moe_s2]
        if self.w1.dtype != torch.int8 or self.w2.dtype != torch.int8:
            raise ValueError(
                f"W8A8 MoE expects INT8 weights, got {self.w1.dtype}/{self.w2.dtype}"
            )
        if self.w1_scale.dtype != torch.float32 or self.w2_scale.dtype != torch.float32:
            raise ValueError(
                "W8A8 MoE expects FP32 scales, got "
                f"{self.w1_scale.dtype}/{self.w2_scale.dtype}"
            )
        if self.w1_scale.shape != (*self.w1.shape[:-1], 1):
            raise ValueError(
                f"w1 scale shape {self.w1_scale.shape} does not match {self.w1.shape}"
            )
        if self.w2_scale.shape != (*self.w2.shape[:-1], 1):
            raise ValueError(
                f"w2 scale shape {self.w2_scale.shape} does not match {self.w2.shape}"
            )
        self.use_nopad = has_deep_gemm_int8_grouped_nopad()
        logger.info(
            "[DeepGemmInt8HybridExecutor] prefill GEMM layout: %s",
            "nopad" if self.use_nopad else "contiguous",
        )
        self._warmup_int8_kernels()

    def _warmup_int8_kernels(self) -> None:
        try:
            warmup_mode, max_masked_m, max_prefill_m = self._get_warmup_limits()
            if warmup_mode == "skip":
                return
            for weight in (
                (self.w1, self.w1_scale),
                (self.w2, self.w2_scale),
            ):
                if max_masked_m:
                    warmup_grouped_int8_gemm(
                        weight,
                        max_m=max_masked_m,
                        layout="masked",
                        mode=warmup_mode,
                        num_sms=self.num_gemm_sms,
                    )
                warmup_grouped_int8_gemm(
                    weight,
                    max_m=max_prefill_m,
                    layout="nopad" if self.use_nopad else "contiguous",
                    mode=warmup_mode,
                    num_sms=self.num_gemm_sms,
                )
        except Exception:
            logger.exception("DeepGEMM INT8 warmup failed; execution will use lazy JIT")

    @staticmethod
    def _get_dispatch_scale(payload: ExpertForwardPayload) -> torch.Tensor:
        hidden_states = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        if hidden_states.dtype != torch.int8:
            raise ValueError(
                f"W8A8 DeepEP Normal expected INT8 input, got {hidden_states.dtype}"
            )
        if hidden_states_scale is None or hidden_states_scale.dtype != torch.float32:
            raise ValueError("W8A8 DeepEP Normal expected an FP32 per-token scale")
        if hidden_states_scale.shape != (hidden_states.shape[0], 1):
            raise ValueError(
                f"input scale shape {hidden_states_scale.shape} does not match "
                f"({hidden_states.shape[0]}, 1)"
            )
        return hidden_states_scale

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            hidden_states = payload.expert_x
            hidden_states_scale = self._get_dispatch_scale(payload)
            topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
            topk_weights = payload.expert_topk_weights
            tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

            token_num = hidden_states.shape[0]
            num_experts = tokens_per_expert.shape[0]
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, ceil_div(token_num * self.top_k, num_experts))
            device = hidden_states.device
            hidden_states_shape = hidden_states.shape

            input_q = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.int8,
            )
            input_scale = torch.empty(
                (self.num_experts_per_partition, alignment, 1),
                device=device,
                dtype=torch.float32,
            )
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(tokens_per_expert)
            ep_scatter_v2_int8(
                hidden_states,
                hidden_states_scale,
                topk_idx,
                alignment,
                expert_start_loc,
                input_q.view(-1, self.hidden_size),
                input_scale.view(-1, 1),
                output_index,
            )
            dispose_tensor(hidden_states)
            dispose_tensor(hidden_states_scale)
            gateup_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.intermediate_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_int8_gemm_nt_masked(
                (input_q, input_scale),
                (self.w1, self.w1_scale),
                gateup_output,
                tokens_per_expert,
                expected_m,
            )
            dispose_tensor(input_q)
            dispose_tensor(input_scale)

            down_q = torch.empty(
                (
                    self.num_experts_per_partition,
                    alignment,
                    self.intermediate_size // 2,
                ),
                device=device,
                dtype=torch.int8,
            )
            down_scale = torch.empty(
                (self.num_experts_per_partition, alignment, 1),
                device=device,
                dtype=torch.float32,
            )
            silu_and_mul_masked_per_token_quant_int8_fwd(
                input=gateup_output,
                output=down_q,
                output_scale=down_scale,
                masked_m=tokens_per_expert,
            )
            dispose_tensor(gateup_output)

            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
            m_grouped_int8_gemm_nt_masked(
                (down_q, down_scale),
                (self.w2, self.w2_scale),
                down_output,
                tokens_per_expert,
                expected_m,
            )
            dispose_tensor(down_q)
            dispose_tensor(down_scale)

            gather_out = torch.empty(
                hidden_states_shape, device=device, dtype=torch.bfloat16
            )
            ep_gather(
                down_output.view(-1, self.hidden_size),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
            dispose_tensor(down_output)
            return CombineForwardPayload(fused_expert_output=gather_out)

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden_states = payload.expert_x
        hidden_states_scale = self._get_dispatch_scale(payload)
        topk_idx = self._to_local_expert_ids(payload.expert_topk_ids)
        topk_weights = payload.expert_topk_weights
        tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens_cpu
        if tokens_per_expert is None:
            tokens_per_expert = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()

        expert_alignment = 1 if self.use_nopad else self.EXPERT_ALIGNMENT
        aligned_tokens = [align(count, expert_alignment) for count in tokens_per_expert]
        all_tokens = sum(aligned_tokens)
        device = hidden_states.device
        hidden_states_shape = hidden_states.shape
        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_shape,
                    device=device,
                    dtype=payload.expert_x_origin_dtype,
                )
            )

        token_counts = torch.tensor(
            aligned_tokens, dtype=torch.int32, pin_memory=True, device="cpu"
        ).to(device=device, non_blocking=True)
        expert_start_loc = torch.empty_like(token_counts)
        m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
        output_index = torch.empty_like(topk_idx)
        input_q = torch.empty(
            (all_tokens, self.hidden_size), device=device, dtype=torch.int8
        )
        input_scale = torch.empty((all_tokens, 1), device=device, dtype=torch.float32)
        ep_scatter_int8(
            hidden_states,
            hidden_states_scale,
            topk_idx,
            token_counts,
            expert_start_loc,
            input_q,
            input_scale,
            m_indices,
            output_index,
        )
        m_indices.clamp_(min=0, max=self.num_experts_per_partition - 1)
        dispose_tensor(hidden_states)
        dispose_tensor(hidden_states_scale)
        gateup_output = torch.empty(
            (all_tokens, self.intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            if self.use_nopad:
                m_grouped_int8_gemm_nt_nopad(
                    (input_q, input_scale),
                    (self.w1, self.w1_scale),
                    gateup_output,
                    m_indices,
                    token_counts,
                )
            else:
                m_grouped_int8_gemm_nt_contiguous(
                    (input_q, input_scale),
                    (self.w1, self.w1_scale),
                    gateup_output,
                    m_indices,
                )
        dispose_tensor(input_q)
        dispose_tensor(input_scale)

        down_bf16 = torch.empty(
            (all_tokens, self.intermediate_size // 2),
            device=device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(down_bf16, gateup_output)
        dispose_tensor(gateup_output)
        down_q, down_scale = per_token_quant_int8(down_bf16)
        dispose_tensor(down_bf16)

        down_output = torch.empty(
            (all_tokens, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            if self.use_nopad:
                m_grouped_int8_gemm_nt_nopad(
                    (down_q, down_scale),
                    (self.w2, self.w2_scale),
                    down_output,
                    m_indices,
                    token_counts,
                )
            else:
                m_grouped_int8_gemm_nt_contiguous(
                    (down_q, down_scale),
                    (self.w2, self.w2_scale),
                    down_output,
                    m_indices,
                )
        dispose_tensor(down_q)
        dispose_tensor(down_scale)

        gather_out = torch.empty(
            hidden_states_shape, device=device, dtype=torch.bfloat16
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        dispose_tensor(down_output)
        return CombineForwardPayload(fused_expert_output=gather_out)

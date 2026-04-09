"""Deterministic MoE executor for ROCm using ATREX split_reduce.

Uses ATREX Triton Gluon kernels with split_reduce mode for deterministic
MoE computation. The split_reduce approach writes each expert's weighted
output to a separate [M, TOPK, N] slot, then reduces along TOPK with a
simple sum — avoiding AtomicAdd entirely.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch

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

log = logging.getLogger(__name__)


def _moe_sorting_python(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure Python MoE sorting that produces ATREX-compatible format.

    ATREX kernel expects sorted_ids encoded as:
        token_index = sorted_id & 0xFFFFFF   (lower 24 bits)
        topk_slot   = sorted_id >> 24        (upper 8 bits)

    Returns:
        sorted_ids:        [max_num_tokens_padded] int32, packed (topk_slot << 24 | token_index)
        sorted_weights:    [max_num_tokens_padded] float32
        sorted_expert_ids: [max_num_m_blocks] int32
        num_valid_ids:     [1] int32, total valid entries (including padding)
    """
    M, TOPK = topk_ids.shape
    device = topk_ids.device

    topk_ids_cpu = topk_ids.cpu()
    topk_weights_cpu = topk_weights.cpu()

    # Group tokens by expert
    expert_to_tokens = [[] for _ in range(num_experts)]
    for token_idx in range(M):
        for topk_slot in range(TOPK):
            expert_id = topk_ids_cpu[token_idx, topk_slot].item()
            if expert_id < 0 or expert_id >= num_experts:
                continue  # Skip invalid expert IDs
            weight = topk_weights_cpu[token_idx, topk_slot].item()
            packed_id = (topk_slot << 24) | token_idx
            expert_to_tokens[expert_id].append((packed_id, weight))

    # Build sorted arrays: tokens grouped by expert, padded to block_size
    sorted_ids_list = []
    sorted_weights_list = []
    sorted_expert_ids_list = []

    PADDING_ID = M * TOPK  # Invalid token index, kernel will skip

    for expert_id in range(num_experts):
        tokens = expert_to_tokens[expert_id]
        n_tokens = len(tokens)
        # Pad to multiple of block_size
        n_padded = ((n_tokens + block_size - 1) // block_size) * block_size
        if n_padded == 0:
            n_padded = block_size  # At least one block per expert

        for i in range(n_padded):
            if i < n_tokens:
                sorted_ids_list.append(tokens[i][0])
                sorted_weights_list.append(tokens[i][1])
            else:
                sorted_ids_list.append(PADDING_ID)
                sorted_weights_list.append(0.0)

        # One expert_id entry per block
        n_blocks = n_padded // block_size
        for _ in range(n_blocks):
            sorted_expert_ids_list.append(expert_id)

    num_valid = len(sorted_ids_list)

    sorted_ids = torch.tensor(sorted_ids_list, dtype=torch.int32, device=device)
    sorted_weights = torch.tensor(
        sorted_weights_list, dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.tensor(
        sorted_expert_ids_list, dtype=torch.int32, device=device
    )
    num_valid_ids = torch.tensor([num_valid], dtype=torch.int32, device=device)

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


class AtrexFusedMoeExecutor(FusedMoeExpertExecutor):
    """Deterministic MoE executor using ATREX split_reduce.

    Uses Python-based moe_sorting + ATREX Triton Gluon kernels:
    - stage1: gate+up GEMM + SiLU activation
    - stage2_split_reduce: down GEMM → [M, TOPK, N] (no AtomicAdd)
    - reduce: sum along TOPK → [M, N]
    """

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        pass

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.num_experts = config.expert_num
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank

        from atrex.src.triton.fused_moe.fused_moe_helper import shuffle_weight

        # shuffle_moe_weight is skipped when USE_ATREX_MOE=1.
        # After mixtbstars moe_stack_w1_pad (permute(0,2,1) + pad + concat):
        #   w1 = [E, inter_dim_padded*2, model_dim], order [up, gate]
        # After no_stack_pad (permute(0,2,1) + pad):
        #   w2 = [E, model_dim, inter_dim_padded]
        #
        # ATREX expects:
        #   w1 = [E, inter_dim*2, model_dim], order [gate, up], (16,16) shuffled
        #   w2 = [E, model_dim, inter_dim], (16,16) shuffled
        # Dimensions already match, just need gate/up swap + shuffle.

        w1_raw = weights[W.moe_w1]  # [E, inter_dim*2, model_dim] or padded
        w2_raw = weights[W.moe_w2]  # [E, model_dim, inter_dim] or padded

        # w1: swap [up, gate] → [gate, up]
        half = w1_raw.shape[1] // 2
        w1_atrex = torch.cat([w1_raw[:, half:, :], w1_raw[:, :half, :]], dim=1)

        # Pad dimensions for ATREX Triton kernel alignment.
        # w1 has [gate, up] concatenated on dim=1 — pad each half separately
        # to match w2's inter_dim, then re-concat.
        ALIGN = 256

        def _pad_to_align(x, dim, align):
            size = x.shape[dim]
            if size % align == 0:
                return x
            pad_size = align - (size % align)
            pad_shape = [0] * (2 * x.dim())
            idx = (x.dim() - 1 - dim) * 2
            pad_shape[idx + 1] = pad_size
            return torch.nn.functional.pad(x, pad_shape)

        # w1: pad each half (gate, up) separately on dim=1
        half = w1_atrex.shape[1] // 2
        w1_gate = _pad_to_align(w1_atrex[:, :half, :], dim=1, align=ALIGN)
        w1_up = _pad_to_align(w1_atrex[:, half:, :], dim=1, align=ALIGN)
        w1_atrex = torch.cat([w1_gate, w1_up], dim=1)

        # w2: pad inter_dim (dim=2)
        w2_raw = _pad_to_align(w2_raw, dim=2, align=ALIGN)

        self.w1 = shuffle_weight(w1_atrex, layout=(16, 16))
        self.w2 = shuffle_weight(w2_raw, layout=(16, 16))

        # Save pre-shuffle weights for debug verification
        if os.environ.get("DEBUG_WEIGHT_CHECK", "0") == "1":
            self._w1_raw = w1_atrex.clone()  # [E, inter*2, model_dim], [gate, up] order
            self._w2_raw = w2_raw.clone()  # [E, model_dim, inter]
            self._debug_done = False

        log.info(
            f"AtrexFusedMoeExecutor: experts={self.num_experts}, "
            f"w1={list(self.w1.shape)}/{self.w1.dtype}, "
            f"w2={list(self.w2.shape)}/{self.w2.dtype}"
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
        from atrex.src.triton.fused_moe.fused_moe_helper import (
            flush_cache,
            get_block_size_m,
            get_m_align,
        )
        from atrex.src.triton.fused_moe.fused_moe_stage1_fp16 import (
            fused_moe_stage1_fp16,
        )
        from atrex.src.triton.fused_moe.fused_moe_stage2_split_reduce_fp16 import (
            fused_moe_stage2_split_reduce_fp16,
        )
        from atrex.src.triton.fused_moe.reduce_fp16 import reduce_fp16

        hidden_states = payload.expert_x
        topk_ids = payload.expert_topk_ids.to(torch.int32)
        topk_weights = payload.expert_topk_weights.to(torch.float32)

        M, K = hidden_states.shape
        TOPK = topk_ids.shape[1]
        E = self.num_experts
        N1 = self.w1.shape[1]  # inter_dim * 2 (with padding)
        N2 = self.w2.shape[1]  # model_dim
        device = hidden_states.device

        if not hasattr(self, "_logged_first"):
            self._logged_first = True
            log.info(
                f"AtrexFusedMoeExecutor.execute: M={M}, K={K}, TOPK={TOPK}, E={E}, "
                f"N1={N1}, N2={N2}, hidden_states={list(hidden_states.shape)}/{hidden_states.dtype}, "
                f"w1={list(self.w1.shape)}/{self.w1.dtype}, w2={list(self.w2.shape)}/{self.w2.dtype}"
            )

        if M == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.empty(
                    (0, N2), dtype=hidden_states.dtype, device=device
                )
            )

        # MoE Sorting — pure Python, produces ATREX-compatible format
        BLOCK_SIZE_M = get_block_size_m(M, TOPK, E)
        M_ALIGN = get_m_align(M, TOPK, E)

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
            _moe_sorting_python(topk_ids, topk_weights, E, BLOCK_SIZE_M)
        )

        # Stage 1: gate+up GEMM + SiLU (ATREX Triton Gluon)
        gemm1_out = torch.empty(
            [M, TOPK, N1 // 2],
            dtype=hidden_states.dtype,
            device=device,
        )

        flush_cache(512)
        fused_moe_stage1_fp16(
            hidden_states,
            self.w1,
            gemm1_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            M_ALIGN,
            BLOCK_SIZE_M,
        )

        # Stage 2: down GEMM with split_reduce (deterministic)
        split_out = torch.empty(
            [M, TOPK, N2],
            dtype=hidden_states.dtype,
            device=device,
        )

        flush_cache(512)
        fused_moe_stage2_split_reduce_fp16(
            gemm1_out,
            self.w2,
            split_out,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            M_ALIGN,
            BLOCK_SIZE_M,
        )

        # Reduce: sum along TOPK [M, TOPK, N2] -> [M, N2]
        flush_cache(512)
        output = reduce_fp16(split_out)

        # === Debug: weight self-consistency check ===
        if (
            os.environ.get("DEBUG_WEIGHT_CHECK", "0") == "1"
            and hasattr(self, "_w1_raw")
            and not self._debug_done
        ):
            self._debug_done = True
            try:
                self._verify_weights(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    gemm1_out,
                    split_out,
                    output,
                    M,
                    TOPK,
                )
            except Exception as e:
                print(f"[DEBUG_WEIGHT_CHECK] Error: {e}", flush=True)
                import traceback

                traceback.print_exc()

        return CombineForwardPayload(fused_expert_output=output)

    def _verify_weights(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        gemm1_out,
        split_out,
        output,
        M,
        TOPK,
    ):
        """Compare kernel output vs torch.matmul using pre-shuffle weights.

        For a token t assigned to expert e at topk_slot k:
          Stage1 ref: silu(x @ w1_gate.T) * (x @ w1_up.T)
          Stage2 ref: stage1_out @ w2_e.T * weight
        """
        print(f"[DEBUG_WEIGHT_CHECK] === Weight self-consistency check ===", flush=True)
        print(
            f"[DEBUG_WEIGHT_CHECK] hidden_states: {list(hidden_states.shape)} {hidden_states.dtype}",
            flush=True,
        )
        print(
            f"[DEBUG_WEIGHT_CHECK] w1_raw: {list(self._w1_raw.shape)} {self._w1_raw.dtype}",
            flush=True,
        )
        print(
            f"[DEBUG_WEIGHT_CHECK] w2_raw: {list(self._w2_raw.shape)} {self._w2_raw.dtype}",
            flush=True,
        )
        print(
            f"[DEBUG_WEIGHT_CHECK] gemm1_out: {list(gemm1_out.shape)} {gemm1_out.dtype}",
            flush=True,
        )
        print(
            f"[DEBUG_WEIGHT_CHECK] split_out: {list(split_out.shape)} {split_out.dtype}",
            flush=True,
        )
        print(f"[DEBUG_WEIGHT_CHECK] topk_ids[0]: {topk_ids[0].tolist()}", flush=True)
        print(
            f"[DEBUG_WEIGHT_CHECK] topk_weights[0]: {topk_weights[0].tolist()}",
            flush=True,
        )

        # Check first 3 (token, topk_slot) pairs
        n_check = min(3, M)
        for t in range(n_check):
            for k in range(min(2, TOPK)):
                expert_id = topk_ids[t, k].item()
                weight_val = topk_weights[t, k].item()

                x = hidden_states[t].float()  # [K]
                w1_e = self._w1_raw[
                    expert_id
                ].float()  # [inter*2, model_dim], [gate, up]
                w2_e = self._w2_raw[expert_id].float()  # [model_dim, inter]

                inter_dim = w1_e.shape[0] // 2
                w1_gate = w1_e[:inter_dim, :]  # [inter, model_dim]
                w1_up = w1_e[inter_dim:, :]  # [inter, model_dim]

                # Stage1 ref: SiGLU = silu(x @ gate.T) * (x @ up.T)
                gate_out = x @ w1_gate.T  # [inter]
                up_out = x @ w1_up.T  # [inter]
                silu_gate = gate_out * torch.sigmoid(gate_out)
                stage1_ref = silu_gate * up_out  # [inter]

                # Kernel stage1 output
                stage1_kernel = gemm1_out[t, k, :].float()  # [inter]

                s1_diff = (stage1_ref - stage1_kernel).abs()
                s1_max = s1_diff.max().item()
                s1_rel = (s1_diff / (stage1_ref.abs().clamp(min=1e-6))).max().item()

                print(
                    f"[DEBUG_WEIGHT_CHECK] token={t} slot={k} expert={expert_id} weight={weight_val:.6f}",
                    flush=True,
                )
                print(
                    f"  Stage1: ref mean={stage1_ref.mean().item():.6f} std={stage1_ref.std().item():.6f}",
                    flush=True,
                )
                print(
                    f"  Stage1: kernel mean={stage1_kernel.mean().item():.6f} std={stage1_kernel.std().item():.6f}",
                    flush=True,
                )
                print(
                    f"  Stage1: max_abs_diff={s1_max:.6f} max_rel_diff={s1_rel:.6f}",
                    flush=True,
                )

                # Stage2 ref: stage1_out @ w2.T * weight
                stage2_ref = (stage1_ref @ w2_e.T) * weight_val  # [model_dim]

                # Kernel stage2 output (split_out already has weight applied)
                stage2_kernel = split_out[t, k, :].float()  # [model_dim]

                s2_diff = (stage2_ref - stage2_kernel).abs()
                s2_max = s2_diff.max().item()
                s2_rel = (s2_diff / (stage2_ref.abs().clamp(min=1e-6))).max().item()

                print(
                    f"  Stage2: ref mean={stage2_ref.mean().item():.6f} std={stage2_ref.std().item():.6f}",
                    flush=True,
                )
                print(
                    f"  Stage2: kernel mean={stage2_kernel.mean().item():.6f} std={stage2_kernel.std().item():.6f}",
                    flush=True,
                )
                print(
                    f"  Stage2: max_abs_diff={s2_max:.6f} max_rel_diff={s2_rel:.6f}",
                    flush=True,
                )

                if s1_max > 1.0:
                    print(
                        f"  !!! Stage1 MISMATCH - weights likely wrong !!!", flush=True
                    )
                if s2_max > 1.0:
                    print(
                        f"  !!! Stage2 MISMATCH - weights likely wrong !!!", flush=True
                    )

        # Also check final reduced output vs manual reduce
        reduce_ref = split_out.float().sum(dim=1)  # [M, N2]
        reduce_kernel = output.float()
        r_diff = (reduce_ref - reduce_kernel).abs().max().item()
        print(f"[DEBUG_WEIGHT_CHECK] Reduce check: max_diff={r_diff:.6f}", flush=True)
        print(f"[DEBUG_WEIGHT_CHECK] === Done ===", flush=True)

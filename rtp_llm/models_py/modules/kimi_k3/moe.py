"""Latent routed and shared-expert MoE implementation for Kimi K3."""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import (
    KimiK3WeightNames as K3W,
    shared_expert_weight_shard_enabled,
)
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
)
from rtp_llm.models_py.modules.base import GroupTopK, RMSNorm
from rtp_llm.models_py.triton_kernels.common.activation import situ_and_mul
from rtp_llm.ops import ParallelismConfig

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

_DEEPGEMM_MEGA_LOGGED_DEVICES: set[int] = set()


def _transient_full_native_column_weight(
    local_weight: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Gather a checkpoint-native ``[out/world_size, in]`` shard transiently."""

    if world_size <= 1:
        return local_weight
    return all_gather(local_weight.contiguous(), group=Group.TP)


def _transient_full_row_weight(
    local_weight: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Gather an ``[in/world_size, out]`` shard without retaining it."""

    if world_size <= 1:
        return local_weight
    return all_gather(local_weight.contiguous(), group=Group.TP)


class KimiK3LatentMoE(nn.Module):
    """K3 latent MoE backed exclusively by DeepGEMM MegaMoE.

    ``fp8_fp4_mega_moe`` owns expert dispatch, both MXFP4 expert GEMMs,
    activation, and combine through a symmetric-memory collective.  K3 does
    not initialize or call RTP's separate DeepEP dispatch implementation.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.expert_num = int(config.expert_num)
        self.top_k = int(config.moe_k)
        self.renormalize = bool(config.has_moe_norm)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.num_expert_group = int(config.moe_n_group)
        self.topk_group = int(config.moe_topk_group)
        self.ep_size = int(parallelism_config.ep_size)
        if self.expert_num % self.ep_size:
            raise ValueError(
                f"expert count {self.expert_num} must divide EP size {self.ep_size}"
            )
        self.local_expert_count = self.expert_num // self.ep_size
        self.attn_tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.ffn_tp_size = int(parallelism_config.get_ffn_tp_size())
        self.ffn_tp_rank = int(parallelism_config.get_ffn_tp_rank())
        self.shared_expert_weight_shard = shared_expert_weight_shard_enabled(
            parallelism_config.role_type
        )
        self.shared_intermediate_size = int(config.inter_size)
        self._validate_shared_expert_weight_layout(config.hidden_size)
        runtime = config.k3_runtime_config
        self.latent_moe_use_norm = runtime.latent_moe_use_norm
        self.beta = runtime.activation_situ_beta
        self.linear_beta = runtime.activation_situ_linear_beta
        self.eps = float(config.layernorm_eps)
        self.routed_norm = (
            RMSNorm(self.weights[K3W.MOE_ROUTED_NORM], self.eps)
            if self.latent_moe_use_norm
            else None
        )
        self.latent_size = int(self.weights[K3W.MOE_ROUTED_DOWN].shape[1])
        self.layer_idx = int(layer_idx)
        self._group_topk = GroupTopK()
        self._setup_deep_gemm_mega()

    def _validate_shared_expert_weight_layout(self, hidden_size: int) -> None:
        intermediate = self.shared_intermediate_size
        if self.shared_expert_weight_shard:
            if self.ffn_tp_size <= 0 or self.ffn_tp_size % 2:
                raise ValueError(
                    "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD=1 requires an even FFN "
                    f"TP size, got {self.ffn_tp_size}"
                )
            if intermediate % (self.ffn_tp_size // 2):
                raise ValueError(
                    "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD=1 requires shared "
                    "intermediate size divisible by FFN TP/2, got "
                    f"intermediate={intermediate} ffn_tp={self.ffn_tp_size}"
                )
            if intermediate % self.ffn_tp_size:
                raise ValueError(
                    "sharded K3 shared-down weight requires shared intermediate "
                    "size divisible by FFN TP, got "
                    f"intermediate={intermediate} ffn_tp={self.ffn_tp_size}"
                )
            gate_up_rows = 2 * intermediate // self.ffn_tp_size
            down_rows = intermediate // self.ffn_tp_size
        else:
            gate_up_rows = 2 * intermediate
            down_rows = intermediate

        gate_up_shape = tuple(self.weights[K3W.MOE_SHARED_GATE_UP].shape)
        down_shape = tuple(self.weights[K3W.MOE_SHARED_DOWN].shape)
        expected_gate_up = (gate_up_rows, hidden_size)
        expected_down = (down_rows, hidden_size)
        if gate_up_shape != expected_gate_up or down_shape != expected_down:
            raise ValueError(
                "K3 shared-expert storage layout does not match "
                "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD="
                f"{int(self.shared_expert_weight_shard)}: gate_up={gate_up_shape} "
                f"expected={expected_gate_up}, down={down_shape} "
                f"expected={expected_down}"
            )

    def _shared_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_gate_up_weight = self.weights[K3W.MOE_SHARED_GATE_UP]
        if self.shared_expert_weight_shard:
            shared_gate_up_weight = _transient_full_native_column_weight(
                shared_gate_up_weight,
                self.ffn_tp_size,
            )
        shared_gate_up = F.linear(hidden_states, shared_gate_up_weight)
        if self.shared_expert_weight_shard:
            del shared_gate_up_weight

        # Both halves are views into the single packed GEMM output.  The
        # strided SiTU kernel consumes them directly without materializing a
        # gate/up reorder or two contiguous activation copies.
        shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
        shared_activation = situ_and_mul(
            shared_gate,
            shared_up,
            self.beta,
            self.linear_beta,
        )
        del shared_gate, shared_up, shared_gate_up

        shared_down_weight = self.weights[K3W.MOE_SHARED_DOWN]
        if self.shared_expert_weight_shard:
            shared_down_weight = _transient_full_row_weight(
                shared_down_weight,
                self.ffn_tp_size,
            )
        shared_output = torch.matmul(shared_activation, shared_down_weight)
        if self.shared_expert_weight_shard:
            del shared_down_weight
        return shared_output

    @staticmethod
    def _packed_fp4_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.int8:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.view(torch.int8)
        raise TypeError(
            "K3 DeepGEMM packed expert weight must be uint8/int8, got "
            f"{tensor.dtype}"
        )

    @staticmethod
    def _ue8m0_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.float8_e8m0fnu:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.view(torch.float8_e8m0fnu)
        raise TypeError(
            "K3 DeepGEMM expert scale must be uint8/float8_e8m0fnu, got "
            f"{tensor.dtype}"
        )

    def _setup_deep_gemm_mega(self) -> None:
        """Transform K3's EP-local MXFP4 weights for SiTU MegaMoE."""

        max_tokens_per_rank = int(
            os.environ.get("MEGA_MOE_MAX_TOKENS_PER_RANK", "65536")
        )
        if max_tokens_per_rank <= 0:
            raise ValueError("MEGA_MOE_MAX_TOKENS_PER_RANK must be positive")

        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.models_py.modules.dsv4.moe.input_packer import (
            get_mega_moe_input_packer,
        )
        from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
            _get_or_create_mega_buf,
            _get_or_create_mega_output,
        )
        from rtp_llm.models_py.modules.dsv4.quant_layouts import (
            FP4_BLOCK,
            prepare_fp4_weight_scale_for_deepgemm,
        )

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError("K3 DeepGEMM MegaMoE requires an SM100+ CUDA GPU")
        if not dist.is_initialized():
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE requires torch.distributed initialization"
            )
        world_size = int(dist.get_world_size())
        if (
            self.ep_size != 8
            or self.attn_tp_size != 8
            or world_size != 8
            or self.local_expert_count != 112
        ):
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE is fixed to "
                "TP8/EP8/world8/112-local-experts; got "
                f"TP={self.attn_tp_size} EP={self.ep_size} world={world_size} "
                f"local_experts={self.local_expert_count}"
            )

        mega_signature = inspect.signature(deep_gemm.fp8_fp4_mega_moe)
        required_parameters = {
            "activation_beta",
            "activation_linear_beta",
            "fast_math",
        }
        missing_parameters = required_parameters.difference(mega_signature.parameters)
        if missing_parameters:
            raise RuntimeError(
                "Kimi K3 DeepGEMM mega resolved an old DeepGEMM "
                "without K3 SiTU support; missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )
        deep_gemm_path = getattr(deep_gemm, "__file__", "")

        st_w1_w = self.weights.pop(K3W.MOE_W1_PACKED)
        st_w1_s = self.weights.pop(K3W.MOE_W1_SCALE)
        st_w3_w = self.weights.pop(K3W.MOE_W3_PACKED)
        st_w3_s = self.weights.pop(K3W.MOE_W3_SCALE)
        if st_w1_w.ndim != 3 or st_w1_w.shape[0] != self.local_expert_count:
            raise RuntimeError(
                "unexpected K3 W1 stack for MegaMoE: " f"{tuple(st_w1_w.shape)}"
            )
        device = st_w1_w.device
        intermediate = int(st_w1_w.shape[1])
        if (
            self.latent_size != 3584
            or intermediate != 3072
            or int(st_w1_w.shape[2]) * 2 != self.latent_size
        ):
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE expects latent/intermediate=3584/3072; "
                f"got {self.latent_size}/{intermediate}"
            )

        expert_count = self.local_expert_count
        w13 = torch.empty(
            (expert_count, 2 * intermediate, self.latent_size // 2),
            dtype=torch.int8,
            device=device,
        )
        s13_raw = torch.empty(
            (expert_count, 2 * intermediate, self.latent_size // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w13[:, :intermediate].copy_(self._packed_fp4_view(st_w1_w))
        w13[:, intermediate:].copy_(self._packed_fp4_view(st_w3_w))
        s13_raw[:, :intermediate].copy_(self._ue8m0_view(st_w1_s))
        s13_raw[:, intermediate:].copy_(self._ue8m0_view(st_w3_s))
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13 = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw,
            2 * intermediate,
            self.latent_size,
            expert_count,
        )
        del s13_raw

        st_w2_w = self.weights.pop(K3W.MOE_W2_PACKED)
        st_w2_s = self.weights.pop(K3W.MOE_W2_SCALE)
        expected_w2_shape = (
            expert_count,
            self.latent_size,
            intermediate // 2,
        )
        if tuple(st_w2_w.shape) != expected_w2_shape:
            raise RuntimeError(
                "unexpected K3 W2 stack for MegaMoE: "
                f"{tuple(st_w2_w.shape)} != {expected_w2_shape}"
            )
        w2 = torch.empty(expected_w2_shape, dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (expert_count, self.latent_size, intermediate // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w2.copy_(self._packed_fp4_view(st_w2_w))
        s2_raw.copy_(self._ue8m0_view(st_w2_s))
        del st_w2_w, st_w2_s
        s2 = prepare_fp4_weight_scale_for_deepgemm(
            s2_raw,
            self.latent_size,
            intermediate,
            expert_count,
        )
        del s2_raw

        (self._mega_l1_w, self._mega_l1_sf), (
            self._mega_l2_w,
            self._mega_l2_sf,
        ) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13),
            (w2, s2),
            activation="situ",
        )
        del w13, s13, w2, s2

        self._mega_group = dist.group.WORLD
        self._mega_buf = _get_or_create_mega_buf(
            group=self._mega_group,
            num_experts=self.expert_num,
            num_max_tokens_per_rank=max_tokens_per_rank,
            num_topk=self.top_k,
            hidden=self.latent_size,
            intermediate_hidden=intermediate,
            use_fp8_dispatch=True,
            activation="situ",
        )
        output_capacity = max(
            max_tokens_per_rank,
            int(getattr(self._mega_buf, "num_max_tokens_per_rank", 0)),
        )
        self._mega_y = _get_or_create_mega_output(
            output_capacity,
            self.latent_size,
            torch.bfloat16,
            device,
        )
        self._mega_input_packer = get_mega_moe_input_packer()

        device_index = device.index if device.index is not None else 0
        if device_index not in _DEEPGEMM_MEGA_LOGGED_DEVICES:
            logging.info(
                "[KimiK3 DeepGEMM MegaMoE] enabled device=%s module=%s "
                "TP=%d EP=%d experts=%d local_experts=%d topk=%d "
                "latent=%d intermediate=%d max_tokens_per_rank=%d "
                "input_packer=%s",
                device,
                deep_gemm_path,
                self.attn_tp_size,
                self.ep_size,
                self.expert_num,
                self.local_expert_count,
                self.top_k,
                self.latent_size,
                intermediate,
                max_tokens_per_rank,
                self._mega_input_packer.name,
            )
            _DEEPGEMM_MEGA_LOGGED_DEVICES.add(device_index)

    def _deep_gemm_mega_expert_sum(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm

        token_count = int(routed_input.shape[0])
        capacity = int(self._mega_buf.num_max_tokens_per_rank)
        if token_count > capacity:
            raise RuntimeError(
                f"K3 MegaMoE tokens/rank={token_count} exceeds capacity={capacity}"
            )
        self._mega_input_packer.pack(
            routed_input,
            routing_weights,
            expert_ids,
            self._mega_buf,
            token_count,
        )
        output = self._mega_y[:token_count]
        deep_gemm.fp8_fp4_mega_moe(
            output,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, 32),
            activation="situ",
            activation_clamp=None,
            activation_beta=float(self.beta),
            activation_linear_beta=(
                None if self.linear_beta is None else float(self.linear_beta)
            ),
            fast_math=True,
        )
        return output

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep all router tensors resident in checkpoint BF16. The FP32
        # conversion is scoped to this layer invocation so 93 layers do not
        # retain about 1.10 GiB of extra allocator segments.
        router_weight = self.weights[K3W.MOE_GATE].float()
        router_logits = torch.matmul(hidden_states.float(), router_weight)
        correction_bias = self.weights[K3W.MOE_CORRECTION_BIAS].float()
        if self._group_topk.fused_sigmoid_supported(
            router_logits,
            correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.top_k,
        ):
            expert_weights = torch.empty(
                (hidden_states.shape[0], self.top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            expert_ids = torch.empty(
                (hidden_states.shape[0], self.top_k),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            self._group_topk.forward_fused_sigmoid(
                expert_weights,
                expert_ids,
                router_logits,
                correction_bias,
                self.top_k,
                self.renormalize,
                self.routed_scaling_factor,
            )
            return expert_ids, expert_weights

        scores = torch.sigmoid(router_logits)
        choice_scores = scores + correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            grouped = choice_scores.reshape(
                hidden_states.shape[0], self.num_expert_group, -1
            )
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected_groups = group_scores.topk(
                self.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = (
                group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice_scores)
            )
            choice_scores = choice_scores.masked_fill(~expert_mask, float("-inf"))
        expert_ids = choice_scores.topk(self.top_k, dim=-1, sorted=False).indices
        expert_weights = scores.gather(1, expert_ids)
        if self.top_k > 1 and self.renormalize:
            expert_weights = expert_weights / (
                expert_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return expert_ids, expert_weights * self.routed_scaling_factor

    def _tp_token_slice(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Avoid dispatching the same replicated TP tokens more than once."""

        token_count = routed_input.shape[0]
        tokens_per_tp_rank = (token_count + self.attn_tp_size - 1) // self.attn_tp_size
        begin = min(tokens_per_tp_rank * self.attn_tp_rank, token_count)
        size = min(tokens_per_tp_rank, token_count - begin)
        return (
            routed_input.narrow(0, begin, size),
            expert_ids.narrow(0, begin, size),
            routing_weights.narrow(0, begin, size),
            tokens_per_tp_rank,
        )

    def _tp_gather(
        self,
        output: torch.Tensor,
        original_token_count: int,
        tokens_per_tp_rank: int,
    ) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return output
        if output.shape[0] < tokens_per_tp_rank:
            output = torch.cat(
                (
                    output,
                    output.new_zeros(
                        tokens_per_tp_rank - output.shape[0], output.shape[1]
                    ),
                ),
                dim=0,
            )
        gathered = all_gather(output, group=Group.TP).reshape(
            self.attn_tp_size * tokens_per_tp_rank, -1
        )
        return gathered[:original_token_count]

    def _mega_expert_sum(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if sequence_parallel:
            return self._deep_gemm_mega_expert_sum(
                routed_input,
                expert_ids,
                routing_weights,
            )
        sliced_input, sliced_ids, sliced_weights, tokens_per_tp_rank = (
            self._tp_token_slice(routed_input, expert_ids, routing_weights)
        )
        local_output = self._deep_gemm_mega_expert_sum(
            sliced_input,
            sliced_ids,
            sliced_weights,
        )
        return self._tp_gather(
            local_output,
            routed_input.shape[0],
            tokens_per_tp_rank,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        sequence_parallel: bool = False,
        valid_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        sp_active = (
            sequence_parallel and self.attn_tp_size > 1 and hidden_states.is_cuda
        )
        expert_ids, routing_weights = self._route(hidden_states)
        if valid_token_count is not None:
            if valid_token_count < 0 or valid_token_count > hidden_states.shape[0]:
                raise ValueError(
                    "valid_token_count is outside the local token shard: "
                    f"valid={valid_token_count}, rows={hidden_states.shape[0]}"
                )
            if valid_token_count < hidden_states.shape[0]:
                expert_ids = expert_ids.clone()
                routing_weights = routing_weights.clone()
                # DeepGEMM validates every expert id before applying its
                # routing weight. Padding rows still need an in-range id;
                # zero weights and the output clear below keep them inert.
                expert_ids[valid_token_count:] = 0
                routing_weights[valid_token_count:] = 0
        routed_input = torch.matmul(
            hidden_states, self.weights[K3W.MOE_ROUTED_DOWN]
        )
        routed_output = self._mega_expert_sum(
            routed_input,
            expert_ids,
            routing_weights,
            sequence_parallel=sp_active,
        )
        if self.routed_norm is not None:
            routed_output = self.routed_norm(routed_output.contiguous())
        routed_output = torch.matmul(
            routed_output, self.weights[K3W.MOE_ROUTED_UP]
        )
        shared_output = self._shared_expert_forward(hidden_states)
        output = routed_output + shared_output
        if valid_token_count is not None and valid_token_count < hidden_states.shape[0]:
            output = output.clone()
            output[valid_token_count:] = 0
        return output

__all__ = [
    "KimiK3LatentMoE",
]

"""Kimi K3 MegaMoE strategy with the BF16 shared expert fused in DeepGEMM.

This module deliberately owns a separate setup, buffer cache and forward path
from :mod:`kimi_k3.moe`.  Both strategies call the public
``deep_gemm.fp8_fp4_mega_moe`` symbol, but only this class supplies the four
``shared_*`` tensors and consumes its independent shared output.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.modules.kimi_k3.input_packer_se import (
    get_kimi_k3_mega_moe_se_input_packer,
)
from rtp_llm.models_py.modules.kimi_k3.mega_se_buf import (
    get_or_create_kimi_k3_mega_moe_se_buf,
    get_or_create_kimi_k3_mega_moe_se_storages,
)
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.ops import ParallelismConfig

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

_DEEPGEMM_MEGA_SE_LOGGED_DEVICES: set[int] = set()


class KimiK3LatentMoESE(KimiK3LatentMoE):
    """Fuse K3 routed experts and the merged BF16 shared MLP in one kernel."""

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        # KimiK3LatentMoE.__init__ invokes _setup_deep_gemm_mega
        # polymorphically after all K3 routing/shared layout fields are ready.
        super().__init__(config, parallelism_config, weights, layer_idx)

    @staticmethod
    def _validate_deep_gemm_se_api(deep_gemm) -> None:
        parameters = list(inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters)
        required = {
            "activation_beta",
            "activation_linear_beta",
            "fast_math",
        }
        missing = required.difference(parameters)
        if missing:
            raise RuntimeError(
                "Kimi K3 mega_moe_se resolved an old DeepGEMM without SiTU "
                "support; missing parameters: " + ", ".join(sorted(missing))
            )
        shared_parameters = [
            "shared_x",
            "shared_y",
            "shared_l1_weights",
            "shared_l2_weights",
        ]
        try:
            shared_start = parameters.index("shared_x")
        except ValueError as error:
            raise RuntimeError(
                "Kimi K3 mega_moe_se requires DeepGEMM's shared_x/shared_y API"
            ) from error
        if parameters[shared_start : shared_start + 4] != shared_parameters:
            raise RuntimeError(
                f"unexpected DeepGEMM fused-SE parameter order: {parameters}"
            )
        if "num_shared_experts" in parameters:
            raise RuntimeError(
                "Kimi K3 mega_moe_se requires the count-free DeepGEMM execution API"
            )
        buffer_parameters = inspect.signature(
            deep_gemm.get_symm_buffer_for_mega_moe
        ).parameters
        if "shared_intermediate_hidden" not in buffer_parameters:
            raise RuntimeError(
                "Kimi K3 mega_moe_se requires DeepGEMM buffer support for "
                "shared_intermediate_hidden"
            )

    def _setup_deep_gemm_mega(self) -> None:
        """Prepare independent routed FP4 and shared BF16 fused-SE state."""

        max_tokens_per_rank = int(
            os.environ.get("MEGA_MOE_MAX_TOKENS_PER_RANK", "65536")
        )
        if max_tokens_per_rank <= 0:
            raise ValueError("MEGA_MOE_MAX_TOKENS_PER_RANK must be positive")
        if self.shared_expert_weight_shard:
            raise ValueError(
                "moe_strategy=mega_moe_se requires full shared-expert weights "
                "on every rank; set KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD=0"
            )

        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.models_py.modules.dsv4.quant_layouts import (
            FP4_BLOCK,
            prepare_fp4_weight_scale_for_deepgemm,
        )

        self._validate_mega_preconditions("K3 DeepGEMM MegaMoE SE")
        if self.top_k > 32:
            raise RuntimeError(
                f"K3 DeepGEMM MegaMoE SE requires topk <= 32, got {self.top_k}"
            )
        self._validate_deep_gemm_se_api(deep_gemm)

        st_w1_w = self.weights.pop(K3W.MOE_W1_PACKED)
        st_w1_s = self.weights.pop(K3W.MOE_W1_SCALE)
        st_w3_w = self.weights.pop(K3W.MOE_W3_PACKED)
        st_w3_s = self.weights.pop(K3W.MOE_W3_SCALE)
        if st_w1_w.ndim != 3 or int(st_w1_w.shape[0]) != self.local_expert_count:
            raise RuntimeError(
                f"unexpected K3 W1 stack for MegaMoE SE: {tuple(st_w1_w.shape)}"
            )
        device = st_w1_w.device
        intermediate = int(st_w1_w.shape[1])
        if (
            self.latent_size != 3584
            or intermediate != 3072
            or int(st_w1_w.shape[2]) * 2 != self.latent_size
        ):
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE SE expects routed latent/intermediate="
                f"3584/3072; got {self.latent_size}/{intermediate}"
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
                "unexpected K3 W2 stack for MegaMoE SE: "
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

        shared_l1 = self.weights.pop(K3W.MOE_SHARED_GATE_UP)
        shared_down_storage = self.weights.pop(K3W.MOE_SHARED_DOWN)
        shared_hidden = int(shared_l1.shape[1])
        shared_intermediate = self.shared_intermediate_size
        expected_shared_l1 = (2 * shared_intermediate, shared_hidden)
        expected_shared_down_storage = (shared_intermediate, shared_hidden)
        if (
            shared_l1.dtype != torch.bfloat16
            or shared_down_storage.dtype != torch.bfloat16
        ):
            raise TypeError(
                "K3 MegaMoE SE shared weights must be BF16, got "
                f"l1={shared_l1.dtype} down={shared_down_storage.dtype}"
            )
        if tuple(shared_l1.shape) != expected_shared_l1:
            raise RuntimeError(
                "unexpected K3 shared L1 for MegaMoE SE: "
                f"{tuple(shared_l1.shape)} != {expected_shared_l1}"
            )
        if tuple(shared_down_storage.shape) != expected_shared_down_storage:
            raise RuntimeError(
                "unexpected K3 shared down storage for MegaMoE SE: "
                f"{tuple(shared_down_storage.shape)} != "
                f"{expected_shared_down_storage}"
            )
        shared_l2 = shared_down_storage.transpose(0, 1).contiguous()
        del shared_down_storage
        self._mega_shared_l1_w, self._mega_shared_l2_w = (
            deep_gemm.transform_weights_for_mega_moe(
                shared_l1.contiguous(),
                shared_l2,
                activation="situ",
            )
        )
        del shared_l1, shared_l2

        self._mega_group = dist.group.WORLD
        self._mega_buf = get_or_create_kimi_k3_mega_moe_se_buf(
            group=self._mega_group,
            num_experts=self.expert_num,
            num_max_tokens_per_rank=max_tokens_per_rank,
            num_topk=self.top_k,
            hidden=self.latent_size,
            intermediate_hidden=intermediate,
            shared_intermediate_hidden=shared_intermediate,
            activation="situ",
        )
        capacity = int(self._mega_buf.num_max_tokens_per_rank)
        (
            self._mega_y,
            self._mega_shared_x,
            self._mega_shared_y,
        ) = get_or_create_kimi_k3_mega_moe_se_storages(
            capacity=capacity,
            routed_hidden=self.latent_size,
            shared_hidden=shared_hidden,
            device=device,
        )
        self._mega_input_packer = get_kimi_k3_mega_moe_se_input_packer()
        self._mega_shared_hidden = shared_hidden
        self._mega_shared_intermediate = shared_intermediate

        device_index = device.index if device.index is not None else 0
        if device_index not in _DEEPGEMM_MEGA_SE_LOGGED_DEVICES:
            logging.info(
                "[KimiK3 DeepGEMM MegaMoE SE] enabled device=%s module=%s "
                "TP=%d EP=%d experts=%d local_experts=%d topk=%d "
                "routed=%d/%d shared=%d/%d capacity=%d input_packer=%s",
                device,
                getattr(deep_gemm, "__file__", ""),
                self.attn_tp_size,
                self.ep_size,
                self.expert_num,
                self.local_expert_count,
                self.top_k,
                self.latent_size,
                intermediate,
                shared_hidden,
                shared_intermediate,
                capacity,
                self._mega_input_packer.name,
            )
            _DEEPGEMM_MEGA_SE_LOGGED_DEVICES.add(device_index)

    def _deep_gemm_mega_expert_sum_with_shared(
        self,
        routed_input: torch.Tensor,
        shared_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import deep_gemm

        token_count = int(routed_input.shape[0])
        capacity = int(self._mega_buf.num_max_tokens_per_rank)
        if token_count > capacity:
            raise RuntimeError(
                "K3 MegaMoE SE tokens/rank="
                f"{token_count} exceeds capacity={capacity}"
            )
        if tuple(shared_input.shape) != (token_count, self._mega_shared_hidden):
            raise RuntimeError(
                "K3 MegaMoE SE shared input shape mismatch: "
                f"got={tuple(shared_input.shape)} expected="
                f"({token_count}, {self._mega_shared_hidden})"
            )
        self._mega_input_packer.pack(
            routed_input,
            routing_weights,
            expert_ids,
            self._mega_buf,
            token_count,
        )
        self._mega_shared_x[:token_count].copy_(shared_input)
        routed_output = self._mega_y[:token_count]
        deep_gemm.fp8_fp4_mega_moe(
            routed_output,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_x=self._mega_shared_x,
            shared_y=self._mega_shared_y,
            shared_l1_weights=self._mega_shared_l1_w,
            shared_l2_weights=self._mega_shared_l2_w,
            recipe=(1, 1, 32),
            activation="situ",
            activation_clamp=None,
            activation_beta=float(self.beta),
            activation_linear_beta=(
                None if self.linear_beta is None else float(self.linear_beta)
            ),
            fast_math=True,
        )
        return routed_output, self._mega_shared_y[:token_count]

    def _mega_expert_sum_with_shared(
        self,
        routed_input: torch.Tensor,
        shared_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_parallel:
            return self._deep_gemm_mega_expert_sum_with_shared(
                routed_input,
                shared_input,
                expert_ids,
                routing_weights,
            )

        token_count = int(routed_input.shape[0])
        tokens_per_tp_rank = (token_count + self.attn_tp_size - 1) // self.attn_tp_size
        begin = min(tokens_per_tp_rank * self.attn_tp_rank, token_count)
        size = min(tokens_per_tp_rank, token_count - begin)
        local_routed, local_shared = self._deep_gemm_mega_expert_sum_with_shared(
            routed_input.narrow(0, begin, size),
            shared_input.narrow(0, begin, size),
            expert_ids.narrow(0, begin, size),
            routing_weights.narrow(0, begin, size),
        )
        return (
            self._tp_gather(local_routed, token_count, tokens_per_tp_rank),
            self._tp_gather(local_shared, token_count, tokens_per_tp_rank),
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
                expert_ids[valid_token_count:] = 0
                routing_weights[valid_token_count:] = 0

        routed_input = torch.matmul(
            hidden_states,
            self.weights[K3W.MOE_ROUTED_DOWN],
        )
        routed_output, shared_output = self._mega_expert_sum_with_shared(
            routed_input,
            hidden_states,
            expert_ids,
            routing_weights,
            sequence_parallel=sp_active,
        )
        if self.routed_norm is not None:
            routed_output = self.routed_norm(routed_output.contiguous())
        routed_output = torch.matmul(
            routed_output,
            self.weights[K3W.MOE_ROUTED_UP],
        )
        output = routed_output + shared_output
        if valid_token_count is not None and valid_token_count < hidden_states.shape[0]:
            output = output.clone()
            output[valid_token_count:] = 0
        return output


__all__ = ["KimiK3LatentMoESE"]

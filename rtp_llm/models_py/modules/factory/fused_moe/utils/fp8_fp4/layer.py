"""Generic FP8-activation/FP4-weight MoE layer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    W13SharedExpert,
    combine_routed_and_shared,
    get_shared_expert_executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_range,
)
from rtp_llm.utils.model_weight import W


class Fp8Fp4MoeRuntimeConfig:
    """Configuration surface consumed by the common Router/Executor API."""

    moe_quant_method = "FP8_FP4"
    activation_type = "silu"
    enable_cuda_graph = False
    use_mori_ep = False
    use_deepep_moe = False
    ll_num_max_token = 0
    masked_max_token_num = 0

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        moe_inter_dim: int,
        expert_num: int,
        moe_k: int,
        n_shared_experts: int,
        swiglu_limit: float,
        ep_size: int,
        ep_rank: int,
        max_tokens_per_rank: int,
        moe_strategy: str,
        world_size: Optional[int] = None,
        world_rank: Optional[int] = None,
        model_type: str = "generic_fp8_fp4_moe",
        has_shared_expert_gate: bool = False,
        warmup_include_capacity: bool = False,
    ) -> None:
        if expert_num % max(ep_size, 1) != 0:
            raise ValueError(
                f"expert_num={expert_num} must be divisible by ep_size={ep_size}"
            )
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.dim = self.hidden_size
        self.moe_inter_dim = int(moe_inter_dim)
        self.expert_num = int(expert_num)
        self.n_routed_experts = self.expert_num
        self.moe_k = int(moe_k)
        self.n_activated_experts = self.moe_k
        self.n_shared_experts = int(n_shared_experts)
        self.has_shared_expert_gate = bool(has_shared_expert_gate)
        self.warmup_include_capacity = bool(warmup_include_capacity)
        self.swiglu_limit = float(swiglu_limit)
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.n_local_experts = self.expert_num // max(self.ep_size, 1)
        self.local_expert_start = self.ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.moe_strategy = str(moe_strategy or "auto")
        self.world_size = int(ep_size if world_size is None else world_size)
        self.world_rank = int(ep_rank if world_rank is None else world_rank)

        self.tp_size = 1
        self.tp_rank = 0
        self.dp_size = self.ep_size
        self.dp_rank = self.ep_rank
        self.local_rank = self.world_rank
        self.data_type = torch.bfloat16
        self.head_num = 0
        self.moe_topk_group = 0
        self.quant_config = None
        self.model_config = SimpleNamespace(
            model_type=model_type,
            quant_config=None,
        )
        self.parallelism_config = SimpleNamespace(tp_size=1)
        self.moe_config = SimpleNamespace(
            moe_strategy=self.moe_strategy,
            use_deepep_low_latency=False,
        )


class Fp8Fp4MoeLayer(nn.Module):
    """Route tokens and execute canonical FP8/FP4 MoE weights via the factory."""

    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str,
        route_scale: float,
        swiglu_limit: float,
        n_hash_layers: int,
        vocab_size: int,
        layer_weights: Dict[str, torch.Tensor],
        ep_size: int = 1,
        ep_rank: int = 0,
        world_size: Optional[int] = None,
        world_rank: Optional[int] = None,
        max_tokens_per_rank: int = 8192,
        strategy: str = "auto",
        model_type: str = "generic_fp8_fp4_moe",
        has_shared_expert_gate: bool = False,
        warmup_include_capacity: bool = False,
    ) -> None:
        super().__init__()
        if n_shared_experts < 0:
            raise ValueError(
                f"n_shared_experts must be non-negative, got {n_shared_experts}"
            )
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.n_shared_experts = int(n_shared_experts)
        self.gate = Gate(
            layer_id,
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func,
            route_scale,
            n_hash_layers,
            vocab_size,
            layer_weights=layer_weights,
        )
        runtime_config = Fp8Fp4MoeRuntimeConfig(
            layer_id=layer_id,
            hidden_size=dim,
            moe_inter_dim=moe_inter_dim,
            expert_num=n_routed_experts,
            moe_k=n_activated_experts,
            n_shared_experts=n_shared_experts,
            swiglu_limit=swiglu_limit,
            ep_size=ep_size,
            ep_rank=ep_rank,
            world_size=world_size,
            world_rank=world_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            moe_strategy=strategy,
            model_type=model_type,
            has_shared_expert_gate=has_shared_expert_gate,
            warmup_include_capacity=warmup_include_capacity,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(
            runtime_config, layer_weights
        )
        self.strategy_name = self.fused_moe.strategy_name
        # MegaMoE formerly fused hash routing with input packing. Preserve its
        # BF16-score/Triton epilogue numerics while keeping routing in the
        # common fused-MoE layer.
        self.gate.fuse_hash_gate = self.strategy_name in {
            "mega_moe",
            "mega_moe_se",
        }

        if self.fused_moe.includes_shared_expert or n_shared_experts == 0:
            self.shared_experts = None
            self._shared_executor = None
        else:
            shared_inter_dim = moe_inter_dim * n_shared_experts
            self.shared_experts = W13SharedExpert(
                dim,
                shared_inter_dim,
                expert_weights={
                    "w13_w": layer_weights[W.ffn_w13],
                    "w13_s": layer_weights[W.ffn_s13],
                    "w2_w": layer_weights[W.ffn_w2],
                    "w2_s": layer_weights[W.ffn_s2],
                },
                swiglu_limit=swiglu_limit,
            )
            self._shared_executor = get_shared_expert_executor(
                max_tokens_per_rank=max_tokens_per_rank,
                dim=dim,
                inter_dim=shared_inter_dim,
                swiglu_limit=swiglu_limit,
            )
            self._shared_executor.prepare(self.shared_experts)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        observer: Optional[Callable[[str, torch.Tensor], None]] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out_flat = None
        if out is not None:
            if out.shape != shape:
                raise ValueError(
                    f"output shape {tuple(out.shape)} does not match input "
                    f"shape {tuple(shape)}"
                )
            if out.dtype != x.dtype or out.device != x.device:
                raise ValueError(
                    "output buffer must use the same dtype and device as the input"
                )
            if not out.is_contiguous():
                raise ValueError("output buffer must be contiguous")
            out_flat = out.view(-1, self.dim)

        if observer is not None:
            observer("input", x)
        flat_ids = None if input_ids is None else input_ids.reshape(-1)
        if flat_ids is not None and flat_ids.numel() != x.size(0):
            raise ValueError(
                f"input_ids has {flat_ids.numel()} tokens, expected {x.size(0)}"
            )
        gate_payload = None
        topk_weights = None
        topk_ids = None
        use_gate_pack = (
            observer is None
            and self.fused_moe.supports_gate_pack
            and self.gate.can_prepare_gate_payload(x, flat_ids)
        )

        def prepare_routes() -> None:
            nonlocal gate_payload, topk_weights, topk_ids
            with record_function_range("moe.gate"):
                if use_gate_pack:
                    gate_payload = self.gate.prepare_gate_payload(x, flat_ids)
                    if gate_payload is None:
                        raise RuntimeError(
                            "fused gate packing became unavailable after selection"
                        )
                else:
                    topk_weights, topk_ids = self.gate(x, flat_ids)
            if observer is not None and not use_gate_pack:
                assert topk_weights is not None and topk_ids is not None
                observer("topk_weights", topk_weights)
                observer("topk_indices", topk_ids)

        def run_routed_experts() -> torch.Tensor:
            if gate_payload is not None:
                return self.fused_moe.forward_gate_pack(
                    hidden_states=x,
                    gate_payload=gate_payload,
                    activation="silu",
                )
            assert topk_weights is not None and topk_ids is not None
            return self.fused_moe(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation="silu",
            )

        if self._shared_executor is None:
            prepare_routes()
            with record_function_range("moe.routed_experts"):
                output = run_routed_experts()
            if observer is not None:
                observer("routed_y", output)
            with record_function_range("moe.add_shared"):
                output = output.to(dtype=x.dtype)
                if out_flat is not None:
                    out_flat.copy_(output)
                    output = out_flat
                output = output.view(shape)
            if observer is not None:
                observer("final_y", output.reshape(-1, self.dim))
            return output

        # Match the original MegaMoE scheduling: its standalone shared expert
        # starts before the BF16 gate GEMM and fused route/input pack, maximizing
        # overlap. Ordinary gate paths keep gate selection ahead of shared work.
        if not use_gate_pack:
            prepare_routes()
        with record_function_range("moe.shared_expert_start"):
            self._shared_executor.start(self.shared_experts, x)
        try:
            if use_gate_pack:
                prepare_routes()
            with record_function_range("moe.routed_experts"):
                routed = run_routed_experts()
        except Exception:
            with record_function_range("moe.shared_expert_finish"):
                self._shared_executor.finish()
            raise
        if observer is not None:
            observer("routed_y", routed)
        with record_function_range("moe.shared_expert_finish"):
            shared = self._shared_executor.finish()
        if observer is not None:
            observer("shared_y", shared)
        with record_function_range("moe.add_shared"):
            output = combine_routed_and_shared(
                routed, shared, x.dtype, out=out_flat
            ).view(shape)
        if observer is not None:
            observer("final_y", output.reshape(-1, self.dim))
        return output

"""PPU shared-compute ops assembled behind the frozen V4.1 backend contract.

`PPUOps` implements `V41BackendOps` (linear, moe, index_score,
sparse_attention, encode_kv, decode_kv, engram_lookup, mhc) by delegating to
the sibling modules in this package. Weight state enters through the bind
methods: routed-expert FP4 shards per layer, and Engram host shards per
Engram layer. The ops object is the single stateful seam the model class
(L2/L3 wiring) talks to; kernels stay in the sibling modules.

All top-level imports are torch-only; accelerator libraries stay lazy inside
the sibling modules, so this module imports on a CPU host.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
from rtp_llm.models_py.modules.dsv41.backend import (
    V41BackendOps,
    V41MHCResult,
    V41QuantizedTensor,
    V41StepContext,
)

from .attention import sparse_attention as _sparse_attention
from .engram import (
    DistributedEnpmGroup,
    EngramShardLayout,
    HostShard,
    LocalEnpmGroup,
)
from .engram import engram_lookup as _engram_lookup
from .indexer import index_score as _index_score
from .kv_codec import decode_kv as _decode_kv
from .kv_codec import encode_kv as _encode_kv
from .linear import PPUResidentBF16Linear, dequant_fp8_block32_ue8m0
from .mhc import PPUMHC, MHCConfig, PPUStepMix
from .mhc_mega import PPUMegaMHC
from .moe import (
    PPUMoE,
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)

_SWIGLU_LIMIT = 10.0


class PPUOps(V41BackendOps):
    """Stateful assembly of the eight contract ops plus weight bindings.

    ``ep_size == 1`` keeps everything single-process (module tests, CPU
    diagnostics); the integrated P/D servers construct it with their EP/ENP
    groups.
    """

    def __init__(
        self,
        *,
        ep_rank: int = 0,
        ep_size: int = 1,
        num_routed_experts: int = 384,
        num_physical_experts: Optional[int] = None,
        num_local_experts: int = 48,
        hidden_size: int = 5120,
        inter_dim: int = 2048,
        swiglu_limit: float = _SWIGLU_LIMIT,
        mhc_config: Optional[MHCConfig] = None,
        enpm_group: Optional[object] = None,
        deepep_group: Optional[object] = None,
        sm_scale: Optional[float] = None,
        window: int = 128,
    ) -> None:
        self._moe = PPUMoE(
            num_routed_experts=num_routed_experts,
            num_physical_experts=num_physical_experts,
            num_local_experts=num_local_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            hidden_size=hidden_size,
            inter_dim=inter_dim,
            swiglu_limit=swiglu_limit,
            group=deepep_group,
        )
        self._mhc = PPUMHC(mhc_config)
        self._mega_mhc = PPUMegaMHC()
        self._enpm_group = enpm_group
        self._engram: dict[int, Tuple[HostShard, EngramShardLayout, object]] = {}
        self._sm_scale = sm_scale
        self._window = int(window)
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)

    # ------------------------------------------------------------------ bindings

    def bind_expert_weights(
        self,
        layer_id: int,
        w13_pair: Tuple[torch.Tensor, torch.Tensor],
        w2_pair: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bind one layer's local FP4 routed-expert shard (see PPUMoE)."""
        self._moe.bind_expert_weights(layer_id, w13_pair, w2_pair)

    def bind_engram_shard(
        self,
        layer_id: int,
        shard: HostShard,
        layout: EngramShardLayout,
        group: Optional[object] = None,
    ) -> None:
        """Bind one Engram layer's host shard, layout and lookup group."""
        self._engram[int(layer_id)] = (shard, layout, group or self._enpm_group)

    def bind_mhc_config(self, config: MHCConfig) -> None:
        self._mhc = PPUMHC(config)

    @property
    def moe_engine(self) -> PPUMoE:
        """The :class:`PPUMoE` instance (``moe`` itself is the contract op)."""
        return self._moe

    # ------------------------------------------------------------------ contract ops

    def linear(
        self,
        x: torch.Tensor,
        weight: V41QuantizedTensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = weight.values
        if weight.format == "bf16":
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
        elif weight.format == "fp8_32x32_ue8m0":
            # Per-call dequant keeps this path usable for fixtures and
            # diagnostics; production binds resident BF16 once through
            # PPUResidentBF16Linear.from_quantized.
            w = dequant_fp8_block32_ue8m0(w, weight.scales)
        else:
            raise ValueError(
                f"unsupported dense linear weight format {weight.format!r}; "
                "expected 'bf16' or 'fp8_32x32_ue8m0' (fp4 is routed-expert only)"
            )
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        return torch.nn.functional.linear(x, w, bias)

    def moe(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        *,
        layer_id: int,
        context: Optional[V41StepContext] = None,
        expert_load=None,
    ) -> torch.Tensor:
        return self._moe.forward(
            x, expert_ids, expert_weights, layer_id=layer_id, expert_load=expert_load
        )

    def index_score(
        self,
        q: V41QuantizedTensor,
        k: V41QuantizedTensor,
        head_weights: torch.Tensor,
        *,
        layer_id: int,
        candidate_blocks: torch.Tensor,
        key_positions: torch.Tensor,
        context: Optional[V41StepContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        return _index_score(
            q,
            k,
            head_weights,
            layer_id=layer_id,
            candidate_blocks=candidate_blocks,
            key_positions=key_positions,
            context=context,
            **kwargs,
        )

    def sparse_attention(
        self,
        q: torch.Tensor,
        selected_blocks: torch.Tensor,
        *,
        cache,
        layer_id: int,
        context: Optional[V41StepContext] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._sm_scale is not None and "sm_scale" not in kwargs:
            kwargs["sm_scale"] = self._sm_scale
        if "window" not in kwargs:
            kwargs["window"] = self._window
        return _sparse_attention(
            q,
            selected_blocks,
            cache=cache,
            layer_id=layer_id,
            context=context,
            **kwargs,
        )

    def encode_kv(
        self, kv: torch.Tensor, *, pool_tag: str, descriptor: object = None
    ) -> torch.Tensor:
        return _encode_kv(kv, pool_tag=pool_tag, descriptor=descriptor)

    def decode_kv(
        self,
        packed: torch.Tensor,
        *,
        pool_tag: str,
        descriptor: object = None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return _decode_kv(packed, pool_tag=pool_tag, descriptor=descriptor, out=out)

    def engram_lookup(
        self,
        rows: torch.Tensor,
        *,
        layer_id: int,
        context: Optional[V41StepContext] = None,
    ) -> torch.Tensor:
        binding = self._engram.get(int(layer_id))
        if binding is None:
            raise RuntimeError(
                f"no Engram shard bound for layer {layer_id}; call "
                "bind_engram_shard first"
            )
        shard, layout, group = binding
        if (
            group is not None
            and hasattr(group, "lookup")
            and not hasattr(group, "all_gather_ids")
        ):
            # Multi-rank simulator (LocalEnpmGroup): this rank's request sits
            # at its own rank slot; the other slots stay empty.
            per_rank = [None] * group.size
            per_rank[shard.rank] = rows
            return group.lookup(per_rank)[shard.rank]
        return _engram_lookup(
            rows, layer_id=int(layer_id), shard=shard, layout=layout, group=group
        )

    def mhc(
        self,
        residual: torch.Tensor,
        branch: Optional[torch.Tensor],
        *,
        phase: str,
        parameters: Optional[Mapping[str, torch.Tensor]] = None,
        context: Optional[V41StepContext] = None,
        state: Optional[V41MHCResult] = None,
    ) -> PPUStepMix:
        return self._mhc(
            residual,
            branch,
            phase=phase,
            parameters=parameters,
            context=context,
            state=state,
        )

    def try_mhc_ffn_transition(
        self, branch, residual, state, parameters, norm, eps, *, phase
    ):
        """PPU-only seam; normalized output is separate from legacy mHC.hidden."""
        return self._mega_mhc.try_transition(
            branch,
            residual,
            state,
            parameters,
            norm,
            eps,
            self._mhc.config,
            phase=phase,
        )

    def mhc_execution_stats(self):
        return self._mega_mhc.execution_stats()


# Model-side helpers re-exported for the L2/L3 wiring convenience; the ops
# methods above remain the contract surface.
__all__ = [
    "PPUOps",
    "PPUResidentBF16Linear",
    "PPUStepMix",
    "MHCConfig",
    "EngramShardLayout",
    "HostShard",
    "LocalEnpmGroup",
    "DistributedEnpmGroup",
    "sqrtsoftplus_gate",
    "pad_topk_for_dispatch",
    "shared_expert_forward",
    "combine_routed_and_shared",
]

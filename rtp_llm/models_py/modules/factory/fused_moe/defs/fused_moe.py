import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, final

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)

logger = logging.getLogger(__name__)

# One-shot guard so the warmup MoE-activation measurement logs once (all layers are
# identical and sequential, so the first one carries the signal).
_MOE_ACT_LOGGED = False

# Executors whose memory scales with received ROWS (coverage) rather than local SLOTS.
# Single source of truth for the row-vs-slot branch in warmup skew (see _warmup_skew_params).
_ROW_BASED_EXECUTORS = {"TritonFusedMoeExecutor"}


def _in_memory_trace() -> bool:
    """True only during a warmup forward being memory-traced (C++ setTraceMemory(true)).

    Lazy import: compute_ops is a compiled extension; importing at call time keeps this
    module importable in environments where the op lib is absent (e.g. pure unit tests).
    """
    try:
        from rtp_llm.ops.compute_ops import is_trace_memory

        return bool(is_trace_memory())
    except Exception:
        return False


def _skew_reserve(mean: float) -> float:
    """Affine reserve over the uniform-routing mean, to cover systematic hot-expert skew.

        reserve = min(1.0, mean * mult + add)     # mult=MOE_SKEW_MULT, add=MOE_SKEW_ADD

    Why affine (not pure multiplicative, not σ):
    - We dropped the old 4σ/√T term: warmup always runs at a large batch (T = ep *
      max_batch_tokens) where 4σ/√T ≈ 0, and σ models *sampling noise* (vanishes with T),
      not *hot experts* (a systematic bias that does NOT vanish with T). See §4 / §6.2:
      online row_share hit 0.98 vs uniform P_hit=0.91, slot_share 0.35 vs 1/ep=0.25.
    - `* mult` covers the proportional imbalance (bigger load → bigger absolute wobble).
    - `+ add` is a non-vanishing floor: a single hot expert contributes an absolute share
      that does NOT shrink with 1/ep, so a pure multiplier under-reserves at high ep. This
      is the §9.3 granularity floor (w_max) made explicit.

    Defaults mult=1.3, add=0.1 cover the measured worst case (slot 1.41x/ep at ep=4) with
    margin. With EPLB on, real imbalance is smaller and both knobs can be lowered.
    """
    try:
        mult = float(os.environ.get("MOE_SKEW_MULT", "1.3"))
        add = float(os.environ.get("MOE_SKEW_ADD", "0.1"))
    except Exception:
        mult, add = 1.3, 0.1
    return min(1.0, mean * mult + add)


def _default_skew_fraction(ep_size: int, expert_num: int, top_k: int) -> float:
    """Row-based executor coverage (TritonFusedMoeExecutor): memory ∝ recv_rows.

        P_hit = 1 - C(E - E/ep, top_k) / C(E, top_k)   # uniform-routing mean coverage
        q     = min(1.0, P_hit * mult + add)           # +hot-expert reserve, capped at 1.0

    recv_rows can never exceed the cluster token count, so q saturates at 1.0.
    """
    if ep_size <= 1:
        return 1.0
    if top_k <= 0:
        return 1.0 / ep_size
    if expert_num > 1 and expert_num % ep_size == 0:
        n_local = expert_num // ep_size
        p_hit = 1.0 - math.comb(expert_num - n_local, top_k) / math.comb(
            expert_num, top_k
        )
    else:
        p_hit = 1.0 - (1.0 - 1.0 / ep_size) ** top_k
    return _skew_reserve(p_hit)


def _default_slot_share(ep_size: int, expert_num: int, top_k: int) -> float:
    """Slot-based executor share (DeepGemmHybrid, Cutlass, etc.): memory ∝ local slots.

        C(ep) = min(1.0, (1/ep) * mult + add)   # uniform 1/ep + hot-expert reserve

    Unlike the row model there is no natural saturation below 1.0, so the reserve is the
    only protection against a hot rank — this is exactly the case §6.2 showed 1/ep under-provisions.
    """
    if ep_size <= 1:
        return 1.0
    if expert_num <= top_k:
        return 1.0
    return _skew_reserve(1.0 / ep_size)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expected_m: Optional[int] = None
    expert_num_tokens: Optional[torch.Tensor] = None
    expert_num_tokens_cpu: Optional[Union[List[int], torch.Tensor]] = None


@dataclass
class ExpertForwardPayload:
    """
    Represents the data payload dispatched to experts for computation.
    """

    expert_x: torch.Tensor
    expert_x_origin_dtype: Optional[torch.dtype] = None
    expert_x_scale: Optional[torch.Tensor] = None
    expert_tokens_meta: Optional[ExpertTokensMetadata] = None
    expert_topk_ids: Optional[torch.Tensor] = None
    expert_topk_weights: Optional[torch.Tensor] = None
    expert_ids_are_local: bool = False


@dataclass
class CombineForwardPayload:
    """
    Represents the data payload for combining the expert outputs.
    """

    fused_expert_output: torch.Tensor


class FusedMoeDataRouter(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize FusedMoeDataRouter with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
        """
        self.config = config
        self.quant_config = quant_config

    @classmethod
    def router_type(cls) -> RouterType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this router can handle the given configuration.

        Subclasses should override this method to check router-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError


class FusedMoeExpertExecutor(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize FusedMoeExpertExecutor with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
            weights: Model weights dictionary
        """
        self.config = config
        self.quant_config = quant_config
        self.weights = weights

    @classmethod
    def executor_type(cls) -> ExecutorType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this executor can handle the given configuration.

        Subclasses should override this method to check executor-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        pass

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    @abstractmethod
    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        raise NotImplementedError


@final
class FusedMoe(torch.nn.Module):
    def __init__(
        self,
        router: FusedMoeDataRouter,
        fused_experts: FusedMoeExpertExecutor,
        expert_num: int,
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts
        self.expert_num = expert_num

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

    def _warmup_skew_params(
        self, ep_size: int, expert_num: int, topk: int
    ) -> Tuple[bool, float, int, str]:
        """Single source of truth for the warmup worst-case routing model.

        Returns (is_row_based, q, s, mode), branched by executor memory bottleneck:

        1. Row-based (TritonFusedMoeExecutor): memory ∝ recv_rows × top_k.
           The executor keeps all top_k slots per recv token and zeros non-local weights,
           so the bottleneck is how many ROWS arrive (coverage).
           → q = P_hit-reserve,  s = 1  (1 rank-0 slot per hot token)

        2. Slot-based (DeepGemmHybrid, Cutlass, etc.): memory ∝ local_slots.
           The executor uses ep_scatter to collect only local (token, expert) pairs,
           so the bottleneck is the total number of local SLOTS.
           → q = C(ep)-reserve,  s = top_k  (all slots on rank-0 per hot token)

        q is clamped to [1/ep, 1]. Tune the reserve via MOE_SKEW_MULT / MOE_SKEW_ADD
        (see _skew_reserve). Used by both _warmup_skew_topk_ids and the [MOE_ACT] log.
        """
        is_row_based = type(self.fused_experts).__name__ in _ROW_BASED_EXECUTORS
        if is_row_based:
            q = _default_skew_fraction(ep_size, expert_num, topk)
            s = 1
            mode = "row-based(P_hit x h)"
        else:
            q = _default_slot_share(ep_size, expert_num, topk)
            s = topk
            mode = "slot-based(1/ep x h)"
        q = max(1.0 / ep_size, min(1.0, q))
        return is_row_based, q, s, mode

    def _warmup_skew_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Rewrite topk_ids to the worst-case routing measured during warmup.

        See _warmup_skew_params for the memory model behind q (hot-token fraction) and
        s (rank-0 slots per hot token).
        """
        router = self.router
        ep_size = int(getattr(router, "ep_size", 1))
        n_local = getattr(router, "expert_num_per_rank", None)
        if ep_size <= 1 or not n_local:
            return topk_ids
        n_local = int(n_local)

        num_tokens, topk = topk_ids.shape[0], topk_ids.shape[1]
        expert_num = ep_size * n_local

        _, q, s, _ = self._warmup_skew_params(ep_size, expert_num, topk)
        n_hot = int(round(num_tokens * q))
        dev, dt = topk_ids.device, topk_ids.dtype
        cols = torch.arange(topk, device=dev)

        off_rank = 1 + (cols % (ep_size - 1))
        off_ids = (off_rank * n_local + (cols % n_local)).clamp_(max=expert_num - 1)

        out = topk_ids.clone()
        if n_hot > 0:
            hot = off_ids.to(dt).unsqueeze(0).repeat(n_hot, 1)
            r0_idx = torch.arange(n_hot, device=dev)
            for j in range(s):
                hot[:, j] = ((r0_idx * s + j) % n_local).to(dt)
            out[:n_hot] = hot
        n_cold = num_tokens - n_hot
        if n_cold > 0:
            out[n_hot:] = off_ids.to(dt).unsqueeze(0)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:

        a1 = hidden_states

        # Warmup-only: force worst-case skew so the measured peak covers the skewed case,
        # and measure the real MoE-section activation (no reset_peak_memory_stats -- that
        # would corrupt the C++ reserved-peak high-water mark read at end of forward).
        _trace = _in_memory_trace()
        if _trace:
            topk_ids = self._warmup_skew_topk_ids(topk_ids)
            _moe_act_m0 = torch.cuda.max_memory_allocated()

        expert_payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_weights,
            topk_ids,
        )

        if expert_payload.expert_topk_ids is None:
            expert_payload.expert_topk_ids = topk_ids
        if expert_payload.expert_topk_weights is None:
            expert_payload.expert_topk_weights = topk_weights

        if expert_payload.expert_x.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            combine_payload = CombineForwardPayload(
                fused_expert_output=torch.empty_like(
                    expert_payload.expert_x, dtype=a1.dtype
                )
            )
        else:
            combine_payload = self.fused_experts.execute(
                expert_payload,
                activation=activation,
                expert_map=expert_map,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
            )

        # pass a1.shape to finalize for shape check
        if extra_finalize_args is None:
            extra_finalize_args = {"a1_shape": a1.shape}
        else:
            extra_finalize_args.update({"a1_shape": a1.shape})

        extra_finalize_args.update({"original_num_tokens": hidden_states.size(0)})

        output = self.router.finalize(
            combine_payload,
            expert_payload.expert_topk_weights,
            expert_payload.expert_topk_ids,
            apply_router_weight_on_input,
            extra_finalize_args,
        )

        assert (
            output.shape == hidden_states.shape
        ), f"output batch size mismatch: expected {hidden_states.shape}, got {output.shape}"

        if _trace:
            global _MOE_ACT_LOGGED
            if not _MOE_ACT_LOGGED:
                # expert_x.shape[0] = tokens dispatched to this rank (worst-case: every rank's
                # tokens, ~ep_size*T_local). Each expands to top_k expert-rows inside the executor,
                # so the actual grouped-GEMM all_tokens ~= dispatched * top_k.
                dispatched = (
                    int(expert_payload.expert_x.shape[0])
                    if expert_payload.expert_x is not None
                    and expert_payload.expert_x.numel() > 0
                    else 0
                )
                topk = topk_ids.shape[1] if topk_ids.dim() == 2 else 1
                all_tokens = dispatched * topk
                ep = int(getattr(self.router, "ep_size", 1))
                n_local = int(getattr(self.router, "expert_num_per_rank", 0) or 0)
                if ep > 1 and n_local:
                    _, skew_p, _, mode = self._warmup_skew_params(
                        ep, ep * n_local, topk
                    )
                else:
                    skew_p = 1.0
                    mode = "single-rank"
                peak = max(0, torch.cuda.max_memory_allocated() - _moe_act_m0)
                per_row = peak // all_tokens if all_tokens else 0
                logger.warning(
                    "[MOE_ACT] warmup %s q=%.3f: ep=%s dispatched_tokens=%d "
                    "all_tokens(~x top_k=%d)=%d moe_section_peak=%.1f MiB per_expert_row~=%dB",
                    mode,
                    skew_p,
                    ep,
                    dispatched,
                    topk,
                    all_tokens,
                    peak / (1024 * 1024),
                    per_row,
                )
                _MOE_ACT_LOGGED = True

        return output

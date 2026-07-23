import logging
import os
import sys
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
from rtp_llm.utils.pre_import_config import warmup_requested

logger = logging.getLogger(__name__)

try:
    from rtp_llm.ops.compute_ops import is_trace_memory as _IS_TRACE_MEMORY

    _TRACE_MEMORY_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, AttributeError) as e:
    _IS_TRACE_MEMORY = None
    _TRACE_MEMORY_IMPORT_ERROR = e

_NONTORCH_BASELINE_MIB: Optional[float] = None
_NONTORCH_RUNTIME_PEAK_MIB = 0.0
_WARMUP_SKEW_LOGGED = False

# Startup warmup is a one-shot lifecycle. Once Python has observed the C++ trace transition from
# active back to inactive, serving forwards can stop crossing the pybind boundary permanently.
_WARMUP_ENABLED = warmup_requested(sys.argv[1:], os.environ)
_TRACE_MEMORY_SEEN = False
_TRACE_MEMORY_FINISHED = not _WARMUP_ENABLED

# Warn-once flags so a persistent misconfiguration surfaces without spamming the log.
_SKEW_ENV_WARNED = False
_RUNTIME_SLOT_CONFIG_WARNED = False

# Optional runtime diagnostic. The slot flag must be identical on every rank because enabled ranks
# enter a collective per MoE layer; keep it disabled except during controlled diagnosis.
_MOE_RUNTIME_MEM_LOG = os.environ.get("MOE_RUNTIME_MEM_LOG", "0") == "1"
_MOE_RUNTIME_SLOT_LOG = os.environ.get("MOE_RUNTIME_SLOT_LOG", "0") == "1"
_RUNTIME_DIAGNOSTICS_ENABLED = _MOE_RUNTIME_MEM_LOG or _MOE_RUNTIME_SLOT_LOG
try:
    _MOE_RUNTIME_SLOT_MIN_SLOTS = max(
        0, int(os.environ.get("MOE_RUNTIME_SLOT_MIN_SLOTS", "0"))
    )
except ValueError:
    _MOE_RUNTIME_SLOT_MIN_SLOTS = 0
_RUNTIME_SLOT_PEAKS: List[float] = []
_RUNTIME_SLOT_LOG_UNSUPPORTED = False


def _nontorch_mib() -> float:
    free_b, total_b = torch.cuda.mem_get_info()
    reserved_b = torch.cuda.memory_reserved()
    return max(0, total_b - free_b - reserved_b) / 1048576.0


def _nontorch_mib_safe() -> Optional[float]:
    # Single capture-guarded entry point for the two non-torch memory snapshots (warmup baseline and
    # runtime peak). A host-side driver query (cudaMemGetInfo / memory_reserved) issued while a CUDA
    # graph is capturing would corrupt/abort the capture, and decode warmup captures its graph while
    # trace-memory is on. Return None during capture so callers skip the snapshot instead of crashing
    # the capture; the reserved-load skew still runs so the captured graph reflects the worst case.
    if torch.cuda.is_current_stream_capturing():
        return None
    return _nontorch_mib()


def _log_runtime_nontorch_peak() -> None:
    # Off by default: this runs on the runtime hot path (every MoE layer, every step) and each call
    # issues cudaMemGetInfo + memory_reserved driver queries. Gate it behind an explicit opt-in so
    # production pays nothing unless a developer is diagnosing non-torch memory growth.
    if not _MOE_RUNTIME_MEM_LOG:
        return
    global _NONTORCH_RUNTIME_PEAK_MIB
    # None while a CUDA graph is capturing (decode runs under CUDA graph): the driver query is
    # unsafe there, so skip this diagnostic step rather than abort the capture.
    current_mib = _nontorch_mib_safe()
    if current_mib is None:
        return
    if current_mib <= _NONTORCH_RUNTIME_PEAK_MIB + 50.0:
        return
    _NONTORCH_RUNTIME_PEAK_MIB = current_mib
    baseline_mib = _NONTORCH_BASELINE_MIB or 0.0
    logger.info(
        "[NONTORCH] runtime_peak=%.0f MiB growth_from_warmup=%.0f MiB",
        current_mib,
        max(0.0, current_mib - baseline_mib),
    )


def _log_runtime_slot_distribution(
    router: "FusedMoeDataRouter", topk_ids: torch.Tensor
) -> None:
    """Log global runtime EP slot-share peaks for TP=1 and DP=EP."""
    if not _MOE_RUNTIME_SLOT_LOG:
        return
    if torch.cuda.is_current_stream_capturing():
        return

    config = router.config
    ep_size = int(config.ep_size)
    tp_size = int(config.tp_size)
    dp_size = int(config.dp_size)
    expert_num = int(config.expert_num)
    if ep_size <= 1:
        return
    if expert_num <= 0 or expert_num % ep_size != 0:
        raise RuntimeError(
            f"expert_num={expert_num} must be positive and divisible by ep_size={ep_size}"
        )
    n_local = expert_num // ep_size

    global _RUNTIME_SLOT_LOG_UNSUPPORTED
    if tp_size != 1 or dp_size != ep_size:
        if not _RUNTIME_SLOT_LOG_UNSUPPORTED:
            logger.warning(
                "[RUNTIME_SLOT] unsupported topology: TP=%d DP=%d EP=%d; "
                "exact logging currently requires TP=1 and DP=EP",
                tp_size,
                dp_size,
                ep_size,
            )
            _RUNTIME_SLOT_LOG_UNSUPPORTED = True
        return

    valid_ids = topk_ids.reshape(-1)
    valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < ep_size * n_local)]
    rank_ids = torch.div(valid_ids, n_local, rounding_mode="floor")
    rank_slots = torch.bincount(rank_ids, minlength=ep_size).to(torch.int64)

    from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

    all_reduce(rank_slots, Group.DP_AND_TP)
    slots = [int(x) for x in rank_slots.cpu().tolist()]
    total_slots = sum(slots)
    if total_slots <= 0 or total_slots < _MOE_RUNTIME_SLOT_MIN_SLOTS:
        return
    shares = [x / total_slots for x in slots]

    global _RUNTIME_SLOT_PEAKS
    if len(_RUNTIME_SLOT_PEAKS) != ep_size:
        _RUNTIME_SLOT_PEAKS = [0.0] * ep_size
    improved = [i for i, share in enumerate(shares) if share > _RUNTIME_SLOT_PEAKS[i]]
    if not improved:
        return
    for i in improved:
        _RUNTIME_SLOT_PEAKS[i] = shares[i]

    if int(config.ep_rank) == 0:
        logger.warning(
            "[RUNTIME_SLOT] new_peak ranks=%s current_slots=%s current_shares=%s "
            "peak_shares=%s total_slots=%d",
            improved,
            slots,
            [round(x, 6) for x in shares],
            [round(x, 6) for x in _RUNTIME_SLOT_PEAKS],
            total_slots,
        )


def _in_memory_trace() -> bool:
    """Return whether this forward is inside the C++ RAII-guarded warmup trace scope.

    The C++ warmup sets the process-global flag immediately before creating its temporary
    executor and restores it to false after destroying that executor, including exception paths.
    """
    global _TRACE_MEMORY_FINISHED, _TRACE_MEMORY_SEEN
    if _TRACE_MEMORY_FINISHED:
        return False
    if _IS_TRACE_MEMORY is None:
        return False
    active = bool(_IS_TRACE_MEMORY())
    if active:
        _TRACE_MEMORY_SEEN = True
    elif _TRACE_MEMORY_SEEN:
        _TRACE_MEMORY_FINISHED = True
    return active


def _skew_reserve(mean: float) -> float:
    """Add configurable headroom for persistent hot-expert skew."""
    global _SKEW_ENV_WARNED
    try:
        mult = float(os.environ.get("MOE_SKEW_MULT", "1.5"))
        add = float(os.environ.get("MOE_SKEW_ADD", "0.1"))
    except ValueError:
        if not _SKEW_ENV_WARNED:
            logger.warning(
                "invalid MOE_SKEW_MULT/MOE_SKEW_ADD env value, using defaults 1.5/0.1"
            )
            _SKEW_ENV_WARNED = True
        mult, add = 1.5, 0.1
    return min(1.0, mean * mult + add)


def _default_slot_share(ep_size: int, expert_num: int, top_k: int) -> float:
    """Reserved local-slot share for slot-based executors."""
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
        self.ep_size = int(router.config.ep_size)
        config_expert_num = int(router.config.expert_num)
        if config_expert_num != expert_num:
            raise ValueError(
                f"router expert_num={config_expert_num} does not match FusedMoe expert_num={expert_num}"
            )
        if self.ep_size <= 0 or expert_num <= 0 or expert_num % self.ep_size != 0:
            raise ValueError(
                f"expert_num={expert_num} must be positive and divisible by ep_size={self.ep_size}"
            )
        self.expert_num_per_rank = expert_num // self.ep_size
        if self.ep_size > 1 and _WARMUP_ENABLED and _IS_TRACE_MEMORY is None:
            raise RuntimeError(
                "EP warmup requires compute_ops.is_trace_memory, but the binding is unavailable: "
                f"{_TRACE_MEMORY_IMPORT_ERROR}"
            )

        global _RUNTIME_SLOT_CONFIG_WARNED
        if _MOE_RUNTIME_SLOT_LOG and not _RUNTIME_SLOT_CONFIG_WARNED:
            logger.warning(
                "MOE_RUNTIME_SLOT_LOG is diagnostic-only, must be set consistently on all ranks, "
                "and performs one collective per MoE layer forward"
            )
            _RUNTIME_SLOT_CONFIG_WARNED = True

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

    def _warmup_skew_params(
        self, ep_size: int, expert_num: int, topk: int
    ) -> Tuple[float, int]:
        """Return the reserved hot-token fraction and rank-0 slots per hot token.

        All supported executors are slot-based (memory scales with local expert slots), so the
        reserved load uses the slot-share model C(ep) with all top_k slots routed to rank 0.
        """
        q = _default_slot_share(ep_size, expert_num, topk)
        s = topk
        q = max(1.0 / ep_size, min(1.0, q))
        return q, s

    def _warmup_skew_topk_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Route the reserved warmup load to rank 0."""
        ep_size = self.ep_size
        if ep_size <= 1:
            return topk_ids
        n_local = self.expert_num_per_rank

        num_tokens, topk = topk_ids.shape[0], topk_ids.shape[1]
        expert_num = self.expert_num

        q, s = self._warmup_skew_params(ep_size, expert_num, topk)
        n_hot = int(round(num_tokens * q))
        global _WARMUP_SKEW_LOGGED
        if not _WARMUP_SKEW_LOGGED:
            logger.warning(
                "[MOE_WARMUP] executor=%s mode=slot ep_size=%d experts=%d "
                "top_k=%d skew_fraction=%.6f hot_tokens=%d total_tokens=%d",
                type(self.fused_experts).__name__,
                ep_size,
                expert_num,
                topk,
                q,
                n_hot,
                num_tokens,
            )
            _WARMUP_SKEW_LOGGED = True
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

        is_warmup = self.ep_size > 1 and _WARMUP_ENABLED and _in_memory_trace()
        if is_warmup:
            # Include reserved MoE skew in the measured warmup peak.
            global _NONTORCH_BASELINE_MIB
            if _NONTORCH_BASELINE_MIB is None:
                # Skipped while a CUDA graph is capturing (returns None); a non-capturing warmup
                # forward runs first and fills the baseline. The skew below must run regardless so
                # the captured graph reserves the worst-case (rank-0) hot-expert load.
                baseline = _nontorch_mib_safe()
                if baseline is not None:
                    _NONTORCH_BASELINE_MIB = baseline
                    logger.info(
                        "[NONTORCH] warmup_baseline=%.0f MiB",
                        _NONTORCH_BASELINE_MIB,
                    )
            topk_ids = self._warmup_skew_topk_ids(topk_ids)
        elif _RUNTIME_DIAGNOSTICS_ENABLED:
            _log_runtime_nontorch_peak()
            _log_runtime_slot_distribution(self.router, topk_ids)

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

        return output

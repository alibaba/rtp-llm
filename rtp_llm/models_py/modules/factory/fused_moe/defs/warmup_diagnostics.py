import logging
import os
from typing import Callable, List, Optional, Protocol

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

logger = logging.getLogger(__name__)


class FusedMoeRouter(Protocol):
    config: MoEConfigAdapter


try:
    from rtp_llm.ops.compute_ops import get_trace_memory_state as _GET_TRACE_MEMORY_STATE

    _TRACE_MEMORY_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, AttributeError) as error:
    _GET_TRACE_MEMORY_STATE = None
    _TRACE_MEMORY_IMPORT_ERROR = error


class MoeWarmupDiagnostics:
    """Process-local state for startup MoE warmup and opt-in diagnostics."""

    def __init__(self) -> None:
        self.get_trace_memory_state: Optional[Callable[[], int]] = (
            _GET_TRACE_MEMORY_STATE
        )
        self.trace_memory_import_error = _TRACE_MEMORY_IMPORT_ERROR
        # The C++ trace state is the sole source of truth for the final warmup gate.
        self.trace_memory_finished = False

        self.nontorch_baseline_mib: Optional[float] = None
        self.nontorch_runtime_peak_mib = 0.0
        self.warmup_skew_logged = False
        self.skew_env_warned = False

        self.runtime_mem_log_enabled = False
        self.runtime_slot_log_requested = False
        self.runtime_slot_min_slots = 0
        self.runtime_slot_log_interval = 100
        self.runtime_slot_log_calls = 0
        self.runtime_slot_peaks: List[float] = []
        self.runtime_slot_log_unsupported = False
        self.runtime_slot_config_warned = False

    def reload_runtime_settings(self) -> None:
        """Refresh server-validated diagnostic settings before MoE construction."""
        self.runtime_mem_log_enabled = (
            os.environ.get("MOE_RUNTIME_MEM_LOG", "0") == "1"
        )
        self.runtime_slot_log_requested = (
            os.environ.get("MOE_RUNTIME_SLOT_LOG", "0") == "1"
        )
        try:
            self.runtime_slot_min_slots = max(
                0, int(os.environ.get("MOE_RUNTIME_SLOT_MIN_SLOTS", "0"))
            )
        except ValueError:
            self.runtime_slot_min_slots = 0
        try:
            self.runtime_slot_log_interval = max(
                1, int(os.environ.get("MOE_RUNTIME_SLOT_LOG_INTERVAL", "100"))
            )
            if self.runtime_slot_log_interval > 2**31 - 1:
                raise ValueError
        except ValueError:
            self.runtime_slot_log_interval = 100

    def require_trace_binding(self, ep_size: int) -> None:
        if ep_size > 1 and self.get_trace_memory_state is None:
            raise RuntimeError(
                "EP warmup requires compute_ops.get_trace_memory_state, but the binding is unavailable: "
                f"{self.trace_memory_import_error}"
            )

    def resolve_runtime_slot_log(self, router: FusedMoeRouter) -> bool:
        """Resolve topology support without communicating on the startup path."""
        if not self.runtime_slot_log_requested:
            return False

        config = router.config
        ep_size = int(config.ep_size)
        ffn_disaggregated = bool(
            config.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate
        )
        if ep_size <= 1:
            return False
        if ffn_disaggregated:
            raise RuntimeError(
                "MOE_RUNTIME_SLOT_LOG is not supported with FFN disaggregation"
            )
        return True

    def warn_runtime_slot_cost_once(self) -> None:
        if self.runtime_slot_config_warned:
            return
        logger.warning(
            "MOE_RUNTIME_SLOT_LOG is diagnostic-only and samples one Group.DP all_reduce plus CPU "
            "sync every %d eligible MoE layer calls; startup requires DP-group agreement, and all "
            "ranks must have symmetric CUDA graph capture state",
            self.runtime_slot_log_interval,
        )
        self.runtime_slot_config_warned = True

    @staticmethod
    def _nontorch_mib() -> float:
        free_b, total_b = torch.cuda.mem_get_info()
        reserved_b = torch.cuda.memory_reserved()
        return max(0, total_b - free_b - reserved_b) / 1048576.0

    def nontorch_mib_safe(self) -> Optional[float]:
        if torch.cuda.is_current_stream_capturing():
            return None
        return self._nontorch_mib()

    def capture_warmup_nontorch_baseline(self) -> None:
        if self.nontorch_baseline_mib is not None:
            return
        baseline = self.nontorch_mib_safe()
        if baseline is None:
            return
        self.nontorch_baseline_mib = baseline
        logger.info("[NONTORCH] warmup_baseline=%.0f MiB", baseline)

    def log_runtime_nontorch_peak(self) -> None:
        if not self.runtime_mem_log_enabled:
            return
        current_mib = self.nontorch_mib_safe()
        if current_mib is None or current_mib <= self.nontorch_runtime_peak_mib + 50.0:
            return
        self.nontorch_runtime_peak_mib = current_mib
        baseline_mib = self.nontorch_baseline_mib or 0.0
        logger.info(
            "[NONTORCH] runtime_peak=%.0f MiB growth_from_warmup=%.0f MiB",
            current_mib,
            max(0.0, current_mib - baseline_mib),
        )

    def log_runtime_slot_distribution(
        self, router: FusedMoeRouter, topk_ids: torch.Tensor
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            return

        self.runtime_slot_log_calls += 1
        if (self.runtime_slot_log_calls - 1) % self.runtime_slot_log_interval != 0:
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
        if tp_size != 1 or dp_size != ep_size:
            if not self.runtime_slot_log_unsupported:
                logger.warning(
                    "[RUNTIME_SLOT] unsupported topology: TP=%d DP=%d EP=%d; "
                    "exact logging currently requires TP=1 and DP=EP",
                    tp_size,
                    dp_size,
                    ep_size,
                )
                self.runtime_slot_log_unsupported = True
            return

        n_local = expert_num // ep_size
        valid_ids = topk_ids.reshape(-1)
        valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < ep_size * n_local)]
        rank_ids = torch.div(valid_ids, n_local, rounding_mode="floor")
        rank_slots = torch.bincount(rank_ids, minlength=ep_size).to(torch.int64)

        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        all_reduce(rank_slots, Group.DP)
        slots = [int(x) for x in rank_slots.cpu().tolist()]
        total_slots = sum(slots)
        if total_slots <= 0 or total_slots < self.runtime_slot_min_slots:
            return
        shares = [x / total_slots for x in slots]

        if len(self.runtime_slot_peaks) != ep_size:
            self.runtime_slot_peaks = [0.0] * ep_size
        improved = [
            i
            for i, share in enumerate(shares)
            if share > self.runtime_slot_peaks[i]
        ]
        if not improved:
            return
        for i in improved:
            self.runtime_slot_peaks[i] = shares[i]

        if int(config.ep_rank) == 0:
            logger.warning(
                "[RUNTIME_SLOT] new_peak ranks=%s current_slots=%s current_shares=%s "
                "peak_shares=%s total_slots=%d",
                improved,
                slots,
                [round(x, 6) for x in shares],
                [round(x, 6) for x in self.runtime_slot_peaks],
                total_slots,
            )

    def in_memory_trace(self, ep_size: int) -> bool:
        if ep_size <= 1 or self.trace_memory_finished:
            return False
        if self.get_trace_memory_state is None:
            return False
        state = int(self.get_trace_memory_state())
        if state == 2:
            self.trace_memory_finished = True
            return False
        return state == 1

    def skew_reserve(self, mean: float) -> float:
        try:
            mult = float(os.environ.get("MOE_SKEW_MULT", "1.5"))
            add = float(os.environ.get("MOE_SKEW_ADD", "0.1"))
        except ValueError:
            if not self.skew_env_warned:
                logger.warning(
                    "invalid MOE_SKEW_MULT/MOE_SKEW_ADD env value, using defaults 1.5/0.1"
                )
                self.skew_env_warned = True
            mult, add = 1.5, 0.1
        return min(1.0, mean * mult + add)

    def default_slot_share(self, ep_size: int, expert_num: int, top_k: int) -> float:
        if ep_size <= 1 or expert_num <= top_k:
            return 1.0
        return self.skew_reserve(1.0 / ep_size)

    def warmup_skew_params(
        self, ep_size: int, expert_num: int, top_k: int
    ) -> tuple[float, int]:
        skew_fraction = self.default_slot_share(ep_size, expert_num, top_k)
        return max(1.0 / ep_size, min(1.0, skew_fraction)), top_k

    def warmup_skew_topk_ids(
        self,
        topk_ids: torch.Tensor,
        ep_size: int,
        expert_num: int,
        executor_name: str,
    ) -> torch.Tensor:
        if ep_size <= 1:
            return topk_ids
        n_local = expert_num // ep_size
        num_tokens, top_k = topk_ids.shape[0], topk_ids.shape[1]
        skew_fraction, slots_per_token = self.warmup_skew_params(
            ep_size, expert_num, top_k
        )
        hot_tokens = int(round(num_tokens * skew_fraction))
        self.log_warmup_skew_once(
            executor_name,
            ep_size,
            expert_num,
            top_k,
            skew_fraction,
            hot_tokens,
            num_tokens,
        )

        device, dtype = topk_ids.device, topk_ids.dtype
        columns = torch.arange(top_k, device=device)
        off_rank = 1 + (columns % (ep_size - 1))
        off_ids = (off_rank * n_local + (columns % n_local)).clamp_(
            max=expert_num - 1
        )

        output = topk_ids.clone()
        if hot_tokens > 0:
            hot_ids = off_ids.to(dtype).unsqueeze(0).repeat(hot_tokens, 1)
            hot_rows = torch.arange(hot_tokens, device=device)
            for slot in range(slots_per_token):
                hot_ids[:, slot] = (
                    (hot_rows * slots_per_token + slot) % n_local
                ).to(dtype)
            output[:hot_tokens] = hot_ids
        if hot_tokens < num_tokens:
            output[hot_tokens:] = off_ids.to(dtype).unsqueeze(0)
        return output

    def log_warmup_skew_once(
        self,
        executor_name: str,
        ep_size: int,
        expert_num: int,
        top_k: int,
        skew_fraction: float,
        hot_tokens: int,
        total_tokens: int,
    ) -> None:
        if self.warmup_skew_logged:
            return
        logger.warning(
            "[MOE_WARMUP] executor=%s mode=slot ep_size=%d experts=%d "
            "top_k=%d skew_fraction=%.6f hot_tokens=%d total_tokens=%d",
            executor_name,
            ep_size,
            expert_num,
            top_k,
            skew_fraction,
            hot_tokens,
            total_tokens,
        )
        self.warmup_skew_logged = True


diagnostics = MoeWarmupDiagnostics()


def reload_runtime_diagnostics() -> None:
    diagnostics.reload_runtime_settings()

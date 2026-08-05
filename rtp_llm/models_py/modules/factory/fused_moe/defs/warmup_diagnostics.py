"""Production MoE warmup sizing support.

The trace-state gate, skew reserve calculation, and top-k rewrite in this module
are part of the PREFILL memory-sizing correctness path: their measured peak feeds
the KV-cache budget.

The Active trace window is a startup-only critical section: it must contain only
the synthetic warmup forward and must reach Finished before any serving request
can enter the model. The skew rewrite relies on this lifecycle because it changes
expert ids without changing weights; the synthetic output is deliberately unused.
"""

import logging
import math
from typing import Callable, Optional

import torch

from rtp_llm.utils.pre_import_config import DEFAULT_MOE_SKEW_MULT

logger = logging.getLogger(__name__)


try:
    from rtp_llm.ops.compute_ops import (
        get_trace_memory_state as _GET_TRACE_MEMORY_STATE,
    )

    _TRACE_MEMORY_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, AttributeError) as error:
    _GET_TRACE_MEMORY_STATE = None
    _TRACE_MEMORY_IMPORT_ERROR = error


class MoeWarmupDiagnostics:
    """Process-local state for production warmup sizing.

    Trace-state contract: the C++ trace state advances Pending -> Active ->
    Finished, and only Finished lets is_moe_warmup_active() stop querying
    the binding on every forward. NormalEngine closes it itself (setTraceMemory
    (false)); every other entrypoint that serves forwards must call
    finishTraceMemory / the finish_trace_memory binding once warmup is over.
    For the current set of entrypoints, grep the tree for finishTraceMemory and
    setTraceMemory -- an enumeration here goes stale silently, and this one
    already had (it predated the EmbeddingEngine call site).
    A new entrypoint that forgets this leaves the state at Pending forever, which
    is not a correctness bug but does cost one pybind call per MoE layer forward.
    """

    def __init__(self) -> None:
        self.get_trace_memory_state: Optional[Callable[[], int]] = (
            _GET_TRACE_MEMORY_STATE
        )
        self.trace_memory_import_error = _TRACE_MEMORY_IMPORT_ERROR
        # The C++ trace state is the sole source of truth for the final warmup gate.
        self.trace_memory_finished = False

        self.warmup_skew_logged = False
        self.warmup_capture_warned = False
        self.skew_mult = DEFAULT_MOE_SKEW_MULT

    def reload_runtime_settings(
        self,
        skew_mult: float = DEFAULT_MOE_SKEW_MULT,
    ) -> None:
        """Refresh production skew settings for a new model-build lifecycle."""
        validated_skew_mult = float(skew_mult)
        if not math.isfinite(validated_skew_mult) or validated_skew_mult <= 1.0:
            # At exactly 1.0 the hot rank carries the plain mean share: the skew
            # degenerates to uniform routing and the measured peak carries no
            # expert-imbalance headroom, so refuse rather than warn.
            raise ValueError(
                f"moe_skew_mult must be finite and greater than 1.0, got {skew_mult!r}"
            )

        # These values describe one model-build/warmup lifecycle. A process may
        # construct another model after the previous trace reached Finished.
        self.trace_memory_finished = False
        self.warmup_skew_logged = False
        self.warmup_capture_warned = False

        self.skew_mult = validated_skew_mult

    def require_trace_binding(self, ep_size: int) -> None:
        """Fail closed when PREFILL EP skew cannot observe the warmup window."""
        if ep_size > 1 and self.get_trace_memory_state is None:
            raise RuntimeError(
                "PREFILL EP warmup skew requires compute_ops.get_trace_memory_state, "
                "but the binding is unavailable: "
                f"{self.trace_memory_import_error}"
            )

    def is_moe_warmup_active(self, ep_size: int) -> bool:
        """Return whether this forward is inside the traced EP warmup."""
        if ep_size <= 1 or self.trace_memory_finished:
            return False
        if self.get_trace_memory_state is None:
            return False
        # Pending remains queryable because NormalEngine starts after model
        # construction. Non-NormalEngine entrypoints must explicitly call the
        # finish_trace_memory binding before serving forwards (see the class
        # docstring for the entrypoint list).
        state = int(self.get_trace_memory_state())
        if state == 2:
            self.trace_memory_finished = True
            return False
        return state == 1

    def skew_fraction(self, ep_size: int, expert_num: int, top_k: int) -> float:
        """Share of the whole cluster's tokens routed onto the hot rank (rank 0).

        1.0 when every rank is hit anyway (no EP, or experts <= top_k). Otherwise
        exactly MOE_SKEW_MULT / ep_size, clamped to 1.0: the hot rank carries
        MOE_SKEW_MULT times the 1/ep_size mean share at every ep_size.

        Every rank applies the same fraction to its own tokens and sends them to
        rank 0, and warmup_skew_topk_ids routes the remaining tokens onto the
        non-rank-0 experts, so this nominal share is the hot rank's actual
        dispatched-slot share -- not a lower bound. When top_k exceeds rank 0's
        n_local experts, each hot row can only place n_local of its slots there;
        warmup_skew_topk_ids compensates by scaling the hot row count up (capped
        at the whole batch) and reports the exact rank-0 slot share in its log.
        """
        if ep_size <= 1 or expert_num <= top_k:
            return 1.0
        return min(1.0, self.skew_mult / ep_size)

    def warmup_skew_topk_ids(
        self,
        topk_ids: torch.Tensor,
        ep_size: int,
        expert_num: int,
        executor_name: str,
    ) -> torch.Tensor:
        """Route the reserved hot share of tokens onto rank 0's experts.

        Cold rows are rewritten onto the non-rank-0 experts, spread evenly, so
        rank 0's share is exactly skew_fraction() rather than a lower bound
        inflated by whatever the model's natural routing added. Row-wise
        uniqueness caps each partition per token at its expert count, so slots
        beyond a partition's capacity overflow into the other partition; cold
        overflow lands on rank 0 (nominal share becomes a floor, the safe
        direction), while hot rows dilute at n_local slots per token and are
        compensated by scaling the hot row count (see skew_fraction). The log
        line reports n_local and the exact rank-0 slot share including both
        effects.
        """
        if ep_size <= 1:
            return topk_ids
        if topk_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            if not self.warmup_capture_warned:
                logger.warning(
                    "[MOE_WARMUP] skipping skew rewrite during CUDA graph capture"
                )
                self.warmup_capture_warned = True
            return topk_ids
        # Divisible layouts retain their exact per-rank size. For a redundant
        # custom layout, ceil gives the logical skew a complete hot partition
        # without consulting its physical replica placement.
        n_local = (expert_num + ep_size - 1) // ep_size
        num_tokens, top_k = topk_ids.shape[0], topk_ids.shape[1]
        if top_k > expert_num:
            raise ValueError(
                f"top_k={top_k} cannot contain unique ids for expert_num={expert_num}"
            )
        skew_fraction = self.skew_fraction(ep_size, expert_num, top_k)
        hot_row_fraction = skew_fraction
        if n_local < top_k:
            # A hot row can place only n_local of its top_k slots on rank 0,
            # diluting the dispatched-slot share by n_local/top_k. Scale the hot
            # row count to compensate so the slot share stays at skew_fraction.
            hot_row_fraction = min(1.0, skew_fraction * top_k / n_local)
        hot_tokens = int(round(num_tokens * hot_row_fraction))
        if num_tokens > 0 and hot_row_fraction > 0.0:
            hot_tokens = max(1, hot_tokens)

        # hot_tokens can only be 0 for an empty batch (a DP rank that received no
        # tokens). Return before logging: the summary fires once per lifecycle, and
        # spending it on an empty batch would record hot_tokens=0 total_tokens=0 and
        # suppress the line for the forward that actually rewrote ids.
        if hot_tokens <= 0:
            return topk_ids

        device, dtype = topk_ids.device, topk_ids.dtype
        cold_tokens = num_tokens - hot_tokens
        n_cold = expert_num - n_local

        def _fill_rows(row_count, primary_start, primary_size, other_start, other_size):
            rows = torch.arange(row_count, device=device)
            ids = torch.empty((row_count, top_k), device=device, dtype=dtype)
            n_primary = min(top_k, primary_size)
            for slot in range(n_primary):
                ids[:, slot] = (primary_start + (rows + slot) % primary_size).to(dtype)
            # Overflow slots stay row-unique because top_k <= expert_num bounds
            # top_k - n_primary by other_size.
            for slot in range(n_primary, top_k):
                ids[:, slot] = (other_start + (rows + slot) % other_size).to(dtype)
            return ids

        output = torch.empty_like(topk_ids)
        output[:hot_tokens] = _fill_rows(hot_tokens, 0, n_local, n_local, n_cold)
        if cold_tokens > 0:
            output[hot_tokens:] = _fill_rows(cold_tokens, n_local, n_cold, 0, n_local)

        hot_rank0_slots = min(top_k, n_local)
        cold_rank0_slots = max(0, top_k - n_cold)
        rank0_slot_share = (
            hot_tokens * hot_rank0_slots + cold_tokens * cold_rank0_slots
        ) / (num_tokens * top_k)
        self.log_warmup_skew_once(
            executor_name,
            ep_size,
            expert_num,
            top_k,
            n_local,
            skew_fraction,
            hot_tokens,
            num_tokens,
            rank0_slot_share,
        )
        return output

    # Contract note: the smoke gate greps this line. multi_inst_case_runner.py
    # matches the "[MOE_WARMUP] executor=" prefix, and suites_h20_oss.bzl pins an
    # exact "skew_fraction=%.6f"-formatted substring via
    # SMOKE_EXPECTED_SKEW_FRACTION. Changing the tag, the field name, or the
    # precision requires updating both in the same commit.
    def log_warmup_skew_once(
        self,
        executor_name: str,
        ep_size: int,
        expert_num: int,
        top_k: int,
        n_local: int,
        skew_fraction: float,
        hot_tokens: int,
        total_tokens: int,
        rank0_slot_share: float,
    ) -> None:
        if self.warmup_skew_logged:
            return
        logger.info(
            "[MOE_WARMUP] executor=%s mode=slot ep_size=%d experts=%d "
            "top_k=%d n_local=%d skew_fraction=%.6f hot_tokens=%d total_tokens=%d "
            "rank0_slot_share=%.6f",
            executor_name,
            ep_size,
            expert_num,
            top_k,
            n_local,
            skew_fraction,
            hot_tokens,
            total_tokens,
            rank0_slot_share,
        )
        self.warmup_skew_logged = True


diagnostics = MoeWarmupDiagnostics()


def reload_runtime_diagnostics(
    skew_mult: float = DEFAULT_MOE_SKEW_MULT,
) -> None:
    diagnostics.reload_runtime_settings(skew_mult)

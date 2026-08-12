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
from typing import Callable, Optional, Tuple

import torch

from rtp_llm.ops import MoeConfig

logger = logging.getLogger(__name__)


def _default_moe_skew_mult() -> float:
    """MoeConfig (rtp_llm/cpp/config/ConfigModules.h) owns this default.

    Read through the binding at call time rather than mirrored as a literal or
    captured into `def`-time default arguments, so reload_runtime_settings
    re-reads the current C++ default instead of a frozen copy. Note this is NOT
    import-time laziness: the module-level `diagnostics` singleton below calls
    this once during import. That is safe only because this module already
    imports the extension itself at line 19 (`from rtp_llm.ops import MoeConfig`)
    -- rtp_llm/__init__ deliberately does NOT import ops eagerly, its module
    __getattr__ exists to avoid exactly that. Consequence to keep in mind: the
    singleton validates the C++ default at import time, so if that default ever
    drops below 1.0 the failure surfaces on import rather than during startup
    argument validation.
    """
    return float(MoeConfig().moe_skew_mult)

# Mirror of TraceMemoryPhase in rtp_llm/models_py/bindings/core/ExecOps.h. The
# values are a cross-language contract pinned by a static_assert next to that
# enum, so a C++ renumbering fails the build rather than silently disabling the
# Active gate below (test_warmup_bindings.py pins the same values through the
# real binding).
_TRACE_PHASE_ACTIVE = 1
_TRACE_PHASE_FINISHED = 2


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
    A new entrypoint that forgets this leaves the state at Pending forever, which
    is not a correctness bug but does cost one pybind call per MoE layer forward.
    """

    def __init__(self) -> None:
        self.get_trace_memory_state: Optional[Callable[[], int]] = (
            _GET_TRACE_MEMORY_STATE
        )
        self.trace_memory_import_error = _TRACE_MEMORY_IMPORT_ERROR
        # Declared for readability; reload_runtime_settings below is what actually
        # initializes these, so validation of skew_mult has a single entry point and the
        # per-lifecycle flags cannot be reset in two places that drift apart.
        self.trace_memory_finished = False
        self.warmup_skew_logged = False
        self.warmup_capture_warned = False
        self.warmup_cold_warned = False
        self.warmup_dilution_warned = False
        # Once-per-lifecycle gate for warn_redundant_nondivisible_once below.
        self.redundant_nondivisible_warned = False
        self.skew_mult = 0.0
        self.reload_runtime_settings()

    def reload_runtime_settings(
        self,
        skew_mult: Optional[float] = None,
    ) -> None:
        """Refresh production skew settings for a new model-build lifecycle.

        skew_mult=None reads the MoeConfig default at call time (not def time).
        """
        validated_skew_mult = float(
            _default_moe_skew_mult() if skew_mult is None else skew_mult
        )
        if not math.isfinite(validated_skew_mult) or validated_skew_mult < 1.0:
            # Below 1.0 the hot rank would carry *less* than the mean share, which
            # inverts the skew's purpose, so refuse rather than warn. Exactly 1.0
            # is legal and means "skew disabled": the hot share equals the mean,
            # the rewrite degenerates to uniform routing, so warmup_skew_topk_ids
            # skips the rewrite entirely and keeps the model's natural routing --
            # the documented rollback knob for keeping warmup sizing while
            # turning the skew off.
            source = (
                "from the MoeConfig default"
                if skew_mult is None
                else "from the passed value"
            )
            raise ValueError(
                f"moe_skew_mult must be finite and at least 1.0, got {validated_skew_mult!r} "
                f"({source})"
            )

        # Per-lifecycle flags, reset for a fresh model build. No entrypoint
        # interleaves serving with another build's warmup: production runs one
        # build per process, speculative/MTP builds both models before the
        # single warmup, and test-process rebuilds are serial with no forwards
        # reaching the old model. So resetting trace_memory_finished cannot
        # re-open the skew gate for live traffic. An entrypoint that serves one
        # model while another warms up would break this assumption and must
        # first add lifecycle ownership to is_moe_warmup_active().
        self.trace_memory_finished = False
        self.warmup_skew_logged = False
        self.warmup_capture_warned = False
        self.warmup_cold_warned = False
        self.warmup_dilution_warned = False
        self.redundant_nondivisible_warned = False

        self.skew_mult = validated_skew_mult

    def warn_redundant_nondivisible_once(
        self,
        expert_num: int,
        ep_size: int,
        phy_exp_num: int,
    ) -> None:
        """Warn once per model build about a redundant non-divisible expert layout.

        Called from FusedMoeDataRouter.experts_per_ep_rank, which exempts redundant
        layouts from the hard divisibility failure to keep their pre-existing floor
        partitioning. Every router on every layer reaches that branch, so the dedup
        flag lives here next to the other one-shot warnings and is cleared by
        reload_runtime_settings on a fresh build.
        """
        if self.redundant_nondivisible_warned:
            return
        self.redundant_nondivisible_warned = True
        logger.warning(
            "redundant non-divisible layout (expert_num=%d, ep_size=%d, phy_exp_num=%d): "
            "floor partitioning leaves the tail %d logical experts unreachable from "
            "any rank's local window. Nothing here consumes phy2log, so this layout is "
            "tolerated rather than supported.",
            expert_num,
            ep_size,
            phy_exp_num,
            expert_num % ep_size,
        )

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
        if state == _TRACE_PHASE_FINISHED:
            self.trace_memory_finished = True
            return False
        return state == _TRACE_PHASE_ACTIVE

    def skew_fraction(self, ep_size: int, expert_num: int, top_k: int) -> float:
        """Share of dispatched expert slots targeted at the hot rank (rank 0).

        1.0 when every rank is hit anyway (no EP, or experts <= top_k). Otherwise
        exactly MOE_SKEW_MULT / ep_size, clamped to 1.0: the hot rank carries
        MOE_SKEW_MULT times the 1/ep_size mean share at every ep_size.

        This is a slot-load guarantee, not a guarantee about distinct tokens seen
        by rank 0: concentrating all top_k slots of a hot row can produce fewer
        distinct rank-0 tokens than natural routing. When top_k exceeds rank 0's
        n_local experts, each hot row can only place n_local of its slots there;
        warmup_skew_topk_ids compensates by scaling the hot row count up (capped
        at the whole batch) and reports the exact rank-0 slot share in its log.

        Cross-rank contract (why an asymmetric skew is safe for KV sizing): each
        rank measures its own warmup peak independently, so every non-rank-0 rank
        systematically measures LESS than rank 0 and would derive a larger KV
        budget on its own. The cluster does not use those per-rank numbers
        directly -- KVCacheManager::allocateAndSync all-gathers block_num and
        takes std::min across the world, so the whole cluster ends up sized by
        the most conservative (hot) rank. This layer therefore produces a
        single-rank budget on purpose; removing that min reduction, or switching
        it to max/avg, would make the skew unsafe.
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

        Exactness: the hot window uses the same floor partitioning as the routers
        (FusedMoeDataRouter.experts_per_ep_rank), so the reported rank0_slot_share is
        rank 0's real *logical* dispatched share for every layout the routers accept --
        divisible or redundant-non-divisible alike.

        Remaining caveat, EPLB: the share is computed over logical expert ids. A custom
        phy2log placement can map those ids onto other physical ranks, so under redundant
        or dynamic EPLB layouts read this value as "logical routing achieved", not as
        "physical share landed on rank 0".
        """
        if ep_size <= 1:
            return topk_ids
        if self.skew_mult <= 1.0:
            # 1.0 is the documented "skew disabled" value: the hot share equals the
            # mean, so a rewrite would only replace the model's natural routing
            # with a synthetic uniform one. Keep the natural routing instead --
            # this is the rollback path for keeping warmup sizing without skew.
            return topk_ids
        if topk_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            if not self.warmup_capture_warned:
                logger.warning(
                    "[MOE_WARMUP_DEGRADED] skew_applied=0 reason=cuda_graph_capture"
                )
                self.warmup_capture_warned = True
            return topk_ids
        # Checked here rather than at the top of the method: the early-return paths above
        # (no EP, skew disabled, CUDA graph capture) hand the tensor back untouched and impose
        # no shape contract, while everything below indexes shape[0]/shape[1] -- where a 3-D
        # input would silently read the wrong dimension and a 0-D one raise an opaque
        # IndexError.
        if topk_ids.dim() != 2:
            raise ValueError(
                f"topk_ids must be 2-D [num_tokens, top_k], got shape {tuple(topk_ids.shape)}"
            )
        num_tokens, top_k = topk_ids.shape[0], topk_ids.shape[1]
        if top_k <= 0 or top_k > expert_num:
            # top_k == 0 would also divide by zero in the rank0_slot_share
            # computation below.
            raise ValueError(
                f"top_k={top_k} must be in [1, expert_num={expert_num}]"
            )
        # Match the router's partitioning exactly: floor, the same formula as
        # FusedMoeDataRouter.experts_per_ep_rank. This is what makes the reported
        # rank0_slot_share the real dispatched share rather than an upper bound -- with ceil,
        # a redundant non-divisible layout put the window's tail ids on other ranks while the
        # log still counted them as rank 0's. No-op for divisible layouts (ceil == floor), so
        # measurements published for supported layouts are unchanged.
        #
        # Keep the floor formula local: FusedMoeDataRouter.experts_per_ep_rank() belongs to the
        # importing router module and also enforces divisibility. Reusing it here would introduce
        # a circular dependency and apply stricter validation to BatchedDataRouter's tp == ep path.
        n_local = expert_num // ep_size
        if n_local == 0:
            # Fewer logical experts than ranks: under floor partitioning rank 0 owns none, so
            # there is no window to concentrate onto. Degenerate for every EP router (they get
            # 0 local experts too); keep the model's natural routing instead of inventing one.
            return topk_ids
        # Named target_slot_fraction, not skew_fraction: a local of that name would
        # shadow self.skew_fraction for the rest of this method.
        target_slot_fraction = self.skew_fraction(ep_size, expert_num, top_k)
        hot_row_fraction = target_slot_fraction
        if n_local < top_k:
            # A hot row can place only n_local of its top_k slots on rank 0,
            # diluting the dispatched-slot share by n_local/top_k. Scale the hot
            # row count to compensate so the slot share stays at the target.
            uncompensated_hot_fraction = target_slot_fraction * top_k / n_local
            hot_row_fraction = min(1.0, uncompensated_hot_fraction)
            if uncompensated_hot_fraction > 1.0 and not self.warmup_dilution_warned:
                logger.warning(
                    "[MOE_WARMUP_DEGRADED] skew_applied=1 reason=dilution_clamped "
                    "requested_hot_row_fraction=%.6f applied_hot_row_fraction=1.000000; "
                    "rank0_slot_share may be below target and KV capacity may be optimistic",
                    uncompensated_hot_fraction,
                )
                self.warmup_dilution_warned = True
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
        if cold_tokens == 0 and not self.warmup_cold_warned:
            logger.warning(
                "[MOE_WARMUP_DEGRADED] skew_applied=1 reason=no_cold_tokens "
                "cold_rank_executor_covered=0 total_tokens=%d; non-hot ranks may defer lazy init",
                num_tokens,
            )
            self.warmup_cold_warned = True

        def _fill_rows(
            row_count: int,
            primary: Tuple[int, int],
            other: Tuple[int, int],
        ) -> torch.Tensor:
            """Fill row_count rows, preferring the primary (start, size) expert window.

            Windows are passed as (start, size) pairs rather than four positional
            ints so a call site cannot silently swap a start with a size.
            """
            primary_start, primary_size = primary
            other_start, other_size = other
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
        output[:hot_tokens] = _fill_rows(hot_tokens, (0, n_local), (n_local, n_cold))
        if cold_tokens > 0:
            output[hot_tokens:] = _fill_rows(
                cold_tokens, (n_local, n_cold), (0, n_local)
            )

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
            target_slot_fraction,
            hot_tokens,
            cold_tokens,
            num_tokens,
            rank0_slot_share,
        )
        return output

    # Contract note: the smoke gate greps this line. multi_inst_case_runner.py
    # matches the "[MOE_WARMUP] executor=" prefix, and suites_h20_oss.bzl pins an
    # exact "skew_fraction=%.6f"-formatted substring via
    # SMOKE_EXPECTED_SKEW_FRACTION. Changing the tag, the field name, or the
    # precision requires updating both in the same commit.
    # rank0_slot_share is the real logical dispatched share for every layout the routers
    # accept (the hot window uses their floor partitioning); under EPLB it describes logical
    # routing, not physical placement (see the warmup_skew_topk_ids docstring).
    def log_warmup_skew_once(
        self,
        executor_name: str,
        ep_size: int,
        expert_num: int,
        top_k: int,
        n_local: int,
        skew_fraction: float,
        hot_tokens: int,
        cold_tokens: int,
        total_tokens: int,
        rank0_slot_share: float,
    ) -> None:
        if self.warmup_skew_logged:
            return
        logger.info(
            "[MOE_WARMUP] executor=%s ep_size=%d experts=%d "
            "top_k=%d n_local=%d skew_fraction=%.6f hot_tokens=%d total_tokens=%d "
            "cold_tokens=%d cold_rank_executor_covered=%d skew_applied=1 rank0_slot_share=%.6f",
            executor_name,
            ep_size,
            expert_num,
            top_k,
            n_local,
            skew_fraction,
            hot_tokens,
            total_tokens,
            cold_tokens,
            int(cold_tokens > 0),
            rank0_slot_share,
        )
        self.warmup_skew_logged = True


diagnostics = MoeWarmupDiagnostics()


def reload_runtime_diagnostics(
    skew_mult: Optional[float] = None,
) -> None:
    diagnostics.reload_runtime_settings(skew_mult)

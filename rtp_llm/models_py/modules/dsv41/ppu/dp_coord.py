"""DP8 decode step coordinator for the D machine (plan section 3.2).

Every decode step, all 8 DP ranks must agree on whether to run and in which
order the collectives execute, regardless of how requests are skewed across
ranks:

1. **Step-state exchange** (a control collective): every rank contributes
   ``(valid requests, tokens, finished requests)`` as a small int tensor;
   the coordinator derives the global view and the run/no-run decision.
2. **Collective-order agreement**: the fixed order is ENP query first, then
   per-layer EP dispatch/combine, then the control/release barrier — all
   serial, no cross-group async interleaving (plan 3.2.4). Every rank
   executes the identical phase sequence; the coordinator verifies this via
   :class:`PhaseTracker` and refuses to release resources otherwise.
3. **Empty-rank placeholders** (plan 3.2.3): a rank with zero local tokens
   still participates in the Engram query and every per-layer EP
   dispatch/combine. Its placeholder rows carry invalid expert ids (``-1``),
   zero routing weights and a false validity mask; they never enter sampling
   or the cache.
4. **Early finish / EOS** (plan 3.2.5): finishing changes only that
   request's validity. The rank keeps executing every collective of the
   step while any rank still has tokens.
5. **Stable owner routing** (plan 3.2.6): a request's KV, token history,
   sampling and output always live on one DP owner rank
   (:class:`OwnerRouter`).

The collective surface is an injected interface
(:class:`CollectiveBackend`) so the whole coordinator is testable with the
single-process 8-rank simulator :class:`LocalCollectiveSimulator` (the same
pattern as ``ppu.engram.LocalEnpmGroup``); :class:`TorchCollectiveBackend`
wraps ``torch.distributed`` (PCCL under the ``nccl`` backend name) for the
real D machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Ints exchanged per rank in the step-state control collective.
STEP_STATE_FIELDS = ("valid_requests", "tokens", "finished_requests")
STEP_STATE_WIDTH = len(STEP_STATE_FIELDS)

# Placeholder routing convention shared with ppu.decode: invalid expert id
# slots and zero weights.
INVALID_EXPERT_ID = -1


class CollectiveBackend:
    """Injected collective surface (small int tensors only).

    ``all_gather_ints`` gathers one equal-shape int64 tensor per rank in
    FIXED rank order; ``all_reduce_max_int`` reduces elementwise;
    ``barrier`` is the control barrier. Implementations must be usable
    per-rank; the coordinator calls them symmetrically on every rank.
    """

    rank: int
    size: int

    def all_gather_ints(self, local: torch.Tensor) -> torch.Tensor: ...
    def all_reduce_max_int(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def barrier(self) -> None: ...


class TorchCollectiveBackend(CollectiveBackend):
    """``torch.distributed`` adapter (PCCL via the ``nccl`` backend).

    The small control tensors are built on the HOST (``RankStepState.tensor``
    and friends), but NCCL/PCCL collectives reject CPU tensors ("No backend
    type associated with device type cpu"), so host tensors are moved to the
    current CUDA device first whenever the group rides the ``nccl`` backend.
    Gloo (the CPU test path) keeps host tensors exactly as before.
    """

    def __init__(self, process_group=None) -> None:
        import torch.distributed as dist

        self.dist = dist
        self.pg = process_group

    @property
    def rank(self) -> int:
        return self.dist.get_rank(self.pg)

    @property
    def size(self) -> int:
        return self.dist.get_world_size(self.pg)

    def _collective_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Place ``tensor`` on a device the active backend supports.

        A collective must never be attempted on a device the backend rejects
        (a failed collective can wedge the whole group), so the placement is
        decided BEFORE the call by introspecting the group's backend.
        """
        if tensor.device.type != "cpu":
            return tensor
        try:
            backend = self.dist.get_backend(self.pg)
        except Exception:  # noqa: BLE001 - bare/test construction without a pg
            backend = None
        backend_name = str(getattr(backend, "value", backend) or "").lower()
        if (
            backend_name == "nccl"
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        ):
            return tensor.to(
                torch.device("cuda", torch.cuda.current_device())
            ).contiguous()
        return tensor

    def all_gather_ints(self, local: torch.Tensor) -> torch.Tensor:
        local = self._collective_device(local.to(torch.int64).contiguous())
        out = torch.empty(
            self.size * local.numel(), dtype=torch.int64, device=local.device
        )
        self.dist.all_gather_into_tensor(out, local, group=self.pg)
        return out.reshape(self.size, local.numel())

    def all_reduce_max_int(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._collective_device(tensor.to(torch.int64).contiguous())
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.MAX, group=self.pg)
        return tensor

    def barrier(self) -> None:
        self.dist.barrier(group=self.pg)


@dataclass
class RankStepState:
    """This rank's contribution to the step-state exchange."""

    valid_requests: int = 0
    tokens: int = 0
    finished_requests: int = 0
    # request ids that finished since the last step (early finish / EOS);
    # they change only their own validity, never the collective schedule.
    finished_ids: Tuple[int, ...] = ()

    def tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.valid_requests, self.tokens, self.finished_requests],
            dtype=torch.int64,
        )


@dataclass
class GlobalStepState:
    """The agreed world view of one step (identical on every rank)."""

    per_rank: torch.Tensor  # [size, 3] int64
    world_tokens: int = 0
    max_rank_tokens: int = 0
    run_step: bool = False

    def tokens_of(self, rank: int) -> int:
        return int(self.per_rank[rank][1])

    def requests_of(self, rank: int) -> int:
        return int(self.per_rank[rank][0])


@dataclass
class StepPlan:
    """What one rank does this step.

    ``collective_phases`` is the agreed order every rank executes verbatim:
    the control exchange, one ENP query phase, per-layer EP dispatch/combine,
    and the release barrier. ``placeholder_rows`` is the empty-rank
    placeholder count (0 on ranks with local tokens).
    """

    rank: int
    run: bool
    local_tokens: int
    world_tokens: int
    placeholder_rows: int
    collective_phases: Tuple[str, ...]
    finished_ids: Tuple[int, ...] = ()

    @property
    def total_rows(self) -> int:
        return self.local_tokens + self.placeholder_rows

    def placeholder_batch(self, slots: int = 8) -> "PlaceholderBatch":
        """Placeholder tokens for an empty rank (invalid ids, zero weights)."""
        rows = self.placeholder_rows
        return PlaceholderBatch(
            input_ids=torch.zeros(rows, dtype=torch.int64),
            positions=torch.full((rows,), -1, dtype=torch.int64),
            request_ids=torch.full((rows,), -1, dtype=torch.int64),
            expert_ids=torch.full((rows, slots), INVALID_EXPERT_ID, dtype=torch.int64),
            expert_weights=torch.zeros(rows, slots, dtype=torch.float32),
            valid_mask=torch.zeros(rows, dtype=torch.bool),
        )


@dataclass
class PlaceholderBatch:
    """Empty-rank placeholder rows: never sampled, never cached."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    request_ids: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    valid_mask: torch.Tensor


class OwnerRouter:
    """Stable request_id -> DP rank routing (plan 3.2.6).

    The default rule is ``request_id % dp_size`` (deterministic, stable for
    the request's whole life, and round-robin for sequential ids). An
    explicit map overrides it — used by the transfer path when a request's
    owner must match where P placed its state.
    """

    def __init__(
        self,
        dp_size: int,
        explicit: Optional[Dict[int, int]] = None,
    ) -> None:
        if dp_size < 1:
            raise ValueError("dp_size must be positive")
        self.dp_size = int(dp_size)
        self._explicit: Dict[int, int] = dict(explicit or {})

    def owner(self, request_id: int) -> int:
        request_id = int(request_id)
        if request_id in self._explicit:
            return self._explicit[request_id]
        return request_id % self.dp_size

    def is_local(self, request_id: int, rank: int) -> bool:
        return self.owner(request_id) == int(rank)

    def owners_of(self, request_ids: Sequence[int]) -> Dict[int, int]:
        return {int(rid): self.owner(rid) for rid in request_ids}


def collective_phase_order(num_layers: int) -> Tuple[str, ...]:
    """The fixed collective order (plan 3.2.4): ENP, then EP per layer.

    Control collectives bracket the step: the step-state exchange runs
    before anything else (it produces the agreement), the release barrier
    runs last so no rank frees resources early.
    """
    phases = ["step_exchange", "enp_query"]
    for layer in range(int(num_layers)):
        phases.append(f"ep_dispatch_L{layer}")
        phases.append(f"ep_combine_L{layer}")
    phases.append("step_barrier")
    return tuple(phases)


def expected_backend_calls(phases: Sequence[str]) -> int:
    """Backend-visible collective count for one full step."""
    count = 0
    for phase in phases:
        if phase == "step_exchange":
            count += 1  # the all_gather
        elif phase == "enp_query":
            count += 2  # gather ids + reduce results
        elif phase == "step_barrier":
            count += 1
        # ep_dispatch/ep_combine ride deep_ep, not this backend.
    return count


class PhaseTracker:
    """Records the phases this rank actually executed, in order.

    ``finish_step`` fails loud when the executed sequence deviates from the
    agreed order — a rank may never skip or reorder a collective.
    """

    def __init__(self) -> None:
        self.executed: List[str] = []

    def enter(self, phase: str) -> None:
        self.executed.append(str(phase))

    def finish_step(self, expected: Sequence[str]) -> None:
        if list(self.executed) != list(expected):
            raise RuntimeError(
                "DP step collective order violated: executed "
                f"{self.executed} != agreed {list(expected)}; no rank may "
                "skip or reorder collectives (plan 3.2.5)"
            )


class DPStepCoordinator:
    """Per-rank coordinator implementing the DP8 execution protocol.

    Usage per step::

        plan, tracker = coord.begin_step(local_state)
        if not plan.run:
            ...  # world is drained; only the control exchange happened
        # execute phases in plan.collective_phases order, calling
        # tracker.enter(name) around each phase's collectives
        coord.finish_step(tracker)  # verifies order + release barrier
    """

    def __init__(
        self,
        backend: CollectiveBackend,
        *,
        num_layers: int = 40,
        owner_router: Optional[OwnerRouter] = None,
        dp_size: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.num_layers = int(num_layers)
        self.phases = collective_phase_order(self.num_layers)
        size = int(getattr(backend, "size", 0) or dp_size or 1)
        self.owner_router = owner_router or OwnerRouter(size)
        self._finished: set[int] = set()

    # -- step state ---------------------------------------------------------

    def exchange_step_state(self, local: RankStepState) -> GlobalStepState:
        """The step-state control collective; identical result on all ranks."""
        gathered = self.backend.all_gather_ints(local.tensor())
        per_rank = gathered.reshape(int(self.backend.size), STEP_STATE_WIDTH)
        world_tokens = int(per_rank[:, 1].sum())
        max_rank_tokens = int(per_rank[:, 1].max()) if per_rank.numel() else 0
        return GlobalStepState(
            per_rank=per_rank,
            world_tokens=world_tokens,
            max_rank_tokens=max_rank_tokens,
            run_step=world_tokens > 0,
        )

    def mark_finished(self, request_id: int) -> None:
        """Register a request finishing (EOS or early stop) on this rank."""
        self._finished.add(int(request_id))

    # -- planning -----------------------------------------------------------

    def plan_step(
        self, local: RankStepState, global_state: GlobalStepState
    ) -> StepPlan:
        """Derive this rank's plan from the agreed global state."""
        rank = int(self.backend.rank)
        local_tokens = int(local.tokens)
        placeholders = 0
        if global_state.run_step and local_tokens == 0:
            # Empty rank: placeholders sized by the group maximum so every
            # collective sees a compatible batch (plan 3.2.3). Kernels that
            # do accept empty tensors may ignore them; they never reach
            # sampling or the cache.
            placeholders = max(1, int(global_state.max_rank_tokens))
        return StepPlan(
            rank=rank,
            run=bool(global_state.run_step),
            local_tokens=local_tokens,
            world_tokens=int(global_state.world_tokens),
            placeholder_rows=placeholders,
            collective_phases=self.phases,
            finished_ids=tuple(sorted(self._finished)),
        )

    def begin_step(self, local: RankStepState) -> Tuple[StepPlan, PhaseTracker]:
        """Exchange + plan + tracker (the control collective runs first)."""
        tracker = PhaseTracker()
        tracker.enter("step_exchange")
        global_state = self.exchange_step_state(local)
        plan = self.plan_step(local, global_state)
        return plan, tracker

    def finish_step(
        self, tracker: PhaseTracker, plan: Optional[StepPlan] = None
    ) -> None:
        """The release barrier, then order verification (plan 3.2.5).

        The barrier runs BEFORE any resources are released. A drained world
        (``plan.run`` False) still exchanges state and barriers out, but
        executes no ENP/EP phases.
        """
        if plan is not None and not plan.run:
            expected: Sequence[str] = ("step_exchange", "step_barrier")
        else:
            expected = self.phases
        self.backend.barrier()
        tracker.enter("step_barrier")
        tracker.finish_step(expected)

    # -- validity -----------------------------------------------------------

    def request_validity(self, request_ids: Sequence[int]) -> Dict[int, bool]:
        """Per-request validity: finished requests drop out, ranks do not."""
        return {int(rid): int(rid) not in self._finished for rid in request_ids}


# ---------------------------------------------------------------------------
# Single-process 8-rank simulator (test harness)
# ---------------------------------------------------------------------------


class SimulatedCollectiveBackend(CollectiveBackend):
    """One rank's view of the simulated group (see LocalCollectiveSimulator)."""

    def __init__(self, simulator: "LocalCollectiveSimulator", rank: int) -> None:
        self._sim = simulator
        self.rank = int(rank)
        self.size = simulator.size

    def all_gather_ints(self, local: torch.Tensor) -> torch.Tensor:
        return self._sim.all_gather_ints(self.rank, local)

    def all_reduce_max_int(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._sim.all_reduce_max_int(self.rank, tensor)

    def barrier(self) -> None:
        self._sim.barrier(self.rank)


class LocalCollectiveSimulator:
    """Single-process 8-rank simulator (mirrors ``ppu.engram.LocalEnpmGroup``).

    Like ``LocalEnpmGroup``, the group holds every rank's pre-registered
    inputs and each collective call returns the FULL group result (the real
    backend blocks until the last rank arrives; the simulator's
    ``set_local_states`` plays that rendezvous). A rank that skips a
    collective is therefore caught by the call log —
    ``verify_collective_order`` checks every rank's call count against the
    agreed phase order — instead of by a deadlock.
    """

    def __init__(self, size: int = 8) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = int(size)
        self.log: List[Tuple[str, int]] = []
        self._gathered: Dict[int, torch.Tensor] = {}
        self._max_inputs: Dict[int, torch.Tensor] = {}
        self._barrier_rounds = 0

    def backend(self, rank: int) -> SimulatedCollectiveBackend:
        return SimulatedCollectiveBackend(self, rank)

    def set_local_states(self, states: Sequence[RankStepState]) -> None:
        """Pre-register every rank's step state (the gather rendezvous)."""
        if len(states) != self.size:
            raise ValueError("one RankStepState per rank required")
        for rank, state in enumerate(states):
            self._gathered[rank] = state.tensor().to(torch.int64)

    def set_max_inputs(self, values: Sequence[torch.Tensor]) -> None:
        """Pre-register every rank's all-reduce inputs (same rendezvous)."""
        if len(values) != self.size:
            raise ValueError("one tensor per rank required")
        for rank, value in enumerate(values):
            self._max_inputs[rank] = value.to(torch.int64).reshape(-1)

    # -- collective execution ------------------------------------------------

    def all_gather_ints(self, rank: int, local: torch.Tensor) -> torch.Tensor:
        """Gather the REGISTERED per-rank states (the rendezvous inputs).

        The passed ``local`` is shape-checked and recorded, but the result
        always reflects the tensors registered via ``set_local_states`` —
        the group-level view, exactly like ``LocalEnpmGroup``. This keeps
        later ad-hoc gathers (the ENP query rounds) from clobbering the
        step states that rank-by-rank exchanges still need.
        """
        if len(self._gathered) != self.size:
            raise AssertionError(
                "simulator rendezvous incomplete: call set_local_states with "
                "all ranks' states before driving any rank"
            )
        if local.numel() != STEP_STATE_WIDTH:
            raise ValueError(
                f"all_gather_ints expects {STEP_STATE_WIDTH} ints per rank; "
                f"got {local.numel()}"
            )
        self.log.append(("all_gather", int(rank)))
        return torch.stack([self._gathered[r].reshape(-1) for r in range(self.size)])

    def all_reduce_max_int(self, rank: int, tensor: torch.Tensor) -> torch.Tensor:
        self._max_inputs[int(rank)] = tensor.to(torch.int64).reshape(-1)
        self.log.append(("all_reduce_max", int(rank)))
        if len(self._max_inputs) != self.size:
            # Partial round: the real backend would block here.
            return self._max_inputs[int(rank)].clone()
        result = torch.stack([self._max_inputs[r] for r in range(self.size)]).amax(
            dim=0
        )
        return result

    def barrier(self, rank: int) -> None:
        self.log.append(("barrier", int(rank)))

    # -- verification --------------------------------------------------------

    def rank_log(self, rank: int) -> List[str]:
        return [op for op, r in self.log if r == rank]

    def verify_all_ranks_participated(self) -> None:
        """Every rank executed the same number of collectives."""
        counts = {rank: 0 for rank in range(self.size)}
        for _, rank in self.log:
            counts[rank] += 1
        if len(set(counts.values())) != 1:
            raise AssertionError(
                f"collective participation mismatch: {counts}; a rank skipped "
                "collectives (plan 3.2.2/3.2.5)"
            )

    def verify_collective_order(self, phases: Sequence[str]) -> None:
        """Each rank's BACKEND call count matches the phases' collectives.

        Only collectives riding this backend are counted (step exchange,
        the ENP gather+reduce, the release barrier); the per-layer EP
        dispatch/combine phases ride deep_ep and are verified through the
        coordinator's :class:`PhaseTracker` instead.
        """
        expected = expected_backend_calls(phases)
        for rank in range(self.size):
            calls = len(self.rank_log(rank))
            if calls != expected:
                raise AssertionError(
                    f"rank {rank} executed {calls} collectives, expected "
                    f"{expected} from phases {list(phases)}"
                )

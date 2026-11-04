"""CP8 zigzag context-parallel prefill for the PPU backend (L2 wave 2).

Ports the CUDA reference CP semantics
(``rtp_llm/models_py/modules/dsv41/cp.py::build_cp_context`` — the C++
``ZigZagProcessor`` coordinates — plus ``dsv41/prefill/forward.py::
set_cp_info``) onto the offline PPU prefill stack:

* **Zigzag split.** Per request the padded stream (length padded up to a
  multiple of ``2 * cp_size``) is split across ranks mirror-style: rank ``r``
  takes the front half's ``r``-th block AND the mirrored back half's ``r``-th
  block. Local row order is ``[front block | back block]`` per request,
  requests concatenated — exactly the reference's ``relative_positions``
  construction. Padding rows sit at global positions ``[L, padded)`` and are
  discarded by the restore permutation.
* **Rank-local query, full KV.** Query tokens (and MoE routing, mHC, the
  output projection) stay rank-local. The compressor/indexer/KV inputs a
  rank needs are all-gathered across the CP group and restored into ORIGINAL
  MODEL ORDER before pooling (``CPPrefillContext.all_gather_restore``), so
  every rank's cache holds the full request state (v1: CP KV sharding is
  OFF per plan section 3.1 — the state is replicated per rank).
* **No padding in real state.** The restore permutation only selects real
  tokens; CP padding rows never enter the KV pools, the Engram hash windows
  (mask via ``local_is_real``) or the expert dispatch ids (prefill masks
  padding rows to ``-1``/zero-weight before MoE).
* **Restore.** The final hidden is gathered and restored to logical token
  order on every rank; results are compared in that order, never in buffer
  layout.

Collective backends:

* :class:`DistributedCPGroup` — ``torch.distributed`` (PCCL on the PPU
  hosts) via ``all_gather_into_tensor``; used by the 8-process offline
  driver ``tools/driver_prefill_cp8.py`` together with real DeepEP (EP8)
  and ``DistributedEnpmGroup`` (ENP8).
* :class:`ThreadedCPGroup` — single-process 8-rank simulation (the
  ``LocalCollectiveSimulator`` / ``LocalEnpmGroup`` pattern adapted to a
  full-forward execution model): one thread per rank, every collective a
  rendezvous. It also simulates EP8 dispatch/combine (each rank computes
  only its local experts over the gathered tokens; the combine is the sum)
  and exposes the ENP group surface (``all_gather_ids`` /
  ``all_reduce_sum`` / ``all_max_count``) so ``ops.engram_lookup`` runs its
  real distributed protocol against it.

Known deviation (offline drivers only, documented): every rank derives the
Engram n-gram hashes from the FULL input sequence instead of receiving
per-token history from a framework scheduler. The hash of a token depends
only on the token and its predecessors, so the per-token math is identical;
the framework-side history plumbing belongs to the integrated path.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "CPRequestSplit",
    "CPZigZagPlan",
    "build_zigzag_plan",
    "CPPrefillContext",
    "DistributedCPGroup",
    "ThreadedCPGroup",
    "cp_context_of",
]


# ---------------------------------------------------------------------------
# Zigzag split math (reference: dsv41/cp.py build_cp_context:549-560)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPRequestSplit:
    """One request's zigzag geometry (identical on every rank)."""

    request_id: int
    length: int  # real tokens L (prefix excluded)
    prefix: int  # tokens already in the cache (continuation prefill)
    padded: int  # P = ceil(L / (2*cp)) * 2*cp
    chunk: int  # P // cp — local rows this request contributes per rank
    pair: int  # chunk // 2 — half-block width
    padded_offset: int  # this request's start in the padded-concat stream
    local_offset: int  # this request's start in each rank's local rows

    @property
    def seq_len_total(self) -> int:
        return self.prefix + self.length


def _padded_chunk(length: int, cp_size: int) -> Tuple[int, int]:
    alignment = 2 * int(cp_size)
    padded = ((int(length) + alignment - 1) // alignment) * alignment
    return padded, padded // int(cp_size)


def _owner_and_local(padded_pos: int, pair: int, padded: int) -> Tuple[int, int]:
    """(owning rank, local row index) of one real padded-concat position.

    Front half ``[0, padded/2)``: rank ``pos // pair``, local ``pos % pair``.
    Back half: mirrored — rank ``(padded - 1 - pos) // pair``, local
    ``pair + (pos - (padded - (rank + 1) * pair))``.
    """
    half = padded // 2
    if padded_pos < half:
        return padded_pos // pair, padded_pos % pair
    rank = (padded - 1 - padded_pos) // pair
    local = pair + (padded_pos - (padded - (rank + 1) * pair))
    return rank, local


@dataclass
class CPZigZagPlan:
    """One rank's derived CP metadata for a prefill forward.

    ``local_*`` tensors describe the rank's rows (padding rows carry a
    clamped position / duplicated token id so every kernel stays in range;
    their outputs are dropped by ``unpad_restore``). ``full_*`` tensors are
    the original-order request-concatenated sequence every rank derives the
    Engram windows (and per-request production order) from.
    """

    cp_size: int
    cp_rank: int
    requests: Tuple[CPRequestSplit, ...]
    chunk_length: int  # total rank-local rows (uniform across ranks)
    padded_seq_len: int
    seq_len_full: int  # total real new tokens across requests
    # rank-local views
    local_input_ids: torch.Tensor  # [chunk_length] int64
    local_positions: torch.Tensor  # [chunk_length] int64 (prefix + clamped rel)
    local_request_ids: torch.Tensor  # [chunk_length] int64
    cu_local: torch.Tensor  # [B + 1] int64 — local request boundaries
    local_is_real: torch.Tensor  # [chunk_length] bool
    relative_positions: torch.Tensor  # [chunk_length] int64 padded-concat coords
    unpad_restore: torch.Tensor  # [seq_len_full] int64 gathered-flat indices
    # original-order views (Engram windows / full-order production)
    full_input_ids: torch.Tensor  # [seq_len_full] int64
    full_positions: torch.Tensor  # [seq_len_full] int64
    full_request_ids: torch.Tensor  # [seq_len_full] int64
    full_cu_seqlens: torch.Tensor  # [B + 1] int64
    full_lookback: Optional[torch.Tensor] = None  # [B, ngram - 1] most-recent-first

    @property
    def cp_rank_token_count(self) -> int:
        return int(self.local_is_real.sum())


def build_zigzag_plan(
    full_input_ids: torch.Tensor,
    full_cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    prefix_lengths: Optional[Sequence[int]] = None,
    full_lookback: Optional[torch.Tensor] = None,
) -> CPZigZagPlan:
    """Derive rank ``cp_rank``'s zigzag split of one prefill batch.

    ``full_input_ids`` / ``full_cu_seqlens`` describe the batch in ORIGINAL
    MODEL ORDER (request-concatenated); ``prefix_lengths`` are per-request
    cached lengths (0 for fresh prefill). The padded chunking follows the
    reference: per request ``chunk = ceil(L / (2 * cp_size)) * 2``.
    """
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    if cp_size < 1:
        raise ValueError("cp_size must be positive")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank {cp_rank} out of range for cp_size {cp_size}")
    ids = full_input_ids.detach().to(torch.int64).reshape(-1)
    cu = full_cu_seqlens.detach().to(torch.int64).reshape(-1)
    if cu.numel() < 1 or int(cu[0]) != 0:
        raise ValueError("full_cu_seqlens must start with 0")
    batch = int(cu.numel()) - 1
    lengths = [int(cu[b + 1]) - int(cu[b]) for b in range(batch)]
    if ids.numel() != int(cu[-1]):
        raise ValueError("full_input_ids length does not match full_cu_seqlens")
    if prefix_lengths is None:
        prefix_lengths = [0] * batch
    if len(prefix_lengths) != batch:
        raise ValueError("prefix_lengths must have one entry per request")

    splits: List[CPRequestSplit] = []
    padded_offset = 0
    local_offset = 0
    for request_id, length in enumerate(lengths):
        padded, chunk = _padded_chunk(length, cp_size)
        splits.append(
            CPRequestSplit(
                request_id=request_id,
                length=length,
                prefix=int(prefix_lengths[request_id]),
                padded=padded,
                chunk=chunk,
                pair=chunk // 2,
                padded_offset=padded_offset,
                local_offset=local_offset,
            )
        )
        padded_offset += padded
        local_offset += chunk
    chunk_length = local_offset

    device = ids.device if ids.numel() else torch.device("cpu")
    local_ids: List[torch.Tensor] = []
    local_pos: List[torch.Tensor] = []
    local_req: List[torch.Tensor] = []
    local_real: List[torch.Tensor] = []
    local_rel: List[torch.Tensor] = []
    for split in splits:
        if split.chunk == 0:
            continue
        pair = split.pair
        arange_pair = torch.arange(pair, dtype=torch.int64, device=device)
        even_rel = split.padded_offset + cp_rank * pair + arange_pair
        odd_rel = (
            split.padded_offset + split.padded - (cp_rank + 1) * pair + arange_pair
        )
        rel = torch.cat((even_rel, odd_rel))
        # Padding rows clamp to the request's last real token (reference
        # `req_relative.clamp_max(max_real_pos)`); their outputs are dropped.
        clamped = (rel - split.padded_offset).clamp_max(max(split.length - 1, 0))
        local_rel.append(rel)
        local_pos.append(split.prefix + clamped)
        local_real.append(rel < split.padded_offset + split.length)
        local_req.append(
            torch.full((split.chunk,), split.request_id, dtype=torch.int64)
        )
        if split.length:
            base = int(cu[split.request_id])
            local_ids.append(ids[base + clamped])
        else:
            local_ids.append(torch.zeros(split.chunk, dtype=torch.int64))

    if local_ids:
        local_ids_t = torch.cat(local_ids)
        local_pos_t = torch.cat(local_pos)
        local_req_t = torch.cat(local_req)
        local_real_t = torch.cat(local_real)
        local_rel_t = torch.cat(local_rel)
    else:
        local_ids_t = torch.zeros(0, dtype=torch.int64)
        local_pos_t = torch.zeros(0, dtype=torch.int64)
        local_req_t = torch.zeros(0, dtype=torch.int64)
        local_real_t = torch.zeros(0, dtype=torch.bool)
        local_rel_t = torch.zeros(0, dtype=torch.int64)

    cu_local = torch.zeros(batch + 1, dtype=torch.int64)
    for b, split in enumerate(splits):
        cu_local[b + 1] = cu_local[b] + split.chunk

    # unpad_restore[global real token index] = gathered-flat row index.
    restore: List[int] = []
    for split in splits:
        if split.length == 0 or split.pair == 0:
            continue
        positions = torch.arange(split.length, dtype=torch.int64)
        for pos in positions.tolist():
            rank, local = _owner_and_local(int(pos), split.pair, split.padded)
            restore.append(rank * chunk_length + split.local_offset + local)
    unpad_restore = torch.tensor(restore, dtype=torch.int64)

    # Original-order full views.
    full_positions = torch.cat(
        [
            torch.arange(split.prefix, split.prefix + split.length, dtype=torch.int64)
            for split in splits
        ]
        or [torch.zeros(0, dtype=torch.int64)]
    )
    full_request_ids = torch.cat(
        [
            torch.full((split.length,), split.request_id, dtype=torch.int64)
            for split in splits
        ]
        or [torch.zeros(0, dtype=torch.int64)]
    )
    return CPZigZagPlan(
        cp_size=cp_size,
        cp_rank=cp_rank,
        requests=tuple(splits),
        chunk_length=chunk_length,
        padded_seq_len=padded_offset,
        seq_len_full=int(sum(s.length for s in splits)),
        local_input_ids=local_ids_t,
        local_positions=local_pos_t,
        local_request_ids=local_req_t,
        cu_local=cu_local,
        local_is_real=local_real_t,
        relative_positions=local_rel_t,
        unpad_restore=unpad_restore,
        full_input_ids=ids,
        full_positions=full_positions,
        full_request_ids=full_request_ids,
        full_cu_seqlens=cu,
        full_lookback=full_lookback,
    )


# ---------------------------------------------------------------------------
# Collective backends
# ---------------------------------------------------------------------------


class DistributedCPGroup:
    """CP group over ``torch.distributed`` (PCCL on the PPU hosts).

    ``all_gather`` returns the rank-major gathered-flat tensor
    ``[size * rows, *trailing]``; callers restore through the plan's
    ``unpad_restore``. Every rank must contribute identical shapes (the
    zigzag chunk length is uniform by construction).

    ``register_ops`` + :meth:`moe` implement a SIMULATED EP8 dispatch/combine
    over the same process group (all-gather the ranks' tokens, each rank
    computes only its local experts through ``PPUMoE._forward_reference_tokens``,
    all-reduce sum, slice the rank's own tokens back out). It mirrors the
    DeepEP normal-mode contract (weights applied per expert, combine only
    sums) and gives CPU multi-process runs — and a P-machine fallback should
    the DeepEP buffer path fail — the same semantics without ``deep_ep``.
    """

    def __init__(self, process_group=None):
        self.pg = process_group
        self._ops = None

    @property
    def rank(self) -> int:
        import torch.distributed as dist

        return int(dist.get_rank(self.pg))

    @property
    def size(self) -> int:
        import torch.distributed as dist

        return int(dist.get_world_size(self.pg))

    def register_ops(self, ops) -> None:
        """Register THIS process's ``PPUOps`` for the simulated EP8 MoE."""
        self._ops = ops

    def all_gather(self, local_rows: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        local = local_rows.contiguous()
        out = torch.empty(
            [self.size * local.shape[0]] + list(local.shape)[1:],
            dtype=local.dtype,
            device=local.device,
        )
        dist.all_gather_into_tensor(out, local, group=self.pg)
        return out

    def moe(self, x: torch.Tensor, ids, weights, layer_id: int) -> torch.Tensor:
        import torch.distributed as dist

        if self._ops is None:
            raise RuntimeError(
                "simulated EP MoE needs this rank's PPUOps; call register_ops"
            )
        # bf16 travels as fp32 (gloo lacks bf16 collectives; the bf16 cast
        # inside the expert compute makes the roundtrip exact).
        xs = self.all_gather(x.to(torch.float32).contiguous())
        idss = self.all_gather(ids.to(torch.int64).contiguous())
        wss = self.all_gather(weights.to(torch.float32).contiguous())
        local_out = self._ops._moe._forward_reference_tokens(
            xs, idss, wss, int(layer_id)
        )
        reduced = local_out.to(torch.float32)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.pg)
        count = x.shape[0]
        base = self.rank * count
        return reduced[base : base + count].to(torch.bfloat16).contiguous()


class _Rendezvous:
    """Generation-based barrier collectives for the threaded simulator.

    Every thread calls :meth:`meet` with ``(op, rank, payload)``; the LAST
    arrival (the leader) runs ``combine`` on the rank-ordered payloads and
    publishes the result; everyone returns the same object. A thread that
    skips or reorders a collective makes the next meet's op names disagree,
    which fails loudly instead of deadlocking.
    """

    def __init__(self, size: int):
        self._size = int(size)
        self._cond = threading.Condition()
        self._arrived: List[Tuple[str, int, object]] = []
        self._result: object = None
        self._round = 0
        self._error: Optional[BaseException] = None
        self.ops_log: List[str] = []

    def meet(self, op: str, rank: int, payload, combine: Callable):
        with self._cond:
            round_id = self._round
            self._arrived.append((op, int(rank), payload))
            if len(self._arrived) < self._size:
                while self._round == round_id and self._error is None:
                    self._cond.wait()
                if self._error is not None:
                    raise self._error
                return self._result
            ops = {entry[0] for entry in self._arrived}
            if len(ops) != 1:
                names = [f"{o}@r{r}" for o, r, _ in self._arrived]
                error = RuntimeError(
                    "collective order mismatch across ranks: " + ", ".join(names)
                )
                self._error = error
                self._arrived = []
                self._round += 1
                self._cond.notify_all()
                raise error
            ordered = sorted(self._arrived, key=lambda entry: entry[1])
            self.ops_log.append(op)
            try:
                self._result = combine(ordered)
            except BaseException as error:
                self._error = error
                self._arrived = []
                self._round += 1
                self._cond.notify_all()
                raise
            self._arrived = []
            self._round += 1
            self._cond.notify_all()
            return self._result


class _ThreadedRankView:
    """One rank's facade over the shared :class:`ThreadedCPGroup` core.

    Exposes the CP surface (``all_gather``) AND the ENP group surface
    (``all_gather_ids`` / ``all_reduce_sum`` / ``all_max_count``) so the same
    rendezvous object can be handed to ``PPUOps(enpm_group=...)`` —
    ``ops.engram_lookup`` then runs its real distributed protocol.
    """

    def __init__(self, group: "ThreadedCPGroup", rank: int):
        self._group = group
        self.rank = int(rank)
        self.size = group.size

    # -- CP surface ---------------------------------------------------------

    def all_gather(self, local_rows: torch.Tensor) -> torch.Tensor:
        return self._group._meet(
            "all_gather",
            self.rank,
            local_rows.contiguous(),
            lambda ordered: torch.cat([payload for _, _, payload in ordered], dim=0),
        )

    # -- ENP surface (engram_lookup's distributed protocol) ------------------

    def all_gather_ids(self, local: torch.Tensor) -> torch.Tensor:
        return self.all_gather(local)

    def all_reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        def combine(ordered):
            reduced = ordered[0][2].clone()
            for _, _, extra in ordered[1:]:
                reduced = reduced + extra
            return reduced

        return self._group._meet("all_reduce_sum", self.rank, values, combine)

    def all_max_count(self, count: int, device) -> int:
        return self._group._meet(
            "all_max_count",
            self.rank,
            int(count),
            lambda ordered: max(payload for _, _, payload in ordered),
        )

    # -- simulated EP8 MoE ----------------------------------------------------

    def moe(self, x: torch.Tensor, ids, weights, layer_id: int) -> torch.Tensor:
        """Simulated DeepEP normal dispatch/combine (see ThreadedCPGroup)."""
        results = self._group._meet(
            "moe",
            self.rank,
            (x, ids, weights, int(layer_id)),
            self._group._moe_combine,
        )
        return results[self.rank]


class ThreadedCPGroup:
    """Single-process CP8 simulator: one thread per rank, rendezvous collectives.

    Mirrors the :class:`LocalCollectiveSimulator` / :class:`LocalEnpmGroup`
    pattern (L3/L1) adapted to full-forward execution: prefill runs
    end-to-end inside each rank's thread, so collectives rendezvous through a
    generation barrier instead of pre-registered inputs.

    The simulated EP8 MoE follows the DeepEP contract documented in
    ``ppu/moe.py``: router weights are applied by the per-expert compute and
    ``combine`` only sums across ranks. Each rank computes ONLY its local
    experts (through ``PPUMoE._forward_reference_tokens``, the rtp
    local-loop oracle) over the dispatch-gathered tokens of all ranks, and
    the combine sums the per-rank contributions before slicing each rank's
    own tokens back out.
    """

    def __init__(self, size: int = 8):
        if size < 1:
            raise ValueError("size must be positive")
        self.size = int(size)
        self._rendezvous = _Rendezvous(self.size)
        self._ops: Dict[int, object] = {}
        self._moe_lock = threading.Lock()

    def rank_view(self, rank: int) -> _ThreadedRankView:
        if not 0 <= int(rank) < self.size:
            raise ValueError(f"rank {rank} out of range")
        return _ThreadedRankView(self, int(rank))

    def register_ops(self, ops_by_rank: Dict[int, object]) -> None:
        """Register every rank's ``PPUOps`` for the simulated EP8 MoE."""
        with self._moe_lock:
            self._ops = {int(rank): ops for rank, ops in ops_by_rank.items()}

    def _meet(self, op: str, rank: int, payload, combine):
        return self._rendezvous.meet(op, rank, payload, combine)

    def _moe_combine(self, ordered):
        with self._moe_lock:
            missing = [rank for rank in range(self.size) if rank not in self._ops]
            if missing:
                raise RuntimeError(
                    f"simulated EP MoE needs every rank's PPUOps registered "
                    f"(missing ranks {missing}); call register_ops first"
                )
            layer_ids = {payload[3] for _, _, payload in ordered}
            if len(layer_ids) != 1:
                raise RuntimeError("MoE layer ids disagree across ranks")
            layer_id = next(iter(layer_ids))
            # dispatch: all ranks' tokens in rank order (DeepEP normal mode).
            xs = torch.cat([payload[0] for _, _, payload in ordered], dim=0)
            ids = torch.cat([payload[1] for _, _, payload in ordered], dim=0)
            weights = torch.cat([payload[2] for _, _, payload in ordered], dim=0)
            # local compute: each rank computes only its local experts over
            # the dispatched tokens (weights applied per the moe.py contract).
            # The combine accumulates in fp32 and rounds ONCE, matching the
            # ep_size==1 reference numerics (rtp local_loop oracle); the real
            # DeepEP combine sums bf16 across ranks — its numerics are a
            # hardware bring-up concern, not this simulator's.
            total = None
            for rank in range(self.size):
                # ``PPUOps.moe`` is the contract-op method; the PPUMoE
                # instance behind it carries the local expert shard.
                moe = getattr(self._ops[rank], "_moe", None) or self._ops[rank].moe
                local_out = moe._forward_reference_tokens(xs, ids, weights, layer_id)
                contribution = local_out.float()
                total = contribution if total is None else total + contribution
            combined = total.to(torch.bfloat16)
            # combine: slice each rank's own tokens back out.
            outputs = []
            cursor = 0
            for _, _, payload in ordered:
                count = payload[0].shape[0]
                outputs.append(
                    combined[cursor : cursor + count].clone()
                    if count
                    else combined[:0].clone()
                )
                cursor += count
            return outputs


# ---------------------------------------------------------------------------
# The prefill-side CP context (the parallel_context hook payload)
# ---------------------------------------------------------------------------


def cp_context_of(parallel_context: Optional[object]) -> Optional["CPPrefillContext"]:
    """The CP context when ``parallel_context`` carries live CP metadata."""
    if parallel_context is None:
        return None
    if int(getattr(parallel_context, "cp_size", 1) or 1) <= 1:
        return None
    if not hasattr(parallel_context, "plan") or not hasattr(parallel_context, "group"):
        raise TypeError(
            "parallel_context with cp_size > 1 must be a CPPrefillContext "
            f"(or duck-type plan/group/all_gather_restore); got {type(parallel_context)}"
        )
    return parallel_context


@dataclass
class CPPrefillContext:
    """Everything ``prefill.run_prefill`` needs to execute one CP rank.

    ``group`` is a :class:`DistributedCPGroup` (real 8-process runs; MoE goes
    through the real DeepEP path inside ``PPUOps``) or a
    :class:`ThreadedCPGroup` rank view (CPU simulation; ``simulate_moe``
    routes ``ops.moe`` through the simulated EP8 dispatch/combine).
    """

    plan: CPZigZagPlan
    group: object
    simulate_moe: bool = False
    _collective_count: int = field(default=0, init=False, repr=False)

    @property
    def cp_size(self) -> int:
        return self.plan.cp_size

    @property
    def cp_rank(self) -> int:
        return self.plan.cp_rank

    @property
    def chunk_length(self) -> int:
        return self.plan.chunk_length

    def all_gather_restore(self, local_rows: torch.Tensor) -> torch.Tensor:
        """Gather rank-local rows and restore them to original model order.

        ``local_rows`` is ``[chunk_length, *trailing]`` (padding rows
        included — the restore permutation drops them); the result is
        ``[seq_len_full, *trailing]`` in global request-concatenated order,
        identical on every rank.
        """
        local = local_rows.contiguous()
        if local.shape[0] != self.plan.chunk_length:
            raise ValueError(
                f"CP gather got {local.shape[0]} rows, the plan's chunk length "
                f"is {self.plan.chunk_length}"
            )
        gathered = self.group.all_gather(local)
        restore = self.plan.unpad_restore.to(gathered.device)
        self._collective_count += 1
        return gathered.index_select(0, restore)

    # -- simulated EP8 hook (CPU tests only) ----------------------------------

    def moe(self, x, ids, weights, layer_id: int):
        """The ``ops.moe`` interception used when ``simulate_moe`` is set."""
        if not self.simulate_moe:
            raise RuntimeError("simulate_moe is off; MoE must go through PPUOps")
        return self.group.moe(x, ids, weights, layer_id)

    @property
    def collective_count(self) -> int:
        return self._collective_count

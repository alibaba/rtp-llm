"""Stage-local expert-parallel context for the CP4EP4PP2 topology.

Why this module exists
----------------------
Under pipeline parallelism the four ranks of a stage are the ONLY peer set an
expert-parallel collective may use. The stage's ranks restart their expert
shard at 0, and the two stages are, by design, running different chunks at the
same moment — so any expert collective that resolves to ``WORLD`` is not merely
inefficient, it is wrong: it would exchange payloads with a stage that is not
executing the same round.

The inherited fork MoE path resolves ``dist.group.WORLD`` inside its hot
helpers. At ``pp_size == 1`` WORLD == stage, which hid the requirement
completely. This module makes the stage group explicit, validates it against the
authoritative :class:`~rtp_llm.models_py.distributed.rank_layout.RankLayout`,
and hands it to the selected backend instead of letting each helper re-derive a
group.

Scope
-----
Constructed once after distributed initialization. Immutable. No collectives are
issued from here, and nothing here overlaps or reorders streams — the first
implementation sequences stage-local collective calls synchronously on the
established stream behaviour (design §6.3).
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import List, Optional, Tuple

from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout

logger = logging.getLogger(__name__)

#: The two named, separately-validated expert backends. These are STRINGS only;
#: neither implies the other and neither is a DeepEP kernel.
BACKEND_PURECP_BF16 = "purecp_bf16"
BACKEND_FORK_NCCL_MXFP8 = "fork_nccl_mxfp8"
PP_EP_BACKENDS: Tuple[str, ...] = (BACKEND_PURECP_BF16, BACKEND_FORK_NCCL_MXFP8)

#: The single accepted PP+EP shape. Kept exact on purpose.
PP_EP_PP_SIZE = 2
PP_EP_DP_SIZE = 1
PP_EP_TP_SIZE = 4
PP_EP_EP_SIZE = 4
PP_EP_WORLD_SIZE = 8


def resolve_pp_ep_opt_in(parallelism_config) -> Tuple[bool, str]:
    """Parse the experimental PP+EP opt-in STRICTLY.

    Returns ``(enabled, backend)``. Off unless ``DSV4_PP_EP_ENABLE`` is exactly
    ``1``. An unknown ``DSV4_PP_EP_BACKEND`` raises instead of falling back: a
    silent downgrade would let a launch believe it exercised a backend it did
    not, and the design explicitly rejects "a raw environment variable implying
    capabilities that were never validated".
    """
    raw_enable = os.environ.get("DSV4_PP_EP_ENABLE", "").strip()
    raw_backend = os.environ.get("DSV4_PP_EP_BACKEND", "").strip()

    if raw_enable == "" or raw_enable == "0":
        if raw_backend:
            raise ValueError(
                "DSV4_PP_EP_BACKEND=%r is set but DSV4_PP_EP_ENABLE is not 1; "
                "the backend name alone must not imply enablement." % raw_backend
            )
        return False, ""
    if raw_enable != "1":
        raise ValueError(
            "DSV4_PP_EP_ENABLE must be exactly '1' (or unset/0); got %r" % raw_enable
        )
    if raw_backend not in PP_EP_BACKENDS:
        raise ValueError(
            "DSV4_PP_EP_ENABLE=1 requires DSV4_PP_EP_BACKEND in %r; got %r"
            % (list(PP_EP_BACKENDS), raw_backend)
        )
    return True, raw_backend


def validate_pp_ep_shape(parallelism_config) -> RankLayout:
    """Validate the narrow target shape and return the authoritative layout.

    Raises ``ValueError`` with every violated condition listed, so a bad launch
    fails before model allocation/collectives rather than inside a hot path.
    """
    problems: List[str] = []

    if int(parallelism_config.pp_size) != PP_EP_PP_SIZE:
        problems.append("pp_size=%d (need %d)" % (parallelism_config.pp_size, PP_EP_PP_SIZE))
    if int(parallelism_config.dp_size) != PP_EP_DP_SIZE:
        problems.append("dp_size=%d (need %d)" % (parallelism_config.dp_size, PP_EP_DP_SIZE))
    if int(parallelism_config.tp_size) != PP_EP_TP_SIZE:
        problems.append("tp_size=%d (need %d)" % (parallelism_config.tp_size, PP_EP_TP_SIZE))
    if int(parallelism_config.ep_size) != PP_EP_EP_SIZE:
        problems.append("ep_size=%d (need %d)" % (parallelism_config.ep_size, PP_EP_EP_SIZE))
    if int(parallelism_config.world_size) != PP_EP_WORLD_SIZE:
        problems.append("world_size=%d (need %d)" % (parallelism_config.world_size, PP_EP_WORLD_SIZE))

    cp_config = getattr(parallelism_config, "prefill_cp_config", None)
    if cp_config is None or not cp_config.is_prefill_enabled():
        problems.append("prefill_cp_config.method is not PREFILL_CP")
    else:
        if cp_config.kv_cache_sharded:
            # A second, within-pool slicer on top of the stage slicing is not
            # validated by anything; the design keeps it off.
            problems.append("prefill_cp_config.kv_cache_sharded is set")
        if int(cp_config.prefill_cp_size) not in (0, PP_EP_TP_SIZE):
            problems.append(
                "prefill_cp_config.prefill_cp_size=%d (need 0 or %d)"
                % (cp_config.prefill_cp_size, PP_EP_TP_SIZE)
            )

    if int(parallelism_config.ep_size) != int(parallelism_config.tp_size) * int(
        parallelism_config.dp_size
    ):
        # The rest of the stack (server_config_setup) enforces this too; checked
        # here so the EP stage context never builds on a shape the loader cannot
        # serve expert shards for.
        problems.append("ep_size must equal tp_size * dp_size")

    if problems:
        raise ValueError(
            "DSV4_PP_EP_ENABLE=1 is only valid for the validated CP4EP4PP2 shape "
            "(pp2 x dp1 x tp4 x ep4, prefill-CP4, unsharded CP cache, world8). "
            "Violations: " + "; ".join(problems)
        )

    layout = RankLayout.from_parallelism_config(parallelism_config)
    if layout.world_size() != PP_EP_WORLD_SIZE:
        raise ValueError(
            "RankLayout(pp=%d,dp=%d,tp=%d) implies world_size=%d, but the config says %d"
            % (
                parallelism_config.pp_size,
                parallelism_config.dp_size,
                parallelism_config.tp_size,
                layout.world_size(),
                parallelism_config.world_size,
            )
        )
    return layout


@dataclasses.dataclass(frozen=True)
class EpStageContext:
    """Immutable, validated view of one pipeline stage's CP/EP roster.

    ``process_group`` is the materialized stage communicator. At ``dp_size == 1``
    a stage coincides with its TP group, and the collective layer already aliases
    ``Group.STAGE`` onto that roster, so no duplicate communicator is created.
    """

    process_group: object
    world_ranks: Tuple[int, ...]
    group_rank: int
    group_size: int
    pp_rank: int
    generation: int
    backend: str

    def __post_init__(self) -> None:
        if self.group_size != len(self.world_ranks):
            raise ValueError(
                "group_size=%d does not match %d world ranks"
                % (self.group_size, len(self.world_ranks))
            )
        if not 0 <= self.group_rank < self.group_size:
            raise ValueError(
                "group_rank=%d outside [0,%d)" % (self.group_rank, self.group_size)
            )
        if self.pp_rank not in (0, 1):
            raise ValueError("pp_rank=%d outside the two-stage target" % self.pp_rank)
        if self.backend not in PP_EP_BACKENDS:
            raise ValueError("unknown expert backend %r" % self.backend)
        if len(set(self.world_ranks)) != self.group_size:
            raise ValueError("world_ranks contains duplicates: %r" % (self.world_ranks,))

    # ---- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        parallelism_config,
        backend: str,
        process_group=None,
        generation: int = 0,
    ) -> "EpStageContext":
        """Resolve the stage roster from ``RankLayout`` and validate it.

        ``process_group`` defaults to the materialized ``Group.STAGE`` handle,
        which at ``dp_size == 1`` is the stage's TP group.
        """
        if backend not in PP_EP_BACKENDS:
            raise ValueError("unknown expert backend %r" % backend)

        layout = validate_pp_ep_shape(parallelism_config)
        world_rank = int(parallelism_config.world_rank)
        if not 0 <= world_rank < PP_EP_WORLD_SIZE:
            raise ValueError("world_rank=%d outside world_size=%d" % (world_rank, PP_EP_WORLD_SIZE))

        stage_ranks = tuple(layout.group_of(Group.STAGE, world_rank))
        group_rank = int(layout.rank_in_group(Group.STAGE, world_rank))
        pp_rank = int(layout.coord_of(world_rank).pp)

        ep_rank = int(layout.ep_rank_of(world_rank, parallelism_config.ep_size))
        if ep_rank != group_rank:
            # At dp1 the lane-local rank and the expert shard must coincide;
            # otherwise the loader's expert offset and the collective's peer
            # order disagree.
            raise ValueError(
                "ep_rank=%d != stage group_rank=%d at dp1 (world_rank=%d)"
                % (ep_rank, group_rank, world_rank)
            )

        if process_group is None:
            process_group = _stage_process_group(parallelism_config)
        _verify_process_group(process_group, stage_ranks, world_rank)
        if ep_rank != int(getattr(parallelism_config, "ep_rank", ep_rank)):
            raise ValueError(
                "loader ep_rank=%r disagrees with the layout's %d for world_rank=%d"
                % (getattr(parallelism_config, "ep_rank", None), ep_rank, world_rank)
            )

        ctx = cls(
            process_group=process_group,
            world_ranks=stage_ranks,
            group_rank=group_rank,
            group_size=len(stage_ranks),
            pp_rank=pp_rank,
            generation=int(generation),
            backend=backend,
        )
        logger.info(
            "[EpStageContext] backend=%s pp_rank=%d group_rank=%d/%d world_ranks=%s",
            ctx.backend,
            ctx.pp_rank,
            ctx.group_rank,
            ctx.group_size,
            list(ctx.world_ranks),
        )
        return ctx

    # ---- construction (single-stage / PP1) --------------------------------

    @classmethod
    def build_single_stage(
        cls,
        parallelism_config,
        backend: str,
        process_group=None,
        generation: int = 0,
    ) -> "EpStageContext":
        """Context for a PP1 launch, where WORLD *is* the expert roster.

        This is a SEPARATE entry point on purpose. `validate_pp_ep_shape` encodes
        the CP4EP4PP2 engine contract (pp2 x dp1 x tp4 x ep4, world8) and must not
        be relaxed to serve a fixture — that guard is what stops an env var from
        enabling an unvalidated shape. Here the identity is instead PROVEN rather
        than assumed: exactly one stage, world size equal to the expert count.

        At pp1 there is no cross-stage hazard, so `_verify_process_group`'s
        "must not be WORLD" rule does not apply; the roster, order and local rank
        are still checked against the live group.
        """
        import torch.distributed as dist

        if backend not in PP_EP_BACKENDS:
            raise ValueError("unknown expert backend %r" % backend)
        if not dist.is_initialized():
            raise ValueError("distributed is not initialized while building EpStageContext")

        pp_size = int(getattr(parallelism_config, "pp_size", 1) or 1)
        ep_size = int(getattr(parallelism_config, "ep_size", 1) or 1)
        world_size = int(getattr(parallelism_config, "world_size", 1) or 1)
        if pp_size != 1:
            raise ValueError(
                "build_single_stage is for pp_size==1; got pp_size=%d. Use build() "
                "for the CP4EP4PP2 two-stage target." % pp_size
            )
        if world_size != ep_size:
            raise ValueError(
                "at pp_size==1 the WORLD group is the expert roster only when "
                "world_size == ep_size; got %d != %d" % (world_size, ep_size)
            )

        world_rank = int(getattr(parallelism_config, "world_rank", 0) or 0)
        if process_group is None:
            process_group = dist.group.WORLD

        roster = tuple(int(r) for r in dist.get_process_group_ranks(process_group))
        if len(roster) != world_size or tuple(sorted(roster)) != tuple(range(world_size)):
            raise ValueError(
                "the pp1 expert roster must be ranks 0..%d in order; got %r"
                % (world_size - 1, list(roster))
            )
        if int(dist.get_rank()) != world_rank:
            raise ValueError(
                "caller world_rank=%d disagrees with the live world rank %d"
                % (world_rank, int(dist.get_rank()))
            )
        group_rank = int(dist.get_rank(process_group))
        ep_rank = int(getattr(parallelism_config, "ep_rank", group_rank))
        if ep_rank != group_rank:
            raise ValueError(
                "loader ep_rank=%d disagrees with the live group rank %d"
                % (ep_rank, group_rank)
            )

        ctx = cls(
            process_group=process_group,
            world_ranks=roster,
            group_rank=group_rank,
            group_size=len(roster),
            pp_rank=0,
            generation=int(generation),
            backend=backend,
        )
        logger.info(
            "[EpStageContext] single-stage backend=%s roster=%s group_rank=%d",
            ctx.backend,
            list(ctx.world_ranks),
            ctx.group_rank,
        )
        return ctx

    # ---- queries ----------------------------------------------------------

    def is_stage_root(self, world_rank: int) -> bool:
        """Stage-local root: rank 0 OF THIS STAGE, never global rank 0."""
        return int(world_rank) == self.world_ranks[0]

    def peer_world_rank(self, group_rank: int) -> int:
        return self.world_ranks[group_rank]

    def __repr__(self) -> str:  # process_group holds no useful repr
        return (
            "EpStageContext(backend=%s, pp_rank=%d, group_rank=%d/%d, "
            "world_ranks=%s, generation=%d)"
            % (
                self.backend,
                self.pp_rank,
                self.group_rank,
                self.group_size,
                list(self.world_ranks),
                self.generation,
            )
        )


def _stage_process_group(parallelism_config):
    """The materialized ``Group.STAGE`` communicator (== the stage's TP group at dp1)."""
    from rtp_llm.models_py.distributed import collective_torch

    return collective_torch._get_group(Group.STAGE)


def _verify_process_group(process_group, stage_ranks: Tuple[int, ...], live_rank: int) -> None:
    """The LIVE communicator must match the layout-derived roster exactly.

    Checks size, membership, ORDER, the caller's local rank inside the group and
    the loader's `ep_rank`. A mismatch means the group map was built for a
    different parallelism configuration than the one being validated — the
    failure mode that would otherwise show up as a hang or as cross-stage
    contamination much later.

    `live_rank` is the world rank the CALLER believes it is; it is compared
    against `dist.get_rank(group.WORLD)` so a caller whose view of its own rank
    disagrees with the process group fails here rather than in a collective.
    """
    import torch.distributed as dist

    if process_group is None:
        raise ValueError("stage process group is None")
    if not dist.is_initialized():
        raise ValueError("distributed is not initialized while building EpStageContext")

    if process_group is dist.group.WORLD:
        raise ValueError(
            "stage process group resolved to WORLD; the stage must be a strict "
            "subset under pp_size>1"
        )
    if len(stage_ranks) < 2:
        raise ValueError("stage roster %r has fewer than 2 ranks; EP needs >=2" % (stage_ranks,))

    actual_world_rank = int(dist.get_rank())
    if int(live_rank) != actual_world_rank:
        raise ValueError(
            "caller world_rank=%d disagrees with the live world rank %d"
            % (int(live_rank), actual_world_rank)
        )

    actual = tuple(int(r) for r in dist.get_process_group_ranks(process_group))
    if len(actual) != len(stage_ranks):
        raise ValueError(
            "stage process group has %d rank(s) but the layout roster has %d: %r vs %r"
            % (len(actual), len(stage_ranks), list(actual), list(stage_ranks))
        )
    if tuple(sorted(actual)) != tuple(sorted(stage_ranks)):
        raise ValueError(
            "stage process group ranks %r do not match the layout roster %r"
            % (list(actual), list(stage_ranks))
        )
    if actual != tuple(stage_ranks):
        # The collective's peer order IS this order; agreeing as a set is not
        # enough, because the send/recv split vectors are indexed by it.
        raise ValueError(
            "stage process group order %r differs from the layout roster order %r"
            % (list(actual), list(stage_ranks))
        )

    local_rank = int(dist.get_rank(process_group))
    expected_local = stage_ranks.index(actual_world_rank)
    if local_rank != expected_local:
        raise ValueError(
            "rank %d has group-local rank %d, but the layout roster %r places it at %d"
            % (actual_world_rank, local_rank, list(stage_ranks), expected_local)
        )


def maybe_build(
    parallelism_config,
    backend: Optional[str] = None,
    process_group=None,
) -> Optional[EpStageContext]:
    """Build the context when the resolved PP+EP opt-in is on, else ``None``.

    A convenience for model construction; it never guesses a backend.
    """
    enabled, resolved_backend = resolve_pp_ep_opt_in(parallelism_config)
    if not enabled:
        return None
    if backend is not None and backend != resolved_backend:
        raise ValueError(
            "explicit backend %r disagrees with the resolved DSV4_PP_EP_BACKEND %r"
            % (backend, resolved_backend)
        )
    return EpStageContext.build(
        parallelism_config, backend=resolved_backend, process_group=process_group
    )

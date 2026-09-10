"""Single authoritative implementation of the world-rank layout:

    world_rank = pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank

Axis order {PP(outermost), DP, TP(innermost)}. Every rank derivation must go
through this module; the C++ side only reads the ParallelismConfig fields
materialized here. Pure stdlib, importable from every startup path.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Group(Enum):
    """Parallel dimensions plus composite groups. Single enum shared by the
    layout model and collective communication. WORLD spans all ranks
    (torch.distributed WORLD; the former "DP_AND_TP", named before PP existed);
    STAGE spans one pipeline stage's dp*tp ranks (WORLD narrowed per stage).
    Both are composite groups, not single-axis slices. STAGE is layout-level
    only: its torch process group is materialized when a consumer appears.
    PCP is reserved, not materialized yet."""

    TP = "TP"
    DP = "DP"
    PP = "PP"
    WORLD = "WORLD"
    STAGE = "STAGE"


@dataclass(frozen=True)
class Coord:
    """Parallel coordinates of one world rank."""

    tp: int = 0
    dp: int = 0
    pp: int = 0


class RankLayout:
    """World-rank lattice over axes {PP, DP, TP}.

    Determinism contract (process group key naming relies on it):
    groups() orders groups lexicographically by the pinned coordinates
    (pinned axes iterated PP, DP, TP); ranks inside a group ascend and the
    position inside the group equals the coordinate along the target axis.
    """

    __slots__ = ("_pp_size", "_dp_size", "_tp_size")

    def __init__(self, pp_size: int = 1, dp_size: int = 1, tp_size: int = 1):
        if pp_size < 1 or dp_size < 1 or tp_size < 1:
            raise ValueError(
                f"RankLayout sizes must all be >= 1, got "
                f"pp_size={pp_size}, dp_size={dp_size}, tp_size={tp_size}"
            )
        self._pp_size = int(pp_size)
        self._dp_size = int(dp_size)
        self._tp_size = int(tp_size)

    @staticmethod
    def from_parallelism_config(cfg) -> "RankLayout":
        """Duck-typed on the three size fields; missing/zero sizes fall back to 1."""
        return RankLayout(
            pp_size=max(int(getattr(cfg, "pp_size", 1)), 1),
            dp_size=max(int(getattr(cfg, "dp_size", 1)), 1),
            tp_size=max(int(getattr(cfg, "tp_size", 1)), 1),
        )

    @property
    def pp_size(self) -> int:
        return self._pp_size

    @property
    def dp_size(self) -> int:
        return self._dp_size

    @property
    def tp_size(self) -> int:
        return self._tp_size

    def size_of(self, group: Group) -> int:
        return {
            Group.TP: self._tp_size,
            Group.DP: self._dp_size,
            Group.PP: self._pp_size,
            Group.WORLD: self.world_size(),
            Group.STAGE: self.lane_stride(),
        }[group]

    def world_size(self) -> int:
        return self._pp_size * self._dp_size * self._tp_size

    def lane_stride(self) -> int:
        """World-rank stride between adjacent PP stages of the same lane."""
        return self._dp_size * self._tp_size

    def coord_of(self, world_rank: int) -> Coord:
        """Validated: raises ValueError when world_rank is outside the lattice."""
        world_rank = int(world_rank)
        if world_rank < 0 or world_rank >= self.world_size():
            raise ValueError(
                f"world_rank {world_rank} out of range [0, {self.world_size()}) "
                f"for layout {self}"
            )
        return self.coord_of_unchecked(world_rank)

    def coord_of_unchecked(self, world_rank: int) -> Coord:
        """No range check: ranks outside the lattice yield out-of-bounds
        coordinates (legacy-compatible semantics for worlds whose process
        count exceeds pp*dp*tp, e.g. FFN-disaggregate replica/service ranks).
        Only for callsites that legitimately see such ranks."""
        world_rank = int(world_rank)
        tp = world_rank % self._tp_size
        dp = (world_rank // self._tp_size) % self._dp_size
        pp = world_rank // (self._dp_size * self._tp_size)
        return Coord(tp=tp, dp=dp, pp=pp)

    def world_rank_of(self, coord: Coord) -> int:
        if not (0 <= coord.tp < self._tp_size):
            raise ValueError(
                f"tp coordinate {coord.tp} out of range [0, {self._tp_size})"
            )
        if not (0 <= coord.dp < self._dp_size):
            raise ValueError(
                f"dp coordinate {coord.dp} out of range [0, {self._dp_size})"
            )
        if not (0 <= coord.pp < self._pp_size):
            raise ValueError(
                f"pp coordinate {coord.pp} out of range [0, {self._pp_size})"
            )
        return (coord.pp * self._dp_size + coord.dp) * self._tp_size + coord.tp

    def groups(self, group: Group) -> List[List[int]]:
        """All groups of `group`, deterministic order (see class docstring)."""
        groups: List[List[int]] = []
        if group is Group.TP:
            for pp in range(self._pp_size):
                for dp in range(self._dp_size):
                    groups.append(
                        [
                            self.world_rank_of(Coord(tp=t, dp=dp, pp=pp))
                            for t in range(self._tp_size)
                        ]
                    )
        elif group is Group.DP:
            for pp in range(self._pp_size):
                for tp in range(self._tp_size):
                    groups.append(
                        [
                            self.world_rank_of(Coord(tp=tp, dp=d, pp=pp))
                            for d in range(self._dp_size)
                        ]
                    )
        elif group is Group.PP:
            for dp in range(self._dp_size):
                for tp in range(self._tp_size):
                    groups.append(
                        [
                            self.world_rank_of(Coord(tp=tp, dp=dp, pp=p))
                            for p in range(self._pp_size)
                        ]
                    )
        elif group is Group.WORLD:
            groups.append(list(range(self.world_size())))
        elif group is Group.STAGE:
            for pp in range(self._pp_size):
                groups.append(
                    [
                        self.world_rank_of(Coord(tp=t, dp=d, pp=pp))
                        for d in range(self._dp_size)
                        for t in range(self._tp_size)
                    ]
                )
        else:  # pragma: no cover - guarded by the Group enum
            raise ValueError(f"unknown group: {group}")
        return groups

    def group_of(self, group: Group, world_rank: int) -> List[int]:
        """The unique group of `group` that contains `world_rank`."""
        coord = self.coord_of(world_rank)
        for members in self.groups(group):
            if world_rank in members:
                return members
        raise AssertionError(  # pragma: no cover - unreachable by construction
            f"world_rank {world_rank} (coord {coord}) not found in any {group} group"
        )

    def rank_in_group(self, group: Group, world_rank: int) -> int:
        """Position inside the `group` group; equals the coordinate along that
        axis, the lane-local position for STAGE, or world_rank for WORLD."""
        coord = self.coord_of(world_rank)
        return {
            Group.TP: coord.tp,
            Group.DP: coord.dp,
            Group.PP: coord.pp,
            Group.WORLD: world_rank,
            Group.STAGE: coord.dp * self._tp_size + coord.tp,
        }[group]

    def ep_rank_of(self, world_rank: int, ep_size: int) -> int:
        """Expert-shard number: lane-local rank modulo ep_size. At pp=1 this
        equals the legacy `world_rank % ep_size` (lane_rank == world_rank).
        ep_size must divide the per-stage rank count; ep_size <= 1 means no EP."""
        ep_size = int(ep_size)
        if ep_size <= 1:
            return 0
        stride = self.lane_stride()
        if stride % ep_size != 0:
            raise ValueError(
                f"ep_size {ep_size} must divide the per-stage rank count "
                f"{stride} for layout {self}"
            )
        return int(world_rank) % stride % ep_size

    def ep_groups(self) -> List[List[int]]:
        """EP communication rosters: one group per PP stage holding all its
        dp*tp ranks (WORLD narrowed per stage; equals [WORLD] at pp=1).
        Identical to groups(Group.STAGE); kept as the EP-view alias."""
        return self.groups(Group.STAGE)

    def __repr__(self) -> str:
        return (
            f"RankLayout(pp_size={self._pp_size}, dp_size={self._dp_size}, "
            f"tp_size={self._tp_size})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RankLayout):
            return NotImplemented
        return (
            self._pp_size == other._pp_size
            and self._dp_size == other._dp_size
            and self._tp_size == other._tp_size
        )

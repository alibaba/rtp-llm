"""World-rank layout with an explicit global process count:

    world_rank = pp_rank * lane_stride + dp_rank * tp_size + tp_rank
    lane_stride = world_size // pp_size

Axis order {PP(outermost), DP, TP(innermost)}. TP/DP group enumeration uses
the declared sizes; rank validation uses the actual world size.
Pure stdlib, importable from every startup path.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Group(Enum):
    """Parallel dimensions plus composite groups. Single enum shared by the
    layout model and collective communication. WORLD spans all ranks;
    STAGE spans one pipeline stage (WORLD narrowed per stage).
    Both are composite groups, not single-axis slices.
    """

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

    __slots__ = ("_world_size", "_pp_size", "_dp_size", "_tp_size")

    def __init__(
        self, world_size: int, pp_size: int = 1, dp_size: int = 1, tp_size: int = 1
    ):
        if world_size < 1 or pp_size < 1 or dp_size < 1 or tp_size < 1:
            raise ValueError(
                f"RankLayout sizes must all be >= 1, got "
                f"world_size={world_size}, pp_size={pp_size}, "
                f"dp_size={dp_size}, tp_size={tp_size}"
            )
        if world_size % pp_size != 0:
            raise ValueError(
                f"world_size {world_size} must be divisible by pp_size {pp_size}"
            )
        self._world_size = int(world_size)
        self._pp_size = int(pp_size)
        self._dp_size = int(dp_size)
        self._tp_size = int(tp_size)

    @staticmethod
    def from_parallelism_config(cfg) -> "RankLayout":
        return RankLayout(
            world_size=cfg.world_size,
            pp_size=cfg.pp_size,
            dp_size=cfg.dp_size,
            tp_size=cfg.tp_size,
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
        return self._world_size

    def lane_stride(self) -> int:
        """World-rank stride between adjacent PP stages of the same lane."""
        return self._world_size // self._pp_size

    def coord_of(self, world_rank: int) -> Coord:
        """Return coordinates; reject world ranks outside the layout."""
        world_rank = int(world_rank)
        if world_rank < 0 or world_rank >= self.world_size():
            raise ValueError(
                f"world_rank {world_rank} out of range [0, {self.world_size()}) "
                f"for layout {self}"
            )

        pp_rank, rank_in_stage = divmod(world_rank, self.lane_stride())
        dp_rank, tp_rank = divmod(rank_in_stage, self._tp_size)
        return Coord(tp=tp_rank, dp=dp_rank, pp=pp_rank)

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
        return coord.pp * self.lane_stride() + coord.dp * self._tp_size + coord.tp

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
            stride = self.lane_stride()
            for pp in range(self._pp_size):
                groups.append(list(range(pp * stride, (pp + 1) * stride)))
        else:  # pragma: no cover - guarded by the Group enum
            raise ValueError(f"unknown group: {group}")
        return groups

    def group_of(self, group: Group, world_rank: int) -> List[int]:
        """The unique group of `group` that contains `world_rank`."""
        coord = self.coord_of(world_rank)
        for members in self.groups(group):
            if world_rank in members:
                return members
        raise AssertionError(
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
        """Expert-shard number within a stage, for a world rank in this layout.
        ep_size must divide the per-stage rank count; ep_size <= 1 means no EP."""
        coord = self.coord_of(world_rank)
        ep_size = int(ep_size)
        if ep_size <= 1:
            return 0
        stride = self.lane_stride()
        if stride % ep_size != 0:
            raise ValueError(
                f"ep_size {ep_size} must divide the per-stage rank count "
                f"{stride} for layout {self}"
            )
        stage_rank = coord.dp * self._tp_size + coord.tp
        return stage_rank % ep_size

    def __repr__(self) -> str:
        return (
            f"RankLayout(world_size={self._world_size}, pp_size={self._pp_size}, "
            f"dp_size={self._dp_size}, tp_size={self._tp_size})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RankLayout):
            return NotImplemented
        return (
            self._world_size == other._world_size
            and self._pp_size == other._pp_size
            and self._dp_size == other._dp_size
            and self._tp_size == other._tp_size
        )

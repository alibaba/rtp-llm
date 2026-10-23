"""Local zigzag row ownership, including canonical router chunk offsets."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ZigzagShardLayout:
    logical_tokens: int
    local_tokens: int
    local_valid_tokens: int
    rank: int
    world_size: int
    canonical_fragments: tuple[tuple[int, int, int], ...]

    @classmethod
    def from_cp(cls, cp):
        fragments = sorted((f.cp_offset, f.canonical_offset, f.length)
                           for f in cp.fragments if f.cp_rank == cp.rank)
        merged = []
        for local, canonical, length in fragments:
            if merged and merged[-1][0] + merged[-1][2] == local and merged[-1][1] + merged[-1][2] == canonical:
                a, b, c = merged[-1]
                merged[-1] = (a, b, c + length)
            else:
                merged.append((local, canonical, length))
        if sum(x[2] for x in merged) != cp.local_cp_tokens:
            raise ValueError('KCP requires unpadded zigzag rows')
        return cls(cp.sp_layout.logical_tokens, cp.local_cp_tokens,
                   cp.local_cp_tokens, cp.rank, cp.world_size, tuple(merged))

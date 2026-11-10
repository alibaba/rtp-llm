"""Fixed-TP4 logical/shard keys, not cross-topology SR coordinates.

The caller must establish each actual parameter's identity, category and
partition metadata from an independently qualified 35B live/source manifest.
"""
from dataclasses import dataclass
from bf16_sr_coordinates_candidate import SRCoordinates


@dataclass(frozen=True)
class FixedTP4Coordinates(SRCoordinates):
    logical_name: str = ''
    sharded: bool = False
    group_size: int = 4
    group_rank: int = 0
    metadata_axis: int | None = None
    metadata_stride: int | None = None

    def validate(self):
        super().validate()
        if not self.logical_name or type(self.sharded) is not bool:
            raise ValueError('logical name and explicit category required')
        if self.group_size!=4 or type(self.group_rank) is not int or not 0<=self.group_rank<4:
            raise ValueError('fixed TP4 rank/size required')
        if self.axis is not None or self.global_shape!=self.local_shape:
            raise ValueError('fixed shard key requires local flat coordinates')
        if self.sharded:
            if type(self.metadata_axis) is not int or not 0<=self.metadata_axis<len(self.local_shape):
                raise ValueError('actual partition axis required')
            if self.metadata_stride is not None and (type(self.metadata_stride) is not int or self.metadata_stride<1):
                raise ValueError('actual partition stride must be positive or absent')
            expected=self.logical_name+'/fixed-tp4-shard-'+str(self.group_rank)
        else:
            if self.metadata_axis is not None or self.metadata_stride is not None:
                raise ValueError('replicated parameter has no TP shard geometry')
            expected=self.logical_name+'/fixed-tp4-replica'
        if self.canonical_name!=expected:raise ValueError('fixed topology key mismatch')
        return self


def make_fixed_tp4_coordinates(logical_name,shape,*,sharded,group_rank,axis=None,stride=None):
    if not isinstance(shape,tuple):raise ValueError('explicit local shape tuple required')
    suffix='/fixed-tp4-shard-'+str(group_rank) if sharded else '/fixed-tp4-replica'
    return FixedTP4Coordinates(logical_name+suffix,shape,shape,logical_name=logical_name,
        sharded=sharded,group_rank=group_rank,metadata_axis=axis,metadata_stride=stride).validate()

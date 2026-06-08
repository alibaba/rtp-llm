from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FlydslMoeBatchRangeTuning:
    min_m: int
    max_m: Optional[int]
    grouped_tile_m: int

    def matches(self, batch_m: int) -> bool:
        return batch_m >= self.min_m and (self.max_m is None or batch_m <= self.max_m)


@dataclass(frozen=True)
class FlydslMoeShapeTuning:
    max_m: int
    grouped_route_min_b: int
    grouped_tile_m_ranges: tuple[FlydslMoeBatchRangeTuning, ...] = ()

    def grouped_tile_m(self, batch_m: int, default: int) -> int:
        for batch_tuning in self.grouped_tile_m_ranges:
            if batch_tuning.matches(batch_m):
                return batch_tuning.grouped_tile_m
        return default


# Qwen3.5-397B-A17B PTPC-FP8 direct-routing tuning table.
# Key is per-rank inter_dim after TP sharding.
_QWEN_PTPC_FP8_TUNING_BY_INTER_DIM = {
    # TP=4. Op tests show FlyDSL wins through M=512 and regresses at M=513.
    256: FlydslMoeShapeTuning(
        max_m=512,
        grouped_route_min_b=8,
        grouped_tile_m_ranges=(
            FlydslMoeBatchRangeTuning(min_m=8, max_m=2047, grouped_tile_m=16),
            FlydslMoeBatchRangeTuning(min_m=2048, max_m=None, grouped_tile_m=64),
        ),
    ),
    # TP=8. No negative boundary was found in the measured range.
    128: FlydslMoeShapeTuning(
        max_m=0,
        grouped_route_min_b=32,
        grouped_tile_m_ranges=(
            FlydslMoeBatchRangeTuning(min_m=128, max_m=1023, grouped_tile_m=16),
            FlydslMoeBatchRangeTuning(min_m=1024, max_m=2047, grouped_tile_m=32),
            FlydslMoeBatchRangeTuning(min_m=2048, max_m=None, grouped_tile_m=64),
        ),
    ),
}


def get_qwen_ptpc_fp8_tuning(inter_dim: int) -> Optional[FlydslMoeShapeTuning]:
    return _QWEN_PTPC_FP8_TUNING_BY_INTER_DIM.get(inter_dim)

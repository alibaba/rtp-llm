"""Pipeline-parallel layout helpers.

Single source of truth for the PP layer partition in Python: the partition
is DECIDED ONCE here (resolve_pp_partition), materialized as per-stage
layer counts on ParallelismConfig.pp_stage_layer_counts, and consumed
purely as data everywhere else — weight loading
(LoadConfig.pp_layer_range / has_pp_embedding / has_pp_lm_head), model
construction (GptModelBase.pp_layer_ids / pp_has_embedding /
pp_has_lm_head), cache geometry and the C++ side
(rtp_llm/cpp/config/RankLayout.h, prefix-sum lookup). Consumers never
re-derive the partition rule (see stage_layer_range).
"""

from enum import Enum
from typing import Callable, List, Optional

from rtp_llm.models_py.distributed.rank_layout import RankLayout


class ModuleKind(Enum):
    """Non-layer modules whose owning stage is decided by placement rules."""

    EMBEDDING = "EMBEDDING"
    LM_HEAD = "LM_HEAD"
    MTP = "MTP"


class MtpPlacementStrategy(Enum):
    PIN_LAST_STAGE = "PIN_LAST_STAGE"  # MTP sits with lm_head/sampler on the last stage
    DISTRIBUTED = "DISTRIBUTED"  # reserved, not implemented


class ModulePlacement:
    """Module -> owning-stage view.

    Data sources: pp_size + materialized layer partition + MTP strategy.
    Current placement rules are fully derivable from these (pure function);
    if a non-derivable placement shape appears later, switch to consuming
    materialized placement data without changing this API.
    """

    def __init__(
        self,
        pp_size: int = 1,
        layer_counts: Optional[List[int]] = None,
        num_layers: Optional[int] = None,
        has_mtp: bool = False,
        mtp_strategy: MtpPlacementStrategy = MtpPlacementStrategy.PIN_LAST_STAGE,
    ):
        if has_mtp and mtp_strategy is MtpPlacementStrategy.DISTRIBUTED:
            raise NotImplementedError("DISTRIBUTED MTP placement is not implemented")
        self._pp_size = max(int(pp_size or 1), 1)
        self._layer_counts = list(layer_counts) if layer_counts else None
        self._num_layers = num_layers
        self._has_mtp = bool(has_mtp)
        self._mtp_strategy = mtp_strategy

    @staticmethod
    def from_parallelism_config(cfg, has_mtp: bool = False) -> "ModulePlacement":
        return ModulePlacement(
            pp_size=getattr(cfg, "pp_size", 1),
            layer_counts=getattr(cfg, "pp_stage_layer_counts", None),
            has_mtp=has_mtp,
        )

    def owner_of(self, module: ModuleKind) -> int:
        if module is ModuleKind.EMBEDDING:
            return 0
        if module is ModuleKind.LM_HEAD:
            return self._pp_size - 1
        if module is ModuleKind.MTP:
            if not self._has_mtp:
                raise ValueError("MTP module queried but has_mtp is False")
            return self._pp_size - 1  # PIN_LAST_STAGE
        raise ValueError(f"unknown module: {module}")

    def owns(self, module: ModuleKind, pp_rank: int) -> bool:
        return self.owner_of(module) == pp_rank

    def layer_range(self, pp_rank: int) -> range:
        """Global layer ids owned by `pp_rank` (delegates to the
        materialized-partition lookup shared with weight loading)."""
        num_layers = self._num_layers
        if num_layers is None and self._layer_counts:
            num_layers = sum(self._layer_counts)
        if num_layers is None:
            raise ValueError("layer_range requires num_layers or layer_counts")
        return stage_layer_range(num_layers, self._pp_size, pp_rank, self._layer_counts)


def even_split_counts(num_layers: int, pp_size: int) -> List[int]:
    """Default partition: even split, remainder to the earlier stages.

    Returns the layer count of every stage in rank order (e.g. 65 layers,
    pp=4 -> [17, 16, 16, 16]).
    """
    base = num_layers // pp_size
    rem = num_layers % pp_size
    return [base + (1 if rank < rem else 0) for rank in range(pp_size)]


"""Model-level partitioner registry: extension point for shape-specialized partitions;
output travels through the materialized-counts channel, so C++ never sees the rule."""

# partition(num_layers, pp_size, model_config) -> per-stage layer counts
PpPartitioner = Callable[[int, int, object], List[int]]
_PP_PARTITIONERS: dict = {}


def register_pp_partitioner(model_type: str, partitioner: PpPartitioner) -> None:
    """Attach an optional model-level layer partitioner to a model type."""
    _PP_PARTITIONERS[model_type] = partitioner


def get_pp_partitioner(model_type: str) -> Optional[PpPartitioner]:
    return _PP_PARTITIONERS.get(model_type)


def resolve_pp_partition(
    num_layers: int,
    pp_size: int,
    model_config=None,
) -> List[int]:
    """Decide the final PP layer partition and return per-stage layer counts.

    Priority: model-registered partitioner > default even split. The result
    is validated (length, positivity, sum) and ready to be materialized on
    ParallelismConfig.pp_stage_layer_counts.
    """
    model_type = (
        getattr(model_config, "model_type", None) if model_config is not None else None
    )
    partitioner = get_pp_partitioner(model_type) if model_type else None
    if partitioner is not None:
        counts = list(partitioner(num_layers, pp_size, model_config))
    else:
        counts = even_split_counts(num_layers, pp_size)
    _check_partition_counts(counts, num_layers, pp_size)
    return counts


def _check_partition_counts(counts: List[int], num_layers: int, pp_size: int) -> None:
    if len(counts) != pp_size:
        raise ValueError(
            f"pp partition has {len(counts)} stages but pp_size={pp_size}: {counts}"
        )
    if any(c <= 0 for c in counts):
        raise ValueError(
            f"pp partition must give every stage at least one layer: {counts}"
        )
    if sum(counts) != num_layers:
        raise ValueError(
            f"pp partition sums to {sum(counts)} but num_layers={num_layers}: {counts}"
        )


def pp_layer_range_from_counts(counts: List[int], pp_rank: int) -> range:
    """Half-open global layer-id range owned by `pp_rank` under a
    materialized partition (prefix-sum lookup)."""
    if not (0 <= pp_rank < len(counts)):
        raise ValueError(f"pp_rank={pp_rank} out of range for partition {counts}")
    begin = sum(counts[:pp_rank])
    return range(begin, begin + counts[pp_rank])


def stage_layer_range(
    num_layers: int,
    pp_size: int,
    pp_rank: int,
    counts: Optional[List[int]] = None,
) -> range:
    """Single consumer entry for the stage layer range.

    Three cases only:
      - materialized partition present -> prefix-sum lookup;
      - pp_size=1 without materialized data (the normal single-stage
        deployment never materializes) -> trivially all layers;
      - pp_size>1 without materialized data -> a startup-path error, NOT a
        silent fallback (the decision point must have written the counts).
    """
    if counts:
        return pp_layer_range_from_counts(counts, pp_rank)
    if pp_size <= 1:
        return range(num_layers)
    raise ValueError(
        f"pp_size={pp_size} requires a materialized layer partition "
        "(pp_stage_layer_counts); it must be written by the startup decision point"
    )


def derive_pp_rank(world_rank: int, dp_size: int, tp_size: int) -> int:
    """Fallback pp_rank for configs that only carry sizes (fake configs in
    tests). Production configs carry pp_rank directly and never need this;
    the rank formula itself lives solely in RankLayout."""
    layout = RankLayout(dp_size=int(dp_size or 1), tp_size=int(tp_size or 1))
    return layout.coord_of_unchecked(int(world_rank or 0)).pp


def stage_has_embedding(pp_rank: int) -> bool:
    """Thin wrapper over ModulePlacement (kept for existing callsites);
    the first stage owns the token/positional embedding."""
    return ModulePlacement().owns(ModuleKind.EMBEDDING, pp_rank)


def stage_has_lm_head(pp_rank: int, pp_size: int) -> bool:
    """Thin wrapper over ModulePlacement (kept for existing callsites);
    the last stage owns lm_head + final_layernorm."""
    return ModulePlacement(pp_size=pp_size).owns(ModuleKind.LM_HEAD, pp_rank)

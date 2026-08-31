"""Pipeline-parallel layout helpers.

Single source of truth for the PP layer partition in Python: the partition
is DECIDED ONCE here (resolve_pp_partition), materialized as per-stage
layer counts on ParallelismConfig.pp_stage_layer_counts, and consumed
purely as data everywhere else — weight loading
(LoadConfig.pp_layer_range / has_pp_embedding / has_pp_lm_head), model
construction (GptModelBase.pp_layer_ids / pp_has_embedding /
pp_has_lm_head), cache geometry and the C++ side
(rtp_llm/cpp/config/PPLayout.h, prefix-sum lookup). Consumers never
re-derive the partition rule (see stage_layer_range).
"""

from typing import Callable, List, Optional


def even_split_counts(num_layers: int, pp_size: int) -> List[int]:
    """Default partition: even split, remainder to the earlier stages.

    Returns the layer count of every stage in rank order (e.g. 65 layers,
    pp=4 -> [17, 16, 16, 16]).
    """
    base = num_layers // pp_size
    rem = num_layers % pp_size
    return [base + (1 if rank < rem else 0) for rank in range(pp_size)]


# ---------------------------------------------------------------------------
# Model-level partitioner registry (extension point for shape-specialized
# partitions of irregular models). A partitioner is Python-only; its output
# travels through the materialized-counts channel, so C++ never needs to
# understand the partitioning rule.
# ---------------------------------------------------------------------------

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
    tests). PP is the outermost dim of the world-rank layout:
    world_rank = pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank.
    Production configs carry pp_rank directly and never need this."""
    dp_size = max(int(dp_size or 1), 1)
    tp_size = max(int(tp_size or 1), 1)
    return int(world_rank or 0) // (dp_size * tp_size)


def stage_has_embedding(pp_rank: int) -> bool:
    """The first stage owns the token/positional embedding."""
    return pp_rank == 0


def stage_has_lm_head(pp_rank: int, pp_size: int) -> bool:
    """The last stage owns lm_head + final_layernorm. With pp_size=1 the
    single stage is both first and last."""
    return pp_rank == pp_size - 1

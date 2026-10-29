"""One PP partition keeps loading and execution on the same layers."""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import ParallelismConfig


def even_split_counts(num_layers: int, pp_size: int) -> List[int]:
    """Default partition: even split, remainder to the earlier stages.

    Returns the layer count of every stage in rank order (e.g. 65 layers,
    pp=4 -> [17, 16, 16, 16]).
    """
    base = num_layers // pp_size
    rem = num_layers % pp_size
    return [base + (1 if rank < rem else 0) for rank in range(pp_size)]


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


def finalize_pp_partition(
    parallelism_config: "ParallelismConfig", num_layers: int
) -> None:
    """Validate or generate stage counts against the final main-model depth."""
    if parallelism_config.pp_size <= 1:
        return
    counts = parallelism_config.pp_stage_layer_counts
    if not counts:
        counts = even_split_counts(num_layers, parallelism_config.pp_size)
    _check_partition_counts(counts, num_layers, parallelism_config.pp_size)
    parallelism_config.pp_stage_layer_counts = counts


def stage_layer_range(counts: List[int], pp_rank: int) -> range:
    """Half-open global layer-id range owned by `pp_rank` under a
    materialized partition (prefix-sum lookup)."""
    begin = sum(counts[:pp_rank])
    return range(begin, begin + counts[pp_rank])


def local_layer_range(
    model_config: "ModelConfig", parallelism_config: "ParallelismConfig"
) -> range:
    """Drafts share the physical topology but own their full layer range."""
    if model_config.is_mtp or parallelism_config.pp_size <= 1:
        return range(model_config.num_layers)
    return stage_layer_range(
        parallelism_config.pp_stage_layer_counts,
        parallelism_config.pp_rank,
    )


def is_last_stage(pp_rank: int, pp_size: int) -> bool:
    return pp_rank == pp_size - 1

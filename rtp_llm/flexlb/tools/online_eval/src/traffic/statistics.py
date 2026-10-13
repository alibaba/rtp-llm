"""Observed arrival, length and structural sharing statistics."""
import math
from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE, prefix_block_size

def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return dict(count=len(ordered), min=ordered[0], max=ordered[-1],
                mean=sum(ordered) / len(ordered),
                **{f"p{p}": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
                   for p in (50, 90, 95, 99)})


def statistics(events, *, version=2, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    """Nearest-rank percentiles; sharing describes the DAG, not cache hits."""
    block_size = prefix_block_size(block_size)
    duration = (events[-1][0] - events[0][0]) / 1000
    counts = [0] * (int(duration) + 1)
    for event in events:
        counts[(event[0] - events[0][0]) // 1000] += 1
    resolution = 1 if version == 3 else block_size
    inputs = [event[1] * resolution for event in events]
    shared = [event[3] * block_size for event in events]
    return dict(
        scope="captured_requests_only_no_missing_shard_extrapolation",
        request_count=len(events), percentile_method="nearest_rank",
        arrival=dict(event_span_s=duration,
                     mean_qps=len(events) / duration if duration else None,
                     requests_per_second=distribution(counts),
                     bucket_origin="first_event", bucket_ms=1000,
                     final_bucket="included_even_if_partial", idle_buckets_included=True),
        input_tokens=dict(**distribution(inputs), total=sum(inputs), resolution_tokens=resolution),
        prefix_structure=dict(shared_tokens=distribution(shared),
                              token_weighted_shared_fraction=sum(shared) / sum(inputs),
                              requests_with_shared_prefix=sum(value > 0 for value in shared),
                              parentless_requests=sum(event[2] == -1 for event in events),
                              interpretation="structural_sharing_not_observed_cache_hit_rate"),
        output_tokens=None, outcomes=None)

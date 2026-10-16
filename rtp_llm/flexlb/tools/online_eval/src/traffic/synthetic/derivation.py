"""Derive statistical source parameters from a pinned lineage model + fit report.

Outputs remain independently specified: the capture is error censored. Session
turns are not identifiable here and default to a declared disabled component.
"""
from traffic.lineage_model import LineageModel
from traffic.structure import joint_distribution

def lineage_sizes(model: LineageModel):
    """Subtree request counts of the empirical prefix-lineage forest, descending.

    The codec stores each request's longest-matching predecessor, so the reuse
    unit a scheduler observes is a lineage tree; zipf concentration is fitted
    against that distribution.
    """
    events = model.events
    subtree = [1] * len(events)
    for i in range(len(events) - 1, -1, -1):
        parent = events[i].parent_index
        if parent >= 0:
            subtree[parent] += subtree[i]
    return sorted((subtree[i] for i, event in enumerate(events) if event.parent_index == -1), reverse=True)


def derive_parameters(model: LineageModel, report, report_sha, *, model_sha256):
    block_size = model.metadata.block_size
    events = model.events
    sizes = sorted(event.input_tokens for event in events)
    # A dense empirical inverse CDF preserves the long tail and mean better than
    # interpolating three percentiles. Deterministic, bounded profile size.
    values=[sizes[int(q*(len(sizes)-1)/1000)] for q in range(1001)]
    family_counts=lineage_sizes(model)
    prefix_depths=[event.shared_full_blocks for event in events if event.shared_full_blocks]
    cold=sum(e.shared_full_blocks == 0 for e in events)/len(events)
    families=max(5,len(family_counts))
    top5=sum(family_counts[:5])/len(events)
    lo,hi=0.0,4.0
    for _ in range(40):
        alpha=(lo+hi)/2
        share=sum((i+1)**-alpha for i in range(5))/sum((i+1)**-alpha for i in range(families))*(1-cold)
        if share < top5:lo=alpha
        else:hi=alpha
    depth=sorted(prefix_depths)[len(prefix_depths)//2] if prefix_depths else 1
    # Prefix depth floor is 4K; concentration below uses lineage trees, not 4K families.
    depth=max(4096 // block_size,depth)
    return dict(schema_version=2, data_kind='synthetic', generator=dict(kind='synthetic',model='realistic',version='1'),
        parameters=dict(block_size=block_size,families=families,shared_blocks=0,
        prefix_blocks=depth,suffix_blocks=1,zipf_alpha=(lo+hi)/2,cold_fraction=cold,
        session_requests=1,session_growth_blocks=0,
        input_distribution=dict(values=values,weights=[1]*len(values)),
        joint_distribution=joint_distribution(model)),
        calibration=dict(model_sha256=model_sha256,fit_report_sha256=report_sha,
            provenance=model.metadata.provenance,held_out_validated=False,
            family_statistic="lineage_tree_request_count",
            structure_statistics=dict(lineage_tree_count=len(family_counts)),
            targets=dict(mean_input_tokens=sum(sizes)/len(sizes),p99_input_tokens=values[990],
                lineage_tree_top5_share=top5,cold_fraction=cold),
            limitations=['session turns not identifiable; disabled until explicitly configured',
                'output distribution independently specified; capture error censored',
                'statistical family model does not preserve empirical pairwise reuse topology',
                'arrival curve retained for playback configuration, not source scheduling'],
            source_minute_windows=report.get('source_minute_windows',{}),
            nearest_prefix_repeat_gap_ms=report.get('nearest_prefix_repeat_gap_ms',{})))

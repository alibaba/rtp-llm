"""Derive statistical source parameters from a pinned lineage model + fit report.

Outputs remain independently specified: the capture is error censored. Session
turns are not identifiable here and default to a declared disabled component.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from traffic.prefix_lineage import decode, expand


def calibrate(raw, report, report_sha):
    metadata, events=decode(raw)
    sizes=sorted(e[1]*512 for e in events)
    # A dense empirical inverse CDF preserves the long tail and mean better than
    # interpolating three percentiles. Deterministic, bounded profile size.
    values=[sizes[int(q*(len(sizes)-1)/1000)] for q in range(1001)]
    counts=collections.Counter()
    prefix_depths=[]
    for event, (_,labels) in zip(events,expand(events)):
        counts[tuple(labels[:8])] += 1
        if event[3]:prefix_depths.append(event[3])
    cold=sum(e[3]==0 for e in events)/len(events)
    families=max(5,len(counts))
    top5=sum(v for _,v in counts.most_common(5))/len(events)
    lo,hi=0.0,4.0
    for _ in range(40):
        alpha=(lo+hi)/2
        share=sum((i+1)**-alpha for i in range(5))/sum((i+1)**-alpha for i in range(families))*(1-cold)
        if share < top5:lo=alpha
        else:hi=alpha
    depth=sorted(prefix_depths)[len(prefix_depths)//2] if prefix_depths else 1
    # Family concentration is defined at 4K in the empirical report.
    depth=max(8,depth)
    return dict(data_kind='synthetic', generator=dict(kind='synthetic',model='realistic',version='1'),
        parameters=dict(block_size=512,families=families,shared_blocks=0,
        prefix_blocks=depth,suffix_blocks=1,zipf_alpha=(lo+hi)/2,cold_fraction=cold,
        session_requests=1,session_growth_blocks=0,
        input_distribution=dict(values=values,weights=[1]*len(values))),
        calibration=dict(model_sha256=hashlib.sha256(raw).hexdigest(),fit_report_sha256=report_sha,
            provenance=metadata['provenance'],held_out_validated=False,
            targets=dict(mean_input_tokens=sum(sizes)/len(sizes),p99_input_tokens=values[990],
                family_top5_share=top5,cold_fraction=cold),
            limitations=['session turns not identifiable; disabled until explicitly configured',
                'output distribution independently specified; capture error censored',
                'statistical family model does not preserve empirical pairwise reuse topology',
                'arrival curve retained for playback configuration, not source scheduling'],
            source_minute_windows=report.get('source_minute_windows',{}),
            nearest_prefix_repeat_gap_ms=report.get('nearest_prefix_repeat_gap_ms',{})))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--fit-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();raw=a.fit_report.read_bytes()
    result=calibrate(a.model.read_bytes(),json.loads(raw),hashlib.sha256(raw).hexdigest())
    result['source_capture']=os.path.relpath(a.model.resolve(), a.out.resolve().parent)
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

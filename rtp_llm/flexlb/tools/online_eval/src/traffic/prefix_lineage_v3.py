"""Exact-length empirical lineage; filter only when projecting a replay plan."""
import hashlib
import json
import lzma
from pathlib import Path

from traffic.prefix_lineage import BLOCK, MAX_EVENTS
from traffic.output_sampling import output_sampler, output_semantics


def validate(events):
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError('invalid lineage event count')
    previous = 0
    for i, event in enumerate(events):
        if len(event) != 4 or any(type(v) is not int for v in event):
            raise ValueError('invalid lineage event')
        ts, length, parent, shared = event
        if not previous <= ts <= 9223372036854775807 or not 1 <= length <= 2147483647:
            raise ValueError('invalid lineage timestamp/length')
        if not -1 <= parent < i or not 0 <= shared <= length // BLOCK:
            raise ValueError('invalid lineage reference')
        if (parent == -1 and shared) or (parent >= 0 and shared > events[parent][1] // BLOCK):
            raise ValueError('invalid lineage shared prefix')
        previous = ts


def encode(events, provenance=None):
    validate(events)
    metadata = dict(version=3, block_size=BLOCK, count=len(events),
                    realism='EMPIRICAL_PREFIX_STRUCTURE', tail='EXACT_LENGTH_PRIVATE_PARTIAL_BLOCK',
                    arrival='TRUE_TIMESTAMPS_PACED_BY_CLIENT', provenance=provenance or {})
    return lzma.compress(json.dumps(dict(metadata=metadata, events=events),
                                   separators=(',', ':'), allow_nan=False).encode(), preset=6)


def decode(raw):
    decoder = lzma.LZMADecompressor(memlimit=128*1024*1024)
    data = decoder.decompress(raw, max_length=128*1024*1024)
    if not decoder.eof or decoder.unused_data:
        raise ValueError('invalid or oversized lineage v3 envelope')
    model = json.loads(data)
    metadata, events = model['metadata'], model['events']
    if metadata.get('version') != 3 or metadata.get('block_size') != BLOCK or metadata.get('count') != len(events):
        raise ValueError('invalid lineage v3 header')
    validate(events)
    return metadata, events


def write_trace(path, parameters, namespace, base_dir, *, max_requests=None):
    """Compatibility entrypoint; content projection lives in lineage_transforms."""
    from traffic.lineage_transforms import write_trace as project
    return project(path, parameters, namespace, base_dir, decode=decode,
                   max_requests=max_requests)

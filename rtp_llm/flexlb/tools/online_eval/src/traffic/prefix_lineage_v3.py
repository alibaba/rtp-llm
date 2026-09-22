"""Exact-length empirical lineage; filter only when projecting a replay plan."""
import hashlib
import json
import lzma
import math
from pathlib import Path

from traffic.prefix_lineage import BLOCK, MAX_EVENTS


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


def output_sampler(parameters):
    """Replay lengths are explicit: sample once by original event index, not run ID."""
    cap = parameters['output_tokens']
    policy = parameters.get('output_distribution')
    if policy is None:
        return lambda index: cap
    if not isinstance(policy, dict) or set(policy) != {'kind', 'mean_tokens', 'seed'}:
        raise ValueError('invalid output distribution fields')
    mean, seed = policy['mean_tokens'], policy['seed']
    if (policy['kind'] != 'geometric' or type(mean) not in (int, float)
            or not math.isfinite(mean) or mean < 1
            or type(seed) is not int or not -(1 << 63) <= seed < (1 << 63)):
        raise ValueError('invalid output distribution parameters')
    def sample(index):
        if mean == 1:
            return 1
        mask = (1 << 64) - 1
        z = (index + seed + 0x9e3779b97f4a7c15) & mask
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        z ^= z >> 31
        uniform = (z >> 11) * 2.0 ** -53
        return min(cap, 1 + math.floor(math.log1p(-uniform) / math.log1p(-1 / mean)))
    return sample


def write_trace(path, parameters, namespace, base_dir, *, max_requests=None):
    p = parameters
    required = {'path', 'sha256', 'count', 'output_tokens', 'priority'}
    if not required <= set(p) or set(p) - required - {'max_input_tokens', 'output_distribution'}:
        raise ValueError('lineage v3 requires pinned model and output settings')
    cap = p.get('max_input_tokens', 2147483647)
    if type(cap) is not int or not 1 <= cap <= 2147483647:
        raise ValueError('invalid playback input length filter')
    if type(p['output_tokens']) is not int or not 1 <= p['output_tokens'] <= 2147483647 or type(p['priority']) is not int or not 1 <= p['priority'] <= 100:
        raise ValueError('invalid output length/priority')
    sample_output = output_sampler(p)
    raw = (Path(base_dir) / p['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != p['sha256']:
        raise ValueError('lineage model checksum mismatch')
    metadata, events = decode(raw)
    if type(p['count']) is not int or p['count'] != len(events):
        raise ValueError('lineage model count mismatch')
    paths, next_label, selected, excluded = [], 1, 0, 0
    with Path(path).open('w') as out:
        for i, (ts, length, parent, shared) in enumerate(events):
            # Expand excluded parents too: filtering must never break prefix identity.
            labels = paths[parent][:shared] if parent >= 0 else []
            fresh = (length + BLOCK - 1) // BLOCK - shared
            if next_label + fresh > 2147483647:
                raise ValueError('lineage label budget exceeded')
            labels.extend(range(next_label, next_label + fresh))
            next_label += fresh
            paths.append(labels)
            if length > cap:
                excluded += 1
                continue
            out.write(json.dumps(dict(rid=f'{namespace}:{i}', ts=ts, il=length,
                ol=sample_output(i), priority=p['priority'], cache_key_block_size=BLOCK,
                input_token_blocks=labels), separators=(',', ':')) + '\n')
            selected += 1
            if max_requests is not None and selected >= max_requests:
                break
    if not selected:
        raise ValueError('empty playback length selection')
    return dict(realism=metadata['realism'], tail=metadata['tail'], arrival=metadata['arrival'],
                provenance=metadata['provenance'], model_sha256=p['sha256'],
                length_filter=dict(max_input_tokens=cap, selected=selected, excluded_before_limit=excluded),
                output_semantics=('EXPLICIT_GEOMETRIC_SPLITMIX64_EVENT_INDEX_V1' if p.get('output_distribution')
                                  else 'INDEPENDENT_FIXED_CAP_NOT_FITTED_FROM_ERROR_OUTPUTS'),
                output_distribution=p.get('output_distribution'), output_cap=p['output_tokens'])

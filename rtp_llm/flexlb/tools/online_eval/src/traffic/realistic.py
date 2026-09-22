"""Statistical prefix/session generator; arrival pacing is owned by the client.

An explicit shape is an experimental workload, not a production claim. The
bundled defaults are derived from a pinned empirical model; no held-out
validation or inference of error-censored output lengths is claimed.
"""
import bisect
import json
import math
import random
from itertools import accumulate
from pathlib import Path

from traffic.datasets import DEFAULT_PROFILE, profile_path

FIELDS = {'seed','count','block_size','families','shared_blocks','prefix_blocks','suffix_blocks',
          'zipf_alpha','cold_fraction','session_requests','session_growth_blocks','output_tokens',
          'priority','input_distribution','output_distribution','pinned_blocks','profile','sampling','joint_distribution'}


def distribution(spec):
    if not isinstance(spec, dict) or set(spec) != {'values','weights'}:
        raise ValueError('distribution requires values and weights')
    values, weights = spec['values'], spec['weights']
    if not isinstance(values, list) or not isinstance(weights, list) or not values or len(values) != len(weights):
        raise ValueError('invalid distribution dimensions')
    if any(type(v) is not int or not 1 <= v <= 2147483647 for v in values):
        raise ValueError('distribution values must be positive int32')
    if any(type(w) not in (int,float) or not math.isfinite(w) or w < 0 for w in weights) or not 0 < sum(weights) < float('inf'):
        raise ValueError('invalid distribution weights')
    return spec


def resolve(parameters, *, document=None):
    if not isinstance(parameters, dict) or set(parameters)-FIELDS:
        raise ValueError('unknown realistic parameter; arrival controls belong to client')
    if type(parameters.get('seed')) is not int or type(parameters.get('count')) is not int or not 1 <= parameters['count'] <= 1000000:
        raise ValueError('explicit integer seed and bounded count required')
    p = dict(parameters)
    profile = p.pop('profile', DEFAULT_PROFILE if 'families' not in p else None)
    provenance = None
    if profile:
        document = document if document is not None else json.loads(profile_path(profile).read_text())
        if document.get('schema_version', 1) not in (1, 2):
            raise ValueError('unsupported calibration schema')
        provenance = document['calibration']
        p = dict(document['parameters'], **p)
        if set(p)-FIELDS:
            raise ValueError('unknown parameter in calibration profile')
    p.setdefault('block_size',512)
    p.setdefault('priority',50)
    p.setdefault('shared_blocks',0)
    p.setdefault('suffix_blocks',1)
    p.setdefault('session_requests',1)
    p.setdefault('session_growth_blocks',0)
    p.setdefault('pinned_blocks',[])
    for k in ('families','prefix_blocks','zipf_alpha','cold_fraction'):
        if k not in p:
            raise ValueError('explicit shape or calibration profile required: '+k)
    if p['block_size'] not in (512,1024):
        raise ValueError('realistic block_size supports 512 or compatibility 1024')
    for k in ('families','shared_blocks','prefix_blocks','suffix_blocks','session_requests','session_growth_blocks','priority'):
        if type(p[k]) is not int or p[k] < 0:
            raise ValueError('invalid '+k)
    if not 1 <= p['families'] <= 100000 or p['session_requests'] < 1 or not 1 <= p['priority'] <= 100:
        raise ValueError('invalid family/session/priority')
    for k, hi in (('zipf_alpha',4),('cold_fraction',1)):
        if type(p[k]) not in (int,float) or not math.isfinite(p[k]) or not 0 <= p[k] <= hi:
            raise ValueError('invalid '+k)
    if 'output_distribution' in p and 'output_tokens' in p:
        raise ValueError('specify exactly one output distribution or output_tokens')
    if 'output_tokens' in p:
        p['output_distribution'] = dict(values=[p.pop('output_tokens')],weights=[1])
    if 'output_distribution' not in p:
        raise ValueError('output_distribution must be specified independently of the censored capture')
    distribution(p['output_distribution'])
    if 'input_distribution' in p:
        distribution(p['input_distribution'])
    pinned = p['pinned_blocks']
    if not isinstance(pinned,list) or any(not isinstance(block,list) or len(block) != p['block_size'] or any(type(t) is not int or not 0 <= t <= 2147483647 for t in block) for block in pinned):
        raise ValueError('pinned_blocks must contain exact full blocks of int32 tokens')
    if 'input_distribution' in p and pinned and min(p['input_distribution']['values']) < len(pinned)*p['block_size']:
        raise ValueError('input distribution truncates pinned content')
    p.setdefault('sampling', 'independent')
    if p['sampling'] not in ('independent', 'joint'):
        raise ValueError('unknown sampling method')
    if p['sampling'] == 'joint':
        joint = p.get('joint_distribution')
        if not isinstance(joint, dict) or joint.get('schema_version') != 1:
            raise ValueError('joint sampling requires a schema 2 calibrated joint distribution')
        warm, cold = joint.get('warm_pairs'), joint.get('cold_blocks')
        if not isinstance(warm, list) or not isinstance(cold, list):
            raise ValueError('joint pools must be arrays')
        if (p['cold_fraction'] < 1 and not warm) or (p['cold_fraction'] > 0 and not cold):
            raise ValueError('joint pool empty for selected cold fraction')
        if any(not isinstance(v, (list, tuple)) or len(v) != 2 or any(type(x) is not int for x in v)
               or not 1 <= v[1] <= v[0] <= 2147483647//512 for v in warm):
            raise ValueError('invalid joint warm pair')
        if any(type(v) is not int or not 1 <= v <= 2147483647//512 for v in cold):
            raise ValueError('invalid joint cold length')
        if p['block_size'] != 512 or pinned or p['shared_blocks'] or p['session_growth_blocks'] or p['session_requests'] != 1:
            raise ValueError('joint sampling does not support pinned/global/session overrides')
        if 'input_distribution' in parameters or 'prefix_blocks' in parameters or 'suffix_blocks' in parameters:
            raise ValueError('joint sampling controls lengths and depths; independent shape overrides are invalid')
    return p, provenance


def iter_requests(p, namespace, *, max_requests=None):
    rng = random.Random(p['seed'])
    # Separate streams: changing output lengths cannot alter input reuse.
    output_rng = random.Random(p['seed'] ^ 0x5D47A319)
    output_cumulative = list(accumulate(p['output_distribution']['weights']))
    input_cumulative = list(accumulate(p['input_distribution']['weights'])) if 'input_distribution' in p else None
    weights = [(i+1)**-p['zipf_alpha'] for i in range(p['families'])]
    cumulative = []
    for w in weights:
        cumulative.append(w+(cumulative[-1] if cumulative else 0))
    next_label = max((t for block in p['pinned_blocks'] for t in block), default=0)+1
    def fresh(n):
        nonlocal next_label
        if n < 0 or next_label+n > 2147483647:
            raise ValueError('token label budget exceeded')
        labels = list(range(next_label,next_label+n))
        next_label += n
        return labels
    shared = p['pinned_blocks'] + fresh(p['shared_blocks'])
    prefixes = ([[] for _ in weights] if p['sampling'] == 'joint' else
                [shared+fresh(p['prefix_blocks']) for _ in weights])
    sessions = [[] for _ in weights]
    rounds = [0 for _ in weights]
    for i in range(min(p['count'], max_requests) if max_requests is not None else p['count']):
        cold = rng.random() < p['cold_fraction']
        family = bisect.bisect_right(cumulative,rng.random()*cumulative[-1])
        if p['sampling'] == 'joint':
            pool = p['joint_distribution']
            if cold:
                n = rng.choice(pool['cold_blocks'])
                labels = fresh(n)
            else:
                n, depth = rng.choice(pool['warm_pairs'])
                prefix = prefixes[family]
                prefix.extend(fresh(max(0, depth-len(prefix))))
                labels = prefix[:depth] + fresh(n-depth)
            il = n*p['block_size']
        elif cold:
            labels = p['pinned_blocks'] + fresh(p['shared_blocks']+p['prefix_blocks'])
        else:
            if rounds[family] % p['session_requests'] == 0:
                sessions[family] = []
            labels = prefixes[family]+sessions[family]
            sessions[family] += fresh(p['session_growth_blocks'])
            rounds[family] += 1
        if p['sampling'] == 'joint':
            pass
        elif 'input_distribution' in p:
            d=p['input_distribution'];il=rng.choices(d['values'],cum_weights=input_cumulative)[0]
            n=(il+p['block_size']-1)//p['block_size']
            # Retain a private suffix when the shape permits one.
            keep=max(len(p['pinned_blocks']),n-p['suffix_blocks'])
            labels=labels[:keep]+fresh(max(0,n-min(len(labels),keep)))
        else:
            labels=labels+fresh(p['suffix_blocks']);il=len(labels)*p['block_size']
        if not 1 <= il <= 2147483647:
            raise ValueError('invalid generated input length')
        d=p['output_distribution'];ol=output_rng.choices(d['values'],cum_weights=output_cumulative)[0]
        yield dict(rid=f'{namespace}:{i}',ts=i,il=il,ol=ol,
            input_token_blocks=labels,priority=p['priority'],cache_key_block_size=p['block_size'],
            family='cold' if cold else str(family))


def write_trace(path, parameters, namespace, base_dir=None, *, max_requests=None):
    p, calibration = resolve(parameters)
    with Path(path).open('w') as out:
        for row in iter_requests(p, namespace, max_requests=max_requests):
            out.write(json.dumps(row, separators=(',', ':')) + '\n')
    return dict(realism='CALIBRATED_STATISTICAL' if calibration else 'NOT_VALIDATED',
                arrival='ORDINAL_PACED_BY_CLIENT',tail='EXACT_CONFIGURED_LENGTHS',
                calibration=calibration,held_out_validated=False,sampling=p['sampling'],
                output_model='INDEPENDENT_EXPLICIT_DISTRIBUTION')

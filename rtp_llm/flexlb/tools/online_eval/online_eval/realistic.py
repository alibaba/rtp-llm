"""Statistical prefix/session generator; arrival pacing is owned by the client.

An explicit shape is an experimental workload, not a production claim. The
bundled defaults are derived from a pinned empirical model; no held-out
validation or inference of error-censored output lengths is claimed.
"""
import bisect
import hashlib
import json
import math
import random
from pathlib import Path

PROFILE = Path(__file__).with_name('traffic_profiles') / 'frontend_20260921.json'
FIELDS = {'seed','count','block_size','families','shared_blocks','prefix_blocks','suffix_blocks',
          'zipf_alpha','cold_fraction','session_requests','session_growth_blocks','output_tokens',
          'priority','input_distribution','output_distribution','pinned_blocks','profile'}


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


def resolve(parameters):
    if not isinstance(parameters, dict) or set(parameters)-FIELDS:
        raise ValueError('unknown realistic parameter; arrival controls belong to client')
    if type(parameters.get('seed')) is not int or type(parameters.get('count')) is not int or not 1 <= parameters['count'] <= 1000000:
        raise ValueError('explicit integer seed and bounded count required')
    p = dict(parameters)
    profile = p.pop('profile', 'frontend_20260921' if 'families' not in p else None)
    provenance = None
    if profile:
        if profile != 'frontend_20260921':
            raise ValueError('unknown calibration profile')
        document = json.loads(PROFILE.read_text())
        provenance = document['calibration']
        p = dict(document['parameters'], **p)
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
    return p, provenance


def write_trace(path, parameters, namespace, base_dir=None):
    p, calibration = resolve(parameters)
    rng = random.Random(p['seed'])
    # Separate streams: changing output lengths cannot alter input reuse.
    output_rng = random.Random(p['seed'] ^ 0x5D47A319)
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
    prefixes = [shared+fresh(p['prefix_blocks']) for _ in weights]
    sessions = [[] for _ in weights]
    rounds = [0 for _ in weights]
    with Path(path).open('w') as out:
        for i in range(p['count']):
            cold = rng.random() < p['cold_fraction']
            family = bisect.bisect_right(cumulative,rng.random()*cumulative[-1])
            if cold:
                labels = p['pinned_blocks'] + fresh(p['shared_blocks']+p['prefix_blocks'])
            else:
                if rounds[family] % p['session_requests'] == 0:
                    sessions[family] = []
                labels = prefixes[family]+sessions[family]
                sessions[family] += fresh(p['session_growth_blocks'])
                rounds[family] += 1
            if 'input_distribution' in p:
                d=p['input_distribution'];il=rng.choices(d['values'],weights=d['weights'])[0]
                n=(il+p['block_size']-1)//p['block_size']
                # Retain a private suffix when the shape permits one.
                keep=max(len(p['pinned_blocks']),n-p['suffix_blocks'])
                labels=labels[:keep]+fresh(max(0,n-min(len(labels),keep)))
            else:
                labels=labels+fresh(p['suffix_blocks']);il=len(labels)*p['block_size']
            if not 1 <= il <= 2147483647:
                raise ValueError('invalid generated input length')
            d=p['output_distribution'];ol=output_rng.choices(d['values'],weights=d['weights'])[0]
            out.write(json.dumps(dict(rid=f'{namespace}:{i}',ts=i,il=il,ol=ol,
                input_token_blocks=labels,priority=p['priority'],cache_key_block_size=p['block_size'],
                family='cold' if cold else str(family)),separators=(',',':'))+'\n')
    return dict(realism='CALIBRATED_STATISTICAL' if calibration else 'NOT_VALIDATED',
                arrival='ORDINAL_PACED_BY_CLIENT',tail='EXACT_CONFIGURED_LENGTHS',
                calibration=calibration,held_out_validated=False,
                output_model='INDEPENDENT_EXPLICIT_DISTRIBUTION')

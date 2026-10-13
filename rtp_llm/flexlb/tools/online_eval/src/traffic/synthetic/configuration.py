"""Statistical prefix/session generator; arrival pacing is owned by the client.

An explicit shape is an experimental workload, not a production claim. The
bundled defaults are derived from a pinned empirical model; no held-out
validation or inference of error-censored output lengths is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
import json
import math

from traffic.request_rules import PRIORITY
from traffic.lineage_model import prefix_block_size

from traffic.catalog import DEFAULT_PROFILE, profile_path


@dataclass(frozen=True)
class WeightedDistribution:
    values: tuple[int, ...]
    cumulative_weights: tuple[float, ...]

    @classmethod
    def read(cls, value):
        distribution(value)
        return cls(tuple(value['values']), tuple(accumulate(value['weights'])))

    def sample(self, rng):
        return rng.choices(self.values, cum_weights=self.cumulative_weights)[0]


@dataclass(frozen=True)
class IndependentShape:
    shared_blocks: int
    prefix_blocks: int
    suffix_blocks: int
    session_requests: int
    session_growth_blocks: int
    pinned_blocks: tuple[tuple[int, ...], ...]
    input_distribution: WeightedDistribution | None


@dataclass(frozen=True)
class JointShape:
    warm_pairs: tuple[tuple[int, int], ...]
    cold_blocks: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticWorkload:
    seed: int
    count: int
    block_size: int
    priority: int
    families: int
    zipf_alpha: float
    cold_fraction: float
    output_distribution: WeightedDistribution
    shape: IndependentShape | JointShape

    @property
    def sampling(self):
        return 'joint' if isinstance(self.shape, JointShape) else 'independent'

    @classmethod
    def from_validated(cls, p):
        if p['sampling'] == 'joint':
            joint = p['joint_distribution']
            shape = JointShape(tuple(tuple(pair) for pair in joint['warm_pairs']), tuple(joint['cold_blocks']))
        else:
            shape = IndependentShape(p['shared_blocks'], p['prefix_blocks'], p['suffix_blocks'],
                p['session_requests'], p['session_growth_blocks'],
                tuple(tuple(block) for block in p['pinned_blocks']),
                WeightedDistribution.read(p['input_distribution']) if 'input_distribution' in p else None)
        return cls(p['seed'], p['count'], p['block_size'], p['priority'], p['families'],
                   p['zipf_alpha'], p['cold_fraction'], WeightedDistribution.read(p['output_distribution']), shape)


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


def resolve(parameters: dict, *, document=None) -> tuple[SyntheticWorkload, dict | None]:
    if not isinstance(parameters, dict) or set(parameters)-FIELDS:
        raise ValueError('unknown realistic parameter; arrival controls belong to client')
    if type(parameters.get('seed')) is not int or type(parameters.get('count')) is not int or not 1 <= parameters['count'] <= 1000000:
        raise ValueError('explicit integer seed and bounded count required')
    p = dict(parameters)
    profile = p.pop('profile', DEFAULT_PROFILE if 'families' not in p else None)
    provenance = None
    if profile:
        document = document if document is not None else json.loads(profile_path(profile).read_text())
        if not isinstance(document, dict) or type(document.get('schema_version', 1)) is not int or document.get('schema_version', 1) not in (1, 2):
            raise ValueError('unsupported synthetic parameter schema')
        if not isinstance(document.get('parameters'), dict) or not isinstance(document.get('calibration'), dict):
            raise ValueError('synthetic profile requires parameters and calibration mappings')
        provenance = document['calibration']
        p = dict(document['parameters'], **p)
        if set(p)-FIELDS:
            raise ValueError('unknown parameter in synthetic profile')
    p.setdefault('block_size',512)
    p.setdefault('priority',50)
    p.setdefault('shared_blocks',0)
    p.setdefault('suffix_blocks',1)
    p.setdefault('session_requests',1)
    p.setdefault('session_growth_blocks',0)
    p.setdefault('pinned_blocks',[])
    for k in ('families','prefix_blocks','zipf_alpha','cold_fraction'):
        if k not in p:
            raise ValueError('explicit shape or synthetic parameter profile required: '+k)
    p['block_size'] = prefix_block_size(p['block_size'])
    for k in ('families','shared_blocks','prefix_blocks','suffix_blocks','session_requests','session_growth_blocks','priority'):
        if type(p[k]) is not int or p[k] < 0:
            raise ValueError('invalid '+k)
    if not 1 <= p['families'] <= 100000 or p['session_requests'] < 1 or not PRIORITY.minimum <= p['priority'] <= PRIORITY.maximum:
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
        if not isinstance(joint, dict) or type(joint.get('schema_version')) is not int or joint.get('schema_version') != 1:
            raise ValueError('joint sampling requires a schema 2 derived joint distribution')
        warm, cold = joint.get('warm_pairs'), joint.get('cold_blocks')
        if not isinstance(warm, list) or not isinstance(cold, list):
            raise ValueError('joint pools must be arrays')
        if (p['cold_fraction'] < 1 and not warm) or (p['cold_fraction'] > 0 and not cold):
            raise ValueError('joint pool empty for selected cold fraction')
        if any(not isinstance(v, (list, tuple)) or len(v) != 2 or any(type(x) is not int for x in v)
               or not 1 <= v[1] <= v[0] <= 2147483647//p['block_size'] for v in warm):
            raise ValueError('invalid joint warm pair')
        if any(type(v) is not int or not 1 <= v <= 2147483647//p['block_size'] for v in cold):
            raise ValueError('invalid joint cold length')
        if pinned or p['shared_blocks'] or p['session_growth_blocks'] or p['session_requests'] != 1:
            raise ValueError('joint sampling does not support pinned/global/session overrides')
        if 'input_distribution' in parameters or 'prefix_blocks' in parameters or 'suffix_blocks' in parameters:
            raise ValueError('joint sampling controls lengths and depths; independent shape overrides are invalid')
    return SyntheticWorkload.from_validated(p), provenance

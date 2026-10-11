"""Seeded request generation; configuration decoding belongs to configuration.py."""
from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from traffic.synthetic.configuration import IndependentShape, JointShape, SyntheticWorkload, resolve


class LabelAllocator:
    def __init__(self, pinned_blocks=()):
        self.next_label = max((t for block in pinned_blocks for t in block), default=0)+1

    def fresh(self, count):
        if count < 0 or self.next_label+count > 2147483647:
            raise ValueError('token label budget exceeded')
        labels = list(range(self.next_label, self.next_label+count))
        self.next_label += count
        return labels


class IndependentSampler:
    def __init__(self, shape: IndependentShape, workload: SyntheticWorkload, allocator: LabelAllocator):
        self.shape, self.workload, self.allocator = shape, workload, allocator
        self.pinned = [list(block) for block in shape.pinned_blocks]
        shared = self.pinned + allocator.fresh(shape.shared_blocks)
        self.prefixes = [shared+allocator.fresh(shape.prefix_blocks) for _ in range(workload.families)]
        self.sessions = [[] for _ in self.prefixes]
        self.rounds = [0 for _ in self.prefixes]

    def sample(self, rng, cold, family):
        shape, allocator = self.shape, self.allocator
        if cold:
            labels = self.pinned + allocator.fresh(shape.shared_blocks+shape.prefix_blocks)
        else:
            if self.rounds[family] % shape.session_requests == 0:
                self.sessions[family] = []
            labels = self.prefixes[family]+self.sessions[family]
            self.sessions[family] += allocator.fresh(shape.session_growth_blocks)
            self.rounds[family] += 1
        if shape.input_distribution is not None:
            length = shape.input_distribution.sample(rng)
            blocks = (length+self.workload.block_size-1)//self.workload.block_size
            keep = max(len(self.pinned), blocks-shape.suffix_blocks)
            labels = labels[:keep]+allocator.fresh(max(0, blocks-min(len(labels), keep)))
        else:
            labels += allocator.fresh(shape.suffix_blocks)
            length = len(labels)*self.workload.block_size
        return length, labels


class JointSampler:
    def __init__(self, shape: JointShape, workload: SyntheticWorkload, allocator: LabelAllocator):
        self.shape, self.workload, self.allocator = shape, workload, allocator
        self.prefixes = [[] for _ in range(workload.families)]

    def sample(self, rng, cold, family):
        if cold:
            blocks = rng.choice(self.shape.cold_blocks)
            labels = self.allocator.fresh(blocks)
        else:
            blocks, depth = rng.choice(self.shape.warm_pairs)
            prefix = self.prefixes[family]
            prefix.extend(self.allocator.fresh(max(0, depth-len(prefix))))
            labels = prefix[:depth] + self.allocator.fresh(blocks-depth)
        return blocks*self.workload.block_size, labels


class StatisticalTrafficGenerator:
    def __init__(self, workload: SyntheticWorkload):
        self.workload = workload

    def requests(self, namespace, *, max_requests=None):
        p = self.workload
        rng = random.Random(p.seed)
        # Separate streams: changing output lengths cannot alter input reuse.
        output_rng = random.Random(p.seed ^ 0x5D47A319)
        cumulative = []
        for i in range(p.families):
            cumulative.append((i+1)**-p.zipf_alpha+(cumulative[-1] if cumulative else 0))
        shape = p.shape
        allocator = LabelAllocator(shape.pinned_blocks if isinstance(shape, IndependentShape) else ())
        sampler = (IndependentSampler(shape, p, allocator) if isinstance(shape, IndependentShape)
                   else JointSampler(shape, p, allocator))
        for i in range(min(p.count, max_requests) if max_requests is not None else p.count):
            cold = rng.random() < p.cold_fraction
            family = bisect.bisect_right(cumulative, rng.random()*cumulative[-1])
            length, labels = sampler.sample(rng, cold, family)
            if not 1 <= length <= 2147483647:
                raise ValueError('invalid generated input length')
            yield dict(rid=f'{namespace}:{i}', ts=i, il=length, ol=p.output_distribution.sample(output_rng),
                input_token_blocks=labels, priority=p.priority, cache_key_block_size=p.block_size,
                family='cold' if cold else str(family))


def iter_requests(workload: SyntheticWorkload, namespace, *, max_requests=None):
    return StatisticalTrafficGenerator(workload).requests(namespace, max_requests=max_requests)


def write_trace(path, parameters, namespace, base_dir=None, *, max_requests=None):
    workload, derivation = resolve(parameters)
    with Path(path).open('w') as out:
        for row in iter_requests(workload, namespace, max_requests=max_requests):
            out.write(json.dumps(row, separators=(',', ':')) + '\n')
    return dict(realism='CALIBRATED_STATISTICAL' if derivation else 'NOT_VALIDATED',
                arrival='ORDINAL_PACED_BY_CLIENT', tail='EXACT_CONFIGURED_LENGTHS',
                calibration=derivation, held_out_validated=False, sampling=workload.sampling,
                output_model='INDEPENDENT_EXPLICIT_DISTRIBUTION')

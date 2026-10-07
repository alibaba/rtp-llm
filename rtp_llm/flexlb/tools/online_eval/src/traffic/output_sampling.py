"""Stable output-length policies, keyed by original (pre-filter) event index."""
import math


def event_uniform(index, seed):
    mask = (1 << 64) - 1
    z = (index + seed + 0x9e3779b97f4a7c15) & mask
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
    z ^= z >> 31
    return (z >> 11) * 2.0 ** -53


def output_sampler(parameters):
    cap = parameters['output_tokens']
    policy = parameters.get('output_distribution')
    if policy is None:
        return lambda index: cap
    if not isinstance(policy, dict):
        raise ValueError('invalid output distribution fields')
    seed = policy.get('seed')
    if type(seed) is not int or not -(1 << 63) <= seed < (1 << 63):
        raise ValueError('output distribution requires explicit signed 64-bit seed')
    if policy.get('kind') == 'geometric':
        if set(policy) != {'kind', 'mean_tokens', 'seed'}:
            raise ValueError('invalid output distribution fields')
        mean = policy['mean_tokens']
        if type(mean) not in (int, float) or not math.isfinite(mean) or mean < 1:
            raise ValueError('invalid output distribution parameters')
        return lambda index: (1 if mean == 1 else min(cap, 1 + math.floor(
            math.log1p(-event_uniform(index, seed)) / math.log1p(-1 / mean))))
    if policy.get('kind') == 'discrete':
        if set(policy) != {'kind', 'values', 'weights', 'seed'}:
            raise ValueError('invalid output distribution fields')
        values, weights = policy['values'], policy['weights']
        if (not isinstance(values, list) or not values or not isinstance(weights, list)
                or len(values) != len(weights)
                or any(type(v) is not int or not 1 <= v <= cap for v in values)
                or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights)):
            raise ValueError('invalid discrete output values/weights (values must not exceed cap)')
        total = sum(weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError('invalid discrete output weight sum')
        def sample(index):
            target, cumulative = event_uniform(index, seed) * total, 0.0
            for value, weight in zip(values, weights):
                cumulative += weight
                if target < cumulative:
                    return value
            return next(v for v, w in reversed(list(zip(values, weights))) if w > 0)
        return sample
    raise ValueError('unknown output distribution kind')


def output_semantics(parameters):
    policy = parameters.get('output_distribution')
    return ('EXPLICIT_' + policy['kind'].upper() + '_SPLITMIX64_EVENT_INDEX_V1' if policy
            else 'INDEPENDENT_FIXED_CAP_NOT_FITTED_FROM_ERROR_OUTPUTS')

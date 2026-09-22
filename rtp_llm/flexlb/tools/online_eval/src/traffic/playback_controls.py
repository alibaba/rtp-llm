"""Opt-in playback controls and offline intensity planner (algorithm version 1)."""
import json
import math
from traffic.output_sampling import event_uniform

RAW = {'RATE_CURVE', 'ARRIVAL_PROCESS', 'LAP_RETAIN_SCHEDULE'}
FIELDS = {'rate_curve', 'arrival', 'retain_schedule'}


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate(p, *, mode, identity, scalar_present, legacy_pacing):
    """Strict validation shared by YAML and raw-env entrypoints; no fallback parsing."""
    out = {k: p[k] for k in FIELDS if k in p}
    if 'arrival' in out and out['arrival'] not in ('deterministic', 'poisson'):
        raise ValueError('arrival must be deterministic or poisson')
    random = out.get('arrival') == 'poisson' or 'retain_schedule' in out
    if (random or (out and 'seed' in p)) and (type(p.get('seed')) is not int or not -(1 << 63) <= p['seed'] < (1 << 63)):
        raise ValueError('new random playback controls require explicit signed 64-bit seed')
    if out.get('arrival') == 'poisson' and mode not in ('uniform',):
        raise ValueError('poisson arrival requires uniform mode')
    if ('rate_curve' in out or out.get('arrival') == 'poisson') and legacy_pacing:
        raise ValueError('rate_curve/poisson cannot combine with legacy burst, wave, ramp or GRADIENT')
    if 'rate_curve' in out:
        if mode not in ('uniform','replay','true-ts'):
            raise ValueError('rate_curve requires uniform or replay mode')
        curve = out['rate_curve']
        if not isinstance(curve, list) or not 1 <= len(curve) <= 4096:
            raise ValueError('rate_curve requires 1..4096 [elapsed_seconds, multiplier] points')
        previous = -1
        for point in curve:
            if (not isinstance(point, list) or len(point) != 2
                    or not finite(point[0], 0, 1e9) or point[0] <= previous
                    or not finite(point[1], 1e-6, 1e6)):
                raise ValueError('invalid rate_curve point/order')
            previous = point[0]
        if curve[0][0] != 0:
            raise ValueError('rate_curve must start at zero seconds')
    if 'retain_schedule' in out:
        schedule = out['retain_schedule']
        if identity != 'partial' or scalar_present:
            raise ValueError('retain_schedule requires partial and excludes retain_probability')
        if not isinstance(schedule, dict):
            raise ValueError('invalid retain_schedule')
        if schedule.get('kind') == 'linear':
            if (set(schedule) != {'kind', 'start', 'end', 'laps'}
                    or not finite(schedule['start'], 0, 1) or not finite(schedule['end'], 0, 1)
                    or type(schedule['laps']) is not int or not 2 <= schedule['laps'] <= 2147483647):
                raise ValueError('invalid linear retain_schedule')
        elif schedule.get('kind') == 'sequence':
            values = schedule.get('values')
            if (set(schedule) != {'kind', 'values'} or not isinstance(values, list)
                    or not 1 <= len(values) <= 4096 or any(not finite(v, 0, 1) for v in values)
                    or not (all(a <= b for a, b in zip(values, values[1:]))
                            or all(a >= b for a, b in zip(values, values[1:])))):
                raise ValueError('retain_schedule sequence must be monotonic probabilities')
        else:
            raise ValueError('unknown retain_schedule kind')
    return out


def from_env(env):
    p = {}
    for raw, key in [('RATE_CURVE', 'rate_curve'), ('LAP_RETAIN_SCHEDULE', 'retain_schedule')]:
        if str(env.get(raw, '')).strip():
            p[key] = json.loads(env[raw])
    if str(env.get('ARRIVAL_PROCESS', '')).strip():
        p['arrival'] = env['ARRIVAL_PROCESS']
    if str(env.get('PLAYBACK_SEED', '')).strip():
        p['seed'] = int(env['PLAYBACK_SEED'])
    if 'rate_curve' in p and env.get('SEND_MODE','replay') == 'replay' and float(env.get('REPLAY_SPEED') or 1) <= 0:
        raise ValueError('replay rate_curve requires positive REPLAY_SPEED')
    return validate(p, mode=env.get('SEND_MODE', 'replay'), identity=env.get('LAP_IDENTITY', 'structural-relabel'),
        scalar_present=bool(str(env.get('LAP_RETAIN_PROBABILITY', '')).strip()),
        legacy_pacing=(float(env.get('BURST_FACTOR') or 1) != 1
            or float(env.get('DIURNAL_AMPLITUDE') or 0) != 0
            or float(env.get('RAMP_UP_SECONDS') or 0) != 0 or str(env.get('GRADIENT', '')).lower() in ('true','1','yes')))


def integral(curve, seconds):
    """Integral of the multiplier, holding its last value after the final knot."""
    curve = curve or [[0, 1]]
    area = 0.0
    for (left, a), (right, b) in zip(curve, curve[1:]):
        width = min(seconds, right) - left
        if width > 0:
            area += width * (a + 0.5 * (b-a) * width/(right-left))
        if seconds <= right:
            return area
    return area + max(0, seconds-curve[-1][0]) * curve[-1][1]


def inverse(curve, area):
    curve = curve or [[0, 1]]
    for (left, a), (right, b) in zip(curve, curve[1:]):
        segment = (right-left) * (a+b)/2
        if area <= segment:
            slope = (b-a)/(right-left)
            return left + 2*area/(a + math.sqrt(max(0, a*a + 2*slope*area)))
        area -= segment
    return curve[-1][0] + area/curve[-1][1]


def poisson_count(intensity, seed, limit=1_000_000):
    """Exact seeded finite-window count; refuses workloads beyond evidence capacity."""
    total = 0.0
    for index in range(limit + 1):
        # Separate arrival randomness from output-length draws with the same user seed.
        u = event_uniform(index, seed ^ 0xd1b54a32d192ed03)
        total += -math.log1p(-u) if u else 2.0**-53
        if total >= intensity:
            return index
    raise ValueError('Poisson plan exceeds evidence event capacity; shorten duration or lower rate')

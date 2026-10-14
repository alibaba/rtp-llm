"""Per-engine status statistics from sampled gauges; no RPC percentile claim."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math


QPS_WINDOW_S = 10.0


@dataclass(frozen=True)
class StatusSeries:
    times: tuple[float, ...]
    rates: tuple[float, ...]
    latencies: tuple[float, ...]
    areas: tuple[float, ...]

    @classmethod
    def read(cls, rows, name, max_gap_s):
        times, rates, latencies, identities = [], [], [], set()
        for row in rows:
            engine = row['engines'].get(name, {})
            values = (row['t'], engine.get('worker_status_qps'), engine.get('worker_status_rt_us'))
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError('missing or invalid WorkerStatus metrics: ' + name)
            if any(value < 0 for value in values[1:]):
                raise ValueError('negative WorkerStatus metrics: ' + name)
            times.append(values[0]); rates.append(values[1]); latencies.append(values[2])
            identities.add((engine.get('grpc_addr'), engine.get('engine_incarnation')))
        if len(identities) != 1:
            raise ValueError('changed WorkerStatus engine identity: ' + name)
        if len(times) < 2 or any(not 0 < b-a <= max_gap_s for a, b in zip(times, times[1:])):
            raise ValueError('WorkerStatus sampling gap or non-increasing clock: ' + name)
        areas = [0.0]
        for i in range(len(times)-1):
            areas.append(areas[-1] + rates[i] * (times[i+1]-times[i]))
        return cls(tuple(times), tuple(rates), tuple(latencies), tuple(areas))

    def integral(self, at):
        if not self.times[0] <= at <= self.times[-1]:
            raise ValueError('incomplete WorkerStatus rolling window coverage')
        index = min(bisect_right(self.times, at)-1, len(self.times)-2)
        return self.areas[index] + self.rates[index] * (at-self.times[index])

    def latency_samples(self, start, end, include_end):
        first = bisect_left(self.times, start)
        stop = bisect_right(self.times, end) if include_end else bisect_left(self.times, end)
        return self.latencies[first:stop]

    def mean_qps(self, at):
        return max(0.0, (self.integral(at)-self.integral(at-QPS_WINDOW_S))/QPS_WINDOW_S)


def sampled_p99(values):
    ordered = sorted(values)
    return ordered[math.ceil(.99*len(ordered))-1]


def summarize(evidence):
    rows = evidence['samples']
    times = [row['t'] for row in rows]
    if not times or any(type(t) not in (int, float) or not math.isfinite(t) for t in times) or any(
            b <= a for a, b in zip(times, times[1:])):
        return {}, None, None, ['invalid WorkerStatus sample clock']
    withdrawals = [event['t'] for event in evidence.get('events', [])
                   if event['id'] in {'withdraw_start', 'intermediate_withdraw_start'}]
    bounds = {'baseline': (evidence['baseline_start'], evidence['baseline_end']),
              'transition': (min(withdrawals) if withdrawals else evidence['baseline_end'], evidence['post_start']),
              'post': (evidence['post_start'], evidence['post_end'])}
    errors = [] if withdrawals else ['missing withdrawal event for status reporting']
    measurements = {}
    for phase, (start, end) in bounds.items():
        names = evidence['initial_engines'] if phase == 'baseline' else evidence['survivors']
        measurement = dict(start=start, end=end, engines=list(names), samples=0,
                           min_rolling_qps=None, p99_rt_us=None, max_rt_us=None,
                           rolling_qps=[], per_engine={}, qps_window_s=QPS_WINDOW_S,
                           qps_method='time-weighted left-held gauge; trailing window',
                           rt_method='per-engine nearest-rank sampled p99, then fleet maximum',
                           include_end=phase == 'post')
        measurements[phase] = measurement
        if end == start:
            measurement['reason'] = 'zero-duration phase; boundary belongs to the following phase'
            continue
        if end < start or not names:
            errors.append('missing status reporting window: ' + phase)
            continue
        first = bisect_left(times, start)
        stop = bisect_right(times, end) if phase == 'post' else bisect_left(times, end)
        selected = rows[first:stop]
        measurement['samples'] = len(selected)
        if not selected:
            errors.append('missing status reporting samples: ' + phase)
            continue
        # Read one bracketing sample on either side, including warmup history.
        # A short transition still has complete trailing windows for survivors.
        left = bisect_right(times, start-QPS_WINDOW_S)-1
        right = bisect_left(times, end)
        if left < 0 or right >= len(rows):
            errors.append('incomplete WorkerStatus rolling window coverage: ' + phase)
            continue
        support = rows[left:right+1]
        candidates = {start, *[row['t'] for row in selected], *[row['t']+QPS_WINDOW_S for row in support]}
        endpoints = sorted(at for at in candidates if start <= at < end or phase == 'post' and at == end)
        if phase == 'post' and end not in endpoints:
            endpoints.append(end)
        minima = [math.inf] * len(endpoints)
        for name in names:
            try:
                series = StatusSeries.read(support, name, evidence['criteria']['max_gap_s'])
                rolling = [series.mean_qps(at) for at in endpoints]
                latencies = series.latency_samples(start, end, phase == 'post')
                measurement['per_engine'][name] = dict(min_rolling_qps=min(rolling),
                    p99_rt_us=sampled_p99(latencies), max_rt_us=max(latencies), rt_samples=len(latencies))
                minima = [min(a, b) for a, b in zip(minima, rolling)]
            except ValueError as exc:
                errors.append(phase + ': ' + str(exc))
        if len(measurement['per_engine']) == len(names):
            engines = measurement['per_engine'].values()
            measurement.update(min_rolling_qps=min(minima),
                p99_rt_us=max(value['p99_rt_us'] for value in engines),
                max_rt_us=max(value['max_rt_us'] for value in engines),
                rolling_qps=[dict(t=at, min_rolling_qps=value) for at, value in zip(endpoints, minima)])
    valid = not errors
    active = [value for value in measurements.values() if value['min_rolling_qps'] is not None]
    return (measurements, min(value['min_rolling_qps'] for value in active) if valid and active else None,
            max(value['p99_rt_us'] for value in active) if valid and active else None, errors)

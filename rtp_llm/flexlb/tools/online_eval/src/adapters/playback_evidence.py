"""Verify sender-owned playback evidence against the materialized input and launch."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PlaybackEvidence:
    snapshot: dict

    @classmethod
    def read(cls, path, trace_sha256, environment):
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict) or value.get('trace_sha256') != trace_sha256:
            raise ValueError('Java playback trace SHA256 differs from materialized plan')
        if value.get('offline_plan_supported') is not True:
            raise ValueError('Java playback cannot reproduce controlled trace')
        if (value.get('planner_version') not in ('PLAYBACK_LEGACY_V1', 'PLAYBACK_CONTROLS_V1')
                or type(value.get('token_stride')) is not int or value['token_stride'] < 1):
            raise ValueError('Java playback identity evidence missing')
        expected = dict(mode=environment.get('SEND_MODE', 'replay'),
                        identity=environment.get('LAP_IDENTITY', 'structural-relabel'),
                        seed=int(environment.get('PLAYBACK_SEED', 0)),
                        max_laps=int(environment.get('MAX_LAPS', 0 if environment.get('LOOP') == 'true' else 1)),
                        num_shards=int(environment.get('NUM_SHARDS', 1)),
                        shard_index=int(environment.get('SHARD_INDEX', 0)),
                        duration_seconds=int(environment.get('DURATION_S', 0)),
                        limit=int(environment.get('LIMIT', 0)))
        for field, target in expected.items():
            if value.get(field) != target:
                raise ValueError('Java playback launch mismatch: ' + field)
        numbers = dict(qps=('SEND_MODE_QPS', 0), speed=('REPLAY_SPEED', 1),
                       ramp_up_seconds=('RAMP_UP_SECONDS', 0),
                       retain_probability=('LAP_RETAIN_PROBABILITY', 0),
                       burst_factor=('BURST_FACTOR', 1), burst_period_seconds=('BURST_PERIOD_SECONDS', 10),
                       burst_duty=('BURST_DUTY', .1), diurnal_amplitude=('DIURNAL_AMPLITUDE', 0),
                       diurnal_period_seconds=('DIURNAL_PERIOD_SECONDS', 86400))
        for field, (name, fallback) in numbers.items():
            if value.get(field) != float(environment.get(name, fallback)):
                raise ValueError('Java playback launch mismatch: ' + field)
        for field, name in dict(rate_curve='RATE_CURVE', retain_schedule='LAP_RETAIN_SCHEDULE').items():
            if name in environment and value.get(field) != json.loads(environment[name]):
                raise ValueError('Java playback launch mismatch: ' + field)
        if 'ARRIVAL_PROCESS' in environment and value.get('arrival') != environment['ARRIVAL_PROCESS']:
            raise ValueError('Java playback launch mismatch: arrival')
        return cls(value)

    def to_dict(self):
        return dict(self.snapshot)

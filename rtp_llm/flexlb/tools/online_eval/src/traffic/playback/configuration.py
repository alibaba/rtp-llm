"""Decode authoring/Java boundaries into a validated playback configuration."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from traffic.playback import intensity as controls


class PacingMode(str, Enum):
    UNIFORM = 'uniform'
    TIMESTAMPS = 'true-ts'
    REPLAY = 'replay'
    BURST = 'burst'
    GRADIENT = 'gradient'


class LapIdentity(str, Enum):
    NONE = 'none'
    RELABEL = 'structural-relabel'
    PARTIAL = 'partial'


@dataclass(frozen=True)
class PlaybackConfiguration:
    mode: PacingMode = PacingMode.UNIFORM
    qps: float = 0
    speed: float = 1
    max_laps: int = 1
    identity: LapIdentity = LapIdentity.RELABEL
    retain_probability: float = 0
    seed: int = 0
    ramp_up_seconds: float = 0
    burst_factor: float = 1
    burst_period_seconds: float = 10
    burst_duty: float = .1
    diurnal_amplitude: float = 0
    diurnal_period_seconds: float = 86400
    intensity: controls.PlaybackIntensity = controls.PlaybackIntensity()

    def __post_init__(self):
        bounds = {'qps': (0, math.inf), 'speed': (1e-9, math.inf),
            'retain_probability': (0, 1), 'ramp_up_seconds': (0, math.inf),
            'burst_factor': (1, math.inf), 'burst_period_seconds': (.001, math.inf),
            'burst_duty': (.001, .999), 'diurnal_amplitude': (0, .99),
            'diurnal_period_seconds': (.001, math.inf)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if not controls.finite(value, low, high):
                raise ValueError('invalid playback '+key)
        if self.mode not in (PacingMode.TIMESTAMPS, PacingMode.REPLAY) and self.qps <= 0:
            raise ValueError('positive playback qps required')
        if type(self.max_laps) is not int or not 0 <= self.max_laps <= 2147483647:
            raise ValueError('max_laps requires a bounded nonnegative integer')
        if type(self.seed) is not int or not -(1 << 63) <= self.seed < (1 << 63):
            raise ValueError('playback seed must be signed 64-bit integer')
        if self.mode == PacingMode.BURST and self.burst_factor <= 1:
            raise ValueError('burst mode requires burst_factor > 1')
        if self.mode == PacingMode.GRADIENT and self.ramp_up_seconds <= 0:
            raise ValueError('gradient requires ramp_up_seconds')
        if self.identity != LapIdentity.PARTIAL and self.retain_probability != 0:
            raise ValueError('retain_probability belongs to partial identity')

    def to_dict(self):
        result = {key: getattr(self, key) for key in SCALAR_FIELDS}
        result.update(mode=self.mode.value, identity=self.identity.value)
        result.update(self.intensity.to_dict())
        return result

    def environment(self):
        result = {raw: str(getattr(self, key)) for key, raw in ENV_FIELDS.items()}
        result.update(SEND_MODE='replay' if self.mode in (PacingMode.TIMESTAMPS, PacingMode.REPLAY) else 'uniform',
                      LAP_IDENTITY=self.identity.value, LOOP='true' if self.max_laps != 1 else 'false')
        for key, value in self.intensity.to_dict().items():
            raw = ADVANCED_ENV[key]
            result[raw] = value if key == 'arrival' else json.dumps(value, separators=(',', ':'))
        if self.intensity.retain_schedule is not None:
            result.pop('LAP_RETAIN_PROBABILITY')
        return result


# The mapping is the external Java protocol, not a list of business branches.
ENV_FIELDS = dict(qps='SEND_MODE_QPS', speed='REPLAY_SPEED', max_laps='MAX_LAPS',
    retain_probability='LAP_RETAIN_PROBABILITY', seed='PLAYBACK_SEED',
    ramp_up_seconds='RAMP_UP_SECONDS', burst_factor='BURST_FACTOR', burst_period_seconds='BURST_PERIOD_SECONDS',
    burst_duty='BURST_DUTY', diurnal_amplitude='DIURNAL_AMPLITUDE', diurnal_period_seconds='DIURNAL_PERIOD_SECONDS')
ADVANCED_ENV = dict(rate_curve='RATE_CURVE', arrival='ARRIVAL_PROCESS', retain_schedule='LAP_RETAIN_SCHEDULE')
SCALAR_FIELDS = tuple(key for key in PlaybackConfiguration.__dataclass_fields__ if key != 'intensity')
FIELDS = set(SCALAR_FIELDS) | controls.FIELDS
RAW_FIELDS = set(ENV_FIELDS.values()) | set(ADVANCED_ENV.values()) | {'SEND_MODE', 'LOOP', 'LAP_IDENTITY', 'GRADIENT'}


class PlaybackConfigurationCodec:
    @staticmethod
    def decode(value, *, duration_s=0):
        if not isinstance(value, dict) or set(value)-FIELDS:
            raise ValueError('unknown playback fields')
        try:
            mode = PacingMode(value.get('mode', 'uniform'))
            identity = LapIdentity(value.get('identity', 'structural-relabel'))
        except (ValueError, TypeError):
            raise ValueError('invalid playback mode or lap identity') from None
        if mode == PacingMode.REPLAY:
            raise ValueError('invalid playback mode')
        if identity != LapIdentity.PARTIAL and 'retain_probability' in value:
            raise ValueError('retain_probability belongs to partial identity')
        if mode == PacingMode.TIMESTAMPS and any(key in value for key in
                ('qps', 'ramp_up_seconds', 'burst_factor', 'diurnal_amplitude')):
            raise ValueError('true-ts uses only source timestamps and replay speed')
        scalar = {key: val for key, val in value.items() if key in SCALAR_FIELDS}
        scalar.update(mode=mode, identity=identity)
        pacing = PlaybackConfiguration(**scalar)
        if pacing.max_laps == 0 and int(duration_s) <= 0:
            raise ValueError('max_laps requires a positive count or 0 with bounded duration')
        intensity = controls.PlaybackIntensity.read(value, mode=mode.value, identity=identity.value,
            scalar_present='retain_probability' in value, legacy_pacing=(mode in (PacingMode.BURST, PacingMode.GRADIENT)
                or pacing.ramp_up_seconds != 0 or pacing.burst_factor != 1 or pacing.diurnal_amplitude != 0))
        if intensity.to_dict():
            intensity = replace(intensity, seed=pacing.seed)
        return PlaybackConfiguration(**scalar, intensity=intensity)

    @staticmethod
    def from_environment(env):
        mode = env.get('SEND_MODE', 'replay')
        if mode not in ('uniform', 'replay'):
            raise ValueError('invalid Java client pacing')
        values = {}
        for key, raw in ENV_FIELDS.items():
            if raw in env:
                values[key] = (int if key in ('seed', 'max_laps') else float)(env[raw])
        values.setdefault('max_laps', 0 if env.get('LOOP') == 'true' else 1)
        try:
            identity = LapIdentity(env.get('LAP_IDENTITY', 'structural-relabel'))
        except (ValueError, TypeError):
            raise ValueError('invalid lap identity') from None
        intensity = controls.PlaybackEnvCodec.decode(env)
        if intensity.to_dict():
            intensity = replace(intensity, seed=values.get('seed', 0))
        result = PlaybackConfiguration(mode=PacingMode(mode), identity=identity, intensity=intensity, **values)
        if result.max_laps == 0 and int(env.get('DURATION_S', 0)) <= 0:
            raise ValueError('max_laps requires a positive count or 0 with bounded duration')
        return result


def normalize(client):
    """Java environment and artifact projection of the same admitted domain object."""
    env = dict(client)
    if 'playback' not in env:
        configuration = PlaybackConfigurationCodec.from_environment(env)
        return env, configuration.to_dict()
    value = env.pop('playback')
    if set(env) & RAW_FIELDS:
        raise ValueError('playback cannot be combined with raw pacing/loop controls')
    configuration = PlaybackConfigurationCodec.decode(value, duration_s=env.get('DURATION_S', 0))
    env.update(configuration.environment())
    return env, configuration.to_dict()

"""Single-round or bounded multi-round uniform/timestamp playback configuration."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


JAVA_REPLAY_SPEED = 10.0


class PacingMode(str, Enum):
    UNIFORM = 'uniform'
    TIMESTAMPS = 'true-ts'
    REPLAY = 'replay'  # Java environment spelling of timestamp playback.


class LapIdentity(str, Enum):
    NONE = 'none'
    RELABEL = 'structural-relabel'


@dataclass(frozen=True)
class PlaybackConfiguration:
    mode: PacingMode = PacingMode.UNIFORM
    qps: float = 0
    speed: float = 1
    max_laps: int = 1
    identity: LapIdentity = LapIdentity.RELABEL
    ramp_up_seconds: float = 0

    def __post_init__(self):
        for key, low in (('qps', 0), ('speed', 1e-9), ('ramp_up_seconds', 0)):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < low:
                raise ValueError('invalid playback ' + key)
            object.__setattr__(self, key, float(value))
        if self.mode == PacingMode.UNIFORM and self.qps <= 0:
            raise ValueError('positive playback qps required')
        if type(self.max_laps) is not int or not 0 <= self.max_laps <= 2147483647:
            raise ValueError('max_laps requires a bounded nonnegative integer')
        if not isinstance(self.mode, PacingMode) or not isinstance(self.identity, LapIdentity):
            raise ValueError('invalid playback mode or lap identity')
        if self.mode != PacingMode.UNIFORM and self.ramp_up_seconds != 0:
            raise ValueError('ramp_up_seconds belongs to uniform playback')

    def to_dict(self):
        return dict(mode=self.mode.value, qps=self.qps, speed=self.speed,
                    max_laps=self.max_laps, identity=self.identity.value,
                    ramp_up_seconds=self.ramp_up_seconds)

    def environment(self):
        result = {raw: str(getattr(self, key)) for key, raw in ENV_FIELDS.items()}
        result.update(SEND_MODE='uniform' if self.mode == PacingMode.UNIFORM else 'replay',
                      LAP_IDENTITY=self.identity.value, LOOP='true' if self.max_laps != 1 else 'false')
        return result


ENV_FIELDS = dict(qps='SEND_MODE_QPS', speed='REPLAY_SPEED', max_laps='MAX_LAPS',
                  ramp_up_seconds='RAMP_UP_SECONDS')
FIELDS = frozenset(PlaybackConfiguration.__dataclass_fields__)
RAW_FIELDS = set(ENV_FIELDS.values()) | {'SEND_MODE', 'LOOP', 'LAP_IDENTITY'}



class PlaybackConfigurationCodec:
    @staticmethod
    def decode(value, *, duration_s=0):
        if not isinstance(value, dict) or set(value) - FIELDS:
            raise ValueError('unknown playback fields')
        try:
            mode = PacingMode(value.get('mode', 'uniform'))
            identity = LapIdentity(value.get('identity', 'structural-relabel'))
        except (ValueError, TypeError):
            raise ValueError('invalid playback mode or lap identity') from None
        if mode == PacingMode.REPLAY:
            raise ValueError('use true-ts for timestamp playback')
        if mode == PacingMode.TIMESTAMPS and any(k in value for k in ('qps', 'ramp_up_seconds')):
            raise ValueError('true-ts uses only source timestamps and replay speed')
        configuration = PlaybackConfiguration(**dict(value, mode=mode, identity=identity))
        _bounded(configuration, duration_s)
        return configuration

    @staticmethod
    def from_environment(env):
        mode = env.get('SEND_MODE', 'replay')
        if mode not in ('uniform', 'replay'):
            raise ValueError('invalid Java client pacing')
        if 'LOOP' in env and env['LOOP'] not in ('true', 'false'):
            raise ValueError('LOOP must be true or false')
        values = {key: (int if key == 'max_laps' else float)(env[raw])
                  for key, raw in ENV_FIELDS.items() if raw in env}
        values.setdefault('speed', JAVA_REPLAY_SPEED)
        values.setdefault('max_laps', 0 if env.get('LOOP') == 'true' else 1)
        if 'LOOP' in env and (env['LOOP'] == 'true') != (values['max_laps'] != 1):
            raise ValueError('LOOP conflicts with MAX_LAPS')
        try:
            identity = LapIdentity(env.get('LAP_IDENTITY', 'structural-relabel'))
        except (ValueError, TypeError):
            raise ValueError('invalid lap identity') from None
        configuration = PlaybackConfiguration(mode=PacingMode(mode), identity=identity, **values)
        _bounded(configuration, env.get('DURATION_S', 0))
        return configuration


def _bounded(configuration, duration_s):
    if configuration.max_laps == 0 and int(duration_s) <= 0:
        raise ValueError('max_laps requires a positive count or 0 with bounded duration')


def normalize(client):
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

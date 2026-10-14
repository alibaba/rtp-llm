"""Single-round or bounded multi-round uniform/timestamp playback configuration."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from traffic.playback.defaults import DEFAULTS


class PacingMode(str, Enum):
    UNIFORM = 'uniform'
    TIMESTAMPS = 'true-ts'


class LapIdentity(str, Enum):
    NONE = 'none'
    RELABEL = 'structural-relabel'


@dataclass(frozen=True)
class PlaybackConfiguration:
    mode: PacingMode = PacingMode.TIMESTAMPS if DEFAULTS['SEND_MODE'] == 'replay' else PacingMode.UNIFORM
    duration_s: int = int(DEFAULTS['DURATION_S'])
    qps: float = float(DEFAULTS['SEND_MODE_QPS'])
    speed: float = float(DEFAULTS['REPLAY_SPEED'])
    max_laps: int = int(DEFAULTS['MAX_LAPS'])
    identity: LapIdentity = LapIdentity(DEFAULTS['LAP_IDENTITY'])
    ramp_up_seconds: float = float(DEFAULTS['RAMP_UP_SECONDS'])

    def __post_init__(self):
        for key, low in (('qps', 0), ('speed', 1e-9), ('ramp_up_seconds', 0)):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < low:
                raise ValueError('invalid playback ' + key)
            object.__setattr__(self, key, float(value))
        if type(self.duration_s) is not int or not 0 <= self.duration_s <= 2147483647:
            raise ValueError('duration_s requires a bounded nonnegative integer')
        if self.max_laps == 0 and self.duration_s <= 0:
            raise ValueError('unbounded laps require a positive duration_s')
        if self.mode == PacingMode.UNIFORM and self.qps <= 0:
            raise ValueError('positive playback qps required')
        if type(self.max_laps) is not int or not 0 <= self.max_laps <= 2147483647:
            raise ValueError('max_laps requires a bounded nonnegative integer')
        if not isinstance(self.mode, PacingMode) or not isinstance(self.identity, LapIdentity):
            raise ValueError('invalid playback mode or lap identity')
        if self.mode != PacingMode.UNIFORM and self.ramp_up_seconds != 0:
            raise ValueError('ramp_up_seconds belongs to uniform playback')

    def to_dict(self):
        return dict(mode=self.mode.value, duration_s=self.duration_s, qps=self.qps, speed=self.speed,
                    max_laps=self.max_laps, identity=self.identity.value,
                    ramp_up_seconds=self.ramp_up_seconds)

    def environment(self):
        result = {raw: str(getattr(self, key)) for key, raw in ENV_FIELDS.items()}
        result.update(SEND_MODE='uniform' if self.mode == PacingMode.UNIFORM else 'replay',
                      LAP_IDENTITY=self.identity.value, LOOP='true' if self.max_laps != 1 else 'false')
        return result


ENV_FIELDS = dict(duration_s='DURATION_S', qps='SEND_MODE_QPS', speed='REPLAY_SPEED', max_laps='MAX_LAPS',
                  ramp_up_seconds='RAMP_UP_SECONDS')
FIELDS = frozenset(PlaybackConfiguration.__dataclass_fields__)


class PlaybackConfigurationCodec:
    @staticmethod
    def decode(value):
        if not isinstance(value, dict) or set(value) - FIELDS:
            raise ValueError('unknown playback fields')
        try:
            mode = PacingMode(value.get('mode', PlaybackConfiguration().mode.value))
            identity = LapIdentity(value.get('identity', DEFAULTS['LAP_IDENTITY']))
        except (ValueError, TypeError):
            raise ValueError('invalid playback mode or lap identity') from None
        if mode == PacingMode.TIMESTAMPS and any(k in value for k in ('qps', 'ramp_up_seconds')):
            raise ValueError('true-ts uses only source timestamps and replay speed')
        configuration = PlaybackConfiguration(**dict(value, mode=mode, identity=identity))
        return configuration

    @staticmethod
    def from_environment(env):
        env = {key: value for key, value in env.items() if value != ""}
        mode = env.get('SEND_MODE', DEFAULTS['SEND_MODE'])
        if mode not in ('uniform', 'replay'):
            raise ValueError('invalid Java client pacing')
        if 'LOOP' in env and env['LOOP'] not in ('true', 'false'):
            raise ValueError('LOOP must be true or false')
        values = {key: (int if key in ('max_laps', 'duration_s') else float)(env[raw])
                  for key, raw in ENV_FIELDS.items() if raw in env}
        values.setdefault('max_laps', 0 if env.get('LOOP') == 'true' else int(DEFAULTS['MAX_LAPS']))
        if 'LOOP' in env and (env['LOOP'] == 'true') != (values['max_laps'] != 1):
            raise ValueError('LOOP conflicts with MAX_LAPS')
        try:
            identity = LapIdentity(env.get('LAP_IDENTITY', DEFAULTS['LAP_IDENTITY']))
        except (ValueError, TypeError):
            raise ValueError('invalid lap identity') from None
        configuration = PlaybackConfiguration(mode=PacingMode.TIMESTAMPS if mode == "replay" else PacingMode.UNIFORM, identity=identity, **values)
        return configuration

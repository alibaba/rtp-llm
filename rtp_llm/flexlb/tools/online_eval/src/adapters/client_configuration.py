"""Typed scenario intent and projection to the Java launch boundary."""
from dataclasses import dataclass, fields

from traffic.playback.configuration import PlaybackConfiguration, PlaybackConfigurationCodec
from traffic.playback.defaults import DEFAULTS


def section(value, model, name):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(model)}:
        raise ValueError('unknown ' + name + ' fields')
    return model(**value)


def positive_integers(value):
    for field in fields(value):
        number = getattr(value, field.name)
        if type(number) is not int or not 0 < number <= 2147483647:
            raise ValueError(field.name + ' must be a positive bounded integer')


@dataclass(frozen=True)
class ClientLimits:
    max_concurrency: int = int(DEFAULTS['MAX_CONCURRENCY'])
    request_timeout_s: int = int(DEFAULTS['TIMEOUT_MS']) // 1000
    drain_timeout_s: int = int(DEFAULTS['RESPONSE_TIMEOUT'])

    def __post_init__(self):
        positive_integers(self)
        if self.request_timeout_s * 1000 > 2147483647:
            raise ValueError('request timeout exceeds Java millisecond capacity')
        if self.drain_timeout_s < self.request_timeout_s:
            raise ValueError('drain_timeout_s must cover request_timeout_s')


@dataclass(frozen=True)
class ClientTransport:
    channels: int = int(DEFAULTS['N_CHANNELS'])
    event_loop_threads: int = int(DEFAULTS['EVENT_LOOP_THREADS'])

    def __post_init__(self):
        positive_integers(self)


@dataclass(frozen=True)
class ClientConfiguration:
    playback: PlaybackConfiguration
    limits: ClientLimits
    transport: ClientTransport

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict) or set(value) - {'playback', 'limits', 'transport'}:
            raise ValueError('unknown client fields; use playback, limits and transport')
        playback = PlaybackConfigurationCodec.decode(value.get('playback', {}))
        if playback.duration_s <= 0:
            raise ValueError('scenario playback.duration_s must be positive')
        return cls(playback, section(value.get('limits', {}), ClientLimits, 'limits'),
                   section(value.get('transport', {}), ClientTransport, 'transport'))

    def environment(self):
        return dict(self.playback.environment(),
                    MAX_CONCURRENCY=str(self.limits.max_concurrency),
                    TIMEOUT_MS=str(self.limits.request_timeout_s * 1000),
                    RESPONSE_TIMEOUT=str(self.limits.drain_timeout_s),
                    N_CHANNELS=str(self.transport.channels),
                    EVENT_LOOP_THREADS=str(self.transport.event_loop_threads),
                    FETCH_OUTPUT_STREAM=DEFAULTS['FETCH_OUTPUT_STREAM'],
                    REPLAY_UNIQUE_PREFIX=DEFAULTS['REPLAY_UNIQUE_PREFIX'])

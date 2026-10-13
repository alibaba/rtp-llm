"""Validated playback value objects and deterministic offline calculations."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union
import json
import math


class ArrivalProcess(str, Enum):
    DETERMINISTIC = "deterministic"
    POISSON = "poisson"


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


@dataclass(frozen=True)
class RateKnot:
    elapsed_seconds: float
    multiplier: float

    def __post_init__(self):
        if not finite(self.elapsed_seconds, 0, 1e9) or not finite(self.multiplier, 1e-6, 1e6):
            raise ValueError("invalid rate_curve point/order")


@dataclass(frozen=True)
class RateCurve:
    knots: tuple[RateKnot, ...]

    def __post_init__(self):
        if not 1 <= len(self.knots) <= 4096:
            raise ValueError("rate_curve requires 1..4096 [elapsed_seconds, multiplier] points")
        if self.knots[0].elapsed_seconds != 0:
            raise ValueError("rate_curve must start at zero seconds")
        if any(a.elapsed_seconds >= b.elapsed_seconds for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError("invalid rate_curve point/order")

    @classmethod
    def read(cls, value):
        if not isinstance(value, list) or not 1 <= len(value) <= 4096:
            raise ValueError("rate_curve requires 1..4096 [elapsed_seconds, multiplier] points")
        if any(not isinstance(point, list) or len(point) != 2 for point in value):
            raise ValueError("invalid rate_curve point/order")
        return cls(tuple(RateKnot(*point) for point in value))

    def to_list(self):
        return [[point.elapsed_seconds, point.multiplier] for point in self.knots]

    def integral(self, seconds: float) -> float:
        area = 0.0
        for left, right in zip(self.knots, self.knots[1:]):
            width = min(seconds, right.elapsed_seconds) - left.elapsed_seconds
            if width > 0:
                area += width * (left.multiplier + 0.5 * (right.multiplier-left.multiplier)
                                 * width/(right.elapsed_seconds-left.elapsed_seconds))
            if seconds <= right.elapsed_seconds:
                return area
        last = self.knots[-1]
        return area + max(0, seconds-last.elapsed_seconds) * last.multiplier


@dataclass(frozen=True)
class LinearRetentionSchedule:
    start: float
    end: float
    laps: int

    def __post_init__(self):
        if (not finite(self.start, 0, 1) or not finite(self.end, 0, 1)
                or type(self.laps) is not int or not 2 <= self.laps <= 2147483647):
            raise ValueError("invalid linear retain_schedule")

    def probability(self, lap: int) -> float:
        if type(lap) is not int or lap < 1:
            raise ValueError("lap must be a positive integer")
        return self.start + (self.end-self.start)*min(1, (lap-1)/(self.laps-1))

    def to_dict(self):
        return dict(kind="linear", start=self.start, end=self.end, laps=self.laps)


@dataclass(frozen=True)
class SequenceRetentionSchedule:
    values: tuple[float, ...]

    def __post_init__(self):
        if (not 1 <= len(self.values) <= 4096 or any(not finite(v, 0, 1) for v in self.values)
                or not (all(a <= b for a, b in zip(self.values, self.values[1:]))
                        or all(a >= b for a, b in zip(self.values, self.values[1:])))):
            raise ValueError("retain_schedule sequence must be monotonic probabilities")

    def probability(self, lap: int) -> float:
        if type(lap) is not int or lap < 1:
            raise ValueError("lap must be a positive integer")
        return self.values[min(lap-1, len(self.values)-1)]

    def to_dict(self):
        return dict(kind="sequence", values=list(self.values))


RetentionSchedule = Union[LinearRetentionSchedule, SequenceRetentionSchedule]


def read_retention_schedule(value):
    if not isinstance(value, dict):
        raise ValueError("invalid retain_schedule")
    if value.get("kind") == "linear":
        if set(value) != {"kind", "start", "end", "laps"}:
            raise ValueError("invalid linear retain_schedule")
        return LinearRetentionSchedule(value["start"], value["end"], value["laps"])
    if value.get("kind") == "sequence":
        if set(value) != {"kind", "values"} or not isinstance(value["values"], list):
            raise ValueError("retain_schedule sequence must be monotonic probabilities")
        return SequenceRetentionSchedule(tuple(value["values"]))
    raise ValueError("unknown retain_schedule kind")


@dataclass(frozen=True)
class PlaybackIntensity:
    rate_curve: RateCurve | None = None
    arrival: ArrivalProcess | None = None
    retain_schedule: RetentionSchedule | None = None
    seed: int | None = None

    def __post_init__(self):
        random = self.arrival == ArrivalProcess.POISSON or self.retain_schedule is not None
        if (random or self.seed is not None) and (
                type(self.seed) is not int or not -(1 << 63) <= self.seed < (1 << 63)):
            raise ValueError("new random playback controls require explicit signed 64-bit seed")

    @classmethod
    def read(cls, value, *, mode, identity, scalar_present, legacy_pacing):
        try:
            arrival = ArrivalProcess(value["arrival"]) if "arrival" in value else None
        except (TypeError, ValueError):
            raise ValueError("arrival must be deterministic or poisson") from None
        if arrival == ArrivalProcess.POISSON and mode != "uniform":
            raise ValueError("poisson arrival requires uniform mode")
        if ("rate_curve" in value or arrival == ArrivalProcess.POISSON) and legacy_pacing:
            raise ValueError("rate_curve/poisson cannot combine with legacy burst, wave, ramp or GRADIENT")
        if "rate_curve" in value and mode not in ("uniform", "replay", "true-ts"):
            raise ValueError("rate_curve requires uniform or replay mode")
        if "retain_schedule" in value and (identity != "partial" or scalar_present):
            raise ValueError("retain_schedule requires partial and excludes retain_probability")
        active = bool(FIELDS.intersection(value))
        if active and "seed" in value and (type(value["seed"]) is not int
                or not -(1 << 63) <= value["seed"] < (1 << 63)):
            raise ValueError("new random playback controls require explicit signed 64-bit seed")
        return cls(RateCurve.read(value["rate_curve"]) if "rate_curve" in value else None,
                   arrival,
                   read_retention_schedule(value["retain_schedule"]) if "retain_schedule" in value else None,
                   value.get("seed") if active else None)

    def to_dict(self):
        result = {}
        if self.rate_curve is not None:
            result["rate_curve"] = self.rate_curve.to_list()
        if self.arrival is not None:
            result["arrival"] = self.arrival.value
        if self.retain_schedule is not None:
            result["retain_schedule"] = self.retain_schedule.to_dict()
        return result


FIELDS = {"rate_curve", "arrival", "retain_schedule"}
RAW = {"RATE_CURVE", "ARRIVAL_PROCESS", "LAP_RETAIN_SCHEDULE"}


class PlaybackEnvCodec:
    @staticmethod
    def decode(env):
        value = {}
        for raw, key in (("RATE_CURVE", "rate_curve"), ("LAP_RETAIN_SCHEDULE", "retain_schedule")):
            if str(env.get(raw, "")).strip():
                value[key] = json.loads(env[raw])
        if str(env.get("ARRIVAL_PROCESS", "")).strip():
            value["arrival"] = env["ARRIVAL_PROCESS"]
        if str(env.get("PLAYBACK_SEED", "")).strip():
            value["seed"] = int(env["PLAYBACK_SEED"])
        if "rate_curve" in value and env.get("SEND_MODE", "replay") == "replay" and float(env.get("REPLAY_SPEED") or 1) <= 0:
            raise ValueError("replay rate_curve requires positive REPLAY_SPEED")
        return PlaybackIntensity.read(value, mode=env.get("SEND_MODE", "replay"),
            identity=env.get("LAP_IDENTITY", "structural-relabel"),
            scalar_present=bool(str(env.get("LAP_RETAIN_PROBABILITY", "")).strip()),
            legacy_pacing=(float(env.get("BURST_FACTOR") or 1) != 1
                or float(env.get("DIURNAL_AMPLITUDE") or 0) != 0
                or float(env.get("RAMP_UP_SECONDS") or 0) != 0
                or str(env.get("GRADIENT", "")).lower() in ("true", "1", "yes")))


def validate(p, *, mode, identity, scalar_present, legacy_pacing):
    return PlaybackIntensity.read(p, mode=mode, identity=identity, scalar_present=scalar_present,
                                  legacy_pacing=legacy_pacing).to_dict()


def from_env(env):
    return PlaybackEnvCodec.decode(env).to_dict()


def integral(curve, seconds):
    return RateCurve.read(curve or [[0, 1]]).integral(seconds)

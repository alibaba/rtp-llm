"""Frozen session settings and service admission limits."""

import math
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class PressureConfig:
    seed_timeout_s: float
    fill_timeout_s: float
    fill_max_requests: int
    fill_no_progress_s: float
    fill_input_tokens: int
    gpu_tolerance_blocks: int
    activity_timeout_s: float
    drain_timeout_s: float
    capture_timeout_s: float
    host_ratio: float = 0.25
    sample_period_s: float = 1.0
    max_sample_gap_s: float = 2.0
    max_sample_age_s: float = 2.0
    steady_s: float = 3.0
    steady_timeout_s: float = 60.0
    measure_s: float = 60.0
    min_coverage: float = 0.95
    max_invalid_s: float = 2.0
    profile_steps: int = 20
    support_s: float = 60.0
    nsys_duration_s: float | None = None
    control_timeout_s: float = 1.0
    family_schedule: str = "round_robin"
    memory_reuse_ratio: float = 0.0
    fill_host_max_ratio: float = 1.0
    rate_control: str = "fixed"
    request_rate_min: float = 0.0
    request_rate_max: float = 0.0
    backlog_low: int = 0
    backlog_high: int = 0
    max_outstanding: int = 0
    cohort_size: int = 0
    evidence_policy: str = "tree_pressure"
    mock_memory_target_ratio: float = 0.0
    mock_disk_target_ratio: float = 0.0
    mock_remote_target_ratio: float = 0.0
    mock_cache_settings: tuple = ()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=0, inclusive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (not inclusive and value == minimum):
        raise ValueError(f"{name} is below its allowed minimum")
    return value


def mock_provider_enabled(config, tier):
    return dict(config.mock_cache_settings).get(
        f"mock_{tier}_cache",
        config.evidence_policy == f"tree_then_mock_{tier}"
        or getattr(config, f"mock_{tier}_target_ratio") > 0,
    )


def mock_target_ppm(config):
    remaining = 1_000_000
    targets = {}
    for tier in ("remote", "disk", "memory"):
        scaled = getattr(config, f"mock_{tier}_target_ratio") * 1_000_000
        fraction, integer = math.modf(scaled)
        targets[tier] = min(int(integer) + (fraction >= 0.5), remaining)
        remaining -= targets[tier]
    return targets


def validate_profile_steps(steps):
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("profile_steps must be an integer between 1 and 1000")


def resolve_pressure_config(values: dict) -> PressureConfig:
    try:
        config = PressureConfig(**values)
        # JSON resumes represent tuples as lists; restore the immutable shape.
        config = replace(
            config,
            mock_cache_settings=tuple(
                tuple(item) for item in config.mock_cache_settings
            ),
        )
    except TypeError as error:
        raise ValueError(f"invalid pressure settings: {error}") from error
    integers = {
        "fill_max_requests",
        "fill_input_tokens",
        "gpu_tolerance_blocks",
        "profile_steps",
        "backlog_low",
        "backlog_high",
        "max_outstanding",
        "cohort_size",
    }
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in {
            "family_schedule",
            "rate_control",
            "evidence_policy",
            "mock_cache_settings",
        }:
            continue
        if field.name == "nsys_duration_s" and value is None:
            continue
        if field.name in integers:
            _integer(
                value,
                field.name,
                (
                    0
                    if field.name
                    in {
                        "gpu_tolerance_blocks",
                        "backlog_low",
                        "backlog_high",
                        "max_outstanding",
                        "cohort_size",
                    }
                    else 1
                ),
            )
        elif field.name in {
            "memory_reuse_ratio",
            "request_rate_min",
            "request_rate_max",
            "mock_memory_target_ratio",
            "mock_disk_target_ratio",
            "mock_remote_target_ratio",
        }:
            _number(value, field.name, inclusive=True)
        else:
            _number(value, field.name)
    validate_profile_steps(config.profile_steps)
    for name in (
        "host_ratio",
        "min_coverage",
        "memory_reuse_ratio",
        "fill_host_max_ratio",
        "mock_memory_target_ratio",
        "mock_disk_target_ratio",
        "mock_remote_target_ratio",
    ):
        if getattr(config, name) > 1:
            raise ValueError(f"{name} must be <= 1")
    if (
        config.mock_memory_target_ratio
        + config.mock_disk_target_ratio
        + config.mock_remote_target_ratio
        > 1
    ):
        raise ValueError("mock cache target ratios must sum to <= 1")
    if config.family_schedule not in {"round_robin", "tiered_rolling"}:
        raise ValueError("family_schedule must be round_robin or tiered_rolling")
    if config.rate_control not in {"fixed", "adaptive_backlog"}:
        raise ValueError("rate_control must be fixed or adaptive_backlog")
    if config.evidence_policy not in {
        "tree_pressure",
        "mock_memory",
        "tree_then_mock_memory",
        "tree_then_mock_disk",
        "tree_then_mock_remote",
    }:
        raise ValueError(
            "evidence_policy must be tree_pressure, mock_memory, tree_then_mock_memory or tree_then_mock_disk/tree_then_mock_remote"
        )
    adaptive_values = (
        config.request_rate_min,
        config.request_rate_max,
        config.backlog_low,
        config.backlog_high,
        config.max_outstanding,
        config.cohort_size,
    )
    if config.rate_control == "adaptive_backlog":
        if not 0 < config.request_rate_min <= config.request_rate_max:
            raise ValueError("adaptive rates must satisfy 0 < min <= max")
        if not 0 <= config.backlog_low < config.backlog_high:
            raise ValueError("adaptive backlog requires 0 <= low < high")
        if config.backlog_high >= config.max_outstanding:
            raise ValueError("adaptive backlog high must be below max outstanding")
        if config.cohort_size <= 0:
            raise ValueError("adaptive backlog requires a positive cohort size")
        if config.cohort_size % 2:
            raise ValueError("adaptive backlog requires an even cohort size")
        if config.family_schedule != "tiered_rolling":
            raise ValueError("adaptive backlog requires tiered_rolling")
    elif any(adaptive_values):
        raise ValueError("adaptive backlog settings require adaptive_backlog control")
    if config.family_schedule == "tiered_rolling":
        if config.memory_reuse_ratio != 0.5:
            raise ValueError("tiered_rolling requires a 50% memory reuse ratio")
        if config.fill_host_max_ratio <= config.host_ratio:
            raise ValueError("fill host maximum must exceed the pressure target")
    elif config.memory_reuse_ratio:
        raise ValueError("memory reuse ratio requires tiered_rolling")
    if min(config.max_sample_age_s, config.max_sample_gap_s) < config.sample_period_s:
        raise ValueError("sample age and gap must be >= sample period")
    return config


@dataclass(frozen=True)
class ExecutionLimits:
    """Verified service limits, in tokens and per-request physical KV blocks.

    available_kv_blocks is the conservative simultaneous request budget after
    non-reclaimable reservations, not total configured cache capacity.
    When supplied, owner_kv_budgets replaces the single-pool projection with
    (owner/group/pool identity, available blocks, tokens per block) per pool.
    """

    model_tokens: int
    admission_requests: int
    batch_tokens: int
    available_kv_blocks: int
    block_tokens: int
    owner_kv_budgets: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.model_tokens,
                    self.admission_requests,
                    self.batch_tokens,
                    self.block_tokens,
                )
            )
            or type(self.available_kv_blocks) is not int
            or self.available_kv_blocks < 0
        ):
            raise ValueError("execution limits must be positive integers")
        if type(self.owner_kv_budgets) is not tuple or any(
            type(budget) is not tuple
            or len(budget) != 3
            or not isinstance(budget[0], str)
            or not budget[0]
            or type(budget[1]) is not int
            or budget[1] < 0
            or type(budget[2]) is not int
            or budget[2] <= 0
            for budget in self.owner_kv_budgets
        ):
            raise ValueError("invalid immutable owner KV budgets")

    def concurrency_for(self, input_tokens):
        if type(input_tokens) is not int or not 0 < input_tokens < self.model_tokens:
            raise ValueError(
                f"input_tokens={input_tokens} plus one output exceeds "
                f"model token limit={self.model_tokens}"
            )
        tokens = input_tokens + 1
        if tokens > self.batch_tokens:
            raise ValueError(
                f"input_tokens={input_tokens} plus one output exceeds "
                f"batch token limit={self.batch_tokens}"
            )
        count = min(self.admission_requests, self.batch_tokens // tokens)
        budgets = self.owner_kv_budgets or (
            ("service", self.available_kv_blocks, self.block_tokens),
        )
        for name, available, block_tokens in budgets:
            blocks = (tokens + block_tokens - 1) // block_tokens
            if blocks > available:
                raise ValueError(
                    f"{name}: KV admission requires {blocks} blocks "
                    f"for input_tokens={input_tokens} plus one output; "
                    f"available_blocks={available} block_tokens={block_tokens}"
                )
            count = min(count, available // blocks)
        return count

    def require_request(self, input_tokens: int) -> None:
        self.concurrency_for(input_tokens)

"""Stable failure classes shared by every unified perf-test entry path."""

from __future__ import annotations


class PerfTestError(Exception):
    """Base class for failures that map to a stable process exit code."""

    exit_code = 1


class ConfigurationError(PerfTestError):
    """Invalid source config, command input, or capability combination."""

    exit_code = 2


class CapabilityError(ConfigurationError):
    """The requested mode, action, and backend combination is unsupported."""


class PlanningError(PerfTestError):
    """A valid config cannot produce a valid execution plan."""

    exit_code = 3


class ResourceError(PlanningError):
    """Requested devices or remote resources cannot satisfy the plan."""


class EngineLifecycleError(PerfTestError):
    """The serving engine failed to start, remain healthy, or stop."""

    exit_code = 4


class CaseExecutionError(PerfTestError):
    """One or more measurement cases failed or produced invalid output."""

    exit_code = 5


class PartialRunError(CaseExecutionError):
    """Some valid case output exists but the required plan did not complete."""


class PostExecutionError(PerfTestError):
    """Collection, merge, report, or required publication failed."""

    exit_code = 6


class CollectionError(PostExecutionError):
    """Remote shard results failed collection or consistency checks."""


class ReportError(PostExecutionError):
    """Required report generation failed."""


class PublisherError(PostExecutionError):
    """A required publisher failed."""

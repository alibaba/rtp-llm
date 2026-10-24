"""One collection-quality assessment; artifact fields are derived projections."""
from dataclasses import dataclass, field


def classify_gaps(gaps, evidence):
    """Only confirmed injections exempt the named source during that interval."""
    anchor = evidence["clock_anchor"]["epoch_s"]
    teardown = [e for e in evidence.get("phases", [])
                if e["action"] == "teardown" and e["event"] == "start"]
    expected, unexpected = {}, {}
    for key, times in gaps.items():
        source = "/".join(key.split("/")[:2])
        for timestamp in times:
            epoch = timestamp + anchor
            allowed = any(o["source"] == source
                          and o["started_epoch_s"] <= epoch <= o["ended_epoch_s"]
                          for o in evidence.get("expected_outages", []))
            allowed = allowed or any(str(e["env_epoch"]) == key.split("/")[0]
                                     and epoch >= e["epoch_s"] for e in teardown)
            (expected if allowed else unexpected).setdefault(key, []).append(timestamp)
    return expected, unexpected


def bounded_collection_gaps(series, gaps, max_gap_s):
    """Tolerate only gaps bracketed by successful scrapes within the budget."""
    tolerated, fatal = {}, {}
    for key, times in gaps.items():
        source = "/".join(key.split("/")[:2])
        healthy = sorted(t for name, points in series.items()
                         if name.startswith(source + "/up/")
                         for t, value in points if value == 1)
        for t in times:
            left = max((x for x in healthy if x < t), default=None)
            right = min((x for x in healthy if x > t), default=None)
            bounded = (left is not None and right is not None
                       and 0 < right - left <= max_gap_s)
            (tolerated if bounded else fatal).setdefault(key, []).append(t)
    return tolerated, fatal


@dataclass(frozen=True)
class CollectionIssue:
    kind: str
    detail: object
    source: str = ""

    @property
    def blocking(self):
        return self.kind in {"fatal", "integrity", "missing"}


@dataclass(frozen=True)
class CollectionQuality:
    metric_gaps: dict = field(default_factory=dict)
    source_gaps: dict = field(default_factory=dict)
    issues: tuple[CollectionIssue, ...] = ()
    max_gap_s: float = 0

    @property
    def valid(self):
        return not any(issue.blocking for issue in self.issues)

    @property
    def completeness(self):
        return "PARTIAL" if self.source_gaps or any(
            issue.kind in {"diagnostic", "integrity"} for issue in self.issues) else "COMPLETE"

    def to_dict(self):
        """Keep the existing report/archive schema without parallel mutable states."""
        classified = {kind: {} for kind in ("expected", "tolerated", "fatal")}
        for issue in self.issues:
            if issue.kind in classified:
                classified[issue.kind][issue.source] = list(issue.detail)
        tolerated, fatal = classified["tolerated"], classified["fatal"]
        unexpected = {}
        for source, times in self.source_gaps.items():
            selected = set(tolerated.get(source, ())) | set(fatal.get(source, ()))
            if selected:
                unexpected[source] = [t for t in times if t in selected]
        result = dict(
            telemetry_gaps=self.metric_gaps,
            collection_gaps=self.source_gaps,
            expected_telemetry_gaps=classified["expected"],
            unexpected_telemetry_gaps=unexpected,
            telemetry_completeness=self.completeness,
            tolerated_telemetry_gaps=tolerated,
            fatal_telemetry_gaps=fatal,
            telemetry_warnings=[dict(source=key, times=times, max_gap_s=self.max_gap_s,
                reason="Short collection gap bracketed by successful scrapes")
                for key, times in tolerated.items()],
            telemetry_diagnostics=[issue.detail for issue in self.issues if issue.kind == "diagnostic"],
            telemetry_integrity_errors=[issue.detail for issue in self.issues if issue.kind == "integrity"],
        )
        missing = [issue.source for issue in self.issues if issue.kind == "missing"]
        if missing:
            result["missing_telemetry"] = missing
        return result


def assess_collection(series, source_gaps, monitor_errors, evidence, *, capture_metrics, max_gap_s):
    metric_gaps = {key: [t for t, value in points if value is None]
                   for key, points in series.items() if any(value is None for _, value in points)}
    expected, unexpected = classify_gaps(source_gaps, evidence)
    tolerated, fatal = bounded_collection_gaps(series, unexpected, max_gap_s)
    issues = [CollectionIssue(kind, tuple(times), key)
              for kind, gaps in (("expected", expected), ("tolerated", tolerated), ("fatal", fatal))
              for key, times in gaps.items()]
    issues.extend(CollectionIssue("diagnostic" if error.get("severity") == "diagnostic" else "integrity",
                  error) for error in monitor_errors)
    if capture_metrics:
        present = {"/".join(key.split("/")[:2]) for key in series}
        expected_sources = evidence.get("expected_telemetry", [])
        missing = sorted(set(expected_sources) - present) if expected_sources else ["expected telemetry inventory absent"]
        issues.extend(CollectionIssue("missing", None, source) for source in missing)
    return CollectionQuality(metric_gaps, source_gaps, tuple(issues), max_gap_s)

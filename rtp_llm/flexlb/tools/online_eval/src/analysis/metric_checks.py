"""Bridge frozen MetricStore observations to pure check results."""

import copy
from analysis.checks import validate_comparison, invalid_check, evaluate


def check_metric(store, identity, metric_id, *, reduction, op, expected,
                 labels=None, start=None, end=None, source=None, epoch=None,
                 min_samples=1, max_gap_s=None, evidence=None, advisory=False):
    """Use frozen definitions and one explicitly selected series, without acquisition.

    MetricStore's inclusive sample window is distinct from request-cohort windows.
    Invalid observations become ERROR; definition/selector contract errors raise.
    """
    from monitoring.contracts import MetricUnavailable

    validate_comparison(op, expected)
    selection = dict(labels=labels, start=start, end=end, source=source, epoch=epoch,
                     min_samples=min_samples, max_gap_s=max_gap_s)
    provenance = dict(evidence or {}, metric=metric_id, selection=selection, reduction=reduction)
    try:
        actual = store.reduce(metric_id, op=reduction, **selection)
    except MetricUnavailable as exc:
        return invalid_check(identity, detail=str(exc), expected=expected, evidence=provenance)
    rows = store.select(metric_id, **selection)
    provenance["definition"] = store.document["definitions"][metric_id]
    provenance["observations"] = [dict(source=row["source"], epoch=row["epoch"],
        labels=row["labels"], provenance=row["provenance"], sample_count=len(row["points"]),
        observed_bounds=[row["points"][0][0], row["points"][-1][0]]) for row in rows]
    return evaluate(identity, actual, op, expected, evidence=copy.deepcopy(provenance), advisory=advisory)

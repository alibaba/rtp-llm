"""Implementation-owned descriptions for calculations that remain case-specific."""


def implementation_measurement(function, *, population, accuracy, request_identity):
    if not callable(function):
        raise ValueError('measurement requires an executable implementation')
    return dict(method=function.__module__ + '.' + function.__name__,
                population=population, accuracy=accuracy,
                requires_request_identity=request_identity)


def validate_measurement(value, path):
    """Describe evidence semantics; this is metadata, never an expression language."""
    if (not isinstance(value, dict)
            or set(value) != {"method", "population", "accuracy", "requires_request_identity"}
            or any(type(value[key]) is not str or not value[key].strip()
                   for key in ("method", "population"))
            or value["accuracy"] not in ("request_ledger", "sampled", "histogram_estimate", "counter_delta")
            or type(value["requires_request_identity"]) is not bool):
        raise ValueError(f"{path}: invalid measurement semantics")



class MetricContractError(ValueError):
    """An ID, identity or definition violates the declared metric contract."""


class MetricUnavailable(ValueError):
    """Declared data cannot establish a valid measurement."""


def validate_observation(row, metric_id):
    """Shared stored-row contract for publication, reading and saving."""
    if (not isinstance(row, dict)
            or not isinstance(row.get("labels"), dict)
            or not isinstance(row.get("provenance"), dict)
            or not isinstance(row["provenance"].get("evidence", {}), dict)):
        raise MetricContractError("invalid metric observation metadata")
    if (row.get("metric_id") != metric_id
            or any(type(row.get(key)) is not str or not row[key].strip()
                   for key in ("source", "epoch", "status"))
            or not isinstance(row.get("points"), list)):
        raise MetricContractError("invalid metric observation identity or fields")


def validate_definition(value, metric_id, *, publication=False):
    if not isinstance(value, dict) or type(value.get('unit')) is not str or not value['unit']:
        raise MetricContractError('metric definition requires unit: ' + metric_id)
    if publication or 'labels' in value:
        labels = value.get('labels')
        if not isinstance(labels, list) or any(type(label) is not str or not label for label in labels):
            raise MetricContractError('metric definition requires label names: ' + metric_id)
    if publication:
        if type(value.get('source_type')) is not str:
            raise MetricContractError('metric definition requires source_type: ' + metric_id)
        try:
            validate_measurement(value.get('measurement'), metric_id)
        except ValueError as exc:
            raise MetricContractError(str(exc)) from exc

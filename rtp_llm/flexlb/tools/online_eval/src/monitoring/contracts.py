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

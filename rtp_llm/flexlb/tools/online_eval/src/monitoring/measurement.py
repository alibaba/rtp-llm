"""Implementation-owned descriptions for calculations that remain case-specific."""


def implementation_measurement(function, *, population, accuracy, request_identity):
    if not callable(function):
        raise ValueError('measurement requires an executable implementation')
    return dict(method=function.__module__ + '.' + function.__name__,
                population=population, accuracy=accuracy,
                requires_request_identity=request_identity)

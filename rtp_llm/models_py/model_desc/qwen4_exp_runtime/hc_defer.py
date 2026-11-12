"""Selection policy for delayed HC residual materialization."""


def should_defer_hc_residual(*, cuda, hip, shape, bf16, sm, graph):
    """Select supported SM90 and existing SM120 residual fusion shapes."""
    if not cuda or hip or len(shape) != 2 or shape[1] != 10240 or not bf16:
        return False
    rows = shape[0]
    existing = not graph and sm == (12, 0) and 4093 <= rows <= 12288
    sm90 = sm == (9, 0) and 1 <= rows <= (12288 if graph else 32768)
    return existing or sm90

"""Selection policy for delayed HC residual materialization."""


def should_defer_hc_residual(*, enabled, cuda, hip, shape, bf16, sm, graph, extended=False):
    """Keep the existing prefill policy; opt into an SM90 small-M candidate.

    The caller captures this decision with the CUDA graph. Changing the flag
    requires a fresh capture. Numerical/latency validation is required before
    enabling the experimental SM90 path in a deployment.
    """
    if not cuda or hip or len(shape) != 2 or shape[1] != 10240 or not bf16:
        return False
    rows = shape[0]
    existing = not graph and sm == (12, 0) and 4093 <= rows <= 12288
    candidate = enabled and sm == (9, 0) and 1 <= rows <= 32
    wide_candidate = extended and sm == (9, 0) and 1 <= rows <= 12288
    return existing or candidate or wide_candidate

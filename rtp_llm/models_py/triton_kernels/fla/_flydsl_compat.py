"""Compatibility imports for the FlyDSL 0.3.1 FLA kernels.

FlyDSL 0.3.1 no longer exports ``buffer_ops`` and ``vector`` from
``flydsl.expr``. AITER 0.1.21 provides the matching compatibility helpers, so
keep that internal dependency isolated here and fail with an actionable error
if the pinned AITER/FlyDSL pair is not installed.

``fla.chunk`` probes this module behind the opt-in gate. Import failure disables
the FlyDSL path for the process and falls back to the existing Triton kernels.
"""

try:
    from aiter.ops.flydsl.kernels import buffer_ops, vector
except ImportError as exc:
    raise ImportError(
        "RTP-LLM FlyDSL FLA kernels require aiter>=0.1.21 with "
        "flydsl==0.3.1; the AITER FlyDSL compatibility helpers "
        "aiter.ops.flydsl.kernels.{buffer_ops,vector} are unavailable"
    ) from exc

__all__ = ["buffer_ops", "vector"]

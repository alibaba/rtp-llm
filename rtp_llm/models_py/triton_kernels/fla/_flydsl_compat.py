"""Compatibility imports for the FlyDSL 0.3.2 FLA kernels.

FlyDSL no longer exports ``buffer_ops`` or ``vector`` from ``flydsl.expr``.
The pinned AITER wheel still provides ``buffer_ops`` but no longer vendors its
legacy ``vector`` wrappers. Keep the latter local to the MI308X megakernels.

``fla.chunk`` probes this module behind the opt-in gate. Import failure disables
the FlyDSL path for the process and falls back to the existing Triton kernels.
"""

try:
    from aiter.ops.flydsl.kernels import buffer_ops
except ImportError as exc:
    raise ImportError(
        "RTP-LLM FlyDSL FLA kernels require the pinned AITER wheel with "
        "flydsl==0.3.2; the AITER FlyDSL compatibility helper "
        "aiter.ops.flydsl.kernels.buffer_ops is unavailable"
    ) from exc

from rtp_llm.models_py.triton_kernels.fla import _flydsl_vector_compat as vector

__all__ = ["buffer_ops", "vector"]

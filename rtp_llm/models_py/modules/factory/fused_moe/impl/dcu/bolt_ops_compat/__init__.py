"""Compatibility shim for the missing ``bolt_ops`` package (DAS aiter 0.1.5+das185).

The DAS aiter 0.1.5+das185 wheel imports MoE activation helpers from a
``bolt_ops`` package (``aiter/ops/triton/fused_moe.py`` and
``aiter/fused_moe_c.py`` do ``from bolt_ops.fused_moe import ...``), but that
wheel is not shipped in the DTK model-zoo images. As a result the whole aiter
MoE stack (triton / asm / moe_c backends) fails to import and MoE models are
blocked on DCU::

    ModuleNotFoundError: No module named 'bolt_ops'

The very same helpers exist in ``aiter.ops.triton.moe_activation`` (private
names, identical signatures), so this shim simply re-exports them under the
``bolt_ops.fused_moe`` namespace aiter expects.

Usage::

    from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.bolt_ops_compat import (
        ensure_bolt_ops_compat,
    )

    ensure_bolt_ops_compat()          # before importing aiter MoE modules
    from aiter.ops.triton.fused_moe import fused_experts_impl
"""

import importlib.util
import os
import sys

_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_bolt_ops_compat() -> bool:
    """Put the vendored ``bolt_ops`` shim on ``sys.path`` if needed.

    Returns True when the shim is active (i.e. no real ``bolt_ops`` was found).
    A real installation, once Hygon ships it, always takes precedence.
    """
    if importlib.util.find_spec("bolt_ops") is not None:
        return False
    if _SHIM_DIR not in sys.path:
        sys.path.insert(0, _SHIM_DIR)
    return True

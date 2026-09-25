"""``bolt_ops.fused_moe.triton.fused_moe`` compatibility facade.

``aiter.moe.aiter_moe`` (TRITON solution branch) imports
``fused_experts_impl`` from here. The canonical implementation lives in
``aiter.ops.triton.fused_moe``, which by the time this runs has already been
imported (or will be imported through this module without a cycle, because
``bolt_ops.fused_moe`` itself no longer references it).
"""

from aiter.ops.triton.fused_moe import fused_experts_impl  # noqa: F401

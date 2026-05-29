"""Single switch for ALL Triton fuse kernels in the model_py path.

Master switch lives on ``HWKernelConfig.enable_fuse_kernels`` (default
``True``) and is settable via CLI ``--enable_fuse_kernels`` / env var
``ENABLE_FUSE_KERNELS``.  Set to ``False`` to globally bypass to the
unfused baseline path (useful for debugging or precision verification).

After the DSV4-style strip, model forward code only contains unfused
baselines.  This switch now controls two things:

  1. **GraphFX installation** — when False, ``maybe_install_graphfx_fusions``
     skips wrapping ``py_model.forward`` with ``torch.compile``, so no
     FX pass fires and all forward code runs the pure PyTorch baseline.
  2. **Always-on fuse kernels** not managed by FX passes:
     - ``flashmla_sparse_impl._fuse_qk_rope_cat_cache_mla`` (custom_op)
     - ``Indexer._fuse_logits_head_gate`` (decode rope+quant fast path)
     - ``flashmla_sparse_impl._apply_input_bmm`` / ``_apply_output_bmm``
       (cuBLAS strided BMM, always preferred)

Callers either:
  * Pass ``hw_kernel_config`` to ``fuse_kernels_enabled(hw_kernel_config)``
    when they hold a reference to it (the canonical path), OR
  * Call ``fuse_kernels_enabled()`` with no arg in modules that don't
    have access to the config — falls back to env var
    ``ENABLE_FUSE_KERNELS`` (default ``True``).
"""

from __future__ import annotations

import os
from typing import Any, Optional


def fuse_kernels_enabled(hw_kernel_config: Optional[Any] = None) -> bool:
    """Return whether Triton fuse kernels should run (vs. unfused baseline).

    Resolution order:
      1. If ``hw_kernel_config`` is provided and has the
         ``enable_fuse_kernels`` attribute, return that.
      2. Otherwise read env var ``ENABLE_FUSE_KERNELS``
         (truthy strings ``1/true/yes/on``); default ``True``.
    """
    if hw_kernel_config is not None and hasattr(
        hw_kernel_config, "enable_fuse_kernels"
    ):
        return bool(hw_kernel_config.enable_fuse_kernels)
    val = os.environ.get("ENABLE_FUSE_KERNELS")
    if val is None:
        return True
    return val.lower() in ("1", "true", "yes", "on")

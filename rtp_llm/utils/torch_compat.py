import logging

import torch.library as torch_library

logger = logging.getLogger(__name__)
_wrap_triton_fallback_logged = False


def ensure_torch_library_wrap_triton() -> None:
    """Install the identity fallback required by older supported Torch releases."""
    global _wrap_triton_fallback_logged

    if not hasattr(torch_library, "wrap_triton"):

        def wrap_triton(fn):
            return fn

        torch_library.wrap_triton = wrap_triton
        if not _wrap_triton_fallback_logged:
            logger.warning(
                "torch.library.wrap_triton is unavailable; using the identity "
                "compatibility fallback"
            )
            _wrap_triton_fallback_logged = True

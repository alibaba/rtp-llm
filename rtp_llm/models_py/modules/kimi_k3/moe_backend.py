"""Select K3's native expert backend without changing other models."""

import logging
import json
from pathlib import Path
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _load_native():
    try:
        import k3_native_deep_gemm as backend
    except ImportError as error:
        raise RuntimeError(
            "K3 vllm_native MoE requires the pinned k3_native_deep_gemm package "
            "built for this RTP Torch runtime"
        ) from error
    if backend.__version__ != "2.8.0":
        raise RuntimeError("K3 requires the pinned DeepGEMM 2.8.0 SiTU backend")
    import torch
    try:
        build = json.loads(Path(backend.__file__).with_name("BUILD_INFO.json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("K3 native MoE is missing its verified build manifest") from error
    if (
        build.get("commit") != "a6bbb8000161c0dc3a85a0300a905f76898a7913"
        or build.get("patch_sha256") != "3685360daa68961c64db6ecce52564fde3cff9c40feafb82095096c64de0e9ef"
        or build.get("torch") != torch.__version__
    ):
        raise RuntimeError("K3 native MoE build does not match the pinned source, Graph patch, or Torch runtime")
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.activation import (
        situ_kwargs,
    )
    situ_kwargs(backend.fp8_fp4_mega_moe, 4.0, 25.0)
    logging.info("[K3 MoE] native backend=%s version=%s", backend.__file__, backend.__version__)
    return backend


def get_k3_moe_backend():
    mode = os.environ.get("KIMI_K3_MOE_BACKEND", "rtp")
    if mode == "rtp":
        return None
    if mode == "vllm_native":
        return _load_native()
    raise ValueError(f"Unknown KIMI_K3_MOE_BACKEND={mode!r}; expected rtp|vllm_native")

"""Opt-in isolated FlashInfer runtime for K3; run before attention imports."""

import os
import sys
from pathlib import Path


def setup_flashinfer_runtime():
    value = os.environ.get("KIMI_K3_FLASHINFER_ROOT")
    if not value:
        return
    if os.environ.get("MODEL_TYPE") not in ("kimi_k3", "kimi_k3_mtp"):
        raise RuntimeError("KIMI_K3_FLASHINFER_ROOT is only supported for K3")
    root = Path(value).resolve(strict=True)
    expected = (root / "flashinfer").resolve(strict=True)
    loaded = sys.modules.get("flashinfer")
    if loaded is not None and Path(loaded.__file__).resolve().parent != expected:
        raise RuntimeError("FlashInfer was imported before K3 runtime selection; restart the process")
    sys.path.insert(0, str(root))
    # FlashInfer caches its architecture probe on first use. Select CUTLASS
    # before any FlashInfer import can cache an unavailable/older DSL.
    from .models_py.utils.cutlass import setup_cutlass_import_path

    setup_cutlass_import_path()

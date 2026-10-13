"""Explicit opt-in bootstrap for the isolated vLLM benchmark interpreter."""

import os

_installed = False


def install():
    global _installed
    if _installed:
        return
    if os.environ.get("QWEN38_FULL_STATE_PD_PLUGIN") != "1":
        raise ValueError("The full-state benchmark plugin requires explicit opt-in")
    import vllm

    from .fp8_weights import install as install_fp8
    from .weight_loader import register

    if vllm.__version__ != "0.30.1rc1.dev256+gd6cce94fd":
        raise ValueError("vLLM core differs from the tested source contract")
    install_fp8()
    register()
    _installed = True

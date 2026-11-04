"""PPU backend entrypoint for the public model factories.

The implementation modules are public; this entrypoint only records their
backend hooks before the factories consume those slots.
"""

import os
import sys
import importlib.util
from pathlib import Path

_is_ppu = False
try:
    from rtp_llm.device.device_type import DeviceType, get_device_type

    _is_ppu = get_device_type() == DeviceType.Ppu
    if _is_ppu:
        # Spawned PPU workers inherit the cache paths through the environment.
        for _p in os.environ.get("_JIT_CACHE_PATHS", "").split(":"):
            if _p and _p not in sys.path:
                sys.path.insert(0, _p)
        # Loading through the factory packages executes their __init__.py files,
        # which initializes the slots before these hooks can be recorded. Load
        # the three lightweight registration modules directly instead.
        _register_root = Path(__file__).parent / "modules" / "factory"

        def _load_register(name: str, package: str):
            module_name = f"rtp_llm._ppu_{name}_register"
            module = sys.modules.get(module_name)
            if module is None:
                path = _register_root / package / "register.py"
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load PPU register module: {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            return module

        attn_register = _load_register("attention", "attention")
        moe_register = _load_register("fused_moe", "fused_moe")
        linear_register = _load_register("linear", "linear")

        attn_register.install()
        moe_register.install()
        linear_register.install()
except Exception as exc:  # noqa: BLE001
    if _is_ppu:
        raise
    import logging

    logging.warning("[ppu] failed to install backend hooks: %s", exc, exc_info=True)

"""Load the real MoE module without top-level model/native registration.

Only namespace/native-registration shells are stubbed, as in test_chunked_moe.
The tested cache helper, Torch allocation, and CUDA graph operations are real.
"""

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".device",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
    ".models_py.modules.dsv4.moe",
    ".models_py.modules.dsv4.moe.strategies",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)
for name in ["librtp_compute_ops", "librtp_compute_ops.rtp_llm_ops"]:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules.setdefault(name, module)

moe_layer = importlib.import_module("rtp_llm.models_py.modules.dsv4.moe.moe_layer")

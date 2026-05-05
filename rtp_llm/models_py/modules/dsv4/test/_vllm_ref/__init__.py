# Frozen vLLM kernel references for DSV4 FP8 KV cache UTs.
# DO NOT modify the vendored .py files. See VLLM_HEAD.txt for source commit.
#
# Import shim: vendored files do `from vllm.triton_utils import tl, triton`,
# but vllm isn't installed here. Register a fake `vllm.triton_utils` module
# that re-exports the real triton so the vendored source runs unmodified.

import sys
import types

import triton
import triton.language as tl

if "vllm" not in sys.modules:
    _vllm_pkg = types.ModuleType("vllm")
    _vllm_pkg.__path__ = []  # mark as package
    sys.modules["vllm"] = _vllm_pkg

if "vllm.triton_utils" not in sys.modules:
    _triton_utils = types.ModuleType("vllm.triton_utils")
    _triton_utils.tl = tl
    _triton_utils.triton = triton
    sys.modules["vllm.triton_utils"] = _triton_utils
    sys.modules["vllm"].triton_utils = _triton_utils

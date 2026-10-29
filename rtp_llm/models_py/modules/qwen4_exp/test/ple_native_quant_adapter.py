"""Load the original CUDA quantizer without the full engine Python package.

This test-only adapter executes the production per-tensor Python wrapper AST
unchanged and binds its native symbol to a Bazel-built registration of the same
production CUDA function. It does not validate the full compute-ops packaging.
"""

import ast
import contextlib
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common import fp8_linear
from rtp_llm.models_py.modules.qwen4_exp.ple import ple

_library = (
    Path(os.environ["TEST_SRCDIR"])
    / os.environ["TEST_WORKSPACE"]
    / "rtp_llm/models_py/modules/qwen4_exp/test/ple_native_quant_test_ops.so"
)
torch.ops.load_library(str(_library))
_source = Path(ple.__file__).parents[3] / "kernels/cuda/fp8_kernel/fp8_kernel.py"
_tree = ast.parse(_source.read_text())
_function = next(
    n
    for n in _tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "scaled_fp8_per_tensor_quant"
)
_namespace = {
    "torch": torch,
    "Optional": Optional,
    "Tuple": Tuple,
    "Union": Union,
    "per_tensor_quant_fp8": torch.ops.ple_component_test.per_tensor_quant_fp8,
}
exec(
    compile(ast.Module(body=[_function], type_ignores=[]), str(_source), "exec"),
    _namespace,
)
scaled_fp8_per_tensor_quant = _namespace["scaled_fp8_per_tensor_quant"]


def provenance():
    return {
        "native_binding_sha256": hashlib.sha256(_library.read_bytes()).hexdigest(),
        "production_wrapper_source_sha256": hashlib.sha256(
            _source.read_bytes()
        ).hexdigest(),
        "production_wrapper_ast_sha256": hashlib.sha256(
            ast.dump(_function).encode()
        ).hexdigest(),
        "cuda_implementation": "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "packaging_scope": "test-only native registration; full rtp_compute_ops import NOT_RUN",
    }


def _quantize(rows, *, role, scale=None, output=None):
    if torch.version.hip is not None:
        raise RuntimeError("This native CUDA test adapter does not validate ROCm")
    if scale is None and role in ("DECODE", "PDFUSION"):
        from rtp_llm.models_py.triton_kernels.common.fp8_small_quant import (
            try_small_fp8_quant,
        )

        result = try_small_fp8_quant(rows, output=output)
        if result is not None:
            return result
    return scaled_fp8_per_tensor_quant(rows, scale=scale, output=output)


@contextlib.contextmanager
def native_projection_adapter():
    with patch.object(fp8_linear, "_quantize_projection_input", _quantize):
        yield

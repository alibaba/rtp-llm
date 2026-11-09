"""Isolated import contracts; these tests do not validate device kernels."""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_CUDA_SYMBOLS = (
    "FusedRopeKVCacheDecodeOp",
    "FusedRopeKVCachePrefillOpQKVOut",
    "FusedRopeKVCachePrefillOpQOut",
)
_ROCM_SYMBOLS = (
    "FusedRopeKVCacheDecodeOpAsm",
    "FusedRopeKVCacheDecodeOpNonAsm",
    "FusedRopeKVCachePrefillOpAsm",
    "FusedRopeKVCachePrefillOpNonAsm",
)


def module_with_symbols(name, symbols=()):
    module = types.ModuleType(name)
    module.__path__ = []
    for symbol in symbols:
        setattr(module, symbol, object())
    return module


class ComputeOpsBackendTest(unittest.TestCase):
    def load(self, backend, native_symbols):
        native = module_with_symbols("librtp_compute_ops.rtp_llm_ops", native_symbols)
        native_parent = module_with_symbols("librtp_compute_ops")
        native_parent.rtp_llm_ops = native
        device = module_with_symbols("rtp_llm.device.device_type")
        device.is_cuda = lambda: backend == "cuda"
        device.is_hip = lambda: backend == "rocm"
        cuda = module_with_symbols("rtp_llm.ops.fused_rope_kvcache_op", _CUDA_SYMBOLS)
        modules = {
            "librtp_compute_ops": native_parent,
            "librtp_compute_ops.rtp_llm_ops": native,
            "rtp_llm": module_with_symbols("rtp_llm"),
            "rtp_llm.ops": module_with_symbols("rtp_llm.ops"),
            "rtp_llm.device": module_with_symbols("rtp_llm.device"),
            "rtp_llm.device.device_type": device,
            "rtp_llm.ops.fused_rope_kvcache_op": cuda,
        }
        source = Path(__file__).resolve().parents[1] / "compute_ops.py"
        spec = importlib.util.spec_from_file_location("rtp_llm.ops.compute_ops", source)
        loaded = importlib.util.module_from_spec(spec)
        # Inject dependencies BEFORE executing the production module, so a
        # stale required symbol cannot be hidden by a prior successful import.
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(loaded)
        return loaded, native, cuda

    def test_rocm_requires_only_layout_specific_bindings(self):
        with self.assertNoLogs(level=logging.WARNING):
            loaded, native, _ = self.load("rocm", _ROCM_SYMBOLS)
        for symbol in _ROCM_SYMBOLS:
            self.assertIs(getattr(loaded, symbol), getattr(native, symbol))
        self.assertFalse(hasattr(loaded, "FusedRopeKVCacheDecodeOp"))

    def test_missing_rocm_binding_fails_instead_of_silent_fallback(self):
        for missing in _ROCM_SYMBOLS:
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ImportError, missing):
                    self.load("rocm", [s for s in _ROCM_SYMBOLS if s != missing])

    def test_cuda_keeps_python_kernel_wrappers(self):
        loaded, _, cuda = self.load("cuda", _CUDA_SYMBOLS)
        for symbol in _CUDA_SYMBOLS:
            self.assertIs(getattr(loaded, symbol), getattr(cuda, symbol))

    def test_ppu_keeps_native_qout_contract(self):
        loaded, native, _ = self.load("ppu", _CUDA_SYMBOLS)
        for symbol in _CUDA_SYMBOLS:
            self.assertIs(getattr(loaded, symbol), getattr(native, symbol))


if __name__ == "__main__":
    unittest.main()

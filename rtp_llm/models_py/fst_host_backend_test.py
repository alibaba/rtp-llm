import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py import fst_host_backend as fst


def reason(backend):
    return fst.fst_host_backend_reason(backend, use_shm=True, use_nogds=False)


class ShmReader:
    def load_safetensors_to_device(self, path):
        raise AssertionError("preflight must not open a checkpoint")


class HostBackendTest(unittest.TestCase):
    def setUp(self):
        self.package = Path("/test/fast_safetensors")
        self.wrapper = SimpleNamespace(
            __name__="fast_safetensors", LoadWithShm=ShmReader
        )
        self.native = SimpleNamespace(
            __name__="fast_safetensors._C",
            LoadWithShm=ShmReader,
            __file__=str(self.package / "_C.so"),
        )

    def modules(self):
        return patch.dict(
            sys.modules,
            {
                self.wrapper.__name__: self.wrapper,
                self.native.__name__: self.native,
            },
        )

    def test_unrelated_or_unsafe_modes_do_not_import_reader(self):
        with patch.object(fst, "_package_build", side_effect=AssertionError):
            self.assertIsNone(reason("Cuda"))
            for backend in ("ROCm", "Ppu"):
                for shm, nogds in ((False, False), (False, True), (True, True)):
                    self.assertIn(
                        "SHM",
                        fst.fst_host_backend_reason(
                            backend, use_shm=shm, use_nogds=nogds
                        ),
                    )

    def test_cross_backend_or_unknown_ppu_never_imports_native(self):
        with patch.object(importlib, "import_module", side_effect=AssertionError):
            for backend, build in (
                ("Ppu", "Cuda"),
                ("Ppu", "ROCm"),
                ("Ppu", None),
                ("ROCm", "Cuda"),
            ):
                with patch.object(
                    fst, "_package_build", return_value=(self.package, build)
                ):
                    self.assertIn(backend.lower(), reason(backend).lower())

    def test_missing_or_broken_extension_is_an_actionable_reason(self):
        with patch.object(
            fst, "_package_build", side_effect=ImportError("wheel not found")
        ):
            self.assertIn("wheel not found", reason("Ppu"))
        with patch.object(fst, "_package_build", return_value=(self.package, "ROCm")):
            with patch.object(
                importlib, "import_module", side_effect=OSError("ABI mismatch")
            ):
                self.assertIn("ABI mismatch", reason("ROCm"))

    def test_matched_runtime_requires_python_and_native_method(self):
        for backend in ("ROCm", "Ppu"):
            with self.modules(), patch.object(
                fst, "_package_build", return_value=(self.package, backend)
            ):
                self.assertIsNone(reason(backend))
                for module in (self.native, self.wrapper):
                    with patch.object(module, "LoadWithShm", object):
                        self.assertIn("load_safetensors_to_device", reason(backend))
                with patch.object(
                    self.native, "__file__", "/another/fast_safetensors/_C.so"
                ):
                    self.assertIn("different packages", reason(backend))

    def test_untagged_rocm_requires_the_loaded_readers_hip_abi(self):
        symbols = dict(
            hipMemcpyAsync=object(), hipHostRegister=object(), hipStreamCreate=object()
        )
        with self.modules(), patch.object(
            fst, "_package_build", return_value=(self.package, None)
        ):
            for exports, expected in (
                (symbols, None),
                ({}, "HIP"),
                ({**symbols, "cudaMemcpyAsync": object()}, "HIP"),
            ):
                with patch.object(
                    fst.ctypes, "CDLL", return_value=SimpleNamespace(**exports)
                ) as load:
                    result = reason("ROCm")
                    if expected:
                        self.assertIn(expected, result)
                    else:
                        self.assertIsNone(result)
                    load.assert_called_once_with(self.native.__file__)

    def test_metadata_follows_bazel_package_not_shadowing_dist_info(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pkg = root / "wheel" / "fast_safetensors"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").touch()
            for parent, tag in (
                (root / "other", "cu130"),
                (pkg.parent, "torch2.1.2.rocm"),
            ):
                info = parent / f"fast_safetensors-0.7.3+{tag}.dist-info"
                info.mkdir(parents=True)
                (info / "METADATA").write_text(
                    f"Name: fast-safetensors\nVersion: 0.7.3+{tag}\n"
                )
            with patch.dict(sys.modules), patch.object(
                sys, "path", [str(root / "other"), str(pkg.parent)] + sys.path
            ):
                # Another test may already have imported the real native wheel.
                # Exercise first resolution on this path, then restore the module.
                sys.modules.pop("fast_safetensors", None)
                self.assertEqual(fst._package_build(), (pkg, "ROCm"))


if __name__ == "__main__":
    unittest.main()

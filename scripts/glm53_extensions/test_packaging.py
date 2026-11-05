"""Exercise wheel preservation and fail-closed native asset validation on CPU."""

import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import build


class PackagingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = self.root / "payload"
        self.payload.mkdir()
        self.files = {
            "flash_mla_glm53_22.abi3.so": b"\x7fELF-native-fixture",
            "mega_mhc_isolated/mhc_isolated.cpython-310-x86_64-linux-gnu.so": b"\x7fELF-mhc-fixture",
            "mega_mhc_isolated/csrc/jit_kernels/impls/sm100_mega_mhc.hpp": b"kernel source",
            "mega_mhc_isolated/include/deep_jit/utils/parser.hpp": b"parser source",
            "mega_mhc_isolated/include/cute/tensor.hpp": b"cute source",
            "mega_mhc_isolated/include/cutlass/cutlass.h": b"cutlass source",
            "mega_mhc_isolated/licenses/deep_gemm.txt": b"upstream license",
            "fp8_mqa_isolated/_rtp_llm_fp8_mqa.cpython-310-x86_64-linux-gnu.so": b"\x7fELF-mqa-fixture",
            "fp8_mqa_isolated/include/deep_gemm/impls/sm100_mqa_logits.cuh": b"mqa kernel",
            "fp8_mqa_isolated/csrc/jit/compiler.hpp": b"private compiler",
            "fp8_mqa_isolated/LICENSE": b"upstream license",
        }
        for name, data in self.files.items():
            path = self.payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest = {
            "cuda_architectures": ["sm_100a"],
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in self.files.items()
            },
        }
        self.write_manifest()
        self.wheel = self.root / "rtp_llm-0.2.0-cp310-cp310-linux_x86_64.whl"
        self.original = {
            "rtp_llm/unchanged.py": b"value=42\n",
            "rtp_llm/libs/unchanged.so": b"other binary",
            "rtp_llm-0.2.0.dist-info/METADATA": b"Name: rtp-llm\nVersion: 0.2.0\n",
            "rtp_llm-0.2.0.dist-info/RECORD": b"",
        }
        with zipfile.ZipFile(self.wheel, "w") as out:
            for name, data in self.original.items():
                out.writestr(name, data)

    def write_manifest(self):
        (self.payload / "glm53_extensions.json").write_text(json.dumps(self.manifest))

    def test_roundtrip_preserves_existing_artifacts(self):
        build.extend_wheel(self.wheel, self.payload)
        build.check_wheel(self.wheel)
        with zipfile.ZipFile(self.wheel) as package:
            for name, data in self.original.items():
                if not name.endswith("/RECORD"):
                    self.assertEqual(package.read(name), data)
            self.assertIsNone(package.testzip())

    def test_repeated_packaging_has_no_duplicate_members(self):
        build.extend_wheel(self.wheel, self.payload)
        build.extend_wheel(self.wheel, self.payload)
        with zipfile.ZipFile(self.wheel) as package:
            self.assertEqual(len(package.namelist()), len(set(package.namelist())))
        build.check_wheel(self.wheel)

    def test_missing_native_binary_leaves_original_wheel_unchanged(self):
        original = self.wheel.read_bytes()
        (self.payload / "flash_mla_glm53_22.abi3.so").unlink()
        with self.assertRaisesRegex(ValueError, "Missing extension"):
            build.extend_wheel(self.wheel, self.payload)
        self.assertEqual(self.wheel.read_bytes(), original)

    def test_stale_jit_header_is_rejected(self):
        (
            self.payload / "mega_mhc_isolated/include/deep_jit/utils/parser.hpp"
        ).write_bytes(b"stale source")
        with self.assertRaisesRegex(ValueError, "Stale or corrupt"):
            build.extend_wheel(self.wheel, self.payload)

    def test_fake_shared_library_is_rejected(self):
        name = "flash_mla_glm53_22.abi3.so"
        data = b"not a shared library"
        (self.payload / name).write_bytes(data)
        self.manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Not an ELF"):
            build.extend_wheel(self.wheel, self.payload)

    def test_duplicate_original_wheel_is_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.wheel, "a") as package:
                package.writestr("rtp_llm/unchanged.py", b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate members"):
            build.extend_wheel(self.wheel, self.payload)
        self.assertFalse(self.wheel.with_suffix(".glm53.tmp").exists())

    def test_corrupt_record_is_rejected(self):
        build.extend_wheel(self.wheel, self.payload)
        source = self.wheel.with_suffix(".old")
        self.wheel.rename(source)
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(self.wheel, "w") as new:
            for name in old.namelist():
                data = b"" if name.endswith("/RECORD") else old.read(name)
                new.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "Invalid wheel RECORD"):
            build.check_wheel(self.wheel)


if __name__ == "__main__":
    unittest.main()

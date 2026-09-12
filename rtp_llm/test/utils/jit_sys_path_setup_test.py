import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import jit_sys_path_setup as setup


class TileLangNativePackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "external/site-packages"
        self.runfiles = self.root / "runfiles/pip_tilelang/site-packages"
        self.cache = self.root / "cache"
        self.version = "0.1.9+git441c3b06.dsv41.123456789abc"
        self.metadata_name = "tilelang-" + self.version + ".dist-info"
        self.payloads = {
            "tilelang/__init__.py": b"__version__ = '" + self.version.encode() + b"'\n",
            "tilelang/lib/libtvm.so": bytes(range(256)) * 8,
            "tilelang/src/transform/thread_storage_sync.cc": b"qualified source\n",
            "tilelang/rtp_build_manifest.json": json.dumps(
                {"version": self.version, "build_identity": "1" * 64}
            ).encode(),
            self.metadata_name
            + "/METADATA": (
                "Metadata-Version: 2.1\nName: tilelang\nVersion: " + self.version + "\n"
            ).encode(),
            self.metadata_name + "/WHEEL": b"Wheel-Version: 1.0\n",
        }
        for relative, payload in self.payloads.items():
            source = self.source / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
            runfile = self.runfiles / relative
            runfile.parent.mkdir(parents=True, exist_ok=True)
            runfile.symlink_to(source)
        self.distribution = importlib.metadata.Distribution.at(
            self.runfiles / self.metadata_name
        )
        patch = mock.patch.object(
            setup.importlib.metadata, "distribution", return_value=self.distribution
        )
        patch.start()
        self.addCleanup(patch.stop)

    def materialize(self):
        return Path(setup.copy_package_with_lock("tilelang", self.cache))

    def test_materializes_matching_package_and_metadata_without_changing_bytes(self):
        self.assertIsNone(self.distribution.files)
        target = self.materialize()
        for relative, payload in self.payloads.items():
            with self.subTest(relative=relative):
                self.assertEqual((target / relative).read_bytes(), payload)
                self.assertFalse((target / relative).is_symlink())
                self.assertEqual((self.source / relative).read_bytes(), payload)
                self.assertTrue((self.runfiles / relative).is_symlink())

        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((str(target), str(self.runfiles)))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.metadata, json, tilelang; from pathlib import Path; "
                "d = importlib.metadata.distribution('tilelang'); "
                "print(json.dumps([str(Path(tilelang.__file__).resolve().parent), "
                "str(Path(d.locate_file('tilelang')).resolve()), d.version]))",
            ],
            cwd=self.root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            [str(target / "tilelang"), str(target / "tilelang"), self.version],
        )

    def test_complete_cache_reuses_the_same_physical_installation(self):
        target = self.materialize()
        with mock.patch.object(setup.shutil, "copytree") as copy:
            self.assertEqual(self.materialize(), target)
            copy.assert_not_called()

    def test_incomplete_package_cannot_reuse_a_completion_marker(self):
        target = self.materialize()
        (target / "tilelang/__init__.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "cache is invalid"):
            self.materialize()

    def test_concurrent_copies_publish_one_complete_installation(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.materialize(), range(4)))
        self.assertEqual(len(set(results)), 1)
        for relative, payload in self.payloads.items():
            self.assertEqual((results[0] / relative).read_bytes(), payload)
        self.assertEqual(len(list(self.cache.glob("tilelang_python-native-*"))), 1)

    def test_metadata_copy_failure_does_not_publish_or_remove_the_source(self):
        copytree = shutil.copytree

        def fail_metadata(source, destination, *args, **kwargs):
            if Path(source).name == self.metadata_name:
                raise OSError("injected metadata copy failure")
            return copytree(source, destination, *args, **kwargs)

        with mock.patch.object(setup.shutil, "copytree", side_effect=fail_metadata):
            with self.assertRaisesRegex(OSError, "injected metadata"):
                self.materialize()
        self.assertEqual(list(self.cache.glob("tilelang_python-native-*")), [])
        self.assertEqual(list(self.cache.rglob(".copy_complete")), [])
        for relative, payload in self.payloads.items():
            self.assertEqual((self.source / relative).read_bytes(), payload)
        self.assertTrue((self.materialize() / "tilelang/__init__.py").is_file())

    def test_rejects_package_and_metadata_from_different_wheels(self):
        other_source = self.root / "other/site-packages"
        shutil.copytree(self.source, other_source)
        source_init = self.runfiles / "tilelang/__init__.py"
        source_init.unlink()
        source_init.symlink_to(other_source / "tilelang/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "different wheels"):
            self.materialize()
        self.assertFalse(self.cache.exists())

    def test_rejects_manifest_version_mismatch(self):
        manifest = self.source / "tilelang/rtp_build_manifest.json"
        manifest.write_text(json.dumps({"version": "0.1.9"}))
        with self.assertRaisesRegex(RuntimeError, "manifest.*versions differ"):
            self.materialize()
        self.assertFalse(self.cache.exists())

    def test_explicit_unqualified_tilelang_keeps_the_existing_copy_behavior(self):
        self.cache.mkdir()
        with mock.patch.object(
            setup.importlib.metadata,
            "distribution",
            return_value=mock.Mock(version="0.1.9"),
        ), mock.patch.object(
            setup,
            "get_package_info",
            return_value=("0.1.9", str(self.source / "tilelang")),
        ):
            target = Path(setup.copy_package_with_lock("tilelang", self.cache))
        self.assertEqual(target, self.cache / "tilelang_python-0.1.9/site-packages")
        self.assertEqual(
            (target / "tilelang/__init__.py").read_bytes(),
            self.payloads["tilelang/__init__.py"],
        )

    def test_default_packages_do_not_interpose_unqualified_tilelang(self):
        with mock.patch.object(
            setup.importlib.metadata,
            "distribution",
            return_value=mock.Mock(version="0.1.9"),
        ), mock.patch.object(
            setup, "copy_package_with_lock", return_value=None
        ) as copy:
            setup.setup_jit_cache(str(self.cache))
        self.assertEqual(
            [call.args[0] for call in copy.call_args_list],
            ["flashinfer", "torch", "deep_gemm", "tvm_ffi"],
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock

from rtp_llm.test.utils import jit_sys_path_setup as setup


class FlashinferCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_op(self, relative, compiler=None):
        op = self.root / relative
        op.mkdir(parents=True)
        (op / "build.ninja").write_text(
            f"cxx = {compiler or sys.executable}\ncflags = -I{self.root}\n"
        )
        return op

    def test_versioned_workspace_drops_removed_compiler_only(self):
        prefix = ".cache/flashinfer/0.6.9/100a/cached_ops"
        stale = self.make_op(f"{prefix}/stale", self.root / "removed-g++")
        valid = self.make_op(f"{prefix}/valid")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(stale.exists())
        self.assertTrue((valid / "build.ninja").exists())

    def test_resolved_workspace_drops_missing_nvcc_host_compiler(self):
        op = self.make_op("cached_ops/stale")
        with (op / "build.ninja").open("a") as f:
            f.write(f"cuda_cflags = -ccbin {self.root}/removed-gcc\n")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(op.exists())

    def test_keeps_busy_op(self):
        op = self.make_op("cached_ops/stale", self.root / "removed-g++")
        lock_path = op.parent / "tmp" / "stale.lock"
        lock_path.parent.mkdir()
        with FileLock(str(lock_path)):
            with patch.object(
                setup, "FileLock", side_effect=lambda p, **kw: FileLock(p, timeout=0)
            ):
                setup._evict_dangling_flashinfer_ops(self.root)
        self.assertTrue(op.exists())

    def test_does_not_scan_unrelated_directories(self):
        op = self.make_op("models/model/cached_ops/stale", self.root / "removed-g++")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertTrue(op.exists())

    def test_drops_runfiles_includes(self):
        op = self.make_op("cached_ops/stale")
        include = self.root / "action.runfiles" / "include"
        include.mkdir(parents=True)
        with (op / "build.ninja").open("a") as f:
            f.write(f"cflags = -isystem {include}\n")
        setup._evict_dangling_flashinfer_ops(self.root)
        self.assertFalse(op.exists())

    def test_patched_wheel_uses_its_own_cached_copy(self):
        paths = []
        cache = self.root / "cache"
        cache.mkdir()
        for contents in ("original", "patched"):
            runfiles = self.root / contents
            site = runfiles / "pip_gpu_cuda13_torch_flashinfer_python" / "site-packages"
            package = site / "flashinfer"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(contents)
            metadata = site / "flashinfer_python-0.7.5.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Name: flashinfer-python\nVersion: 0.7.5\n"
            )
            (metadata / "RECORD").write_text(f"flashinfer/__init__.py,{contents}\n")
            with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
                cached_site = Path(setup.copy_package_with_lock("flashinfer", cache))
            self.assertEqual(
                (cached_site / "flashinfer/__init__.py").read_text(), contents
            )
            paths.append(cached_site)
        self.assertNotEqual(*paths)

    def test_interrupted_libs_copy_is_redone(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self.root / "runfiles"
        site = runfiles / "pip_gpu_cuda13_torch_flashinfer_python" / "site-packages"
        package = site / "flashinfer"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("package")
        libs = site / "flashinfer.libs"
        libs.mkdir()
        (libs / "libdep.so").write_text("lib")
        metadata = site / "flashinfer_python-0.7.5.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: flashinfer-python\nVersion: 0.7.5\n")
        (metadata / "RECORD").write_text("flashinfer/__init__.py,package\n")
        copytree = setup.shutil.copytree

        def interrupted(source, target, **kwargs):
            if Path(source).name == "flashinfer.libs":
                Path(target).mkdir(parents=True)
                raise OSError("interrupted")
            return copytree(source, target, **kwargs)

        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            with patch.object(setup.shutil, "copytree", side_effect=interrupted):
                self.assertIsNone(setup.copy_package_with_lock("flashinfer", cache))
            cached_site = Path(setup.copy_package_with_lock("flashinfer", cache))
        self.assertEqual(
            (cached_site / "flashinfer.libs" / "libdep.so").read_text(), "lib"
        )

    def _make_mock_deep_gemm_runfiles(self) -> Path:
        runfiles = self.root / "deep_gemm_runfiles"
        site = runfiles / "pip_gpu_cuda13_torch_deep_gemm" / "site-packages"
        package = site / "deep_gemm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("__version__ = '2.8.2'\n")
        sm120_header = site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
        sm120_header.parent.mkdir(parents=True)
        sm120_header.write_text("#pragma once\n")
        sm120_helper = site / "deep_gemm_sm120" / "shim" / "include" / "helper.cuh"
        sm120_helper.write_text("#define HELPER 1\n")
        metadata = site / "deep_gemm-2.8.2.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: deep-gemm\nVersion: 2.8.2\n")
        (metadata / "RECORD").write_text(
            "deep_gemm/__init__.py,abc\n"
            "deep_gemm_sm120/shim/include/header.cuh,def\n"
            "deep_gemm_sm120/shim/include/helper.cuh,ghi\n"
        )
        return runfiles

    def test_deep_gemm_copies_sm120_companion_dir(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertTrue((cached_site / "deep_gemm" / "__init__.py").is_file())
        self.assertEqual(
            (
                cached_site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
            ).read_text(),
            "#pragma once\n",
        )

    def test_deep_gemm_repopulates_cache_missing_sm120_companion_dir(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            init_ino = (cached_site / "deep_gemm" / "__init__.py").stat().st_ino
            marker_ino = (cached_site.parent / ".copy_complete").stat().st_ino
            setup.shutil.rmtree(cached_site / "deep_gemm_sm120")
            self.assertTrue((cached_site.parent / ".copy_complete").exists())
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertEqual(repaired_site, cached_site)
        self.assertEqual(
            (repaired_site / "deep_gemm" / "__init__.py").stat().st_ino, init_ino
        )
        self.assertEqual(
            (repaired_site.parent / ".copy_complete").stat().st_ino, marker_ino
        )
        self.assertEqual(
            (
                repaired_site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
            ).read_text(),
            "#pragma once\n",
        )

    def test_deep_gemm_repairs_dangling_symlink_in_sm120_companion_dir(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            init_ino = (cached_site / "deep_gemm" / "__init__.py").stat().st_ino
            header = cached_site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
            header.unlink()
            os.symlink("/nonexistent/header.cuh", header)
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertEqual(repaired_site, cached_site)
        self.assertEqual(
            (repaired_site / "deep_gemm" / "__init__.py").stat().st_ino, init_ino
        )
        self.assertFalse(header.is_symlink())
        self.assertEqual(header.read_text(), "#pragma once\n")

    def test_deep_gemm_interrupted_companion_repair_preserves_live_package(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        copytree = setup.shutil.copytree

        def fail_companion(source, target, **kwargs):
            if Path(source).name == "deep_gemm_sm120":
                Path(target).mkdir(parents=True)
                raise OSError("interrupted repair")
            return copytree(source, target, **kwargs)

        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            init_ino = (cached_site / "deep_gemm" / "__init__.py").stat().st_ino
            marker_ino = (cached_site.parent / ".copy_complete").stat().st_ino
            setup.shutil.rmtree(cached_site / "deep_gemm_sm120")
            with patch.object(setup.shutil, "copytree", side_effect=fail_companion):
                self.assertIsNone(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertEqual(
            (cached_site / "deep_gemm" / "__init__.py").stat().st_ino, init_ino
        )
        self.assertEqual(
            (cached_site.parent / ".copy_complete").stat().st_ino, marker_ino
        )
        self.assertEqual(list(cached_site.glob(".deep_gemm_sm120.tmp.*")), [])

    def test_deep_gemm_repairs_partially_deleted_companion_dir(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            header = cached_site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
            helper = cached_site / "deep_gemm_sm120" / "shim" / "include" / "helper.cuh"
            init_ino = (cached_site / "deep_gemm" / "__init__.py").stat().st_ino
            marker_ino = (cached_site.parent / ".copy_complete").stat().st_ino
            header_ino = header.stat().st_ino
            helper.unlink()
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertEqual(repaired_site, cached_site)
        self.assertEqual(
            (repaired_site / "deep_gemm" / "__init__.py").stat().st_ino, init_ino
        )
        self.assertEqual(
            (repaired_site.parent / ".copy_complete").stat().st_ino, marker_ino
        )
        self.assertEqual(header.stat().st_ino, header_ino)
        self.assertEqual(helper.read_text(), "#define HELPER 1\n")

    def test_deep_gemm_companion_repair_keeps_live_files_readable_during_swap(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        real_replace = setup.os.replace
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            companion_dir = cached_site / "deep_gemm_sm120"
            header = companion_dir / "shim" / "include" / "header.cuh"
            helper = companion_dir / "shim" / "include" / "helper.cuh"
            header_ino = header.stat().st_ino
            helper.unlink()
            seen = []

            def monitored_replace(src, dst):
                self.assertTrue(companion_dir.is_dir())
                seen.append((header.read_text(), header.stat().st_ino))
                real_replace(src, dst)
                self.assertTrue(companion_dir.is_dir())
                seen.append((header.read_text(), header.stat().st_ino))

            with patch.object(setup.os, "replace", side_effect=monitored_replace):
                repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
        self.assertEqual(repaired_site, cached_site)
        self.assertTrue(seen)
        self.assertTrue(
            all(text == "#pragma once\n" and ino == header_ino for text, ino in seen)
        )
        self.assertEqual(helper.read_text(), "#define HELPER 1\n")

    def test_deep_gemm_repairs_missing_relative_symlink_and_nested_target_together(
        self,
    ):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        src_companion = (
            runfiles
            / "pip_gpu_cuda13_torch_deep_gemm"
            / "site-packages"
            / "deep_gemm_sm120"
        )
        impl = src_companion / "shim" / "include" / "impl.cuh"
        impl.write_text("#define IMPL 1\n")
        os.symlink("shim/include/impl.cuh", src_companion / "alias.cuh")
        os.symlink("alias.cuh", src_companion / "chain.cuh")

        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            companion = cached_site / "deep_gemm_sm120"
            header = companion / "shim" / "include" / "header.cuh"
            header_ino = header.stat().st_ino
            (companion / "alias.cuh").unlink()
            (companion / "chain.cuh").unlink()
            (companion / "shim" / "include" / "impl.cuh").unlink()
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertEqual(header.stat().st_ino, header_ino)
        self.assertTrue((companion / "alias.cuh").is_symlink())
        self.assertTrue((companion / "chain.cuh").is_symlink())
        self.assertEqual((companion / "chain.cuh").read_text(), "#define IMPL 1\n")

    def test_deep_gemm_repairs_truncated_companion_file(self):
        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            header = cached_site / "deep_gemm_sm120" / "shim" / "include" / "header.cuh"
            helper = cached_site / "deep_gemm_sm120" / "shim" / "include" / "helper.cuh"
            header_ino = header.stat().st_ino
            helper.write_text("")
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertEqual(header.stat().st_ino, header_ino)
        self.assertEqual(helper.read_text(), "#define HELPER 1\n")

    def test_deep_gemm_replaces_symlinked_companion_root_without_touching_referent(
        self,
    ):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        referent = self.root / "external_referent_root"
        (referent / "shim" / "include").mkdir(parents=True)
        sentinel = referent / "sentinel.txt"
        sentinel.write_text("keep-me\n")
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            companion_dir = cached_site / "deep_gemm_sm120"
            shutil.rmtree(companion_dir)
            os.symlink(referent, companion_dir)
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertFalse(companion_dir.is_symlink())
        self.assertTrue(companion_dir.is_dir())
        self.assertEqual(
            (companion_dir / "shim" / "include" / "header.cuh").read_text(),
            "#pragma once\n",
        )
        self.assertEqual(
            (companion_dir / "shim" / "include" / "helper.cuh").read_text(),
            "#define HELPER 1\n",
        )
        self.assertEqual(sentinel.read_text(), "keep-me\n")
        self.assertFalse((referent / "shim" / "include" / "header.cuh").exists())
        self.assertFalse((referent / "shim" / "include" / "helper.cuh").exists())

    def test_deep_gemm_replaces_symlinked_companion_subdir_without_touching_referent(
        self,
    ):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        referent = self.root / "external_referent_subdir"
        referent.mkdir()
        sentinel = referent / "sentinel.txt"
        sentinel.write_text("keep-sub\n")
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            companion_dir = cached_site / "deep_gemm_sm120"
            include_dir = companion_dir / "shim" / "include"
            shutil.rmtree(include_dir)
            os.symlink(referent, include_dir)
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertFalse(include_dir.is_symlink())
        self.assertTrue(include_dir.is_dir())
        self.assertEqual((include_dir / "header.cuh").read_text(), "#pragma once\n")
        self.assertEqual((include_dir / "helper.cuh").read_text(), "#define HELPER 1\n")
        self.assertEqual(sentinel.read_text(), "keep-sub\n")
        self.assertFalse((referent / "header.cuh").exists())
        self.assertFalse((referent / "helper.cuh").exists())

    def test_deep_gemm_replaces_symlinked_site_packages_ancestor_without_touching_referent(
        self,
    ):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        referent = self.root / "external_referent_site_packages"
        (referent / "deep_gemm").mkdir(parents=True)
        (referent / "deep_gemm" / "__init__.py").write_text("# referent\n")
        (referent / "deep_gemm_sm120" / "shim" / "include").mkdir(parents=True)
        (referent / "deep_gemm_sm120" / "shim" / "include" / "header.cuh").write_text(
            "#pragma once\n"
        )
        sentinel = referent / "sentinel.txt"
        sentinel.write_text("keep-site\n")

        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            shutil.rmtree(cached_site)
            os.symlink(referent, cached_site)
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertFalse(cached_site.is_symlink())
        self.assertTrue(cached_site.is_dir())
        include_dir = cached_site / "deep_gemm_sm120" / "shim" / "include"
        self.assertEqual((include_dir / "header.cuh").read_text(), "#pragma once\n")
        self.assertEqual((include_dir / "helper.cuh").read_text(), "#define HELPER 1\n")
        self.assertEqual(sentinel.read_text(), "keep-site\n")
        self.assertFalse(
            (referent / "deep_gemm_sm120" / "shim" / "include" / "helper.cuh").exists()
        )

    def test_deep_gemm_replaces_symlinked_generation_root_without_touching_referent(
        self,
    ):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        referent = self.root / "external_referent_generation"
        (referent / "site-packages" / "deep_gemm").mkdir(parents=True)
        (referent / "site-packages" / "deep_gemm" / "__init__.py").write_text(
            "# referent\n"
        )
        (referent / "site-packages" / "deep_gemm_sm120" / "shim" / "include").mkdir(
            parents=True
        )
        (
            referent
            / "site-packages"
            / "deep_gemm_sm120"
            / "shim"
            / "include"
            / "header.cuh"
        ).write_text("#pragma once\n")
        (referent / ".copy_complete").write_text("deep_gemm==2.8.2\n")
        sentinel = referent / "sentinel.txt"
        sentinel.write_text("keep-gen\n")

        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            target_base = cached_site.parent
            shutil.rmtree(target_base)
            os.symlink(referent, target_base)
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertFalse(target_base.is_symlink())
        self.assertTrue(target_base.is_dir())
        self.assertFalse(cached_site.is_symlink())
        self.assertTrue(cached_site.is_dir())
        include_dir = cached_site / "deep_gemm_sm120" / "shim" / "include"
        self.assertEqual((include_dir / "header.cuh").read_text(), "#pragma once\n")
        self.assertEqual((include_dir / "helper.cuh").read_text(), "#define HELPER 1\n")
        self.assertEqual(sentinel.read_text(), "keep-gen\n")
        self.assertFalse(
            (
                referent
                / "site-packages"
                / "deep_gemm_sm120"
                / "shim"
                / "include"
                / "helper.cuh"
            ).exists()
        )

    def test_deep_gemm_recovers_interrupted_generation_with_nonempty_companion_dir(
        self,
    ):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        runfiles = self._make_mock_deep_gemm_runfiles()
        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            cached_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            companion_dir = cached_site / "deep_gemm_sm120"
            (cached_site.parent / ".copy_complete").unlink()
            shutil.rmtree(cached_site / "deep_gemm")
            (companion_dir / "shim" / "include" / "helper.cuh").unlink()
            (companion_dir / "stale.tmp").write_text("leftover\n")
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        self.assertEqual(repaired_site, cached_site)
        self.assertTrue((cached_site.parent / ".copy_complete").exists())
        self.assertTrue((cached_site / "deep_gemm" / "__init__.py").exists())
        self.assertTrue(companion_dir.is_dir())
        self.assertEqual(
            (companion_dir / "shim" / "include" / "header.cuh").read_text(),
            "#pragma once\n",
        )
        self.assertEqual(
            (companion_dir / "shim" / "include" / "helper.cuh").read_text(),
            "#define HELPER 1\n",
        )

    def test_deep_gemm_materializes_external_symlinks_and_cleans_abandoned_staging(self):
        import shutil

        cache = self.root / "cache"
        cache.mkdir()
        external_dir = self.root / "external_runfiles_src"
        external_dir.mkdir()
        ext_header = external_dir / "external.cuh"
        ext_header.write_text("#define EXT_HEADER 42\n")

        runfiles = self._make_mock_deep_gemm_runfiles()
        src_companion = runfiles / "site-packages" / "deep_gemm_sm120"
        os.symlink(ext_header, src_companion / "shim" / "include" / "external.cuh")

        with patch.dict(os.environ, {"RUNFILES_DIR": str(runfiles)}):
            first_site = Path(setup.copy_package_with_lock("deep_gemm", cache))
            stale_staging = first_site / ".deep_gemm_sm120.tmp.999999"
            stale_staging.mkdir()
            (stale_staging / "leftover.cuh").write_text("// dead\n")
            shutil.rmtree(first_site / "deep_gemm_sm120")
            (first_site.parent / ".copy_complete").unlink()
            repaired_site = Path(setup.copy_package_with_lock("deep_gemm", cache))

        staged_ext = repaired_site / "deep_gemm_sm120" / "shim" / "include" / "external.cuh"
        self.assertFalse(staged_ext.is_symlink())
        self.assertFalse(stale_staging.exists())
        shutil.rmtree(external_dir)
        self.assertEqual(staged_ext.read_text(), "#define EXT_HEADER 42\n")


if __name__ == "__main__":
    unittest.main()

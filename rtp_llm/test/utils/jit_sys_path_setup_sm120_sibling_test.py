import queue
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.test.utils import jit_sys_path_setup

_VERSION = "2.8.2+522942a.cu132"
_VARIANT = "pip_gpu_cuda13_sm120_dsv4_torch_deep_gemm"
_SM120_HEADER = Path(
    "deep_gemm_sm120",
    "vendor",
    "deep_gemm",
    "include",
    "deep_gemm",
    "impls",
    "sm120_fp8_fp4_gemm_1d1d.cuh",
)
_LAYOUT_SUFFIX = jit_sys_path_setup._SIBLING_LAYOUT_KEY_SUFFIX


class JitSysPathSetupSm120SiblingTest(unittest.TestCase):
    def _make_site_packages(self, root, package, with_sibling):
        site_packages = root / _VARIANT / "site-packages"
        package_dir = site_packages / package
        (package_dir / "impls").mkdir(parents=True)
        (package_dir / "impls" / "kernel.py").write_text("# kernel\n")
        if with_sibling:
            header = site_packages / _SM120_HEADER
            header.parent.mkdir(parents=True)
            header.write_text("// header\n")
        return package_dir

    def _copy(self, package, package_dir, cache_dir):
        with mock.patch.object(
            jit_sys_path_setup,
            "get_package_info",
            return_value=(_VERSION, str(package_dir)),
        ):
            return jit_sys_path_setup.copy_package_with_lock(package, cache_dir)

    def _main_cache_key(self, package, package_dir):
        # The cache key main computes for this source, before any layout
        # versioning.
        return jit_sys_path_setup._cache_key(package, _VERSION, str(package_dir))

    def _tree_bytes(self, root):
        return {
            path.relative_to(root): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_deep_gemm_sibling_copied_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "deep_gemm", True)
            cache_dir = root / "cache"

            site_packages = Path(self._copy("deep_gemm", package_dir, cache_dir))

            self.assertTrue(
                (site_packages / "deep_gemm" / "impls" / "kernel.py").is_file()
            )
            self.assertTrue((site_packages / _SM120_HEADER).is_file())
            self.assertTrue((site_packages.parent / ".copy_complete").is_file())
            self.assertEqual(
                site_packages.parent,
                cache_dir
                / (self._main_cache_key("deep_gemm", package_dir) + _LAYOUT_SUFFIX),
            )

    def test_old_marker_only_generation_untouched_and_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "deep_gemm", True)
            cache_dir = root / "cache"

            # Generation staged by the pre-sibling copy rule: marker + package,
            # no sibling. It is published, so it must be left alone entirely.
            old_key = self._main_cache_key("deep_gemm", package_dir)
            old_base = cache_dir / old_key
            stale_package = old_base / "site-packages" / "deep_gemm"
            stale_package.mkdir(parents=True)
            (stale_package / "stale.py").write_text("# stale\n")
            (old_base / ".copy_complete").write_text("deep_gemm==x\n")
            snapshot = self._tree_bytes(old_base)

            site_packages = Path(self._copy("deep_gemm", package_dir, cache_dir))

            # The sibling layout stages under its own key; the published
            # generation is neither reused nor modified.
            self.assertEqual(
                site_packages.parent, cache_dir / (old_key + _LAYOUT_SUFFIX)
            )
            self.assertEqual(self._tree_bytes(old_base), snapshot)
            self.assertTrue((site_packages / _SM120_HEADER).is_file())
            self.assertTrue((site_packages.parent / ".copy_complete").is_file())

    def test_interrupted_copy_without_marker_is_redone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "deep_gemm", True)
            cache_dir = root / "cache"

            # A copy interrupted after the sibling directory appeared: package
            # and sibling dirs present, no completion marker. Never published,
            # so the next call must redo it under the lock, not accept it.
            layout_key = self._main_cache_key("deep_gemm", package_dir) + _LAYOUT_SUFFIX
            target_site_packages = cache_dir / layout_key / "site-packages"
            stale_package = target_site_packages / "deep_gemm"
            stale_package.mkdir(parents=True)
            (stale_package / "stale.py").write_text("# interrupted\n")
            (target_site_packages / "deep_gemm_sm120").mkdir()

            site_packages = Path(self._copy("deep_gemm", package_dir, cache_dir))

            self.assertEqual(site_packages, target_site_packages)
            self.assertFalse((stale_package / "stale.py").exists())
            self.assertTrue((site_packages / _SM120_HEADER).is_file())
            self.assertTrue((cache_dir / layout_key / ".copy_complete").is_file())

    def test_concurrent_reader_waits_for_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "deep_gemm", True)
            cache_dir = root / "cache"
            marker = (
                cache_dir
                / (self._main_cache_key("deep_gemm", package_dir) + _LAYOUT_SUFFIX)
                / ".copy_complete"
            )

            copy_started = threading.Event()
            allow_finish = threading.Event()
            self.addCleanup(allow_finish.set)
            real_copytree = shutil.copytree

            def gated_copytree(src, dst, *args, **kwargs):
                # copytree recurses into itself via the module global, so the
                # patch sees subdirectory calls too (positional args).
                result = real_copytree(src, dst, *args, **kwargs)
                if Path(dst).name == "deep_gemm_sm120":
                    # Pause mid-copy: package and sibling present, marker not
                    # yet written, writer still holding the lock.
                    copy_started.set()
                    allow_finish.wait(timeout=60)
                return result

            results = queue.Queue()

            def call():
                try:
                    results.put(
                        jit_sys_path_setup.copy_package_with_lock(
                            "deep_gemm", cache_dir
                        )
                    )
                except Exception as error:  # pragma: no cover - debugging aid
                    results.put(error)

            with mock.patch.object(
                jit_sys_path_setup,
                "get_package_info",
                return_value=(_VERSION, str(package_dir)),
            ), mock.patch.object(shutil, "copytree", gated_copytree):
                writer = threading.Thread(target=call, daemon=True)
                writer.start()
                self.assertTrue(copy_started.wait(timeout=30))

                reader = threading.Thread(target=call, daemon=True)
                reader.start()
                time.sleep(0.5)
                # The tree is still unmarked: the reader must not be handed the
                # in-progress copy; it waits on the lock until the writer
                # publishes the generation.
                self.assertTrue(results.empty())
                self.assertFalse(marker.exists())
                allow_finish.set()
                writer.join(timeout=60)
                reader.join(timeout=60)

            self.assertFalse(writer.is_alive())
            self.assertFalse(reader.is_alive())
            self.assertEqual(results.qsize(), 2)
            while not results.empty():
                outcome = results.get()
                self.assertIsInstance(outcome, str)
                self.assertTrue((Path(outcome) / _SM120_HEADER).is_file())
                self.assertTrue(marker.is_file())

    def test_deep_gemm_without_sibling_keeps_main_key_and_warm_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "deep_gemm", False)
            cache_dir = root / "cache"

            site_packages = Path(self._copy("deep_gemm", package_dir, cache_dir))
            self.assertTrue((site_packages / "deep_gemm").is_dir())
            self.assertFalse((site_packages / "deep_gemm_sm120").exists())
            # No sibling required -> same cache key as main.
            self.assertEqual(
                site_packages.parent.name,
                self._main_cache_key("deep_gemm", package_dir),
            )

            # A marker-complete cache stays warm: a second call must not
            # re-copy.
            sentinel = site_packages / "deep_gemm" / "sentinel.py"
            sentinel.write_text("# cached\n")
            self.assertEqual(
                Path(self._copy("deep_gemm", package_dir, cache_dir)), site_packages
            )
            self.assertTrue(sentinel.is_file())

    def test_non_deep_gemm_package_does_not_copy_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = self._make_site_packages(root / "src", "torch", True)

            site_packages = Path(self._copy("torch", package_dir, root / "cache"))

            self.assertTrue((site_packages / "torch").is_dir())
            self.assertFalse((site_packages / "deep_gemm_sm120").exists())
            self.assertEqual(
                site_packages.parent.name,
                self._main_cache_key("torch", package_dir),
            )


if __name__ == "__main__":
    unittest.main()

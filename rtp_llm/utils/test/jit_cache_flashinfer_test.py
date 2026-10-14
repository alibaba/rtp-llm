import os
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from filelock import FileLock

from rtp_llm.test.utils import jit_sys_path_setup as wrapper
from rtp_llm.utils import jit_cache_flashinfer as flash
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


class FlashInferCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.workspace = self.root / "managed" / "v2" / "scope" / "flashinfer"
        item = next(c for c in jit.COMPONENTS if c.name == "flashinfer")
        self.scope = jit.Scope(
            "scope",
            self.workspace.parent,
            (replace(item, local_dir=self.workspace),),
        )
        self.config = types.SimpleNamespace(
            manage_jit_cache=True, remote_jit_dir="", jit_cache_setup_timeout_s=-1
        )
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ["TEST_JIT_LOCAL_DIR"] = str(self.root / "managed")
        jit.setup_jit_cache_env.cache_clear()
        self.addCleanup(jit.setup_jit_cache_env.cache_clear)

    def op(self, name="stale", include=None, workspace=None):
        workspace = workspace or self.workspace
        path = workspace / ".cache/flashinfer/0.6.9/89/cached_ops" / name
        path.mkdir(parents=True)
        include = include or self.root / "old.runfiles/flashinfer/include"
        (path / "build.ninja").write_text(f"cflags = -I{include}\n")
        return path

    def start(self):
        with mock.patch.object(
            jit, "resolve_scope", return_value=self.scope
        ), mock.patch.object(jit, "_prepare_shared_root"):
            return jit.start_from_config(self.config)

    def test_runtime_handoff_checks_managed_workspace(self):
        stale = self.op()
        with mock.patch.object(wrapper.Path, "home", return_value=self.root / "home"):
            wrapper.setup_jit_cache(packages=[])
        self.assertTrue(stale.exists())  # wrapper checked a different workspace
        self.start()
        self.assertEqual(os.environ["FLASHINFER_WORKSPACE_BASE"], str(self.workspace))
        self.assertFalse(stale.exists())

    def test_warm_cache_checked_when_remote_unavailable(self):
        stale = self.op()
        self.config.remote_jit_dir = "/unavailable"
        with mock.patch.object(store, "resolve_remote", return_value=None):
            self.start()
        self.assertFalse(stale.exists())

    def test_restored_cache_checked_before_watching(self):
        source = self.op()
        rel = source.relative_to(self.scope.root) / "build.ninja"
        remote = store.RemoteSnapshotStore(self.root / "remote", "")
        remote.remote_root.mkdir()
        remote.publish_snapshot({rel.as_posix(): source / "build.ninja"})
        # A separate empty root simulates a cold consumer.
        cold = self.root / "cold/v2/scope"
        self.scope = jit.Scope(
            "scope",
            cold,
            (replace(self.scope.components[0], local_dir=cold / "flashinfer"),),
        )
        self.config.remote_jit_dir = str(remote.remote_root)
        restored = cold / rel

        def watch(manager):
            self.assertFalse(restored.exists())
            return True

        with mock.patch.object(
            store, "resolve_remote", return_value=remote
        ), mock.patch.object(jit.JitCacheManager, "_start_watch", watch):
            manager = self.start()
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager._restored)
        manager.stop()

    def test_disabled_management_leaves_cache_untouched(self):
        stale = self.op()
        self.config.manage_jit_cache = False
        self.start()
        self.assertTrue(stale.exists())

    def test_unmanaged_component_left_untouched(self):
        stale = self.op()
        self.scope = self.scope._replace(components=())
        self.start()
        self.assertTrue(stale.exists())

    def test_valid_cache_and_unrelated_ninja_preserved(self):
        valid = self.op("valid", self.root)
        unrelated = self.workspace / "elsewhere"
        unrelated.mkdir(parents=True)
        (unrelated / "build.ninja").write_text("cflags = -I/missing\n")
        flash.evict_dangling_flashinfer_ops(self.workspace)
        self.assertTrue(valid.exists())
        self.assertTrue(unrelated.exists())

    def test_runfiles_rejected_even_while_source_exists(self):
        include = self.root / "live.runfiles/include"
        include.mkdir(parents=True)
        stale = self.op(include=include)
        flash.evict_dangling_flashinfer_ops(self.workspace)
        self.assertFalse(stale.exists())

    def test_missing_include_and_dangling_src_rejected(self):
        for scenario in ("include", "src"):
            with self.subTest(scenario=scenario):
                path = self.op(scenario, self.root)
                if scenario == "src":
                    (path / "src").symlink_to(self.root / "gone")
                else:
                    (path / "build.ninja").write_text(
                        f"flags = -isystem {self.root}/gone\n"
                    )
                flash.evict_dangling_flashinfer_ops(self.workspace)
                self.assertFalse(path.exists())

    def test_busy_build_lock_is_not_removed(self):
        stale = self.op()
        lock_path = stale.parent / "tmp" / f"{stale.name}.lock"
        lock_path.parent.mkdir()
        with FileLock(str(lock_path)):
            with mock.patch.object(
                flash,
                "FileLock",
                side_effect=lambda path, timeout: FileLock(path, timeout=0),
            ):
                flash.evict_dangling_flashinfer_ops(self.workspace)
            self.assertTrue(stale.exists())
            self.assertTrue(lock_path.exists())

    def test_staleness_rechecked_under_build_lock(self):
        stale = self.op()

        class RefreshLock:
            def __enter__(self_):
                (stale / "build.ninja").write_text(f"flags = -I{self.root}\n")

            def __exit__(self_, *args):
                pass

        with mock.patch.object(flash, "FileLock", return_value=RefreshLock()):
            flash.evict_dangling_flashinfer_ops(self.workspace)
        self.assertTrue(stale.exists())


if __name__ == "__main__":
    unittest.main()

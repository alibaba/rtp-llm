import concurrent.futures
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rtp_llm import start_backend_server as backend
from rtp_llm.model_loader.tipc import ffi as tipc_ffi
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


def _fake_probes(hip=None, arch="sm_90", pkg="1_0", toolkit="nvcc-test"):
    # Stub device/compiler probes so scope resolution is hermetic.
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch("torch.version.hip", hip))
    stack.enter_context(mock.patch("torch.version.cuda", None if hip else "12.8"))
    probes = {
        "_accelerator_scope": f"{'rocm' if hip else 'cuda'}-{hip or '12.8'}-{arch}",
        "_toolkit_scope": toolkit,
        "_cpp_runtime_scope": "cxx-test",
    }
    for name, value in probes.items():
        stack.enter_context(mock.patch.object(jit, name, return_value=value))
    resolver = pkg if callable(pkg) else lambda _name: pkg
    stack.enter_context(
        mock.patch.object(jit.importlib.metadata, "version", side_effect=resolver)
    )
    return stack


def _fake_fuser(umount=None):
    module = types.ModuleType("rtp_llm.utils.fuser")
    module.umount_file = umount or mock.Mock()
    module.MountRwMode = mock.Mock()
    module.fetch_remote_file_to_local = mock.Mock()
    return mock.patch.dict(sys.modules, {"rtp_llm.utils.fuser": module})


def contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def snapshots(snap_store):
    return sorted(snap_store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))


class JitCacheTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        old_umask = os.umask(0)
        os.umask(old_umask)
        self.addCleanup(os.umask, old_umask)
        self.old_env = os.environ.copy()
        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        from rtp_llm.utils.util import COMPILE_FLAG_ENVS

        for name in COMPILE_FLAG_ENVS:
            os.environ.pop(name, None)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old_env)))
        self._seq = 0

    def make_store(self, name="remote"):
        remote = self.root / name
        remote.mkdir(parents=True, exist_ok=True)
        return store.RemoteSnapshotStore(remote)

    def publish(self, snap_store, files: dict[str, bytes]):
        generation = {}
        for rel, data in files.items():
            self._seq += 1
            source = self.root / f"src_{self._seq}"
            source.write_bytes(data)
            generation[rel] = source
        return snap_store.publish_snapshot(generation)

    def restore(self, snap_store, target: Path) -> bool:
        # The manager's restore recipe: lock, skip non-empty, prepare, commit.
        fd = store.acquire_flock(target.with_name(f"{target.name}.lock"))
        try:
            if store.scope_root_usable(target):
                return False
            staging = snap_store.prepare_restore(
                target.with_name(f"{target.name}.staging")
            )
            if staging is None:
                return False
            return store.commit_restore(staging.staging, target)
        finally:
            os.close(fd)


class StoreTest(JitCacheTestBase):
    def test_publish_keeps_immutable_generations(self):
        snap_store, expected = self.make_store(), {}
        remote = snap_store.remote_root
        for index in range(3):
            expected[f"triton/hash/k-{index}.cubin"] = f"v{index}".encode()
            self.publish(snap_store, expected)
        snapshots = list(remote.glob(f"*{store.SNAPSHOT_SUFFIX}"))
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(f"-{os.uname().nodename}" in p.name for p in snapshots))
        target = self.root / "target"
        self.assertTrue(self.restore(snap_store, target))
        self.assertEqual(contents(target), expected)

    def test_publish_keeps_latest_snapshots_and_reaps_stale_tmp(self):
        snap_store = self.make_store()
        stale_tmp = snap_store.remote_root / f"0-x{store.SNAPSHOT_SUFFIX}.tmp"
        fresh_tmp = snap_store.remote_root / f"1-y{store.SNAPSHOT_SUFFIX}.tmp"
        stale_tmp.touch()
        fresh_tmp.touch()
        old = time.time() - store.STALE_REMOTE_TMP_S - 1
        os.utime(stale_tmp, (old, old))
        with mock.patch.object(store, "SNAPSHOT_KEEP", 2):
            for index in range(3):
                self.publish(snap_store, {"triton/k.cubin": f"v{index}".encode()})
        self.assertEqual(len(snapshots(snap_store)), 2)
        self.assertFalse(stale_tmp.exists())
        self.assertTrue(fresh_tmp.exists())
        restored = self.root / "restored"
        self.assertTrue(self.restore(snap_store, restored))
        self.assertEqual(contents(restored), {"triton/k.cubin": b"v2"})

    def test_reap_only_stale_existence_batons(self):
        old, old_aiter, fresh, flock_style = (
            self.root / name
            for name in ("old/lock", "build/lock_module_fmha", "new/lock", "x.lock")
        )
        for path in (old, old_aiter, fresh, flock_style):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        stale = time.time() - store.STALE_BATON_S - 1
        for path in (old, old_aiter, flock_style):
            os.utime(path, (stale, stale))
        store.reap_stale_batons(self.root)
        self.assertFalse(old.exists())
        self.assertFalse(old_aiter.exists())
        self.assertTrue(fresh.exists())
        # flock files live forever and may be held right now; age proves nothing.
        self.assertTrue(flock_style.exists())

    def test_scope_root_usable_ignores_empty_directories(self):
        root = self.root / "scope"
        (root / "triton").mkdir(parents=True)
        self.assertFalse(store.scope_root_usable(root))

    def test_scope_root_usable_fails_open_on_scan_error(self):
        with mock.patch.object(Path, "rglob", side_effect=OSError):
            self.assertFalse(store.scope_root_usable(self.root / "scope"))

    def test_extractor_rejects_unsafe_members(self):
        target = self.root / "target"
        for name, kind in (
            ("../escape", tarfile.REGTYPE),
            ("/escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE),
            ("hardlink", tarfile.LNKTYPE),
        ):
            member = tarfile.TarInfo(name)
            member.type = kind
            with self.subTest(name=name), self.assertRaises(ValueError):
                list(store._safe_members([member], target))
        with self.assertRaises(ValueError):
            store._safe_path(target, "../manifest")

    def test_extract_strips_setuid_bits(self):
        source = self.root / "suid.so"
        source.write_bytes(b"x")
        source.chmod(0o6755)
        archive = self.root / "a.tar.zst"
        store.pack_zstd_tar(archive, {"triton/suid.so": source})
        store.extract_zstd_tar(archive, self.root / "out")
        mode = (self.root / "out" / "triton" / "suid.so").stat().st_mode
        self.assertFalse(mode & 0o7000)

    def test_snapshot_without_mtime_manifest_fails_open(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/op.so": b"x"})
        with mock.patch.object(store, "MTIME_MANIFEST", ".missing"), self.assertLogs(
            level="WARNING"
        ):
            self.assertIsNone(snap_store.prepare_restore(self.root / "staging"))

    def test_corrupt_snapshots_fall_back_then_fail_open(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/old": b"old"})
        self.publish(snap_store, {"triton/old": b"old", "triton/new": b"new"})
        bad = snap_store.remote_root / f"{'9' * 20}-bad{store.SNAPSHOT_SUFFIX}"
        bad.write_bytes(b"bad")
        with self.assertLogs(level="WARNING"):
            restored = self.root / "restored"
            self.assertTrue(self.restore(snap_store, restored))
        self.assertEqual(
            contents(restored), {"triton/old": b"old", "triton/new": b"new"}
        )
        # All snapshots corrupt: prepare yields None, nothing committed or left.
        broken = self.make_store("broken")
        (broken.remote_root / f"{1:020d}{store.SNAPSHOT_SUFFIX}").write_bytes(b"x")
        cold = self.root / "cold"
        with self.assertLogs(level="WARNING"):
            self.assertFalse(self.restore(broken, cold))
        self.assertFalse(cold.exists())
        self.assertFalse(list((self.root / "cold.staging").glob("*")))

    def test_commit_never_replaces_nonempty_scope_root(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/keep/op.so": b"remote"})
        warm = self.root / "warm"
        staging = snap_store.prepare_restore(self.root / "staging")
        # A peer builder warms the tree between prepare and commit.
        (warm / "triton").mkdir(parents=True)
        (warm / "triton/op.so").write_bytes(b"local")
        self.assertFalse(store.commit_restore(staging.staging, warm))
        self.assertEqual(contents(warm), {"triton/op.so": b"local"})
        self.assertFalse(staging.staging.exists())

    def test_concurrent_publishers_write_independent_snapshots(self):
        snap_store = self.make_store()
        barrier = threading.Barrier(2)
        real_copyfile = shutil.copyfile

        def interleaved(src, dst, *args, **kwargs):
            barrier.wait(timeout=10)
            return real_copyfile(src, dst, *args, **kwargs)

        one, two = self.root / "one", self.root / "two"
        one.write_bytes(b"one")
        two.write_bytes(b"two")
        # One store per publisher, as in production: each owns its own mount lease.
        peers = (snap_store, self.make_store())
        with mock.patch.object(
            store.shutil, "copyfile", side_effect=interleaved
        ), concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(peer.publish_snapshot, {"triton/x": path})
                for peer, path in zip(peers, (one, two))
            ]
            for future in futures:
                future.result()
        self.assertEqual(
            len(list(snap_store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))), 2
        )
        restored = self.root / "restored"
        self.assertTrue(self.restore(snap_store, restored))
        self.assertIn(contents(restored), ({"triton/x": b"one"}, {"triton/x": b"two"}))

    def test_shared_inode_packs_as_full_content(self):
        dup_a, dup_b = self.root / "a.cubin", self.root / "b.cubin"
        dup_a.write_bytes(b"dup")
        os.link(dup_a, dup_b)
        snap_store = self.make_store()
        snap_store.publish_snapshot({"triton/a.cubin": dup_a, "triton/b.cubin": dup_b})
        target = self.root / "restored"
        self.assertTrue(self.restore(snap_store, target))
        self.assertEqual(
            contents(target), {"triton/a.cubin": b"dup", "triton/b.cubin": b"dup"}
        )

    def test_new_generation_replaces_source_and_binary_together(self):
        snap_store = self.make_store()
        self.publish(
            snap_store,
            {"triton/u/op.cu": b"v1", "triton/u/op.so": b"v1", "triton/u/old.o": b"x"},
        )
        self.publish(snap_store, {"triton/u/op.cu": b"v2", "triton/u/op.so": b"v2"})
        restored = self.root / "restored"
        self.assertTrue(self.restore(snap_store, restored))
        self.assertEqual(
            contents(restored), {"triton/u/op.cu": b"v2", "triton/u/op.so": b"v2"}
        )

    def test_publish_defers_when_builder_writes_during_pack(self):
        snap_store = self.make_store()
        source = self.root / "op.so"
        source.write_bytes(b"partial")
        # Signature seen before add, then a moved signature on the verify pass:
        # a builder rewrote the artifact mid-pack, so the generation must defer.
        sigs = iter([(1, 2, 7, 100), (1, 2, 42, 200)])
        with mock.patch.object(store, "_file_sig", side_effect=lambda _: next(sigs)):
            with self.assertRaises(store.SnapshotRaced):
                snap_store.publish_snapshot({"triton/op.so": source})
        self.assertEqual(snapshots(snap_store), [])
        self.assertEqual(
            list(snap_store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}.tmp")), []
        )

    def test_unmount_waits_for_in_flight_publish(self):
        snap_store = self.make_store()
        snap_store._mounted = "/mnt/fake"
        source = self.root / "op.so"
        source.write_bytes(b"payload")
        released, packing, release = [], threading.Event(), threading.Event()
        self.addCleanup(release.set)
        real_pack = store.pack_zstd_tar

        def pack(archive, files):
            packing.set()
            self.assertTrue(release.wait(5))
            real_pack(archive, files)

        with _fake_fuser(umount=released.append), mock.patch.object(
            store, "pack_zstd_tar", side_effect=pack
        ):
            publisher = threading.Thread(
                target=snap_store.publish_snapshot, args=({"triton/op.so": source},)
            )
            publisher.start()
            self.assertTrue(packing.wait(5))
            closer = threading.Thread(target=snap_store.close)
            closer.start()
            closer.join(0.3)
            self.assertTrue(closer.is_alive())  # the live copy holds the mount
            self.assertEqual(released, [])
            release.set()
            for worker in (publisher, closer):
                worker.join(5)
        self.assertEqual(released, ["/mnt/fake"])
        self.assertEqual(len(snapshots(snap_store)), 1)
        self.publish(snap_store, {"triton/late.cubin": b"late"})  # dropped once closed
        self.assertEqual(len(snapshots(snap_store)), 1)

    def test_restore_preserves_nanosecond_mtime(self):
        source = self.root / "op.so"
        source.write_bytes(b"payload")
        mtime_ns = 1_700_000_000_123_456_789
        os.utime(source, ns=(mtime_ns, mtime_ns))
        snap_store = self.make_store()
        snap_store.publish_snapshot({"tree/op.so": source})
        restored = self.root / "restored"
        self.assertTrue(self.restore(snap_store, restored))
        self.assertEqual((restored / "tree/op.so").stat().st_mtime_ns, mtime_ns)


class ScopeTest(JitCacheTestBase):
    def resolve(self, **overrides):
        with _fake_probes(**overrides):
            scope = jit.resolve_scope(self.root)
        self.assertIsNotNone(scope)
        return scope

    def scope_id(self, **overrides):
        return self.resolve(**overrides).scope_id

    def test_scope_id_covers_every_environment_axis(self):
        base = self.scope_id()
        self.assertEqual(base, self.scope_id())  # deterministic
        self.assertNotEqual(base, self.scope_id(arch="sm_80"))
        self.assertNotEqual(base, self.scope_id(toolkit="nvcc-new"))
        with mock.patch("torch.__version__", "0.0.0+scope"):
            self.assertNotEqual(base, self.scope_id())
        with _fake_probes(pkg="2_0"):
            bumped = jit.resolve_scope(self.root).scope_id
        self.assertNotEqual(base, bumped)  # any managed package version moves it
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "preset")
        self.assertNotEqual(base, self.scope_id())  # presets change the identity
        os.environ.pop("TRITON_CACHE_DIR")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        self.assertNotEqual(base, self.scope_id())  # compile flags change it too

        missing = mock.Mock(side_effect=jit.importlib.metadata.PackageNotFoundError)
        with _fake_probes(pkg=missing):
            triton_only = jit.resolve_scope(self.root)
        self.assertEqual([x.name for x in triton_only.components], ["triton"])
        with _fake_probes(pkg=missing, arch="sm_80"):
            self.assertNotEqual(
                triton_only.scope_id, jit.resolve_scope(self.root).scope_id
            )
        with mock.patch("torch.__version__", "0.0.0+scope"), _fake_probes(pkg=missing):
            self.assertNotEqual(
                triton_only.scope_id, jit.resolve_scope(self.root).scope_id
            )

    def test_torch_extensions_scope_tracks_rtp_kernel_build(self):
        def with_kernel(version):
            def resolver(name):
                if name != "rtp_kernel":
                    return "1_0"
                if version is None:
                    raise jit.importlib.metadata.PackageNotFoundError(name)
                return version

            return resolver

        # A different source fingerprint must never map to the same snapshot scope.
        self.assertNotEqual(
            self.scope_id(pkg=with_kernel("0.1.0+aaa")),
            self.scope_id(pkg=with_kernel("0.1.0+bbb")),
        )
        names = {c.name for c in self.resolve(pkg=with_kernel(None)).components}
        self.assertNotIn("torch_extensions", names)  # no identity -> not managed
        self.assertIn("triton", names)

    def test_rocm_scope_and_component_selection(self):
        with _fake_probes(hip="6.2.41133", arch="gfx942"):
            scope = jit.resolve_scope(self.root)
        names = {item.name for item in scope.components}
        self.assertLessEqual({"aiter", "flydsl", "triton"}, names)
        self.assertFalse(names & {"flashinfer", "deep_gemm", "tvm_ffi", "cute_dsl"})

    def test_setup_env_redirects_and_respects_presets(self):
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "preset")
        with _fake_probes(), mock.patch.object(
            jit, "LOCAL_JIT_ROOT", self.root / ".jit_cache"
        ):
            scope = jit.setup_jit_cache_env()
        self.assertNotIn("triton", {item.name for item in scope.components})
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(self.root / "preset"))
        torch_ext = next(x for x in scope.components if x.name == "torch_extensions")
        self.assertEqual(os.environ["TORCH_EXTENSIONS_DIR"], str(torch_ext.local_dir))
        self.assertIn(scope.scope_id, os.environ["TORCH_EXTENSIONS_DIR"])

    def test_env_override_relocates_local_root(self):
        override = self.root / "isolated" / ".jit_cache"
        os.environ["RTP_JIT_LOCAL_ROOT"] = str(override)
        with _fake_probes():
            scope = jit.setup_jit_cache_env()
        self.assertEqual(scope.root.parents[1], override)
        torch_ext = next(x for x in scope.components if x.name == "torch_extensions")
        self.assertTrue(str(torch_ext.local_dir).startswith(str(override)))

    def test_local_root_is_shared(self):
        owned = self.root / "owned"
        old_file = owned / ".jit_cache" / "v1" / "scope" / "old.so"
        old_file.parent.mkdir(parents=True)
        old_file.write_bytes(b"old")
        os.chmod(owned, 0o700)
        os.chmod(old_file.parent, 0o700)
        os.chmod(old_file, 0o600)
        with mock.patch.object(
            jit, "LOCAL_JIT_ROOT", owned / ".jit_cache"
        ), mock.patch.object(
            jit, "resolve_scope", return_value=None
        ), mock.patch.object(
            jit.os, "umask"
        ) as umask:
            self.assertIsNone(jit.setup_jit_cache_env())
        umask.assert_called_once_with(0)
        # Both roots stay sticky; inside the tree commit_restore still needs rename.
        self.assertEqual(os.stat(owned).st_mode & 0o7777, 0o1777)
        self.assertEqual(os.stat(owned / ".jit_cache").st_mode & 0o7777, 0o1777)
        self.assertEqual(os.stat(old_file.parent).st_mode & 0o7777, 0o777)
        self.assertEqual(os.stat(old_file).st_mode & 0o777, 0o666)

    def test_symlinked_local_root_is_refused(self):
        target = self.root / "target"
        target.mkdir(mode=0o700)
        linked = self.root / "linked"
        linked.symlink_to(target, target_is_directory=True)
        os.environ["RTP_JIT_LOCAL_ROOT"] = str(linked / ".jit_cache")
        with self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(jit.setup_jit_cache_env())
        self.assertIn("untrusted JIT root", "\n".join(logs.output))
        self.assertEqual(os.stat(target).st_mode & 0o777, 0o700)  # never widened
        self.assertFalse((target / ".jit_cache").exists())

    def test_component_suffix_rules(self):
        triton = next(x for x in jit.COMPONENTS if x.name == "triton")
        self.assertTrue(triton.should_sync("hash/kernel.cubin"))
        self.assertTrue(triton.should_sync("hash/kernel.cubin", "closed"))
        self.assertFalse(triton.should_sync("hash/t.autotune.json"))
        self.assertFalse(triton.should_sync("hash/t.autotune.json", "closed"))
        self.assertFalse(triton.should_sync("../escape.cubin"))
        self.assertFalse(triton.should_sync("tmp/partial.cubin"))
        self.assertFalse(triton.should_sync("hash/readme.txt"))

    def test_probe_failure_fails_open_and_is_not_cached(self):
        outputs = [
            subprocess.TimeoutExpired(cmd="c++", timeout=5),
            "g++ (test) 13.0\n",
            "/usr/lib/libstdc++.so.6\n",
        ]
        with mock.patch.object(
            jit.subprocess, "check_output", side_effect=outputs
        ), mock.patch.object(jit.Path, "is_file", return_value=True), mock.patch.object(
            jit.Path, "read_bytes", return_value=b"solib"
        ):
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(jit._cpp_runtime_scope())
            self.assertIsNotNone(jit._cpp_runtime_scope())

    def test_gpu_probe_runs_only_in_subprocess(self):
        with mock.patch.object(
            jit.subprocess, "check_output", return_value="sm_90\n"
        ) as probe:
            self.assertEqual(jit._accelerator_scope("cuda", "12.8"), "cuda-12.8-sm_90")
        self.assertEqual(probe.call_args.args[0][:2], [sys.executable, "-c"])

    def test_toolkit_scope_uses_actual_compiler_version(self):
        with mock.patch.object(jit, "shutil") as shutil_module, mock.patch.object(
            jit.subprocess, "check_output", return_value="nvcc 12.9\n"
        ) as probe:
            shutil_module.which.return_value = "/toolkit/bin/nvcc"
            with mock.patch("torch.utils.cpp_extension.CUDA_HOME", None), mock.patch(
                "torch.utils.cpp_extension.ROCM_HOME", None
            ):
                scope = jit._toolkit_scope(jit.CUDA)
        self.assertEqual(scope, "nvcc-nvcc 12.9")
        self.assertEqual(probe.call_args.args[0], ["/toolkit/bin/nvcc", "--version"])


class ManagerTest(JitCacheTestBase):
    def setUp(self):
        super().setUp()
        self.managers = []
        self.addCleanup(lambda: [manager.stop() for manager in self.managers])

    def make_scope(self):
        with _fake_probes(), mock.patch.object(
            jit, "LOCAL_JIT_ROOT", self.root / "local"
        ):
            return jit.setup_jit_cache_env()

    def make_manager(self, scope, remote="remote"):
        (self.root / remote).mkdir(parents=True, exist_ok=True)
        manager = jit.JitCacheManager(scope, str(self.root / remote))
        self.managers.append(manager)
        return manager

    def write_artifact(self, scope, rel, data=b"payload"):
        """Emit a product into the scope tree, like a builder would."""
        path = scope.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def publish_remote(self, manager, scope, rel, data):
        """Publish one member straight to the scope's remote, as a peer host would."""
        remote = Path(manager._remote_value) / jit.RTP_JIT_VERSION / scope.scope_id
        remote.mkdir(parents=True, exist_ok=True)
        source = self.root / Path(rel).name
        source.write_bytes(data)
        store.RemoteSnapshotStore(remote).publish_snapshot({rel: source})

    def test_bootstrap_restores_before_ranks_then_publishes(self):
        scope = self.make_scope()
        producer = self.make_manager(scope)
        self.assertTrue(producer.bootstrap(timeout_s=30))
        artifact = self.write_artifact(scope, "triton/hash/kernel.cubin")
        producer._dirty.set()
        producer.publish_pending_snapshot()
        self.assertTrue(snapshots(producer.store))
        producer.stop()

        # A fresh host (empty scope root) restores that generation up front.
        shutil.rmtree(scope.root)
        consumer = self.make_manager(scope)
        self.assertTrue(consumer.bootstrap(timeout_s=30))
        self.assertEqual(artifact.read_bytes(), b"payload")

    def test_publish_defers_and_rearms_dirty_on_pack_race(self):
        scope = self.make_scope()
        producer = self.make_manager(scope)
        self.assertTrue(producer.bootstrap(timeout_s=30))
        self.write_artifact(scope, "triton/hash/kernel.cubin")
        producer._dirty.set()
        with mock.patch.object(
            producer.store, "publish_snapshot", side_effect=store.SnapshotRaced("race")
        ):
            producer.publish_pending_snapshot()
        self.assertTrue(producer._dirty.is_set())
        self.assertEqual(snapshots(producer.store), [])
        producer.stop()

    def test_scan_failure_rearms_dirty_and_frees_publish_lock(self):
        scope = self.make_scope()
        producer = self.make_manager(scope)
        self.assertTrue(producer.bootstrap(timeout_s=30))
        self.write_artifact(scope, "triton/kernel.cubin")
        producer._dirty.set()
        with mock.patch.object(
            producer, "_snapshot_files", side_effect=OSError("scan blew up")
        ), self.assertRaises(OSError):
            producer.publish_pending_snapshot()
        # Neither the pending work nor the lock may be lost, or this scope can
        # never publish again for the lifetime of the process.
        self.assertTrue(producer._dirty.is_set())
        producer.publish_pending_snapshot()
        self.assertTrue(snapshots(producer.store))

    def test_empty_run_does_not_block_later_restore(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        self.assertTrue(manager.bootstrap(timeout_s=30))
        self.assertTrue(scope.root.exists())
        self.assertFalse(store.scope_root_usable(scope.root))
        self.assertTrue(manager._dirty.is_set())
        manager.stop()

        self.publish_remote(manager, scope, "triton/fresh.cubin", b"fresh")
        consumer = self.make_manager(scope)
        self.assertTrue(consumer.bootstrap(timeout_s=30))
        self.assertEqual((scope.root / "triton" / "fresh.cubin").read_bytes(), b"fresh")

    def test_live_peer_empty_tree_swap_still_backfills(self):
        # A peer between _start_watch and its first artifact owns only empty dirs,
        # so a later restore may swap the tree out from under it. That costs the
        # peer its inotify watches, not correctness: it writes by path, and its next
        # full scan packs both its own artifact and the restored one.
        scope = self.make_scope()
        peer = self.make_manager(scope)
        self.assertTrue(peer.bootstrap(timeout_s=30))
        self.assertFalse(store.scope_root_usable(scope.root))

        self.publish_remote(peer, scope, "triton/third_host.cubin", b"third")
        consumer = self.make_manager(scope)
        self.assertTrue(consumer.bootstrap(timeout_s=30))
        self.assertEqual(
            (scope.root / "triton" / "third_host.cubin").read_bytes(), b"third"
        )

        self.write_artifact(scope, "triton/peer.cubin", b"peer")
        peer._dirty.set()  # stands in for the watcher event of that write
        peer.publish_pending_snapshot()
        peek = self.root / "peek"
        peek.mkdir()
        store.extract_zstd_tar(snapshots(peer.store)[-1], peek)
        self.assertEqual(
            contents(peek),
            {"triton/third_host.cubin": b"third", "triton/peer.cubin": b"peer"},
        )

    def test_bootstrap_skips_restore_on_peer_warm_tree(self):
        scope = self.make_scope()
        peer_file = scope.root / "triton" / "peer.cubin"
        peer_file.parent.mkdir(parents=True, exist_ok=True)
        peer_file.write_bytes(b"peer")
        manager = self.make_manager(scope)
        with mock.patch.object(
            store, "extract_zstd_tar", side_effect=AssertionError("re-extracted")
        ):
            self.assertTrue(manager.bootstrap(timeout_s=30))
        self.assertEqual(peer_file.read_bytes(), b"peer")

    def test_warm_tree_seeds_empty_remote_and_restart_stays_quiet(self):
        scope = self.make_scope()
        seeder = self.make_manager(scope)
        self.write_artifact(scope, "triton/warm.cubin", b"warm")
        self.assertTrue(seeder.bootstrap(timeout_s=30))
        self.assertTrue(seeder._dirty.is_set())  # empty remote gets seeded once
        seeder.publish_pending_snapshot()
        self.assertEqual(len(snapshots(seeder.store)), 1)
        seeder.stop()

        restart = self.make_manager(scope)
        self.assertTrue(restart.bootstrap(timeout_s=30))
        self.assertFalse(restart._dirty.is_set())  # published remote: no re-upload
        restart.publish_pending_snapshot()
        self.assertEqual(len(snapshots(restart.store)), 1)

        # A later build still publishes the complete current tree.
        self.write_artifact(scope, "triton/fresh.cubin", b"fresh")
        restart._dirty.set()  # stands in for the watcher event of that write
        restart.publish_pending_snapshot()
        peek = self.root / "peek"
        peek.mkdir()
        store.extract_zstd_tar(snapshots(restart.store)[-1], peek)
        self.assertEqual(
            contents(peek),
            {"triton/warm.cubin": b"warm", "triton/fresh.cubin": b"fresh"},
        )

    def test_bootstrap_timeout_abandons_and_late_worker_cleans_up(self):
        scope = self.make_scope()
        release = threading.Event()
        self.addCleanup(release.set)
        real_prepare = store.RemoteSnapshotStore.prepare_restore

        def slow_prepare(self_store, staging_root):
            release.wait(10)
            return real_prepare(self_store, staging_root)

        manager = self.make_manager(scope)
        self.publish_remote(manager, scope, "triton/k.cubin", b"fresh")
        with mock.patch.object(
            store.RemoteSnapshotStore, "prepare_restore", slow_prepare
        ), self.assertLogs(level="WARNING"):
            self.assertFalse(manager.bootstrap(timeout_s=1))
        self.write_artifact(scope, "triton/local.cubin", b"local")
        release.set()
        manager._prepare_thread.join(timeout=10)
        self.assertFalse(manager._prepare_thread.is_alive())
        retry = self.make_manager(scope)
        self.assertTrue(retry.bootstrap(timeout_s=1))
        retry._dirty.set()
        retry.publish_pending_snapshot()
        self.assertEqual(len(snapshots(retry.store)), 2)
        # The timed-out worker cleans staging and never overwrites the live tree.
        self.assertEqual((scope.root / "triton" / "local.cubin").read_bytes(), b"local")
        staging = scope.root.parent / ".staging" / scope.scope_id
        self.assertFalse(list(staging.glob("*")))

    def test_bootstrap_timeout_discards_result_delivered_at_boundary(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        staging = self.root / "prepared"
        staging.mkdir()
        delivered, discarded = threading.Event(), threading.Event()
        released = []
        pending_store = store.RemoteSnapshotStore(self.root, "/mnt/fake")

        def prepare(_):
            manager._prepared.put(
                (pending_store, store.Restored(staging, mock.Mock()), None)
            )
            delivered.set()

        def boundary_get(*args, **kwargs):
            if kwargs.get("timeout") is not None:
                self.assertTrue(delivered.wait(1))
                raise jit.queue.Empty
            return jit.queue.Queue.get(manager._prepared, *args, **kwargs)

        def unmount(path):
            released.append(path)
            discarded.set()

        with mock.patch.object(
            manager, "_prepare", side_effect=prepare
        ), mock.patch.object(
            manager._prepared, "get", side_effect=boundary_get
        ), _fake_fuser(
            umount=unmount
        ), self.assertLogs(
            level="WARNING"
        ):
            self.assertFalse(manager.bootstrap(timeout_s=1))
            manager._prepare_thread.join(timeout=1)
            self.assertTrue(discarded.wait(1))
        self.assertFalse(staging.exists())
        self.assertEqual(released, ["/mnt/fake"])

    def test_event_handler_marks_dirty_only_for_trigger_files(self):
        names = ("triton", "flydsl", "cute_dsl", "tilelang")
        components = tuple(
            replace(
                next(x for x in jit.COMPONENTS if x.name == name),
                local_dir=self.root / name,
            )
            for name in names
        )
        scope = jit.Scope("test", self.root, components)
        manager = self.make_manager(scope)
        triton = next(x for x in scope.components if x.name == "triton")
        quiet = triton.local_dir / "hash" / "t.autotune.json"
        loud = triton.local_dir / "hash" / "k.cubin"
        quiet.parent.mkdir(parents=True, exist_ok=True)
        for path, expect in ((quiet, False), (loud, True)):
            manager._dirty.clear()
            path.write_bytes(b"x")
            manager.on_any_event(
                mock.Mock(event_type="closed", src_path=str(path), is_directory=False)
            )
            self.assertEqual(manager._dirty.is_set(), expect, path.name)

        for name, suffix in (
            ("flydsl", ".pkl"),
            ("cute_dsl", ".mlir"),
            ("tilelang", ".so"),
        ):
            manager._dirty.clear()
            item = next(x for x in components if x.name == name)
            dest = item.local_dir / f"kernel{suffix}"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"x")
            manager.on_any_event(
                mock.Mock(
                    event_type="moved",
                    src_path=str(self.root / "ignored.tmp"),
                    dest_path=str(dest),
                    is_directory=False,
                )
            )
            self.assertTrue(manager._dirty.is_set(), name)

    def test_restore_waits_before_mounting_remote(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        lock = scope.root.parent / ".locks" / f"{scope.scope_id}.restore.lock"
        restore_fd = store.acquire_flock(lock)
        result = []
        real_resolve = store.resolve_remote
        with mock.patch.object(store, "resolve_remote", wraps=real_resolve) as resolve:
            worker = threading.Thread(
                target=lambda: result.append(manager.bootstrap(5))
            )
            worker.start()
            time.sleep(0.1)
            self.assertFalse(resolve.called)
            os.close(restore_fd)
            worker.join(5)
        self.assertEqual(result, [True])

    def test_restore_lock_wait_honors_setup_timeout(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        lock = scope.root.parent / ".locks" / f"{scope.scope_id}.restore.lock"
        restore_fd = store.acquire_flock(lock)
        try:
            with mock.patch.object(store, "resolve_remote") as resolve, self.assertLogs(
                level="WARNING"
            ):
                self.assertFalse(manager.bootstrap(0.05))
            resolve.assert_not_called()
        finally:
            os.close(restore_fd)
        manager._prepare_thread.join(5)

    def test_failed_watch_start_is_stopped_during_cleanup(self):
        manager = jit.JitCacheManager(mock.Mock(components=()), "")
        observer = mock.Mock()
        observer.start.side_effect = RuntimeError("watch failed")
        with mock.patch.object(jit, "Observer", return_value=observer), self.assertLogs(
            level="ERROR"
        ):
            self.assertFalse(manager._start_watch())
        manager.stop()
        observer.stop.assert_called_once_with()
        observer.join.assert_called_once_with()

    def test_watch_scopes_each_component(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        observer = mock.Mock()
        with mock.patch.object(jit, "Observer", return_value=observer):
            self.assertTrue(manager._start_watch())
        self.assertEqual(
            observer.schedule.call_args_list,
            [
                mock.call(manager, str(item.local_dir), recursive=True)
                for item in scope.components
            ],
        )
        self.assertTrue(all(item.local_dir.is_dir() for item in scope.components))

    def test_second_stop_retries_failed_unmount(self):
        manager = jit.JitCacheManager(mock.Mock(), "")
        manager.store = store.RemoteSnapshotStore(self.root, "/mnt/fake")
        attempts = []

        def unmount(path):
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError("busy")

        with _fake_fuser(umount=unmount):
            with self.assertLogs(level="WARNING"):
                manager.stop()
            manager.stop()
        self.assertEqual(attempts, ["/mnt/fake", "/mnt/fake"])

    def test_stop_timeout_starts_daemon_unmount(self):
        manager = jit.JitCacheManager(mock.Mock(), "")
        manager.store = store.RemoteSnapshotStore(self.root, "/mnt/fake")
        publishing, release, unmounted = (threading.Event() for _ in range(3))
        self.addCleanup(release.set)

        def publish():
            publishing.set()
            release.wait(5)

        manager._dirty.set()
        with mock.patch.object(
            manager, "publish_pending_snapshot", side_effect=publish
        ), _fake_fuser(umount=lambda _: unmounted.set()), mock.patch.object(
            jit, "STOP_TIMEOUT_S", 0.05
        ), self.assertLogs(
            level="WARNING"
        ):
            manager._worker = threading.Thread(target=manager._sync_loop)
            manager._worker.start()
            manager.stop()
            self.assertTrue(publishing.wait(1))
            self.assertTrue(unmounted.wait(1))
            release.set()

    def test_mount_owned_and_released_exactly_once(self):
        released = []
        with _fake_fuser(umount=released.append):
            manager = jit.JitCacheManager(mock.Mock(), "")
            manager.store = store.RemoteSnapshotStore(self.root, "/mnt/fake")
            stops = [threading.Thread(target=manager.stop) for _ in range(2)]
            for stop in stops:
                stop.start()
            for stop in stops:
                stop.join(1)
            manager.stop()
        self.assertEqual(released, ["/mnt/fake"])
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(
                store.resolve_remote(
                    str(self.root / "missing"), jit.RTP_JIT_VERSION, "scope"
                )
            )
        snap_store = store.resolve_remote(str(self.root), jit.RTP_JIT_VERSION, "scope")
        self.assertEqual(
            snap_store.remote_root,
            self.root / jit.RTP_JIT_VERSION / "scope",
        )

    def test_mount_failure_paths_release_exactly_once(self):
        released = []
        with _fake_fuser(umount=released.append):
            fuser = sys.modules["rtp_llm.utils.fuser"]
            fuser.fetch_remote_file_to_local = mock.Mock(return_value="/mnt/fake")
            # Mounted, but the mounted path is unusable: lease must close.
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(
                    store.resolve_remote("oss://bucket/x", jit.RTP_JIT_VERSION, "scope")
                )
            self.assertEqual(released, ["/mnt/fake"])
            fuser.fetch_remote_file_to_local = mock.Mock(side_effect=RuntimeError)
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(
                    store.resolve_remote("oss://bucket/x", jit.RTP_JIT_VERSION, "scope")
                )
            self.assertEqual(released, ["/mnt/fake"])  # nothing new mounted

    def test_each_manager_can_publish(self):
        scope = self.make_scope()
        managers = (self.make_manager(scope), self.make_manager(scope))
        self.assertTrue(all(manager.bootstrap(30) for manager in managers))
        artifact = scope.root / "triton" / "kernel.cubin"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"payload")
        publish_lock = scope.root.parent / ".locks" / f"{scope.scope_id}.publish.lock"
        publisher_fd = store.acquire_flock(publish_lock)
        managers[0]._dirty.set()
        managers[0].publish_pending_snapshot()
        self.assertTrue(managers[0]._dirty.is_set())
        self.assertFalse(snapshots(managers[0].store))
        os.close(publisher_fd)
        for manager in managers:
            manager._dirty.set()
            manager.publish_pending_snapshot()
        self.assertEqual(len(snapshots(managers[0].store)), 2)
        self.assertTrue(all(manager._worker for manager in managers))


class BackendTest(JitCacheTestBase):
    def make_configs(self, remote=""):
        configs = mock.Mock()
        configs.jit_config.remote_jit_dir = remote
        configs.jit_config.jit_cache_setup_timeout_s = 5
        return configs

    def test_local_only_redirects_env_without_manager(self):
        with _fake_probes(), mock.patch.object(
            jit, "LOCAL_JIT_ROOT", self.root / "local"
        ):
            self.assertIsNone(jit.start_from_config(self.make_configs().jit_config))
        self.assertIn("TORCH_EXTENSIONS_DIR", os.environ)

    def test_cpu_path_forwards_pipe_writer_and_skips_jit(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        controller, configs, pipe_writer = mock.Mock(), mock.Mock(), mock.Mock()
        with mock.patch.object(
            backend.torch.cuda, "is_available", return_value=False
        ), mock.patch.object(backend, "load_gpu_nic_affinity"), mock.patch.object(
            backend, "local_rank_start", return_value="served"
        ) as rank_start, mock.patch.object(
            jit, "start_from_config", side_effect=AssertionError("no JIT on CPU path")
        ):
            result = backend.start_backend_server(controller, configs, pipe_writer)
        self.assertEqual(result, "served")
        rank_start.assert_called_once_with(controller, configs, 0, pipe_writer)

    def test_bootstrap_failure_fails_open(self):
        scope = mock.Mock()
        manager = mock.Mock()
        manager.bootstrap.return_value = False
        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=scope
        ), mock.patch.object(jit, "JitCacheManager", return_value=manager):
            self.assertIsNone(
                jit.start_from_config(self.make_configs(remote="/r").jit_config)
            )
        manager.bootstrap.assert_called_once_with(5)
        manager.stop.assert_called_once_with()

    def test_bootstrap_exception_fails_open(self):
        scope = mock.Mock()
        manager = mock.Mock()
        manager.bootstrap.side_effect = RuntimeError("restore failed")
        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=scope
        ), mock.patch.object(
            jit, "JitCacheManager", return_value=manager
        ), self.assertLogs(
            level="ERROR"
        ):
            self.assertIsNone(
                jit.start_from_config(self.make_configs(remote="/r").jit_config)
            )
        manager.stop.assert_called_once_with()

    def test_entry_stops_manager_after_ranks_exit(self):
        manager = mock.Mock()
        configs = self.make_configs(remote="/r")
        configs.parallelism_config.world_size = 2
        with mock.patch.object(
            jit, "start_from_config", return_value=manager
        ), mock.patch.object(
            backend, "multi_rank_start", return_value=["proc"]
        ) as ranks, mock.patch.object(
            backend, "load_gpu_nic_affinity"
        ), mock.patch.object(
            backend.torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            backend.torch.cuda, "device_count", return_value=2
        ), mock.patch.object(
            backend, "setproctitle"
        ), mock.patch.object(
            backend.os, "makedirs"
        ), mock.patch.object(
            backend.signal, "signal"
        ):
            backend.start_backend_server(None, configs)
        ranks.assert_called_once()
        manager.stop.assert_called_once_with()

    def test_runtime_handler_installed_after_start_requests_shutdown(self):
        handlers = {}

        class FakeBackendManager:
            instance = None

            def __init__(self, _configs):
                self.request_shutdown = mock.Mock()
                self.serve_forever = mock.Mock()
                FakeBackendManager.instance = self

            def start(self):
                # While the engine is starting, no rank-level handler exists.
                assert backend.signal.SIGTERM not in handlers

        backend_module = types.ModuleType("rtp_llm.server.backend_manager")
        backend_module.BackendManager = FakeBackendManager
        configs = mock.Mock()
        configs.parallelism_config.local_rank = 0
        configs.parallelism_config.world_size = 1
        status = mock.Mock()
        with mock.patch.dict(
            sys.modules, {"rtp_llm.server.backend_manager": backend_module}
        ), contextlib.ExitStack() as stack:
            for name in (
                "copy_gemm_config",
                "set_parallelism_config",
                "setup_cuda_device_and_accl_env",
                "set_global_controller",
                "setproctitle",
            ):
                stack.enter_context(mock.patch.object(backend, name))
            stack.enter_context(
                mock.patch.object(
                    backend.signal,
                    "signal",
                    side_effect=lambda s, h: handlers.__setitem__(s, h),
                )
            )
            stack.enter_context(mock.patch.object(backend, "_send_pipe_status", status))
            backend.local_rank_start(None, configs, pipe_writer=object())
        self.assertEqual(status.call_args[0][1], "success")
        handlers[backend.signal.SIGTERM](backend.signal.SIGTERM, None)
        FakeBackendManager.instance.request_shutdown.assert_called_once_with()
        FakeBackendManager.instance.serve_forever.assert_called_once_with()

    def test_parent_signal_during_jit_setup_prevents_rank_start(self):
        handlers = {}
        configs = self.make_configs(remote="/r")
        configs.parallelism_config.world_size = 2

        def start_jit(_config):
            handlers[backend.signal.SIGTERM](backend.signal.SIGTERM, None)

        with mock.patch.object(
            backend.signal,
            "signal",
            side_effect=lambda signum, handler: handlers.__setitem__(signum, handler),
        ), mock.patch.object(
            jit, "start_from_config", side_effect=start_jit
        ), mock.patch.object(
            backend, "multi_rank_start"
        ) as ranks, mock.patch.object(
            backend, "load_gpu_nic_affinity"
        ), mock.patch.object(
            backend.torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            backend.torch.cuda, "device_count", return_value=2
        ), mock.patch.object(
            backend, "setproctitle"
        ), mock.patch.object(
            backend.os, "makedirs"
        ):
            with self.assertRaises(KeyboardInterrupt):
                backend.start_backend_server(None, configs)
        ranks.assert_not_called()

    def test_start_from_config_releases_manager_on_startup_abort(self):
        manager = mock.Mock()
        manager.bootstrap.side_effect = KeyboardInterrupt("signal 15 during startup")
        config = self.make_configs(remote="/r").jit_config
        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=mock.Mock()
        ), mock.patch.object(jit, "JitCacheManager", return_value=manager):
            with self.assertRaises(KeyboardInterrupt):
                jit.start_from_config(config)
        manager.stop.assert_called_once_with()

    @staticmethod
    def _fake_create(proc):
        def fake(_gc, _cfg, _ctx, processes, readers):
            processes.append(proc)
            readers.append(mock.Mock())

        return fake

    def test_rank_wait_teardown_covers_startup_abort(self):
        configs = self.make_configs(remote="/r")
        configs.distribute_config.fake_gang_env = False
        proc = mock.Mock()
        proc.name, proc.pid = "rank-0", 123
        proc.is_alive.side_effect = [True, False, False]  # dies after terminate
        status = mock.Mock()

        with mock.patch.object(
            backend.multiprocessing, "get_context"
        ), mock.patch.object(
            backend, "_create_rank_processes", side_effect=self._fake_create(proc)
        ), mock.patch.object(
            backend,
            "_wait_for_ranks_startup",
            side_effect=KeyboardInterrupt("signal 15 during startup"),
        ), mock.patch.object(
            backend, "_send_pipe_status", status
        ):
            # The abort must never be downgraded to a catchable Exception.
            with self.assertRaisesRegex(KeyboardInterrupt, "signal 15"):
                backend.multi_rank_start(None, configs, pipe_writer=object())
        proc.terminate.assert_called_once_with()
        self.assertEqual(status.call_args[0][1], "failed")
        self.assertIn("KeyboardInterrupt", status.call_args[0][3])
        self.assertIn("signal 15 during startup", status.call_args[0][3])

    def test_spawn_failure_terminates_already_started_ranks(self):
        configs = self.make_configs(remote="/r")
        configs.distribute_config.fake_gang_env = False
        configs.parallelism_config.world_rank = 0
        configs.parallelism_config.dp_size = 1
        first = mock.Mock()
        first.name, first.pid = "rank-0", 123
        first.is_alive.side_effect = [True, False, False]
        second = mock.Mock()
        second.start.side_effect = RuntimeError("spawn failed")
        second.is_alive.return_value = False
        ctx = mock.Mock()
        ctx.Process.side_effect = [first, second]
        ctx.Pipe.return_value = (mock.Mock(), mock.Mock())
        with mock.patch.object(
            backend.multiprocessing, "get_context", return_value=ctx
        ), mock.patch.object(
            backend, "_get_local_world_size", return_value=2
        ), mock.patch.object(
            backend, "_get_cuda_device_list", return_value=["0", "1"]
        ), mock.patch.object(
            backend, "_send_pipe_status"
        ):
            with self.assertRaisesRegex(Exception, "spawn failed"):
                backend.multi_rank_start(None, configs)
        first.terminate.assert_called_once_with()  # started before the failure

    def test_failed_rank_raises_even_when_reporting_last(self):
        def pipe(status):
            reader = mock.Mock()
            reader.poll.return_value = True
            reader.recv.return_value = {"status": status, "message": "boom"}
            return reader

        proc = mock.Mock()
        proc.is_alive.return_value = True
        proc.exitcode = None
        with self.assertRaisesRegex(Exception, "Rank 1 startup failed: boom"):
            backend._wait_for_ranks_startup(
                [proc, proc], [pipe("success"), pipe("failed")], 2
            )

    def test_process_manager_cleanup_runs_before_hard_exit(self):
        from rtp_llm.utils import process_manager as pm

        cleanup = mock.Mock()
        with mock.patch.object(pm.signal, "signal"):
            manager = pm.ProcessManager(pre_exit_cleanup=cleanup)
        manager.failure_detected = True
        with mock.patch.object(pm.os, "_exit", side_effect=SystemExit(1)) as hard_exit:
            with self.assertRaises(SystemExit):
                manager.monitor_and_release_processes()
        cleanup.assert_called_once_with()
        hard_exit.assert_called_once_with(1)

    def test_hard_exit_runs_bounded_cleanup(self):
        configs = self.make_configs(remote="/r")
        configs.distribute_config.fake_gang_env = False
        proc = mock.Mock()
        proc.name, proc.pid = "rank-0", 123
        proc.is_alive.return_value = True
        cleanup = mock.Mock()

        with mock.patch.object(
            backend.multiprocessing, "get_context"
        ), mock.patch.object(
            backend, "_create_rank_processes", side_effect=self._fake_create(proc)
        ), mock.patch.object(
            backend, "_wait_for_ranks_startup", side_effect=RuntimeError("failed")
        ), mock.patch.object(
            backend, "_send_pipe_status"
        ), mock.patch.object(
            backend.os, "_exit", side_effect=SystemExit(1)
        ):
            with self.assertRaises(SystemExit):
                backend.multi_rank_start(None, configs, cleanup=cleanup)
        cleanup.assert_called_once_with()
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()

    def test_normal_rank_returns_without_hard_exit(self):
        instance = mock.Mock()
        backend_module = types.ModuleType("rtp_llm.server.backend_manager")
        backend_module.BackendManager = mock.Mock(return_value=instance)
        configs = mock.Mock()
        configs.parallelism_config.local_rank = 0
        configs.parallelism_config.world_size = 1
        with mock.patch.dict(
            sys.modules, {"rtp_llm.server.backend_manager": backend_module}
        ), contextlib.ExitStack() as stack:
            for name in (
                "copy_gemm_config",
                "set_parallelism_config",
                "setup_cuda_device_and_accl_env",
                "set_global_controller",
                "setproctitle",
            ):
                stack.enter_context(mock.patch.object(backend, name))
            stack.enter_context(mock.patch.object(backend.signal, "signal"))
            stack.enter_context(mock.patch.object(backend, "_send_pipe_status"))
            hard_exit = stack.enter_context(mock.patch.object(backend.os, "_exit"))
            backend.local_rank_start(None, configs)
        hard_exit.assert_not_called()
        instance.serve_forever.assert_called_once_with()


class TipcTest(JitCacheTestBase):
    def test_source_signature_tracks_content(self):
        source_root = self.root / "tipc"
        source_root.mkdir()
        (source_root / "extension.cc").write_text("int f() { return 1; }\n")
        header = source_root / "extension.h"
        header.write_text("#pragma once\n")
        args = [["-O3"], None, None, False]
        signature = tipc_ffi._source_signature(source_root, args)
        clone = self.root / "clone"
        shutil.copytree(source_root, clone)
        self.assertEqual(signature, tipc_ffi._source_signature(clone, args))
        header.write_text("#pragma once\n// changed\n")
        self.assertNotEqual(signature, tipc_ffi._source_signature(source_root, args))

    def test_compile_scopes_by_arch_and_clears_stale_baton(self):
        from rtp_llm.utils.util import torch_abi_fingerprint

        build_root = self.root / "build"
        cap = (8, 0)
        build_dir = build_root / "tipc" / "fixed-signature"
        build_dir.mkdir(parents=True)
        (build_dir / "lock").touch()  # SIGKILL'd FileBaton corpse hangs load()

        def loaded(*_args, **kwargs):
            self.assertEqual(kwargs["build_directory"], str(build_dir))
            self.assertFalse((build_dir / "lock").exists())  # cleared pre-load
            return "module"

        with mock.patch.dict(
            os.environ,
            {
                "TORCH_EXTENSIONS_DIR": str(build_root),
                "NVCC_APPEND_FLAGS": "--threads=4",
            },
        ), mock.patch.object(
            tipc_ffi.torch.cuda, "get_device_capability", return_value=cap
        ), mock.patch.object(
            tipc_ffi, "_source_signature", return_value="fixed-signature"
        ) as signature, mock.patch.object(
            tipc_ffi, "load", side_effect=loaded
        ) as load:
            self.assertEqual(tipc_ffi.__CompileHelper__().compile(), "module")
            load.assert_called_once()
        source_root, build_args = signature.call_args.args
        self.assertEqual(source_root, Path(tipc_ffi.__file__).with_name("csrc"))
        self.assertLessEqual(
            {"sm_80", "--threads=4", *map(str, torch_abi_fingerprint())},
            set(build_args),
        )


if __name__ == "__main__":
    unittest.main()

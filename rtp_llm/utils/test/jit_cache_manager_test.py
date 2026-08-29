import contextlib
import os
import shutil
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
from rtp_llm.utils.util import COMPILE_FLAG_ENVS, torch_abi_fingerprint


def _fake_probes(hip=None, arch="sm_90", pkg="1_0", toolkit="nvcc-test"):
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


def _shared_acl_supported():
    try:  # gate on production's own requirement, not a copy of its ACL spec
        with tempfile.TemporaryDirectory() as root:
            jit._prepare_shared_root(Path(root))
            return True
    except (OSError, subprocess.SubprocessError):
        return False


requires_non_root = unittest.skipIf(
    os.geteuid() == 0, "root bypasses the permission bits under test"
)
requires_shared_acl = unittest.skipUnless(
    _shared_acl_supported(), "setfacl/getfacl with default ACL support is required"
)


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
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ["TEST_JIT_LOCAL_DIR"] = self.tmp.name
        for name in (*(item.env_name for item in jit.COMPONENTS), *COMPILE_FLAG_ENVS):
            os.environ.pop(name, None)
        jit.setup_jit_cache_env.cache_clear()  # memoized: every test starts cold
        self.addCleanup(jit.setup_jit_cache_env.cache_clear)
        self._seq = 0

    def make_store(self, name="remote", mounted=""):
        remote = self.root / name
        remote.mkdir(parents=True, exist_ok=True)
        return store.RemoteSnapshotStore(remote, mounted)

    def publish(self, snap_store, files: dict[str, bytes]):
        generation = {}
        for rel, data in files.items():
            self._seq += 1
            source = self.root / f"src_{self._seq}"
            source.write_bytes(data)
            generation[rel] = source
        return snap_store.publish_snapshot(generation)


class StoreTest(JitCacheTestBase):
    def test_publish_keeps_immutable_generations(self):
        snap_store, expected = self.make_store(), {}
        for index in range(3):
            expected[f"triton/hash/k-{index}.cubin"] = f"v{index}".encode()
            self.publish(snap_store, expected)
        generations = snapshots(snap_store)
        self.assertEqual(len(generations), 3)
        self.assertTrue(all(f"-{os.uname().nodename}" in p.name for p in generations))
        restored = snap_store.prepare_restore(self.root / "target.staging")
        self.assertEqual(contents(restored.staging), expected)

    def test_published_snapshot_is_readable_under_a_private_umask(self):
        os.umask(0o077)
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/op.so": b"x"})
        self.assertEqual(snapshots(snap_store)[0].stat().st_mode & 0o777, 0o644)

    def test_extract_strips_setuid_bits(self):
        source = self.root / "suid.so"
        source.write_bytes(b"x")
        source.chmod(0o6755)
        archive = self.root / "a.tar.zst"
        store.pack_zstd_tar(archive, {"triton/suid.so": source})
        store.extract_zstd_tar(archive, self.root / "out")
        mode = (self.root / "out" / "triton" / "suid.so").stat().st_mode
        self.assertFalse(mode & 0o7000)

    def test_newest_snapshot_wins_by_name_not_by_mtime(self):
        snap_store = self.make_store()
        for index in range(2):
            self.publish(snap_store, {"triton/k.cubin": f"v{index}".encode()})
        newest = snapshots(snap_store)[-1]  # names embed time_ns: name == age
        # A FUSE mount reports mtimes too coarse (or copy-preserved) to order by.
        os.utime(newest, (0, 0))
        restored = snap_store.prepare_restore(self.root / "staging")
        self.assertEqual(restored.snapshot, newest)
        self.assertEqual(contents(restored.staging), {"triton/k.cubin": b"v1"})

    def test_snapshot_without_mtime_manifest_fails_open(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/op.so": b"x"})
        with mock.patch.object(store, "MTIME_MANIFEST", ".missing"), self.assertLogs(
            level="WARNING"
        ):
            self.assertIsNone(snap_store.prepare_restore(self.root / "staging"))

    def test_restored_tree_is_shared_with_co_tenants(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/op.so": b"x"})
        restored = snap_store.prepare_restore(self.root / "staging")
        target = self.root / "scope"
        self.assertTrue(store.commit_restore(restored.staging, target))
        # Shared producers use temp-file rename, so every build directory must
        # permit a trusted peer to replace another uid's artifact.
        self.assertEqual(target.stat().st_mode & 0o7777, 0o777)
        self.assertEqual((target / "triton").stat().st_mode & 0o7777, 0o777)
        self.assertEqual((target / "triton/op.so").stat().st_mode & 0o777, 0o666)

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
        restored = snap_store.prepare_restore(self.root / "restored.staging")
        self.assertEqual(contents(restored.staging), {"triton/k.cubin": b"v2"})

    def test_stale_baton_reaper_does_not_unlink_replacement(self):
        baton = self.root / "lock"
        baton.touch()
        stale = time.time() - store.STALE_BATON_S - 1
        os.utime(baton, (stale, stale))
        real_stat, swapped = Path.stat, False

        def swap_then_stat(path, *args, **kwargs):
            nonlocal swapped
            if path == baton and not swapped:
                swapped = True
                path.unlink()
                path.touch()
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(Path, "stat", swap_then_stat):
            store.reap_stale_batons(self.root)
        self.assertTrue(baton.exists())

    def test_stale_baton_reaper_does_not_wait_for_peer_lock(self):
        baton = self.root / "lock"
        baton.touch()
        os.utime(baton, (time.time() - store.STALE_BATON_S - 1,) * 2)
        fd = store.acquire_flock(baton)
        done = threading.Event()

        def reap():
            store.reap_stale_batons(self.root)
            done.set()

        worker = threading.Thread(target=reap)
        worker.start()
        try:
            self.assertTrue(done.wait(1))
            self.assertTrue(baton.exists())
        finally:
            os.close(fd)
            worker.join(1)

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

    def test_corrupt_snapshots_fall_back_then_fail_open(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/old": b"old"})
        self.publish(snap_store, {"triton/old": b"old", "triton/new": b"new"})
        bad = snap_store.remote_root / f"{'9' * 20}-bad{store.SNAPSHOT_SUFFIX}"
        bad.write_bytes(b"bad")
        with self.assertLogs(level="WARNING"):
            restored = snap_store.prepare_restore(self.root / "restored.staging")
            self.assertIsNotNone(restored)
        self.assertEqual(
            contents(restored.staging), {"triton/old": b"old", "triton/new": b"new"}
        )
        broken = self.make_store("broken")
        (broken.remote_root / f"{1:020d}{store.SNAPSHOT_SUFFIX}").write_bytes(b"x")
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(broken.prepare_restore(self.root / "cold.staging"))
        self.assertFalse(list((self.root / "cold.staging").glob("*")))

    def test_commit_replaces_existing_empty_scope_root(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/keep/op.so": b"remote"})
        warm = self.root / "warm"
        staging = snap_store.prepare_restore(self.root / "staging")
        (warm / "triton").mkdir(parents=True)
        self.assertTrue(store.commit_restore(staging.staging, warm))
        self.assertEqual(contents(warm), {"triton/keep/op.so": b"remote"})
        self.assertFalse(staging.staging.exists())

    def test_commit_never_replaces_nonempty_scope_root(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/remote.so": b"remote"})
        warm = self.root / "warm"
        (warm / "zz_empty_peer").mkdir(parents=True)
        (warm / "aa_live" / "live.so").parent.mkdir(parents=True)
        (warm / "aa_live" / "live.so").write_bytes(b"live")
        staging = snap_store.prepare_restore(self.root / "staging")
        with self.assertLogs(level="WARNING"):
            self.assertFalse(store.commit_restore(staging.staging, warm))
        self.assertEqual(contents(warm), {"aa_live/live.so": b"live"})
        self.assertTrue((warm / "zz_empty_peer").is_dir())
        self.assertFalse(staging.staging.exists())

    def test_commit_failure_keeps_live_scope_unchanged(self):
        snap_store = self.make_store()
        self.publish(snap_store, {"triton/keep/op.so": b"remote"})
        staging = snap_store.prepare_restore(self.root / "staging")
        target = self.root / "cold"
        with mock.patch.object(store.os, "rename", side_effect=OSError("disk full")):
            with self.assertLogs(level="WARNING"):
                self.assertFalse(store.commit_restore(staging.staging, target))
        self.assertFalse(target.exists())
        self.assertFalse(staging.staging.exists())

    def test_failed_publish_is_atomic(self):
        signatures = iter([(1, 2, 7, 100), (1, 2, 42, 200)])
        cases = (
            (
                "rewritten",
                mock.patch.object(
                    store, "file_sig", side_effect=lambda _: next(signatures)
                ),
                False,
                store.SnapshotRaced,
            ),
            (
                "disappeared",
                mock.patch.object(store, "file_sig", side_effect=FileNotFoundError),
                False,
                store.SnapshotRaced,
            ),
            (
                "chmod",
                mock.patch.object(Path, "chmod", side_effect=OSError("denied")),
                False,
                OSError,
            ),
            ("grew", contextlib.nullcontext(), True, store.SnapshotRaced),
        )
        for name, patcher, grows, error in cases:
            snap_store = self.make_store(name)
            source = self.root / f"{name}.so"
            source.write_bytes(b"payload")
            files, rescan = {f"triton/{name}.so": source}, None
            if grows:  # a builder emits one more product while we pack
                extra = self.root / "extra.so"
                extra.write_bytes(b"extra")
                rescan = lambda: {**files, "triton/extra.so": extra}
            with self.subTest(name=name), patcher, self.assertRaises(error):
                snap_store.publish_snapshot(files, rescan)
            self.assertFalse(snapshots(snap_store))
            self.assertFalse(list(snap_store.remote_root.glob("*.tmp")))

    def test_unmount_waits_for_in_flight_publish(self):
        snap_store = self.make_store(mounted="/mnt/fake")
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
            closer.join(0.05)
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
        restored = snap_store.prepare_restore(self.root / "restored.staging")
        self.assertEqual((restored.staging / "tree/op.so").stat().st_mtime_ns, mtime_ns)


@requires_shared_acl
class ScopeTest(JitCacheTestBase):
    def resolve(self, root=None, **overrides):
        with _fake_probes(**overrides):
            scope = jit.resolve_scope(root or self.root)
        self.assertIsNotNone(scope)
        return scope

    def scope_id(self, root=None, **overrides):
        return self.resolve(root, **overrides).scope_id

    def test_scope_id_covers_every_environment_axis(self):
        base = self.scope_id()
        self.assertEqual(base, self.scope_id())  # deterministic
        self.assertNotEqual(base, self.scope_id(arch="sm_80"))
        self.assertNotEqual(base, self.scope_id(toolkit="nvcc-new"))
        # artifacts bake the local root in, so a relocated tree needs its own remote
        self.assertNotEqual(base, self.scope_id(root=self.root / "elsewhere"))
        with mock.patch("torch.__version__", "0.0.0+scope"):
            self.assertNotEqual(base, self.scope_id())
        with mock.patch("torch.__file__", "/other/prefix/torch/__init__.py"):
            self.assertNotEqual(base, self.scope_id())  # ninja files bake in the prefix
        with _fake_probes(pkg="2_0"):
            bumped = jit.resolve_scope(self.root).scope_id
        self.assertNotEqual(base, bumped)  # any managed package version moves it
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "preset")
        # An opted-out component must fork the id, not share it: this node's snapshots
        # omit that component, and restore is a whole-tree rename rather than a merge,
        # so a shared id would let a partial snapshot strip a full peer's cache.
        self.assertNotEqual(base, self.scope_id())
        os.environ.pop("TRITON_CACHE_DIR")
        for name in COMPILE_FLAG_ENVS:  # every compile flag is part of the identity
            os.environ[name] = "8.0"
            self.assertNotEqual(base, self.scope_id(), name)
            os.environ.pop(name)

        missing = mock.Mock(side_effect=jit.importlib.metadata.PackageNotFoundError)
        with _fake_probes(pkg=missing):
            triton_only = jit.resolve_scope(self.root)
        self.assertEqual([x.name for x in triton_only.components], ["triton"])

    def test_torch_extensions_scope_tracks_rtp_kernel_build(self):
        def with_kernel(version):
            def resolver(name):
                if name != "rtp_kernel":
                    return "1_0"
                if version is None:
                    raise jit.importlib.metadata.PackageNotFoundError(name)
                return version

            return resolver

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
        with _fake_probes():
            scope = jit.setup_jit_cache_env()
        self.assertNotIn("triton", {item.name for item in scope.components})
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(self.root / "preset"))
        torch_ext = next(x for x in scope.components if x.name == "torch_extensions")
        self.assertEqual(os.environ["TORCH_EXTENSIONS_DIR"], str(torch_ext.local_dir))
        self.assertIn(scope.scope_id, os.environ["TORCH_EXTENSIONS_DIR"])

    def test_presetting_every_component_disables_all_redirection(self):
        off = self.root / "off"  # the documented rollback: no root, no ACL, no env
        os.environ["TEST_JIT_LOCAL_DIR"] = str(off)
        for item in jit.COMPONENTS:
            os.environ[item.env_name] = str(self.root / item.name)
        with _fake_probes():
            self.assertIsNone(jit.setup_jit_cache_env())
        self.assertFalse(off.exists())

    def test_baton_reaping_spares_flock_components(self):
        with _fake_probes():
            scope = jit.setup_jit_cache_env()
            dirs = {item.name: item.local_dir for item in scope.components}
            torch_dir, tvm_dir = dirs["torch_extensions"], dirs["tvm_ffi"]
            reaped = [torch_dir / name for name in ("lock", "m/lock_module_fmha")]
            flocks = (torch_dir / "m/.load.lock", tvm_dir / "lock")
            spared = [*flocks, torch_dir / "m/k.so", torch_dir / "m/locking.h"]
            stale = time.time() - store.STALE_BATON_S - 1
            for path in reaped + spared:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                os.utime(path, (stale, stale))
            for item in scope.components:  # a preset env var opts the component out
                os.environ.pop(item.env_name, None)
            jit.setup_jit_cache_env.cache_clear()  # a restart, not a repeat call
            jit.setup_jit_cache_env()
        self.assertEqual([p for p in reaped if p.exists()], [])
        self.assertEqual([p for p in spared if not p.exists()], [])

    def test_local_root_is_shared_across_uids(self):
        owned = self.root / "owned"
        owned.mkdir()
        os.chmod(owned, 0o700)
        os.environ.pop("TEST_JIT_LOCAL_DIR", None)
        with _fake_probes(), mock.patch.object(
            jit, "LOCAL_JIT_ROOT", owned / ".jit_cache"
        ):
            scope = jit.setup_jit_cache_env()
            self.assertIsNotNone(scope)
            self.assertIs(scope, jit.setup_jit_cache_env())  # repeat calls are stable
        self.assertEqual(os.stat(owned).st_mode & 0o7777, 0o1777)
        self.assertEqual(os.stat(owned / ".jit_cache").st_mode & 0o7777, 0o1777)

        os.umask(0o022)
        inherited = owned / ".jit_cache" / "new" / "nested"
        inherited.mkdir(parents=True)
        (inherited / "new.so").touch()
        outside = self.root / "outside"
        outside.touch()
        self.assertEqual(inherited.stat().st_mode & 0o7777, 0o777)
        self.assertEqual((inherited / "new.so").stat().st_mode & 0o777, 0o666)
        self.assertEqual(outside.stat().st_mode & 0o777, 0o644)

    def test_managed_directories_inherit_shared_acl(self):
        with _fake_probes():
            scope = jit.setup_jit_cache_env()
        version_root = scope.root.parent
        locks, staging = version_root / jit.LOCKS_DIR, version_root / jit.STAGING_DIR
        self.assertFalse(version_root.exists())
        lock = locks / "probe.lock"
        os.close(store.acquire_flock(lock))
        staging.mkdir()
        for path in (version_root, locks, staging):
            self.assertEqual(os.stat(path).st_mode & 0o7777, 0o777, path)
        self.assertEqual(lock.stat().st_mode & 0o777, 0o666)

    @requires_non_root
    def test_unwritable_scope_root_fails_open(self):
        with _fake_probes():
            scope = jit.setup_jit_cache_env()
            self.assertIsNotNone(scope)
            scope.root.mkdir(parents=True)
            scope.root.chmod(0o500)  # a co-tenant tree we cannot add kernels to
            for item in jit.COMPONENTS:  # a second resolve needs a clean env
                os.environ.pop(item.env_name, None)
            jit.setup_jit_cache_env.cache_clear()
            with self.assertLogs(level="WARNING") as logs:
                self.assertIsNone(jit.setup_jit_cache_env())
            scope.root.chmod(0o700)
        self.assertIn("not shared by its owner", "\n".join(logs.output))

    def test_missing_acl_support_fails_open(self):
        os.umask(0o022)
        with _fake_probes(), mock.patch.object(
            jit, "_run", side_effect=[OSError("no setfacl"), "user::rwx"]
        ):
            with self.assertLogs(level="WARNING") as logs:
                self.assertIsNone(jit.setup_jit_cache_env())
        self.assertIn("shared ACL unavailable", "\n".join(logs.output))

    def test_preconfigured_foreign_root_does_not_require_ownership(self):
        shared = self.root / ".jit_cache"
        shared.mkdir()
        os.chmod(self.root, 0o777)
        os.chmod(shared, 0o777)
        jit._prepare_shared_root(shared)
        with _fake_probes(), mock.patch.object(
            jit.os, "chmod", side_effect=PermissionError
        ):
            with mock.patch.object(
                jit,
                "_run",
                side_effect=[PermissionError, "other::rwx\ndefault:other::rwx"],
            ):
                self.assertIsNotNone(jit.setup_jit_cache_env())

    def test_symlinked_local_root_is_refused(self):
        target = self.root / "target"
        target.mkdir(mode=0o700)
        linked = self.root / "linked"
        linked.symlink_to(target, target_is_directory=True)
        os.environ["TEST_JIT_LOCAL_DIR"] = str(linked)
        with _fake_probes(), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(jit.setup_jit_cache_env())
        self.assertIn("untrusted JIT dir", "\n".join(logs.output))
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

    def test_optimized_gpu_probe_rejects_mixed_architectures(self):
        fake_torch = (
            "class C:\n device_count=lambda s:2\n"
            " get_device_capability=lambda s,i:((8,0),(9,0))[i]\n"
            "class V:hip=None\nclass T:cuda=C();version=V()\ntorch=T()"
        )
        with mock.patch.object(
            jit, "GPU_PROBE", jit.GPU_PROBE.replace("import torch", fake_torch)
        ), mock.patch.dict(os.environ, {"PYTHONOPTIMIZE": "1"}), self.assertLogs(
            level="WARNING"
        ) as logs:
            self.assertIsNone(jit._accelerator_scope("cuda", "12.8"))
        self.assertIn("mixed GPU architectures", "\n".join(logs.output))


@requires_shared_acl
class ManagerTest(JitCacheTestBase):
    def setUp(self):
        super().setUp()
        self.managers = []
        self.addCleanup(lambda: [manager.stop() for manager in self.managers])

    def make_scope(self):
        with _fake_probes():
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

    def publish_remote(self, manager, rel, data):
        """Publish one member straight to the scope's remote, as a peer host would."""
        version_root = Path(manager._remote_value) / jit.RTP_JIT_VERSION
        remote = version_root / manager.scope.scope_id
        remote.mkdir(parents=True, exist_ok=True)
        self.publish(store.RemoteSnapshotStore(remote), {rel: data})

    def test_non_positive_setup_timeout_starts_cold(self):
        # argparse rejects these; a raw env read must degrade, not raise.
        for timeout_s in (0, -2):
            blocked = threading.Event()
            manager = self.make_manager(self.make_scope())
            try:
                with self.subTest(timeout_s=timeout_s), mock.patch.object(
                    jit.JitCacheManager, "_prepare", lambda _: blocked.wait(10)
                ), self.assertLogs(level="WARNING") as logs:
                    self.assertFalse(manager.bootstrap(timeout_s))
                self.assertIn("setup timed out", "\n".join(logs.output))
            finally:
                blocked.set()

    def test_publish_then_restore_round_trip(self):
        scope = self.make_scope()
        producer = self.make_manager(scope)
        self.assertTrue(producer.bootstrap(timeout_s=30))
        artifact = self.write_artifact(scope, "triton/hash/kernel.cubin")
        producer._dirty.set()
        producer.publish_pending_snapshot()
        self.assertTrue(snapshots(producer.store))
        producer.stop()

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

    def test_live_peer_empty_tree_swap_still_backfills(self):
        scope = self.make_scope()
        peer = self.make_manager(scope)
        self.assertTrue(peer.bootstrap(timeout_s=30))
        self.publish_remote(peer, "triton/remote.cubin", b"remote")
        consumer = self.make_manager(scope)
        self.assertTrue(consumer.bootstrap(timeout_s=30))
        self.write_artifact(scope, "triton/peer.cubin", b"peer")
        peer._dirty.set()
        peer.publish_pending_snapshot()
        peek = self.root / "peek"
        peek.mkdir()
        store.extract_zstd_tar(snapshots(peer.store)[-1], peek)
        self.assertEqual(
            contents(peek),
            {"triton/remote.cubin": b"remote", "triton/peer.cubin": b"peer"},
        )

    def test_bootstrap_skips_restore_on_peer_warm_tree(self):
        scope = self.make_scope()
        self.write_artifact(scope, "triton/peer.cubin", b"peer")
        manager = self.make_manager(scope)
        self.publish_remote(manager, "triton/remote.cubin", b"remote")
        self.assertTrue(manager.bootstrap(timeout_s=30))
        # A live peer owns this tree: nothing may be unpacked, let alone renamed in.
        self.assertFalse(
            (scope.root.parent / jit.STAGING_DIR / scope.scope_id).exists()
        )
        self.assertEqual(contents(scope.root), {"triton/peer.cubin": b"peer"})

    def test_warm_tree_seeds_empty_remote_and_restart_stays_quiet(self):
        scope = self.make_scope()
        seeder = self.make_manager(scope)
        self.write_artifact(scope, "triton/warm.cubin", b"warm")
        self.assertTrue(seeder.bootstrap(timeout_s=30))
        seeder.publish_pending_snapshot()
        self.assertEqual(len(snapshots(seeder.store)), 1)  # empty remote seeded once
        seeder.stop()

        restart = self.make_manager(scope)
        self.assertTrue(restart.bootstrap(timeout_s=30))
        restart.publish_pending_snapshot()
        self.assertEqual(len(snapshots(restart.store)), 1)  # published: no re-upload

        # No watcher event stands in here: the rescan alone must notice the write,
        # which is the only path left once inotify has gone blind.
        self.write_artifact(scope, "triton/fresh.cubin", b"fresh")
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
        self.publish_remote(manager, "triton/k.cubin", b"fresh")
        with mock.patch.object(
            store.RemoteSnapshotStore, "prepare_restore", slow_prepare
        ), self.assertLogs(level="WARNING"):
            self.assertFalse(manager.bootstrap(timeout_s=0.05))
        self.write_artifact(scope, "triton/local.cubin", b"local")
        release.set()
        manager._prepare_thread.join(timeout=10)
        self.assertFalse(manager._prepare_thread.is_alive())
        retry = self.make_manager(scope)
        self.assertTrue(retry.bootstrap(timeout_s=30))
        retry._dirty.set()
        retry.publish_pending_snapshot()
        self.assertEqual(len(snapshots(retry.store)), 2)
        self.assertEqual((scope.root / "triton" / "local.cubin").read_bytes(), b"local")
        staging = scope.root.parent / jit.STAGING_DIR / scope.scope_id
        self.assertFalse(list(staging.glob("*")))

    def test_event_handler_marks_dirty_only_for_trigger_files(self):
        triton = replace(
            next(x for x in jit.COMPONENTS if x.name == "triton"),
            local_dir=self.root / "triton",
        )
        manager = self.make_manager(jit.Scope("test", self.root, (triton,)))
        quiet = triton.local_dir / "hash" / "t.autotune.json"
        loud = triton.local_dir / "hash" / "k.cubin"
        quiet.parent.mkdir(parents=True, exist_ok=True)
        for path, event_type, expect in (
            (quiet, "closed", False),
            (loud, "moved", True),
        ):
            manager._dirty.clear()
            path.write_bytes(b"x")
            manager.on_any_event(
                mock.Mock(
                    event_type=event_type,
                    src_path=str(quiet),
                    dest_path=str(path),
                    is_directory=False,
                )
            )
            self.assertEqual(manager._dirty.is_set(), expect, path.name)

    def test_restore_lock_wait_honors_setup_timeout(self):
        scope = self.make_scope()
        manager = self.make_manager(scope)
        lock = scope.root.parent / jit.LOCKS_DIR / f"{scope.scope_id}.restore.lock"
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

    def test_stop_is_bounded_by_a_hung_final_publish(self):
        manager = jit.JitCacheManager(mock.Mock(), "")
        manager.store = store.RemoteSnapshotStore(self.root, "/mnt/fake")
        publishing, release, unmounted = (threading.Event() for _ in range(3))
        self.addCleanup(release.set)

        def publish():  # a real publish holds the mount lock while it packs
            with manager.store._mount_lock:
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
            manager.stop()  # returns without waiting for the publish to land
            self.assertTrue(publishing.wait(1))
            # close() shares the mount lock, so unmount waits the publish out
            # instead of truncating it.
            self.assertFalse(unmounted.is_set())
            release.set()
            self.assertTrue(unmounted.wait(5))

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

    def test_publish_lock_defers_peer_managers(self):
        scope = self.make_scope()
        managers = (self.make_manager(scope), self.make_manager(scope))
        self.assertTrue(all(manager.bootstrap(30) for manager in managers))
        self.write_artifact(scope, "triton/kernel.cubin")
        publish_lock = (
            scope.root.parent / jit.LOCKS_DIR / f"{scope.scope_id}.publish.lock"
        )
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


class BackendTest(JitCacheTestBase):
    def make_configs(self, remote="", world_size=1):
        configs = mock.Mock()
        configs.jit_config.remote_jit_dir = remote
        configs.jit_config.jit_cache_setup_timeout_s = 5
        configs.jit_config.manage_jit_cache = True
        configs.parallelism_config.world_size = world_size
        return configs

    def patched_backend(self, *, cuda=True, device_count=1, signal_handler=None):
        """Neutralize start_backend_server's process-wide startup side effects."""
        stack = contextlib.ExitStack()
        for target, name, kwargs in (
            (backend, "load_gpu_nic_affinity", {}),
            (backend, "setproctitle", {}),
            (backend.os, "makedirs", {}),
            (backend.signal, "signal", {"side_effect": signal_handler}),
            (backend.torch.cuda, "is_available", {"return_value": cuda}),
            (backend.torch.cuda, "device_count", {"return_value": device_count}),
        ):
            stack.enter_context(mock.patch.object(target, name, **kwargs))
        return stack

    def test_no_remote_sets_up_local_jit_env(self):
        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=mock.Mock()
        ) as setup, mock.patch.object(jit, "JitCacheManager") as manager:
            self.assertIsNone(jit.start_from_config(self.make_configs().jit_config))
        setup.assert_called_once_with()
        manager.assert_not_called()

    def test_manage_jit_cache_false_skips_all_setup(self):
        config = self.make_configs(remote="/r").jit_config
        config.manage_jit_cache = False
        with mock.patch.object(jit, "setup_jit_cache_env") as setup, mock.patch.object(
            jit, "JitCacheManager"
        ) as manager:
            self.assertIsNone(jit.start_from_config(config))
        setup.assert_not_called()
        manager.assert_not_called()

    def test_cpu_path_forwards_pipe_writer_and_skips_jit(self):
        controller, configs, pipe_writer = mock.Mock(), mock.Mock(), mock.Mock()
        with self.patched_backend(cuda=False), mock.patch.object(
            backend, "local_rank_start", return_value="served"
        ) as rank_start, mock.patch.object(jit, "start_from_config") as jit_start:
            result = backend.start_backend_server(controller, configs, pipe_writer)
        self.assertEqual(result, "served")
        rank_start.assert_called_once_with(controller, configs, 0, pipe_writer)
        jit_start.assert_not_called()

    def test_bootstrap_failure_releases_the_manager(self):
        # start_from_config owns no fail-open: a false return, an Exception, and a
        # startup-abort BaseException must all stop the manager and propagate.
        for failure in (
            False,
            RuntimeError("restore failed"),
            KeyboardInterrupt("sig 15"),
        ):
            manager = mock.Mock()
            manager.bootstrap.side_effect = failure or None
            manager.bootstrap.return_value = False
            config = self.make_configs(remote="/r").jit_config
            with self.subTest(failure=failure), mock.patch.object(
                jit, "setup_jit_cache_env", return_value=mock.Mock()
            ), mock.patch.object(jit, "JitCacheManager", return_value=manager):
                if failure:  # the only fail-open handler is the caller's, not ours
                    self.assertRaises(type(failure), jit.start_from_config, config)
                else:
                    self.assertIsNone(jit.start_from_config(config))
                manager.bootstrap.assert_called_once_with(5)
                manager.stop.assert_called_once_with()

    def test_jit_setup_failure_still_starts_the_engine(self):
        configs = self.make_configs(remote="/r", world_size=1)
        with self.patched_backend(), mock.patch.object(
            jit, "start_from_config", side_effect=RuntimeError("setup blew up")
        ), mock.patch.object(
            backend, "local_rank_start", return_value="served"
        ) as rank_start, self.assertLogs(
            level="ERROR"
        ):
            self.assertEqual(backend.start_backend_server(None, configs), "served")
        rank_start.assert_called_once()

    def test_multi_rank_start_owns_manager_cleanup(self):
        manager = mock.Mock()
        configs = self.make_configs(remote="/r", world_size=2)
        with self.patched_backend(device_count=2), mock.patch.object(
            jit, "start_from_config", return_value=manager
        ), mock.patch.object(
            backend, "multi_rank_start", return_value=["proc"]
        ) as ranks:
            backend.start_backend_server(None, configs)
        ranks.assert_called_once()
        self.assertIs(ranks.call_args.kwargs["cleanup"], manager.stop)
        manager.stop.assert_called_once_with()

    def test_single_rank_returns_after_stopping_the_manager(self):
        events = []
        manager = mock.Mock()
        manager.stop.side_effect = lambda: events.append("stop")
        configs = self.make_configs(remote="/r", world_size=1)
        with self.patched_backend(device_count=1), mock.patch.object(
            jit, "start_from_config", return_value=manager
        ), mock.patch.object(
            backend,
            "local_rank_start",
            side_effect=lambda *_: events.append("rank") or "served",
        ), mock.patch.object(
            # A hard exit here would truncate the caller's own teardown.
            backend.os,
            "_exit",
            side_effect=AssertionError("single rank must return, not hard exit"),
        ):
            self.assertEqual(backend.start_backend_server(None, configs), "served")
        self.assertEqual(events, ["rank", "stop"])

    def test_runtime_handler_installed_after_start_requests_shutdown(self):
        handlers = {}

        class FakeBackendManager:
            instance = None

            def __init__(self, _configs):
                self.request_shutdown = mock.Mock()
                self.serve_forever = mock.Mock()
                FakeBackendManager.instance = self

            def start(self):
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
        configs = self.make_configs(remote="/r", world_size=2)

        def start_jit(_config):
            handlers[backend.signal.SIGTERM](backend.signal.SIGTERM, None)

        with self.patched_backend(
            device_count=2, signal_handler=handlers.__setitem__
        ), mock.patch.object(
            jit, "start_from_config", side_effect=start_jit
        ), mock.patch.object(
            backend, "multi_rank_start"
        ) as ranks:
            with self.assertRaises(KeyboardInterrupt):
                backend.start_backend_server(None, configs)
        ranks.assert_not_called()

    @staticmethod
    def _fake_create(proc):
        def fake(_gc, _cfg, _ctx, processes, readers, shutdown_ready_events):
            processes.append(proc)
            readers.append(mock.Mock())
            shutdown_ready_events.append(None)

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

    def test_rank_wait_repolls_without_an_extra_sleep(self):
        reader = mock.Mock()
        reader.poll.side_effect = [False, True]
        reader.recv.return_value = {
            "status": "success",
            "message": "rank ready",
        }
        proc = mock.Mock()
        proc.is_alive.return_value = True
        proc.exitcode = None

        with mock.patch.object(backend.time, "sleep") as sleep:
            backend._wait_for_ranks_startup([proc], [reader], 1)

        self.assertEqual(reader.poll.call_count, 2)
        sleep.assert_not_called()

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
        self.assertNotEqual(signature, tipc_ffi._source_signature(clone, args))
        header.write_text("#pragma once\n// changed\n")
        self.assertNotEqual(signature, tipc_ffi._source_signature(source_root, args))

    def test_compile_scopes_by_arch_and_clears_stale_baton(self):
        build_root = self.root / "build"
        cap = (8, 0)
        build_dir = build_root / "tipc" / "fixed-signature"
        build_dir.mkdir(parents=True)
        (build_dir / "lock").touch()  # SIGKILL'd FileBaton corpse hangs load()

        def loaded(*_args, **kwargs):
            self.assertEqual(kwargs["build_directory"], str(build_dir))
            self.assertFalse((build_dir / "lock").exists())  # cleared pre-load
            return "module"

        with mock.patch.object(
            tipc_ffi.os, "open", wraps=os.open
        ) as open_file, mock.patch.dict(
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
            self.assertTrue(open_file.call_args.args[1] & os.O_NOFOLLOW)
            source_root, build_args = signature.call_args.args
            self.assertEqual(source_root, Path(tipc_ffi.__file__).with_name("csrc"))
            self.assertTrue(
                {
                    "sm_80",
                    *map(str, torch_abi_fingerprint()),
                    *(
                        f"{name}={os.environ.get(name, '')}"
                        for name in COMPILE_FLAG_ENVS
                    ),
                }.issubset(build_args)
            )

    def test_compile_key_names_environment_inputs(self):
        def build_args(env):
            env = {"TORCH_EXTENSIONS_DIR": str(self.root / "build"), **env}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                tipc_ffi.torch.cuda, "get_device_capability", return_value=(8, 0)
            ), mock.patch.object(
                tipc_ffi, "_source_signature", return_value="fixed-signature"
            ) as signature, mock.patch.object(
                tipc_ffi, "load", return_value="module"
            ):
                tipc_ffi.__CompileHelper__().compile()
                return signature.call_args.args[1]

        append = build_args({"NVCC_APPEND_FLAGS": "same"})
        prepend = build_args({"NVCC_PREPEND_FLAGS": "same"})
        self.assertNotEqual(append, prepend)

        base = build_args({})
        for name in ("CXX", "CC", "PYTORCH_NVCC"):
            with self.subTest(name=name):
                self.assertNotEqual(base, build_args({name: "/toolchain/compiler"}))


if __name__ == "__main__":
    unittest.main()

import concurrent.futures
import contextlib
import fcntl
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm import start_backend_server as backend
from rtp_llm.model_loader.tipc import ffi as tipc_ffi
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


def _fake_scopes(local=None):
    # Stub the backend/scope probes (and optionally local root) so tests don't
    # shell out to a compiler or query a GPU. Returns an ExitStack usable as `with`.
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch("torch.version.hip", None))
    stack.enter_context(mock.patch("torch.version.cuda", "12.8"))
    stack.enter_context(
        mock.patch.object(jit, "_accelerator_scope", return_value="cuda-test")
    )
    stack.enter_context(
        mock.patch.object(jit, "_torch_scope", return_value="torch-test")
    )
    stack.enter_context(
        mock.patch.object(jit, "_cpp_runtime_scope", return_value="cxx-test")
    )
    stack.enter_context(mock.patch.object(jit, "_pkg_version", return_value="1_0"))
    if local is not None:
        stack.enter_context(mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)))
    return stack


def component(components, name):
    return next(item for item in components if item.name == name)


def contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def tipc_build_dir(build_root: Path, cap: tuple[int, int]) -> Path:
    # Must match ffi.compile()'s flattened build_args exactly.
    args = [
        "-O3",
        "-O3",
        "-use_fast_math",
        f"sm_{cap[0]}{cap[1]}",
        tipc_ffi.torch.version.cuda,
    ]
    source_root = Path(tipc_ffi.__file__).with_name("csrc")
    return build_root / "tipc" / tipc_ffi._source_signature(source_root, args)


def peer_setup(event, entered, result):
    with mock.patch.object(jit, "setup_jit_cache_env", return_value=((), True)):
        started = time.monotonic()
        entered.set()
        backend._setup_jit_cache("/remote", 1, event)
        result.put(time.monotonic() - started)


class JitCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.stop()
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(self, remote=None):
        local = self.root / "local"
        remote = Path(remote or self.root / "remote")
        remote.mkdir(parents=True, exist_ok=True)
        with _fake_scopes(local):
            components, compatible = jit.setup_jit_cache_env()
            remote_root = jit.resolve_remote_root(remote) if compatible else None
            manager = jit.JitCacheManager(remote_root, components)
        self.managers.append(manager)
        return manager

    def test_startup_scope_events_and_shutdown(self):
        local_peer, local_owner, peer, ready = (
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
            mock.Mock(),
        )
        manager = mock.Mock()
        manager.start_background_sync.side_effect = RuntimeError("boom")

        lock_held = False

        @contextlib.contextmanager
        def lifecycle_lock(target):
            nonlocal lock_held
            self.assertEqual(target, self.root / "local")
            self.assertFalse(lock_held)
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

        def resolve_remote(_remote):
            self.assertTrue(lock_held)
            return self.root / "remote"

        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=((object(),), True)
        ) as setup, mock.patch.object(
            jit, "JitCacheManager", return_value=manager
        ) as manager_cls, mock.patch.object(
            jit, "resolve_remote_root", side_effect=resolve_remote
        ), mock.patch.object(
            jit, "LOCAL_JIT_DIR", str(self.root / "local")
        ), mock.patch.object(
            store, "restore_lock", side_effect=lifecycle_lock
        ):
            self.assertIsNone(backend._setup_jit_cache("", 1, local_peer))
            self.assertIsNone(backend._setup_jit_cache("", 0, local_owner))
            self.assertIsNone(backend._setup_jit_cache("/remote", 1, peer))
            with self.assertLogs(level="ERROR"):
                # A worker failure is swallowed and returns None (cold start).
                self.assertIsNone(backend._setup_jit_cache("/remote", 0, ready))
        self.assertEqual(setup.call_count, 4)  # local env is set up on every path
        self.assertEqual(manager_cls.call_count, 1)
        local_peer.wait.assert_not_called()  # no remote: nothing to wait for
        local_owner.set.assert_called_once_with()
        peer.wait.assert_called_once_with(timeout=backend.JIT_CACHE_SETUP_TIMEOUT_S + 5)
        self.assertEqual(ready.set.call_count, 1)
        self.assertIsNotNone(manager.start_background_sync.call_args.kwargs["commit"])

        # A wedged setup thread (hard mount) must not hang rank0: the hard join times
        # out and we cold-start (None) without blocking on the daemon worker.
        release = threading.Event()
        self.addCleanup(release.set)
        hung = mock.Mock()
        hung.start_background_sync.side_effect = lambda **_kwargs: release.wait(10)
        timed = mock.Mock()
        started = time.monotonic()
        with mock.patch.object(
            backend, "JIT_CACHE_SETUP_TIMEOUT_S", 0.05
        ), mock.patch.object(
            jit, "setup_jit_cache_env", return_value=((object(),), True)
        ), mock.patch.object(
            jit, "JitCacheManager", return_value=hung
        ), mock.patch.object(
            jit, "resolve_remote_root", side_effect=resolve_remote
        ), mock.patch.object(
            jit, "LOCAL_JIT_DIR", str(self.root / "local")
        ), mock.patch.object(
            store, "restore_lock", side_effect=lifecycle_lock
        ), self.assertLogs(
            level="WARNING"
        ):
            self.assertIsNone(backend._setup_jit_cache("/remote", 0, timed))
        self.assertGreaterEqual(time.monotonic() - started, 0.04)
        timed.set.assert_called_once()
        release.set()

        ctx = multiprocessing.get_context("spawn")
        elapsed_times = []
        for is_ready in (True, False):
            event, entered, result = ctx.Event(), ctx.Event(), ctx.Queue()
            if is_ready:
                event.set()
            process = ctx.Process(target=peer_setup, args=(event, entered, result))
            process.start()
            self.assertTrue(entered.wait(timeout=10))
            if not is_ready:
                time.sleep(0.1)
                event.set()
            elapsed = result.get(timeout=10)
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
            self.assertLess(elapsed, 1)
            elapsed_times.append(elapsed)
            result.close()
        self.assertLess(elapsed_times[0], elapsed_times[1])

        with mock.patch("torch.version.cuda", "12.9"), mock.patch(
            "torch.cuda.current_device", return_value=2
        ), mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ) as capability:
            self.assertEqual(jit._accelerator_scope(jit.CUDA), "cuda-12_9-sm_90")
            capability.assert_called_once_with(2)

        root, external = self.root / "cache", self.root / "external"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(external)
        with _fake_scopes(root):
            resolved = jit._resolve_components()
            managed, _ = jit.setup_jit_cache_env()
            with mock.patch.object(jit, "_pkg_version", return_value=None):
                versionless = jit._resolve_components()
        self.assertNotIn("torch_extensions", {item.name for item in managed})
        self.assertEqual(os.environ["TORCH_EXTENSIONS_DIR"], str(external))
        self.assertEqual(
            {item.name for item in versionless}, {"torch_extensions", "triton"}
        )
        self.assertEqual(
            {item.name for item in resolved},
            {item.name for item in jit.COMPONENTS if item.backend != jit.ROCM},
        )
        for name, path in (
            ("deep_gemm", "kernel.cubin"),
            ("trtllm_deep_gemm", "nvcc_kernel.cubin"),
            ("tvm_ffi", "kernel.so"),
            ("triton", "kernel.so"),
        ):
            item = component(resolved, name)
            self.assertTrue(item.should_sync(path, "created"))
            self.assertTrue(item.should_sync(path, "closed"))
            self.assertTrue(item.should_sync(path, "moved"))

        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        event_manager = self.make_manager()
        with mock.patch.object(jit, "Observer", return_value=mock.Mock()):
            event_manager.start_background_sync()
            triton = component(event_manager.components, "triton")
            path = triton.local_dir / "hash/final.cubin"
            path.parent.mkdir(parents=True)
            handler = jit._EventHandler(event_manager)
            path.touch()
            handler.on_any_event(
                mock.Mock(event_type="created", src_path=str(path), is_directory=False)
            )
            path.write_bytes(b"partial")
            handler.on_any_event(
                mock.Mock(event_type="created", src_path=str(path), is_directory=False)
            )
            path.write_bytes(b"final")
            handler.on_any_event(
                mock.Mock(event_type="closed", src_path=str(path), is_directory=False)
            )
            event_manager.stop()
        restored = self.root / "event_target"
        event_manager.store.restore(restored)
        self.assertEqual(
            (restored / path.relative_to(event_manager.local_root)).read_bytes(),
            b"final",
        )

        blocked, release = self.make_manager(), threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        blocked._sync_thread = worker
        started = time.monotonic()
        with mock.patch.object(jit, "SHUTDOWN_TIMEOUT_S", 0.01):
            blocked.stop()
        self.assertLess(time.monotonic() - started, 1)
        release.set()
        worker.join(timeout=1)

        idle = self.make_manager()
        idle._stop = mock.Mock()
        idle._stop.wait.side_effect = (False, True)
        idle._last_event_at = 1.0
        with mock.patch.object(
            jit.time, "monotonic", return_value=1.0
        ), mock.patch.object(idle, "publish_pending_snapshot") as flush:
            idle._sync_loop()
        flush.assert_called_once()  # final flush only; recent events are coalesced

    def test_observer_partial_start_failure_is_cleaned_up(self):
        manager = self.make_manager()
        observer = mock.Mock()
        observer.start.side_effect = RuntimeError("emitter failed")
        observer.is_alive.return_value = False
        with mock.patch.object(jit, "Observer", return_value=observer):
            with self.assertRaisesRegex(RuntimeError, "emitter failed"):
                manager.start_background_sync()
        observer.stop.assert_called_once()
        self.assertIsNone(manager._observer)

    def test_snapshot_scan_skips_files_vanishing_mid_scan(self):
        # A file pruned or rebuilt away between the rglob walk and its stat must be
        # skipped, not abort the whole publish cycle with FileNotFoundError.
        manager = self.make_manager()
        triton = component(manager.components, "triton")
        gone = triton.local_dir / "hash/gone.cubin"
        gone.parent.mkdir(parents=True, exist_ok=True)
        gone.write_bytes(b"x")
        real_stat, attempts = Path.stat, []

        def flaky_stat(path, *args, **kwargs):
            if path == gone:
                attempts.append(1)
                if len(attempts) > 2:  # is_symlink/is_file see it, scan stat loses it
                    raise FileNotFoundError(gone)
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(Path, "stat", flaky_stat):
            self.assertEqual(manager._snapshot_files(), {})

    def test_snapshot_lifecycle(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        expected = {}
        generation = {}
        with mock.patch.object(store, "SNAPSHOT_KEEP", 2):
            for index in range(3):
                source = self.root / f"kernel-{index}.cubin"
                source.write_bytes(f"value-{index}".encode())
                name = f"triton/hash/kernel-{index}.cubin"
                expected[name] = source.read_bytes()
                generation[name] = source
                snapshot_store.publish_snapshot(lambda: generation)
            # GC is mtime-gated (never unlink-races a reader), so all 3 fresh
            # snapshots survive until they age past the cutoff.
            self.assertEqual(len(list(remote.glob(f"*{store.SNAPSHOT_SUFFIX}"))), 3)
            # Backdate the two oldest past the cutoff; the next publish trims them.
            for snap in sorted(remote.glob(f"*{store.SNAPSHOT_SUFFIX}"))[:2]:
                stale = time.time() - store.IDLE_REAP_S - 1
                os.utime(snap, (stale, stale))
            source = self.root / "kernel-3.cubin"
            source.write_bytes(b"value-3")
            name = "triton/hash/kernel-3.cubin"
            expected[name] = source.read_bytes()
            generation[name] = source
            snapshot_store.publish_snapshot(lambda: generation)
        target = self.root / "target"
        self.assertTrue(snapshot_store.restore(target))
        self.assertEqual(contents(target), expected)
        self.assertEqual(len(list(remote.glob(f"*{store.SNAPSHOT_SUFFIX}"))), 2)
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

        broken = self.root / "broken"
        broken.mkdir()
        (broken / f"{1:020d}{store.SNAPSHOT_SUFFIX}").write_bytes(b"broken")

        def partial_extract(_snapshot, staging):
            (staging / "triton").mkdir(parents=True)
            (staging / "triton/partial").write_bytes(b"partial")
            raise RuntimeError("killed mid-extract")

        # A failed extract — crash or wedged mount — never commits a half-written
        # tree, leaves no staging temp behind, and never claims the tree (.ready),
        # so the next cold start retries.
        for index, side_effect in enumerate((partial_extract, TimeoutError)):
            cold = self.root / f"cold-{index}"
            with mock.patch.object(
                store, "extract_zstd_tar", side_effect=side_effect
            ), self.assertLogs(level="WARNING"):
                self.assertFalse(store.RemoteSnapshotStore(broken).restore(cold))
            self.assertFalse(cold.exists())
            self.assertFalse(cold.with_name(f"{cold.name}.ready").exists())
            self.assertFalse(list(self.root.glob("*.stage*")))

        old = self.root / "old"
        old.write_bytes(b"old")
        new = self.root / "new"
        new.write_bytes(b"new")

        # Restore is cold-only. A warm tree belongs to its active builder and is
        # never merged with a remote generation.
        merge = self.root / "merge_remote"
        merge.mkdir()
        merge_store = store.RemoteSnapshotStore(merge)
        merge_store.publish_snapshot(
            lambda: {"triton/keep/op.so": old, "triton/late/op.so": new}
        )

        # A warm target remains byte-for-byte untouched.
        warm = self.root / "merge_live"
        (warm / "triton/keep").mkdir(parents=True)
        (warm / "triton/keep/op.so").write_bytes(b"healthy")
        self.assertFalse(merge_store.restore(warm))
        self.assertEqual(contents(warm), {"triton/keep/op.so": b"healthy"})

        # A fresh local lifecycle can still restore the complete generation.
        cold = self.root / "cold_retry"
        merge_store.restore(cold)
        self.assertEqual(
            contents(cold),
            {"triton/keep/op.so": b"old", "triton/late/op.so": b"new"},
        )
        self.assertFalse(list(self.root.glob("*.stage*")))

        healthy = self.root / "healthy"
        healthy.mkdir()
        healthy_store = store.RemoteSnapshotStore(healthy)
        healthy_generation = {"triton/old": old}
        healthy_store.publish_snapshot(lambda: healthy_generation)
        healthy_generation["triton/new"] = new
        healthy_store.publish_snapshot(lambda: healthy_generation)
        (healthy / f"{'9' * 20}-bad{store.SNAPSHOT_SUFFIX}").write_bytes(b"bad")
        with self.assertLogs(level="WARNING"):
            restored = self.root / "restored"
            healthy_store.restore(restored)
        self.assertEqual(
            contents(restored), {"triton/old": b"old", "triton/new": b"new"}
        )
        # A fresh temp upload (possibly a peer's in-flight one) is never reclaimed.
        live_tmp = healthy / f"{'1' * 20}-host{store.SNAPSHOT_SUFFIX}.tmp"
        live_tmp.write_bytes(b"orphan")
        healthy_store.publish_snapshot(lambda: healthy_generation)
        self.assertEqual(live_tmp.read_bytes(), b"orphan")
        # A long-dead one (mtime past IDLE_REAP_S) is reclaimed next publish.
        dead = healthy / f"{'2' * 20}-host{store.SNAPSHOT_SUFFIX}.tmp"
        dead.write_bytes(b"orphan")
        stale = time.time() - store.IDLE_REAP_S - 1
        os.utime(dead, (stale, stale))
        healthy_store.publish_snapshot(lambda: healthy_generation)
        self.assertFalse(dead.exists())

        concurrent_root = self.root / "concurrent"
        concurrent_root.mkdir()
        concurrent_store = store.RemoteSnapshotStore(concurrent_root)
        sources = {"triton/a": old, "triton/b": new}
        # Barrier both publishers into the upload window at once so neither can
        # delete a peer's in-flight per-writer .tmp; Future.result surfaces worker
        # exceptions so a thread-internal failure fails the test.
        barrier = threading.Barrier(len(sources))
        real_copyfile = shutil.copyfile

        def interleaved_copyfile(src, dst, *args, **kwargs):
            barrier.wait(timeout=10)
            return real_copyfile(src, dst, *args, **kwargs)

        with mock.patch.object(
            store.shutil, "copyfile", side_effect=interleaved_copyfile
        ), concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = [
                pool.submit(
                    concurrent_store.publish_snapshot, lambda n=name, p=path: {n: p}
                )
                for name, path in sources.items()
            ]
            for future in futures:
                future.result()
        # Both complete generations land, but restore() selects only the newest.
        self.assertEqual(
            len(list(concurrent_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))), len(sources)
        )
        restored = self.root / "concurrent_restored"
        self.assertTrue(concurrent_store.restore(restored))
        self.assertIn(contents(restored), ({"triton/a": b"old"}, {"triton/b": b"new"}))

        # Two published names sharing one inode must not become a tar hardlink
        # member — the extractor rejects non-file members on restore.
        dup_a, dup_b = self.root / "dup-a.cubin", self.root / "dup-b.cubin"
        dup_a.write_bytes(b"dup")
        os.link(dup_a, dup_b)
        dup_store = store.RemoteSnapshotStore(self.root / "dup_remote")
        dup_store.remote_root.mkdir()
        dup_store.publish_snapshot(
            lambda: {"triton/a.cubin": dup_a, "triton/b.cubin": dup_b}
        )
        dup_target = self.root / "dup_restored"
        self.assertTrue(dup_store.restore(dup_target))
        self.assertEqual(
            contents(dup_target), {"triton/a.cubin": b"dup", "triton/b.cubin": b"dup"}
        )

        ninja = shutil.which("ninja")
        if ninja:
            source = self.root / "ninja"
            source.mkdir()
            (source / "input.cu").write_text("payload")
            (source / "build.ninja").write_text(
                "rule copy\n  command = cp $in $out\nbuild output.o: copy input.cu\n"
            )
            subprocess.run([ninja, "-C", str(source)], check=True, capture_output=True)
            ninja_store = store.RemoteSnapshotStore(self.root / "ninja_remote")
            ninja_store.remote_root.mkdir()
            ninja_store.publish_snapshot(
                lambda: {f"tree/{path.name}": path for path in source.iterdir()}
            )
            restored = self.root / "ninja_restored"
            ninja_store.restore(restored)
            result = subprocess.run(
                [ninja, "-C", str(restored / "tree"), "-n"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("no work to do", result.stdout)

        self.assertIsNone(jit.resolve_remote_root(self.root / "missing"))

    def test_restore_never_leaves_ninja_seeing_stale_source(self):
        # Restore never combines a remote generation with a live local generation.
        # A warm tree belongs to its builder and remains byte-for-byte untouched.
        ninja = shutil.which("ninja")
        if not ninja:
            self.skipTest("ninja not available")

        def build_tree(where: Path, value: str) -> None:
            where.mkdir(parents=True, exist_ok=True)
            (where / "op.cu").write_text(value)
            (where / "build.ninja").write_text(
                "rule cc\n  command = cp $in $out\n  restat = 1\n"
                "build op.so: cc op.cu\ndefault op.so\n"
            )
            subprocess.run([ninja, "-C", str(where)], check=True, capture_output=True)

        def ninja_dirty(where: Path) -> bool:
            out = subprocess.run(
                [ninja, "-C", str(where), "-n"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            return "no work to do" not in out

        remote = self.root / "stale_remote"
        remote.mkdir()
        snap_store = store.RemoteSnapshotStore(remote)
        new_build = self.root / "new_build"
        build_tree(new_build, "int v = 2;")  # snapshot holds NEW source + its .so
        snap_store.publish_snapshot(
            lambda: {f"comp/{p.name}": p for p in new_build.iterdir()}
        )

        # Warm cache: an OLDER self-consistent build. restore() must keep it as-is.
        target = self.root / "live" / "v1"
        build_tree(target / "comp", "int v = 1;")
        self.assertFalse(snap_store.restore(target))
        self.assertEqual((target / "comp/op.cu").read_text(), "int v = 1;")
        self.assertEqual((target / "comp/op.so").read_text(), "int v = 1;")
        self.assertFalse(ninja_dirty(target / "comp"))  # .so still matches its source

    def test_new_generation_replaces_source_and_binary_together(self):
        remote = self.root / "generation_remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        source = self.root / "op.cu"
        binary = self.root / "op.so"
        obsolete = self.root / "obsolete.o"
        generation = {
            "triton/unit/op.cu": source,
            "triton/unit/op.so": binary,
            "triton/unit/obsolete.o": obsolete,
        }

        source.write_bytes(b"source-v1")
        binary.write_bytes(b"binary-v1")
        obsolete.write_bytes(b"obsolete-v1")
        snapshot_store.publish_snapshot(lambda: generation)

        source.write_bytes(b"source-v2")
        binary.write_bytes(b"binary-v2")
        generation.pop("triton/unit/obsolete.o")
        snapshot_store.publish_snapshot(lambda: generation)

        extra = self.root / "late.cubin"
        extra.write_bytes(b"late")
        snapshot_count = len(list(remote.glob(f"*{store.SNAPSHOT_SUFFIX}")))
        # First scan packs `generation`; the rescan after copy sees an extra file.
        scans = iter([generation, {**generation, "triton/unit/late.cubin": extra}])
        with self.assertLogs(level="WARNING") as logs:
            self.assertFalse(snapshot_store.publish_snapshot(lambda: next(scans)))
        self.assertIn("changed during snapshot", "".join(logs.output))
        self.assertEqual(
            len(list(remote.glob(f"*{store.SNAPSHOT_SUFFIX}"))), snapshot_count
        )

        restored = self.root / "generation_restored"
        self.assertTrue(snapshot_store.restore(restored))
        self.assertEqual(
            contents(restored),
            {
                "triton/unit/op.cu": b"source-v2",
                "triton/unit/op.so": b"binary-v2",
            },
        )

    def test_restore_cancel_before_swap_leaves_tree_untouched(self):
        # The timeout path and restore commit share one lock. Once the caller wins
        # that lock and starts a cold build, a fully extracted snapshot cannot swap
        # over the live target.
        remote = self.root / "remote"
        remote.mkdir()
        snap_store = store.RemoteSnapshotStore(remote)
        source = self.root / "kernel.cubin"
        source.write_bytes(b"fresh")
        snap_store.publish_snapshot(lambda: {"triton/hash/kernel.cubin": source})
        target = self.root / "live"
        target.mkdir(parents=True)
        cancel = threading.Event()
        commit_lock = threading.Lock()
        extracted = threading.Event()
        real_extract = store.extract_zstd_tar

        def extract_then_wait(snapshot, staging):
            real_extract(snapshot, staging)
            extracted.set()

        commit_lock.acquire()
        try:
            with mock.patch.object(
                store, "extract_zstd_tar", side_effect=extract_then_wait
            ), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    snap_store.restore,
                    target,
                    cancel,
                    commit_lock,
                )
                self.assertTrue(extracted.wait(timeout=10))
                cancel.set()
                (target / "cold-build.cu").write_bytes(b"local")
                commit_lock.release()
                self.assertFalse(future.result(timeout=10))
        finally:
            if commit_lock.locked():
                commit_lock.release()
        self.assertEqual(contents(target), {"cold-build.cu": b"local"})
        self.assertFalse(list(self.root.glob("*.stage*")))

    def test_tipc_warm_load_and_rebuild(self):
        source_root = self.root / "tipc"
        source_root.mkdir()
        source = source_root / "extension.cc"
        source.write_text("int f() { return 1; }\n")
        header = source_root / "extension.h"
        header.write_text("#pragma once\n")
        args = [["-O3"], None, None, False]
        signature = tipc_ffi._source_signature(source_root, args)
        clone = self.root / "clone"
        shutil.copytree(source_root, clone)
        self.assertEqual(signature, tipc_ffi._source_signature(clone, args))
        header.write_text("#pragma once\n// changed\n")
        self.assertNotEqual(signature, tipc_ffi._source_signature(source_root, args))

        build_root = self.root / "build"
        cap = (8, 0)
        build_dir = tipc_build_dir(build_root, cap)
        build_dir.mkdir(parents=True)
        so = build_dir / "tipc.so"
        so.write_bytes(b"cached")
        (build_dir / ".load.lock").touch()  # A stale flock path must not block.
        module, spec = object(), mock.Mock()
        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
        ), mock.patch.object(
            tipc_ffi.torch.cuda, "get_device_capability", return_value=cap
        ), mock.patch.object(
            tipc_ffi.importlib.util, "spec_from_file_location", return_value=spec
        ), mock.patch.object(
            tipc_ffi.importlib.util, "module_from_spec", return_value=module
        ), mock.patch.object(
            tipc_ffi, "load"
        ) as load:
            self.assertIs(tipc_ffi.__CompileHelper__().compile(), module)
            load.assert_not_called()

        # A different device capability resolves to a different build dir, so the
        # cached .so for the old arch is never reused: load() must run and the new
        # dir starts empty.
        new_cap = (9, 0)
        self.assertNotEqual(tipc_build_dir(build_root, new_cap), build_dir)
        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
        ), mock.patch.object(
            tipc_ffi.torch.cuda, "get_device_capability", return_value=new_cap
        ), mock.patch.object(
            tipc_ffi, "load", return_value="rebuilt-for-new-arch"
        ) as load:
            self.assertEqual(
                tipc_ffi.__CompileHelper__().compile(), "rebuilt-for-new-arch"
            )
            load.assert_called_once()

        def rebuild(*_args, **_kwargs):
            self.assertFalse(so.exists())
            return "rebuilt"

        so.write_bytes(b"broken")
        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
        ), mock.patch.object(
            tipc_ffi.torch.cuda, "get_device_capability", return_value=cap
        ), mock.patch.object(
            tipc_ffi.importlib.util, "module_from_spec", side_effect=ImportError
        ), mock.patch.object(
            tipc_ffi, "load", side_effect=rebuild
        ), self.assertLogs(
            level="WARNING"
        ):
            self.assertEqual(tipc_ffi.__CompileHelper__().compile(), "rebuilt")

    def test_tipc_stale_baton_lock_reclaimed(self):
        # A builder SIGKILL'd inside torch's load() leaves a bare `lock` FileBaton
        # in build_dir; without reclaiming it, the next load() waits on it forever.
        build_root = self.root / "build"
        build_dir = tipc_build_dir(build_root, (8, 0))
        build_dir.mkdir(parents=True)
        (build_dir / "lock").touch()  # FileBaton corpse; no tipc.so -> must rebuild

        sentinel = object()

        def rebuild(*_args, **_kwargs):
            self.assertFalse(
                (build_dir / "lock").exists(),
                "stale baton lock must be cleared before load()",
            )
            return sentinel

        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
        ), mock.patch.object(
            tipc_ffi.torch.cuda, "get_device_capability", return_value=(8, 0)
        ), mock.patch.object(
            tipc_ffi, "load", side_effect=rebuild
        ) as load:
            self.assertIs(tipc_ffi.__CompileHelper__().compile(), sentinel)
            load.assert_called_once()
        self.assertFalse((build_dir / "lock").exists())

    def test_runtime_fingerprint_scopes_torch_extensions_dir(self):
        # TIPC keys its build dir on TORCH_EXTENSIONS_DIR; the manager folds the
        # runtime fingerprint into that path, so a drift in GPU arch, CUDA/Torch
        # version or C++ ABI yields a distinct scope and a stale-ABI extension is
        # never reused. An undetectable cpp scope disables reuse (None) instead.
        base = jit._torch_scope("cuda-12_9-sm_90", "cxx-fixed")
        # GPU arch and CUDA version both move the scope.
        for accelerator in ("cuda-12_9-sm_80", "cuda-12_6-sm_90"):
            self.assertNotEqual(base, jit._torch_scope(accelerator, "cxx-fixed"))
        with mock.patch("torch.__version__", "0.0.0+fingerprint"):
            self.assertNotEqual(base, jit._torch_scope("cuda-12_9-sm_90", "cxx-fixed"))
        self.assertIsNone(jit._torch_scope("cuda-12_9-sm_90", None))

        # The load-bearing seam: setup_jit_cache_env exports the fingerprint-scoped
        # dir into TORCH_EXTENSIONS_DIR, which is what TIPC keys its build dir on
        # (that keying is covered by test_tipc_warm_load_and_rebuild).
        with _fake_scopes(self.root / "local"):
            managed, _ = jit.setup_jit_cache_env()
        torch_ext = component(managed, "torch_extensions")
        self.assertEqual(os.environ["TORCH_EXTENSIONS_DIR"], str(torch_ext.local_dir))
        self.assertIn("torch-test", os.environ["TORCH_EXTENSIONS_DIR"])

    def test_preset_component_dir_excluded_and_warns(self):
        # A producer pinned to a preset dir is neither redirected nor observed: it
        # drops out of `managed` and the operator is warned it won't sync.
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "preset_triton")
        with _fake_scopes(self.root / "local"), self.assertLogs(
            level="WARNING"
        ) as logs:
            managed, _ = jit.setup_jit_cache_env()
        self.assertNotIn("triton", {item.name for item in managed})
        self.assertTrue(
            any("triton" in line and "TRITON_CACHE_DIR" in line for line in logs.output)
        )

    def test_restore_replaces_cold_leftovers_as_one_tree(self):
        # Empty directories and lock corpses do not make a cache warm. The first
        # owner can replace those leftovers with one complete remote generation.
        remote = self.root / "remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        source = self.root / "kernel.cubin"
        source.write_bytes(b"real")
        snapshot_store.publish_snapshot(lambda: {"triton/hash/kernel.cubin": source})

        target = self.root / "leftovers"
        (target / "triton" / "empty-scope").mkdir(parents=True)  # empty component dir
        (target / "triton" / "lock").write_bytes(b"")  # torch FileBaton corpse
        (target / "triton" / "build_lock").write_bytes(b"")  # our builder lock
        (target / "triton" / "x.lock").write_bytes(b"")
        (target / store.MTIME_MANIFEST).write_bytes(b"{}")

        self.assertTrue(snapshot_store.restore(target))
        self.assertEqual(contents(target), {"triton/hash/kernel.cubin": b"real"})
        self.assertFalse(list(self.root.glob("*.stage*")))

    def test_restore_serializes_concurrent_cold_restore(self):
        # Two co-located owners serialize on the restore flock. Exactly one commits
        # the complete tree; the other observes the claimed tree and reuses it.
        remote = self.root / "remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        source = self.root / "kernel.cubin"
        source.write_bytes(b"real")
        snapshot_store.publish_snapshot(lambda: {"triton/hash/kernel.cubin": source})

        target = self.root / "shared" / "v1"
        barrier = threading.Barrier(2)

        def racer():
            barrier.wait(timeout=10)
            return snapshot_store.restore(target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = sorted(
                f.result() for f in (pool.submit(racer), pool.submit(racer))
            )
        self.assertEqual(results, [False, True])  # exactly one cold restore ran
        self.assertEqual(contents(target), {"triton/hash/kernel.cubin": b"real"})

    def test_restore_ready_marker_claims_shared_tree(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        target = self.root / "shared" / "v1"

        # The first service claims an empty remote before its builder starts.
        self.assertFalse(snapshot_store.restore(target))
        (target / "triton").mkdir(parents=True)
        (target / "triton" / "lock").write_bytes(b"")

        # A snapshot appearing now must not replace the claimed, artifact-free tree.
        source = self.root / "kernel.cubin"
        source.write_bytes(b"from-remote")
        snapshot_store.publish_snapshot(lambda: {"triton/hash/kernel.cubin": source})
        with mock.patch.object(
            store, "extract_zstd_tar", side_effect=AssertionError("re-extracted")
        ):
            self.assertFalse(snapshot_store.restore(target))
        self.assertEqual(contents(target), {"triton/lock": b""})

        # An explicit cache reset removes both the tree and its lifecycle marker.
        shutil.rmtree(target)
        target.with_name("v1.ready").unlink()
        with store.restore_lock(target):
            self.assertTrue(snapshot_store.restore(target, commit=threading.Lock()))
        self.assertEqual(contents(target), {"triton/hash/kernel.cubin": b"from-remote"})

    def test_stale_lock_cleanup_walk_error_is_fail_open(self):
        # An unreadable local cache tree (rglob raising mid-walk) must not break
        # setup: JIT cache is fail-open, so cleanup is skipped and setup still
        # returns the managed components.
        with _fake_scopes(self.root / "local"), mock.patch.object(
            jit.Path, "rglob", side_effect=OSError("unreadable cache tree")
        ):
            jit.clear_jit_locks()  # swallows the walk error, must not raise
            managed, compatible = jit.setup_jit_cache_env()
        self.assertTrue(compatible)
        self.assertTrue(managed)

    def test_clear_jit_locks_reaps_dead_but_spares_live_baton(self):
        # torch's FileBaton lock is a bare "lock"; a SIGKILL'd builder leaves it and
        # FileBaton.wait() spins on os.path.exists forever, hanging the next load(). Our
        # *_lock/.lock are flock-probed, so an unheld one is a corpse reaped at once. The
        # bare baton can't be probed, so on a shared tree a FRESH one may be a live
        # co-located builder's and is spared; only one idle past a full compile is reaped.
        # Real artifacts always survive.
        local = self.root / "jit"
        (local / "a").mkdir(parents=True)
        (local / "b").mkdir(parents=True)
        stale_baton = local / "a" / "lock"  # torch FileBaton corpse, aged out
        stale_baton.write_bytes(b"")
        old = time.time() - store.STALE_BATON_S - 1
        os.utime(stale_baton, (old, old))
        fresh_baton = local / "b" / "lock"  # a live co-located builder's baton
        fresh_baton.write_bytes(b"")
        our_lock = local / "a" / "build_lock"  # our unheld builder lock -> corpse
        our_lock.write_bytes(b"")
        dot_lock = local / "a" / "x.lock"
        dot_lock.write_bytes(b"")
        artifact = local / "a" / "kernel.so"  # never a lock, must survive
        artifact.write_bytes(b"real")
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)):
            jit.clear_jit_locks()
        self.assertFalse(stale_baton.exists())  # aged-out corpse reaped
        self.assertTrue(fresh_baton.exists())  # possibly-live baton spared
        self.assertFalse(our_lock.exists())  # unheld flock'd lock reaped
        self.assertFalse(dot_lock.exists())
        self.assertTrue(artifact.exists())

    def test_clear_jit_locks_preserves_live_builder_lock(self):
        # A co-located instance (shared /tmp) may hold an flock on its builder lock
        # while another instance starts up. The startup sweep must reap unheld corpses
        # but leave a live builder's flock'd lock alone, or it would enable a double
        # build and a corrupt tree.
        local = self.root / "jit"
        (local / "a").mkdir(parents=True)
        dead = local / "a" / "dead.lock"  # nobody holds it -> corpse
        dead.write_bytes(b"")
        live = local / "a" / "build.lock"  # a live builder's flock
        live.write_bytes(b"")
        fd = os.open(live, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)):
                jit.clear_jit_locks()
            self.assertFalse(dead.exists())  # unheld corpse reaped
            self.assertTrue(live.exists())  # live builder's lock preserved
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_cpp_runtime_scope_does_not_cache_transient_failure(self):
        # A transient probe failure (e.g. TimeoutExpired when the 5s `c++` probe
        # loses a CPU-contended startup race) must not disable the remote JIT cache:
        # the scope is never memoized, so the next call re-probes and succeeds.
        outputs = [
            subprocess.TimeoutExpired(cmd="c++", timeout=5),  # first probe times out
            "g++ (test) 13.0\n",  # retry: --version
            "/usr/lib/libstdc++.so.6\n",  # retry: libstdc++ path
        ]
        with mock.patch.object(
            jit.subprocess, "check_output", side_effect=outputs
        ), mock.patch.object(jit.Path, "is_file", return_value=True), mock.patch.object(
            jit.Path, "read_bytes", return_value=b"solib"
        ):
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(jit._cpp_runtime_scope())  # not cached
            scope = jit._cpp_runtime_scope()  # re-probe succeeds
        self.assertIsNotNone(scope)
        self.assertTrue(scope.startswith("cxx-"))


if __name__ == "__main__":
    unittest.main()

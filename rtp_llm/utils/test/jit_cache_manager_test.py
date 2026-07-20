import concurrent.futures
import contextlib
import multiprocessing
import os
import shutil
import signal
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
    # Stub the scope probes (and optionally LOCAL_JIT_DIR) so tests don't shell out
    # to a compiler or query a GPU. Returns an ExitStack usable as a `with`.
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(jit, "_backend", return_value=jit.CUDA))
    stack.enter_context(
        mock.patch.object(jit, "_accelerator_scope", return_value="cuda-test")
    )
    stack.enter_context(
        mock.patch.object(jit, "_torch_scope", return_value="torch-test")
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


def peer_setup(event, result):
    # Peer waits JIT_CACHE_SETUP_TIMEOUT_S + 5; -4.9 → a 0.1s wait so the
    # not-ready case still returns in well under the asserted 1s.
    backend.JIT_CACHE_SETUP_TIMEOUT_S = -4.9
    with mock.patch.object(jit, "setup_jit_cache_env", return_value=((), True)):
        started = time.monotonic()
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
        local, peer, ready = mock.Mock(), mock.Mock(), mock.Mock()
        peer.wait.return_value = False
        manager = mock.Mock()
        manager.start_background_sync.side_effect = backend._jit_cache_timeout_handler
        previous = object()
        with mock.patch.object(
            jit, "setup_jit_cache_env", return_value=((object(),), True)
        ) as setup, mock.patch.object(
            jit, "JitCacheManager", return_value=manager
        ) as manager_cls, mock.patch.object(
            jit, "resolve_remote_root", return_value=None
        ):
            self.assertIsNone(backend._setup_jit_cache("", 1, local))
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(backend._setup_jit_cache("/remote", 1, peer))
            with mock.patch.object(
                backend.signal, "signal", return_value=previous
            ) as set_signal, mock.patch.object(
                backend.signal, "alarm"
            ) as alarm, self.assertLogs(
                level="ERROR"
            ):
                # Setup failure stops the half-started manager and returns None.
                self.assertIsNone(backend._setup_jit_cache("/remote", 0, ready))
            manager.start_background_sync.side_effect = SystemExit(143)
            with mock.patch.object(
                backend.signal, "signal", return_value=previous
            ), mock.patch.object(backend.signal, "alarm"), self.assertRaises(
                SystemExit
            ):
                backend._setup_jit_cache("/remote", 0, ready)
        self.assertEqual(setup.call_count, 4)  # local env is set up on every path
        self.assertEqual(manager_cls.call_count, 2)
        self.assertEqual(manager.stop.call_count, 2)  # timeout + SystemExit paths
        local.wait.assert_not_called()
        peer.wait.assert_called_once_with(timeout=backend.JIT_CACHE_SETUP_TIMEOUT_S + 5)
        self.assertEqual(ready.set.call_count, 2)
        self.assertEqual(
            alarm.call_args_list,
            [mock.call(backend.JIT_CACHE_SETUP_TIMEOUT_S), mock.call(0)],
        )
        self.assertEqual(
            set_signal.call_args_list[-1], mock.call(signal.SIGALRM, previous)
        )

        ctx = multiprocessing.get_context("spawn")
        elapsed_times = []
        for is_ready in (True, False):
            event, result = ctx.Event(), ctx.Queue()
            if is_ready:
                event.set()
            process = ctx.Process(target=peer_setup, args=(event, result))
            process.start()
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
        with _fake_scopes():
            resolved = jit._resolve_components(root)
            managed, _ = jit.setup_jit_cache_env()
            with mock.patch.object(jit, "_pkg_version", return_value=None):
                versionless = jit._resolve_components(root)
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
        tilelang = component(resolved, "tilelang")
        self.assertFalse(
            tilelang.should_sync("0.1.9/.staging/hash/device_kernel.cu", "closed")
        )
        self.assertTrue(tilelang.should_sync("0.1.9/hash/device_kernel.cu", "closed"))

        for item in jit.COMPONENTS:
            os.environ.pop(item.env_name, None)
        event_manager = self.make_manager()
        with mock.patch.object(jit, "Observer", return_value=mock.Mock()):
            event_manager.start_background_sync()
            triton = component(event_manager.components, "triton")
            path = triton.local_dir / "hash/final.cubin"
            path.parent.mkdir(parents=True)
            tilelang = component(event_manager.components, "tilelang")
            tilelang_path = tilelang.local_dir / "0.1.9/hash/device_kernel.cu"
            tilelang_path.parent.mkdir(parents=True)
            tilelang_path.write_bytes(b"stable-directory-move")
            handler = jit._EventHandler(triton, event_manager)
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
        self.assertEqual(
            (
                restored / tilelang_path.relative_to(event_manager.local_root)
            ).read_bytes(),
            b"stable-directory-move",
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
        ), mock.patch.object(idle, "flush_delta2remote") as flush:
            idle._sync_loop()
        flush.assert_called_once()  # final flush only; recent events are coalesced

    def test_snapshot_lifecycle(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot_store = store.RemoteSnapshotStore(remote)
        expected = {}
        with mock.patch.object(store, "SNAPSHOT_KEEP", 2):
            for index in range(3):
                source = self.root / f"kernel-{index}.cubin"
                source.write_bytes(f"value-{index}".encode())
                name = f"triton/hash/kernel-{index}.cubin"
                expected[name] = source.read_bytes()
                snapshot_store.publish_snapshot({name: source})
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
            snapshot_store.publish_snapshot({name: source})
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
        live = self.root / "live"
        live.mkdir()
        (live / "existing").write_bytes(b"existing")

        def partial_extract(_snapshot, staging):
            (staging / "partial").write_bytes(b"partial")
            raise RuntimeError("broken")

        with mock.patch.object(
            store, "extract_zstd_tar", side_effect=partial_extract
        ), self.assertLogs(level="WARNING"):
            self.assertFalse(store.RemoteSnapshotStore(broken).restore(live))
        self.assertEqual(contents(live), {"existing": b"existing"})
        with mock.patch.object(store, "extract_zstd_tar", side_effect=TimeoutError):
            self.assertRaises(
                TimeoutError, store.RemoteSnapshotStore(broken).restore, live
            )

        old = self.root / "old"
        old.write_bytes(b"old")
        new = self.root / "new"
        new.write_bytes(b"new")

        # A merge that dies midway must leave existing files whole: os.replace
        # swaps atomically, so a healthy prior cache is never truncated.
        merge = self.root / "merge_remote"
        merge.mkdir()
        merge_store = store.RemoteSnapshotStore(merge)
        merge_store.publish_snapshot(
            {"triton/keep": old, "triton/late": new},
        )
        merged = self.root / "merge_live"
        merged.mkdir()
        (merged / "triton").mkdir()
        (merged / "triton/keep").write_bytes(b"healthy")
        real_copy2 = shutil.copy2

        def copy2_then_fail(src, dst, *args, **kwargs):
            if Path(src).name == "late":
                raise OSError("disk full mid-merge")
            return real_copy2(src, dst, *args, **kwargs)

        with mock.patch.object(store.shutil, "copy2", side_effect=copy2_then_fail):
            self.assertRaises(OSError, merge_store.restore, merged)
        # "keep" is a whole value (healthy or snapshot bytes), never a truncated
        # mix; no .tmp left over.
        self.assertIn((merged / "triton/keep").read_bytes(), (b"healthy", b"old"))
        self.assertFalse(list(merged.rglob("*.tmp")))

        healthy = self.root / "healthy"
        healthy.mkdir()
        healthy_store = store.RemoteSnapshotStore(healthy)
        healthy_store.publish_snapshot({"triton/old": old})
        vanished = self.root / "vanished"
        healthy_store.publish_snapshot({"triton/old": vanished})
        restored = self.root / "restored_after_vanished_source"
        self.assertTrue(healthy_store.restore(restored))
        self.assertEqual(contents(restored), {"triton/old": b"old"})
        (healthy / f"{'9' * 20}-bad{store.SNAPSHOT_SUFFIX}").write_bytes(b"bad")
        with self.assertLogs(level="WARNING"):
            healthy_store.publish_snapshot({"triton/new": new})
            restored = self.root / "restored"
            healthy_store.restore(restored)
        self.assertEqual(
            contents(restored), {"triton/old": b"old", "triton/new": b"new"}
        )
        # A fresh temp upload (possibly a peer's in-flight one) is never reclaimed.
        (healthy / ".upload.live.tmp").write_bytes(b"orphan")
        healthy_store.publish_snapshot({"triton/new": new})
        self.assertEqual((healthy / ".upload.live.tmp").read_bytes(), b"orphan")
        # A long-dead one (mtime past IDLE_REAP_S) is reclaimed next publish.
        dead = healthy / ".upload.dead.tmp"
        dead.write_bytes(b"orphan")
        stale = time.time() - store.IDLE_REAP_S - 1
        os.utime(dead, (stale, stale))
        healthy_store.publish_snapshot({"triton/new": new})
        self.assertFalse(dead.exists())

        concurrent_root = self.root / "concurrent"
        concurrent_root.mkdir()
        concurrent_store = store.RemoteSnapshotStore(concurrent_root)
        sources = {"triton/a": old, "triton/b": new}
        # Barrier both publishers into the upload window at once so neither can
        # delete a peer's in-flight .upload.<token>.tmp; Future.result surfaces
        # worker exceptions so a thread-internal failure fails the test.
        barrier = threading.Barrier(len(sources))
        real_copyfile = shutil.copyfile

        def interleaved_copyfile(src, dst, *args, **kwargs):
            barrier.wait(timeout=10)
            return real_copyfile(src, dst, *args, **kwargs)

        with mock.patch.object(
            store.shutil, "copyfile", side_effect=interleaved_copyfile
        ), concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = [
                pool.submit(concurrent_store.publish_snapshot, {name: path})
                for name, path in sources.items()
            ]
            for future in futures:
                future.result()
        # Both commits land, but restore() is newest-only by design: a concurrent
        # fork drops one publisher's delta (hit-rate cache self-heals on the next
        # miss via recompile+republish). Assert that real contract, not a union of
        # archives that would falsely imply both deltas survive.
        self.assertEqual(
            len(list(concurrent_root.glob(f"*{store.SNAPSHOT_SUFFIX}"))), len(sources)
        )
        restored = self.root / "concurrent_restored"
        self.assertTrue(concurrent_store.restore(restored))
        self.assertIn(contents(restored), ({"triton/a": b"old"}, {"triton/b": b"new"}))

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
                {f"tree/{path.name}": path for path in source.iterdir()}
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
        mounted = self.root / "mounted"
        mounted.mkdir()
        with mock.patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local", return_value=str(mounted)
        ) as fetch:
            self.assertEqual(
                jit.resolve_remote_root("oss://bucket/cache"), mounted / "v1"
            )
        self.assertEqual(fetch.call_args.args[0], "oss://bucket/cache")

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
        source_root = Path(tipc_ffi.__file__).with_name("csrc")
        build_dir = (
            build_root
            / "tipc"
            / tipc_ffi._source_signature(
                source_root, [["-O3"], ["-O3", "-use_fast_math"], None, True]
            )
        )
        build_dir.mkdir(parents=True)
        so = build_dir / "tipc.so"
        so.write_bytes(b"cached")
        (build_dir / ".load.lock").touch()  # A stale flock path must not block.
        module, spec = object(), mock.Mock()
        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
        ), mock.patch.object(
            tipc_ffi.importlib.util, "spec_from_file_location", return_value=spec
        ), mock.patch.object(
            tipc_ffi.importlib.util, "module_from_spec", return_value=module
        ), mock.patch.object(
            tipc_ffi, "load"
        ) as load:
            self.assertIs(tipc_ffi.__CompileHelper__().compile(), module)
            load.assert_not_called()

        def rebuild(*_args, **_kwargs):
            self.assertFalse(so.exists())
            return "rebuilt"

        so.write_bytes(b"broken")
        with mock.patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(build_root)}
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
        source_root = Path(tipc_ffi.__file__).with_name("csrc")
        build_dir = (
            build_root
            / "tipc"
            / tipc_ffi._source_signature(
                source_root, [["-O3"], ["-O3", "-use_fast_math"], None, True]
            )
        )
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
        ), mock.patch.object(tipc_ffi, "load", side_effect=rebuild) as load:
            self.assertIs(tipc_ffi.__CompileHelper__().compile(), sentinel)
            load.assert_called_once()
        self.assertFalse((build_dir / "lock").exists())


if __name__ == "__main__":
    unittest.main()

import concurrent.futures
import copy
import errno
import hashlib
import io
import json
import multiprocessing
import os
import tarfile
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import zstandard as zstd

from rtp_llm.utils import jit_cache_deep_gemm as dg
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store

SCOPE = "deep_gemm-cuda-13_0-sm_100-2_8_0-deepjit-v1-" + "a" * 64


def build_manifest():
    return {
        "schema_version": 1,
        "cache_entry_format": "deepjit-v1",
        "sources": {
            "deepgemm": "66081d4c9c7d7c44f13fea402e5b622aa0f409c2",
            "deepjit": "e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16",
            "cutlass": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
        },
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
        "build": {
            "python_soabi": "cpython-310-aarch64-linux-gnu",
            "torch_version": "2.11.0+cu130",
            "cxx11_abi": True,
            "host_arch": "aarch64",
            "cuda_version": "13.2",
            "nvcc_version": "13.2.0",
            "host_compiler": "gcc 13.2.1",
            "libstdcxx_sha256": "b" * 64,
            "target_archs": ["sm_100a"],
            "flags": ["-std=c++20"],
        },
    }


def entry_payloads():
    return {
        "kernel.cu": b'extern "C" __global__ void test() {}',
        "kernel.cubin": b"unit-test-payload-not-a-loadability-test",
        "meta.json": b'{"name":"test","compiler_info":{},"compiler_options":{}}',
        ".committed": b"",
    }


def write_entry(entry, files=None):
    entry.mkdir(parents=True, exist_ok=True)
    for name, payload in (entry_payloads() if files is None else files).items():
        (entry / name).write_bytes(payload)
    return entry


def tree_contents(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _restore_process(remote, target, start, results):
    results.put(("ready", os.getpid(), None))
    if not start.wait(15):
        raise TimeoutError("restore processes did not start together")
    restored = store.RemoteSnapshotStore(Path(remote)).restore(Path(target))
    results.put(("restored", os.getpid(), restored))


def _hold_restore_lock_process(target, acquired, release):
    with store.restore_lock(Path(target)):
        acquired.set()
        if not release.wait(30):
            raise TimeoutError("restore lock holder was not released")


class DeepGemmCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manager(self):
        local, remote = self.root / "local", self.root / "remote"
        remote.mkdir(exist_ok=True)
        component = replace(
            next(item for item in jit.COMPONENTS if item.name == "deep_gemm"),
            local_dir=local / "deep_gemm" / SCOPE,
        )
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)):
            manager = jit.JitCacheManager(remote, (component,))
        self.addCleanup(manager.stop)
        return manager, component.local_dir

    def identity(self, manifest=None, version="2.8.0+local", runtime="runtime-a"):
        path = self.root / "manifest.json"
        if manifest is not None:
            path.write_text(json.dumps(manifest), encoding="utf-8")
        distribution = SimpleNamespace(version=version, locate_file=lambda _: path)
        with mock.patch.object(
            dg.importlib.metadata, "distribution", return_value=distribution
        ):
            return dg.deep_gemm_build_scope(runtime)

    def test_old_wheel_scope_stays_unchanged(self):
        self.assertIsNone(self.identity(version="2.6.1+09cd3ee.cu132"))

    def test_source_patch_toolchain_and_runtime_identities_do_not_collide(self):
        original = build_manifest()
        baseline = self.identity(original)
        self.assertTrue(dg.is_deepjit_scope(f"deep_gemm-test-{baseline}"))
        reordered = dict(reversed(list(original.items())))
        self.assertEqual(self.identity(reordered), baseline)
        variants = []
        for name in original["sources"]:
            variant = copy.deepcopy(original)
            variant["sources"][name] = "0" * 40
            variants.append(variant)
        variant = copy.deepcopy(original)
        variant["patch_sha256"] = "0" * 64
        variants.append(variant)
        for name, value in (
            ("libstdcxx_sha256", "0" * 64),
            ("flags", ["-std=c++20", "--use_fast_math"]),
            ("target_archs", ["sm_100f"]),
            ("nvcc_version", "13.3.0"),
            ("host_arch", "x86_64"),
            ("cxx11_abi", False),
        ):
            variant = copy.deepcopy(original)
            variant["build"][name] = value
            variants.append(variant)
        observed = {baseline}
        for variant in variants:
            with self.subTest(manifest=variant):
                scope = self.identity(variant)
                self.assertNotIn(scope, observed)
                observed.add(scope)
        self.assertNotEqual(self.identity(original, runtime="runtime-b"), baseline)
        with mock.patch.dict(os.environ, {"NVCC_PREPEND_FLAGS": "-lineinfo"}):
            self.assertNotEqual(self.identity(original), baseline)

    def test_missing_or_invalid_new_build_identity_does_not_fail_open(self):
        with self.assertRaises(dg.DeepGemmBuildIdentityError):
            self.identity()
        for field in ("sources", "patch_sha256", "build", "cache_entry_format"):
            manifest = build_manifest()
            del manifest[field]
            with self.subTest(field=field), self.assertRaises(
                dg.DeepGemmBuildIdentityError
            ):
                self.identity(manifest)
        with self.assertRaises(dg.DeepGemmBuildIdentityError):
            self.identity(build_manifest(), runtime=None)
        with mock.patch.object(
            jit,
            "_resolve_components",
            side_effect=dg.DeepGemmBuildIdentityError("bad pin"),
        ), self.assertRaises(dg.DeepGemmBuildIdentityError):
            jit.setup_jit_cache_env()
        with mock.patch.object(
            jit, "_resolve_components", side_effect=OSError("transient old component")
        ), self.assertLogs(level="ERROR"):
            self.assertEqual(jit.setup_jit_cache_env(), ((), False))

    def test_new_entries_are_collected_whole_and_legacy_rules_are_preserved(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        (entry / "kernel.sass").write_bytes(b"debug code")
        partial = write_entry(scope / "cache/partial.123")
        (partial / "meta.json").unlink()
        write_entry(scope / "tmp/inflight")
        old = manager.local_root / "deep_gemm/deep_gemm-old/cache/kernel.123"
        write_entry(old)
        other = manager.local_root / "triton/triton-old/hash"
        other.mkdir(parents=True)
        (other / "kernel.cubin").write_bytes(b"old triton")
        (other / "empty.cubin").touch()
        files = manager._snapshot_files()
        expected = {
            path.relative_to(manager.local_root).as_posix() for path in entry.iterdir()
        }
        expected.update(
            (path.relative_to(manager.local_root).as_posix())
            for path in (
                old / "kernel.cu",
                old / "kernel.cubin",
                other / "kernel.cubin",
            )
        )
        self.assertEqual(set(files), expected)
        self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        target = self.root / "restored"
        self.assertTrue(manager.store.restore(target))
        self.assertEqual(
            tree_contents(target),
            {name: path.read_bytes() for name, path in files.items()},
        )

    def test_directory_commit_event_and_marker_touch(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        handler = jit._EventHandler(manager)
        handler.on_any_event(
            SimpleNamespace(
                event_type="moved",
                src_path=str(scope / "tmp/source"),
                dest_path=str(entry),
                is_directory=True,
            )
        )
        self.assertTrue(manager._dirty.is_set())
        manager.publish_pending_snapshot()
        self.assertFalse(manager._dirty.is_set())
        marker = entry / ".committed"
        os.utime(marker, ns=(100000000000, 100000000000))
        handler.on_any_event(
            SimpleNamespace(
                event_type="modified", src_path=str(marker), is_directory=False
            )
        )
        self.assertFalse(manager._dirty.is_set())
        self.assertEqual(len(manager.store._snapshots()), 1)

    def test_marker_first_does_not_publish_an_incomplete_entry(self):
        manager, scope = self.manager()
        entry = scope / "cache/op.123"
        entry.mkdir(parents=True)
        handler = jit._EventHandler(manager)
        files = entry_payloads()
        for index, name in enumerate(
            (".committed", "kernel.cu", "kernel.cubin", "meta.json")
        ):
            path = entry / name
            path.write_bytes(files[name])
            handler.on_any_event(
                SimpleNamespace(
                    event_type="closed", src_path=str(path), is_directory=False
                )
            )
            self.assertEqual(manager._dirty.is_set(), index == 3)
        self.assertEqual(len(manager._snapshot_files()), 4)

    def test_real_watcher_observes_whole_directory_rename(self):
        manager, scope = self.manager()
        with mock.patch.object(jit, "SYNC_POLL_S", 0.025):
            manager.start_background_sync()
            source = write_entry(scope / "tmp/build-uuid")
            target = scope / "cache/op.123"
            target.parent.mkdir()
            source.rename(target)
            deadline = time.monotonic() + 3
            while not manager.store._snapshots() and time.monotonic() < deadline:
                time.sleep(0.02)
            manager.stop()
        self.assertTrue(manager.store._snapshots())
        restored = self.root / "watcher-restored"
        self.assertTrue(manager.store.restore(restored))
        self.assertEqual(tree_contents(restored), tree_contents(manager.local_root))

    def test_marker_access_during_pack_does_not_starve_snapshot(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        original = store.pack_zstd_tar

        def pack_and_touch(archive, staging):
            original(archive, staging)
            marker = entry / ".committed"
            os.utime(marker, ns=(100000000000, 100000000000))

        with mock.patch.object(store, "pack_zstd_tar", side_effect=pack_and_touch):
            self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        self.assertEqual(len(manager.store._snapshots()), 1)

    def test_payload_change_during_pack_defers_the_complete_generation(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        original = store.pack_zstd_tar

        def pack_and_change(archive, staging):
            original(archive, staging)
            (entry / "kernel.cubin").write_bytes(b"recompiled")

        with mock.patch.object(
            store, "pack_zstd_tar", side_effect=pack_and_change
        ), self.assertLogs(level="WARNING"):
            self.assertFalse(manager.store.publish_snapshot(manager._snapshot_files))
        self.assertFalse(manager.store._snapshots())

    def write_archive(self, path, files):
        with zstd.open(path, "wb") as body, tarfile.open(
            fileobj=body, mode="w|"
        ) as tar:
            for name, payload in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    def test_bad_newest_entries_fall_back_without_exposing_marker(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        payloads = tree_contents(manager.local_root)
        prefix = entry.relative_to(manager.local_root).as_posix()
        checksums = {
            name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
        }
        manifest = json.dumps({"schema_version": 1, "files": checksums}).encode()
        variants = []
        for name in (".committed", "kernel.cu", "kernel.cubin", "meta.json"):
            files = dict(payloads)
            del files[f"{prefix}/{name}"]
            variants.append((f"missing-{name}", files))
        for name, value in (
            (".committed", b"unexpected"),
            ("meta.json", b"broken json"),
            ("meta.json", b"[]"),
            ("kernel.cubin", payloads[f"{prefix}/kernel.cubin"][:3]),
        ):
            files = dict(payloads)
            files[f"{prefix}/{name}"] = value
            variants.append((f"broken-{name}", files))
        variants.append(("missing-checksums", dict(payloads)))
        bad = manager.store.remote_root / f"{'9' * 20}-bad{store.SNAPSHOT_SUFFIX}"
        for index, (reason, files) in enumerate(variants):
            if reason != "missing-checksums":
                files[dg.SNAPSHOT_MANIFEST] = manifest
            self.write_archive(bad, files)
            target = self.root / f"restored-{index}"
            with self.subTest(reason=reason), self.assertLogs(level="WARNING"):
                self.assertTrue(manager.store.restore(target))
            self.assertEqual(tree_contents(target), payloads)
            self.assertFalse(list(self.root.glob("*.stage.*")))

    def test_new_tmp_and_incomplete_entries_do_not_claim_a_warm_tree(self):
        manager, scope = self.manager()
        write_entry(scope / "tmp/build-uuid")
        incomplete = write_entry(scope / "cache/op.123")
        (incomplete / ".committed").unlink()
        self.assertFalse(store._tree_is_warm(manager.local_root))
        other = self.root / "complete"
        write_entry(other / "deep_gemm" / SCOPE / "cache/op.123")
        manager.store.publish_snapshot(lambda: dg.deepjit_snapshot_files(other))
        self.assertTrue(manager.store.restore(manager.local_root))
        self.assertEqual(tree_contents(manager.local_root), tree_contents(other))

    def test_concurrent_cold_restores_publish_one_complete_tree(self):
        manager, scope = self.manager()
        write_entry(scope / "cache/op.123")
        manager.store.publish_snapshot(manager._snapshot_files)
        target = self.root / "concurrent-restore"
        barrier = threading.Barrier(8)

        def restore():
            barrier.wait(timeout=5)
            return manager.store.restore(target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(restore) for _ in range(8)]
            results = [future.result(timeout=5) for future in futures]
        self.assertEqual(sum(results), 1)
        self.assertEqual(tree_contents(target), tree_contents(manager.local_root))
        self.assertFalse(list(self.root.glob("*.stage.*")))

    def test_unreadable_deepgemm_does_not_erase_other_warm_components(self):
        manager, scope = self.manager()
        scope.mkdir(parents=True)
        path = manager.local_root / "triton/old/kernel.cubin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"active")
        original = Path.iterdir

        def iterdir(directory):
            if directory == manager.local_root / "deep_gemm":
                raise PermissionError("unreadable DeepGEMM component")
            return original(directory)

        with mock.patch.object(Path, "iterdir", iterdir):
            self.assertTrue(store._tree_is_warm(manager.local_root))
            self.assertEqual(
                manager._snapshot_files(),
                {path.relative_to(manager.local_root).as_posix(): path},
            )

    def test_symlinks_in_new_entries_are_not_published(self):
        manager, scope = self.manager()
        entry = write_entry(scope / "cache/op.123")
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        (entry / "kernel.cubin").unlink()
        (entry / "kernel.cubin").symlink_to(outside)
        self.assertFalse(manager._snapshot_files())
        self.assertFalse(store._tree_is_warm(manager.local_root))

    def test_other_component_or_legacy_artifact_preserves_the_whole_warm_tree(self):
        manager, scope = self.manager()
        write_entry(scope / "tmp/build-uuid")
        for index, relative in enumerate(
            ("triton/old/kernel.cubin", "deep_gemm/deep_gemm-old/cache/kernel.cubin")
        ):
            target = self.root / f"warm-{index}"
            write_entry(target / "deep_gemm" / SCOPE / "tmp/build-uuid")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"active old artifact")
            before = tree_contents(target)
            self.assertTrue(store._tree_is_warm(target))
            self.assertFalse(manager.store.restore(target))
            self.assertEqual(tree_contents(target), before)

    def test_non_deepjit_nanosecond_mtimes_are_preserved_and_rescanned(self):
        manager, _ = self.manager()
        path = manager.local_root / "triton/old/hash/kernel.cubin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"triton")
        mtime = 1700000000123456789
        os.utime(path, ns=(mtime, mtime))
        self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        target = self.root / "mtime-restored"
        self.assertTrue(manager.store.restore(target))
        self.assertEqual(
            (target / path.relative_to(manager.local_root)).stat().st_mtime_ns, mtime
        )
        original = store.pack_zstd_tar

        def pack_and_touch(archive, staging):
            original(archive, staging)
            os.utime(path, ns=(mtime + 1, mtime + 1))

        with mock.patch.object(
            store, "pack_zstd_tar", side_effect=pack_and_touch
        ), self.assertLogs(level="WARNING"):
            self.assertFalse(manager.store.publish_snapshot(manager._snapshot_files))
        self.assertEqual(len(manager.store._snapshots()), 1)

    def test_cancel_before_swap_preserves_cold_leftovers_and_no_ready_claim(self):
        manager, scope = self.manager()
        write_entry(scope / "cache/op.123")
        manager.store.publish_snapshot(manager._snapshot_files)
        target = self.root / "cancelled"
        write_entry(target / "deep_gemm" / SCOPE / "tmp/inflight")
        before = tree_contents(target)
        cancel = threading.Event()
        original = store.extract_zstd_tar

        def extract_then_cancel(archive, staging):
            original(archive, staging)
            cancel.set()

        with mock.patch.object(
            store, "extract_zstd_tar", side_effect=extract_then_cancel
        ):
            self.assertFalse(manager.store.restore(target, cancel))
        self.assertEqual(tree_contents(target), before)
        self.assertFalse(target.with_name(f"{target.name}.ready").exists())

    def wait_for(self, condition, message, timeout=10):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                self.fail(message)
            time.sleep(0.01)

    def test_restore_error_keeps_real_watcher_and_recovers_remote_publish(self):
        manager, scope = self.manager()
        with mock.patch.object(jit, "SYNC_POLL_S", 0.025):
            with mock.patch.object(
                manager.store, "restore", side_effect=OSError("remote unavailable")
            ), self.assertLogs(level="ERROR"):
                manager.start_background_sync()
            self.assertTrue(manager._observer.is_alive())
            self.assertFalse(
                manager.local_root.with_name(
                    f"{manager.local_root.name}.ready"
                ).exists()
            )
            source = write_entry(scope / "tmp/local-build")
            target = scope / "cache/op.local-build"
            target.parent.mkdir()
            source.rename(target)
            self.wait_for(manager.store._snapshots, "local build was not published")
            manager.stop()
        restored = self.root / "after-remote-recovery"
        self.assertTrue(manager.store.restore(restored))
        self.assertEqual(tree_contents(restored), tree_contents(manager.local_root))

    def test_remote_readability_timeout_keeps_dirty_and_shutdown_retries(self):
        manager, scope = self.manager()
        copyfile = store.shutil.copyfile

        def corrupt_remote_copy(source, destination, *args, **kwargs):
            result = copyfile(source, destination, *args, **kwargs)
            path = Path(destination)
            if path.name.endswith(f"{store.SNAPSHOT_SUFFIX}.tmp"):
                payload = path.read_bytes()
                path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            return result

        with mock.patch.object(jit, "SYNC_POLL_S", 3600):
            manager.start_background_sync()
            write_entry(scope / "cache/op.readable")
            self.wait_for(manager._dirty.is_set, "completed entry was not observed")
            before = tree_contents(manager.local_root)
            with mock.patch.object(
                store.shutil, "copyfile", side_effect=corrupt_remote_copy
            ), mock.patch.object(
                store, "REMOTE_READY_TIMEOUT_S", 0.025
            ), self.assertRaises(
                TimeoutError
            ):
                manager.publish_pending_snapshot()
            self.assertTrue(manager._dirty.is_set())
            self.assertFalse(manager.store._snapshots())
            self.assertFalse(list(manager.store.remote_root.glob("*.tmp")))
            self.assertEqual(tree_contents(manager.local_root), before)
            manager.stop()
        self.assertFalse(manager._dirty.is_set())
        restored = self.root / "readability-retry"
        self.assertTrue(manager.store.restore(restored))
        self.assertEqual(tree_contents(restored), before)

    def test_disk_full_during_upload_cleans_partial_and_preserves_retry(self):
        manager, scope = self.manager()
        copyfile = store.shutil.copyfile

        def disk_full(source, destination, *args, **kwargs):
            path = Path(destination)
            if path.name.endswith(f"{store.SNAPSHOT_SUFFIX}.tmp"):
                path.write_bytes(b"partial")
                raise OSError(errno.ENOSPC, "injected full snapshot filesystem")
            return copyfile(source, destination, *args, **kwargs)

        with mock.patch.object(jit, "SYNC_POLL_S", 3600):
            manager.start_background_sync()
            write_entry(scope / "cache/op.disk-full")
            self.wait_for(manager._dirty.is_set, "completed entry was not observed")
            before = tree_contents(manager.local_root)
            with mock.patch.object(
                store.shutil, "copyfile", side_effect=disk_full
            ), self.assertRaises(OSError) as failure:
                manager.publish_pending_snapshot()
            self.assertEqual(failure.exception.errno, errno.ENOSPC)
            self.assertTrue(manager._dirty.is_set())
            self.assertFalse(manager.store._snapshots())
            self.assertFalse(list(manager.store.remote_root.glob("*.tmp")))
            self.assertEqual(tree_contents(manager.local_root), before)
            manager.stop()
        restored = self.root / "disk-full-retry"
        self.assertTrue(manager.store.restore(restored))
        self.assertEqual(tree_contents(restored), before)

    def test_shutdown_is_bounded_when_remote_upload_stalls(self):
        manager, scope = self.manager()
        uploading, release = threading.Event(), threading.Event()

        def stalled_readability(remote, local):
            uploading.set()
            if not release.wait(10):
                raise RuntimeError("test did not release the stalled upload")
            raise TimeoutError("injected remote visibility timeout")

        with mock.patch.object(jit, "SYNC_POLL_S", 3600), mock.patch.object(
            jit, "SHUTDOWN_TIMEOUT_S", 0.1
        ):
            manager.start_background_sync()
            write_entry(scope / "cache/op.stalled")
            self.wait_for(manager._dirty.is_set, "completed entry was not observed")
            worker = manager._sync_thread
            with mock.patch.object(
                manager.store, "_wait_remote_ready", side_effect=stalled_readability
            ), self.assertLogs(level="ERROR"):
                try:
                    manager._stop.set()
                    self.assertTrue(uploading.wait(5))
                    manager.stop()
                    self.assertTrue(worker.is_alive())
                    self.assertFalse(manager.store._snapshots())
                finally:
                    release.set()
                    worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(manager._dirty.is_set())
            self.assertFalse(list(manager.store.remote_root.glob("*.tmp")))

    def cleanup_process(self, process):
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    def test_sixteen_spawned_processes_restore_one_complete_tree(self):
        manager, scope = self.manager()
        write_entry(scope / "cache/op.shared")
        self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        target = self.root / "spawned-restore"
        context = multiprocessing.get_context("spawn")
        start, results = context.Event(), context.Queue()
        self.addCleanup(results.close)
        workers = [
            context.Process(
                target=_restore_process,
                args=(str(manager.store.remote_root), str(target), start, results),
            )
            for _ in range(16)
        ]
        for worker in workers:
            worker.start()
            self.addCleanup(self.cleanup_process, worker)
        ready = [results.get(timeout=15) for _ in workers]
        self.assertEqual({row[0] for row in ready}, {"ready"})
        self.assertEqual(len({row[1] for row in ready}), 16)
        start.set()
        restored = [results.get(timeout=15) for _ in workers]
        self.assertEqual({row[0] for row in restored}, {"restored"})
        self.assertEqual({row[1] for row in ready}, {row[1] for row in restored})
        self.assertEqual(sum(row[2] for row in restored), 1)
        for worker in workers:
            worker.join(5)
            self.assertEqual(worker.exitcode, 0)
        self.assertEqual(tree_contents(target), tree_contents(manager.local_root))
        self.assertTrue(target.with_name(f"{target.name}.ready").exists())
        self.assertFalse(list(self.root.glob("*.stage.*")))

    def test_killed_restore_lock_owner_does_not_wedge_next_process(self):
        manager, scope = self.manager()
        write_entry(scope / "cache/op.after-kill")
        self.assertTrue(manager.store.publish_snapshot(manager._snapshot_files))
        target = self.root / "killed-owner"
        context = multiprocessing.get_context("spawn")
        acquired, release = context.Event(), context.Event()
        worker = context.Process(
            target=_hold_restore_lock_process,
            args=(str(target), acquired, release),
        )
        worker.start()
        self.addCleanup(self.cleanup_process, worker)
        self.assertTrue(acquired.wait(10))
        self.assertFalse(target.with_name(f"{target.name}.ready").exists())
        worker.terminate()
        worker.join(5)
        self.assertIsNotNone(worker.exitcode)
        self.assertNotEqual(worker.exitcode, 0)
        self.assertTrue(manager.store.restore(target))
        self.assertEqual(tree_contents(target), tree_contents(manager.local_root))
        self.assertTrue(target.with_name(f"{target.name}.ready").exists())


if __name__ == "__main__":
    unittest.main()

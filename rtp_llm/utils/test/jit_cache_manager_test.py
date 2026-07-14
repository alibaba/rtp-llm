import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as jit_store
from rtp_llm.utils.test import jit_cache_smoke_test as smoke


class FakeFileEvent:
    def __init__(self, event_type: str, src: str, dest: str = ""):
        self.event_type = event_type
        self.src_path = src
        self.dest_path = dest
        self.is_directory = False


class FakeObserver:
    def __init__(self, **_kwargs):
        self.handlers = []
        self.started = False

    def schedule(self, handler, path, recursive=True):
        self.handlers.append((handler, path, recursive))

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def join(self, timeout=None):
        pass


def write_archive(path: Path, entries: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jit-archive.", dir=path.parent) as tmp:
        for name, payload in entries.items():
            source = Path(tmp) / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
        jit_store.pack_zstd_tar(path, Path(tmp))
    return path


def archive_members(path: Path) -> dict[str, bytes]:
    result = {}
    with jit_store.zstd_tar(path, "r") as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as source:
                    result[member.name] = source.read()
    return result


def snapshot_paths(remote: Path) -> list[Path]:
    return sorted(remote.glob(f"*{jit_store.SNAPSHOT_SUFFIX}"))


def write_snapshot(remote: Path, timestamp: int, entries: dict[str, bytes]) -> Path:
    name = f"{timestamp:020d}{jit_store.SNAPSHOT_SUFFIX}"
    return write_archive(remote / name, entries)


def effective_members(remote: Path) -> dict[str, bytes]:
    snapshots = snapshot_paths(remote)
    return archive_members(snapshots[-1]) if snapshots else {}


def component(components, name: str):
    return next(item for item in components if item.name == name)


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class JitCacheManagerTest(unittest.TestCase):
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

    def make_manager(
        self,
        remote: Path | str | None = None,
        *,
        create_remote: bool = True,
    ) -> jit.JitCacheManager:
        remote = self.root / "remote" if remote is None else remote
        if remote and create_remote:
            Path(remote).mkdir(parents=True, exist_ok=True)
        local = self.root / f"local_{len(self.managers)}"
        config = JITConfig()
        config.remote_jit_dir = str(remote or "")
        jit.setup_jit_cache_env(local)
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)):
            manager = jit.JitCacheManager(config)
        manager.bootstrap()
        self.managers.append(manager)
        return manager

    def write_component(
        self, manager: jit.JitCacheManager, name: str, rel: str, data: bytes
    ) -> tuple[jit.Component, Path]:
        item = component(manager.components, name)
        path = item.local_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return item, path

    def test_environment_and_component_contracts(self):
        os.environ["TRITON_CACHE_DIR"] = str(self.root / "outside")
        manager = self.make_manager()
        self.assertEqual(len(manager.components), 7)
        self.assertEqual(
            os.environ["TRITON_CACHE_DIR"],
            str(self.root / "outside"),
        )
        with mock.patch.object(jit, "Observer", FakeObserver):
            manager.start_background_sync()
        watched = {path for _handler, path, _recursive in manager._observer.handlers}
        self.assertNotIn(
            str(component(manager.components, "triton").local_dir), watched
        )
        self.assertEqual(len(watched), len(manager.components) - 1)

        flashinfer = component(jit.COMPONENTS, "flashinfer")
        torch_extensions = component(jit.COMPONENTS, "torch_extensions")
        deep_gemm = component(jit.COMPONENTS, "deep_gemm")
        trtllm = component(jit.COMPONENTS, "tensorrt_llm_deep_gemm")
        tvm_ffi = component(jit.COMPONENTS, "tvm_ffi")
        cute_dsl = component(jit.COMPONENTS, "cute_dsl")
        triton = component(jit.COMPONENTS, "triton")
        self.assertEqual(flashinfer.upload_events, frozenset({"closed", "moved"}))
        self.assertEqual(torch_extensions.upload_events, frozenset({"closed", "moved"}))
        terminal_events = frozenset({"created", "moved"})
        self.assertEqual(deep_gemm.upload_events, terminal_events)
        self.assertEqual(trtllm.upload_events, terminal_events)
        self.assertEqual(tvm_ffi.upload_events, terminal_events)
        self.assertEqual(cute_dsl.upload_events, frozenset({"moved"}))
        self.assertEqual(triton.upload_events, terminal_events)

        for rel in (
            "generated/op/kernel.cu",
            "generated/op/config.inc",
            "generated/op/fmha_v2_api.h",
            "cached_ops/op/kernel.cuda.o",
            "cached_ops/op/op.so",
        ):
            self.assertTrue(flashinfer.should_sync(rel), rel)
        for rel in (
            "cached_ops/op/build.ninja",
            "cached_ops/op/rules.ninja",
            "cached_ops/op/.ninja_log",
            "cached_ops/op/.ninja_deps",
        ):
            self.assertFalse(flashinfer.should_sync(rel), rel)
        self.assertFalse(torch_extensions.should_sync("tipc/main.cpp"))
        self.assertFalse(torch_extensions.should_sync("tipc/build.ninja"))
        self.assertFalse(torch_extensions.should_sync("tipc/.ninja_log"))
        self.assertFalse(torch_extensions.should_sync("tipc/.ninja_deps"))
        self.assertTrue(torch_extensions.should_sync("tipc/ipc.o"))
        self.assertTrue(torch_extensions.should_sync("tipc/tipc.so"))
        self.assertTrue(trtllm.should_sync("cache/nvcc_kernel.cubin"))
        self.assertFalse(trtllm.should_sync("cache/kernel.cubin"))
        self.assertTrue(tvm_ffi.should_sync("libtorch_c_dlpack_addon_torch28-cuda.so"))
        self.assertFalse(
            tvm_ffi.should_sync("libtorch_c_dlpack_addon_torch28-cuda.so.lock")
        )
        self.assertTrue(cute_dsl.should_sync("cute_dsl_0123456789abcdef.mlir"))
        self.assertFalse(
            cute_dsl.should_sync("tmp.pid_123/cute_dsl_0123456789abcdef.mlir")
        )
        self.assertFalse(triton.should_sync("hash/kernel.autotune.json"))
        self.assertTrue(triton.should_sync("hash/kernel.json"))
        self.assertTrue(triton.should_sync("driver/cuda_utils.test.so"))
        self.assertFalse(flashinfer.should_sync("tmp/op.so"))
        self.assertFalse(flashinfer.should_sync("../op.so"))

    def test_trtllm_scope_tracks_flashinfer_version(self):
        scope = component(jit.COMPONENTS, "tensorrt_llm_deep_gemm").scope_func
        with mock.patch.object(jit, "_cuda_scope", return_value="cuda-12_8"):
            with mock.patch.object(jit, "_dist_version", return_value="0.3.1"):
                first = scope()
            with mock.patch.object(jit, "_dist_version", return_value="0.3.2"):
                second = scope()
        self.assertNotEqual(first, second)

    def test_cuda_component_scopes_track_package_versions(self):
        tvm_scope = component(jit.COMPONENTS, "tvm_ffi").scope_func
        cute_scope = component(jit.COMPONENTS, "cute_dsl").scope_func
        with mock.patch.object(jit, "_torch_extensions_scope", return_value="torch"):
            with mock.patch.object(jit, "_dist_version", return_value="0.1.8"):
                first_tvm = tvm_scope()
            with mock.patch.object(jit, "_dist_version", return_value="0.1.9"):
                second_tvm = tvm_scope()
        with mock.patch.object(jit, "_cuda_scope", return_value="cuda-12_9"):
            with mock.patch.object(jit, "_dist_version", return_value="4.3.5"):
                first_cute = cute_scope()
            with mock.patch.object(jit, "_dist_version", return_value="4.4.1"):
                second_cute = cute_scope()
        self.assertNotEqual(first_tvm, second_tvm)
        self.assertNotEqual(first_cute, second_cute)

    def test_restore_snapshot_and_rebase(self):
        remote = self.root / "remote"
        remote.mkdir()
        invalid_paths = {
            "": "/producer/empty",
            ".": "/producer/dot",
            "..": "/producer/parent",
            "nested/kernel.cubin": "/producer/nested/kernel.cubin",
        }
        group = {
            "child_paths": {
                "kernel.cubin": "/producer/kernel.cubin",
                **invalid_paths,
            }
        }
        write_snapshot(
            remote,
            1,
            {
                "triton/hash/__grp__kernel.json": json.dumps(group).encode(),
                "triton/hash/kernel.cubin": b"cached",
            },
        )
        target = self.root / "restored"

        self.assertTrue(jit_store.RemoteSnapshotStore(remote).restore(target))
        self.assertEqual((target / "triton/hash/kernel.cubin").read_bytes(), b"cached")
        restored = json.loads((target / "triton/hash/__grp__kernel.json").read_text())
        self.assertEqual(
            restored["child_paths"]["kernel.cubin"],
            str(target / "triton/hash/kernel.cubin"),
        )
        for name, original_path in invalid_paths.items():
            self.assertEqual(restored["child_paths"][name], original_path)

    def test_corrupt_newest_snapshot_falls_back(self):
        remote = self.root / "remote"
        remote.mkdir()
        write_snapshot(remote, 1, {"triton/base.cubin": b"base"})
        bad = remote / f"{2:020d}{jit_store.SNAPSHOT_SUFFIX}"
        bad.write_bytes(b"not a zstd archive")
        target = self.root / "restored"
        target.mkdir()
        (target / "existing.cubin").write_bytes(b"existing")

        self.assertTrue(jit_store.RemoteSnapshotStore(remote).restore(target))
        self.assertEqual((target / "existing.cubin").read_bytes(), b"existing")
        self.assertEqual((target / "triton/base.cubin").read_bytes(), b"base")
        self.assertFalse(list(target.parent.glob(f"{target.name}.restore.*")))

    def test_restore_copy_failure_does_not_fall_back(self):
        remote = self.root / "remote"
        remote.mkdir()
        write_snapshot(remote, 1, {"triton/old.cubin": b"old"})
        write_snapshot(remote, 2, {"triton/new.cubin": b"new"})
        target = self.root / "restored"

        def fail_copy(_staging, destination, **_kwargs):
            (destination / "partial.cubin").write_bytes(b"partial")
            raise OSError("copy failed")

        with mock.patch.object(
            jit_store.shutil, "copytree", side_effect=fail_copy
        ) as copytree:
            with self.assertRaisesRegex(OSError, "copy failed"):
                jit_store.RemoteSnapshotStore(remote).restore(target)

        self.assertEqual(copytree.call_count, 1)
        self.assertFalse((target / "triton/old.cubin").exists())

    def test_publish_snapshot_merges_and_rotates_versions(self):
        remote = self.root / "remote"
        remote.mkdir()
        store = jit_store.RemoteSnapshotStore(remote)
        expected = {}
        with mock.patch.object(jit_store, "SNAPSHOT_KEEP", 2):
            for index in range(3):
                source = self.root / f"kernel-{index}.cubin"
                source.write_bytes(f"value-{index}".encode())
                name = f"triton/hash/kernel-{index}.cubin"
                expected[name] = source.read_bytes()
                self.assertTrue(store.publish_snapshot({name: source}))

        snapshots = snapshot_paths(remote)
        # Lock-free: names are time_ns+uuid (unique), pruned to SNAPSHOT_KEEP,
        # and the newest holds the cumulative merge of every published file.
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(archive_members(snapshots[-1]), expected)

    def test_publish_builds_on_latest_and_writes_fresh_name(self):
        remote = self.root / "remote"
        remote.mkdir()
        existing = write_snapshot(remote, 1, {"triton/base.cubin": b"base"})
        source = self.root / "next.cubin"
        source.write_bytes(b"next")

        self.assertTrue(
            jit_store.RemoteSnapshotStore(remote).publish_snapshot(
                {"triton/next.cubin": source}
            )
        )
        snapshots = snapshot_paths(remote)
        self.assertEqual(len(snapshots), 2)
        self.assertIn(existing, snapshots)  # pre-existing snapshot never overwritten
        self.assertEqual(
            archive_members(snapshots[-1]),
            {"triton/base.cubin": b"base", "triton/next.cubin": b"next"},
        )

    def test_concurrent_publishes_all_persist_best_effort(self):
        remote = self.root / "remote"
        remote.mkdir()
        first = self.root / "first.cubin"
        second = self.root / "second.cubin"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        barrier = threading.Barrier(2)
        wait_remote_ready = jit_store.RemoteSnapshotStore._wait_remote_ready

        def synchronized_ready(path, expected_size):
            wait_remote_ready(path, expected_size)
            if path.name.startswith(".upload."):
                barrier.wait(timeout=5)

        with mock.patch.object(
            jit_store.RemoteSnapshotStore,
            "_wait_remote_ready",
            side_effect=synchronized_ready,
        ), ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(
                    jit_store.RemoteSnapshotStore(remote).publish_snapshot,
                    {"triton/first.cubin": first},
                ),
                pool.submit(
                    jit_store.RemoteSnapshotStore(remote).publish_snapshot,
                    {"triton/second.cubin": second},
                ),
            )
            self.assertTrue(all(future.result(timeout=10) for future in futures))

        # Lock-free: each publisher lands its own uniquely-named snapshot; the
        # union across all snapshots holds both, and no upload temp leaks.
        union: dict[str, bytes] = {}
        for snapshot in snapshot_paths(remote):
            union.update(archive_members(snapshot))
        self.assertEqual(
            union,
            {"triton/first.cubin": b"first", "triton/second.cubin": b"second"},
        )
        self.assertEqual(len(snapshot_paths(remote)), 2)
        self.assertFalse(list(remote.glob(".upload.*")))

    def test_manager_does_not_publish_ninja_artifacts(self):
        manager = self.make_manager()
        expected = {}
        for component_name in ("flashinfer", "torch_extensions"):
            item = component(manager.components, component_name)
            for rel in ("op/build.ninja", "op/.ninja_log", "op/.ninja_deps"):
                path = item.local_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"ninja")
                manager.stage_delta_file(item, rel)

            shared_object = item.local_dir / "op/op.so"
            shared_object.write_bytes(component_name.encode())
            manager.stage_delta_file(item, "op/op.so")
            expected[shared_object.relative_to(manager.local_root).as_posix()] = (
                component_name.encode()
            )

        manager.flush_delta2remote()
        self.assertEqual(effective_members(self.root / "remote"), expected)

    def test_restore_finishes_before_observer_starts(self):
        remote = self.root / "remote"
        remote.mkdir()
        write_snapshot(remote, 1, {"triton/a.cubin": b"cached"})
        manager = self.make_manager(remote)

        class CheckingObserver(FakeObserver):
            def start(self):
                self.restored = (manager.local_root / "triton/a.cubin").read_bytes()
                super().start()

        with mock.patch.object(jit, "Observer", CheckingObserver):
            manager.start_background_sync()

        self.assertEqual(manager._observer.restored, b"cached")

    def test_missing_remote_disables_observer(self):
        manager = self.make_manager(self.root / "missing", create_remote=False)
        with mock.patch.object(jit, "Observer", FakeObserver):
            manager.start_background_sync()
        self.assertIsNone(manager.store)
        self.assertIsNone(manager._observer)

    def test_handler_filters_component_terminal_events(self):
        manager = self.make_manager()
        flashinfer, direct = self.write_component(
            manager, "flashinfer", "cached_ops/op/a.so", b"data"
        )
        triton, atomic = self.write_component(
            manager, "triton", "hash/kernel.cubin", b"data"
        )
        staged = []
        stage = lambda *args: staged.append(args)
        direct_handler = jit._JitFileEventHandler(flashinfer, stage)
        atomic_handler = jit._JitFileEventHandler(triton, stage)

        direct_handler.on_any_event(FakeFileEvent("created", str(direct)))
        direct_handler.on_any_event(FakeFileEvent("closed", str(direct)))
        direct_handler.on_any_event(FakeFileEvent("modified", str(direct)))
        source = direct.with_name("generated.cu")
        direct_handler.on_any_event(FakeFileEvent("created", str(source)))
        ninja = direct.with_name("build.ninja")
        direct_handler.on_any_event(FakeFileEvent("created", str(ninja)))
        moved = direct.with_name("moved.so")
        moved.write_bytes(b"data")
        direct_handler.on_any_event(
            FakeFileEvent("moved", str(self.root / "tmp.so"), str(moved))
        )
        atomic_handler.on_any_event(FakeFileEvent("created", str(atomic)))
        empty = atomic.with_name("empty.cubin")
        empty.touch()
        atomic_handler.on_any_event(FakeFileEvent("created", str(empty)))

        # FlashInfer stages only terminal writes; component filtering is
        # covered separately. Triton stages atomic creation.
        self.assertEqual(
            staged,
            [
                (flashinfer, direct.relative_to(flashinfer.local_dir).as_posix()),
                (flashinfer, moved.relative_to(flashinfer.local_dir).as_posix()),
                (triton, atomic.relative_to(triton.local_dir).as_posix()),
            ],
        )

    def test_cuda_component_event_signals(self):
        manager = self.make_manager()
        tvm_ffi, shared_object = self.write_component(
            manager,
            "tvm_ffi",
            "libtorch_c_dlpack_addon_torch28-cuda.so",
            b"elf",
        )
        cute_dsl, mlir = self.write_component(
            manager, "cute_dsl", "cute_dsl_0123456789abcdef.mlir", b"mlir"
        )
        staged = []
        tvm_handler = jit._JitFileEventHandler(
            tvm_ffi, lambda *args: staged.append(args)
        )
        cute_handler = jit._JitFileEventHandler(
            cute_dsl, lambda *args: staged.append(args)
        )

        tvm_handler.on_any_event(FakeFileEvent("created", str(shared_object)))
        cute_handler.on_any_event(FakeFileEvent("created", str(mlir)))
        cute_handler.on_any_event(
            FakeFileEvent("moved", str(self.root / "tmp.mlir"), str(mlir))
        )

        self.assertEqual(
            staged,
            [
                (
                    tvm_ffi,
                    shared_object.relative_to(tvm_ffi.local_dir).as_posix(),
                ),
                (cute_dsl, mlir.relative_to(cute_dsl.local_dir).as_posix()),
            ],
        )

    def test_existing_file_without_event_is_not_uploaded(self):
        manager = self.make_manager()
        self.write_component(manager, "triton", "hash/existing.cubin", b"existing")
        manager.flush_delta2remote()
        self.assertEqual(effective_members(self.root / "remote"), {})

    def test_duplicate_events_are_batched_into_one_snapshot(self):
        manager = self.make_manager()
        item, path = self.write_component(manager, "triton", "hash/a.cubin", b"data")
        manager.stage_delta_file(item, "hash/a.cubin")
        manager.stage_delta_file(item, "hash/a.cubin")

        manager.flush_delta2remote()

        self.assertEqual(len(snapshot_paths(self.root / "remote")), 1)
        self.assertEqual(
            effective_members(self.root / "remote"),
            {path.relative_to(manager.local_root).as_posix(): b"data"},
        )

    def test_publish_failure_requeues_the_batch(self):
        manager = self.make_manager()
        item, path = self.write_component(manager, "triton", "hash/a.cubin", b"data")
        manager.stage_delta_file(item, "hash/a.cubin")

        with mock.patch.object(
            manager.store,
            "publish_snapshot",
            side_effect=OSError("remote unavailable"),
        ):
            with self.assertRaisesRegex(OSError, "remote unavailable"):
                manager.flush_delta2remote()

        manager.flush_delta2remote()
        self.assertEqual(
            effective_members(self.root / "remote"),
            {path.relative_to(manager.local_root).as_posix(): b"data"},
        )

    def test_worker_flushes_and_stop_drains(self):
        # The loop publishes full snapshots, and stop() drains whatever was staged.
        manager = self.make_manager()
        with mock.patch.object(jit, "Observer", FakeObserver), mock.patch.object(
            jit, "SYNC_POLL_S", 0.01
        ):
            manager.start_background_sync()
            item, first = self.write_component(
                manager, "triton", "hash/a.cubin", b"first"
            )
            manager.stage_delta_file(item, "hash/a.cubin")
            first_name = first.relative_to(manager.local_root).as_posix()
            self.assertTrue(
                wait_until(
                    lambda: effective_members(self.root / "remote").get(first_name)
                    == b"first"
                )
            )

            item, second = self.write_component(
                manager, "triton", "hash/b.cubin", b"second"
            )
            manager.stage_delta_file(item, "hash/b.cubin")
            manager.stop()

        self.assertEqual(
            effective_members(self.root / "remote"),
            {
                first_name: b"first",
                second.relative_to(manager.local_root).as_posix(): b"second",
            },
        )

    def test_real_observer_waits_for_direct_write_close(self):
        manager = self.make_manager()
        item = component(manager.components, "flashinfer")
        path = item.local_dir / "cached_ops/op/real.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        manager.start_background_sync()

        with path.open("wb") as output:
            output.write(b"partial")
            output.flush()
            time.sleep(0.1)
            manager.flush_delta2remote()
            self.assertEqual(effective_members(self.root / "remote"), {})
            output.write(b"-final")

        name = path.relative_to(manager.local_root).as_posix()
        self.assertTrue(
            wait_until(
                lambda: (manager.flush_delta2remote() or True)
                and effective_members(self.root / "remote").get(name)
                == b"partial-final"
            )
        )

    def test_real_observer_uploads_atomic_rename(self):
        manager = self.make_manager()
        item = component(manager.components, "deep_gemm")
        tmp = item.local_dir / "tmp/build/kernel.cubin"
        final = item.local_dir / "cache/op/kernel.cubin"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        manager.start_background_sync()

        tmp.write_bytes(b"complete")
        tmp.replace(final)
        name = final.relative_to(manager.local_root).as_posix()
        self.assertTrue(
            wait_until(
                lambda: (manager.flush_delta2remote() or True)
                and effective_members(self.root / "remote").get(name) == b"complete"
            )
        )

    def test_uri_uses_original_fuser_interface(self):
        mounted = self.root / "mounted"
        mounted.mkdir()
        with mock.patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local",
            return_value=str(mounted),
        ) as fetch:
            manager = self.make_manager("oss://bucket/cache", create_remote=False)

        self.assertEqual(manager.store.remote_root, mounted)
        self.assertEqual(fetch.call_args.args[0], "oss://bucket/cache")
        self.assertEqual(fetch.call_args.args[2], True)

    def test_smoke_chat_rejects_stream_without_choices(self):
        response = mock.MagicMock()
        response.__enter__.return_value = [
            b'data: {"choices":[]}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(smoke.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "no completion choices"):
                smoke._send_chat_completion(10000, "model", "verify", 64, 1, 1)

    def test_smoke_manifest_ignores_triton_restore_root(self):
        name = "triton/hash/__grp__kernel.json"

        def manifest(root):
            return json.dumps(
                {"child_paths": {"kernel.cubin": f"/{root}/kernel.cubin"}}
            ).encode()

        self.assertEqual(
            smoke._manifest_value(name, manifest("producer")),
            smoke._manifest_value(name, manifest("consumer")),
        )

    def test_smoke_runtime_identity_detects_atomic_replace(self):
        local_root = self.root / "identity"
        triton = component(jit.COMPONENTS, "triton").resolve(local_root)
        path = triton.local_dir / "hash/kernel.cubin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same")
        mtime_ns = path.stat().st_mtime_ns
        manifest = smoke._local_syncable_manifest(local_root)
        identity = smoke._local_runtime_identity(local_root)

        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"same")
        os.utime(replacement, ns=(mtime_ns, mtime_ns))
        replacement.replace(path)

        self.assertEqual(manifest, smoke._local_syncable_manifest(local_root))
        self.assertNotEqual(identity, smoke._local_runtime_identity(local_root))

    def test_smoke_verifier_writes_summary_before_failure(self):
        local_root, remote_root = self.root / "verify", self.root / "remote_verify"
        remote_root.mkdir()
        triton = component(jit.COMPONENTS, "triton").resolve(local_root)
        artifact = triton.local_dir / "hash/kernel.cubin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"local-only")
        summary_path = self.root / "summary.json"
        case = smoke.JitCacheSmokeTest()
        case.server = None

        with self.assertRaisesRegex(AssertionError, "missing remotely"):
            case._finish_run(
                smoke.DEEPSEEK_V2_LITE,
                local_root,
                remote_root,
                {},
                summary_path,
                {},
            )
        summary = json.loads(summary_path.read_text("utf-8"))
        self.assertFalse(summary["post_run_passed"])
        self.assertEqual(
            summary["remote_compare"]["local_not_remote"],
            ["triton/hash/kernel.cubin"],
        )


if __name__ == "__main__":
    unittest.main()

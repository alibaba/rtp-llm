import json
import os
import tarfile
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.jit_cache_manager import (
    COMPONENT_SPECS,
    SNAPSHOT_NAME,
    JitCacheManager,
)


class FakeFileEvent:
    def __init__(self, event_type: str, src_path: str, dest_path: str = ""):
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = False


def iter_component_files(root: Path, component):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            if stat.st_size > 0 and jit_cache_module.should_sync_file(component, rel):
                yield path, rel


def add_path_to_tracker(tracker, component, path: Path, root: Path) -> None:
    rel = path.relative_to(root).as_posix()
    if jit_cache_module.should_sync_file(component, rel):
        tracker.enqueue_upload(component, rel)


def write_snapshot(remote: Path) -> Path:
    snapshot = remote / SNAPSHOT_NAME
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    cctx = jit_cache_module.zstd.ZstdCompressor()
    with snapshot.open("wb") as raw:
        with cctx.stream_writer(raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for component in COMPONENT_SPECS:
                    component_root = remote / component.name
                    for path, rel in iter_component_files(component_root, component):
                        info = tar.gettarinfo(
                            str(path),
                            arcname=f"{component.name}/{rel}",
                        )
                        with path.open("rb") as source:
                            tar.addfile(info, source)
    return snapshot


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            self.wait_for_snapshot_publish(manager)
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self,
        remote="",
        timeout_s: float = 5,
        create_remote: bool = True,
        run_id: str | None = None,
        local_root: Path | None = None,
    ) -> JitCacheManager:
        if remote:
            remote_path = Path(remote)
            if create_remote and remote_path.is_absolute():
                remote_path.mkdir(parents=True, exist_ok=True)
        config = JITConfig()
        config.local_jit_cache_dir = str(local_root or self.root / "local")
        config.jit_remote_timeout_s = timeout_s
        config.remote_jit_dir = remote
        manager = JitCacheManager(config, run_id=run_id)
        self.managers.append(manager)
        manager.bootstrap()
        return manager

    def write_snapshot(self, remote: Path):
        return write_snapshot(remote)

    def read_summary(self):
        return json.loads(
            (self.root / "local" / ".rtp_jit_summary.json").read_text(encoding="utf-8")
        )

    def component(self, name: str):
        return jit_cache_module.COMPONENT_BY_NAME[name]

    def snapshot_members(self, snapshot: Path) -> dict[str, bytes]:
        members = {}
        dctx = jit_cache_module.zstd.ZstdDecompressor()
        with snapshot.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        source = tar.extractfile(member)
                        if source is not None:
                            with source:
                                members[member.name] = source.read()
        return members

    def enqueue_file(self, manager: JitCacheManager, component_name: str, key: str):
        component = self.component(component_name)
        root = manager.component_dirs.get(
            component_name, (self.root / "local" / component_name, None)
        )[0]
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(component_name, encoding="utf-8")
        tracker = manager.dirty_tracker or jit_cache_module.JitDirtyTracker(
            manager.component_dirs, manager.enqueue_upload
        )
        manager.dirty_tracker = tracker
        add_path_to_tracker(tracker, component, path, root)
        return path

    def wait_for_pending_empty(self, manager: JitCacheManager):
        if not manager._pending_empty.wait(timeout=5):
            self.fail("timed out waiting for pending uploads to drain")

    def wait_for_snapshot_publish(self, manager: JitCacheManager):
        deadline = time.time() + 5
        while manager._snapshot_publishing:
            if time.time() >= deadline:
                self.fail("timed out waiting for snapshot publish to finish")
            time.sleep(0.05)

    def test_run_id_generation_and_override(self):
        config = JITConfig()
        config.local_jit_cache_dir = str(self.root / "local")
        manager = JitCacheManager(config, run_id="shared-run")
        manager.bootstrap()

        self.assertEqual(manager.run_id, "shared-run")

        with mock.patch.object(jit_cache_module.time, "time", return_value=1.0):
            with mock.patch.object(
                jit_cache_module.uuid,
                "uuid4",
                return_value=mock.Mock(hex="child-run"),
            ):
                run_id = jit_cache_module.new_jit_cache_run_id()

        self.assertEqual(run_id, f"1000-{os.getpid()}-child-run")

    def test_bootstrap_creates_component_cache_dirs(self):
        manager = self.make_manager(str(self.root / "remote"))

        self.assertTrue(manager.enabled)
        self.assertEqual(
            set(manager.component_dirs),
            {component.name for component in COMPONENT_SPECS},
        )
        for component in COMPONENT_SPECS:
            self.assertTrue((self.root / "local" / component.name).is_dir())

    def test_triton_autotune_cache_dir_and_env_are_gpu_scoped(self):
        os.environ["TRITON_AUTOTUNE_CACHE_MODE"] = "disabled"
        with mock.patch.object(
            jit_cache_module,
            "get_gpu_scope",
            return_value="NVIDIA_H20",
        ):
            manager = self.make_manager(str(self.root / "remote"))

            self.assertEqual(
                manager.component_dirs["triton_autotune"][0],
                self.root / "local" / "triton_autotune" / "NVIDIA_H20",
            )
            self.assertEqual(
                manager.component_dirs["triton_autotune"][1],
                self.root / "remote" / "triton_autotune" / "NVIDIA_H20",
            )
            self.assertEqual(
                os.environ["TRITON_CACHE_DIR"],
                str(manager.component_dirs["triton"][0]),
            )
            self.assertEqual(
                os.environ["TRITON_AUTOTUNE_CONFIG_DIR"],
                str(manager.component_dirs["triton_autotune"][0]),
            )

            jit_cache_module.apply_jit_cache_env(self.root / "local_env")

        self.assertEqual(
            os.environ["TRITON_AUTOTUNE_CONFIG_DIR"],
            str(self.root / "local_env" / "triton_autotune" / "NVIDIA_H20"),
        )
        self.assertEqual(os.environ["TRITON_AUTOTUNE_CACHE_MODE"], "cached")

    def test_bootstrap_clears_configured_startup_files(self):
        local_root = self.root / "local"
        deep_gemm_dir = local_root / "deep_gemm"
        stale_lock = deep_gemm_dir / "cache" / "kernel_lock"
        nested_lock = deep_gemm_dir / "cache" / "nested" / "build_lock"
        keep_file = deep_gemm_dir / "cache" / "kernel.cubin"
        legacy_lock = self.root / "deep_gemm_runtime" / "legacy_lock"
        triton_lock = local_root / "triton" / "cache" / "kernel_lock"
        lock_dir = deep_gemm_dir / "cache" / "dir_lock"
        for path in (stale_lock, nested_lock, keep_file, legacy_lock, triton_lock):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data", encoding="utf-8")
        lock_dir.mkdir()

        self.make_manager(local_root=local_root)

        self.assertFalse(stale_lock.exists())
        self.assertFalse(nested_lock.exists())
        self.assertTrue(keep_file.exists())
        self.assertTrue(legacy_lock.exists())
        self.assertTrue(triton_lock.exists())
        self.assertTrue(lock_dir.exists())

    def test_prepare_without_remote_returns_disabled_summary(self):
        manager = self.make_manager()

        summary = manager.prepare()

        self.assertFalse(manager.enabled)
        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "disabled")
        self.assertNotIn("components", summary)

    def test_remote_config_validates_path(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            self.make_manager("relative/jit_cache")

        manager = self.make_manager(
            str(self.root / "missing_remote"), create_remote=False
        )
        self.assertFalse(manager.enabled)
        self.assertEqual(manager.component_dirs, {})

    def test_remote_config_mounts_uri_before_validation(self):
        mounted_remote = self.root / "mounted_remote"
        mounted_remote.mkdir()

        with mock.patch.object(
            jit_cache_module,
            "fetch_remote_file_to_local",
            return_value=str(mounted_remote),
        ) as fetch_remote:
            manager = self.make_manager("oss://bucket/jit-cache", create_remote=False)

        fetch_remote.assert_called_once_with(
            "oss://bucket/jit-cache",
            jit_cache_module.MountRwMode.RWMODE_RW,
        )
        self.assertEqual(manager.config.remote_root, mounted_remote)
        self.assertEqual(manager.component_dirs["triton"][1], mounted_remote / "triton")

    def test_remote_config_disables_cache_when_uri_mount_fails(self):
        with mock.patch.object(
            jit_cache_module,
            "fetch_remote_file_to_local",
            side_effect=RuntimeError("mount failed"),
        ) as fetch_remote:
            manager = self.make_manager("oss://bucket/jit-cache", create_remote=False)

        fetch_remote.assert_called_once_with(
            "oss://bucket/jit-cache",
            jit_cache_module.MountRwMode.RWMODE_RW,
        )
        self.assertFalse(manager.enabled)
        self.assertEqual(manager.component_dirs, {})

    def test_prepare_snapshot_miss_does_not_scan_remote_files(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")
        self.assertTrue(manager.snapshot_complete_path.exists())
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertEqual(self.read_summary()["mode"], "periodic_flush")

    def test_snapshot_miss_marker_unblocks_non_owner(self):
        remote = self.root / "remote"
        remote.mkdir()
        first = self.make_manager(str(remote))
        second = self.make_manager(str(remote), timeout_s=0)

        try:
            first_summary = first.prepare()
            second_summary = second.prepare()
        finally:
            second.stop()
            first.stop()

        self.assertEqual(first_summary["cache_state"], "snapshot_miss")
        self.assertEqual(first_summary["result"], "skipped")
        self.assertEqual(second_summary["cache_state"], "leader_completed")
        self.assertEqual(second_summary["result"], "success")

    def test_prepare_uses_external_snapshot_and_marks_memory_cache(self):
        remote = self.root / "remote"
        remote_file = remote / "triton" / "kernel" / "a.so"
        remote_file.parent.mkdir(parents=True)
        remote_file.write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        sync = manager.sync_once("periodic_flush")
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["extracted_files"], 1)
        self.assertTrue((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertTrue(manager.snapshot_complete_path.exists())
        self.assertEqual(sync["uploaded_files"], 0)
        self.assertEqual(sync["components"], {})

    def test_snapshot_rejects_path_traversal_before_writing(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME

        for member_name, escaped_name, payload in (
            ("../../escaped.txt", "escaped.txt", b"escaped"),
            ("../../escaped_dir", "escaped_dir", None),
        ):
            with self.subTest(member_name=member_name):
                cctx = jit_cache_module.zstd.ZstdCompressor()
                with snapshot.open("wb") as raw:
                    with cctx.stream_writer(raw) as compressed:
                        with tarfile.open(fileobj=compressed, mode="w|") as tar:
                            info = tarfile.TarInfo(member_name)
                            if payload is None:
                                info.type = tarfile.DIRTYPE
                                tar.addfile(info)
                            else:
                                info.size = len(payload)
                                tar.addfile(info, BytesIO(payload))
                manager = self.make_manager(str(remote))

                summary = manager.prepare()
                manager.stop()

                self.assertEqual(summary["cache_state"], "snapshot_error")
                self.assertEqual(summary["result"], "failed")
                self.assertFalse((self.root / escaped_name).exists())
                snapshot.unlink(missing_ok=True)

    def test_extract_snapshot_closes_member_source_on_timeout(self):
        remote = self.root / "remote"
        remote_file = remote / "triton" / "kernel" / "a.so"
        remote_file.parent.mkdir(parents=True)
        remote_file.write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))

        class TimeoutSource(BytesIO):
            def readinto(self, buffer):
                raise TimeoutError("copy timeout")

        source = TimeoutSource(b"so")
        with mock.patch.object(tarfile.TarFile, "extractfile", return_value=source):
            summary = manager.prepare()

        self.assertEqual(summary["result"], "timeout")
        self.assertTrue(source.closed)

    def test_snapshot_skips_unknown_component_entries(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME
        payload = b"legacy"
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    info = tarfile.TarInfo("legacy_component/kernel/a.so")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary.get("extracted_files", 0), 0)
        self.assertEqual(summary.get("extracted_bytes", 0), 0)
        self.assertFalse(
            (self.root / "local" / "legacy_component" / "kernel" / "a.so").exists()
        )

    def test_snapshot_extracts_only_current_gpu_scope(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME
        cctx = jit_cache_module.zstd.ZstdCompressor()
        entries = {
            "triton_autotune/NVIDIA_H20/configs/keep.json": b"keep",
            "triton_autotune/NVIDIA_A100/configs/drop.json": b"drop",
        }
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for name, payload in entries.items():
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        tar.addfile(info, BytesIO(payload))
        manager = self.make_manager(str(remote))

        with mock.patch.object(
            jit_cache_module,
            "get_gpu_scope",
            return_value="NVIDIA_H20",
        ):
            summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["extracted_files"], 1)
        self.assertEqual(
            (
                self.root
                / "local"
                / "triton_autotune"
                / "NVIDIA_H20"
                / "configs"
                / "keep.json"
            ).read_text(encoding="utf-8"),
            "keep",
        )
        self.assertFalse(
            (
                self.root
                / "local"
                / "triton_autotune"
                / "NVIDIA_A100"
                / "configs"
                / "drop.json"
            ).exists()
        )

    def test_upload_queue_reuploads_same_size_file_without_remote_stat(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        local_file = self.enqueue_file(manager, "flashinfer", "kernel/a.cubin")

        first = manager.sync_once()
        local_file.write_text("same_size!", encoding="utf-8")
        add_path_to_tracker(
            manager.dirty_tracker,
            self.component("flashinfer"),
            local_file,
            self.root / "local" / "flashinfer",
        )
        second = manager.sync_once()

        self.assertEqual(first["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(second["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "flashinfer" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "same_size!",
        )

    def test_upload_covers_all_jit_components(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        filenames = {
            "flashinfer": "kernel/a.cubin",
            "triton": "kernel/a.so",
            "triton_autotune": "configs/a.pkl",
            "deep_gemm": "kernel/a.cubin",
            "torch_extensions": "extension/a.so",
        }

        for component, key in filenames.items():
            self.enqueue_file(manager, component, key)

        summary = manager.sync_once("periodic_flush")

        self.assertEqual(summary["uploaded_files"], len(COMPONENT_SPECS))
        for component, key in filenames.items():
            self.assertEqual(
                (manager.component_dirs[component][1] / key).read_text(
                    encoding="utf-8"
                ),
                component,
            )

    def test_upload_skips_tmp_jit_paths(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")
        key = "hash/tmp.pid_1_abc/a.cubin"
        local_file = self.root / "local" / component.name / key
        local_file.parent.mkdir(parents=True)
        local_file.write_text("tmp", encoding="utf-8")

        manager.enqueue_upload(component, key)
        summary = manager.sync_once()

        self.assertEqual(summary["uploaded_files"], 0)
        self.assertFalse((remote / component.name / key).exists())

    def test_usage_tracker_increments_on_upload_and_extract(self):
        from rtp_llm.metrics import GaugeMetrics, kmonitor

        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))

        manager.prepare()
        usage = manager._usage_summary()
        self.assertEqual(usage["local_cache"]["components"]["triton"]["files"], 1)
        self.assertGreater(usage["local_cache"]["components"]["triton"]["bytes"], 0)

        self.enqueue_file(manager, "triton", "kernel/b.so")
        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report"
        ) as report:
            summary = manager.sync_once("periodic_flush")

        usage = manager._usage_summary()
        self.assertEqual(usage["remote_cache"]["components"]["triton"]["files"], 1)
        self.assertIn("remote_cache", summary)
        self.assertGreater(summary["remote_cache"]["files"], 0)
        self.assertTrue(report.called)
        manager.stop()

    def test_usage_tracker_is_manager_scoped(self):
        first = self.make_manager(str(self.root / "remote1"))
        second = self.make_manager(str(self.root / "remote2"))

        first._add_usage("remote", "triton", 4)

        self.assertEqual(first._usage_summary()["remote_cache"]["files"], 1)
        self.assertEqual(second._usage_summary(), {})

    def test_sync_once_ignores_kmonitor_report_errors(self):
        from rtp_llm.metrics import kmonitor

        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager._add_usage("remote", "triton", 4)

        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report", side_effect=RuntimeError("kmonitor unavailable")
        ), self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["remote_cache"]["files"], 1)
        self.assertIn(
            "failed to report JIT cache usage metrics", "\n".join(logs.output)
        )

    def test_upload_skips_empty_files(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")
        key = "kernel/empty.cubin"
        local_file = self.root / "local" / component.name / key
        local_file.parent.mkdir(parents=True)
        local_file.write_bytes(b"")

        manager.enqueue_upload(component, key)
        summary = manager.sync_once()

        self.assertEqual(summary["components"]["triton"]["skipped_files"], 1)
        self.assertFalse((remote / component.name / key).exists())

    def test__copy_atomic_preserves_existing_file_when_copy_fails(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"
        dst.parent.mkdir(parents=True)
        dst.write_text("old", encoding="utf-8")

        def failing_copy(src_path, dst_path, deadline_s):
            Path(dst_path).write_text("partial", encoding="utf-8")
            raise OSError("copy failed")

        with mock.patch.object(
            jit_cache_module, "_copy_with_deadline", side_effect=failing_copy
        ):
            with self.assertRaises(OSError):
                manager._copy_atomic(src, dst, time.monotonic() + 5)

        self.assertEqual(dst.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(dst.parent.glob(f".{dst.name}.*.tmp")), [])

    def test__copy_atomic_preserves_source_mtime(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        mtime_ns = 1_700_000_000_123_456_789
        os.utime(src, ns=(mtime_ns, mtime_ns))
        dst = remote / "triton" / "kernel" / "a.cubin"

        manager._copy_atomic(src, dst, time.monotonic() + 5)

        self.assertEqual(dst.read_text(encoding="utf-8"), "new")
        self.assertEqual(dst.stat().st_mtime_ns, mtime_ns)

    def test_watcher_enqueue_filters_events(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.component_dirs, manager.enqueue_upload
        )
        root = self.root / "local" / "triton"
        kernel = root / "kernel"
        kernel.mkdir(parents=True)
        valid_file = kernel / "a.cubin"
        valid_file.write_text("cubin", encoding="utf-8")

        with mock.patch.object(manager, "_upload_task"):
            add_path_to_tracker(tracker, self.component("triton"), valid_file, root)
            for name in ("a.cubin.tmp", "a.o", "a.cu", "compile.log", "build.ninja"):
                skipped_file = kernel / name
                skipped_file.write_text("skip", encoding="utf-8")
                add_path_to_tracker(
                    tracker, self.component("triton"), skipped_file, root
                )

            self.assertEqual(manager._pending_count, 1)

    def test_watcher_filters_paths_and_only_enqueues_completed_events(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.component_dirs, manager.enqueue_upload
        )
        root = self.root / "local" / "triton"
        kernel = root / "kernel"
        kernel.mkdir(parents=True)
        handler = tracker._make_handler(self.component("triton"), root)

        with mock.patch.object(manager, "_upload_task"):
            handler.on_any_event(
                FakeFileEvent(
                    "moved",
                    str(root / "kernel" / "moved_out.so"),
                    str(self.root / "outside" / "moved_out.so"),
                )
            )
            self.assertEqual(manager._pending_count, 0)

            ignored_file = kernel / "ignored.so"
            ignored_file.write_text("ignored", encoding="utf-8")
            for event_type in ("opened", "deleted", "modified"):
                handler.on_any_event(FakeFileEvent(event_type, str(ignored_file)))
            self.assertEqual(manager._pending_count, 0)

            tmp_file = root / "hash" / "tmp.pid_1_abc" / "tmp.cubin"
            tmp_file.parent.mkdir(parents=True)
            tmp_file.write_text("tmp", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("closed", str(tmp_file)))
            handler.on_any_event(FakeFileEvent("created", str(tmp_file)))
            self.assertEqual(manager._pending_count, 0)

            created_file = kernel / "created.so"
            created_file.write_text("so", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("created", str(created_file)))
            self.assertEqual(manager._pending_count, 0)

            closed_file = kernel / "closed.so"
            closed_file.write_text("so", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("closed", str(closed_file)))
            self.assertEqual(manager._pending_count, 0)

            moved_file = kernel / "moved.so"
            moved_file.write_text("so", encoding="utf-8")
            handler.on_any_event(
                FakeFileEvent("moved", str(kernel / "tmp.so"), str(moved_file))
            )
            self.assertEqual(manager._pending_count, 1)

    def test_watcher_uses_component_specific_completion_events(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.component_dirs, manager.enqueue_upload
        )

        def handler_for(component_name: str):
            root = manager.component_dirs[component_name][0]
            root.mkdir(parents=True, exist_ok=True)
            return (
                tracker._make_handler(self.component(component_name), root),
                root,
            )

        cases = (
            ("flashinfer", "kernel.so", "created", "closed", "so"),
            ("torch_extensions", "extension.so", "moved", "closed", "so"),
            ("triton", "kernel/a.so", "closed", "moved", "so"),
            ("triton_autotune", "kernel.json", "created", "closed", "{}"),
            ("deep_gemm", "cache/kernel.cubin", "closed", "created", "cubin"),
        )
        with mock.patch.object(manager, "_upload_task"):
            for component_name, key, ignored_event, upload_event, content in cases:
                with self.subTest(component=component_name):
                    handler, root = handler_for(component_name)
                    path = root / key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    before = manager._pending_count

                    handler.on_any_event(FakeFileEvent(ignored_event, str(path)))
                    self.assertEqual(manager._pending_count, before)

                    if upload_event == "moved":
                        event = FakeFileEvent(
                            "moved", str(path.with_suffix(".tmp")), str(path)
                        )
                    else:
                        event = FakeFileEvent(upload_event, str(path))
                    handler.on_any_event(event)
                    self.assertEqual(manager._pending_count, before + 1)

    def test_upload_failure_releases_pending_candidate(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        original_copy = manager._copy_atomic
        failures = 0

        def raise_copy(src, dst, deadline_s=None):
            nonlocal failures
            failures += 1
            raise OSError("remote write failed")

        manager._copy_atomic = raise_copy
        with self.assertLogs(level="WARNING"):
            self.enqueue_file(manager, "triton", "kernel/a.cubin")
            first = manager.sync_once()
        self.assertEqual(first["result"], "failed")
        self.assertEqual(first["components"]["triton"]["failed_files"], 1)
        self.assertEqual(failures, 1)
        self.assertTrue(manager._pending_empty.is_set())

        manager._copy_atomic = original_copy
        self.enqueue_file(manager, "triton", "kernel/a.cubin")
        second = manager.sync_once()

        self.assertEqual(second["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "triton",
        )

    def test_enqueue_upload_cleans_pending_when_executor_rejects_task(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")

        with mock.patch.object(
            manager.sync_executor, "submit", side_effect=RuntimeError("shutdown")
        ):
            self.assertFalse(manager.enqueue_upload(component, "kernel/a.so"))

        self.assertEqual(manager._pending_count, 0)
        self.assertTrue(manager._pending_empty.is_set())

    def test_upload_task_releases_pending_after_failure(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")
        manager._pending_count = 1
        manager._pending_empty.clear()
        manager._do_upload = mock.Mock(side_effect=OSError("remote write failed"))

        with self.assertLogs(level="WARNING"):
            manager._upload_task(component, "kernel/a.so")

        self.assertEqual(manager._pending_count, 0)
        self.assertTrue(manager._pending_empty.is_set())
        self.assertEqual(manager.sync_stats[component.name].failed_files, 1)

    def test_background_worker_records_upload_failure(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        local_file = self.root / "local" / "triton" / "kernel" / "a.so"
        local_file.parent.mkdir(parents=True)
        local_file.write_text("so", encoding="utf-8")
        attempts = 0

        def failing_copy(src, dst, deadline_s=None):
            nonlocal attempts
            attempts += 1
            raise OSError("remote write failed")

        manager._copy_atomic = failing_copy
        with self.assertLogs(level="WARNING"):
            add_path_to_tracker(
                manager.dirty_tracker,
                self.component("triton"),
                local_file,
                self.root / "local" / "triton",
            )
            self.wait_for_pending_empty(manager)
        manager.stop()

        self.assertEqual(attempts, 1)
        self.assertFalse((remote / "triton" / "kernel" / "a.so").exists())

    def test_sync_drains_pending_uploads(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        self.enqueue_file(manager, "triton", "kernel/a.so")

        summary = manager.sync_once()

        self.assertEqual(summary["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.so").read_text(encoding="utf-8"),
            "triton",
        )

    def test_sync_times_out_waiting_for_stuck_upload_worker(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")
        block = threading.Event()

        def blocking_upload(*a, **kw):
            block.wait()
            return 0

        manager._do_upload = blocking_upload
        manager.enqueue_upload(component, "kernel/a.so")

        manager.config.remote_timeout_s = 0.01
        with self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        block.set()
        self.assertEqual(summary["result"], "failed")
        self.assertTrue(summary.get("drain_timed_out", False))
        self.assertIn("drain timed out", "\n".join(logs.output))

    def test_sync_once_marks_failed_when_drain_raises(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        with mock.patch.object(
            manager, "_drain_upload_queue", side_effect=RuntimeError("drain failed")
        ), self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        self.assertEqual(summary["result"], "failed")
        self.assertTrue(summary.get("drain_timed_out", False))
        self.assertIn("failed to sync remote JIT cache", "\n".join(logs.output))

    def test_sync_once_schedules_snapshot_publish_as_best_effort(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="async-publisher")
        manager.prepare()
        self.enqueue_file(manager, "triton", "kernel/a.so")
        started = threading.Event()
        release = threading.Event()

        def blocking_publish():
            started.set()
            self.assertTrue(release.wait(timeout=5))

        with mock.patch.object(
            manager, "_publish_snapshot", side_effect=blocking_publish
        ):
            with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
                summary = manager.sync_once()
            self.assertEqual(summary["result"], "success")
            self.assertTrue(started.wait(timeout=1))
            self.assertTrue(manager._snapshot_publishing)
            release.set()
            self.wait_for_snapshot_publish(manager)

    def test_sync_once_publishes_snapshot_after_successful_drain(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="publisher")
        manager.prepare()
        self.enqueue_file(manager, "triton", "kernel/a.so")

        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            summary = manager.sync_once()

        self.wait_for_snapshot_publish(manager)
        snapshot = remote / SNAPSHOT_NAME
        self.assertEqual(summary["result"], "success")
        self.assertTrue(snapshot.is_file())
        self.assertEqual(
            self.snapshot_members(snapshot),
            {"triton/kernel/a.so": b"triton"},
        )
        self.assertTrue((remote / ".jit_snapshot_publish_lease.1").is_dir())
        self.assertEqual(list(remote.glob(f"{SNAPSHOT_NAME}.*.tmp")), [])
        manager.stop()

    def test_snapshot_publish_builds_archive_locally_before_remote_copy(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="local-publisher")
        manager.prepare()
        archive_paths = []

        def write_local_archive(archive: Path, deadline_s: float = 0.0):
            archive_paths.append(archive)
            self.assertEqual(archive.parent, manager.config.local_root)
            archive.write_bytes(b"local snapshot bytes")
            return 1, len(b"local snapshot bytes")

        manager._create_snapshot_archive = write_local_archive

        self.assertTrue(manager._publish_snapshot())

        self.assertEqual(len(archive_paths), 1)
        self.assertFalse(archive_paths[0].exists())
        self.assertEqual(
            (remote / SNAPSHOT_NAME).read_bytes(),
            b"local snapshot bytes",
        )
        self.assertEqual(list(remote.glob(f"{SNAPSHOT_NAME}.*.tmp")), [])

    def test_snapshot_publish_lease_allows_one_publisher_per_bucket(self):
        remote = self.root / "remote"
        first = self.make_manager(
            str(remote),
            run_id="first",
            local_root=self.root / "local_first",
        )
        second = self.make_manager(
            str(remote),
            run_id="second",
            local_root=self.root / "local_second",
        )
        first.prepare()
        second.prepare()
        self.enqueue_file(first, "triton", "kernel/first.so")

        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            first.sync_once()
        self.wait_for_snapshot_publish(first)

        self.enqueue_file(second, "triton", "kernel/second.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            second.sync_once()
        self.wait_for_snapshot_publish(second)

        self.assertEqual(
            self.snapshot_members(remote / SNAPSHOT_NAME),
            {"triton/kernel/first.so": b"triton"},
        )
        self.assertTrue((remote / "triton" / "kernel" / "second.so").is_file())
        self.assertEqual(len(list(remote.glob(".jit_snapshot_publish_lease.*"))), 1)
        second.stop()
        first.stop()

    def test_snapshot_publish_lease_expiry_allows_next_bucket(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="lease-expiry")
        manager.prepare()
        stale_lease = remote / ".jit_snapshot_publish_lease.1"
        stale_lease.mkdir()

        self.enqueue_file(manager, "triton", "kernel/a.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            first = manager.sync_once()
        self.wait_for_snapshot_publish(manager)

        self.assertEqual(first["result"], "success")
        self.assertFalse((remote / SNAPSHOT_NAME).exists())
        self.assertEqual(
            sorted(path.name for path in remote.glob(".jit_snapshot_publish_lease.*")),
            [".jit_snapshot_publish_lease.1"],
        )

        self.enqueue_file(manager, "triton", "kernel/a.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=2400.0):
            second = manager.sync_once()
        self.wait_for_snapshot_publish(manager)

        self.assertEqual(second["result"], "success")
        self.assertEqual(
            self.snapshot_members(remote / SNAPSHOT_NAME),
            {"triton/kernel/a.so": b"triton"},
        )
        self.assertEqual(
            sorted(path.name for path in remote.glob(".jit_snapshot_publish_lease.*")),
            [".jit_snapshot_publish_lease.1", ".jit_snapshot_publish_lease.2"],
        )
        manager.stop()

    def test_snapshot_publish_skips_empty_archive_and_preserves_existing_snapshot(self):
        remote = self.root / "remote"
        remote_file = remote / "triton" / "kernel" / "old.so"
        remote_file.parent.mkdir(parents=True)
        remote_file.write_text("old", encoding="utf-8")
        self.write_snapshot(remote)
        remote_file.unlink()
        manager = self.make_manager(str(remote), run_id="empty-publisher")
        manager.prepare()

        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            summary = manager.sync_once()

        self.wait_for_snapshot_publish(manager)
        self.assertEqual(summary["result"], "success")
        self.assertEqual(
            self.snapshot_members(remote / SNAPSHOT_NAME),
            {"triton/kernel/old.so": b"old"},
        )
        self.assertFalse((remote / ".jit_snapshot_publish_lease.1").exists())
        self.assertEqual(list(remote.glob(f"{SNAPSHOT_NAME}.*.tmp")), [])

        self.enqueue_file(manager, "triton", "kernel/new.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            manager.sync_once()
        self.wait_for_snapshot_publish(manager)

        self.assertEqual(
            self.snapshot_members(remote / SNAPSHOT_NAME),
            {"triton/kernel/new.so": b"triton"},
        )
        self.assertTrue((remote / ".jit_snapshot_publish_lease.1").is_dir())
        manager.stop()

    def test_snapshot_publish_failure_does_not_fail_sync_once(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        self.enqueue_file(manager, "triton", "kernel/a.so")

        with mock.patch.object(
            manager, "_publish_snapshot", side_effect=OSError("publish failed")
        ), self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()
            self.wait_for_snapshot_publish(manager)

        self.assertEqual(summary["result"], "success")
        self.assertIn("failed to publish JIT cache snapshot", "\n".join(logs.output))
        self.assertFalse(any(remote.glob(".jit_snapshot_publish_lease.*")))
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.so").read_text(encoding="utf-8"),
            "triton",
        )
        manager.stop()

    def test_start_background_sync_uploads_files_moved_after_prepare(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.start_background_sync()
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        time.sleep(0.05)
        final_file = local_kernel / "a.cubin"
        final_file.write_text("cubin", encoding="utf-8")
        handler = manager.dirty_tracker._make_handler(
            self.component("triton"),
            self.root / "local" / "triton",
        )
        handler.on_any_event(
            FakeFileEvent("moved", str(local_kernel / "tmp.cubin"), str(final_file))
        )
        self.wait_for_pending_empty(manager)
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )
        stored = self.read_summary()
        self.assertEqual(stored["mode"], "periodic_flush")

    def test_background_sync_runs_periodic_flush(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        called = threading.Event()
        original_sync_once = manager.sync_once

        def wrapped_sync_once(mode="manual_sync"):
            if mode == "periodic_flush":
                called.set()
            return original_sync_once(mode)

        manager.sync_once = wrapped_sync_once
        with mock.patch.object(jit_cache_module, "PERIODIC_SYNC_INTERVAL_S", 0.01):
            manager.start_background_sync()
            self.assertTrue(called.wait(timeout=1))

        manager.stop()
        self.assertIsNone(manager._periodic_sync_thread)

    def test_stop_freezes_watcher_before_final_flush(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        order = []
        original_tracker_stop = manager.dirty_tracker.stop
        original_sync_once = manager.sync_once

        def wrapped_tracker_stop():
            order.append("tracker_stop")
            original_tracker_stop()

        def wrapped_sync_once(mode="manual_sync"):
            order.append(f"sync_once:{mode}")
            self.assertIsNone(manager.dirty_tracker)
            return original_sync_once(mode)

        manager.dirty_tracker.stop = wrapped_tracker_stop
        manager.sync_once = wrapped_sync_once

        manager.stop()

        self.assertEqual(order, ["tracker_stop", "sync_once:periodic_flush"])
        self.assertFalse(manager.owns_startup_lock)

    def test_non_owner_takes_over_after_owner_lock_is_released(self):
        remote = self.root / "remote"
        remote.mkdir()
        first = self.make_manager(str(remote))
        self.assertTrue(first.acquire_startup_lock())
        second = self.make_manager(str(remote), timeout_s=1)
        release_owner = threading.Timer(0.05, first.stop)

        release_owner.start()
        try:
            summary = second.prepare()
        finally:
            release_owner.join()
            second.stop()
            first.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")

    def test_prepare_returns_cached_result_on_second_call(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        with mock.patch.object(
            manager, "_pull_snapshot", wraps=manager._pull_snapshot
        ) as pull_snapshot:
            first = manager.prepare()
            second = manager.prepare()

        self.assertIs(first, second)
        self.assertEqual(pull_snapshot.call_count, 1)
        manager.stop()

    def test_prepare_timeout_writes_summary_event(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote), timeout_s=0)

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertEqual(self.read_summary()["cache_state"], "timeout")

    def test_local_marker_skips_remote_snapshot(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("remote", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))
        manager.snapshot_complete_path.touch()

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "local_hit")
        self.assertEqual(summary["result"], "skipped")
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())


if __name__ == "__main__":
    unittest.main()

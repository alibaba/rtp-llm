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
    SNAPSHOT_PREFIX,
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


def find_latest_snapshot(remote: Path) -> Path | None:
    return JitCacheManager._find_latest_snapshot(remote)


def write_snapshot(remote: Path) -> Path:
    snapshot = remote / JitCacheManager._new_snapshot_name()
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


def wait_for_snapshot_publish(manager: JitCacheManager, timeout_s: float = 5) -> None:
    deadline = time.time() + timeout_s
    while manager._snapshot_publishing:
        if time.time() >= deadline:
            raise TimeoutError("timed out waiting for snapshot publish")
        time.sleep(0.05)


def snapshot_member_names(snapshot_path: Path) -> set[str]:
    dctx = jit_cache_module.zstd.ZstdDecompressor()
    with snapshot_path.open("rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                return {member.name for member in tar}


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            wait_for_snapshot_publish(manager)
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
        config.remote_sync_timeout_s = timeout_s
        config.remote_jit_cache_dir = remote
        manager = JitCacheManager(config, run_id=run_id)
        self.managers.append(manager)
        manager.bootstrap()
        return manager

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
            "get_gpu_info",
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

            for c in COMPONENT_SPECS:
                os.environ.pop(c.env_name, None)
            os.environ.pop("TRITON_AUTOTUNE_CACHE_MODE", None)
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
        # Marker is written by _prepare_as_leader regardless of pull outcome so
        # followers don't block; cross-boot re-pull is guaranteed by the unlink
        # at the start of _prepare_as_leader rather than by skipping the touch.
        self.assertTrue(manager.snapshot_complete_path.exists())
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertEqual(self.read_summary()["mode"], "snapshot_download")

    def test_snapshot_miss_retries_on_next_boot(self):
        # First "boot" sees an empty remote → snapshot_miss; the second "boot"
        # (a fresh manager sharing local_root) must re-attempt the pull and
        # find the snapshot the first one missed.
        remote = self.root / "remote"
        remote.mkdir()
        local = self.root / "local"

        first = self.make_manager(str(remote), local_root=local)
        first_summary = first.prepare()
        first.stop()

        # Another node publishes a snapshot between boots.
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        write_snapshot(remote)

        second = self.make_manager(str(remote), local_root=local)
        second_summary = second.prepare()
        second.stop()

        self.assertEqual(first_summary["cache_state"], "snapshot_miss")
        self.assertEqual(second_summary["cache_state"], "snapshot_hit")
        self.assertEqual(second_summary["result"], "success")
        self.assertTrue((local / "triton" / "kernel" / "a.so").exists())

    def test_prepare_uses_external_snapshot_and_marks_memory_cache(self):
        remote = self.root / "remote"
        remote_file = remote / "triton" / "kernel" / "a.so"
        remote_file.parent.mkdir(parents=True)
        remote_file.write_text("so", encoding="utf-8")
        write_snapshot(remote)
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
        snapshot = remote / JitCacheManager._new_snapshot_name()

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
        write_snapshot(remote)
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
        snapshot = remote / JitCacheManager._new_snapshot_name()
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
        snapshot = remote / JitCacheManager._new_snapshot_name()
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
            "get_gpu_info",
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

    def test_upload_skips_existing_remote_file(self):
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
        self.assertEqual(second["components"]["flashinfer"]["uploaded_files"], 0)
        self.assertEqual(
            (remote / "flashinfer" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "flashinfer",
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
        write_snapshot(remote)
        manager = self.make_manager(str(remote))

        manager.prepare()
        usage = manager._usage_summary(manager._usage_snapshot())
        self.assertEqual(usage["local_cache"]["components"]["triton"]["files"], 1)
        self.assertGreater(usage["local_cache"]["components"]["triton"]["bytes"], 0)

        self.enqueue_file(manager, "triton", "kernel/b.so")
        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report"
        ) as report:
            summary = manager.sync_once("periodic_flush")

        usage = manager._usage_summary(manager._usage_snapshot())
        self.assertEqual(usage["remote_cache"]["components"]["triton"]["files"], 1)
        self.assertIn("remote_cache", summary)
        self.assertGreater(summary["remote_cache"]["files"], 0)
        self.assertTrue(report.called)
        manager.stop()

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

    def test_sync_once_lock_contention_returns_standard_summary(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        manager._sync_lock.acquire()
        try:
            summary = manager.sync_once("manual_contention")
        finally:
            manager._sync_lock.release()

        self.assertEqual(summary["mode"], "manual_contention")
        self.assertEqual(summary["result"], "skipped")
        self.assertEqual(summary["reason"], "sync in progress")
        self.assertIn("timestamp_ms", summary)
        self.assertIn("total_cost_ms", summary)
        self.assertEqual(self.read_summary()["result"], "skipped")

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

    def test__copy_atomic_skips_existing_destination(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"
        dst.parent.mkdir(parents=True)
        dst.write_text("old", encoding="utf-8")

        result = manager._copy_atomic(src, dst, time.monotonic() + 5)

        self.assertEqual(result, 0)
        self.assertEqual(dst.read_text(encoding="utf-8"), "old")

    def test__copy_atomic_copies_file_content(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"

        result = manager._copy_atomic(src, dst, time.monotonic() + 5)

        self.assertEqual(dst.read_text(encoding="utf-8"), "new")
        self.assertGreater(result, 0)

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

            for name in ("a.cubin.tmp", "a.o", "a.cu", "compile.log", "build.ninja"):
                bad_suffix = kernel / name
                bad_suffix.write_text("skip", encoding="utf-8")
                handler.on_any_event(
                    FakeFileEvent("moved", str(kernel / "tmp"), str(bad_suffix))
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

        manager.config.remote_sync_timeout_s = 0.01
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
            wait_for_snapshot_publish(manager)

    def test_sync_once_publishes_snapshot_after_successful_drain(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="publisher")
        manager.prepare()
        self.enqueue_file(manager, "triton", "kernel/a.so")

        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            summary = manager.sync_once()

        wait_for_snapshot_publish(manager)
        snapshot = find_latest_snapshot(remote)
        self.assertIsNotNone(snapshot)
        self.assertEqual(summary["result"], "success")
        self.assertTrue(snapshot.is_file())
        self.assertTrue(snapshot.name.startswith(SNAPSHOT_PREFIX))
        self.assertEqual(
            self.snapshot_members(snapshot),
            {"triton/kernel/a.so": b"triton"},
        )
        self.assertTrue((remote / ".jit_snapshot_publish_lease.1").is_dir())
        manager.stop()

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
        wait_for_snapshot_publish(first)

        self.enqueue_file(second, "triton", "kernel/second.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            second.sync_once()
        wait_for_snapshot_publish(second)

        snapshot = find_latest_snapshot(remote)
        self.assertIsNotNone(snapshot)
        self.assertEqual(
            self.snapshot_members(snapshot),
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
        wait_for_snapshot_publish(manager)

        self.assertEqual(first["result"], "success")
        self.assertIsNone(find_latest_snapshot(remote))
        self.assertEqual(
            sorted(path.name for path in remote.glob(".jit_snapshot_publish_lease.*")),
            [".jit_snapshot_publish_lease.1"],
        )

        self.enqueue_file(manager, "triton", "kernel/a.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=2400.0):
            second = manager.sync_once()
        wait_for_snapshot_publish(manager)

        self.assertEqual(second["result"], "success")
        snapshot = find_latest_snapshot(remote)
        self.assertIsNotNone(snapshot)
        self.assertEqual(
            self.snapshot_members(snapshot),
            {"triton/kernel/a.so": b"triton"},
        )
        self.assertEqual(
            sorted(path.name for path in remote.glob(".jit_snapshot_publish_lease.*")),
            [".jit_snapshot_publish_lease.1", ".jit_snapshot_publish_lease.2"],
        )
        manager.stop()

    def test_snapshot_publish_skips_empty_archive_and_preserves_existing_snapshot(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote), run_id="empty-publisher")
        manager.prepare()
        # No local JIT files — publish should produce empty archive and skip.

        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            summary = manager.sync_once()

        wait_for_snapshot_publish(manager)
        self.assertEqual(summary["result"], "success")
        self.assertIsNone(find_latest_snapshot(remote))
        self.assertFalse((remote / ".jit_snapshot_publish_lease.1").exists())

        self.enqueue_file(manager, "triton", "kernel/new.so")
        with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
            manager.sync_once()
        wait_for_snapshot_publish(manager)

        snapshot = find_latest_snapshot(remote)
        self.assertIsNotNone(snapshot)
        self.assertEqual(
            self.snapshot_members(snapshot),
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
            wait_for_snapshot_publish(manager)

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
        self.assertEqual(stored["mode"], "snapshot_download")

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
        manager.config.remote_sync_longer_timeout_s = 0.01
        manager.start_background_sync()
        self.assertTrue(called.wait(timeout=1))

        manager.stop()
        self.assertIsNone(manager._periodic_sync_thread)

    def test_periodic_sync_clamps_nonpositive_interval(self):
        manager = self.make_manager()
        manager.config.remote_sync_longer_timeout_s = 0

        with mock.patch.object(
            manager._periodic_sync_stop, "wait", return_value=True
        ) as wait:
            manager._periodic_sync_loop()

        wait.assert_called_once_with(jit_cache_module.STARTUP_WAIT_POLL_S)

    def test_stop_freezes_watcher_before_executor_shutdown(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        order = []
        original_tracker_stop = manager.dirty_tracker.stop

        def wrapped_tracker_stop():
            order.append("tracker_stop")
            original_tracker_stop()

        manager.dirty_tracker.stop = wrapped_tracker_stop

        manager.stop()

        self.assertEqual(order, ["tracker_stop"])
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
        write_snapshot(remote)
        manager = self.make_manager(str(remote), timeout_s=0)

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertEqual(self.read_summary()["cache_state"], "timeout")

    def test_stop_cancels_pending_uploads(self):
        manager = self.make_manager()
        with mock.patch.object(
            manager.sync_executor, "shutdown", wraps=manager.sync_executor.shutdown
        ) as shutdown:
            manager.stop()
        shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_snapshot_iter_skips_component_when_dir_probe_fails(self):
        # An OSError on a single component's local dir must not abort the
        # iteration; remaining components must still be included.
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        local = manager.config.local_root
        for name in ("triton", "flashinfer"):
            d = local / name / "kernel"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.so").write_text(name, encoding="utf-8")

        real_is_dir = Path.is_dir

        def flaky_is_dir(self):
            if "flashinfer" in self.parts and self.name == "flashinfer":
                raise OSError("simulated local FS error")
            return real_is_dir(self)

        with mock.patch.object(Path, "is_dir", flaky_is_dir):
            collected = list(manager._iter_snapshot_files(manager.config.local_root))
        manager.stop()

        names = {comp.name for comp, _, _ in collected}
        self.assertIn("triton", names)
        self.assertNotIn("flashinfer", names)

    def test_apply_jit_cache_env_respects_pre_set_cache_mode(self):
        # Smoke tests set TRITON_AUTOTUNE_CACHE_MODE explicitly; bootstrap
        # must not clobber it.
        os.environ["TRITON_AUTOTUNE_CACHE_MODE"] = "disabled"
        jit_cache_module.apply_jit_cache_env(self.root / "local_apply")
        self.assertEqual(os.environ["TRITON_AUTOTUNE_CACHE_MODE"], "disabled")

    def test_leader_ignores_pre_existing_marker(self):
        # The marker is boot-scoped: a marker left over from a previous process
        # must not short-circuit this boot's pull attempt, otherwise an early
        # cold-start snapshot_miss would permanently disable snapshot bootstrap.
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("remote", encoding="utf-8")
        write_snapshot(remote)
        manager = self.make_manager(str(remote))
        manager.snapshot_complete_path.touch()

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertTrue((self.root / "local" / "triton" / "kernel" / "a.so").exists())


class SnapshotPublishConsumerTest(unittest.TestCase):
    """Publish snapshot then verify a new manager extracts it (no GPU needed)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        os.environ["TRITON_AUTOTUNE_GPU_NAME"] = "NVIDIA_H20"
        jit_cache_module.get_gpu_info.cache_clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        jit_cache_module.get_gpu_info.cache_clear()
        self.tmp.cleanup()

    def make_manager(self, local_root, remote_root, run_id):
        remote_root.mkdir(parents=True, exist_ok=True)
        config = JITConfig()
        config.local_jit_cache_dir = str(local_root)
        config.remote_jit_cache_dir = str(remote_root)
        config.remote_sync_timeout_s = 10
        manager = JitCacheManager(config, run_id=run_id)
        manager.bootstrap()
        return manager

    def test_publish_then_consumer_extracts_snapshot(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root, "publisher")
        try:
            prepare = first.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_miss")

            expected_members = set()
            for component in COMPONENT_SPECS:
                local_root, _ = first.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                local_file = local_root / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_text(component.name, encoding="utf-8")
                self.assertTrue(first.enqueue_upload(component, filename))
                if component.gpu_scoped:
                    expected_members.add(
                        f"{component.name}/{jit_cache_module.get_gpu_info()}/{filename}"
                    )
                else:
                    expected_members.add(f"{component.name}/{filename}")

            with mock.patch.object(jit_cache_module.time, "time", return_value=1200.0):
                summary = first.sync_once("publish_consumer_test")
            wait_for_snapshot_publish(first, timeout_s=10)

            snapshot_path = find_latest_snapshot(remote_root)
            self.assertIsNotNone(snapshot_path)
            self.assertEqual(summary["result"], "success")
            self.assertTrue(snapshot_path.is_file())
            self.assertTrue(snapshot_path.name.startswith(SNAPSHOT_PREFIX))
            self.assertEqual(snapshot_member_names(snapshot_path), expected_members)
            self.assertTrue((remote_root / ".jit_snapshot_publish_lease.1").is_dir())
        finally:
            first.stop()

        second = self.make_manager(self.root / "local_second", remote_root, "consumer")
        try:
            prepare = second.prepare()
            self.assertEqual(prepare["cache_state"], "snapshot_hit")
            self.assertEqual(prepare["result"], "success")
            self.assertEqual(prepare["extracted_files"], len(COMPONENT_SPECS))

            for component in COMPONENT_SPECS:
                local_root, _ = second.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                self.assertEqual(
                    (local_root / filename).read_text(encoding="utf-8"),
                    component.name,
                )
        finally:
            second.stop()


if __name__ == "__main__":
    unittest.main()

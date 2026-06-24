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
            if stat.st_size > 0 and jit_cache_module.is_syncable_file(component, rel):
                yield path, rel, jit_cache_module.FileMeta(
                    size=stat.st_size, mtime_ns=stat.st_mtime_ns
                )


def add_path_to_tracker(tracker, component, path: Path, root: Path) -> None:
    rel = path.relative_to(root).as_posix()
    if jit_cache_module.is_syncable_file(component, rel):
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
                    for path, rel, _ in iter_component_files(component_root, component):
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

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self,
        remote="",
        timeout_s: float = 5,
        create_remote: bool = True,
        run_id: str | None = None,
    ) -> JitCacheManager:
        if remote:
            remote_path = Path(remote)
            if create_remote and remote_path.is_absolute():
                remote_path.mkdir(parents=True, exist_ok=True)
        config = JITConfig()
        config.local_jit_cache_dir = str(self.root / "local")
        config.jit_remote_timeout_s = timeout_s
        config.remote_jit_dir = remote
        manager = JitCacheManager(config, run_id=run_id)
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
        deadline = time.time() + 5
        with manager._lock:
            while manager.pending_uploads:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.fail("timed out waiting for pending uploads to drain")
                manager._lock.wait(timeout=remaining)

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
        with self.assertRaisesRegex(ValueError, "existing directory"):
            self.make_manager(str(self.root / "missing_remote"), create_remote=False)

    def test_file_meta_requires_exact_metadata_match(self):
        left = jit_cache_module.FileMeta(size=16, mtime_ns=10_000)

        self.assertNotEqual(left, jit_cache_module.FileMeta(size=16, mtime_ns=10_999))
        self.assertNotEqual(left, jit_cache_module.FileMeta(size=17, mtime_ns=10_000))
        self.assertEqual(left, left)

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
        marker = json.loads(manager.snapshot_complete_path.read_text(encoding="utf-8"))
        self.assertEqual(marker["event"], "jit_cache_snapshot_complete")
        self.assertEqual(marker["run_id"], manager.run_id)
        self.assertEqual(marker["extracted_files"], 1)
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
        sources = []

        def raise_timeout(source, out, deadline_s, **kwargs):
            sources.append(source)
            raise TimeoutError("copy timeout")

        with mock.patch.object(
            jit_cache_module,
            "copy_stream_with_deadline",
            side_effect=raise_timeout,
        ):
            summary = manager.prepare()

        self.assertEqual(summary["result"], "timeout")
        self.assertTrue(sources)
        self.assertTrue(sources[0].closed)

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
        self.assertNotIn("extracted_files", summary)
        self.assertFalse(
            (self.root / "local" / "legacy_component" / "kernel" / "a.so").exists()
        )

    def test_apply_extract_renames_extracted_tree_into_local_cache(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        extract_root = self.root / "extract"
        new_file = extract_root / "new_component" / "kernel" / "a.so"
        new_file.parent.mkdir(parents=True)
        new_file.write_text("so", encoding="utf-8")
        existing_file = self.root / "local" / "triton" / "kernel" / "old.so"
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text("old", encoding="utf-8")
        replacement = extract_root / "triton" / "kernel" / "old.so"
        replacement.parent.mkdir(parents=True)
        replacement.write_text("new", encoding="utf-8")

        manager._apply_extract(extract_root)

        self.assertFalse(list(extract_root.rglob("*.*")))
        self.assertEqual(
            (self.root / "local" / "new_component" / "kernel" / "a.so").read_text(
                encoding="utf-8"
            ),
            "so",
        )
        self.assertEqual(existing_file.read_text(encoding="utf-8"), "new")

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

        tracker = manager.usage_tracker

        manager.prepare()
        usage = tracker.to_summary_dict()
        self.assertEqual(usage["local_cache"]["components"]["triton"]["files"], 1)
        self.assertGreater(usage["local_cache"]["components"]["triton"]["bytes"], 0)

        self.enqueue_file(manager, "triton", "kernel/b.so")
        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report"
        ) as report:
            summary = manager.sync_once("periodic_flush")

        usage = tracker.to_summary_dict()
        self.assertEqual(usage["remote_cache"]["components"]["triton"]["files"], 1)
        self.assertIn("remote_cache", summary)
        self.assertGreater(summary["remote_cache"]["files"], 0)
        self.assertTrue(report.called)
        manager.stop()

    def test_usage_tracker_is_manager_scoped(self):
        first = self.make_manager(str(self.root / "remote1"))
        second = self.make_manager(str(self.root / "remote2"))

        first.usage_tracker.add("remote", "triton", 4)

        self.assertEqual(
            first.usage_tracker.to_summary_dict()["remote_cache"]["files"], 1
        )
        self.assertEqual(second.usage_tracker.to_summary_dict(), {})

    def test_sync_once_ignores_kmonitor_report_errors(self):
        from rtp_llm.metrics import kmonitor

        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.usage_tracker.add("remote", "triton", 4)

        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report", side_effect=RuntimeError("kmonitor unavailable")
        ), self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["remote_cache"]["files"], 1)
        self.assertIn(
            "failed to report JIT cache usage metrics", "\n".join(logs.output)
        )

    def test_upload_retries_when_file_changes_during_copy(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        call_count = 0
        original_file_meta = jit_cache_module.file_meta
        local_file = None

        def mutating_stat(path, **kwargs):
            nonlocal call_count
            meta = original_file_meta(path, **kwargs)
            call_count += 1
            if call_count == 2:
                local_file.write_text("mutated-mid-upload", encoding="utf-8")
            return original_file_meta(path, **kwargs) if call_count == 2 else meta

        with mock.patch.object(
            jit_cache_module, "file_meta", side_effect=mutating_stat
        ):
            local_file = self.enqueue_file(manager, "triton", "kernel/a.cubin")
            first = manager.sync_once()

        self.assertEqual(first["components"]["triton"]["retried_files"], 1)
        self.assertEqual(first["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "mutated-mid-upload",
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

    def test__copy_verified_preserves_existing_file_when_copy_fails(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"
        dst.parent.mkdir(parents=True)
        dst.write_text("old", encoding="utf-8")

        def failing_copy(src_path, dst_path):
            Path(dst_path).write_text("partial", encoding="utf-8")
            raise OSError("copy failed")

        with mock.patch.object(
            jit_cache_module.shutil, "copyfile", side_effect=failing_copy
        ):
            with self.assertRaises(OSError):
                manager._copy_verified(src, dst, jit_cache_module.file_meta(src))

        self.assertEqual(dst.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(dst.parent.glob(f".{dst.name}.*.tmp")), [])

    def test__copy_verified_preserves_source_mtime(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        mtime_ns = 1_700_000_000_123_456_789
        os.utime(src, ns=(mtime_ns, mtime_ns))
        dst = remote / "triton" / "kernel" / "a.cubin"

        manager._copy_verified(src, dst, jit_cache_module.file_meta(src))

        self.assertEqual(dst.read_text(encoding="utf-8"), "new")
        self.assertEqual(dst.stat().st_mtime_ns, mtime_ns)

    def test__copy_verified_rejects_source_changed_during_copy(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("old", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"
        dst.parent.mkdir(parents=True)
        dst.write_text("remote-old", encoding="utf-8")
        src_meta = jit_cache_module.file_meta(src)
        original_copyfile = jit_cache_module.shutil.copyfile

        def mutating_copy(src_path, dst_path):
            result = original_copyfile(src_path, dst_path)
            Path(src_path).write_text("changed-after-copy", encoding="utf-8")
            return result

        with mock.patch.object(
            jit_cache_module.shutil, "copyfile", side_effect=mutating_copy
        ):
            with self.assertRaisesRegex(RuntimeError, "source changed during upload"):
                manager._copy_verified(src, dst, src_meta)

        self.assertEqual(dst.read_text(encoding="utf-8"), "remote-old")
        self.assertEqual(list(dst.parent.glob(f".{dst.name}.*.tmp")), [])

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

        with mock.patch.object(manager, "_process_upload_task"):
            add_path_to_tracker(tracker, self.component("triton"), valid_file, root)
            add_path_to_tracker(tracker, self.component("triton"), valid_file, root)
            for name in ("a.cubin.tmp", "a.o", "a.cu", "compile.log", "build.ninja"):
                skipped_file = kernel / name
                skipped_file.write_text("skip", encoding="utf-8")
                add_path_to_tracker(
                    tracker, self.component("triton"), skipped_file, root
                )

            self.assertEqual(len(manager.pending_uploads), 1)
            self.assertIn(("triton", "kernel/a.cubin"), manager.pending_uploads)

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

        with mock.patch.object(manager, "_process_upload_task"):
            handler.on_any_event(
                FakeFileEvent(
                    "moved",
                    str(root / "kernel" / "moved_out.so"),
                    str(self.root / "outside" / "moved_out.so"),
                )
            )
            self.assertEqual(len(manager.pending_uploads), 0)

            ignored_file = kernel / "ignored.so"
            ignored_file.write_text("ignored", encoding="utf-8")
            for event_type in ("opened", "deleted", "modified"):
                handler.on_any_event(FakeFileEvent(event_type, str(ignored_file)))
            self.assertEqual(len(manager.pending_uploads), 0)

            tmp_file = root / "hash" / "tmp.pid_1_abc" / "tmp.cubin"
            tmp_file.parent.mkdir(parents=True)
            tmp_file.write_text("tmp", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("closed", str(tmp_file)))
            handler.on_any_event(FakeFileEvent("created", str(tmp_file)))
            self.assertEqual(len(manager.pending_uploads), 0)

            created_file = kernel / "created.so"
            created_file.write_text("so", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("created", str(created_file)))
            self.assertEqual(len(manager.pending_uploads), 0)

            closed_file = kernel / "closed.so"
            closed_file.write_text("so", encoding="utf-8")
            handler.on_any_event(FakeFileEvent("closed", str(closed_file)))
            self.assertEqual(len(manager.pending_uploads), 0)

            moved_file = kernel / "moved.so"
            moved_file.write_text("so", encoding="utf-8")
            handler.on_any_event(
                FakeFileEvent("moved", str(kernel / "tmp.so"), str(moved_file))
            )
            self.assertIn(("triton", "kernel/moved.so"), manager.pending_uploads)
            self.assertEqual(len(manager.pending_uploads), 1)

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
            ("triton_autotune", "kernel.json", "created", "closed", "{}"),
            ("deep_gemm", "cache/kernel.cubin", "closed", "created", "cubin"),
        )
        with mock.patch.object(manager, "_process_upload_task"):
            for component_name, key, ignored_event, upload_event, content in cases:
                with self.subTest(component=component_name):
                    handler, root = handler_for(component_name)
                    path = root / key
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    before = len(manager.pending_uploads)

                    handler.on_any_event(FakeFileEvent(ignored_event, str(path)))
                    self.assertEqual(len(manager.pending_uploads), before)

                    handler.on_any_event(FakeFileEvent(upload_event, str(path)))
                    self.assertIn((component_name, key), manager.pending_uploads)

    def test_upload_failure_retries_then_releases_pending_candidate(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        original_copy = manager._copy_verified
        failures = 0

        def raise_copy(src, dst, src_meta):
            nonlocal failures
            failures += 1
            raise OSError("remote write failed")

        manager._copy_verified = raise_copy
        self.enqueue_file(manager, "triton", "kernel/a.cubin")
        with self.assertLogs(level="WARNING"):
            first = manager.sync_once()
        self.assertEqual(first["result"], "failed")
        self.assertEqual(first["components"]["triton"]["failed_files"], 1)
        self.assertEqual(
            first["components"]["triton"]["retried_files"],
            jit_cache_module.MAX_UPLOAD_ATTEMPTS - 1,
        )
        self.assertEqual(failures, jit_cache_module.MAX_UPLOAD_ATTEMPTS)
        self.assertEqual(len(manager.pending_uploads), 0)

        manager._copy_verified = original_copy
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

        self.assertEqual(manager.pending_uploads, set())

    def test_process_upload_task_releases_pending_when_retry_submit_fails(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        component = self.component("triton")
        pending_key = (component.name, "kernel/a.so")
        manager.pending_uploads.add(pending_key)
        manager._try_upload = lambda *a, **kw: jit_cache_module.SyncResult(
            "retry", reason="remote write failed"
        )

        with mock.patch.object(
            manager, "_submit_upload_task", return_value=False
        ), self.assertLogs(level="WARNING"):
            manager._process_upload_task(
                jit_cache_module.UploadTask(component, "kernel/a.so")
            )

        self.assertEqual(manager.pending_uploads, set())
        self.assertEqual(manager.sync_stats[component.name].failed_files, 1)

    def test_background_worker_retries_transient_upload_failure(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        local_file = self.root / "local" / "triton" / "kernel" / "a.so"
        local_file.parent.mkdir(parents=True)
        local_file.write_text("so", encoding="utf-8")
        original_copy = manager._copy_verified
        attempts = 0

        def flaky_copy(src, dst, src_meta):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient remote write failed")
            return original_copy(src, dst, src_meta)

        manager._copy_verified = flaky_copy
        add_path_to_tracker(
            manager.dirty_tracker,
            self.component("triton"),
            local_file,
            self.root / "local" / "triton",
        )
        self.wait_for_pending_empty(manager)
        manager.stop()

        self.assertEqual(attempts, 2)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.so").read_text(encoding="utf-8"),
            "so",
        )

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
        manager._try_upload = lambda *a, **kw: (
            block.wait(),
            jit_cache_module.SyncResult("skipped"),
        )[1]
        manager.enqueue_upload(component, "kernel/a.so")

        manager.config.remote_timeout_s = 0.01
        with self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        block.set()
        self.assertEqual(summary["result"], "failed")
        self.assertEqual(summary["components"]["triton"]["drain_timed_out"], 1)
        self.assertIn("drain timed out", "\n".join(logs.output))

    def test_sync_once_marks_failed_when_drain_raises(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        with mock.patch.object(
            manager, "_drain_upload_queue", side_effect=RuntimeError("drain failed")
        ), self.assertLogs(level="WARNING") as logs:
            summary = manager.sync_once()

        self.assertEqual(summary["result"], "failed")
        self.assertEqual(summary["drain_timed_out"], len(manager.component_dirs))
        self.assertIn("failed to sync remote JIT cache", "\n".join(logs.output))

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

    def test_stop_freezes_watcher_before_final_flush(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        order = []
        original_stop_dirty_tracker = manager.stop_dirty_tracker
        original_sync_once = manager.sync_once

        def wrapped_stop_dirty_tracker():
            order.append("stop_dirty_tracker")
            original_stop_dirty_tracker()

        def wrapped_sync_once(mode="manual_sync"):
            order.append(f"sync_once:{mode}")
            self.assertIsNone(manager.dirty_tracker)
            return original_sync_once(mode)

        manager.stop_dirty_tracker = wrapped_stop_dirty_tracker
        manager.sync_once = wrapped_sync_once

        manager.stop()

        self.assertEqual(order, ["stop_dirty_tracker", "sync_once:periodic_flush"])
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

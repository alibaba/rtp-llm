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
    JitCacheManager,
    iter_component_sync_files,
    open_snapshot_reader,
    open_snapshot_writer,
    snapshot_path,
)


class FakeFileEvent:
    def __init__(self, event_type: str, src_path: str, dest_path: str = ""):
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = False


def write_snapshot(remote: Path) -> Path:
    snapshot = remote / jit_cache_module.SNAPSHOT_NAME
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with open_snapshot_writer(snapshot) as tar:
        for component in jit_cache_module.COMPONENT_SPECS:
            component_root = jit_cache_module.component_cache_dir(remote, component)
            for path, rel in iter_component_sync_files(component_root, component):
                arcname = f"{component.name}/{jit_cache_module.component_snapshot_rel(component, rel)}"
                tar.add(str(path), arcname=arcname, recursive=False)
    return snapshot


def write_raw_snapshot(remote: Path, entries: dict[str, bytes | None]) -> Path:
    snapshot = remote / jit_cache_module.SNAPSHOT_NAME
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with open_snapshot_writer(snapshot) as tar:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
    return snapshot


def snapshot_member_names(snapshot_path: Path) -> set[str]:
    with open_snapshot_reader(snapshot_path) as tar:
        return {member.name for member in tar}


def snapshot_members(snapshot_path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with open_snapshot_reader(snapshot_path) as tar:
        for member in tar:
            source = tar.extractfile(member)
            if source is not None:
                with source:
                    members[member.name] = source.read()
    return members


def make_manager(
    root: Path,
    remote: str = "",
    *,
    local_root: Path | None = None,
    create_remote: bool = True,
    debounce_s: float = 0,
) -> JitCacheManager:
    """Helper used by both test classes; debounce_s=0 makes publish synchronous."""
    if remote and create_remote:
        Path(remote).mkdir(parents=True, exist_ok=True)
    config = JITConfig()
    config.local_jit_dir = str(local_root or root / "local")
    config.remote_jit_dir = remote
    manager = JitCacheManager(config, debounce_s=debounce_s)
    manager.bootstrap()
    return manager


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        self.managers: list[JitCacheManager] = []

    def tearDown(self):
        for manager in self.managers:
            manager.stop()
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self,
        remote: str = "",
        *,
        local_root: Path | None = None,
        create_remote: bool = True,
        debounce_s: float = 0,
    ) -> JitCacheManager:
        manager = make_manager(
            self.root,
            remote,
            local_root=local_root,
            create_remote=create_remote,
            debounce_s=debounce_s,
        )
        self.managers.append(manager)
        return manager

    def make_remote_manager(self, **kwargs) -> tuple[Path, JitCacheManager]:
        remote = self.root / "remote"
        return remote, self.make_manager(str(remote), **kwargs)

    def component(self, name: str):
        return jit_cache_module.COMPONENT_BY_NAME[name]

    def component_dir(self, root: Path, name: str) -> Path:
        return jit_cache_module.component_cache_dir(root, self.component(name))

    def upload_file_helper(
        self, manager: JitCacheManager, component_name: str, rel: str
    ):
        component = self.component(component_name)
        local_root = manager.component_dirs[component_name]
        path = local_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(component_name, encoding="utf-8")
        self.assertTrue(manager.upload_file(component, rel))
        return path

    def test_bootstrap_creates_managed_component_dirs(self):
        remote, manager = self.make_remote_manager()

        self.assertTrue(manager.enabled)
        self.assertEqual(
            set(manager.component_dirs),
            {component.name for component in jit_cache_module.COMPONENT_SPECS},
        )
        for component in jit_cache_module.COMPONENT_SPECS:
            self.assertTrue(
                jit_cache_module.component_cache_dir(
                    self.root / "local", component
                ).is_dir()
            )
            self.assertEqual(
                manager.component_dirs[component.name],
                jit_cache_module.component_cache_dir(self.root / "local", component),
            )

    def test_prepare_without_remote_is_disabled(self):
        manager = self.make_manager()

        summary = manager.prepare()

        self.assertEqual(summary["result"], "skipped")
        self.assertEqual(summary["cache_state"], "disabled")

    def test_prepare_miss_skips_snapshot_download(self):
        _remote, manager = self.make_remote_manager()

        summary = manager.prepare()

        self.assertEqual(summary["result"], "skipped")
        self.assertEqual(summary["cache_state"], "snapshot_miss")

    def test_prepare_extracts_fixed_snapshot_every_boot_and_overwrites_local(self):
        remote = self.root / "remote"
        local = self.root / "local"
        first = self.make_manager(str(remote), local_root=local)
        remote_file = self.component_dir(remote, "triton") / "kernel/a.so"
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("first", encoding="utf-8")
        write_snapshot(remote)

        first_summary = first.prepare()
        self.assertEqual(first_summary["cache_state"], "snapshot_hit")
        target = self.component_dir(local, "triton") / "kernel/a.so"
        self.assertEqual(target.read_text(encoding="utf-8"), "first")

        remote_file.write_text("second", encoding="utf-8")
        write_snapshot(remote)
        second = self.make_manager(str(remote), local_root=local)
        second_summary = second.prepare()

        self.assertEqual(second_summary["cache_state"], "snapshot_hit")
        self.assertEqual(target.read_text(encoding="utf-8"), "second")

    def test_concurrent_prepare_same_local_is_serialized(self):
        remote = self.root / "remote"
        local = self.root / "local"
        first = self.make_manager(str(remote), local_root=local)
        second = self.make_manager(str(remote), local_root=local)
        remote_file = self.component_dir(remote, "triton") / "kernel/a.so"
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("shared", encoding="utf-8")
        write_snapshot(remote)

        first_started = threading.Event()
        release_first = threading.Event()
        second_entered_extract = threading.Event()
        original_first_extract = first._extract_snapshot
        original_second_extract = second._extract_snapshot

        def slow_first_extract(archive: Path):
            first_started.set()
            self.assertTrue(release_first.wait(timeout=5))
            return original_first_extract(archive)

        def recording_second_extract(archive: Path):
            second_entered_extract.set()
            return original_second_extract(archive)

        results: list[dict] = []
        with (
            mock.patch.object(
                first, "_extract_snapshot", side_effect=slow_first_extract
            ),
            mock.patch.object(
                second, "_extract_snapshot", side_effect=recording_second_extract
            ),
        ):
            leader = threading.Thread(target=lambda: results.append(first.prepare()))
            follower = threading.Thread(target=lambda: results.append(second.prepare()))
            leader.start()
            self.assertTrue(first_started.wait(timeout=5))
            follower.start()
            time.sleep(0.1)
            self.assertFalse(second_entered_extract.is_set())
            release_first.set()
            leader.join(timeout=5)
            follower.join(timeout=5)

        self.assertEqual([r["result"] for r in results], ["success", "success"])
        self.assertEqual(
            (self.component_dir(local, "triton") / "kernel/a.so").read_text(), "shared"
        )

    def test_prepare_rejects_unsafe_snapshot_paths(self):
        remote, manager = self.make_remote_manager()
        write_raw_snapshot(remote, {"../../escape.so": b"bad"})

        summary = manager.prepare()

        self.assertEqual(summary["result"], "failed")
        self.assertEqual(summary["cache_state"], "snapshot_error")
        self.assertFalse((self.root / "escape.so").exists())

    def test_prepare_skips_other_component_scopes(self):
        remote, manager = self.make_remote_manager()
        write_raw_snapshot(
            remote,
            {
                "deep_gemm/cache/old.cubin": b"old",
                "triton/kernel/keep.so": b"keep",
            },
        )

        summary = manager.prepare()

        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["extracted_files"], 1)
        self.assertFalse(
            (manager.component_dirs["deep_gemm"] / "cache/old.cubin").exists()
        )
        self.assertEqual(
            (manager.component_dirs["triton"] / "kernel/keep.so").read_text(
                encoding="utf-8"
            ),
            "keep",
        )

    def test_upload_file_publishes_snapshot_without_writing_remote_file_tree(self):
        remote, manager = self.make_remote_manager()
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        remote_root = jit_cache_module.component_cache_dir(remote, component)
        local_file = local_root / "kernel/a.so"
        remote_file = remote_root / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.parent.mkdir(parents=True, exist_ok=True)
        remote_file.write_text("old", encoding="utf-8")
        local_file.write_text("new", encoding="utf-8")

        self.assertTrue(manager.upload_file(component, "kernel/a.so"))
        summary = manager.sync_once()

        self.assertEqual(remote_file.read_text(encoding="utf-8"), "old")
        self.assertEqual(
            snapshot_members(snapshot_path(remote)), {"triton/kernel/a.so": b"new"}
        )
        self.assertEqual(summary["result"], "success")

    def test_upload_file_reports_failures(self):
        _remote, manager = self.make_remote_manager()
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        local_file = local_root / "kernel/a.so"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text("so", encoding="utf-8")

        with mock.patch.object(
            manager, "_publish_snapshot", side_effect=OSError("publish failed")
        ):
            self.assertFalse(manager.upload_file(component, "kernel/a.so"))

    def test_sync_once_publishes_one_fixed_snapshot_file(self):
        remote, manager = self.make_remote_manager()
        manager.prepare()
        self.upload_file_helper(manager, "triton", "kernel/a.so")

        summary = manager.sync_once()
        snapshot = snapshot_path(remote)

        self.assertEqual(summary["result"], "success")
        self.assertEqual(snapshot, remote / jit_cache_module.SNAPSHOT_NAME)
        self.assertEqual(snapshot_members(snapshot), {"triton/kernel/a.so": b"triton"})

    def test_snapshot_publish_overwrites_existing_snapshot(self):
        remote, manager = self.make_remote_manager()
        manager.prepare()
        self.upload_file_helper(manager, "triton", "kernel/a.so")
        manager.sync_once()
        self.upload_file_helper(manager, "triton", "kernel/b.so")
        manager.sync_once()

        snapshot = snapshot_path(remote)

        self.assertEqual(
            snapshot_members(snapshot),
            {"triton/kernel/a.so": b"triton", "triton/kernel/b.so": b"triton"},
        )
        self.assertEqual(
            [p.name for p in remote.glob("*.tar.zst")], [jit_cache_module.SNAPSHOT_NAME]
        )

    def test_snapshot_publish_keeps_existing_remote_members(self):
        remote, manager = self.make_remote_manager()
        write_raw_snapshot(
            remote,
            {
                "triton/kernel/a.so": b"remote",
                "triton/kernel/other.so": b"other",
            },
        )
        self.upload_file_helper(manager, "triton", "kernel/a.so")
        self.upload_file_helper(manager, "triton", "kernel/b.so")

        self.assertEqual(
            snapshot_members(snapshot_path(remote)),
            {
                "triton/kernel/a.so": b"remote",
                "triton/kernel/other.so": b"other",
                "triton/kernel/b.so": b"triton",
            },
        )

    def test_watcher_uses_component_specific_completion_events(self):
        _remote, manager = self.make_remote_manager()
        calls = []
        cases = (
            ("flashinfer", "kernel.so", "created", "closed"),
            ("torch_extensions", "extension.so", "moved", "closed"),
            ("triton", "kernel/a.so", "closed", "moved"),
            ("triton_autotune", "kernel.json", "created", "closed"),
            ("deep_gemm", "cache/kernel.cubin", "closed", "created"),
            ("deep_gemm", "cache/kernel2.cubin", "closed", "moved"),
        )
        for component_name, rel, ignored_event, upload_event in cases:
            component = self.component(component_name)
            root = manager.component_dirs[component_name]
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
            handler = jit_cache_module._JitFileEventHandler(
                component,
                root,
                lambda component, rel: calls.append((component.name, rel)) or True,
            )
            before = len(calls)

            handler.on_any_event(FakeFileEvent(ignored_event, str(path)))
            self.assertEqual(len(calls), before)

            if upload_event == "moved":
                event = FakeFileEvent("moved", str(path.with_suffix(".tmp")), str(path))
            else:
                event = FakeFileEvent(upload_event, str(path))
            handler.on_any_event(event)
            self.assertEqual(len(calls), before + 1)

    def test_file_event_handler_uploads_syncable_file(self):
        remote, manager = self.make_remote_manager()
        manager.prepare()
        local_root = manager.component_dirs["triton"]
        final_file = local_root / "kernel/a.cubin"
        final_file.parent.mkdir(parents=True, exist_ok=True)
        final_file.write_text("cubin", encoding="utf-8")
        handler = manager._make_handler(self.component("triton"), local_root)

        handler.on_any_event(
            FakeFileEvent("moved", str(final_file.with_suffix(".tmp")), str(final_file))
        )

        self.assertEqual(
            snapshot_members(snapshot_path(remote)), {"triton/kernel/a.cubin": b"cubin"}
        )

    def test_remote_config_mounts_uri_before_validation(self):
        from rtp_llm.utils.fuser import MountRwMode

        mounted_remote = self.root / "mounted_remote"
        mounted_remote.mkdir()

        with mock.patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local",
            return_value=str(mounted_remote),
        ) as fetch_remote:
            manager = self.make_manager("oss://bucket/jit-cache", create_remote=False)

        fetch_remote.assert_called_once_with(
            "oss://bucket/jit-cache", MountRwMode.RWMODE_RW
        )
        self.assertEqual(manager.remote_root, mounted_remote)

    def test_debounce_batches_rapid_uploads_into_single_publish(self):
        remote, manager = self.make_remote_manager(debounce_s=0.1)
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]

        publish_calls: list[None] = []
        original = manager._publish_snapshot

        def counting_publish():
            publish_calls.append(None)
            return original()

        with mock.patch.object(
            manager, "_publish_snapshot", side_effect=counting_publish
        ):
            for i in range(5):
                path = local_root / f"kernel/k{i}.so"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
                manager.upload_file(component, f"kernel/k{i}.so")

            time.sleep(0.5)

        self.assertEqual(
            len(publish_calls),
            1,
            "5 rapid uploads should trigger exactly 1 debounced publish",
        )

    def test_debounce_keeps_existing_pending_timer(self):
        _remote, manager = self.make_remote_manager(debounce_s=60)
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        first = local_root / "kernel/a.so"
        second = local_root / "kernel/b.so"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        self.assertTrue(manager.upload_file(component, "kernel/a.so"))
        timer = manager._debounce_timer
        self.assertIsNotNone(timer)
        self.assertTrue(manager.upload_file(component, "kernel/b.so"))

        self.assertIs(manager._debounce_timer, timer)

    def test_concurrent_publish_does_not_corrupt_snapshot(self):
        """Parallel _publish_snapshot calls must not produce a truncated archive."""
        remote, manager = self.make_remote_manager()
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        for i in range(10):
            p = local_root / f"kernel/k{i}.so"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 1024)

        errors: list[Exception] = []

        def publish_worker():
            try:
                manager._publish_snapshot()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=publish_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertFalse(errors, f"concurrent publish raised: {errors}")
        # Snapshot must be readable after concurrent writes.
        names = snapshot_member_names(snapshot_path(remote))
        self.assertGreater(len(names), 0)

    def test_snapshot_mklock_times_out_when_lock_dir_is_left_behind(self):
        remote = self.root / "remote"
        remote.mkdir()
        (remote / jit_cache_module.SNAPSHOT_LOCK_DIR_NAME).mkdir()

        with (
            mock.patch.object(jit_cache_module, "SNAPSHOT_LOCK_TIMEOUT_S", 0.01),
            mock.patch.object(jit_cache_module, "SNAPSHOT_LOCK_POLL_S", 0.001),
        ):
            with self.assertRaises(TimeoutError):
                with jit_cache_module._snapshot_mklock(remote):
                    pass

    def test_two_services_accumulate_entries_without_loss(self):
        """Simulates two services writing to the same remote dir sequentially
        (worst-case overlap without flock released between them) and verifies
        neither service's entries are lost."""
        remote = self.root / "remote"
        local_a = self.root / "local_a"
        local_b = self.root / "local_b"

        svc_a = self.make_manager(str(remote), local_root=local_a)
        svc_b = self.make_manager(str(remote), local_root=local_b)

        component = self.component("triton")

        # Service A writes its file and publishes.
        path_a = svc_a.component_dirs["triton"] / "kernel/a_kernel.so"
        path_a.parent.mkdir(parents=True, exist_ok=True)
        path_a.write_bytes(b"svc_a_kernel")
        svc_a.upload_file(component, "kernel/a_kernel.so")
        svc_a.sync_once()

        # Service B writes a *different* file and publishes.
        path_b = svc_b.component_dirs["triton"] / "kernel/b_kernel.so"
        path_b.parent.mkdir(parents=True, exist_ok=True)
        path_b.write_bytes(b"svc_b_kernel")
        svc_b.upload_file(component, "kernel/b_kernel.so")
        svc_b.sync_once()

        members = snapshot_members(snapshot_path(remote))
        self.assertIn(
            "triton/kernel/a_kernel.so",
            members,
            "service A entry lost after service B publish",
        )
        self.assertIn(
            "triton/kernel/b_kernel.so",
            members,
            "service B entry missing from snapshot",
        )
        self.assertEqual(members["triton/kernel/a_kernel.so"], b"svc_a_kernel")
        self.assertEqual(members["triton/kernel/b_kernel.so"], b"svc_b_kernel")

    def test_upload_file_returns_false_for_missing_file(self):
        _remote, manager = self.make_remote_manager()
        component = self.component("triton")

        result = manager.upload_file(component, "nonexistent/path.so")

        self.assertFalse(result)

    def test_sync_once_cancels_pending_debounce_timer(self):
        remote, manager = self.make_remote_manager(debounce_s=60)
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        path = local_root / "kernel/a.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

        manager.upload_file(component, "kernel/a.so")
        self.assertIsNotNone(manager._debounce_timer)

        manager.sync_once()

        self.assertIsNone(
            manager._debounce_timer,
            "sync_once should cancel the pending debounce timer",
        )
        self.assertTrue(snapshot_path(remote).is_file())

    def test_stop_cancels_debounce_timer(self):
        _remote, manager = self.make_remote_manager(debounce_s=60)
        component = self.component("triton")
        local_root = manager.component_dirs["triton"]
        path = local_root / "kernel/a.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")

        manager.upload_file(component, "kernel/a.so")
        self.assertIsNotNone(manager._debounce_timer)

        manager.stop()

        self.assertIsNone(
            manager._debounce_timer, "stop should cancel the pending debounce timer"
        )


class SnapshotPublishConsumerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(self, local_root: Path, remote_root: Path) -> JitCacheManager:
        remote_root.mkdir(parents=True, exist_ok=True)
        config = JITConfig()
        config.local_jit_dir = str(local_root)
        config.remote_jit_dir = str(remote_root)
        manager = JitCacheManager(config, debounce_s=0)
        manager.bootstrap()
        return manager

    def test_publish_then_consumer_extracts_snapshot(self):
        remote_root = self.root / "remote"
        first = self.make_manager(self.root / "local_first", remote_root)
        try:
            first.prepare()
            expected_members = set()
            for component in jit_cache_module.COMPONENT_SPECS:
                local_root = first.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                local_file = local_root / filename
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_text(component.name, encoding="utf-8")
                first.upload_file(component, filename)
                expected_members.add(
                    f"{component.name}/{jit_cache_module.component_snapshot_rel(component, filename)}"
                )
            first.sync_once("publish_consumer_test")
            snapshot = snapshot_path(remote_root)
            self.assertEqual(snapshot_member_names(snapshot), expected_members)
        finally:
            first.stop()

        for component in jit_cache_module.COMPONENT_SPECS:
            os.environ.pop(component.env_name, None)

        second = self.make_manager(self.root / "local_second", remote_root)
        try:
            prepare = second.prepare()
            self.assertEqual(prepare["result"], "success")
            self.assertEqual(
                prepare["extracted_files"], len(jit_cache_module.COMPONENT_SPECS)
            )
            for component in jit_cache_module.COMPONENT_SPECS:
                local_root = second.component_dirs[component.name]
                filename = f"kernel/{component.name}{component.sync_suffixes[0]}"
                self.assertEqual(
                    (local_root / filename).read_text(encoding="utf-8"), component.name
                )
        finally:
            second.stop()


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as jit_store
from rtp_llm.utils.test import jit_cache_smoke_test as smoke


class FakeFileEvent:
    def __init__(self, event_type: str, src: Path, dest: Path | None = None):
        self.event_type = event_type
        self.src_path = str(src)
        self.dest_path = str(dest) if dest else ""
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


def component(components, name: str):
    return next(item for item in components if item.name == name)


def snapshots(remote: Path) -> list[Path]:
    return sorted(remote.glob(f"*{jit_store.SNAPSHOT_SUFFIX}"))


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
        self, remote: Path | str | None = None, *, create_remote: bool = True
    ) -> jit.JitCacheManager:
        remote = self.root / "remote" if remote is None else remote
        if remote and create_remote:
            Path(remote).mkdir(parents=True, exist_ok=True)
        local = self.root / f"local_{len(self.managers)}"
        config = JITConfig()
        config.remote_jit_dir = str(remote or "")
        jit.setup_jit_cache_env(local)
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(local)), mock.patch.object(
            jit, "_DEFAULT_ROOT", Path(local).expanduser().absolute()
        ):
            manager = jit.JitCacheManager(config)
        manager.bootstrap()
        self.managers.append(manager)
        return manager

    @staticmethod
    def write_artifact(
        manager: jit.JitCacheManager, name: str, rel: str, data: bytes
    ) -> tuple[jit.Component, Path]:
        item = component(manager.components, name)
        path = item.local_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return item, path

    def test_nondefault_local_root_disables_remote(self):
        remote = self.root / "remote"
        remote.mkdir()
        config = JITConfig()
        config.remote_jit_dir = str(remote)
        with mock.patch.object(jit, "LOCAL_JIT_DIR", str(self.root / "local")):
            manager = jit.JitCacheManager(config)
        self.assertIsNone(manager.store)

    def test_restore_tolerates_missing_mtime_manifest(self):
        remote = self.root / "remote"
        remote.mkdir()
        legacy = self.root / "legacy" / "triton"
        legacy.mkdir(parents=True)
        (legacy / "k.cubin").write_bytes(b"x")
        with jit_store.zstd_tar(
            remote / f"{1:020d}{jit_store.SNAPSHOT_SUFFIX}", "w"
        ) as out:
            out.add(legacy.parent, arcname="")
        target = self.root / "restored"
        self.assertTrue(jit_store.RemoteSnapshotStore(remote).restore(target))
        self.assertEqual((target / "triton/k.cubin").read_bytes(), b"x")

    def test_external_cache_dir_is_preserved_and_not_watched(self):
        outside = self.root / "outside"
        os.environ["TRITON_CACHE_DIR"] = str(outside)
        manager = self.make_manager()

        with mock.patch.object(jit, "Observer", FakeObserver):
            manager.start_background_sync()

        watched = {path for _handler, path, _recursive in manager._observer.handlers}
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(outside))
        self.assertNotIn(
            str(component(manager.components, "triton").local_dir), watched
        )
        self.assertEqual(len(watched), len(manager.components) - 1)

    def test_sync_contract(self):
        cases = {
            "flashinfer": {
                "yes": (
                    "generated/op/kernel.cu",
                    "generated/op/config.inc",
                    "generated/op/api.h",
                    "cached/op/build.ninja",
                    "cached/op/.ninja_log",
                    "cached/op/.ninja_deps",
                    "cached/op/kernel.o",
                    "cached/op/kernel.so",
                ),
                "no": ("cached/op/rules.ninja", "tmp/op.so", "../op.so"),
            },
            "deep_gemm": {
                "yes": ("shape/kernel.cu", "shape/kernel.cubin"),
                "no": ("shape/kernel.so",),
            },
            "tensorrt_llm_deep_gemm": {
                "yes": ("shape/nvcc_kernel.cubin",),
                "no": ("shape/kernel.cubin",),
            },
            "torch_extensions": {
                "yes": (
                    "tipc/main.cpp",
                    "tipc/cuda.cu",
                    "tipc/build.ninja",
                    "tipc/.ninja_log",
                    "tipc/.ninja_deps",
                    "tipc/ipc.o",
                    "tipc/tipc.so",
                ),
                "no": ("tipc/lock",),
            },
            "tvm_ffi": {
                "yes": ("libtorch_c_dlpack_addon_torch28-cuda.so",),
                "no": ("mod/main.cpp", "mod/build.ninja", "addon.so.lock"),
            },
            "cute_dsl": {
                "yes": ("cute_dsl_0123456789abcdef.mlir",),
                "no": ("tmp.pid_123/cute_dsl_0123456789abcdef.mlir",),
            },
            "triton": {
                "yes": ("hash/kernel.json", "hash/kernel.cubin", "hash/driver.so"),
                "no": ("hash/kernel.autotune.json",),
            },
        }
        for name, contract in cases.items():
            item = component(jit.COMPONENTS, name)
            for rel in contract["yes"]:
                with self.subTest(component=name, path=rel):
                    self.assertTrue(item.should_sync(rel))
            for rel in contract["no"]:
                with self.subTest(component=name, path=rel):
                    self.assertFalse(item.should_sync(rel))

    def test_component_scopes_encode_binary_compatibility(self):
        trtllm = component(jit.COMPONENTS, "tensorrt_llm_deep_gemm").scope_func
        tvm_ffi = component(jit.COMPONENTS, "tvm_ffi").scope_func
        cute_dsl = component(jit.COMPONENTS, "cute_dsl").scope_func

        with mock.patch.object(jit, "_cuda_scope", return_value="cuda-12_9"):
            with mock.patch.object(jit, "_dist_version", return_value="0.3.1"):
                self.assertEqual(trtllm(), "cuda-12_9-flashinfer-0_3_1")
            with mock.patch.object(jit, "_dist_version", return_value="4.3.5"):
                self.assertEqual(cute_dsl(), "cuda-12_9-cutlass-dsl-4_3_5")
        with mock.patch.object(jit, "_torch_extensions_scope", return_value="torch"):
            with mock.patch.object(jit, "_dist_version", return_value="0.1.8"):
                self.assertEqual(tvm_ffi(), "tvm-ffi-0_1_8-torch")

    def test_snapshot_merge_rotation_and_restore(self):
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
                store.publish_snapshot({name: source})

        consumer = self.root / "consumer"
        self.assertTrue(store.restore(consumer))
        restored = {
            path.relative_to(consumer).as_posix(): path.read_bytes()
            for path in consumer.rglob("*")
            if path.is_file()
        }
        self.assertEqual(len(snapshots(remote)), 2)
        self.assertEqual(restored, expected)

    def test_restored_ninja_tree_has_no_work(self):
        ninja = shutil.which("ninja")
        if ninja is None:
            self.skipTest("ninja is not available")

        producer = self.root / "producer"
        producer.mkdir()
        source = producer / "input.cu"
        source.write_text("payload")
        source_mtime_ns = 1_783_782_660_123_456_789
        output_mtime_ns = 1_783_782_662_123_456_789
        os.utime(source, ns=(source_mtime_ns, source_mtime_ns))
        (producer / "build.ninja").write_text(
            "rule copy\n"
            "  command = cp $in $out && "
            f"touch -d @{output_mtime_ns // 1_000_000_000}."
            f"{output_mtime_ns % 1_000_000_000:09d} $out\n"
            "build output.o: copy input.cu\n"
        )
        subprocess.run([ninja, "-C", str(producer)], check=True, capture_output=True)

        remote = self.root / "remote"
        remote.mkdir()
        store = jit_store.RemoteSnapshotStore(remote)
        prefix = "flashinfer/cached_ops/ninja_test"
        store.publish_snapshot(
            {f"{prefix}/{path.name}": path for path in producer.iterdir()}
        )
        consumer = self.root / "consumer"
        store.restore(consumer)
        result = subprocess.run(
            [ninja, "-C", str(consumer / prefix), "-n", "-d", "explain"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.stdout,
            f"ninja: Entering directory `{consumer / prefix}'\n"
            "ninja: no work to do.\n",
        )
        self.assertEqual(result.stderr, "")

    def test_restore_finishes_before_observer_starts(self):
        producer = self.root / "cached.cubin"
        producer.write_bytes(b"cached")
        remote = self.root / "remote"
        remote.mkdir()
        jit_store.RemoteSnapshotStore(remote).publish_snapshot(
            {"triton/a.cubin": producer}
        )
        manager = self.make_manager(remote)

        class CheckingObserver(FakeObserver):
            def start(inner_self):
                inner_self.restored = (
                    manager.local_root / "triton/a.cubin"
                ).read_bytes()
                super().start()

        with mock.patch.object(jit, "Observer", CheckingObserver):
            manager.start_background_sync()

        self.assertEqual(manager._observer.restored, b"cached")

    def test_terminal_events_publish_complete_artifacts(self):
        manager = self.make_manager()
        events = []
        for name, rel, event_type in (
            ("flashinfer", "cached/op.so", "closed"),
            ("deep_gemm", "shape/kernel.cubin", "moved"),
            ("cute_dsl", "shape/kernel.mlir", "moved"),
            ("triton", "hash/kernel.cubin", "created"),
        ):
            item, path = self.write_artifact(manager, name, rel, name.encode())
            handler = jit._JitFileEventHandler(item, manager.stage_delta_file)
            if event_type == "moved":
                event = FakeFileEvent(event_type, self.root / "tmp", path)
            else:
                event = FakeFileEvent(event_type, path)
            handler.on_any_event(event)
            events.append(
                (path.relative_to(manager.local_root).as_posix(), name.encode())
            )

        manager.flush_delta2remote()
        consumer = self.root / "consumer"
        manager.store.restore(consumer)
        self.assertEqual(
            {
                path.relative_to(consumer).as_posix(): path.read_bytes()
                for path in consumer.rglob("*")
                if path.is_file()
            },
            dict(events),
        )

    def test_stop_flushes_pending_artifact(self):
        manager = self.make_manager()
        with mock.patch.object(jit, "Observer", FakeObserver):
            manager.start_background_sync()
            item, path = self.write_artifact(
                manager, "triton", "hash/final.cubin", b"final"
            )
            manager.stage_delta_file(item, "hash/final.cubin")
            manager.stop()

        consumer = self.root / "consumer"
        manager.store.restore(consumer)
        self.assertEqual(
            (consumer / path.relative_to(manager.local_root)).read_bytes(), b"final"
        )

    def test_missing_remote_disables_background_sync(self):
        manager = self.make_manager(self.root / "missing", create_remote=False)
        with mock.patch.object(jit, "Observer", FakeObserver):
            manager.start_background_sync()
        self.assertIsNone(manager.store)
        self.assertIsNone(manager._observer)

    def test_remote_uri_uses_rw_fuser_mount(self):
        mounted = self.root / "mounted"
        mounted.mkdir()
        with mock.patch(
            "rtp_llm.utils.fuser.fetch_remote_file_to_local",
            return_value=str(mounted),
        ) as fetch:
            manager = self.make_manager("oss://bucket/cache", create_remote=False)

        self.assertEqual(manager.store.remote_root, mounted)
        self.assertEqual(fetch.call_args.args[0], "oss://bucket/cache")
        self.assertTrue(fetch.call_args.args[2])

    def test_runtime_identity_detects_same_content_atomic_replace(self):
        local_root = self.root / "identity"
        item = component(jit.COMPONENTS, "triton").resolve(local_root)
        path = item.local_dir / "hash/kernel.cubin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"same")
        mtime_ns = path.stat().st_mtime_ns
        before = smoke._local_runtime_identity(local_root)["triton/hash/kernel.cubin"]

        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"same")
        os.utime(replacement, ns=(mtime_ns, mtime_ns))
        replacement.replace(path)
        after = smoke._local_runtime_identity(local_root)["triton/hash/kernel.cubin"]

        self.assertEqual(after["sha256"], before["sha256"])
        self.assertEqual(after["mtime_ns"], before["mtime_ns"])
        self.assertNotEqual(after["inode"], before["inode"])


if __name__ == "__main__":
    unittest.main()

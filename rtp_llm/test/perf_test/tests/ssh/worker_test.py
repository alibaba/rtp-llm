"""Performance framework behavior and failure regressions."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.ssh.source import build_identity, resolve_source
from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT


def load_worker():
    spec = importlib.util.spec_from_file_location(
        "cold_ssh_worker", PERF_TEST_ROOT / "ssh" / "worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkerTest(unittest.TestCase):

    def test_file_entry_describe_has_no_runtime_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(PERF_TEST_ROOT / "ssh" / "worker.py"),
                "--operation",
                "describe",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema_version"], 1)

    def test_cancel_rejects_reused_pid_and_does_not_signal_neighbor(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = {
                "schema_version": 1,
                "run_id": "run1",
                "member_id": "0",
                "run_dir": str(root),
            }
            proc = subprocess.Popen(
                [sys.executable, "-c", "import signal;signal.pause()"],
                start_new_session=True,
            )
            self.addCleanup(lambda: proc.poll() is None and proc.kill())
            owner = {
                "run_id": "run1",
                "member_id": "0",
                "pid": proc.pid,
                "pgid": proc.pid,
                "start_ticks": "incorrect",
            }
            (root / "owner.json").write_text(json.dumps(owner))
            with self.assertRaisesRegex(ValueError, "identity"):
                worker.cancel(descriptor)
            self.assertIsNone(proc.poll())
            owner["start_ticks"] = worker.process_identity(proc.pid)
            (root / "owner.json").write_text(json.dumps(owner))
            worker.cancel(descriptor)
            self.assertIsNotNone(proc.wait(timeout=5))

    def test_build_driver_receipt_can_be_reused_and_corrupt_native_is_rejected(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (source_dir, build_dir, run_dir) = (
                root / "source",
                root / "build",
                root / "run",
            )
            source_dir.mkdir()
            run_dir.mkdir()
            driver = root / "driver.py"
            program = "import hashlib,json,pathlib,sys\np=json.loads(pathlib.Path(sys.argv[-1]).read_text())\nartifact=pathlib.Path(p['source_dir'])/'lib.so'; artifact.write_bytes(b'native')\nr={'schema_version':1,'status':'completed','bazel_exit_code':0,'source_sha':p['source_sha'],\n   'artifacts':{str(artifact):hashlib.sha256(artifact.read_bytes()).hexdigest()},\n   'driver_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),\n   'toolchain':TOOLCHAIN}\npathlib.Path(p['receipt_path']).write_text(json.dumps(r))\n".replace(
                "TOOLCHAIN", repr(worker.toolchain_snapshot())
            )
            driver.write_text(program)
            source = {
                "ref": "a" * 40,
                "build_policy": "build",
                "build_driver": str(driver),
                "bazel_config": "rocm",
                "bazel_targets": ["//:native"],
                "bazel_args": [],
            }
            descriptor = {
                "schema_version": 1,
                "run_id": "run1",
                "member_id": "0",
                "run_dir": str(run_dir),
                "source_dir": str(source_dir),
                "build_dir": str(build_dir),
                "source": source,
            }
            with patch.object(worker, "preflight", return_value={}):
                built = worker.build(descriptor)
                self.assertEqual(built["status"], "completed")
                owner_path = run_dir / "build_owner.json"
                self.assertTrue(json.loads(owner_path.read_text())["cleanup_complete"])
                with patch.object(
                    worker, "group_members", return_value={"123": "reused"}
                ):
                    self.assertTrue(worker.status(descriptor)["cleanup_complete"])
                descriptor["source"] = {
                    **source,
                    "build_policy": "reuse",
                    "build_driver": None,
                }
                self.assertEqual(
                    worker.build(descriptor)["build_identity"], built["build_identity"]
                )
                (source_dir / "lib.so").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    worker.build(descriptor)

            # If ownership cannot be written, the detached driver never runs.
            descriptor["source"] = source
            (source_dir / "lib.so").unlink()
            write = worker.atomic_json
            launched = []
            popen = subprocess.Popen

            def launch(*args, **kwargs):
                process = popen(*args, **kwargs)
                launched.append(process)
                return process

            def fail_owner(path, value):
                if Path(path).name == "build_owner.json":
                    self.assertFalse((source_dir / "lib.so").exists())
                    raise OSError("owner disk full")
                return write(path, value)

            with patch.object(worker, "preflight", return_value={}), patch.object(
                worker, "atomic_json", side_effect=fail_owner
            ), patch.object(worker.subprocess, "Popen", side_effect=launch):
                with self.assertRaisesRegex(OSError, "owner disk full"):
                    worker.build(descriptor)
            for process in launched:
                self.assertNotEqual(0, process.wait(timeout=5))
            self.assertFalse((source_dir / "lib.so").exists())

    def test_verify_bundle_rejects_modified_plan_bytes(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/execution_plan.json").write_text("{}")
            descriptor = {
                "schema_version": 1,
                "run_id": "run1",
                "member_id": "0",
                "run_dir": str(root),
                "bundle_hashes": {"config/execution_plan.json": "0" * 64},
            }
            with self.assertRaisesRegex(ValueError, "bundle hash mismatch"):
                worker.verify_bundle(descriptor)

    def test_cancel_before_start_fences_late_unacknowledged_launch(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            descriptor = {
                "schema_version": 1,
                "run_id": "run1",
                "member_id": "0",
                "run_dir": directory,
            }
            worker.cancel(descriptor)
            with patch.object(worker, "verify_bundle", return_value={}):
                with patch.object(
                    worker.subprocess,
                    "Popen",
                    side_effect=AssertionError("must not launch"),
                ):
                    with self.assertRaisesRegex(ValueError, "cancelled"):
                        worker.start(
                            descriptor, str(Path(directory) / "descriptor.json")
                        )

    def test_runtime_native_dependency_metadata_and_bytes_must_match_receipt(self):
        import hashlib
        import platform

        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "deep_gemm.so"
            binary.write_bytes(b"native")

            class Distribution:
                version = "2.8.1"
                files = ["deep_gemm.so"]

                def read_text(self, name):
                    return name + "-metadata"

                def locate_file(self, name):
                    return Path(directory) / name

            machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(
                platform.machine().lower(), platform.machine().lower()
            )
            expected = {
                "deep_gemm": {
                    "version": "2.8.1",
                    "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                    "machine": machine,
                    "wheel_sha256": hashlib.sha256(b"WHEEL-metadata").hexdigest(),
                    "record_sha256": hashlib.sha256(b"RECORD-metadata").hexdigest(),
                    "binaries": {"deep_gemm.so": hashlib.sha256(b"native").hexdigest()},
                }
            }
            with patch("importlib.metadata.distribution", return_value=Distribution()):
                worker.verify_runtime_dependencies(expected, "cuda13")
                binary.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "differs"):
                    worker.verify_runtime_dependencies(expected, "cuda13")
            with self.assertRaisesRegex(ValueError, "requires"):
                worker.verify_runtime_dependencies({}, "cuda13")

    def launch_owned_group_with_stubborn_child(self, worker, root):
        import selectors

        child_code = "import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(os.getpid(),flush=True);signal.pause()"
        parent_code = "import subprocess,sys,signal;p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE,text=True);print(p.stdout.readline().strip(),flush=True);signal.pause()"
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code, child_code],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        def cleanup():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()

        self.addCleanup(cleanup)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(5), "child did not become ready")
        child = int(process.stdout.readline())
        owner = {
            "run_id": "run1",
            "member_id": "0",
            "pid": process.pid,
            "pgid": process.pid,
            "start_ticks": worker.process_identity(process.pid),
        }
        (root / "owner.json").write_text(json.dumps(owner))
        (root / "status.json").write_text(json.dumps({"state": "running"}))
        descriptor = {
            "schema_version": 1,
            "run_id": "run1",
            "member_id": "0",
            "run_dir": str(root),
        }
        return (process, child, owner, descriptor)

    def test_cancel_kills_stubborn_group_child_after_leader_exits(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            (process, child, owner, descriptor) = (
                self.launch_owned_group_with_stubborn_child(worker, Path(directory))
            )
            with patch.object(worker, "CANCEL_GRACE_SECONDS", 0.05):
                result = worker.cancel(descriptor)
            process.wait(timeout=5)
            self.assertIsNone(
                worker.process_identity(child), "owned child survived cancellation"
            )
            self.assertFalse(result["alive"])
            self.assertTrue(result["cleanup_complete"])

    def test_status_and_clean_retain_evidence_when_leader_dies_before_child(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (process, child, owner, descriptor) = (
                self.launch_owned_group_with_stubborn_child(worker, root)
            )
            owner["members"] = {str(child): worker.process_identity(child)}
            (root / "owner.json").write_text(json.dumps(owner))
            process.terminate()
            process.wait(timeout=5)
            current = worker.status(descriptor)
            self.assertTrue(
                current["alive"], "leader death hid owned residual processes"
            )
            self.assertFalse(current["cleanup_complete"])
            with self.assertRaises(ValueError):
                worker.clean(descriptor)
            self.assertTrue((root / "owner.json").exists())
            with patch.object(worker, "CANCEL_GRACE_SECONDS", 0.05):
                worker.cancel(descriptor)
            self.assertIsNone(worker.process_identity(child))

    def launch_cpu_supervisor(self, worker, root, *, leave_child):
        """Exercise start/supervise; replace only model/build prerequisites."""
        driver = root / "cpu_supervisor.py"
        driver.write_text(
            "import contextlib,importlib.util,pathlib,signal,subprocess,sys,types\n"
            f"spec=importlib.util.spec_from_file_location('worker',{str(PERF_TEST_ROOT / 'ssh/worker.py')!r})\n"
            "worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)\n"
            "worker.runtime_build_lease=lambda descriptor:contextlib.nullcontext()\n"
            "worker.verify_bundle=lambda descriptor:None\n"
            "worker.verify_source=lambda descriptor:None\n"
            "worker.verify_build_receipt=lambda *args,**kwargs:None\n"
            "device=types.ModuleType('rtp_llm.test.utils.device_resource')\n"
            "device.get_cuda_info=lambda:['cpu-test'];sys.modules[device.__name__]=device\n"
            "def measure(descriptor,root,cuda_info,check_cancelled):\n"
            + (
                "    child=subprocess.Popen([sys.executable,'-c',"
                + repr(
                    "import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    "print(os.getpid(),flush=True);signal.pause()"
                )
                + "],stdout=subprocess.PIPE,text=True)\n"
                "    print(child.stdout.readline().strip(),flush=True)\n"
                "    signal.pause()\n"
                if leave_child
                else "    print('completed',flush=True)\n    return 0\n"
            )
            + "worker._measure_locked=measure\n"
            "raise SystemExit(worker.main())\n"
        )
        (root / "build_receipt.json").write_text("{}")
        descriptor = {
            "schema_version": 1,
            "run_id": "run1",
            "member_id": "0",
            "run_dir": str(root),
            "source_dir": str(root),
        }
        descriptor_path = root / "descriptor.json"
        descriptor_path.write_text(json.dumps(descriptor))
        processes = []
        popen = subprocess.Popen

        def launch(*args, **kwargs):
            kwargs["stdout"] = subprocess.PIPE
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(worker, "verify_bundle"), patch.object(
            worker, "verify_source"
        ), patch.object(worker, "__file__", str(driver)), patch.object(
            worker.subprocess, "Popen", side_effect=launch
        ):
            worker.start(descriptor, descriptor_path)
        process = processes[0]
        owned = {str(process.pid): worker.process_identity(process.pid)}

        def cleanup():
            for pid, ticks in owned.items():
                if worker.process_identity(int(pid)) == ticks and ticks is not None:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait(timeout=5)
            process.stdout.close()

        self.addCleanup(cleanup)
        import selectors

        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(5), "CPU measurement did not become ready")
        ready = process.stdout.readline().decode().strip()
        self.assertTrue(ready, (root / "supervisor.log").read_text())
        owned.update(worker.group_members(process.pid))
        return descriptor, process, ready

    def test_cancel_reclaims_descendants_after_supervisor_is_killed(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, process, ready = self.launch_cpu_supervisor(
                worker, root, leave_child=True
            )
            child = int(ready)
            process.kill()
            process.wait(timeout=5)
            self.assertIsNotNone(worker.process_identity(child))
            with patch.object(worker, "CANCEL_GRACE_SECONDS", 0.05):
                result = worker.cancel(descriptor)
            self.assertIsNone(worker.process_identity(child))
            self.assertTrue(result["cleanup_complete"])

    def test_completed_supervisor_leaves_no_owned_processes(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor, process, _ready = self.launch_cpu_supervisor(
                worker, root, leave_child=False
            )
            result = worker.wait(descriptor)
            process.wait(timeout=5)
            self.assertFalse(result["alive"])
            self.assertEqual("completed", result["state"])
            self.assertTrue(result["cleanup_complete"])

    def assert_environment_lock_contract(
        self, *, parent_exclusive, child_exclusive, blocked
    ):
        import fcntl
        import selectors

        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (first, second) = (root / "build-identity-a", root / "build-identity-b")
            first.mkdir()
            second.mkdir()
            child_code = "import fcntl,os,runpy,sys\nns=runpy.run_path(sys.argv[1]);os.write(1,b'R')\nwith ns['environment_lock'](exclusive=sys.argv[3]=='exclusive'):\n    with open(sys.argv[2]+'/.lock','a') as build:\n        fcntl.flock(build,fcntl.LOCK_EX);os.write(1,b'A')\n"
            with patch.dict(os.environ, {"HOME": str(root)}):
                with worker.environment_lock(exclusive=parent_exclusive):
                    with (first / ".lock").open("a") as build:
                        fcntl.flock(build, fcntl.LOCK_EX)
                        child = subprocess.Popen(
                            [
                                sys.executable,
                                "-c",
                                child_code,
                                str(PERF_TEST_ROOT / "ssh/worker.py"),
                                str(second),
                                "exclusive" if child_exclusive else "shared",
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        def cleanup():
                            if child.poll() is None:
                                child.kill()
                            child.wait(timeout=5)
                            child.stdout.close()
                            child.stderr.close()

                        self.addCleanup(cleanup)
                        self.assertEqual(child.stdout.read(1), b"R")
                        with selectors.DefaultSelector() as selector:
                            selector.register(child.stdout, selectors.EVENT_READ)
                            if blocked:
                                self.assertFalse(
                                    selector.select(0.05),
                                    "shared Python environment lock did not block",
                                )
                            else:
                                self.assertTrue(
                                    selector.select(5),
                                    "unrelated read-only builds should share the environment",
                                )
                                self.assertEqual(child.stdout.read(1), b"A")
                if blocked:
                    with selectors.DefaultSelector() as selector:
                        selector.register(child.stdout, selectors.EVENT_READ)
                        self.assertTrue(
                            selector.select(5), "environment lock was not released"
                        )
                        self.assertEqual(child.stdout.read(1), b"A")
                (_output, error) = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, error.decode())

    def test_dependency_install_blocks_other_build_identity_measurement(self):
        self.assert_environment_lock_contract(
            parent_exclusive=True, child_exclusive=False, blocked=True
        )

    def test_measurement_blocks_other_build_identity_dependency_install(self):
        self.assert_environment_lock_contract(
            parent_exclusive=False, child_exclusive=True, blocked=True
        )

    def test_readonly_builds_share_python_environment_across_identities(self):
        self.assert_environment_lock_contract(
            parent_exclusive=False, child_exclusive=False, blocked=False
        )

    def test_remote_tracked_source_drift_is_rejected_even_when_head_matches(self):
        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.py").write_text("value = 1")
            for args in (
                ["init"],
                ["config", "user.email", "test@example.invalid"],
                ["config", "user.name", "Test"],
                ["add", "code.py"],
                ["commit", "-m", "source"],
            ):
                subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            descriptor = {"source_dir": str(root), "source": {"ref": revision}}
            self.assertEqual(worker.verify_source(descriptor), revision)
            (root / "code.py").write_text("value = 2")
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                revision,
            )
            with self.assertRaisesRegex(ValueError, "uncommitted tracked"):
                worker.verify_source(descriptor)

    def test_supervisor_holds_build_lease_before_source_or_runtime_checks(self):
        import contextlib
        import io

        worker = load_worker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            descriptor = {
                "schema_version": 1,
                "run_id": "run1",
                "member_id": "0",
                "run_dir": str(root),
                "build_dir": str(build_dir),
            }
            observed = []
            child_code = "import fcntl,sys\nwith open(sys.argv[1],'a') as lock:\n    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n    except BlockingIOError: sys.exit(71)\n"

            def check_source(_descriptor):
                observed.append(
                    subprocess.run(
                        [sys.executable, "-c", child_code, str(build_dir / ".lock")],
                        check=False,
                    ).returncode
                )
                raise RuntimeError("stop before native imports")

            previous = {
                sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
            }
            try:
                with patch.dict(os.environ, {"HOME": str(root)}):
                    with patch.object(worker, "verify_bundle", return_value={}):
                        with patch.object(
                            worker, "verify_source", side_effect=check_source
                        ):
                            with contextlib.redirect_stderr(io.StringIO()):
                                self.assertEqual(worker.supervise(descriptor), 1)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            self.assertEqual(
                observed,
                [71],
                "source checks happened before acquiring the native build lease",
            )


class SourceTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.root / "code.py").write_text("value = 1\n")
        self.git("add", "code.py")
        self.git("commit", "-m", "test")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def source(self):
        return {
            "worktree": str(self.root),
            "ref": "auto",
            "build_policy": "reuse",
            "build_driver": None,
            "bazel_config": "cuda13",
            "bazel_targets": ["//:target"],
            "bazel_args": [],
        }

    def test_auto_freezes_exact_public_commit_and_untracked_is_excluded(self):
        (self.root / "personal.json").write_text("{}")
        frozen = resolve_source(self.source(), source_dir=self.root)
        self.assertEqual(frozen["ref"], self.git("rev-parse", "HEAD"))
        self.assertEqual(frozen["worktree"], str(self.root))

    def test_dirty_tracked_code_is_rejected(self):
        (self.root / "code.py").write_text("value = 2\n")
        with self.assertRaisesRegex(ConfigurationError, "uncommitted"):
            resolve_source(self.source(), source_dir=self.root)

    def test_build_identity_changes_with_flags(self):
        source = resolve_source(self.source(), source_dir=self.root)
        changed = {**source, "bazel_args": ["--jobs=2"]}
        self.assertNotEqual(build_identity(source), build_identity(changed))


if __name__ == "__main__":
    unittest.main()

"""Versioned SSH node worker. Cold operations import only the standard library.

Execute this file directly before the engine dependencies have been installed.
The supervisor owns one run's process group; no process-name cleanup is used.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

TERMINAL = {"completed", "failed", "interrupted", "lost"}
CANCEL_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 5


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def toolchain_snapshot():
    """Standard-library build ABI identity shared with the machine driver."""
    import platform
    import sysconfig

    result = {
        "python_abi": sysconfig.get_config_var("SOABI"),
        "python_cache_tag": sys.implementation.cache_tag,
        "machine": platform.machine(),
    }
    for name, base, relative in (
        ("cuda", "/usr/local/cuda", "version.json"),
        ("rocm", "/opt/rocm", ".info/version"),
    ):
        root = Path(base)
        if root.exists():
            marker = root / relative
            result[name] = {
                "root": str(root.resolve()),
                "version_hash": digest(marker) if marker.exists() else None,
            }
        else:
            result[name] = None
    return result


def verify_runtime_dependencies(dependencies, platform):
    """Verify installed native-wheel bytes without importing the extension."""
    if platform not in {"cuda13", "cuda13_arm"}:
        if dependencies not in ({}, None):
            raise ValueError("unexpected runtime dependencies for this platform")
        return
    import importlib.metadata
    import platform as host_platform

    if not isinstance(dependencies, dict) or set(dependencies) != {"deep_gemm"}:
        raise ValueError("CUDA13 build receipt requires deep_gemm runtime dependencies")
    try:
        distribution = importlib.metadata.distribution("deep-gemm")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("installed deep-gemm dependency is missing") from error
    wheel, record = distribution.read_text("WHEEL"), distribution.read_text("RECORD")
    if not wheel or not record:
        raise ValueError("installed deep-gemm metadata is incomplete")
    machine = host_platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    actual = {
        "version": distribution.version,
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "machine": machine,
        "wheel_sha256": hashlib.sha256(wheel.encode()).hexdigest(),
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "binaries": {
            str(name): digest(distribution.locate_file(name))
            for name in distribution.files or ()
            if str(name).endswith(".so")
        },
    }
    if not actual["binaries"] or actual != dependencies["deep_gemm"]:
        raise ValueError("installed deep-gemm dependency differs from build receipt")


@contextmanager
def environment_lock(*, exclusive):
    """Serialize site-packages mutation against all builds and measurements."""
    identity = json_hash(
        {
            "executable": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).resolve()),
        }
    )
    root = Path.home() / ".cache" / "rtp-llm" / "perf-env-locks"
    root.mkdir(parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise ValueError(
            "Python environment lock directory is not owned by current user"
        )
    fd = os.open(
        root / (identity + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "a") as lock:
        if os.fstat(lock.fileno()).st_uid != os.getuid():
            raise ValueError("Python environment lock is not owned by current user")
        # Lock order throughout this module is environment -> build identity.
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def verify_source(descriptor):
    source_dir = Path(descriptor["source_dir"])
    actual = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != descriptor["source"]["ref"]:
        raise ValueError("remote source SHA differs from frozen source")
    checked = subprocess.run(
        ["git", "-C", str(source_dir), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if checked.returncode:
        raise ValueError(
            "remote source has uncommitted tracked changes; refusing to execute"
        )
    return actual


def process_record(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start_ticks": fields[19], "pgid": int(fields[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_identity(pid):
    record = process_record(pid)
    return record["start_ticks"] if record else None


def group_members(pgid):
    members = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            record = process_record(int(path.name))
        except PermissionError:
            continue
        if record and record["pgid"] == pgid:
            members[path.name] = record["start_ticks"]
    return members


def owner_group_state(descriptor, owner):
    if (
        owner.get("run_id") != descriptor["run_id"]
        or owner.get("member_id") != descriptor["member_id"]
    ):
        raise ValueError("run owner identity mismatch")
    pid = owner.get("pid")
    if type(pid) is not int or pid <= 0 or owner.get("pgid") != pid:
        raise ValueError("invalid process group owner identity")
    if owner.get("cleanup_complete") is True:
        # This group was observed empty; a subsequently reused PGID is unrelated.
        return {
            "leader_alive": False,
            "members": {},
            "verified": True,
            "cleanup_complete": True,
        }
    members = group_members(pid)
    known = {str(pid): owner.get("start_ticks"), **owner.get("members", {})}
    verified = any(
        ticks is not None and members.get(member) == ticks
        for member, ticks in known.items()
    )
    leader_alive = (
        members.get(str(pid)) == owner.get("start_ticks")
        and owner.get("start_ticks") is not None
    )
    return {
        "leader_alive": leader_alive,
        "members": members,
        "verified": verified,
        "cleanup_complete": not members,
    }


def wait_members(members, pgid, timeout):
    """Wait on every observed member's pidfd, not merely the session leader."""
    import selectors

    deadline = None if timeout is None else time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        try:
            for member, ticks in members.items():
                record = process_record(int(member))
                if (
                    not record
                    or record["start_ticks"] != ticks
                    or record["pgid"] != pgid
                ):
                    continue
                try:
                    fd = pidfd_open(int(member))
                except ProcessLookupError:
                    continue
                selector.register(fd, selectors.EVENT_READ)
            while selector.get_map():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                events = selector.select(remaining)
                if not events:
                    break
                for key, _event in events:
                    selector.unregister(key.fd)
                    os.close(key.fd)
        finally:
            for key in tuple(selector.get_map().values()):
                selector.unregister(key.fd)
                os.close(key.fd)


def pidfd_open(pid):
    """Conda builds may omit os.pidfd_open although the Linux kernel supports it."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes
    import errno

    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "pidfd_open", None)
    value = (
        function(int(pid), 0)
        if function is not None
        else library.syscall(434, int(pid), 0)
    )
    if value < 0:
        number = ctypes.get_errno()
        if number == errno.ESRCH:
            raise ProcessLookupError(number, os.strerror(number))
        raise OSError(number, "Linux pidfd support is required: " + os.strerror(number))
    return value


def validate_descriptor(descriptor):
    if descriptor.get("schema_version") != 1:
        raise ValueError("unsupported SSH descriptor version")
    for field in ("run_id", "member_id"):
        if not isinstance(descriptor.get(field), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", descriptor[field]
        ):
            raise ValueError(f"invalid descriptor {field}")
    path = Path(descriptor["run_dir"])
    if not path.is_absolute():
        raise ValueError("run_dir must be absolute")
    return path


def status(descriptor):
    root = validate_descriptor(descriptor)
    path = root / "status.json"
    result = read_json(path) if path.exists() else {"state": "starting"}
    groups = []
    for name in ("owner.json", "build_owner.json"):
        if (root / name).exists():
            owner = read_json(root / name)
            groups.append(owner_group_state(descriptor, owner))
    result["alive"] = any(group["members"] for group in groups)
    result["leader_alive"] = any(group["leader_alive"] for group in groups)
    result["cleanup_complete"] = all(group["cleanup_complete"] for group in groups)
    result["residual_pids"] = sorted(
        {pid for group in groups for pid in group["members"]}, key=int
    )
    if result["alive"] and not result["leader_alive"]:
        result["state"] = "cleanup_required"
    elif not result["alive"] and result["state"] not in TERMINAL:
        result["state"] = "lost" if groups else result["state"]
    return {
        **result,
        "run_id": descriptor["run_id"],
        "member_id": descriptor["member_id"],
    }


def cancel(descriptor):
    root = validate_descriptor(descriptor)
    with (root / ".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # This tombstone fences a start whose SSH acknowledgement was lost.
        atomic_json(
            root / "cancel_requested.json",
            {"run_id": descriptor["run_id"], "member_id": descriptor["member_id"]},
        )
        owners = [
            (root / name, read_json(root / name))
            for name in ("owner.json", "build_owner.json")
            if (root / name).exists()
        ]
    for owner_path, owner in owners:
        terminate_owner(descriptor, owner, owner_path=owner_path)
    result = status(descriptor)
    if not result["cleanup_complete"]:
        raise ValueError(
            "owned process group cleanup is incomplete; retaining owner evidence"
        )
    return result


def terminate_owner(descriptor, owner, owner_path=None):
    root = validate_descriptor(descriptor)
    owner_path = Path(owner_path) if owner_path else root / "owner.json"
    current = owner_group_state(descriptor, owner)
    if not current["cleanup_complete"]:
        if not current["verified"]:
            raise ValueError(
                "process identity changed or group ownership cannot be proved; refusing to signal"
            )
        # Persist the complete known group before TERM can remove its leader.
        owner["members"] = current["members"]
        atomic_json(owner_path, owner)
        current = owner_group_state(descriptor, owner)
        if current["members"] and current["verified"]:
            os.killpg(owner["pgid"], signal.SIGTERM)
        wait_members(owner["members"], owner["pgid"], CANCEL_GRACE_SECONDS)
        current = owner_group_state(descriptor, owner)
        if current["members"]:
            if not current["verified"]:
                raise ValueError(
                    "residual process group ownership cannot be proved; retaining owner evidence"
                )
            # A surviving recorded member proves this PGID still belongs to us.
            owner["members"] = current["members"]
            atomic_json(owner_path, owner)
            if owner_group_state(descriptor, owner)["verified"]:
                os.killpg(owner["pgid"], signal.SIGKILL)
            wait_members(owner["members"], owner["pgid"], KILL_GRACE_SECONDS)
        current = owner_group_state(descriptor, owner)
        if current["members"]:
            raise ValueError(
                "owned process group did not exit; retaining owner evidence"
            )
    owner["cleanup_complete"] = True
    atomic_json(owner_path, owner)
    return status(descriptor)


def preflight(descriptor):
    root = validate_descriptor(descriptor)
    if os.getuid() == 0:
        raise ValueError("SSH performance workers must run as a non-root user")
    actual = verify_source(descriptor)
    if not Path(descriptor["worker_path"]).is_file():
        raise ValueError("frozen source does not contain the SSH worker protocol")
    ports = set(range(descriptor["start_port"], descriptor["start_port"] + 6))
    ports.update((descriptor["start_port"] + 1024, descriptor["start_port"] + 1025))
    for port in sorted(ports):
        with socket.socket() as connection:
            connection.bind(("0.0.0.0", port))
    root.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": 1,
        "status": "ready",
        "source_sha": actual,
        "uid": os.getuid(),
    }


def verify_bundle(descriptor):
    root = validate_descriptor(descriptor)
    for name, expected in descriptor["bundle_hashes"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid bundle path")
        if digest(root / path) != expected:
            raise ValueError(f"frozen bundle hash mismatch: {name}")
    plan = read_json(root / "config" / "execution_plan.json")
    if json_hash(plan) != descriptor["controller_plan_hash"]:
        raise ValueError("controller plan hash mismatch")
    if plan["backend"] != "ssh" or plan["action"] != "run":
        raise ValueError("descriptor requires a frozen SSH run plan")
    if descriptor["source"] != plan["resolved_config"]["execution"]["source"]:
        raise ValueError("descriptor source differs from the frozen plan")
    index = descriptor["member_index"]
    resource = plan["resource_plan"]
    if type(index) is not int or not 0 <= index < resource["machine_count"]:
        raise ValueError("invalid member index")
    if (
        descriptor["local_world_size"] != resource["local_world_size"]
        or descriptor["world_rank"] != index * resource["local_world_size"]
        or str(index) != descriptor["member_id"]
    ):
        raise ValueError("descriptor ranks differ from the frozen layout")
    return plan


def build(descriptor):
    with environment_lock(
        exclusive=descriptor["source"]["build_policy"] == "build_with_deps"
    ):
        return _build_locked(descriptor)


def _build_locked(descriptor):
    preflight(descriptor)
    source = descriptor["source"]
    root = Path(descriptor["run_dir"])
    identity = json_hash(
        {
            key: source.get(key)
            for key in ("ref", "bazel_config", "bazel_targets", "bazel_args")
        }
    )
    build_dir = Path(descriptor["build_dir"])
    build_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = build_dir / "receipt.json"
    with (build_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source["build_policy"] != "reuse":
            driver = source.get("build_driver")
            if not driver or not Path(driver).is_file():
                raise ValueError(
                    "build driver is missing; no bare Bazel fallback is permitted"
                )
            request = {
                "schema_version": 1,
                "operation": source["build_policy"],
                "run_id": descriptor["run_id"],
                "member_id": descriptor["member_id"],
                "source_sha": source["ref"],
                "source_dir": descriptor["source_dir"],
                "platform": source["bazel_config"],
                "targets": source["bazel_targets"],
                "args": source["bazel_args"],
                "output_user_root": str(build_dir / "bazel"),
                "repository_cache": str(build_dir.parent / "repository_cache"),
                "log_path": str(root / "build.log"),
                "receipt_path": str(receipt_path),
            }
            request_path = root / "build_request.json"
            atomic_json(request_path, request)
            with (root / "build_driver.log").open("ab") as log:
                with (root / ".start.lock").open("a") as start_lock:
                    fcntl.flock(start_lock, fcntl.LOCK_EX)
                    if (root / "cancel_requested.json").exists():
                        raise ValueError(
                            "run has been cancelled; refusing a late build"
                        )
                    receipt_path.unlink(missing_ok=True)
                    gate_read, gate_write = os.pipe()
                    gate = (
                        "import os,sys; fd=int(sys.argv[1]); ready=os.read(fd,1); "
                        "os.close(fd); "
                        "sys.exit(1) if ready != b'R' else os.execv(sys.argv[2],sys.argv[2:])"
                    )
                    process = None
                    try:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-c",
                                gate,
                                str(gate_read),
                                sys.executable,
                                driver,
                                "--request",
                                str(request_path),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(gate_read,),
                        )
                        owner = {
                            "run_id": descriptor["run_id"],
                            "member_id": descriptor["member_id"],
                            "pid": process.pid,
                            "pgid": process.pid,
                            "start_ticks": process_identity(process.pid),
                        }
                        atomic_json(root / "build_owner.json", owner)
                        os.write(gate_write, b"R")
                    except BaseException:
                        os.close(gate_write)
                        gate_write = None
                        if process is not None:
                            process.wait(timeout=5)
                        raise
                    finally:
                        os.close(gate_read)
                        if gate_write is not None:
                            os.close(gate_write)
                code = process.wait()
                if group_members(process.pid):
                    raise ValueError(
                        "build driver left processes in its group; cancellation required"
                    )
                owner["cleanup_complete"] = True
                atomic_json(root / "build_owner.json", owner)
                if code:
                    raise ValueError(f"build driver failed with exit {code}")
        receipt = read_json(receipt_path)
        verify_build_receipt(
            descriptor, receipt, require_identity=source["build_policy"] == "reuse"
        )
        receipt["build_identity"] = identity
        atomic_json(receipt_path, receipt)
        atomic_json(root / "build_receipt.json", receipt)
        return receipt


def verify_build_receipt(descriptor, receipt, *, require_identity):
    source = descriptor["source"]
    build_dir = Path(descriptor["build_dir"])
    identity = json_hash(
        {
            key: source.get(key)
            for key in ("ref", "bazel_config", "bazel_targets", "bazel_args")
        }
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "completed"
        or receipt.get("bazel_exit_code") != 0
        or receipt.get("source_sha") != source["ref"]
    ):
        raise ValueError("build receipt does not prove successful completion")
    if require_identity and receipt.get("build_identity") != identity:
        raise ValueError("reuse build identity does not match frozen build")
    if receipt.get("toolchain") != toolchain_snapshot():
        raise ValueError("build toolchain ABI does not match current node")
    if source.get("build_driver") and receipt.get("driver_sha256") != digest(
        source["build_driver"]
    ):
        raise ValueError("build driver identity does not match receipt")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("build receipt has no native artifact hashes")
    for name, expected in artifacts.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(descriptor["source_dir"]) / path
        allowed = (
            Path(descriptor["source_dir"]).resolve(),
            (build_dir / "bazel").resolve(),
        )
        if not any(path.resolve().is_relative_to(base) for base in allowed):
            raise ValueError(f"build artifact outside approved roots: {name}")
        if digest(path) != expected:
            raise ValueError(f"build artifact hash mismatch: {name}")
    verify_runtime_dependencies(
        receipt.get("runtime_dependencies"), source["bazel_config"]
    )
    return identity


def start(descriptor, descriptor_path):
    verify_bundle(descriptor)
    root = Path(descriptor["run_dir"])
    with (root / ".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "cancel_requested.json").exists():
            raise ValueError("run has been cancelled; refusing a late start")
        if (root / "owner.json").exists():
            # Lost acknowledgements return the existing invocation, never relaunch.
            return status(descriptor)
        verify_source(descriptor)
        with (root / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--operation",
                    "supervise",
                    "--descriptor",
                    str(Path(descriptor_path).resolve()),
                ],
                cwd=descriptor["source_dir"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        owner = {
            "run_id": descriptor["run_id"],
            "member_id": descriptor["member_id"],
            "pid": process.pid,
            "pgid": process.pid,
            "start_ticks": process_identity(process.pid),
        }
        atomic_json(root / "owner.json", owner)
    return {
        "state": "starting",
        "pid": process.pid,
        "run_id": descriptor["run_id"],
        "member_id": descriptor["member_id"],
    }


@contextmanager
def runtime_build_lease(descriptor):
    # Both locks precede receipt/source checks and every import of built code.
    with environment_lock(exclusive=False):
        with (Path(descriptor["build_dir"]) / ".lock").open("a") as build_lease:
            fcntl.flock(build_lease, fcntl.LOCK_SH)
            yield


def ownership_anchor(read_fd, pgid):
    """Keep a recorded group member alive if the supervisor dies first."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.write(sys.stdout.fileno(), b"R")
    while os.read(read_fd, 1):
        pass
    os.close(read_fd)
    # The write end is CLOEXEC and never passed to engines. EOF therefore
    # follows runtime completion or supervisor exit even if engines survive.
    while True:
        members = group_members(pgid)
        members.pop(str(os.getpid()), None)
        # After EOF the supervisor cannot launch more runtime children. Do
        # not keep a successful supervisor alive solely to reap its anchor.
        members.pop(str(pgid), None)
        if not members:
            return
        wait_members(members, pgid, None)


@contextmanager
def supervisor_ownership(descriptor):
    """Persist an identity witness before runtime code can spawn children.

    A periodic descendant snapshot leaves a SIGKILL window. This tiny child
    stays in our group until engine descendants exit, retaining verifiable
    PID/start-time evidence without repeated owner-file writes.
    """
    import selectors

    root = validate_descriptor(descriptor)
    read_fd, write_fd = os.pipe()  # Both ends default to CLOEXEC.
    anchor = None
    try:
        with (root / ".start.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (root / "cancel_requested.json").exists():
                raise ValueError("run has been cancelled; refusing runtime startup")
            owner = read_json(root / "owner.json")
            if (
                owner.get("pid") != os.getpid()
                or owner.get("start_ticks") != process_identity(os.getpid())
                or not owner_group_state(descriptor, owner)["leader_alive"]
            ):
                raise ValueError("supervisor process identity differs from owner")
            anchor = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--operation",
                    "ownership-anchor",
                    "--anchor-fd",
                    str(read_fd),
                    "--anchor-pgid",
                    str(owner["pgid"]),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                pass_fds=(read_fd,),
            )
            os.close(read_fd)
            read_fd = None
            with anchor.stdout, selectors.DefaultSelector() as selector:
                selector.register(anchor.stdout, selectors.EVENT_READ)
                if not selector.select(5) or anchor.stdout.read(1) != b"R":
                    raise RuntimeError("ownership anchor did not become ready")
            ticks = process_identity(anchor.pid)
            if ticks is None:
                raise RuntimeError("ownership anchor exited before registration")
            owner["members"] = {**owner.get("members", {}), str(anchor.pid): ticks}
            atomic_json(root / "owner.json", owner)
        yield
    finally:
        if read_fd is not None:
            os.close(read_fd)
        os.close(write_fd)
        # It exits itself once runtime descendants are gone, including after
        # an abrupt supervisor exit that skips this finally block entirely.
        if anchor is not None and not (
            set(group_members(os.getpgrp())) - {str(os.getpid()), str(anchor.pid)}
        ):
            # Normal completion must reap the anchor before wait() observes
            # the supervisor exiting; otherwise it looks like leaked engines.
            anchor.wait(timeout=5)


def supervise(descriptor):
    root = validate_descriptor(descriptor)
    interrupted = False

    def signal_cancel(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    def check_cancelled():
        if interrupted:
            raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, signal_cancel)
    signal.signal(signal.SIGINT, signal_cancel)
    state, exit_code, error = "failed", 1, None
    atomic_json(root / "status.json", {"state": "running"})
    try:
        with runtime_build_lease(descriptor):
            verify_bundle(descriptor)
            verify_source(descriptor)
            verify_build_receipt(
                descriptor,
                read_json(root / "build_receipt.json"),
                require_identity=True,
            )
            with supervisor_ownership(descriptor):
                sys.path.insert(0, descriptor["source_dir"])
                from rtp_llm.test.utils.device_resource import get_cuda_info

                cuda_info = get_cuda_info()
                if not cuda_info:
                    raise ValueError(
                        "SSH model measurement requires visible accelerator devices"
                    )
                exit_code = _measure_locked(
                    descriptor, root, cuda_info, check_cancelled
                )
                state = "completed" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        state, exit_code = "interrupted", 130
    except BaseException as failure:
        import traceback

        traceback.print_exc()
        error = str(failure)
    finally:
        atomic_json(
            root / "status.json",
            {"state": state, "exit_code": exit_code, "error": error},
        )
    return exit_code


def _measure_locked(descriptor, root, cuda_info, check_cancelled):
    from rtp_llm.test.perf_test.execution.distributed.member import run_member
    from rtp_llm.test.perf_test.plan import ExecutionPlan
    from rtp_llm.test.utils.device_resource import DeviceResource

    with DeviceResource(
        descriptor["local_world_size"], cuda_info=cuda_info
    ) as resource:
        descriptor = {
            **descriptor,
            "devices": list(range(descriptor["local_world_size"])),
        }
        os.environ[
            (
                "CUDA_VISIBLE_DEVICES"
                if "308" not in cuda_info[0]
                else "HIP_VISIBLE_DEVICES"
            )
        ] = ",".join(resource.gpu_ids)
        from rtp_llm.test.perf_test.execution.devices import GPU_LOCKED_DEVICES_ENV

        os.environ[GPU_LOCKED_DEVICES_ENV] = ",".join(resource.gpu_ids)
        plan = ExecutionPlan.from_dict(verify_bundle(descriptor))
        source = read_json(root / "config" / "source_config.json")
        check_cancelled()
        return run_member(
            plan,
            source_config=source,
            descriptor=descriptor,
            check_cancelled=check_cancelled,
        )


def wait(descriptor):
    root = validate_descriptor(descriptor)
    owner = read_json(root / "owner.json")
    current = status(descriptor)
    if not current["alive"]:
        return current
    if process_identity(owner["pid"]) == owner["start_ticks"]:
        import selectors

        try:
            fd = pidfd_open(owner["pid"])
        except ProcessLookupError:
            return status(descriptor)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_READ)
                # Wait on process completion, not a sleep/poll loop.
                selector.select(86400)
        finally:
            os.close(fd)
    else:
        raise ValueError(
            "Linux pidfd support is required to wait for a live supervisor"
        )
    return status(descriptor)


def clean(descriptor):
    root = validate_descriptor(descriptor)
    current = status(descriptor)
    if (
        current["alive"]
        or not current["cleanup_complete"]
        or current["state"] not in TERMINAL
    ):
        raise ValueError("refusing to clean an active or unowned run")
    if (
        root.resolve() != root
        or root.name != descriptor["member_id"]
        or root.parent.name != descriptor["run_id"]
        or root.parent.parent.name != "runs"
    ):
        raise ValueError("refusing cleanup outside the dedicated run directory")
    shutil.rmtree(root)
    return {
        "state": "cleaned",
        "run_id": descriptor["run_id"],
        "member_id": descriptor["member_id"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        required=True,
        choices=(
            "describe",
            "preflight",
            "build",
            "start",
            "supervise",
            "ownership-anchor",
            "wait",
            "status",
            "cancel",
            "clean",
        ),
    )
    parser.add_argument("--descriptor")
    parser.add_argument("--anchor-fd", type=int)
    parser.add_argument("--anchor-pgid", type=int)
    args = parser.parse_args(argv)
    try:
        if args.operation == "ownership-anchor":
            ownership_anchor(args.anchor_fd, args.anchor_pgid)
            return 0
        if args.operation == "describe":
            result = {"schema_version": 1, "protocol": "rtp-perf-ssh-worker"}
        else:
            descriptor = read_json(args.descriptor)
            if args.operation == "supervise":
                return supervise(descriptor)
            result = (
                start(descriptor, args.descriptor)
                if args.operation == "start"
                else globals()[args.operation](descriptor)
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

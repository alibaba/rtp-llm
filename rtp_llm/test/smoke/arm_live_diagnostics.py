"""Opt-in, bounded observation for an isolated ARM diagnostic run only."""

import json
import logging
import os
import shutil


def _btc_arm_logged_pids(text):
    import re

    ranks = set()
    other = set()
    for line in text.splitlines():
        match = re.search(r"\[(\d+):\d+\]\s+\[RANK\s+\d+\]", line)
        if match:
            ranks.add(int(match.group(1)))
        match = re.search(r"\[process-(\d+)\]", line)
        if match:
            pid = int(match.group(1))
            other.add(pid)
            if "start_backend_server.py:local_rank_start()" in line:
                ranks.add(pid)
    return sorted(ranks), sorted(other - ranks)


def _btc_arm_identity(process, expected, root_pid, uid):
    import psutil

    try:
        # Recompute the descendant set before every attach/read, never trust a PID
        # from a log or the first discovery alone (PID reuse / exited processes).
        descendants = {p.pid for p in psutil.Process(root_pid).children(recursive=True)}
        return (
            process.pid in descendants
            and process.uids().real == uid
            and process.uids().effective == uid
            and process.create_time() == expected
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (OSError, psutil.Error):
        return False


def start_observer(suite):
    if os.environ.get("RTP_BTC_ARM_LIVE_STACKS") != "1":
        return None
    targets = {
        "dense_tp2_sm100",
        "deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm",
        "moe_nvfp4_deepep_ll_tp2_sm100",
        "moe_nvfp4_deepep_normal_dp2_sm100",
        "next_moe_nvfp4_cudagraph_tp2_sm100",
    }
    if suite not in targets or os.getuid() != os.geteuid():
        return None

    import concurrent.futures
    import datetime
    import re
    import signal
    import subprocess
    import sys
    import threading
    import time
    import types
    from pathlib import Path

    import psutil

    output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    artifacts = output / "arm_live_diagnostics"
    artifacts.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    root_pid, uid = os.getpid(), os.getuid()
    started = time.monotonic()
    start_wall = time.time()
    identities = {}
    owned = set()
    owned_lock = threading.Lock()

    def register(process):
        with owned_lock:
            owned.add(process)
        return process

    def bounded_output(process, stream, limit):
        state = {"bytes_seen": 0, "bytes_saved": 0, "limit": limit}

        def drain():
            while True:
                block = process.stdout.read(65536)
                if not block:
                    return
                state["bytes_seen"] += len(block)
                data = block[: max(0, limit - state["bytes_saved"])]
                try:
                    stream.write(data)
                    state["bytes_saved"] += len(data)
                except (OSError, ValueError):
                    state["write_error"] = True

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        return reader, state

    def finish_owned(process, detach=False):
        errors = []
        if process.poll() is None and detach:
            try:
                process.send_signal(signal.SIGINT)
                process.stdin.write(b"\ndetach\nquit\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                errors.append(repr(exc))
        for action in ("terminate", "kill"):
            if process.poll() is None:
                try:
                    getattr(process, action)()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(repr(exc))
        if process.poll() is not None:
            with owned_lock:
                owned.discard(process)
        return errors

    def read_limited(path, limit=65536):
        try:
            with open(path, "rb") as stream:
                data = stream.read(limit + 1)
            return {
                "text": data[:limit].decode(errors="replace"),
                "truncated": len(data) > limit,
            }
        except OSError as exc:
            return {"error": str(exc)}

    def command(argv, path, timeout=8):
        # Owned diagnostics only; even GPU/FUSE output has a hard byte cap.
        result = {"argv": argv}
        if stopped.is_set():
            return dict(result, skipped="observer stopping")
        try:
            with path.open("wb") as stream:
                process = register(
                    subprocess.Popen(
                        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                    )
                )
                reader, state = bounded_output(process, stream, 262144)
                try:
                    result["returncode"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    result["timeout"] = timeout
                finally:
                    result["cleanup_errors"] = finish_owned(process)
                    reader.join(timeout=2)
                    result["output"] = dict(state)
                    result["reader_finished"] = not reader.is_alive()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["error"] = repr(exc)
        return result

    def process_snapshot(process, identity):
        pid = process.pid
        if not _btc_arm_identity(process, identity, root_pid, uid):
            return {"pid": pid, "rejected_identity": True}
        proc = Path("/proc") / str(pid)
        result = {"pid": pid, "create_time": identity, "uid": uid}
        for name in ("status", "stat", "statm", "io", "wchan", "syscall"):
            result[name] = read_limited(proc / name)
        maps = read_limited(proc / "maps", 1048576)
        result["checkpoint_maps"] = {
            "lines": [
                line
                for line in maps.get("text", "").splitlines()
                if any(
                    key in line
                    for key in ("/mnt/nas", ".safetensors", "engram", "/dev/shm/")
                )
            ][:512],
            "source_truncated": maps.get("truncated"),
            "error": maps.get("error"),
        }
        result["engram_files"] = []
        try:
            descriptors = sorted((proc / "fd").iterdir(), key=lambda p: int(p.name))
            for descriptor in descriptors[:1024]:
                target = os.readlink(descriptor)
                if target.startswith(
                    (f"/dev/shm/rtp_llm_engram_{uid}_", f"/tmp/rtp_llm_engram_{uid}_")
                ):
                    result["engram_files"].append(
                        {"fd": descriptor.name, "target": target}
                    )
            result["fds_truncated"] = len(descriptors) > 1024
        except OSError as exc:
            result["fd_error"] = str(exc)
        result["tasks"] = {}
        try:
            tasks = sorted((proc / "task").iterdir(), key=lambda p: int(p.name))
            result["task_count"] = len(tasks)
            for task in tasks[:512]:
                result["tasks"][task.name] = {
                    name: read_limited(task / name, 8192)
                    for name in ("wchan", "status", "syscall", "stack")
                }
            result["tasks_truncated"] = len(tasks) > 512
        except OSError as exc:
            result["task_error"] = str(exc)
        return result

    def attach(process, identity, directory):
        pid = process.pid
        receipt = {"pid": pid, "create_time": identity, "uid": uid}
        if stopped.is_set():
            return dict(receipt, not_attached="observer stopping")
        receipt["before"] = process_snapshot(process, identity)
        debugger = shutil.which("gdb")
        receipt["debugger"] = debugger
        if debugger is None or not _btc_arm_identity(process, identity, root_pid, uid):
            receipt["not_attached"] = "gdb unavailable or identity no longer valid"
            return receipt
        before = receipt["before"].get("status", {}).get("text", "")
        state_line = next(
            (x for x in before.splitlines() if x.startswith("State:")), ""
        )
        tracer = next(
            (x for x in before.splitlines() if x.startswith("TracerPid:")), ""
        )
        if "\tT" in state_line or "\tt" in state_line or tracer.split()[-1:] != ["0"]:
            receipt["not_attached"] = (
                "already stopped/traced or no readable identity status"
            )
            return receipt
        commands = [
            "set pagination off",
            "set confirm off",
            "set auto-load off",
            "set auto-load libthread-db on",
            "set libthread-db-search-path /lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:/lib64:/usr/lib64",
            "set auto-load safe-path /lib:/usr/lib:/lib64:/usr/lib64",
            "set print elements 64",
            "set print frame-arguments none",
            "set backtrace limit 40",
            f"attach {pid}",
            "thread apply all bt 35",
            "thread apply all py-bt",
            "detach",
            "quit",
        ]
        trusted_python_helper = None
        for candidate in (
            Path("/opt/conda310/bin/python3.10-gdb.py"),
            Path("/usr/share/gdb/auto-load/usr/bin/python3.10-gdb.py"),
        ):
            try:
                real = candidate.resolve(strict=True)
                if not any(
                    str(real).startswith(prefix)
                    for prefix in ("/opt/conda310/", "/usr/share/gdb/")
                ):
                    continue
                chain = [real] + list(real.parents)
                if all(
                    item.stat().st_uid == 0 and not item.stat().st_mode & 0o022
                    for item in chain
                ):
                    trusted_python_helper = str(real)
                    commands.insert(
                        commands.index("thread apply all py-bt"), "source " + str(real)
                    )
                    break
            except OSError:
                continue
        receipt["trusted_python_helper"] = trusted_python_helper
        receipt["python_stack_availability"] = (
            "py-bt attempted; actual debugger output is authoritative"
        )
        if stopped.is_set() or not _btc_arm_identity(process, identity, root_pid, uid):
            return dict(
                receipt, not_attached="stopping or identity changed before spawn"
            )
        # PDEATHSIG also covers an outer watchdog SIGKILL, where Python
        # finally cannot run. The child verifies the parent before exec gdb.
        launcher = (
            "import ctypes,os,signal,sys; "
            "p=int(sys.argv[1]); "
            "libc=ctypes.CDLL(None,use_errno=True); "
            "r=libc.prctl(1,signal.SIGTERM); "
            "sys.exit(125) if r!=0 or os.getppid()!=p else None; "
            "os.execv(sys.argv[2],sys.argv[2:])"
        )
        with (directory / f"rank_{pid}.gdb.log").open("wb") as stream:
            debugger_process = register(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        launcher,
                        str(root_pid),
                        debugger,
                        "-nx",
                        "-q",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            )
            receipt["debugger_pid"] = debugger_process.pid
            receipt["debugger_parent_death_signal"] = (
                "SIGTERM; parent PID checked before exec"
            )
            reader, output_state = bounded_output(debugger_process, stream, 4194304)
            try:
                debugger_process.stdin.write(("\n".join(commands) + "\n").encode())
                debugger_process.stdin.flush()
                receipt["gdb_returncode"] = debugger_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                receipt["gdb_timeout_seconds"] = 30
            except (OSError, ValueError) as exc:
                receipt["gdb_io_error"] = repr(exc)
            finally:
                receipt["cleanup_errors"] = finish_owned(debugger_process, detach=True)
                try:
                    debugger_process.stdin.close()
                except (OSError, ValueError) as exc:
                    receipt["stdin_close_error"] = repr(exc)
                reader.join(timeout=2)
                receipt["debugger_output"] = dict(output_state)
                receipt["debugger_output"]["truncated"] = (
                    output_state["bytes_seen"] > output_state["bytes_saved"]
                )
                receipt["debugger_output"]["reader_finished"] = not reader.is_alive()
        receipt["after"] = process_snapshot(process, identity)
        after = receipt["after"].get("status", {}).get("text", "")
        receipt["residual_stopped_or_traced"] = any(
            line.startswith("State:")
            and ("\tT" in line or "\tt" in line)
            or line.startswith("TracerPid:")
            and line.split()[-1] != "0"
            for line in after.splitlines()
        )
        # Restore only a stop caused by this observer's attach: never resume a
        # pre-existing stop, another tracer, a reused PID or a non-descendant.
        after_tracer = next(
            (
                line.split()[-1]
                for line in after.splitlines()
                if line.startswith("TracerPid:")
            ),
            None,
        )
        after_stopped = any(
            line.startswith("State:") and ("\tT" in line or "\tt" in line)
            for line in after.splitlines()
        )
        if (
            after_stopped
            and after_tracer == "0"
            and debugger_process.poll() is not None
            and _btc_arm_identity(process, identity, root_pid, uid)
        ):
            receipt["restore_sigcont"] = {
                "reason": "restore this attach's stop",
                "identity_rechecked": True,
                "debugger_exited": True,
                "tracer_pid": 0,
            }
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError as exc:
                receipt["restore_error"] = repr(exc)
            receipt["after_restore"] = process_snapshot(process, identity)
        # Every residual stop remains visible. No test business signal, cancel,
        # kill, timeout adjustment, request or production sleep is introduced.
        return receipt

    def observe(elapsed):
        directory = artifacts / f"elapsed_{elapsed:04d}"
        directory.mkdir(exist_ok=True)
        snapshot = {
            "suite": suite,
            "elapsed_requested": elapsed,
            "elapsed_actual": time.monotonic() - started,
            "root_pid": root_pid,
            "uid": uid,
            "ptrace_scope": read_limited("/proc/sys/kernel/yama/ptrace_scope"),
            "roles": {},
            "candidate_identity_proof": {},
        }
        candidates = {}
        for log in sorted(output.glob("*logs/process.log"))[:8]:
            text = read_limited(log, 16 * 1024 * 1024).get("text", "")
            ranks, others = _btc_arm_logged_pids(text)
            snapshot["roles"][str(log.relative_to(output))] = {
                "rank_pids": ranks,
                "other_logged_pids": others,
            }
            # Prefer actual model ranks. Before they exist, capture up to two
            # logged Python startup processes for the otherwise opaque startup.
            for pid in ranks if ranks else others[:2]:
                try:
                    process = psutil.Process(pid)
                    expected = identities.setdefault(pid, process.create_time())
                    logged_times = []
                    for line in text.splitlines():
                        if f"[process-{pid}]" not in line and not re.search(
                            r"\[" + str(pid) + r":\d+\]", line
                        ):
                            continue
                        match = re.search(
                            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[.,]\d+", line
                        )
                        if match:
                            logged_times.append(
                                datetime.datetime.strptime(
                                    match.group(1), "%Y-%m-%d %H:%M:%S"
                                ).timestamp()
                            )
                    if (
                        not logged_times
                        or not start_wall - 5 <= expected <= min(logged_times) + 2
                    ):
                        continue
                    if _btc_arm_identity(process, expected, root_pid, uid):
                        candidates[pid] = (process, expected)
                        snapshot["candidate_identity_proof"][str(pid)] = {
                            "source_log": str(log.relative_to(output)),
                            "earliest_logged_time": min(logged_times),
                            "create_time": expected,
                            "uid": uid,
                            "descendant_rechecked": True,
                        }
                except (OSError, psutil.Error):
                    pass
        selected = list(sorted(candidates.values(), key=lambda item: item[0].pid))[:4]
        snapshot["candidate_count"] = len(candidates)
        snapshot["selection_truncated"] = len(candidates) > 4
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                snapshot["processes"] = list(
                    executor.map(lambda item: attach(*item, directory), selected)
                )
        except Exception as exc:
            snapshot["attach_exception"] = repr(exc)
        selected_pids = {str(p.pid) for p, _ in selected}
        locks = read_limited("/proc/locks", 2 * 1024 * 1024)
        snapshot["process_locks"] = []
        for line in locks.get("text", "").splitlines():
            match = re.search(r"\b(?:READ|WRITE)\s+(-?\d+)\s", line)
            if match and match.group(1) in selected_pids:
                snapshot["process_locks"].append(line)
        snapshot["process_locks"] = snapshot["process_locks"][:256]
        snapshot["mounts"] = [
            line
            for line in read_limited("/proc/self/mountinfo", 1048576)
            .get("text", "")
            .splitlines()
            if "fuse" in line.lower() or "/mnt/nas" in line
        ][:64]
        try:
            stat = os.statvfs("/dev/shm")
            snapshot["dev_shm"] = {
                "blocks": stat.f_blocks,
                "free_blocks": stat.f_bavail,
                "block_size": stat.f_frsize,
                "free_inodes": stat.f_favail,
            }
            shared = set()
            for process_info in snapshot.get("processes", []):
                for entry in process_info.get("before", {}).get("engram_files", []):
                    name = Path(entry["target"].removesuffix(" (deleted)")).name
                    if name.endswith(".lock"):
                        name = name[:-5]
                    if name.startswith(f"rtp_llm_engram_{uid}_"):
                        shared.add(Path("/dev/shm") / name)
            snapshot["shared_staging"] = []
            for path in sorted(shared)[:32]:
                item = {
                    "path": str(path),
                    "reason": "open FD in a validated target process",
                }
                for label, candidate in (
                    ("data", path),
                    ("lock", Path("/tmp") / (path.name + ".lock")),
                ):
                    try:
                        value = candidate.stat()
                        item[label] = {
                            "path": str(candidate),
                            "size": value.st_size,
                            "mtime": value.st_mtime,
                        }
                    except OSError as exc:
                        item[label] = {"path": str(candidate), "error": str(exc)}
                snapshot["shared_staging"].append(item)
        except OSError as exc:
            snapshot["shared_staging_error"] = str(exc)
        smi = shutil.which("nvidia-smi")
        snapshot["gpu"] = (
            command([smi, "-q"], directory / "gpu-state.log")
            if smi
            else {"error": "nvidia-smi unavailable"}
        )
        snapshot["gpu_device_kernel_stack"] = (
            "Not claimed: ordinary gdb captures native host stacks; nvidia-smi captures GPU state."
        )
        # Tail only known bounded FUSE diagnostic paths; never read model data.
        fuse_logs = [
            output / "fuse.log",
            Path("/tmp/fuse.log"),
            Path("/home/admin/fuse.log"),
            Path("/var/log/fuse.log"),
        ]
        for index, path in enumerate(fuse_logs):
            if path.is_file():
                snapshot[f"fuse_tail_{index}"] = command(
                    ["tail", "-c", "65536", str(path)],
                    directory / f"fuse_{index}.tail.log",
                    5,
                )
        (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")

    def run():
        times = [600, 1200, 1600]
        if suite == "deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm":
            times.append(3000)
        for elapsed in times:
            if stopped.wait(max(0, elapsed - (time.monotonic() - started))):
                return
            try:
                observe(elapsed)
            except Exception:
                logging.exception(
                    "opt-in ARM live diagnostics failed at elapsed=%s", elapsed
                )

    thread = threading.Thread(target=run, name="btc-arm-live-diagnostics", daemon=True)

    def cleanup_observer():
        # This is optional diagnostics: every cleanup step is best-effort and
        # no ordinary cleanup/logging/receipt error may replace test outcomes.
        errors = []

        def attempt(label, action):
            try:
                return action()
            except Exception as exc:
                errors.append({"step": label, "error": repr(exc)})
                return None

        attempt("signal_observer", stopped.set)
        attempt("join_observer", lambda: thread.join(timeout=110))

        def owned_snapshot():
            with owned_lock:
                return list(owned)

        remaining = attempt("list_owned", owned_snapshot) or []
        for process in remaining:
            attempt(
                f"cleanup_owned_{process.pid}",
                lambda process=process: finish_owned(
                    process, detach=process.stdin is not None
                ),
            )
        attempt("final_join", lambda: thread.join(timeout=10))

        def write_receipt():
            with owned_lock:
                alive = [p.pid for p in owned if p.poll() is None]
            (artifacts / "observer-stop.json").write_text(
                json.dumps(
                    {
                        "thread_finished": not thread.is_alive(),
                        "owned_subprocesses_still_alive": alive,
                        "cleanup_complete": not thread.is_alive()
                        and not alive
                        and not errors,
                        "cleanup_errors": errors,
                    },
                    indent=2,
                )
                + "\n"
            )

        attempt("write_stop_receipt", write_receipt)
        if errors:
            try:
                logging.error("opt-in ARM observer cleanup incomplete: %s", errors)
            except Exception:
                pass

    def stop_observer():
        try:
            cleanup_observer()
        except Exception:
            # A second guard also covers unexpected failures in bookkeeping.
            try:
                logging.exception("opt-in ARM observer cleanup failed")
            except Exception:
                pass

    (artifacts / "observer-start.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "root_pid": root_pid,
                "uid": uid,
                "effective_uid": os.geteuid(),
                "schedule_seconds": [600, 1200, 1600]
                + (
                    [3000]
                    if suite == "deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm"
                    else []
                ),
                "gdb_process_limit": 4,
                "concurrent_gdb_limit": 2,
                "gdb_timeout_seconds": 30,
                "gdb_output_bytes_limit": 4194304,
                "diagnostic_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    thread.start()
    return types.SimpleNamespace(stop=stop_observer)

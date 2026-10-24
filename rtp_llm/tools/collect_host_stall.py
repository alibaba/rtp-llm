#!/usr/bin/env python3
"""Manually collect one bounded, read-only Linux GDB host-stall snapshot.

Attaching pauses the target. This tool never calls inferior functions, resumes
execution for sampling, or signals the target PID. Timeout cleanup can take up
to seven additional seconds. A successful result requires independently verified
detach; a timeout remains a failure even if detach was verified.

Example (supply the actual /proc/PID/stat field 22, not a guessed value)::

    python collect_host_stall.py --pid 123 --startticks 456 --tid 789 \
        --output /tmp/stall-new --version-label deployed-image-and-sha \
        --trace-file /path/to/gpu_stage_trace.123.jsonl

Only the Python standard library and GDB with Python support are required.
Artifacts can contain command-line arguments, stack locals and process memory;
the new output directory is private. The tool does not dump environment variables.
"""

import argparse
import datetime
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

PROC_ROOT = Path("/proc")
TRACE_MAX_BYTES = 16 * 1024 * 1024


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_proc_stat(raw):
    # comm may contain spaces and ')'; fields following its final ')' start at 3.
    left, separator, right = raw.rpartition(")")
    if not separator:
        raise ValueError("malformed /proc stat")
    fields = right.split()
    return {
        "pid": int(left.split("(", 1)[0]),
        "state": fields[0],
        "startticks": int(fields[19]),
    }


def read_identity(pid, tids=()):
    directory = PROC_ROOT / str(pid)
    identity = parse_proc_stat((directory / "stat").read_text())
    status = dict(
        line.split(":", 1)
        for line in (directory / "status").read_text().splitlines()
        if ":" in line
    )
    identity["tracer_pid"] = int(status["TracerPid"].strip())
    identity["tgid"] = int(status["Tgid"].strip())
    identity["threads"] = {}
    for tid in tids:
        task = directory / "task" / str(tid)
        task_identity = parse_proc_stat((task / "stat").read_text())
        task_status = dict(
            line.split(":", 1)
            for line in (task / "status").read_text().splitlines()
            if ":" in line
        )
        if int(task_status["Tgid"].strip()) != pid or task_identity["pid"] != tid:
            raise ValueError("selected TID does not belong to target PID")
        task_identity["tracer_pid"] = int(task_status["TracerPid"].strip())
        identity["threads"][str(tid)] = task_identity
    return identity


def validate_identity(identity, pid, startticks):
    if identity["pid"] != pid or identity["tgid"] != pid:
        raise ValueError("target PID/Tgid mismatch")
    if identity["startticks"] != startticks:
        raise ValueError("target startticks changed or does not match")
    if identity["tracer_pid"] != 0:
        raise ValueError("target already has a tracer")
    if any(task["tracer_pid"] != 0 for task in identity["threads"].values()):
        raise ValueError("selected TID already has a tracer")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_proc_files(output, pid, name):
    destination = output / name
    destination.mkdir(mode=0o700)
    errors = []
    for filename in ("stat", "status", "cmdline", "maps"):
        try:
            (destination / filename).write_bytes(
                (PROC_ROOT / str(pid) / filename).read_bytes()
            )
        except OSError as error:
            (destination / (filename + ".error")).write_text(str(error) + "\n")
            errors.append(filename + ": " + str(error))
    return errors


def snapshot_trace(source, destination):
    source = source.resolve(strict=True)
    if "environ" in source.parts and str(source).startswith("/proc/"):
        raise ValueError("/proc environ is not a trace file")
    # O_NONBLOCK prevents an accidental FIFO from blocking before fstat rejects it.
    descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("trace snapshot requires a regular file")
        offset = max(0, info.st_size - TRACE_MAX_BYTES)
        handle.seek(offset)
        data = handle.read(min(info.st_size, TRACE_MAX_BYTES))
        destination.write_bytes(data)
    return {
        "source": str(source),
        "captured_at_utc": utc_now(),
        "source_size": info.st_size,
        "source_mtime_ns": info.st_mtime_ns,
        "offset": offset,
        "bytes": len(data),
        "tail_truncated": offset != 0,
        "atomic_snapshot": False,
    }


def build_gdb_script(pid, startticks, thread_identities, marker):
    config = {
        "pid": pid,
        "startticks": startticks,
        "threads": thread_identities,
        "marker": marker,
    }
    body = r"""
        import gdb, json, os
        config = json.loads(CONFIG_JSON)
        errors = []
        detached = False

        def emit(event, **fields):
            fields["event"] = event
            gdb.write(config["marker"] + json.dumps(fields, sort_keys=True) + "\n")

        def identity(path):
            with open(path + "/stat") as handle:
                raw = handle.read()
            startticks = int(raw.rsplit(")", 1)[1].split()[19])
            with open(path + "/status") as handle:
                status = dict(line.split(":", 1) for line in handle if ":" in line)
            return startticks, int(status["TracerPid"]), int(status["Tgid"])

        def command(text):
            emit("command", text=text)
            try:
                gdb.execute(text, from_tty=False)
            except gdb.error as error:
                errors.append(text + ": " + str(error))
                emit("command_failed", text=text, error=str(error))

        try:
            # These settings affect GDB, not inferior memory or execution.
            for setting in (
                "set pagination off", "set confirm off", "set height 0",
                "set width 0", "set backtrace limit unlimited",
                "set print elements 32", "set print repeats 16",
                "set print frame-arguments all", "set print pretty off",
                "set may-call-functions off", "set may-write-memory off",
                "set may-write-registers off", "set write off",
                "set auto-load python-scripts off", "set auto-load gdb-scripts off",
                "set auto-load local-gdbinit off",
            ):
                command(setting)
            if errors:
                raise RuntimeError("required GDB safety settings unavailable")
            path = "/proc/" + str(config["pid"])
            ticks, tracer, tgid = identity(path)
            if (ticks, tracer, tgid) != (config["startticks"], 0, config["pid"]):
                raise RuntimeError("identity/tracer changed immediately before attach")
            emit("attach_started", pid=config["pid"])
            gdb.execute("attach " + str(config["pid"]), from_tty=False)
            ticks, tracer, tgid = identity(path)
            if (ticks, tracer, tgid) != (config["startticks"], os.getpid(), config["pid"]):
                raise RuntimeError("identity/tracer mismatch after attach")
            emit("attach_verified", pid=config["pid"], startticks=ticks)
            command("show version")
            command("info threads")
            threads = {thread.ptid[1]: thread for thread in gdb.selected_inferior().threads()}
            for tid_string, before in config["threads"].items():
                tid = int(tid_string)
                if tid not in threads:
                    raise RuntimeError("selected TID exited before attach: " + tid_string)
                task_ticks, _, task_tgid = identity(path + "/task/" + tid_string)
                if task_ticks != before["startticks"] or task_tgid != config["pid"]:
                    raise RuntimeError("selected TID identity changed: " + tid_string)
                threads[tid].switch()
                emit("selected_thread", tid=tid)
                command("bt full")
                command("frame 0")
                command("info frame")
                command("info args")
                command("info registers")
                command("x/16gx $sp")
                command("x/16i $pc")
                frame = gdb.newest_frame()
                level = 0
                while frame is not None:
                    name = frame.name() or ""
                    if any(part in name.lower() for part in (
                        "pthread_rwlock", "pthread_mutex", "futex", "lll_lock",
                    )):
                        frame.select()
                        emit("lock_wait_frame", tid=tid, level=level, name=name)
                        command("info frame")
                        command("info args")
                    frame = frame.older()
                    level += 1
            command("thread apply all bt")
            command("info sharedlibrary")
            sections_help = gdb.execute("help maintenance info sections", to_string=True)
            sections_option = "-all-objects" if "-all-objects" in sections_help else "ALLOBJ"
            command("maintenance info sections " + sections_option)
            for objfile in gdb.objfiles():
                # Older GDB builds may not expose build_id; do not invent it.
                build_id = getattr(objfile, "build_id", None)
                if isinstance(build_id, bytes):
                    build_id = build_id.hex()
                emit("objfile", filename=objfile.filename,
                     build_id=build_id)
            emit("collection_finished", errors=errors)
        except BaseException as error:
            errors.append(type(error).__name__ + ": " + str(error))
            emit("collection_failed", error=errors[-1])
        finally:
            try:
                if gdb.selected_inferior().pid:
                    gdb.execute("detach", from_tty=False)
                    detached = True
                    emit("detach_returned")
            except BaseException as error:
                errors.append("detach: " + str(error))
                emit("detach_failed", error=str(error))
            emit("gdb_finished", detached=detached, errors=errors)
            gdb.execute("quit " + ("0" if detached and not errors else "1"), from_tty=False)
    """
    body = textwrap.dedent(body).replace("CONFIG_JSON", repr(json.dumps(config)))
    return "python\n" + body + "\nend\n"


def stop_gdb(process, actions):
    """Only signal this unreaped child; never signal the business PID/group."""
    if process.poll() is not None:
        return
    actions.append({"action": "gdb_sigint", "at_utc": utc_now()})
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.stdin.write("set confirm off\ndetach\nquit 1\n")
        process.stdin.flush()
        actions.append({"action": "gdb_detach_commands_queued", "at_utc": utc_now()})
    except (BrokenPipeError, OSError):
        pass
    for delay, action in ((3, process.terminate), (2, process.kill), (2, None)):
        try:
            process.wait(timeout=delay)
            return
        except subprocess.TimeoutExpired:
            if action is not None:
                actions.append(
                    {"action": "gdb_" + action.__name__, "at_utc": utc_now()}
                )
                try:
                    action()
                except ProcessLookupError:
                    pass
    raise RuntimeError("spawned GDB did not exit after bounded cleanup")


def read_gdb_events(path, marker):
    events = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith(marker):
                try:
                    events.append(json.loads(line[len(marker) :]))
                except ValueError:
                    pass
    return events


def collect(args):
    if sys.platform != "linux":
        raise ValueError("this tool must be invoked manually on Linux")
    if args.pid == os.getpid():
        raise ValueError("refusing to attach the collector itself")
    output = args.output.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    os.chmod(output, 0o700)
    metadata = {
        "started_at_utc": utc_now(),
        "pid": args.pid,
        "expected_startticks": args.startticks,
        "tids": sorted(set(args.tid)),
        "version_label": args.version_label,
        "version_label_provenance": "caller_supplied_unverified",
        "timeout_seconds": args.timeout,
        "cleanup_grace_seconds": 7,
        "host": list(os.uname()),
        "success": False,
        "detach_verified": False,
        "cleanup_actions": [],
    }
    write_json(output / "manifest.json", metadata)
    process = None
    marker = "__HOST_STALL_" + secrets.token_hex(12) + "__"
    old_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError("collector received signal " + str(signum))

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        metadata["before_proc_errors"] = save_proc_files(output, args.pid, "before")
        if metadata["before_proc_errors"]:
            raise RuntimeError("required pre-attach /proc files could not be saved")
        before = read_identity(args.pid, metadata["tids"])
        metadata["before"] = before
        validate_identity(before, args.pid, args.startticks)
        for name, path in (
            ("nvidia-driver-version", PROC_ROOT / "driver/nvidia/version"),
            ("nvidia-module-version", Path("/sys/module/nvidia/version")),
            ("kernel-version", PROC_ROOT / "version"),
            ("ptrace-scope", PROC_ROOT / "sys/kernel/yama/ptrace_scope"),
        ):
            try:
                (output / name).write_bytes(path.read_bytes())
            except OSError as error:
                (output / (name + ".error")).write_text(str(error) + "\n")
        metadata["traces"] = [
            snapshot_trace(path, output / ("trace-%03d.snapshot" % index))
            for index, path in enumerate(args.trace_file)
        ]
        command_path = output / "gdb.commands"
        command_path.write_text(
            build_gdb_script(args.pid, args.startticks, before["threads"], marker)
        )
        current = read_identity(args.pid, metadata["tids"])
        validate_identity(current, args.pid, args.startticks)
        if current["threads"] != before["threads"]:
            # Task state may legitimately change; only startticks identify reuse.
            for tid, task in before["threads"].items():
                if current["threads"][tid]["startticks"] != task["startticks"]:
                    raise ValueError("selected TID changed before attach")
        metadata["before_attach_at_utc"] = utc_now()
        write_json(output / "manifest.json", metadata)
        with (output / "gdb.stdout").open("w") as log:
            deadline = time.monotonic() + args.timeout
            process = subprocess.Popen(
                [args.gdb, "-nx", "-nh", "-q", "-x", str(command_path)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env={**os.environ, "DEBUGINFOD_URLS": ""},
            )
            metadata["gdb_pid"] = process.pid
            write_json(output / "manifest.json", metadata)
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                metadata["timed_out"] = True
                raise RuntimeError("GDB collection exceeded timeout")
    except BaseException as error:
        metadata["error"] = type(error).__name__ + ": " + str(error)
    finally:
        # Repeated user signals cannot interrupt bounded detach cleanup.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None:
            try:
                stop_gdb(process, metadata["cleanup_actions"])
            except BaseException as error:
                metadata["cleanup_error"] = str(error)
            metadata["gdb_returncode"] = process.poll()
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        try:
            metadata["after_proc_errors"] = save_proc_files(output, args.pid, "after")
        except OSError as error:
            metadata["after_proc_errors"] = [str(error)]
        try:
            all_tids = sorted(
                int(task.name)
                for task in (PROC_ROOT / str(args.pid) / "task").iterdir()
                if task.name.isdecimal()
            )
            after = read_identity(args.pid, all_tids)
            metadata["after"] = after
            metadata["after_all_threads_checked"] = all_tids
            validate_identity(after, args.pid, args.startticks)
            stopped_before = metadata.get("before", {}).get("state") in ("T", "t")
            if after["state"] in ("T", "t") and not stopped_before:
                raise ValueError("target remains stopped after collection")
            # The task scan is not atomic; bracket its completion with another
            # leader identity check before claiming the original process detached.
            final_identity = read_identity(args.pid)
            metadata["final_identity"] = final_identity
            validate_identity(final_identity, args.pid, args.startticks)
            metadata["untraced_after_verified"] = True
        except (OSError, ValueError, KeyError, IndexError) as error:
            metadata["after_verification_error"] = str(error)
        try:
            if (output / "gdb.stdout").exists():
                metadata["gdb_events"] = read_gdb_events(output / "gdb.stdout", marker)
        except OSError as error:
            metadata["event_read_error"] = str(error)
        events = metadata.get("gdb_events", [])
        names = {event.get("event") for event in events}
        metadata["detach_verified"] = (
            "attach_verified" in names
            and metadata.get("untraced_after_verified", False)
            and process is not None
            and process.poll() is not None
            and "cleanup_error" not in metadata
        )
        clean_finish = any(
            event.get("event") == "gdb_finished"
            and event.get("detached")
            and not event.get("errors")
            for event in events
        )
        metadata["success"] = (
            not any(
                key in metadata
                for key in (
                    "error",
                    "cleanup_error",
                    "after_verification_error",
                    "event_read_error",
                    "timed_out",
                )
            )
            and not metadata.get("after_proc_errors")
            and metadata["detach_verified"]
            and metadata.get("gdb_returncode") == 0
            and {"attach_verified", "collection_finished", "detach_returned"} <= names
            and clean_finish
        )
        metadata["finished_at_utc"] = utc_now()
        try:
            write_json(output / "manifest.json", metadata)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return metadata["success"], output


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def timeout_seconds(value):
    number = float(value)
    if not 5 <= number <= 60:
        raise argparse.ArgumentTypeError("must be between 5 and 60 seconds")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=positive_integer)
    parser.add_argument("--startticks", required=True, type=positive_integer)
    parser.add_argument("--tid", action="append", default=[], type=positive_integer)
    parser.add_argument(
        "--output", required=True, type=Path, help="new private directory"
    )
    parser.add_argument("--version-label", required=True)
    parser.add_argument("--timeout", type=timeout_seconds, default=30)
    parser.add_argument("--trace-file", action="append", default=[], type=Path)
    parser.add_argument(
        "--gdb", default="gdb", help="GDB executable, no shell commands"
    )
    args = parser.parse_args()
    os.umask(0o077)
    try:
        success, output = collect(args)
    except (OSError, ValueError) as error:
        print("Collection failed: " + str(error), file=sys.stderr)
        return 1
    print(("SUCCESS: " if success else "FAILED; inspect manifest: ") + str(output))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

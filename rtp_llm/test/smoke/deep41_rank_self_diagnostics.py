"""Diagnostic-only rank self stacks; no attach, child signals, or workload changes."""

import json
import os
import time
import types
import uuid
from pathlib import Path

_TARGET = "deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm"
_active = None
_retained_file = None


def start_rank_watchdog():
    global _active
    if (
        os.environ.get("RTP_BTC_ARM_RANK_SELF_TRACE") != "1"
        or os.environ.get("RTP_BTC_DIAG_SUITE") != _TARGET
        or not os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        or _active is not None
        or _retained_file is not None
    ):
        return None

    try:
        import faulthandler
        import sys
    except BaseException:
        return None

    owner_pid = os.getpid()
    trace_file = prefix = None
    attempted = armed = stopped = False

    def write_stop(data, suffix="stop"):
        try:
            if prefix is not None:
                Path(str(prefix) + "." + suffix + ".json").write_text(
                    json.dumps(data) + "\n"
                )
        except BaseException:
            pass

    def cleanup():
        nonlocal stopped
        global _active, _retained_file
        if os.getpid() != owner_pid:
            # A later weight-loader fork may inherit this closure/FD. Keep its
            # local FD copy alive via a strong retained reference; never cancel a child-owned timer
            # or claim the parent's timer was cleaned up from another PID.
            _retained_file = trace_file
            write_stop(
                {
                    "owner_pid": owner_pid,
                    "current_pid": os.getpid(),
                    "parent_timer_cleanup_claimed": False,
                    "timer_cancel_attempted": False,
                    "local_fd_policy": "retained until process exit; no cancel/close",
                },
                "fork-copy-" + str(os.getpid()),
            )
            return
        if stopped:
            return
        stopped = True
        cancelled = not attempted
        errors = []
        if attempted:
            try:
                faulthandler.cancel_dump_traceback_later()
                cancelled = True
            except BaseException:
                errors.append("cancel_failed")
        if trace_file is not None:
            if cancelled:
                try:
                    trace_file.close()
                except BaseException:
                    errors.append("close_failed")
                    _retained_file = trace_file
            else:
                # Closing/recycling a descriptor with an unconfirmed timer is
                # unsafe; retain it and refuse another timer in this process.
                _retained_file = trace_file
        _active = None
        write_stop(
            {
                "owner_pid": owner_pid,
                "current_pid": os.getpid(),
                "timer_registered": armed,
                "timer_cancelled": cancelled,
                "fd_closed": bool(trace_file is None or trace_file.closed),
                "cleanup_errors": errors,
            }
        )

    def stop():
        try:
            cleanup()
        except BaseException:
            # Diagnostic cleanup cannot replace Return/Exception/SystemExit.
            pass

    try:
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        if not root.is_absolute() or not root.resolve(strict=True).is_dir():
            return None
        directory = root / "deep41_rank_selfdiag"
        directory.mkdir(mode=0o700, exist_ok=True)
        prefix = directory / f"rank_{owner_pid}_{uuid.uuid4().hex}"
        trace_file = Path(str(prefix) + ".python-stacks.log").open("x")
        handle = types.SimpleNamespace(stop=stop)
        Path(str(prefix) + ".prepared.json").write_text(
            json.dumps(
                {
                    "status": "prepared_before_timer; not an armed-timer proof",
                    "owner_pid": owner_pid,
                    "uid": os.getuid(),
                    "effective_uid": os.geteuid(),
                    "suite": _TARGET,
                    "python": sys.version,
                    "prepared_wall_ns": time.time_ns(),
                    "timer_seconds": 1200,
                    "repeat": True,
                    "exit": False,
                    "scope": "self-process Python file/function/line; no locals/data",
                    "custom_c_fd_byte_cap": False,
                    "timer_owner_api_available": False,
                }
            )
            + "\n"
        )
        attempted = True
        faulthandler.dump_traceback_later(
            1200, repeat=True, file=trace_file, exit=False
        )
        armed = True
        _active = handle
        return handle
    except BaseException:
        stop()
        return None

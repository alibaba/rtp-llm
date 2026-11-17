"""Opt-in diagnostics of this smoke test process only; never attach or signal."""

import json
import os
import threading
import time
from pathlib import Path

_TARGET = "smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100"
_out = None
_events = 0
_retained_fd = None


def phase(name):
    global _events
    try:
        if _out is None or _events >= 128:
            return
        _events += 1
        with (_out / "phases.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"phase": name, "monotonic_ns": time.monotonic_ns()}) + "\n"
            )
    except Exception:
        pass


def start(suite):
    global _out, _retained_fd, _events
    if _out is not None or _retained_fd is not None:
        return None
    if os.environ.get("RTP_BTC_0731_SELF_DIAG") != "1" or suite != _TARGET:
        return None
    import faulthandler
    import sys

    stop_event = threading.Event()
    thread = trace_file = None
    timer_attempted = False

    def cleanup():
        global _out, _retained_fd
        _out = None
        cancelled = not timer_attempted
        try:
            if timer_attempted:
                faulthandler.cancel_dump_traceback_later()
                cancelled = True
        except Exception:
            pass
        for action in (
            stop_event.set,
            lambda: thread.join(timeout=1) if thread else None,
        ):
            try:
                action()
            except Exception:
                pass
        if trace_file is not None:
            if cancelled:
                try:
                    trace_file.close()
                except Exception:
                    pass
            else:
                # Never recycle an fd if cancellation could not be established.
                _retained_fd = trace_file

    def sample():
        begun = time.monotonic()
        for seconds in (600, 1200, 1600):
            if stop_event.wait(max(0, seconds - (time.monotonic() - begun))):
                return
            try:
                current = sys._current_frames()
                main = threading.main_thread().ident
                frames = {}
                for tid in sorted(current, key=lambda x: (x != main, x))[:16]:
                    frame, stack = current[tid], []
                    while frame is not None and len(stack) < 24:
                        stack.append(
                            {
                                "file": frame.f_code.co_filename[:256],
                                "function": frame.f_code.co_name[:96],
                                "line": frame.f_lineno,
                            }
                        )
                        frame = frame.f_back
                    frames[str(tid)] = stack
                current.clear()
                proc = {}
                for name in (
                    "self/io",
                    "self/stat",
                    "self/wchan",
                    "loadavg",
                    "pressure/cpu",
                    "pressure/memory",
                    "pressure/io",
                ):
                    try:
                        with open("/proc/" + name) as stream:
                            proc[name] = stream.read(4096)
                    except OSError as exc:
                        proc[name] = str(exc)
                data = {
                    "elapsed_requested": seconds,
                    "monotonic_ns": time.monotonic_ns(),
                    "frames": frames,
                    "proc": proc,
                    "locals_collected": False,
                }
                payload = json.dumps(data)
                if len(payload.encode()) > 262144:
                    data["frames"] = {str(main): frames.get(str(main), [])}
                    data["truncated"] = True
                    payload = json.dumps(data)
                (directory / f"self_{seconds}.json").write_text(payload + "\n")
            except Exception:
                pass

    try:
        directory = (
            Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
            / "btc_0731_selfdiag"
        )
        directory.mkdir(parents=True, exist_ok=True)
        trace_file = (directory / "self_faulthandler.log").open("w")
        (directory / "start.json").write_text(
            json.dumps(
                {
                    "status": "prepared_before_timer_and_thread",
                    "suite": suite,
                    "pid": os.getpid(),
                    "python": sys.version,
                    "sample_seconds": [600, 1200, 1600],
                    "faulthandler_seconds": 300,
                    "faulthandler_exit": False,
                }
            )
            + "\n"
        )
        thread = threading.Thread(
            target=sample, name="btc-self-only-diagnostics", daemon=True
        )
        timer_attempted = True
        faulthandler.dump_traceback_later(300, repeat=True, file=trace_file, exit=False)
        thread.start()
        _out = directory
        _events = 0
        return cleanup
    except Exception:
        cleanup()
        return None

import argparse
import glob
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Type, Union

logging.basicConfig(
    level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from smoke.case_runner import CaseRunner
from smoke.common_def import REL_PATH
from smoke.gpu_diagnostics import (
    ExceptionType,
    classify_exception,
    dump_gpu_state,
    snapshot_dmesg,
)
from smoke.multi_inst_case_runner import (
    DpSeperationCaseRunner,
    FrontAppSeperationCaseRunner,
    PdSeperationCaseRunner,
    VitSeperationCaseRunner,
)
from smoke.task_info import TaskInfo
from smoke.utils import resolve_prompt_refs

from rtp_llm.utils.util import str_to_bool


def get_runner_type(
    env_args: Union[List[str], Dict[str, List[str]]]
) -> Type[CaseRunner]:
    if isinstance(env_args, list):
        return CaseRunner
    else:
        if "prefill" in env_args:
            if "DECODE_ENTRANCE=1" in env_args["prefill"]:
                return DpSeperationCaseRunner
            else:
                return PdSeperationCaseRunner
        elif "llm" in env_args and any(
            role == "vit" or role.startswith("vit_") for role in env_args
        ):
            return VitSeperationCaseRunner
        elif "frontend" in env_args:
            return FrontAppSeperationCaseRunner
        else:
            raise Exception(f"unknow env_args for runner selection: {env_args}")


def _parse_kv_list(raw: str) -> Dict[str, str]:
    """Parse a JSON list of 'KEY=VALUE' strings into a dict."""
    items = json.loads(raw)
    result = {}
    for item in items:
        k, v = item.split("=", 1)
        result[k] = v
    return result


def _btc_pending_decode_drains(text):
    import re

    pending = {}
    for line in text.splitlines():
        if "BlockTreeCache::~BlockTreeCache()" not in line:
            continue
        identity = re.search(r"\[(\d+):(\d+)\]\s+\[RANK\s", line)
        if identity is None:
            continue
        key = tuple(map(int, identity.groups()))
        if "load tickets closed, draining cache tasks..." in line:
            pending[key] = line
        elif line.endswith("] destroyed"):
            pending.pop(key, None)
    return pending


def _btc_verified_decode_process(pid, tid, test_pid, process_api):
    """Reject log PIDs unless they are live, same-UID descendants of this test."""
    test = process_api.Process(test_pid)
    if test.uids().real != os.getuid() or test.uids().effective != os.getuid():
        raise RuntimeError("test UID changed")
    children = {child.pid: child for child in test.children(recursive=True)}
    if pid not in children or pid == test_pid:
        raise RuntimeError("logged PID is not a current test descendant")
    process = children[pid]
    if process.uids().real != os.getuid() or process.uids().effective != os.getuid():
        raise RuntimeError("backend UID differs from test UID")
    if not process.is_running() or tid not in {
        thread.id for thread in process.threads()
    }:
        raise RuntimeError("backend or logged native thread no longer exists")
    return process


def _btc_capture_decode_drain_stack(
    output_dir, pid, tid, log_path, observed_stall_seconds=0, expected_marker=None
):
    import subprocess
    from pathlib import Path

    import psutil

    output = Path(output_dir)
    stem = output / f"startup_native_stack_decode_{pid}"
    result = {
        "pid": pid,
        "drain_tid": tid,
        "test_pid": os.getpid(),
        "role_log": str(log_path.relative_to(output)),
        "observed_stall_seconds": observed_stall_seconds,
        "gdb_path": shutil.which("gdb"),
        "returncode": None,
        "exception": None,
        "attach_attempted": False,
    }

    def read_proc(path):
        try:
            return Path(path).read_text().strip()
        except OSError as error:
            return type(error).__name__

    try:
        process = _btc_verified_decode_process(pid, tid, os.getpid(), psutil)
        created = process.create_time()
        result["backend_create_time"] = created
        result["uid_verified"] = os.getuid()
        result["pid_wchan"] = read_proc(f"/proc/{pid}/wchan")
        result["drain_tid_wchan"] = read_proc(f"/proc/{pid}/task/{tid}/wchan")
        result["ptrace_scope"] = read_proc("/proc/sys/kernel/yama/ptrace_scope")
        if result["gdb_path"] is None:
            raise RuntimeError("gdb is not installed")
        with stem.with_suffix(".log").open("x") as stream:
            # Refresh ancestry, UID, lifetime and thread ownership immediately before attach.
            current = _btc_verified_decode_process(pid, tid, os.getpid(), psutil)
            if current.create_time() != created:
                raise RuntimeError("backend PID was reused")
            active = _btc_pending_decode_drains(log_path.read_text(errors="replace"))
            if (pid, tid) not in active or (
                expected_marker is not None and active[(pid, tid)] != expected_marker
            ):
                raise RuntimeError("logged drain completed or changed before attach")
            result["attach_attempted"] = True
            completed = subprocess.run(
                [
                    result["gdb_path"],
                    "-nx",
                    "-batch",
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "set debuginfod enabled off",
                    "-ex",
                    "set auto-load off",
                    "-ex",
                    "set print frame-arguments none",
                    "-ex",
                    f"attach {pid}",
                    "-ex",
                    "thread apply all bt 35",
                    "-ex",
                    "detach",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=45,
                check=False,
            )
            result["returncode"] = completed.returncode
    except Exception as error:
        result["exception"] = f"{type(error).__name__}: {error}"
    finally:
        with stem.with_suffix(".json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


def _start_btc_ppu_drain_stack_observer():
    """Opt-in startup diagnostics only; leave the normal smoke path untouched."""
    if os.environ.get("RTP_BTC_PPU_DRAIN_STACKS") != "1":
        return None
    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not output_dir:
        logging.warning("BTC drain stacks requested without an artifact directory")
        return None

    import threading
    import time
    from pathlib import Path

    output = Path(output_dir).resolve()
    stop = threading.Event()

    def observe():
        waiting = {}
        attempted = set()
        while not stop.wait(1):
            pending = {}
            try:
                for path in output.glob("**/decode_logs/process.log"):
                    path = path.resolve()
                    if output not in path.parents:
                        continue
                    for key, marker in _btc_pending_decode_drains(
                        path.read_text(errors="replace")
                    ).items():
                        pending[key] = (path, marker)
                now = time.monotonic()
                for key in list(waiting):
                    if key not in pending or pending[key] != waiting[key][1]:
                        waiting.pop(key)
                for (pid, tid), marker in pending.items():
                    if pid in attempted:
                        continue
                    started, _ = waiting.setdefault((pid, tid), (now, marker))
                    if now - started < 5:
                        continue
                    attempted.add(pid)
                    _btc_capture_decode_drain_stack(
                        output, pid, tid, marker[0], now - started, marker[1]
                    )
            except Exception as error:
                logging.warning("BTC drain observer failed: %s", type(error).__name__)

    observer = threading.Thread(target=observe, name="btc-drain-stacks", daemon=True)
    observer.start()
    return stop, observer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="smoke runner")
    parser.add_argument("--suite_name", type=str, required=True, help="suite_name")
    parser.add_argument("--task_info", type=str, required=True, help="task_info")
    parser.add_argument("--envs", type=str, required=False, default="", help="envs")
    parser.add_argument(
        "--smoke_args", type=str, required=False, default="", help="smoke_args"
    )
    parser.add_argument(
        "--gpu_card", type=str, required=True, default="", help="gpu_card"
    )
    parser.add_argument(
        "--kvcm_envs", type=str, required=False, default="[]", help="KVCM server config"
    )
    parser.add_argument(
        "--sleep_time_qr", type=int, default=0, help="sleep seconds between queries"
    )
    parser.add_argument(
        "--kill_remote", type=str, default="False", help="kill KVCM server mid-test"
    )
    parser.add_argument(
        "--concurrency_test", type=str, default="False", help="concurrent request mode"
    )
    args, _ = parser.parse_known_args()

    logging.info(
        "cwd: %s envs: %s, smoke_args: %s", os.getcwd(), args.envs, args.smoke_args
    )
    with open(os.path.join(REL_PATH, args.task_info), "r") as f:
        try:
            import json5

            x = json5.load(f)
        except ImportError:
            x = json.load(f)
        if "query_result" in x:
            x["query_result"] = [resolve_prompt_refs(qr) for qr in x["query_result"]]
        task_info = TaskInfo(
            **x, taskinfo_rel_path=os.path.join(REL_PATH, args.task_info)
        )

    env_args = json.loads(args.envs)
    smoke_args = args.smoke_args
    if smoke_args:
        smoke_args_clean = smoke_args.strip().strip("'\"")
        if smoke_args_clean.startswith("{") and smoke_args_clean.endswith("}"):
            try:
                smoke_args = json.loads(smoke_args_clean)
            except json.JSONDecodeError:
                pass
        elif smoke_args.startswith('"') and smoke_args.endswith('"'):
            smoke_args = smoke_args[1:-1]

    kvcm_config = _parse_kv_list(args.kvcm_envs)

    runner_class = get_runner_type(env_args)
    logging.info("runner_class: %s", str(runner_class))
    runner_params: Dict[str, Any] = {
        "task_info": task_info,
        "env_args": env_args,
        "gpu_card": args.gpu_card,
        "smoke_args": smoke_args,
        "kvcm_config": kvcm_config,
        "sleep_time_qr": args.sleep_time_qr,
        "kill_remote": str_to_bool(args.kill_remote),
        "concurrency_test": str_to_bool(args.concurrency_test),
    }
    # prompt_batch queries are now routed to /batch_infer per-query in case_runner
    # (see CaseRunner._resolve_endpoint), no env-level switch needed.

    runner = runner_class(**runner_params)

    dmesg_baseline = snapshot_dmesg()
    btc_drain_observer = _start_btc_ppu_drain_stack_observer()

    try:
        task_states = runner.run()
    except Exception as e:
        exc_type = classify_exception(e)
        if exc_type != ExceptionType.NOT_GPU_ERROR:
            output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
            dump_gpu_state(
                exc=e,
                failure_context=f"runner exception ({exc_type.value})",
                log_path=os.path.join(output_dir, "gpu_state_runner_crash.log"),
                dmesg_baseline=dmesg_baseline,
            )
        raise
    finally:
        if btc_drain_observer is not None:
            btc_drain_observer[0].set()
            btc_drain_observer[1].join(timeout=50)
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        rocm_debug_dir = os.path.join(output_dir, "rocm_debug")
        for pattern in [
            "/tmp/rocm_debug_agent*",
            "/tmp/rocm_code_objects/*",
            "/tmp/gpucore.*",
        ]:
            for src in glob.glob(pattern):
                os.makedirs(rocm_debug_dir, exist_ok=True)
                dst = os.path.join(rocm_debug_dir, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                    logging.info("copied rocm debug artifact: %s -> %s", src, dst)
                except Exception:
                    pass

    logging.info("raw info: %s", str(task_states))
    assert task_states.ret == True, f"smoke task run failed\n{task_states}"

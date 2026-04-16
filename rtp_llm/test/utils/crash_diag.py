#!/usr/bin/env python3
"""xdist worker crash diagnostic script.

Runs 4 layers of diagnostics to capture crash site information when
pytest-xdist workers crash with "Not properly terminated" on remote
GPU workers.

All subprocess output is written to /tmp/rtp_crash_diag/ files, then
dumped to stdout at the end. This avoids output loss from pytest capture
or xdist stderr swallowing.

Usage (inside REAPI session, after venv activation):
    python rtp_llm/test/utils/crash_diag.py [--layers L1,L2,L3,L4]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys


FAULT_DIR = "/tmp/rtp_xdist_crash"
DIAG_DIR = "/tmp/rtp_crash_diag"


def _ensure_diag_dir():
    os.makedirs(DIAG_DIR, exist_ok=True)


def _banner(layer: str, desc: str):
    print(f"\n{'='*60}")
    print(f">>>DIAG_{layer}_START {desc}")
    print(f"{'='*60}")
    sys.stdout.flush()


def _footer(layer: str, exit_code: int):
    print(f">>>DIAG_{layer}_EXIT={exit_code}")
    print(f">>>DIAG_{layer}_END")
    sys.stdout.flush()


def _dump_file(path: str, label: str, max_bytes: int = 30000):
    """Read a file and print its contents with a label."""
    print(f"\n>>>FILE:{label}")
    try:
        content = open(path).read()
        if not content.strip():
            print("(empty)")
        elif len(content) > max_bytes:
            print(f"... (truncated, showing last {max_bytes} bytes)")
            print(content[-max_bytes:])
        else:
            print(content)
    except FileNotFoundError:
        print("(file not found)")
    except Exception as e:
        print(f"(error: {e})")
    print(f">>>END_FILE:{label}")
    sys.stdout.flush()


def layer0_per_dir_collect():
    """Layer 0: Collect tests per-directory to isolate which import crashes.

    Runs --collect-only on each testpath individually. The first directory
    that returns exit 255 contains the crashing import.
    """
    _banner("L0", "Per-directory collection (isolate crashing import)")
    _ensure_diag_dir()

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0]

    # Drill into rtp_llm/models_py subdirectories (identified as crash source)
    models_py = "rtp_llm/models_py"
    testpaths = []
    if os.path.isdir(models_py):
        for entry in sorted(os.listdir(models_py)):
            subdir = os.path.join(models_py, entry)
            if os.path.isdir(subdir):
                # Look for test/ subdirectories
                test_dir = os.path.join(subdir, "test")
                if os.path.isdir(test_dir):
                    testpaths.append(test_dir)
                else:
                    # Some dirs have tests deeper (e.g. kernels/cuda/test)
                    for root, dirs, files in os.walk(subdir):
                        if "test" in dirs:
                            testpaths.append(os.path.join(root, "test"))
    if not testpaths:
        # Fallback to full testpaths
        testpaths = [
            "rtp_llm/test",
            "rtp_llm/models_py",
            "rtp_llm/cpp/models",
        ]
    print(f"  L0 scanning {len(testpaths)} subdirectories of {models_py}")

    results = {}
    for tp in testpaths:
        stdout_log = f"{DIAG_DIR}/L0_{tp.replace('/', '_')}_stdout.log"
        stderr_log = f"{DIAG_DIR}/L0_{tp.replace('/', '_')}_stderr.log"

        with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "pytest",
                        "--collect-only", "-q",
                        "-p", "no:remote-gpu", "-p", "no:rtp-ci-profile",
                        "--override-ini=addopts=",
                        "--continue-on-collection-errors",
                        "--override-ini=testpaths=" + tp,
                    ],
                    env=env, timeout=60,
                    stdout=fout, stderr=ferr,
                )
                ec = result.returncode
            except subprocess.TimeoutExpired:
                ec = -999

        status = "OK" if ec == 0 else f"EXIT={ec}"
        results[tp] = ec
        # Only dump details for failures
        if ec != 0 and ec != 5:  # 5 = no tests collected (OK)
            print(f"  {tp}: {status} *** CRASH ***")
            _dump_file(stdout_log, f"L0_{tp}_stdout", max_bytes=3000)
            _dump_file(stderr_log, f"L0_{tp}_stderr", max_bytes=3000)
        else:
            print(f"  {tp}: {status}")

    print(f"\n>>>L0_RESULTS {results}")
    crashed = [tp for tp, ec in results.items() if ec not in (0, 5)]
    if crashed:
        print(f">>>L0_CRASHED_DIRS {crashed}")
    else:
        print(">>>L0_ALL_OK (crash may be from testpath interaction, not individual dirs)")

    _footer("L0", 0 if not crashed else 1)
    return 0 if not crashed else 1


def _build_exit_trap_so() -> str:
    """Compile a tiny LD_PRELOAD .so that intercepts exit() and prints backtrace."""
    so_path = f"{DIAG_DIR}/exit_trap.so"
    c_path = f"{DIAG_DIR}/exit_trap.c"
    _ensure_diag_dir()

    c_code = r"""
#define _GNU_SOURCE
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

/* Intercept both exit() and _exit() to catch os._exit() from Python */

void exit(int status) {
    void *bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n=== EXIT(%d) INTERCEPTED — C backtrace ===\n", status);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    fprintf(stderr, "=== END EXIT TRACE ===\n");
    fflush(stderr);
    /* Call the real _exit via syscall to avoid recursion */
    syscall(231, status); /* __NR_exit_group = 231 on x86_64 */
}

void _exit(int status) {
    void *bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n=== _EXIT(%d) INTERCEPTED — C backtrace ===\n", status);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    fprintf(stderr, "=== END _EXIT TRACE ===\n");
    fflush(stderr);
    syscall(231, status); /* __NR_exit_group */
}

void _Exit(int status) {
    void *bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n=== _Exit(%d) INTERCEPTED — C backtrace ===\n", status);
    backtrace_symbols_fd(bt, n, STDERR_FILENO);
    fprintf(stderr, "=== END _Exit TRACE ===\n");
    fflush(stderr);
    syscall(231, status);
}
"""
    with open(c_path, "w") as f:
        f.write(c_code)

    result = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", so_path, c_path, "-ldl"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[exit_trap] gcc failed: {result.stderr}")
        return ""
    print(f"[exit_trap] compiled {so_path}")
    return so_path


def layer_exit_trap():
    """Layer E: Intercept exit() via LD_PRELOAD to get C++ backtrace.

    Compiles a tiny .so that wraps exit() with backtrace_symbols_fd(),
    then runs pytest --collect-only with LD_PRELOAD. When any code calls
    exit(-1), the backtrace is printed to stderr BEFORE the process dies.
    """
    _banner("LE", "LD_PRELOAD exit() trap (C++ backtrace on exit)")
    _ensure_diag_dir()

    # Use strace to intercept at kernel level — the only way to catch raw syscalls
    strace_log = f"{DIAG_DIR}/LE_strace.log"
    stdout_log = f"{DIAG_DIR}/LE_stdout.log"
    stderr_log = f"{DIAG_DIR}/LE_stderr.log"

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0]

    has_strace = shutil.which("strace") is not None
    if not has_strace:
        print("(strace not available, trying gdb fallback)")
        return layer_gdb_catch_exit()

    # strace -f: follow forks
    # -e trace=exit_group,exit,rt_sigaction,rt_sigprocmask: catch exits and signals
    # -e signal=all: show signal delivery
    # -t: timestamps
    # -o: output to file (keeps program stdout/stderr clean)
    cmd = [
        "strace", "-f", "-t",
        "-e", "trace=exit_group,exit,kill,tgkill,rt_sigaction",
        "-e", "signal=all",
        "-o", strace_log,
        sys.executable, "-m", "pytest",
        "--collect-only", "-q",
        "-p", "no:remote-gpu", "-p", "no:rtp-ci-profile",
        "--override-ini=addopts=",
        "--continue-on-collection-errors",
    ]

    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(
            cmd, env=env, timeout=180,
            stdout=fout, stderr=ferr,
        )

    _dump_file(stdout_log, "LE_stdout")
    _dump_file(stderr_log, "LE_stderr")

    # Parse strace output for exit_group calls
    print("\n>>>STRACE_EXIT_CALLS")
    try:
        with open(strace_log) as f:
            lines = f.readlines()
        # Find exit_group and exit calls
        exit_lines = [l for l in lines if "exit_group" in l or "exit(" in l or "+++ exited" in l]
        # Find the last 200 lines before the exit_group(-1) call
        for i, l in enumerate(lines):
            if "exit_group" in l and "255" in l or "exit_group" in l and "-1" in l:
                print(f">>> FOUND exit_group(-1) at strace line {i+1}")
                # Print 200 lines before it for context
                start = max(0, i - 200)
                print(f">>> Context ({start+1} to {i+1}):")
                for ctx_line in lines[start:i+1]:
                    print(ctx_line.rstrip())
                break
        else:
            # Didn't find exit_group(-1), print all exit calls
            print(f"(no exit_group(-1/255) found, showing all {len(exit_lines)} exit calls)")
            for l in exit_lines[-30:]:
                print(l.rstrip())
    except Exception as e:
        print(f"(strace parse error: {e})")

    # Also dump strace file size and tail
    print(f"\n>>>STRACE_FILE_INFO size={os.path.getsize(strace_log)} bytes")
    _dump_file(strace_log, "LE_strace_tail", max_bytes=10000)

    _footer("LE", result.returncode)
    return result.returncode


def layer_gdb_catch_exit():
    """Layer G: Use gdb 'catch syscall exit_group' to get C++ backtrace.

    Unlike LD_PRELOAD, gdb catch syscall intercepts at kernel level AND
    provides full C++ backtrace. This is the definitive way to identify
    which function calls exit_group(-1).
    """
    _banner("LG", "gdb catch syscall exit_group (C++ backtrace)")
    _ensure_diag_dir()

    if not shutil.which("gdb"):
        print("(gdb not available, skipping)")
        _footer("LG", -1)
        return -1

    gdb_log = f"{DIAG_DIR}/LG_gdb.log"
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0]

    script = (
        "import pytest; pytest.main(['--collect-only', '-q', "
        "'-p', 'no:remote-gpu', '-p', 'no:rtp-ci-profile', "
        "'--override-ini=addopts=', '--continue-on-collection-errors'])"
    )

    with open(gdb_log, "w") as f:
        result = subprocess.run(
            [
                "gdb", "-batch",
                "-ex", "set confirm off",
                "-ex", "set pagination off",
                # Catch the exit_group syscall at kernel level
                "-ex", "catch syscall exit_group",
                "-ex", "run",
                # When caught, print full backtrace
                "-ex", "echo \\n=== exit_group CAUGHT — backtrace ===\\n",
                "-ex", "bt 50",
                "-ex", "echo \\n=== full backtrace ===\\n",
                "-ex", "bt full 30",
                "-ex", "echo \\n=== all threads ===\\n",
                "-ex", "info threads",
                "-ex", "echo \\n=== current thread registers ===\\n",
                "-ex", "info registers rdi",
                "-ex", "quit",
                "--args", sys.executable, "-c", script,
            ],
            env=env, timeout=180,
            stdout=f, stderr=subprocess.STDOUT,
        )

    _dump_file(gdb_log, "LG_gdb", max_bytes=20000)
    _footer("LG", result.returncode)
    return result.returncode


def layer1_no_xdist_collect():
    """Layer 1: Collect tests without xdist (-n 0).

    If the crash happens during import/collection, it will occur in the
    main process where faulthandler output goes directly to the log file.
    """
    _banner("L1", "No-xdist collection (crash = import-time CUDA issue)")
    _ensure_diag_dir()

    stdout_log = f"{DIAG_DIR}/L1_stdout.log"
    stderr_log = f"{DIAG_DIR}/L1_stderr.log"

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    # Use first GPU only
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        env["CUDA_VISIBLE_DEVICES"] = cvd.split(",")[0]

    script = (
        "import faulthandler, sys, atexit, traceback, os\n"
        "faulthandler.enable(file=sys.stderr, all_threads=True)\n"
        "# Catch SystemExit and unhandled exceptions\n"
        "def _diag_excepthook(exc_type, exc_val, exc_tb):\n"
        "    sys.stderr.write('[L1_EXCEPTHOOK] %s: %s\\n' % (exc_type.__name__, exc_val))\n"
        "    traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)\n"
        "    sys.stderr.flush()\n"
        "sys.excepthook = _diag_excepthook\n"
        "atexit.register(lambda: sys.stderr.write('[L1_ATEXIT] normal exit\\n'))\n"
        "print('[L1] faulthandler enabled, PID=%d CVD=%s' % "
        "(os.getpid(), os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')))\n"
        "sys.stdout.flush()\n"
        "import pytest\n"
        "ec = pytest.main([\n"
        "    '--collect-only', '-q',\n"
        "    '-p', 'no:remote-gpu', '-p', 'no:rtp-ci-profile',\n"
        "    '--override-ini=addopts=',\n"
        "    '--continue-on-collection-errors',\n"
        "])\n"
        "print('[L1] collection completed, exit_code=%d' % ec)\n"
    )

    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, timeout=120,
            stdout=fout, stderr=ferr,
        )

    _dump_file(stdout_log, "L1_stdout")
    _dump_file(stderr_log, "L1_stderr")
    _footer("L1", result.returncode)
    return result.returncode


def layer2_simulated_worker():
    """Layer 2: Simulate a single xdist worker in-process.

    Sets PYTEST_XDIST_WORKER=gw0 and executes conftest.py, then collects
    tests. Crash happens in current process so faulthandler is visible.
    """
    _banner("L2", "Simulated xdist worker (single process)")
    _ensure_diag_dir()

    stdout_log = f"{DIAG_DIR}/L2_stdout.log"
    stderr_log = f"{DIAG_DIR}/L2_stderr.log"

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTEST_XDIST_WORKER"] = "gw0"
    env["GPU_COUNT_PER_WORKER"] = "1"
    # Ensure CVD is set to all GPUs (simulating device_resource.py output)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    script = (
        "import faulthandler, sys, os, atexit, traceback\n"
        "faulthandler.enable(file=sys.stderr, all_threads=True)\n"
        "def _diag_excepthook(et, ev, tb):\n"
        "    sys.stderr.write('[L2_EXCEPTHOOK] %s: %s\\n' % (et.__name__, ev))\n"
        "    traceback.print_exception(et, ev, tb, file=sys.stderr)\n"
        "sys.excepthook = _diag_excepthook\n"
        "atexit.register(lambda: sys.stderr.write('[L2_ATEXIT] normal exit\\n'))\n"
        "print('[L2] PID=%d CVD=%s WORKER=%s' % (\n"
        "    os.getpid(),\n"
        "    os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'),\n"
        "    os.environ.get('PYTEST_XDIST_WORKER', 'unset')))\n"
        "sys.stdout.flush()\n"
        "\n"
        "# Step 1: Load conftest.py (what xdist worker does first)\n"
        "sys.path.insert(0, '.')\n"
        "exec(open('conftest.py').read())\n"
        "print('[L2] After conftest: CVD=%s' % os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'))\n"
        "sys.stdout.flush()\n"
        "\n"
        "# Step 2: Collect tests (triggers imports)\n"
        "import pytest\n"
        "ec = pytest.main([\n"
        "    '--collect-only', '-q',\n"
        "    '-p', 'no:remote-gpu', '-p', 'no:rtp-ci-profile',\n"
        "    '--override-ini=addopts=',\n"
        "    '--continue-on-collection-errors',\n"
        "])\n"
        "print('[L2] collection exit_code=%d' % ec)\n"
    )

    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, timeout=120,
            stdout=fout, stderr=ferr,
        )

    _dump_file(stdout_log, "L2_stdout")
    _dump_file(stderr_log, "L2_stderr")
    _footer("L2", result.returncode)
    return result.returncode


def layer3_file_faulthandler_xdist():
    """Layer 3: Run xdist with file-based faulthandler.

    conftest.py writes faulthandler output to /tmp/rtp_xdist_crash/*.fault.
    After the run, we read and dump those files.
    """
    _banner("L3", "xdist with file-based faulthandler")
    _ensure_diag_dir()

    stdout_log = f"{DIAG_DIR}/L3_stdout.log"
    stderr_log = f"{DIAG_DIR}/L3_stderr.log"

    # Clean up previous fault files
    if os.path.exists(FAULT_DIR):
        shutil.rmtree(FAULT_DIR, ignore_errors=True)
    os.makedirs(FAULT_DIR, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["GPU_COUNT_PER_WORKER"] = "1"

    # Run through device_resource.py wrapper (sets CVD)
    cmd = [
        sys.executable, "rtp_llm/test/utils/device_resource.py",
        sys.executable, "-m", "pytest",
        "-p", "no:remote-gpu", "-p", "no:rtp-ci-profile",
        "--collect-only", "-n", "4",
        "--override-ini=addopts=",
        "--continue-on-collection-errors",
    ]

    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(cmd, env=env, timeout=180, stdout=fout, stderr=ferr)

    _dump_file(stdout_log, "L3_stdout")
    _dump_file(stderr_log, "L3_stderr")
    _footer("L3", result.returncode)

    # Dump fault files
    print("\n>>>DIAG_L3_FAULT_FILES")
    fault_files = sorted(glob.glob(f"{FAULT_DIR}/*.fault"))
    if not fault_files:
        print("(no fault files found)")
    for f in fault_files:
        _dump_file(f, f"fault_{os.path.basename(f)}")

    return result.returncode


def layer4_system_tools():
    """Layer 4: strace + core dump analysis.

    Uses OS-level tools to capture crash signals and core dumps.
    """
    _banner("L4", "System-level crash analysis (strace/coredump)")
    _ensure_diag_dir()

    stdout_log = f"{DIAG_DIR}/L4_stdout.log"
    stderr_log = f"{DIAG_DIR}/L4_stderr.log"

    # 4a: Check tool availability
    has_gdb = shutil.which("gdb") is not None
    has_strace = shutil.which("strace") is not None
    print(f"GDB={'yes' if has_gdb else 'no'}")
    print(f"STRACE={'yes' if has_strace else 'no'}")

    # Read core pattern
    try:
        core_pattern = open("/proc/sys/kernel/core_pattern").read().strip()
        print(f"core_pattern={core_pattern}")
    except Exception:
        print("core_pattern=UNREADABLE")

    # 4b: Set up core dumps
    cores_dir = "/tmp/cores"
    os.makedirs(cores_dir, exist_ok=True)
    subprocess.run(["bash", "-c", "ulimit -c unlimited"], check=False)
    subprocess.run(
        ["bash", "-c", f'echo "{cores_dir}/core.%e.%p" > /proc/sys/kernel/core_pattern'],
        check=False,
    )

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["GPU_COUNT_PER_WORKER"] = "1"

    base_cmd = [
        sys.executable, "rtp_llm/test/utils/device_resource.py",
        sys.executable, "-m", "pytest",
        "-p", "no:remote-gpu", "-p", "no:rtp-ci-profile",
        "--collect-only", "-n", "2",
        "--override-ini=addopts=",
        "--continue-on-collection-errors",
    ]

    # 4c: strace if available
    if has_strace:
        strace_log = "/tmp/strace_xdist.log"
        cmd = [
            "strace", "-f", "-e", "trace=signal",
            "-o", strace_log,
        ] + base_cmd
        with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
            result = subprocess.run(cmd, env=env, timeout=180, stdout=fout, stderr=ferr)
        _footer("L4_STRACE", result.returncode)

        # Parse strace output
        print("\n>>>DIAG_L4_STRACE_SIGNALS")
        try:
            lines = open(strace_log).readlines()
            sig_lines = [l for l in lines if any(k in l for k in ["SIG", "kill", "exit_group"])]
            for line in sig_lines[-50:]:
                print(line.rstrip())
        except Exception as e:
            print(f"(strace parse error: {e})")
    else:
        # Run without strace, hope for core dumps
        with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
            result = subprocess.run(base_cmd, env=env, timeout=180, stdout=fout, stderr=ferr)
        _footer("L4_NOSTRACE", result.returncode)

    _dump_file(stdout_log, "L4_stdout")
    _dump_file(stderr_log, "L4_stderr")

    # 4d: Analyze core dumps
    print("\n>>>DIAG_L4_CORES")
    core_files = []
    for search_dir in [cores_dir, "."]:
        for f in glob.glob(f"{search_dir}/core*"):
            if os.path.getsize(f) > 0:
                core_files.append(f)
    core_files = core_files[:3]

    if not core_files:
        print("(no core files found)")
    elif has_gdb:
        python_bin = sys.executable
        for core in core_files:
            print(f"\n--- analyzing {core} ---")
            gdb_result = subprocess.run(
                [
                    "gdb", "-batch",
                    "-ex", "bt",
                    "-ex", "bt full",
                    "-ex", "info threads",
                    "-ex", "thread apply all bt full",
                    "-ex", "quit",
                    python_bin, core,
                ],
                capture_output=True, text=True, timeout=60,
            )
            output = gdb_result.stdout[:8000]
            print(output)
            if gdb_result.stderr.strip():
                print(f"gdb stderr: {gdb_result.stderr[:2000]}")
    else:
        for core in core_files:
            print(f"{core}: {os.path.getsize(core)} bytes")
            subprocess.run(["file", core], check=False)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="xdist crash diagnostics")
    parser.add_argument(
        "--layers", default="LG,LE,L0,L1,L2,L3,L4",
        help="Comma-separated layers to run (default: LG,LE,L0,L1,L2,L3,L4)",
    )
    args = parser.parse_args()
    layers = [l.strip().upper() for l in args.layers.split(",")]

    _ensure_diag_dir()

    print(f">>>DIAG_START layers={','.join(layers)}")
    print(f"PID={os.getpid()} CWD={os.getcwd()}")
    print(f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"HVD={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}")
    print(f"PYTHON={sys.executable}")
    print(f"DIAG_DIR={DIAG_DIR}")
    sys.stdout.flush()

    results = {}

    if "LG" in layers:
        results["LG"] = layer_gdb_catch_exit()

    if "LE" in layers:
        results["LE"] = layer_exit_trap()

    if "L0" in layers:
        results["L0"] = layer0_per_dir_collect()

    if "L1" in layers:
        results["L1"] = layer1_no_xdist_collect()

    if "L2" in layers:
        results["L2"] = layer2_simulated_worker()

    if "L3" in layers:
        results["L3"] = layer3_file_faulthandler_xdist()

    if "L4" in layers:
        results["L4"] = layer4_system_tools()

    # Dump all diag files inventory
    print("\n>>>DIAG_FILES_INVENTORY")
    for d in [DIAG_DIR, FAULT_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                print(f"  {fp} ({os.path.getsize(fp)} bytes)")

    print(f"\n>>>DIAG_SUMMARY {results}")
    print(">>>DIAG_END")


if __name__ == "__main__":
    main()

"""Check whether 48 task-local pread workers pass the first FP8 layer."""

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


ROOT = Path("/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927")
RUNFILES = Path(
    "/data0/luohaocheng.lhc/.cache/bazel/k3-integrated-perf-20260929/"
    "72fe907babae969d3cf28e3db89db495/execroot/rtp_llm/"
    "bazel-out/k8-opt/bin/rtp_llm/rtp_llm_server.runfiles"
)
LOADER = RUNFILES / "rtp_llm/rtp_llm/model_loader/loader.py"
INSTRUMENT = ROOT / "full93-diag-loader-instrumentation-115-20260929.py"
SOURCE_SCRIPT = ROOT / "run_fp8_3fs_integrated_full93_single_diag_20260929.sh"
SCRIPT = ROOT / "run_fp8_3fs_integrated_full93_pread48_diag_20260929.sh"
RUN = Path("/data0/luohaocheng.lhc/k3integrated-115-fp8-93l-20260929-single-diag-r6")
SUMMARY = ROOT / "full93-pread48-single-diag-115-20260929.json"


def progress():
    status = {}
    counts = []
    for rank in range(8):
        log = RUN / "logs" / f"main_{rank}.log"
        if log.exists():
            events = re.findall(r"K3_DIAG_(?:PROGRESS|LOAD_BEGIN|LOAD_END) count=(\d+)[^\n]*", log.read_text(errors="replace"))
        else:
            events = []
        counts.append(max((int(count) for count in events), default=0))
        status[str(rank)] = counts[-1]
    return status, all(count >= 20 for count in counts)


def main():
    assert os.getuid() != 0
    assert LOADER.is_symlink()
    link_target = os.readlink(LOADER)
    assert Path(link_target).is_file()
    assert not RUN.exists() and not Path(str(RUN) + ".server-stdio.log").exists()
    script = SOURCE_SCRIPT.read_text()
    assert "single-diag-r2" in script
    assert "export K3_3FS_PREAD_THREADS=64" in script
    assert "libparallel_3fs_pread_20260929.so:$deps" in script
    script = script.replace("single-diag-r2", "single-diag-r6")
    script = script.replace("export K3_3FS_PREAD_THREADS=64", "export K3_3FS_PREAD_THREADS=48")
    SCRIPT.write_text(script)
    SCRIPT.chmod(0o700)
    started = time.monotonic()
    proc = None
    reached = False
    LOADER.unlink()
    try:
        shutil.copyfile(INSTRUMENT, LOADER)
        env = os.environ.copy()
        env.pop("FASTSAFETENSORS_NOGDS", None)
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "115", "run"],
            env=env,
            start_new_session=True,
        )
        deadline = started + 180
        while proc.poll() is None and time.monotonic() < deadline:
            _, reached = progress()
            if reached:
                break
            time.sleep(1)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # The server can fork a child whose parent exits after TERM.
            # The process group is task-owned and must be cleared as a whole.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
        LOADER.unlink(missing_ok=True)
        LOADER.symlink_to(link_target)
    counts, final_reached = progress()
    result = {
        "purpose": "bounded loading-path diagnosis; no smoke request or benchmark",
        "fastsafetensors_nogds": False,
        "pread_helper": True,
        "pread_threads": 48,
        "elapsed_s": round(time.monotonic() - started, 1),
        "launcher_exit": proc.returncode if proc else None,
        "all_ranks_reached_tensor_20": final_reached,
        "rank_max_load_event_count": counts,
        "run": str(RUN),
        "runfile_symlink_restored": LOADER.is_symlink() and os.readlink(LOADER) == link_target,
    }
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

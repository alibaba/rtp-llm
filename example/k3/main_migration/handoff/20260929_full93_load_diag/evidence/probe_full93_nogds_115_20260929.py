"""Bounded task-only startup probe; always restore the Bazel runfile symlink."""

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
SCRIPT = ROOT / "run_fp8_3fs_integrated_full93_nogds_diag_20260929.sh"
RUN = Path("/data0/luohaocheng.lhc/k3integrated-115-fp8-93l-20260929-single-diag-r3-nogds")
SUMMARY = ROOT / "full93-nogds-single-diag-115-20260929.json"


def main():
    assert os.getuid() != 0
    assert LOADER.is_symlink()
    link_target = os.readlink(LOADER)
    assert Path(link_target).is_file()
    assert not RUN.exists() and not Path(str(RUN) + ".server-stdio.log").exists()
    script = SOURCE_SCRIPT.read_text()
    assert "single-diag-r2" in script
    SCRIPT.write_text(script.replace("single-diag-r2", "single-diag-r3-nogds"))
    SCRIPT.chmod(0o700)
    started = time.monotonic()
    proc = None
    LOADER.unlink()
    try:
        shutil.copyfile(INSTRUMENT, LOADER)
        env = os.environ.copy()
        env["FASTSAFETENSORS_NOGDS"] = "1"
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "115", "run"],
            env=env,
            start_new_session=True,
        )
        deadline = started + 180
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(1)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
        LOADER.unlink(missing_ok=True)
        LOADER.symlink_to(link_target)
    rank_status = {}
    for rank in range(8):
        log = RUN / "logs" / f"main_{rank}.log"
        if not log.exists():
            rank_status[str(rank)] = "no_log"
            continue
        events = re.findall(r"K3_DIAG_(?:PROGRESS|LOAD_BEGIN|LOAD_END) count=\d+[^\n]*", log.read_text(errors="replace"))
        rank_status[str(rank)] = events[-1] if events else "no_progress"
    result = {
        "purpose": "bounded startup diagnosis, no smoke requests or benchmark measurement",
        "nogds": True,
        "pread_threads": 64,
        "elapsed_s": round(time.monotonic() - started, 1),
        "launcher_exit": proc.returncode if proc else None,
        "run": str(RUN),
        "rank_last_events": rank_status,
        "runfile_symlink_restored": LOADER.is_symlink() and os.readlink(LOADER) == link_target,
    }
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

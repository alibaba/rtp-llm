#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only container visibility report. Does not collect environment variables or credentials."""
import glob
import json
import os
import platform
import stat
import subprocess


def emit(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


emit("platform", hostname=platform.node(), machine=platform.machine(), kernel=platform.release(), uid=os.geteuid())
for pattern in ("/dev/nvidia*", "/dev/nvidia-caps/*", "/dev/nvidia-caps-imex-channels/*", "/dev/infiniband/*"):
    paths = sorted(glob.glob(pattern))
    emit("device_glob", pattern=pattern, count=len(paths))
    for path in paths:
        try:
            s = os.stat(path)
            fields = dict(path=path, mode=oct(stat.S_IMODE(s.st_mode)), uid=s.st_uid, gid=s.st_gid,
                          is_character_device=stat.S_ISCHR(s.st_mode),
                          read_access=os.access(path, os.R_OK), write_access=os.access(path, os.W_OK))
            if stat.S_ISCHR(s.st_mode):
                fields.update(major=os.major(s.st_rdev), minor=os.minor(s.st_rdev))
            emit("device_visibility", **fields)
        except OSError as exc:
            emit("device_visibility", path=path, error=str(exc))

for path in ("/proc/driver/nvidia/version", "/sys/devices/system/node/online", "/proc/self/cgroup",
             "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/cpuset.mems.effective",
             "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/cpuset/cpuset.mems"):
    try:
        with open(path, encoding="utf-8") as f:
            emit("system_file", path=path, value=f.read(16384).strip())
    except OSError as exc:
        emit("system_file", path=path, error=str(exc))

for path in sorted(glob.glob("/sys/devices/system/node/node*/meminfo")):
    try:
        with open(path, encoding="utf-8") as f:
            emit("numa_meminfo", path=path, value=f.read(16384).strip())
    except OSError as exc:
        emit("numa_meminfo", path=path, error=str(exc))

for command in (
    ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"],
    ["nvidia-smi", "topo", "-m"],
    ["nvidia-smi", "nvlink", "--status"],
    ["nvcc", "--version"],
):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        emit("command", command=command, returncode=result.returncode,
             stdout=result.stdout[:32768], stderr=result.stderr[:8192])
    except (OSError, subprocess.TimeoutExpired) as exc:
        emit("command", command=command, error=str(exc))

emit("notice", message="Device nodes/access bits and local topology are inventory only. Remote EGM readiness requires owner + remote GPU consumer PASS; this script does not prove a shared NVL72 domain.")

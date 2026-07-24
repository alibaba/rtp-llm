#!/usr/bin/env python3
"""G1: lock 新鲜度。校验 lock 头部的 input-hash 是否等于当前 requirements 输入的 sha256。

input-hash 由 update_locks.sh 写入（格式: `# input-hash: <hex>`）。
无 stamp 的 lock 仅告警（迁移期兼容 pip-compile 产物）。
用法: python3 deps/check_lock_freshness.py [deps_dir]
"""
import hashlib
import re
import sys
from pathlib import Path

DEPS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

# lock 文件 -> 输入 requirements（与 update_locks.sh 保持一致：base, arch_src, bazel_provided）
LOCK_INPUTS = {
    "requirements_lock_torch_gpu_cuda12.txt": ["requirements_base.txt", "requirements_torch_gpu_cuda12.txt", "bazel_provided_cuda12.txt"],
    "requirements_lock_torch_gpu_cuda12_9.txt": ["requirements_base.txt", "requirements_torch_gpu_cuda12_9.txt", "bazel_provided_cuda12_9.txt"],
    "requirements_lock_rocm.txt": ["requirements_base.txt", "requirements_rocm.txt", "bazel_provided_rocm.txt"],
    "requirements_lock_torch_cpu.txt": ["requirements_base.txt", "requirements_torch_cpu.txt", "bazel_provided_cpu.txt"],
    "requirements_lock_torch_arm.txt": ["requirements_base.txt", "requirements_cpu_arm.txt", "bazel_provided_arm.txt"],
    "requirements_lock_cuda12_arm.txt": ["requirements_base.txt", "requirements_cuda12_arm.txt", "bazel_provided_cuda12_arm.txt"],
}

STAMP_RE = re.compile(r"^#\s*input-hash:\s*([0-9a-f]{64})", re.M)


def input_hash(files):
    h = hashlib.sha256()
    for f in files:
        h.update((DEPS / f).read_bytes())
    return h.hexdigest()


def main():
    stale, unstamped = [], []
    for lock_name, inputs in LOCK_INPUTS.items():
        lock = DEPS / lock_name
        if not lock.exists():
            continue
        m = STAMP_RE.search(lock.read_text())
        if not m:
            unstamped.append(lock_name)
            continue
        if m.group(1) != input_hash(inputs):
            stale.append(lock_name)
    for n in unstamped:
        print(f"WARN {n}: 无 input-hash stamp（旧 pip-compile 产物），跑 deps/update_locks.sh 后消除")
    if stale:
        for n in stale:
            print(f"FAIL {n}: requirements 已改但 lock 未重编")
        print("修复: bash deps/update_locks.sh")
        sys.exit(1)
    print(f"OK: {len(LOCK_INPUTS) - len(unstamped)} 份 lock 新鲜（{len(unstamped)} 份无 stamp 仅告警）")


if __name__ == "__main__":
    main()

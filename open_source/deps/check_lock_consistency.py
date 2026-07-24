#!/usr/bin/env python3
"""G3: 跨 lock hash 一致性。同一 name==version 在**同平台**多份 lock 中的 hash 集合不允许矛盾。

不同 CPU 平台（x86_64 vs aarch64）的 wheel hash 本就不同，只在同平台组内比较。
同平台组内 hash 集合无交集，说明锁的是内容不同的产物（典型原因: 私有 OSS 上同版本被覆盖）。
用法: python3 deps/check_lock_consistency.py [deps_dir]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

DEPS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

# lock -> 平台组（与 update_locks.sh 的 --python-platform 保持一致）
PLATFORM_GROUPS = {
    "requirements_lock_torch_gpu_cuda12.txt": "x86_64",
    "requirements_lock_torch_gpu_cuda12_9.txt": "x86_64",
    "requirements_lock_rocm.txt": "x86_64",
    "requirements_lock_torch_cpu.txt": "x86_64",
    "requirements_lock_torch_arm.txt": "aarch64",
    "requirements_lock_cuda12_arm.txt": "aarch64",
}

PKG_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ \\;]+)")
HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")


def parse_lock(path):
    """返回 {(name, version): set(hashes)}"""
    out = defaultdict(set)
    cur = None
    for line in path.read_text().splitlines():
        s = line.strip()
        m = PKG_RE.match(s)
        if m:
            cur = (m.group(1).lower().replace("_", "-"), m.group(2))
        for h in HASH_RE.findall(s):
            if cur:
                out[cur].add(h)
    return out


def main():
    locks = sorted(DEPS.glob("requirements_lock_*.txt"))
    if not locks:
        print("SKIP: 未找到 lock 文件")
        return
    seen = {}  # (platform, name, version) -> (lockname, hashes)
    conflicts = []
    total = 0
    for lock in locks:
        plat = PLATFORM_GROUPS.get(lock.name, "unknown")
        for (name, ver), hashes in parse_lock(lock).items():
            key = (plat, name, ver)
            if key in seen:
                prev_lock, prev_hashes = seen[key]
                if not (hashes & prev_hashes):
                    conflicts.append(((name, ver), prev_lock, lock.name))
            else:
                seen[key] = (lock.name, hashes)
                total += 1
    if conflicts:
        for (name, ver), l1, l2 in conflicts:
            print(f"FAIL {name}=={ver}: {l1} 与 {l2} 的 hash 集合无交集（疑似 OSS 同版本被覆盖）")
        print("修复: 确认覆盖来源，重传为新 local version 后重编所有受影响 lock")
        sys.exit(1)
    print(f"OK: {len(locks)} 份 lock，{total} 个 (platform,name,version)，无 hash 冲突")


if __name__ == "__main__":
    main()

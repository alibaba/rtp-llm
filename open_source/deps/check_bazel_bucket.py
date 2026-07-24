#!/usr/bin/env python3
"""门禁：bazel 桶自洽性（治痛点 9 的版本项）。

FAIL 项（逐 arch）：桶里的包出现在同 arch 的 lock 中 ⇒ 双供给（bazel 与 pip 同时提供），
  说明 --no-emit-package 没生效或 requirements 又写回了桶包。
WARN 项（尽力而为）：桶 pin 的版本串在 bazel 侧定义文件（http.bzl/git.bzl/arch_select.bzl）
  中找不到——源构建包（deep-ep 等）的版本只体现在 +sha 上，属预期；torch 类 URL 嵌版本应能找到。
用法: python3 deps/check_bazel_bucket.py [repo_root]
"""
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
DEPS = ROOT / "deps"

CFG = {
    "cuda12": "requirements_lock_torch_gpu_cuda12.txt",
    "cuda12_9": "requirements_lock_torch_gpu_cuda12_9.txt",
    "rocm": "requirements_lock_rocm.txt",
    "cpu": "requirements_lock_torch_cpu.txt",
    "arm": "requirements_lock_torch_arm.txt",
    "cuda12_arm": "requirements_lock_cuda12_arm.txt",
}
BAZEL_FILES = [DEPS / "http.bzl", DEPS / "git.bzl", ROOT / "arch_config" / "arch_select.bzl"]


def norm(n):
    return n.lower().replace("_", "-")


def bucket(arch):
    out = {}
    p = DEPS / f"bazel_provided_{arch}.txt"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s and "==" in s:
            name, ver = s.split("==", 1)
            out[norm(name)] = ver
    return out


def lock_names(path):
    names = set()
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==|@)", line.strip())
        if m:
            names.add(norm(m.group(1)))
    return names


def main():
    bazel_text = "".join(p.read_text() for p in BAZEL_FILES if p.exists())
    leaks, warns = [], []
    for arch, lock_name in CFG.items():
        b = bucket(arch)
        lock = DEPS / lock_name
        if not b or not lock.exists():
            continue
        in_lock = lock_names(lock)
        for name, ver in b.items():
            if name in in_lock:
                leaks.append(f"{arch}: {name} 同时在 bazel 桶与 {lock_name}（双供给）")
            cands = {ver, ver.replace("+", "%2B"), urllib.parse.quote(ver)}
            if not any(c in bazel_text for c in cands):
                warns.append(f"{arch}: {name}=={ver} 版本串未在 bazel 侧文件中发现（源构建包属预期）")
    for w in warns:
        print(f"WARN {w}")
    if leaks:
        for l in leaks:
            print(f"FAIL {l}")
        print("修复：从对应 requirements_<arch>.txt 移除该包（桶是唯一真源）后重编 lock")
        sys.exit(1)
    print(f"OK: {len(CFG)} arch 桶∩lock=∅（无双供给；{len(warns)} 条版本 WARN 见上）")


if __name__ == "__main__":
    main()

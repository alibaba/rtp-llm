#!/usr/bin/env python3
"""门禁：rtp_llm/BUILD 里 requirement() 登记的每个包，必须能在「所有 lock ∪ 所有 bazel 桶」里找到。

治痛点 10②：否则 `requirement("拼错名")` 或引用了谁都没提供的包，要等 `bazel build` 到某 arch
才报 `no such target`（构建期、per-arch 的晚暴露）。此检查纯文本、秒级、无 bazel 依赖，提前拦下。
（联合判定，不做 per-arch 精确匹配，避免对 rocm-only 等 arch 专属包误报。）
用法: python3 deps/check_requirement_subset.py [repo_root]
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
DEPS = ROOT / "deps"
BUILD = ROOT / "rtp_llm" / "BUILD"


def norm(n):
    return n.lower().replace("_", "-").replace(".", "-")


def build_requirement_names(text):
    names = set()
    for m in re.finditer(r'requirement\(\s*\[([^\]]*)\]', text, re.S):
        names |= {norm(x) for x in re.findall(r'"([^"]+)"', m.group(1))}
    for m in re.finditer(r'requirement\(\s*"([^"]+)"', text):
        names.add(norm(m.group(1)))
    return names


def names_in(path, pat):
    out = set()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            out.add(norm(m.group(1)))
    return out


def main():
    if not BUILD.exists():
        print(f"SKIP: 未找到 {BUILD}")
        return
    req = build_requirement_names(BUILD.read_text())

    provided = set()
    lock_pat = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==|@)")
    name_pat = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
    for lock in DEPS.glob("requirements_lock_*.txt"):
        provided |= names_in(lock, lock_pat)
    for bp in DEPS.glob("bazel_provided_*.txt"):
        provided |= names_in(bp, name_pat)

    missing = sorted(req - provided)
    if missing:
        for m in missing:
            print(f"FAIL requirement(\"{m}\")：任何 lock 与 bazel 桶都未提供（拼错名 / 漏加依赖）")
        print("修复：加进对应 requirements_*.txt 后重编 lock，或加进 bazel_provided_<arch>.txt")
        sys.exit(1)
    print(f"OK: BUILD 的 {len(req)} 个 requirement() 均在 lock∪bazel桶 中")


if __name__ == "__main__":
    main()

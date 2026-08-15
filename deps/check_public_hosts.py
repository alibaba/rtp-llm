#!/usr/bin/env python3
"""The open-source dependency declaration surface must not contain internal/closed-source hosts.

Only scans the "dependency and build configuration" leak channel
(requirements/lock/*.bzl/bazelrc/MODULE.bazel), not the whole repo -- leaks all happen
here, and scanning the whole repo would drown in docs and comments. Criterion: a machine
without internal DNS can clone the open-source repo and build with `--config=cpu`.

Usage: check_public_hosts.py <repo root>
"""

import os
import re
import sys

# Internal/closed-source hosts and prefixes. rtp-opensource is a publicly readable
# open-source bucket and is not listed here.
DENY = [
    (r"rtp-maga", "PPU closed-source bucket"),
    (r"ppu_sdk", "PPU private index prefix"),
    (r"artlab\.alibaba-inc\.com", "internal artifactory"),
    (r"[A-Za-z0-9.-]+-internal\.aliyuncs\.com", "OSS internal endpoint"),
    (r"search-cicd[A-Za-z0-9.-]*", "internal CI mirror bucket"),
    # With only -internal.aliyuncs.com listed, the public-endpoint spelling of the same
    # internal region (*-zmf.aliyuncs.com) and the bare bucket name oss://search-ad slip
    # through as a whole class, so all three spellings are listed.
    (r"oss://search-ad", "internal zmf bucket (oss:// form)"),
    (r"search-ad\.oss[A-Za-z0-9.-]*", "internal zmf bucket (https form)"),
    (r"[A-Za-z0-9.-]+-zmf\.aliyuncs\.com", "OSS zmf endpoint"),
    (r"com\.taobao\.[A-Za-z0-9.]+", "internal Bazel cache/downloader endpoint"),
    (r"[A-Za-z0-9.-]*\.vipserver", "internal vipserver endpoint"),
    (r"[A-Za-z0-9.-]*\.alibaba-inc\.com", "internal domain"),
]

# Scan surface relative to the repo root. Directory entries only scan these suffixes.
# .bazeliskrc is listed: it decides where the bazel binary is downloaded from, the first
# hop of "can a fresh clone start building" (the internal-side mirror is overridden via
# ENV in the dev image).
# The .json files under deps are listed: deps.json is the single declaration source of
# truth; skipping it leaves the main leak channel wide open.
FILES = [".bazelrc", ".bazeliskrc", "MODULE.bazel", "WORKSPACE", "def.bzl"]
DIRS = [
    ("deps", (".txt", ".bzl", ".sh", ".json")),
    # .py under bazel/ is listed: tool scripts executed by rules live here and can embed
    # internal endpoints too.
    ("bazel", (".bzl", ".bazelrc", ".cfg", ".py")),
    ("arch_config", (".bzl",)),
]

# One known spot [not yet] in the scan surface: package/*.Dockerfile.
# package/base_gpu_cuda12.Dockerfile still fetches 5 NVIDIA redistributables from the
# internal zmf bucket via a BuildKit secret; inclusion first requires mirroring them into
# the public bucket. Registered here rather than silently unscanned.


def fetched_part(line):
    """The part that will actually be fetched.

    In a downloader cfg, `rewrite <match> <replacement>`: the match side only says "who
    requested this URL"; the replacement side is the actual download address. An internal
    host on the left precisely means it is rewritten to a public mirror -- flagging it
    would be a false positive; only look at the right side. `allow`/`block` likewise are
    pure match expressions, skip the whole line.
    """
    head = line.split()[:1]
    if head == ["rewrite"]:
        return line.split()[-1]
    if head in (["allow"], ["block"]):
        return ""
    return line


def scan(path, rel, hits):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            text = fetched_part(line)
            for pattern, why in DENY:
                found = re.search(pattern, text)
                if found:
                    hits.append((rel, lineno, found.group(0), why))
                    break


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    hits, scanned = [], 0
    targets = [f for f in FILES if os.path.isfile(os.path.join(root, f))]
    for name, suffixes in DIRS:
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        # Recursive: archive declarations all live in the deps/extensions/ subdirectory,
        # which a plain os.listdir would skip wholesale.
        for cur, _dirs, files in os.walk(directory):
            for entry in sorted(files):
                if entry.endswith(suffixes):
                    full = os.path.join(cur, entry)
                    targets.append(os.path.relpath(full, root))
    for rel in targets:
        scanned += 1
        scan(os.path.join(root, rel), rel, hits)
    if hits:
        print(
            f"FAIL: {len(hits)} internal hosts found on the open-source dependency declaration surface (scanned {scanned} files):"
        )
        for rel, lineno, text, why in hits:
            print(f"  {rel}:{lineno}: {text}  <- {why}")
        print(
            "  Fix: move the declaration into internal_source/ (.internal_bazelrc / internal_source/deps/); the open-source side keeps only publicly reachable entries"
        )
        return 1
    print(
        f"OK: {scanned} open-source dependency declaration files have no internal host references"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

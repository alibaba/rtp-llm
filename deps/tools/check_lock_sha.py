#!/usr/bin/env python3
"""Verify deps/requirements_lock_*.txt still resolve byte-identical wheels to
the closed-source originals.

The committed baseline (deps/tools/lock_sha_baseline.json) freezes the sha256
set of every package in every lock as of the deps unification, taken from the
internal overlay. The unification re-pointed URL hosts at public mirrors, but
the wheel bytes -- and therefore every sha256 -- must stay identical.

Usage:
  python3 deps/tools/check_lock_sha.py                          # vs baseline
  python3 deps/tools/check_lock_sha.py --internal REPO --ref R  # vs internal_source/deps@R
  python3 deps/tools/check_lock_sha.py --regen                  # after a reviewed re-resolution

Exits non-zero on any missing/extra package or sha-set difference.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")
URL_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s+@\s+(\S+)")
URL_SHA_RE = re.compile(r"#sha256=([0-9a-f]{64})")


def parse_lock(text):
    """Map package name -> set of pinned sha256 digests.

    Handles both lock entry forms: ``name==version`` with ``--hash``
    continuation lines, and ``name @ url`` with the digest either in the URL
    fragment or on the continuation lines."""
    entries, cur = {}, None
    for line in text.splitlines():
        if line.startswith(" ") or line.startswith("#") or not line.strip():
            if cur is not None:
                entries[cur].update(HASH_RE.findall(line))
            continue
        m = PIN_RE.match(line)
        if m:
            cur = m.group(1)
            entries.setdefault(cur, set())
            entries[cur].update(HASH_RE.findall(line))
            continue
        m = URL_RE.match(line)
        if m:
            name, url = m.group(1), m.group(2)
            entries.setdefault(name, set())
            sha = URL_SHA_RE.search(url)
            if sha:
                entries[name].add(sha.group(1))
                cur = None
            else:
                cur = name  # digest arrives on the --hash continuation lines
            continue
        cur = None
    return entries


def git_show(repo, ref, path):
    r = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (ref, path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("error: git show %s:%s in %s failed: %s" % (ref, path, repo, r.stderr.strip()))
    return r.stdout


def compare(lock, got, want, problems):
    for pkg in sorted(set(got) | set(want)):
        g, w = got.get(pkg), want.get(pkg)
        if g is None:
            problems.append("%s: %s missing from the current locks" % (lock, pkg))
        elif w is None:
            problems.append("%s: %s not in the reference" % (lock, pkg))
        elif g != w:
            problems.append("%s: %s sha mismatch (current %d, reference %d; current-only %s, "
                            "reference-only %s)" % (lock, pkg, len(g), len(w),
                            [s[:12] for s in sorted(g - w)][:3],
                            [s[:12] for s in sorted(w - g)][:3]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deps", default="deps", help="deps dir holding requirements_lock_*.txt")
    ap.add_argument("--baseline", default=None, help="baseline json (default <deps>/tools/lock_sha_baseline.json)")
    ap.add_argument("--regen", action="store_true", help="rewrite the baseline from the CURRENT locks")
    ap.add_argument("--internal", metavar="REPO", default=None,
                    help="verify against internal_source/deps in REPO (requires --ref)")
    ap.add_argument("--ref", default=None, help="git ref for --internal")
    args = ap.parse_args()
    deps = Path(args.deps)
    baseline_path = Path(args.baseline) if args.baseline else deps / "tools" / "lock_sha_baseline.json"
    locks = sorted(p.name for p in deps.glob("requirements_lock_*.txt"))
    if not locks:
        sys.exit("error: no requirements_lock_*.txt under %s" % deps)
    if args.internal and not args.ref:
        sys.exit("error: --internal requires --ref")

    current = {lf: parse_lock((deps / lf).read_text()) for lf in locks}

    if args.regen:
        data = {"_meta": {"note": "regenerated from the current locks "
                                  "(only after a reviewed re-resolution)"}}
        for lf in locks:
            data[lf] = {k: sorted(v) for k, v in current[lf].items()}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print("baseline rewritten: %s (%d locks)" % (baseline_path, len(locks)))
        return

    problems = []
    if args.internal:
        r = subprocess.run(["git", "-C", args.internal, "ls-tree", "--name-only", args.ref,
                            "internal_source/deps/"], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("error: git ls-tree %s in %s failed: %s"
                     % (args.ref, args.internal, r.stderr.strip()))
        internal_locks = set(x.strip() for x in r.stdout.splitlines()
                             if re.match(r"requirements_lock_.*\.txt$", x.strip()))
        for lf in locks:
            ref = parse_lock(git_show(args.internal, args.ref, "internal_source/deps/" + lf))
            compare(lf, current[lf], ref, problems)
        for lf in sorted(internal_locks - set(locks)):
            problems.append("%s: lock file missing (present in the internal tree)" % lf)
    else:
        if not baseline_path.exists():
            sys.exit("error: baseline not found: %s" % baseline_path)
        data = json.loads(baseline_path.read_text())
        for lf in locks:
            if lf not in data:
                problems.append("%s: no baseline entry" % lf)
                continue
            compare(lf, current[lf], {k: set(v) for k, v in data[lf].items()}, problems)
        for lf in sorted(k for k in data if k != "_meta" and k not in current):
            problems.append("%s: lock file missing (present in baseline)" % lf)

    if problems:
        print("MISMATCH (%d):" % len(problems))
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    print("OK: %d locks, %d packages, every sha256 set matches the reference"
          % (len(locks), sum(len(v) for v in current.values())))


if __name__ == "__main__":
    main()

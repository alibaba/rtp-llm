#!/usr/bin/env python3
"""Same-version artifact-swap detection. Two sides of one proposition, sharing one lock parse.

`name==version` cannot tell which build it is, so two complementary criteria:
* Cross-lock: hash sets of the same `name==version` across locks on the same platform must
  intersect (an empty intersection means the same version was overwritten on the private
  OSS). Wheel hashes naturally differ across CPU platforms, hence same-platform groups.
* Against declarations: for packages with `sha256` in deps.json, the lock's --hash set must
  equal it exactly -- covers the cross-lock blind spot of packages appearing in only one lock.

Usage: python3 deps/check_lock_artifacts.py [deps_dir]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pkgname import PIN_RE, norm

DEPS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_lock(path):
    """{normalized name: (version, {hashes})} -- the single parse shared by both criteria."""
    out = {}
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = PIN_RE.match(s)
        if m:
            cur = norm(m.group(1))
            out[cur] = (m.group(2), set())
        for h in HASH_RE.findall(s):
            if cur:
                out[cur][1].add(h)
    return out


def platform_groups(manifest):
    """lock file name -> platform group (first segment of profile.platform: x86_64-... -> x86_64)."""
    return {p["lock"]: p["platform"].split("-", 1)[0] for p in manifest["profiles"]}


def declarations(manifest):
    """[(normalized name, profile, sha256)], common expanded to all profiles."""
    all_profiles = [p["name"] for p in manifest["profiles"]]
    out = []
    for pkg in manifest["packages"]:
        name = norm(pkg["name"])
        scopes = list(pkg.get("per_profile", {}).items())
        if "common" in pkg:
            scopes += [(prof, pkg["common"]) for prof in all_profiles]
        for profile, info in scopes:
            if "sha256" in info:
                out.append((name, profile, info["sha256"]))
    return out


def check_cross_lock(manifest, parsed):
    """Every pair of hash sets for the same name==version within a platform group must intersect."""
    groups = platform_groups(manifest)
    seen = defaultdict(list)  # (platform, name, version) -> [(lock name, hashes)]
    problems, total = [], 0
    for lock_name, entries in sorted(parsed.items()):
        plat = groups.get(lock_name, "unknown")
        for name, (ver, hashes) in entries.items():
            key = (plat, name, ver)
            for prev_lock, prev_hashes in seen[key]:
                if not (hashes & prev_hashes):
                    problems.append(
                        f"{name}=={ver}: hash sets of {prev_lock} and {lock_name} have no intersection"
                        " (likely the same version overwritten on OSS); fix: confirm the overwrite source,"
                        " re-upload as a new local version, then recompile all affected locks"
                    )
            if not seen[key]:
                total += 1
            seen[key].append((lock_name, hashes))
    return problems, total


def check_against_declared(manifest, parsed):
    """For packages declaring sha256, the lock must pin exactly that one artifact."""
    lock_of = {p["name"]: p["lock"] for p in manifest["profiles"]}
    problems = []
    decls = declarations(manifest)
    for name, profile, sha in decls:
        if not SHA_RE.match(sha):
            problems.append(
                f"{name}[{profile}]: sha256 is not 64 lowercase hex chars: {sha!r}"
            )
            continue
        lock_name = lock_of[profile]
        entry = parsed.get(lock_name, {}).get(name)
        if entry is None:
            problems.append(
                f"{name}[{profile}]: declares sha256 but {lock_name} does not contain this package"
            )
            continue
        version, hashes = entry
        if hashes != {sha}:
            got = ", ".join(sorted(h[:12] for h in hashes)) or "(no --hash)"
            problems.append(
                f"{name}=={version}[{profile}]: lock pins {got}, declaration expects {sha[:12]}"
                "; fix: if the bucket object was swapped, correct the artifact first then recompile that"
                " profile's lock; if the build change is intended, update sha256 in deps.json and record"
                " the rationale in _comment"
            )
    return problems, len(decls)


def main():
    manifest = json.loads((DEPS / "deps.json").read_text(encoding="utf-8"))
    locks = sorted(DEPS.glob("requirements_lock_*.txt"))
    expected = {p["lock"] for p in manifest["profiles"]}
    actual = {p.name for p in locks}
    missing = sorted(expected - actual)
    if missing:
        for lock in missing:
            print(f"FAIL: {lock} missing, cannot complete artifact validation")
        sys.exit(1)
    if not locks:
        print("FAIL: no lock files found")
        sys.exit(1)
    parsed = {p.name: parse_lock(p) for p in locks}

    cross, pins = check_cross_lock(manifest, parsed)
    declared, decl_count = check_against_declared(manifest, parsed)

    problems = cross + declared
    if problems:
        for p in problems:
            print("FAIL " + p)
        sys.exit(1)
    print(
        "OK: %d locks, %d (platform,name,version) tuples without hash conflicts; %d sha256 declarations match the locks"
        % (len(locks), pins, decl_count)
    )


if __name__ == "__main__":
    main()

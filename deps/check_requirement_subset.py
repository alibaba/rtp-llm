#!/usr/bin/env python3
"""Validate requirement() references across the repo per profile.

Scan surface = requirement([...]) and requirement(var) (var being a string list in the
same file) across all BUILD and *.bzl files. By default a package must appear in all 6
profile locks; profile-specific packages must be registered in deps.json
exceptions[].exists_in, and the registered coverage set is cross-checked both ways against
actual lock coverage: actually missing = lost package FAIL, actually extra = stale
exception FAIL.

Usage: python3 deps/check_requirement_subset.py [repo_root]
"""
import json
import re
import sys
from pathlib import Path

from pkgname import norm

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
DEPS = Path(__file__).resolve().parent

# The single source of truth is deps.json: profiles give {profile name: lock};
# exceptions come from the manifest's exceptions[].exists_in
# (exists_in/allowed_refs).
with open(DEPS / "deps.json", encoding="utf-8") as _fh:
    MANIFEST = json.load(_fh)

PROFILES = {p["name"]: p["lock"] for p in MANIFEST["profiles"]}


def _var_list_names(text, var, seen=None):
    """Expand string names of a module-level ``var = ["a", "b"] + other`` (incl. nested vars, cycle-safe).

    requirement() is actually written as ``requirement([...] + deep + flashinfer_with_cache + ...)``:
    capturing only inline lists misses names inside variables, and such misses only blow
    up at CI build time on the corresponding profile.
    """
    seen = seen or set()
    if var in seen:
        return set()
    seen.add(var)
    m = re.search(rf"^{re.escape(var)}\s*=\s*(\[[^\]]*\][^\n]*)", text, re.M)
    if not m:
        return set()
    expr = m.group(1)
    out = {norm(x) for x in re.findall(r'"([^"]+)"', expr)}
    for nested in re.findall(r"\+\s*([A-Za-z_][A-Za-z_0-9]*)", expr):
        out |= _var_list_names(text, nested, seen)
    return out


def _call_text(text, start):
    """Take the call text from the left paren of ``requirement(`` to the matching right paren."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def requirement_names(root):
    """requirement() references across the repo → {norm_name: set(referencing file relpaths)}."""
    names = {}
    files = list(root.rglob("BUILD")) + list(root.rglob("*.bzl"))
    for p in files:
        sp = str(p)
        if any(
            seg in sp
            for seg in (
                ".git",
                "bazel-",
                ".bazel741_ob",
                "/external/",
                "arch_select.bzl",
            )
        ):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "requirement(" not in text:
            continue
        found = set()
        for m in re.finditer(r"requirement\(", text):
            call = _call_text(text, m.end() - 1)
            found |= {norm(x) for x in re.findall(r'"([^"]+)"', call)}
            for var in re.findall(r"[(+]\s*([A-Za-z_][A-Za-z_0-9]*)\s*(?=[+)])", call):
                if var != "names":
                    found |= _var_list_names(text, var)
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = sp
        for n in found:
            names.setdefault(n, set()).add(rel)
    return names


def lock_names(path):
    out = set()
    pat = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==|@)")
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            out.add(norm(m.group(1)))
    return out


def load_exceptions():
    """Derive {norm(name): (expected_profiles, allowed_sites)} from deps.json exceptions.

    exists_in gives the expected set directly; allowed_refs restricts the reference sites
    (empty = unrestricted)."""
    exc = {}
    for ex in MANIFEST.get("exceptions", []):
        exc[norm(ex["name"])] = (
            set(ex.get("exists_in", [])),
            set(ex.get("allowed_refs", [])),
        )
    return exc


def main():
    req = requirement_names(ROOT)
    provided = {
        a: lock_names(DEPS / f) for a, f in PROFILES.items() if (DEPS / f).exists()
    }
    exceptions = load_exceptions()
    problems = []
    for name, sites in sorted(req.items()):
        have = {a for a, s in provided.items() if name in s}
        want, allowed_sites = exceptions.get(name, (set(PROFILES), set()))
        unknown_prof = want - set(PROFILES)
        if unknown_prof:
            problems.append(
                f"{name}: exception registers unknown profiles {sorted(unknown_prof)}"
            )
            continue
        missing = want - have
        extra = have - want
        where = ",".join(sorted(sites))
        if not have:
            problems.append(
                f"{name} (referenced by {where}): not provided by any lock (typo/missing dependency)"
            )
            continue
        if missing:
            problems.append(
                f"{name} (referenced by {where}): should cover {sorted(want)} but is missing {sorted(missing)} -- "
                "add pin and relock, or fix exceptions[].exists_in in deps.json"
            )
        if extra and name in exceptions:
            problems.append(
                f"{name}: exception registers only {sorted(want)}, but it also exists in {sorted(extra)} -- stale exception, update the registration"
            )
        if name in exceptions and allowed_sites:
            rogue = sites - allowed_sites
            if rogue:
                problems.append(
                    f"{name}: profile-specific package referenced at unregistered sites {sorted(rogue)} -- "
                    "that consumption site lacks a config guard and will blow up at build time on the other profiles; confirm the guard then register it in the exception file"
                )
    if problems:
        print("FAIL: per-profile requirement() validation failed:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    n_exc = sum(1 for n in req if n in exceptions)
    print(
        f"OK: {len(req)} requirement() names across the repo passed per-profile validation"
        f" (all-profile {len(req) - n_exc} / registered exceptions {n_exc}, reference sites restricted)"
    )


if __name__ == "__main__":
    main()

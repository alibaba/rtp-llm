#!/usr/bin/env python3
"""Every profile's consumption sites must not request packages absent in that profile.

Absence facts = the complement of deps.json exceptions[].exists_in;
private overlay absence facts are loaded when this is an internal checkout; consumption facts
are branch contents of select({"@//:<config>": [...]}) in all BUILD files, with config → profile
mapping following .bazelrc --config semantics. Any (site, profile, package) hitting an absence
fails offline, instead of only blowing up when that profile reaches analysis phase.
"""
import json
import os
import re
import sys

from pkgname import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPER = os.path.dirname(ROOT)

# config_setting → the profiles where it is true. Alias configs (using_cuda/using_cuda12)
# cover multiple profiles, which is exactly how guards get missed.
CONFIG_PROFILES = {
    "using_cpu": ["cpu"],
    "using_arm": ["arm"],
    "using_rocm": ["rocm"],
    "cuda_pre_12_9": ["cuda12_6"],
    "using_cuda12_9_x86": ["cuda12_9"],
    "using_cuda12_arm": ["cuda12_9_arm"],
    "using_cuda12_9": ["cuda12_9", "cuda12_9_arm"],
    "using_cu12_9_or_13_x86": ["cuda12_9", "cuda13"],
    "using_cuda12": ["cuda12_6", "cuda12_9", "cuda12_9_arm"],
    "using_cuda": ["cuda12_6", "cuda12_9", "cuda12_9_arm"],
}

# config_settings that select on a feature toggle rather than a profile, so no profile can be
# derived from them. Listed explicitly: an unlisted, unmapped config means someone added a
# profile guard without registering it, which the scan must refuse rather than skip.
NON_PROFILE_CONFIGS = frozenset(["using_remote_kv_cache", "use_accl_ep", "xft_use_icx"])

# Positive test tag → profiles where that test will definitely be built (taken from the
# --test_tag_filters in .aoneci/main.yaml: the gb200 job uses SM100_ARM, the amd job uses
# rocm). Unconditional deps of tests carrying such tags must exist in that profile.
# The smoke suite's gpu_type values (MI308X*/PPU-ZW*) are not listed: they only become
# tags inside the smoke_test macro, which also adds manual, so those targets never enter
# the wildcard surface (CI runs them by explicit label).
TAG_PROFILES = {
    "SM100_ARM": ["cuda12_9_arm"],
    "rocm": ["rocm"],
}


def load_absent(manifest, private=None):
    profiles = [p["name"] for p in manifest["profiles"]]
    absent = {}
    for exc in manifest.get("exceptions", []):
        key = norm(exc.get("name", ""))
        if not key:
            continue
        exists = {norm(x) for x in exc.get("exists_in", [])}
        absent[key] = {p for p in profiles if norm(p) not in exists}
    private_profiles = (private or {}).get("profiles", {})
    if isinstance(private_profiles, dict):
        for profile, info in private_profiles.items():
            if not isinstance(info, dict) or not isinstance(info.get("absent"), list):
                continue
            for name in info["absent"]:
                absent.setdefault(norm(name), set()).add(profile)
    return absent


def load_private_overlay():
    """Load the internal-only overlay when present; public clones simply have none."""
    path = os.path.join(SUPER, "internal_source", "deps", "ppu.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def private_profile_problems(private):
    """Check private profile locks exist without duplicating their package contents."""
    profiles = (private or {}).get("profiles", {})
    if not isinstance(profiles, dict):
        return ["internal_source/deps/ppu.json profiles must be an object"]
    internal_root = os.path.abspath(os.path.join(SUPER, "internal_source"))
    problems = []
    for profile, info in profiles.items():
        settings = info.get("config_settings") if isinstance(info, dict) else None
        if not isinstance(settings, list) or not settings:
            problems.append("private profile %s is missing config_settings" % profile)
        lock = info.get("lock") if isinstance(info, dict) else None
        if not isinstance(lock, str) or not lock:
            problems.append("private profile %s is missing lock" % profile)
            continue
        path = os.path.abspath(os.path.join(SUPER, lock))
        try:
            inside = os.path.commonpath([internal_root, path]) == internal_root
        except ValueError:
            inside = False
        if not inside:
            problems.append(
                "private profile %s lock escapes internal_source: %s" % (profile, lock)
            )
        elif not os.path.isfile(path):
            problems.append("private profile %s lock is missing: %s" % (profile, lock))
    return problems


def private_config_profiles(private):
    """Private profile → the config_settings it declares.

    Guessing names by concatenation cannot distinguish cuda13 from cuda13_arm, so the
    overlay states them; a profile without them would go unchecked, which
    private_profile_problems reports. Only the most-specific setting is listed: broader
    aliases that are also true (using_cuda12_arm under cuda13_arm) lose select resolution
    to the specialized branch.
    """
    profiles = (private or {}).get("profiles", {})
    if not isinstance(profiles, dict):
        return {}
    out = {}
    for profile, info in profiles.items():
        settings = info.get("config_settings") if isinstance(info, dict) else None
        for config in settings or []:
            out.setdefault(config, []).append(profile)
    return out


def pkg_path(base):
    """Directory → workspace package path. internal_source hangs under the workspace root
    as a symlink; relpath yields ``../internal_source/...`` while labels are written as
    ``//internal_source/...``.
    """
    rel = os.path.relpath(base, ROOT).replace(os.sep, "/")
    while rel.startswith("../"):
        rel = rel[3:]
    return rel


def requirement_packages():
    """Packages across the repo that call requirement([...]) (posix paths relative to the workspace root).

    Only these packages contain wheel shims generated by requirement(); same-named targets
    elsewhere are not shims. ``//rtp_llm/models_py/triton_kernels:triton_kernels`` is a
    local kernel package sharing the name of the ``triton-kernels`` wheel -- taking only
    the label's last segment would false-positive on it.
    """
    pkgs = set()
    for tree in (os.path.join(ROOT, "rtp_llm"), os.path.join(SUPER, "internal_source")):
        for base, dirs, files in os.walk(tree):
            dirs[:] = [d for d in dirs if not d.startswith("bazel-") and d != ".git"]
            if "BUILD" not in files:
                continue
            path = os.path.join(base, "BUILD")
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if re.search(r"^requirement\(", text, re.M):
                pkgs.add(pkg_path(base))
    return pkgs


def shim_name(label, own_pkg, req_pkgs):
    """Dependency label → wheel shim package name; returns None for non-shims.

    Must return the bare target name, not the whole label, or norm() can never match an
    absent key and the branch goes silently unchecked (false green).
    """
    s = label.strip()
    absolute = "//" in s
    tail = s.split("//", 1)[1] if absolute else s
    if ":" in tail:
        pkg, name = tail.rsplit(":", 1)
    elif absolute or "/" in tail:
        return None  # //a/b implicit target or a path, not a shim
    else:
        pkg, name = "", tail
    if not absolute:
        pkg = own_pkg
    return name if pkg in req_pkgs else None


def list_vars(text):
    """Module-level ``name = ["a", "b"]`` → {name: [a, b]}."""
    out = {}
    for m in re.finditer(r"^(\w+)\s*=\s*\[([^\]]*)\]", text, re.M):
        out[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return out


def package_names(body, variables, own_pkg, req_pkgs):
    """select branch body → set of requested shim package names.

    A branch body may be an inline list, a variable name, or an expression chaining them
    with ``+``; all three must be expanded, otherwise variables inside expressions are
    skipped wholesale (false green).
    """
    names = []
    for part in re.split(r"\s*\+\s*", body.strip()):
        part = part.strip()
        if not part:
            continue
        raw = (
            re.findall(r'"([^"]+)"', part)
            if part.startswith("[")
            else variables.get(part, [])
        )
        names += [n for n in (shim_name(x, own_pkg, req_pkgs) for x in raw) if n]
    return names


def unconditional_deps(text, variables, own_pkg, req_pkgs):
    """Entries in deps = [...] that are not inside a select -- effective for all profiles.

    Absent packages must live in profile-guarded select branches; an unconditional absent
    dep only fails at analysis phase of the affected profile.
    """
    # Carve out all select({...}) sections by brace balancing: non-greedy matching up to the
    # first "})" cuts wrong when a branch body itself contains "}", treating the default
    # branch as unconditional (false positive).
    stripped = []
    i = 0
    while True:
        m = re.search(r"select\(\{", text[i:])
        if not m:
            stripped.append(text[i:])
            break
        start = i + m.start()
        stripped.append(text[i:start])
        depth = 0
        j = i + m.end() - 1  # points at '{'
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1
    stripped = "".join(stripped)
    # Commented-out entries are not deps.
    stripped = re.sub(r"#[^\n]*", "", stripped)
    names = []
    # All three deps shapes must be inspected: literal list, "list + variable" expression,
    # bare variable. select sections were already carved out above, so variables remaining
    # here are unconditionally effective; function calls like internal_deps() are skipped
    # (their bodies are selects themselves, covered by the branch check).
    for m in re.finditer(
        r"deps\s*=\s*((?:\[[^\]]*\]|[\w.]+(?:\(\))?)"
        r"(?:\s*\+\s*(?:\[[^\]]*\]|[\w.]+(?:\(\))?))*)",
        stripped,
        re.S,
    ):
        for part in re.split(r"\s*\+\s*", m.group(1)):
            part = part.strip()
            if part.startswith("["):
                raw = re.findall(r'"([^"]+)"', part)
            elif part and not part.endswith("()"):
                raw = variables.get(part, [])
            else:
                raw = []
            names += [n for n in (shim_name(x, own_pkg, req_pkgs) for x in raw) if n]
    return names


def rule_blocks(text):
    """Top-level rule calls → [block texts]. Split on line-start ``name(``; good enough without a starlark parser."""
    starts = [m.start() for m in re.finditer(r"^\w+\s*\(", text, re.M)]
    for i, s in enumerate(starts):
        yield text[s : starts[i + 1] if i + 1 < len(starts) else len(text)]


def tag_profiles(block):
    """A test target's tags → the set of profiles where it will definitely be built.

    Only jobs with a [positive] tag filter give this certainty: the gb200 job uses
    SM100_ARM, the amd job uses rocm. The x86 CUDA jobs only have negative filters and
    are not narrowed by this. Targets tagged manual never enter the wildcard surface.
    """
    m = re.search(r"tags\s*=\s*\[([^\]]*)\]", block)
    tags = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    if "manual" in tags:
        return set()
    out = set()
    for tag in tags:
        out |= set(TAG_PROFILES.get(tag, []))
    return out


def scan(tree, absent, problems, req_pkgs, config_profiles=None):
    config_profiles = config_profiles or CONFIG_PROFILES
    for base, dirs, files in os.walk(tree):
        dirs[:] = [d for d in dirs if not d.startswith("bazel-") and d != ".git"]
        own_pkg = pkg_path(base)
        for fname in files:
            if fname != "BUILD" and not fname.endswith("arch_select.bzl"):
                continue
            path = os.path.join(base, fname)
            # Targets under test/ are mostly excluded per profile by --test_tag_filters,
            # so they cannot uniformly be judged as "must exist in every profile" (11
            # CUDA-only tests would false-red); but tests with positive tags are exactly
            # the ones certain to build on that profile, and their unconditional deps
            # must be checked against it.
            is_test = os.sep + "test" + os.sep in path + os.sep
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            variables = list_vars(text)
            rel = os.path.relpath(path, SUPER)
            if is_test:
                for block in rule_blocks(text):
                    built_in = tag_profiles(block)
                    if not built_in:
                        continue
                    for pkg in unconditional_deps(block, variables, own_pkg, req_pkgs):
                        miss = built_in & absent.get(norm(pkg), set())
                        if miss:
                            name = re.search(r'name\s*=\s*"([^"]+)"', block)
                            problems.append(
                                "%s: test %s with positive tags builds on profile %s, but depends on"
                                " %s which is absent in that profile"
                                % (
                                    rel,
                                    name.group(1) if name else "?",
                                    ", ".join(sorted(miss)),
                                    pkg,
                                )
                            )
            else:
                for pkg in unconditional_deps(text, variables, own_pkg, req_pkgs):
                    miss = absent.get(norm(pkg), set())
                    if miss:
                        problems.append(
                            "%s: %s is an unconditional dep, but it is absent in profile %s -- move it into a "
                            "profile-guarded select branch"
                            % (rel, pkg, ", ".join(sorted(miss)))
                        )
            # All three branch-body shapes must be covered: inline list, single variable,
            # and "a + b + [...]" expressions. Missing the expression shape gives false green.
            for m in re.finditer(
                r'"@[^" ]*//[^":]*:(\w+)":\s*((?:\[[^\]]*\]|[\w.]+)'
                r"(?:\s*\+\s*(?:\[[^\]]*\]|[\w.]+))*)",
                text,
            ):
                config, body = m.group(1), m.group(2)
                if config not in config_profiles:
                    if config not in NON_PROFILE_CONFIGS:
                        problems.append(
                            "%s: config %s is not registered as a profile guard, so its branch "
                            "goes unchecked -- map it in CONFIG_PROFILES / a profile's "
                            "config_settings, or list it in NON_PROFILE_CONFIGS"
                            % (os.path.relpath(path, SUPER), config)
                        )
                    continue
                for profile in config_profiles[config]:
                    for pkg in package_names(body, variables, own_pkg, req_pkgs):
                        if profile in absent.get(norm(pkg), set()):
                            rel = os.path.relpath(path, SUPER)
                            problems.append(
                                "%s: config %s covers profile %s, but requests %s which is absent "
                                "in that profile -- drop it from this branch per the lock facts, or add a pin and relock"
                                % (rel, config, profile, pkg)
                            )


def main():
    with open(os.path.join(ROOT, "deps", "deps.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    try:
        private = load_private_overlay()
    except (OSError, ValueError) as exc:
        print("FAIL: internal_source/deps/ppu.json is unreadable: %s" % exc)
        sys.exit(1)
    absent = load_absent(manifest, private)
    config_profiles = dict(CONFIG_PROFILES)
    for config, profiles in private_config_profiles(private).items():
        config_profiles.setdefault(config, []).extend(profiles)
    req_pkgs = requirement_packages()
    problems = private_profile_problems(private)
    scan(os.path.join(ROOT, "rtp_llm"), absent, problems, req_pkgs, config_profiles)
    internal = os.path.join(SUPER, "internal_source", "bazel")
    if os.path.isdir(internal):
        scan(internal, absent, problems, req_pkgs, config_profiles)
    if problems:
        print("FAIL: consumption sites request packages absent in their profile:")
        for p in sorted(set(problems)):
            print("  " + p)
        sys.exit(1)
    print("OK: packages in all select branches are available in their profile")


if __name__ == "__main__":
    main()

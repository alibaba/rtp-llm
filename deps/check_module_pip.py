#!/usr/bin/env python3
"""The 6 pip.parse blocks in MODULE.bazel must be a faithful projection of deps.json profiles.

MODULE.bazel cannot load(), so the pip configuration is hand-written while deps.json also
holds the profile matrix; the duplication can only be validated. Most critical is the index
override set: rules_python 0.33.2 takes, per package, only "the first readable page" and
never merges indexes, so packages supplied by our own index must each be overridden back to
the flavor view -- a missing entry does not turn the build red, it silently reroutes to
aliyun and falls back to pip, so it must be pinned offline. The "supplied by own index"
criterion: explicit source=oss_flavor, or inferred from a spec carrying a PEP 440 local
version; plus cc_view, wheel.arch_names, and local-version platform_pins entries.

Usage: check_module_pip.py <repo root>
"""

import ast
import json
import os
import re
import sys


def _pip_parse_blocks(text):
    """Cut out each top-level pip.parse(...) call by brace balancing, then take literal kwargs via ast."""
    blocks = []
    for m in re.finditer(r"\bpip\.parse\(", text):
        depth = 0
        for j in range(m.end() - 1, len(text)):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(text[m.start() : j + 1])
                    break
    out = []
    for b in blocks:
        call = ast.parse(b.replace("pip.parse(", "f(", 1), mode="eval").body
        kw = {}
        for k in call.keywords:
            if k.arg is None:
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except ValueError:
                kw[k.arg] = None  # non-literal: not consumed by this check
        out.append(kw)
    return out


def _has_local_version(spec):
    """PEP 440 local version segment. '@<URL>' direct links do not count (they carry deterministic bytes)."""
    return "+" in (spec or "") and not (spec or "").lstrip().startswith("@")


def _supplied_by_own_index(manifest, profile):
    """Package names supplied by our own index for this profile -> criterion (to explain the source in errors)."""
    out = {}
    for pkg in manifest["packages"]:
        name = pkg["name"]
        for entry in (pkg.get("common"), pkg.get("per_profile", {}).get(profile)):
            if not entry:
                continue
            source = entry.get("source")
            if source == "oss_flavor":
                out[name] = "source=oss_flavor"
            elif source is None and _has_local_version(entry.get("spec")):
                out.setdefault(name, "spec has local version")
        if "cc_view" in pkg.get("per_profile", {}).get(profile, {}):
            out.setdefault(name, "cc_view")
    for name in manifest["wheel"]["arch_names"].get(profile, []):
        out.setdefault(name, "wheel.arch_names")
    for pin in manifest["wheel"]["platform_pins"].get(profile, []):
        if _has_local_version(pin):
            out.setdefault(re.split(r"[=<>!~@]", pin)[0].strip(), "platform_pins")
    return out


def check(root):
    manifest_path = os.path.join(root, "deps", "deps.json")
    module_path = os.path.join(root, "MODULE.bazel")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(module_path, encoding="utf-8") as fh:
        blocks = _pip_parse_blocks(fh.read())

    by_hub = {p["hub"]: p for p in manifest["profiles"]}
    problems = []

    seen = [b.get("hub_name") for b in blocks]
    if sorted(x for x in seen if x) != sorted(by_hub):
        problems.append(
            f"pip.parse hub set differs from deps.json profiles: MODULE.bazel={sorted(x for x in seen if x)} "
            f"deps.json={sorted(by_hub)}"
        )

    index_urls, interpreters = set(), set()
    for kw in blocks:
        hub = kw.get("hub_name")
        profile = by_hub.get(hub)
        if profile is None:
            continue
        name, view = profile["name"], profile["index_view"]

        want_lock = "//deps:" + profile["lock"]
        if kw.get("requirements_lock") != want_lock:
            problems.append(
                f"[{name}] requirements_lock={kw.get('requirements_lock')!r}, "
                f"deps.json profiles.lock requires {want_lock!r}"
            )

        extras = kw.get("experimental_extra_index_urls") or []
        if len(extras) != 1 or not extras[0].rstrip("/").endswith("/" + view):
            problems.append(
                f"[{name}] experimental_extra_index_urls={extras}, "
                f"should be exactly one entry pointing at this profile's index_view {view!r}"
            )
        flavor_view = extras[0] if len(extras) == 1 else None

        if kw.get("python_version") != manifest["python"]["version"]:
            problems.append(
                f"[{name}] python_version={kw.get('python_version')!r}, "
                f"deps.json python.version={manifest['python']['version']!r}"
            )
        index_urls.add(kw.get("experimental_index_url"))
        interpreters.add(kw.get("python_interpreter"))

        overrides = kw.get("experimental_index_url_overrides") or {}
        wrong_view = {k: v for k, v in overrides.items() if v != flavor_view}
        if flavor_view and wrong_view:
            problems.append(
                f"[{name}] these overrides do not point at this profile's flavor view {flavor_view!r}"
                f" (would fetch artifacts from another arch): {wrong_view}"
            )

        supplied = _supplied_by_own_index(manifest, name)
        unrecorded = sorted(set(overrides) - set(supplied))
        unprojected = sorted(set(supplied) - set(overrides))
        if unrecorded:
            problems.append(
                f"[{name}] overrides declare self-supply, but deps.json does not record it: {unrecorded}"
                f' -- add source="oss_flavor", otherwise nobody knows why these packages bypass aliyun'
            )
        if unprojected:
            problems.append(
                f"[{name}] deps.json says self-supplied, but the override is missing: "
                f"{[(n, supplied[n]) for n in unprojected]}"
                f" -- under 0.33.2 this silently reroutes to aliyun, then falls back to pip once the lock's hash is not found"
            )

    if len(index_urls) > 1:
        problems.append(
            f"experimental_index_url differs across the 6 hubs: {sorted(index_urls)}"
            " (all hubs must resolve from the same primary index)"
        )
    if len(interpreters) > 1:
        problems.append(
            f"python_interpreter differs across the 6 hubs: {sorted(interpreters)}"
        )
    return problems


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ".."
    if not os.path.exists(os.path.join(root, "MODULE.bazel")):
        print(
            "FAIL: MODULE.bazel does not exist -- a Bzlmod-only repo cannot skip the pip projection check"
        )
        sys.exit(1)
    problems = check(root)
    if problems:
        print("FAIL: MODULE.bazel pip.parse drifted from deps.json profiles:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(
        "OK: the 6 pip.parse blocks match deps.json profiles (hub/lock/index/self-supply set)"
    )


if __name__ == "__main__":
    main()

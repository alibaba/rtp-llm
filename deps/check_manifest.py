#!/usr/bin/env python3
"""Validate self-consistency and derivation consistency of the single dependency manifest deps/deps.json.

Requires jsonschema: schema validation, semantic self-consistency (exceptions vs composed sets,
provable wheel.exclude entries, arch_names resolvable in locks, platform_pins/groups shape),
and a read-only comparison of the generated deps/absent_map.bzl against the manifest
derivation. Any mismatch prints a precise diff and exits non-zero.
Usage: check_manifest.py [deps dir]   default = directory of this script
"""

import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# When loaded by importlib from a [file path] (this is how internal-source tests load it),
# sys.path does not contain deps/, so the sibling pkgname cannot be imported; direct
# execution and py_binary never had this problem.
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from compose import manifest_common, manifest_per_profile
from pkgname import norm

# Fixed profile order. **This constant is an intentional second hand-written copy, not a
# missed update**: it is a tripwire pinning the manifest -- `prof_names != PROFILES` below
# compares the manifest's list and order verbatim, so silently adding/removing/reordering a
# public profile requires changing this too. Deriving it from the manifest would make the
# assertion always-true. The other checkers all derive from `manifest["profiles"]` --
# they are consumers, not the judge.
PROFILES = ["cpu", "arm", "cuda12_6", "cuda12_9", "cuda12_9_arm", "rocm"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def names_in_file(path):
    """Set of normalized package names in a lock / name-list file (for provided-by/arch evidence checks)."""
    names = set()
    if not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            s = raw.split("#", 1)[0].strip()
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", s)
            if m:
                names.add(norm(m.group(1)))
    return names


# ---------------------------------------------------------------------------
# Manifest-side derivation (manifest_common / manifest_per_profile live in compose.py)
# ---------------------------------------------------------------------------
def manifest_composed(manifest, profile):
    """Direct-dependency set composed from the manifest for a profile: common overridden by per_profile."""
    merged = dict(manifest_common(manifest))
    for name, info in manifest_per_profile(manifest, profile).items():
        merged[name] = info
    return merged


# ---------------------------------------------------------------------------
# schema-lite validation
# ---------------------------------------------------------------------------
def load_schema(deps_dir):
    """deps.schema.json -- the field source of truth for schema-lite (enum/field sets/pattern are no longer replicated locally)."""
    path = os.path.join(deps_dir, "deps.schema.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_with_jsonschema(manifest, schema):
    """Full validation with the same deps.schema.json. Missing jsonschema is an error."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is required for full deps.schema.json validation"], False
    try:
        jsonschema.validate(manifest, schema)
        return [], True
    except jsonschema.ValidationError as e:
        return [
            "jsonschema: %s @ %s"
            % (e.message, "/".join(str(p) for p in e.absolute_path))
        ], True
    except jsonschema.SchemaError as e:
        return ["jsonschema schema is invalid: %s" % e], True


def _dep_entry_problems(schema, where, entry):
    """Field/enum/pattern validation of one depEntry (a common / per_profile value) against the schema.

    Field set, source enum, and sha256 pattern are all read from deps.schema.json -- the gate
    and the schema always share one authority; unknown source fails right here, never
    silently defaults to pypi."""
    props = schema["definitions"]["depEntry"]["properties"]
    problems = []
    for key in entry:
        if key not in props:
            problems.append(
                f"{where} has field {key!r} outside the schema (valid: {sorted(props)})"
            )
    src = entry.get("source")
    if src is not None and src not in props["source"]["enum"]:
        problems.append(
            f"{where} source={src!r} not in enum {props['source']['enum']} (unknown source must fail)"
        )
    sha = entry.get("sha256")
    if sha is not None and not re.fullmatch(props["sha256"]["pattern"], str(sha)):
        problems.append(
            f"{where} sha256 has invalid shape (need {props['sha256']['pattern']}): {sha!r}"
        )
    return problems


def validate_schema_lite(manifest, schema=None):
    problems = []
    for key in (
        "schema_version",
        "python",
        "indexes",
        "profiles",
        "packages",
        "wheel",
        "exceptions",
    ):
        if key not in manifest:
            problems.append(f"top-level key {key} missing")
    if problems:
        return problems

    if manifest["schema_version"] != 1:
        problems.append(
            f"schema_version should be 1, got {manifest['schema_version']!r}"
        )
    if "version" not in manifest.get("python", {}):
        problems.append("python.version missing")
    if "oss_base" not in manifest.get("indexes", {}):
        problems.append("indexes.oss_base missing")

    prof_names = [p.get("name") for p in manifest["profiles"]]
    if prof_names != PROFILES:
        problems.append(f"profiles order/list should be {PROFILES}, got {prof_names}")
    for p in manifest["profiles"]:
        for f in (
            "name",
            "platform",
            "bazel_config",
            "index_view",
            "hub",
            "lock",
        ):
            if f not in p:
                problems.append(f"profile {p.get('name')} missing field {f}")

    valid = set(PROFILES)
    pkg_props = (
        (schema or {})
        .get("properties", {})
        .get("packages", {})
        .get("items", {})
        .get("properties", {})
    )
    for pkg in manifest["packages"]:
        if "name" not in pkg:
            problems.append("a record in packages is missing name")
            continue
        nm = pkg["name"]
        if pkg_props:
            for key in pkg:
                if key not in pkg_props:
                    problems.append(
                        f"package {nm} has field {key!r} outside the schema (valid: {sorted(pkg_props)})"
                    )
        if "common" not in pkg and "per_profile" not in pkg:
            problems.append(f"package {nm} has neither common nor per_profile")
        if schema is not None and "common" in pkg:
            problems += _dep_entry_problems(
                schema, f"package {nm}.common", pkg["common"]
            )
        for prof in pkg.get("per_profile", {}):
            if prof not in valid:
                problems.append(
                    f"package {nm} per_profile key {prof} is not a valid profile"
                )
        for prof, info in pkg.get("per_profile", {}).items():
            if schema is not None:
                problems += _dep_entry_problems(schema, f"package {nm}[{prof}]", info)
            cv = info.get("cc_view")
            if cv is not None:
                for f in ("name", "sha256", "urls", "build_file"):
                    if not cv.get(f):
                        problems.append(
                            f"package {nm}[{prof}] cc_view missing field {f}"
                        )

    for exc in manifest["exceptions"]:
        for prof in exc.get("exists_in", []):
            if prof not in valid:
                problems.append(
                    f"exceptions[{exc.get('name')}] contains invalid profile {prof}"
                )
        if not exc.get("reason"):
            problems.append(f"exceptions[{exc.get('name')}] missing reason")

    wheel = manifest["wheel"]
    for k in ("groups", "exclude", "arch_names", "platform_pins"):
        if k not in wheel:
            problems.append(f"wheel missing sub-key {k}")
    for prof in wheel.get("arch_names", {}):
        if prof not in valid:
            problems.append(f"wheel.arch_names key {prof} is not a valid profile")

    for ex in manifest["exceptions"]:
        for f in ("name", "exists_in", "allowed_refs", "reason"):
            if f not in ex:
                problems.append(f"exceptions record {ex.get('name')} missing field {f}")
        for prof in ex.get("exists_in", []):
            if prof not in valid:
                problems.append(
                    f"exceptions[{ex.get('name')}] exists_in contains invalid profile {prof}"
                )
    return problems


# ---------------------------------------------------------------------------
# Semantic self-consistency + retained checks + derivation consistency
# ---------------------------------------------------------------------------
def _profile_machine(manifest, profile):
    """profile 的 platform_machine —— marker 求值只需要它。"""
    for prof in manifest.get("profiles", []):
        if prof.get("name") == profile:
            return prof.get("platform", "").split("-", 1)[0]
    return ""


def check_absent(manifest):
    """A package absent from a profile (not in its exists_in) must not be declared for it.

    The composed set is the declaration surface, before uv evaluates markers, so an entry
    whose marker already excludes that profile's machine is not a contradiction: that is
    exactly how av/decord stay off the aarch64 locks.
    """
    problems = []
    for exc in manifest["exceptions"]:
        name = exc.get("name", "")
        exists = set(exc.get("exists_in", []))
        for profile in PROFILES:
            if profile in exists:
                continue
            entry = manifest_composed(manifest, profile).get(name)
            if entry is None:
                continue
            marker = entry.get("marker") or ""
            machine = _profile_machine(manifest, profile)
            # Only an equality marker proves exclusion: `== "x86_64"` on an aarch64 profile
            # resolves the package away, so the lock never carries it. A substring test would
            # read `!= "x86_64"` as excluding x86_64 and turn a real contradiction green.
            pinned = re.findall(r'platform_machine\s*==\s*["\']([^"\']+)["\']', marker)
            if pinned and machine and machine not in pinned:
                continue
            problems.append(
                f"[exceptions] {name} is not in exists_in for {profile}, but the manifest"
                f" still declares it there (marker={marker!r})"
            )
    return problems


def _manifest_provided_names(manifest):
    """Supply sources of the "non-base-wheel set" inside the manifest -> normalized package
    name sets, keyed by evidence label: platform_pins.<key> and groups.<name>
    (for checking wheel.exclude.provided_by)."""
    ev = {}
    for key, pins in manifest["wheel"]["platform_pins"].items():
        names = set()
        for p in pins:
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", str(p).strip())
            if m:
                names.add(norm(m.group(1)))
        ev["platform_pins.%s" % key] = names
    for grp, gnames in manifest["wheel"]["groups"].items():
        names = set()
        for n in gnames:
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", str(n).strip())
            if m:
                names.add(norm(m.group(1)))
        ev["groups.%s" % grp] = names
    return ev


def check_wheel_exclude_provable(manifest, deps_dir):
    """Every wheel.exclude entry must be provable.

    - the excluded name must be a known package (in manifest.common or any per_profile);
    - shipped:false must come with a reason;
    - each provided_by evidence must be platform_pins.<key>/groups.<name> inside the
      manifest, and the package must actually exist in that source."""
    problems = []
    known = {norm(p["name"]) for p in manifest["packages"]}
    evidence = _manifest_provided_names(manifest)
    for name, spec in manifest["wheel"]["exclude"].items():
        if norm(name) not in known:
            problems.append(
                f"wheel.exclude[{name}]: not a known package (typo/stale entry, delete it)"
            )
        if "provided_by" in spec:
            providers = spec["provided_by"]
            if not providers:
                problems.append(f"wheel.exclude[{name}]: provided_by is empty")
            for f in providers:
                if f not in evidence:
                    problems.append(
                        f"wheel.exclude[{name}]: provided_by evidence {f!r} invalid"
                        f" (must be platform_pins.<key> or groups.<name>)"
                    )
                elif norm(name) not in evidence[f]:
                    problems.append(
                        f"wheel.exclude[{name}]: claims to be provided by {f}, but that source does not contain it (av-class defect)"
                    )
        elif spec.get("shipped") is False:
            if not spec.get("reason"):
                problems.append(f"wheel.exclude[{name}]: shipped:false missing reason")
        else:
            problems.append(
                f"wheel.exclude[{name}]: missing provable annotation -- need shipped:false+reason or provided_by"
            )
    return problems


def check_arch_names_resolvable(manifest, deps_dir):
    """Every big-artifact name in wheel.arch_names[profile] must resolve to a pin in that
    profile's lock. Replicates the raise in gen_wheel_requires.gen_leaf: a missing pin ->
    the derived wheel would lack the big artifact."""
    problems = []
    lock_by_profile = {p["name"]: p.get("lock") for p in manifest["profiles"]}
    for profile, names in manifest["wheel"]["arch_names"].items():
        lock = lock_by_profile.get(profile)
        if not lock:
            problems.append(
                f"wheel.arch_names[{profile}]: profile has no lock registered"
            )
            continue
        have = names_in_file(os.path.join(deps_dir, lock))
        for name in names:
            if norm(name) not in have:
                problems.append(
                    f"wheel.arch_names[{profile}] entry {name} not found in {lock} (add pin + relock)"
                )
    return problems


def check_wheel_shape(manifest):
    """Every wheel.platform_pins entry has name==version shape; every wheel.groups entry is a valid package name."""
    problems = []
    pin_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(==|@|>=|<=|~=|!=|<|>).+")
    name_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
    for key, pins in manifest["wheel"]["platform_pins"].items():
        for p in pins:
            if not isinstance(p, str) or not pin_re.match(p.strip()):
                problems.append(
                    f"wheel.platform_pins.{key} entry not in name==version shape: {p!r}"
                )
    for grp, names in manifest["wheel"]["groups"].items():
        for n in names:
            if not isinstance(n, str) or not name_re.match(n.strip()):
                problems.append(f"wheel.groups.{grp} entry name invalid: {n!r}")
    return problems


def _parse_starlark_dict(text, varname):
    """Extract the `varname = {...}` literal from .bzl text and literal_eval it (values are
    all simple list/str, no nested dicts, so literal_eval is a safe parser). Returns None
    if missing."""
    m = re.search(
        r"^%s\s*=\s*(\{.*?\n\})" % re.escape(varname), text, re.DOTALL | re.MULTILINE
    )
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


def check_absent_map(manifest, deps_dir):
    """ABSENT/ABSENT_REASON in //deps:absent_map.bzl == derivation from manifest.exceptions.

    absent_map.bzl is generated by deps/relock.py (hand-editing forbidden); this is a
    read-only comparison to prevent hand-edit drift."""
    path = os.path.join(deps_dir, "absent_map.bzl")
    if not os.path.exists(path):
        return [
            "deps/absent_map.bzl missing (run `scripts/rtpcli deps sync --all` to generate; "
            "arch_select.requirement relies on it to route absent branches)"
        ]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    got_absent = _parse_starlark_dict(text, "ABSENT")
    got_reason = _parse_starlark_dict(text, "ABSENT_REASON")
    if got_absent is None or got_reason is None:
        return [
            "failed to parse absent_map.bzl: must contain ABSENT = {...} and ABSENT_REASON = {...} literals"
        ]
    # exceptions[].exists_in is the single registration surface; a package's absent set is
    # its complement. Missing that complement would let a package registered in exceptions
    # land as a real hub label in a profile that lacks it, failing at build time with
    # "directory is not a package" instead of an explicit absence error.
    all_profiles = [p["name"] for p in manifest.get("profiles", [])]
    exp_absent = {}
    exp_reason = {}
    for exc in manifest.get("exceptions", []):
        key = norm(exc.get("name", ""))
        if not key:
            continue
        exists = {norm(x) for x in exc.get("exists_in", [])}
        missing = [x for x in all_profiles if norm(x) not in exists]
        if not missing:
            continue
        exp_absent[key] = missing
        exp_reason[key] = exc.get("reason", "") or "only exists in %s" % ", ".join(
            exc.get("exists_in", [])
        )
    problems = []
    if got_absent != exp_absent:
        problems.append(
            f"ABSENT in absent_map.bzl != derivation from manifest.exceptions: file={got_absent} manifest={exp_absent}"
        )
    if got_reason != exp_reason:
        problems.append(
            f"ABSENT_REASON in absent_map.bzl != manifest derivation: file={got_reason} manifest={exp_reason}"
        )
    return problems


def main():
    deps_dir = sys.argv[1] if len(sys.argv) > 1 else HERE
    manifest_path = os.path.join(deps_dir, "deps.json")
    if not os.path.exists(manifest_path):
        print(f"FAIL: {manifest_path} not found")
        sys.exit(1)
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except ValueError as e:
        print(f"FAIL: failed to parse deps.json: {e}")
        sys.exit(1)

    try:
        schema = load_schema(deps_dir)
    except (OSError, ValueError) as e:
        print(
            f"FAIL: deps.schema.json missing/unparsable (the schema is the validation source of truth and must exist): {e}"
        )
        sys.exit(1)

    problems = validate_schema_lite(manifest, schema)
    schema_problems, _ = validate_with_jsonschema(manifest, schema)
    problems += schema_problems
    if problems:
        print("FAIL: deps.json schema validation failed (schema=deps.schema.json):")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    all_problems = []
    all_problems += check_absent(manifest)
    all_problems += check_wheel_exclude_provable(manifest, deps_dir)
    all_problems += check_arch_names_resolvable(manifest, deps_dir)
    all_problems += check_wheel_shape(manifest)
    all_problems += check_absent_map(manifest, deps_dir)

    if all_problems:
        print(
            "FAIL: deps.json self-consistency/derivation-consistency validation failed:"
        )
        for p in all_problems:
            print("  " + p)
        print(
            "  Fix: adjust deps.json per the diff above, then run `scripts/rtpcli deps sync --all`"
        )
        sys.exit(1)

    n_pkgs = len(manifest["packages"])
    n_cc = sum(
        1
        for pkg in manifest["packages"]
        for info in pkg.get("per_profile", {}).values()
        if info.get("cc_view")
    )
    wheel = manifest["wheel"]
    n_wheel = (
        len(wheel["exclude"])
        + sum(len(v) for v in wheel["groups"].values())
        + sum(len(v) for v in wheel["arch_names"].values())
        + sum(len(v) for v in wheel["platform_pins"].values())
    )
    print(
        f"OK: deps.json self-consistency/derivation-consistency passed -- packages={n_pkgs} profiles={len(manifest['profiles'])} "
        f"wheel_records={n_wheel} cc_view_records={n_cc}"
    )


if __name__ == "__main__":
    main()

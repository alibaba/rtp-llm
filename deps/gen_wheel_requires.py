#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate the aggregated Requires-Dist list for the wheel.
#
# Python stdlib only, so it can run directly at build time as a Bazel py_binary.
# Semantics: versions of base names and arch big-artifact names are both taken from the
# selected profile's own lock, so the wheel declaration matches the dependency graph that
# profile actually built and tested; platform pins and the kserve extra are manifest verbatim.
# Aggregation = sort -u (base names ∪ per-profile leaf ∪ platform pins [∪ kserve extra]).
#
# Inputs: --manifest deps.json, --lock requirements_lock_*.txt (profile derived from the
# lock basename via deps.json profiles), --mode full|kserve, --out output path; profiles
# not in deps.json must be declared by --overlay, and that overlay is their single source
# of truth.
import argparse
import json
import os
import re
import sys

# Direct execution and Bazel py_binary both put deps/ on sys.path, but importlib loading
# by [file path] does not -- the internal-source tests load this file exactly that way,
# and this file also describes itself as a standalone stdlib-only script. So add our own
# directory to sys.path first, then import the sibling pkgname.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from compose import manifest_common
from pkgname import norm


# ---------------------------------------------------------------------------
# Profile resolution: public profiles come from deps.json; anything else [must] be
# declared by the passed-in overlay. This file therefore contains no private profile
# names or private lock file names -- the open-source tool only recognizes "public facts
# + the caller's overlay"; all private knowledge arrives with the overlay.
# ---------------------------------------------------------------------------
def overlay_profiles(overlay):
    """Profiles declared by the overlay: name -> record. The overlay keys a dict by name (deps.json uses a list)."""
    if not overlay:
        return {}
    return dict(overlay.get("profiles", {}))


def lock_to_profile_map(manifest, overlay=None):
    """lock basename -> profile name. The 6 public ones come from deps.json profiles (single source of truth);
    private ones take the basename of the overlay's profiles[].lock."""
    out = {p["lock"]: p["name"] for p in manifest.get("profiles", [])}
    for name, prof in overlay_profiles(overlay).items():
        lock = prof.get("lock")
        if lock:
            out[os.path.basename(lock)] = name
    return out


def profile_record(manifest, profile, overlay=None):
    for p in manifest.get("profiles", []):
        if p["name"] == profile:
            return p
    return overlay_profiles(overlay).get(profile)


def platform_pins_key(manifest, profile, overlay=None):
    """profile -> key into wheel.platform_pins (None = no platform pins appended).

    Derived from facts instead of copying another table:
      - aarch64 platforms carry no platform pins (arm / cuda12_9_arm);
      - if the profile name itself is a platform_pins key, use it (rocm);
      - otherwise use default (cpu / cuda12_6 / cuda12_9, and x86_64 private profiles).
    """
    record = profile_record(manifest, profile, overlay)
    if record is None:
        raise RuntimeError(
            "profile %r not declared in deps.json or the overlay" % profile
        )
    if record.get("platform", "").startswith("aarch64"):
        return None
    if profile in manifest["wheel"]["platform_pins"]:
        return profile
    return "default"


# ---------------------------------------------------------------------------
# Helpers shared with the manifest relock implementation
# ---------------------------------------------------------------------------


def _version_from_url(url):
    """Version in the basename of a direct link (wheel/sdist file name); None if unparsable."""
    from urllib.parse import unquote

    base = unquote(url.rsplit("/", 1)[-1].split("#", 1)[0])
    m = re.match(
        r"^[A-Za-z0-9_.]+-([^-]+)(?:-.+\.whl|\.(?:tar\.gz|zip|tar\.bz2))$", base
    )
    return m.group(1) if m else None


def lock_versions(lock_text):
    # {norm_name: version}.
    # Both ``name==version`` and PEP 508 direct links ``name @ URL`` must be recognized:
    # self-built artifacts pinned via direct links would vanish entirely from the wheel
    # declaration if only ``==`` were recognized.
    out = {}
    for line in lock_text.splitlines():
        s = line.strip()
        if not s or s[0] in "#-":
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ ;\\]+)", s)
        if m:
            out[norm(m.group(1))] = m.group(2)
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)", s)
        if m:
            ver = _version_from_url(m.group(2).rstrip("\\").strip())
            if ver:
                out[norm(m.group(1))] = ver
    return out


# ---------------------------------------------------------------------------
# Wheel metadata generators
# ---------------------------------------------------------------------------
def _wheel_requires(manifest, ver):
    exclude = set(norm(n) for n in manifest["wheel"]["exclude"])
    lines = []

    def emit_name(display, marker):
        n = norm(display)
        if n in exclude:
            return None
        mk = " ; " + marker if marker else ""
        if n in ver:
            return "%s==%s%s" % (display, ver[n], mk)
        return None  # only for extras-only unlocked packages (the wheel group below falls back to bare names)

    seen = set()
    for name, info in manifest_common(manifest).items():
        e = emit_name(name, info.get("marker"))
        if e and e not in seen:
            seen.add(e)
            lines.append(e)
    for name in manifest["wheel"]["groups"].get("wheel", []):
        n = norm(name)
        if n in exclude:
            continue
        if n in ver:
            e = "%s==%s" % (name, ver[n])
        else:
            e = name
        if e not in seen:
            seen.add(e)
            lines.append(e)
    return lines


def generate_wheel_leaf(manifest, profile_lock_text, profile):
    """Per-profile arch big-artifact names (wheel.arch_names[profile]) + versions (from the profile lock).
    Raises if an arch name is not found in the profile lock.
    Profiles declared by an overlay have no public arch_names registration and go through
    generate_overlay_leaf (the leaf is always empty)."""
    return _wheel_leaf(manifest, lock_versions(profile_lock_text), profile)


def _wheel_leaf(manifest, ver, profile):
    lock_by_profile = {p["name"]: p["lock"] for p in manifest.get("profiles", [])}
    lock_name = lock_by_profile.get(profile, "<profile lock>")
    names = manifest["wheel"]["arch_names"].get(profile, [])
    lines = []
    missing = []
    for name in names:
        n = norm(name)
        if n in ver:
            lines.append("%s==%s" % (name, ver[n]))
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "wheel.arch_names[%s] %s not in %s (add pin + relock)"
            % (profile, missing, lock_name)
        )
    return lines


def generate_overlay_leaf(overlay, profile):
    """For overlay-declared profiles, the arch leaf is [always empty] -- not an omission but a consequence of the supply channel.

    The SDK stack of such profiles is not installed via pip; it is packed into the
    rtp_llm wheel with the bazel runfiles. Those packages exist only on the private
    index, which the index mounted during image builds does not contain; once written
    into Requires-Dist, the image build necessarily dies unable to resolve that version.

    Its only difference from public profiles is therefore [the version source of the base
    names]: versions come from that profile's own lock, so shared packages are declared
    at the versions it actually resolved, not some public profile's resolution.

    The overlay is still validated: it is the credential for that profile's existence
    (absence means the private overlay has not landed, and should fail at the entry point
    rather than passing off another profile's facts as its own).
    """
    profiles = overlay_profiles(overlay)
    record = profiles.get(profile)
    if not isinstance(record, dict) or not record.get("packages"):
        raise RuntimeError(
            "overlay profile %r has no packages: its source of truth is absent; "
            "refusing to impersonate it with another profile's lock" % profile
        )
    return []


def generate_platform_pins(manifest, profile, overlay=None):
    """Platform verbatim pins (key derived by platform_pins_key)."""
    key = platform_pins_key(manifest, profile, overlay)
    if key is None:
        return []
    return list(manifest["wheel"]["platform_pins"].get(key, []))


def generate_kserve_extra(manifest):
    """kserve extra verbatim names."""
    return list(manifest["wheel"]["groups"].get("kserve", []))


# ---------------------------------------------------------------------------
# Aggregation + CLI
# ---------------------------------------------------------------------------
def aggregate(manifest, profile_lock_text, profile, mode, overlay=None):
    """Both base and arch leaf take the selected profile's own lock, so "declared is
    tested" (a fixed reference lock would make the wheel's Requires-Dist disagree with
    the graph the profile actually built against).

    Public profiles use the wheel.arch_names leaf; other profiles must be declared by an
    overlay (leaf always empty).
    """
    union = set()
    ver = lock_versions(profile_lock_text)
    union.update(_wheel_requires(manifest, ver))
    public = {p["name"] for p in manifest.get("profiles", [])}
    if profile in public:
        union.update(_wheel_leaf(manifest, ver, profile))
    else:
        if overlay is None:
            raise RuntimeError(
                "profile %r is not a public profile, --overlay is required" % profile
            )
        union.update(generate_overlay_leaf(overlay, profile))
    union.update(generate_platform_pins(manifest, profile, overlay))
    if mode == "kserve":
        union.update(generate_kserve_extra(manifest))
    return sorted(x for x in union if x.strip())


def _read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate wheel Requires-Dist aggregate")
    ap.add_argument("--manifest", required=True, help="deps.json path")
    ap.add_argument("--lock", required=True, help="requirements_lock_*.txt path")
    ap.add_argument("--mode", required=True, choices=["full", "kserve"])
    ap.add_argument(
        "--profile",
        default=None,
        help="override profile (default derived from the --lock basename); non-public profiles also need --overlay.",
    )
    ap.add_argument(
        "--overlay",
        default=None,
        help="private overlay path: the single source of truth declaring non-public profiles.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    manifest = json.loads(_read_text(args.manifest))
    # Read the overlay first: it may declare the profile to be used this run, so it must
    # participate in profile validity judgement.
    overlay = json.loads(_read_text(args.overlay)) if args.overlay else None

    public = {p["name"] for p in manifest.get("profiles", [])}
    known = public | set(overlay_profiles(overlay))
    lock_to_profile = lock_to_profile_map(manifest, overlay)

    if args.profile is not None:
        profile = args.profile
        if profile not in known:
            ap.error(
                "unknown --profile %r (expected one of %s)" % (profile, sorted(known))
            )
    else:
        lock_base = os.path.basename(args.lock)
        if lock_base not in lock_to_profile:
            ap.error(
                "unknown lock basename %r (expected one of %s)"
                % (lock_base, sorted(lock_to_profile))
            )
        profile = lock_to_profile[lock_base]

    if profile not in public:
        if overlay is None:
            ap.error(
                "profile %r is not in deps.json; --overlay must declare it (the overlay is its single source of truth)"
                % profile
            )
    elif overlay is not None:
        ap.error(
            "--overlay is only meaningful for non-public profiles declared by an overlay"
        )

    profile_lock_text = _read_text(args.lock)

    lines = aggregate(manifest, profile_lock_text, profile, args.mode, overlay=overlay)
    out = ("\n".join(lines) + "\n") if lines else ""
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

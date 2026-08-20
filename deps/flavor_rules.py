#!/usr/bin/env python3
"""Single source of truth for arch flavor rules.

One table feeds the lock flavor check and the OSS PEP 503 index builder (which splits
the index tree per arch), preventing rule drift between supply and consumption.

The criterion matches on **segments of the local version**, not by prefix: the wild
carries a real `fast_safetensors-0.7.3+torch2.1.2.rocm`, and prefix matching (`^rocm`)
cannot see the rocm inside, which would hang a ROCm wheel on the CUDA index.

The two consumers **must** treat `unknown` (a local naming no arch, e.g.
`+gfa35072d0.d20260402`, `+ali`) differently -- that is each one's correct answer:
- lock gate = allowlist: unknown FAILs directly (an unregistered local is either a human
  pin that should be registered, or a resolver's own choice); blocking costs almost nothing.
- index split = denylist (`excluded_from_index`): only remove locals that name another
  arch; unknown is arch-neutral and stays published, otherwise wheels already pinned in
  the locks would be hidden from the index -- a resolution failure, far worse.
"""
import json
import os
import re

# arch -> own flavor tokens. Full-segment fullmatch, so versioned shapes must be spelled
# out in the pattern (the wild carries `+rocm7.2.0.gitb919bd0c` and
# `+cu12torch2.8cxx11abitrue`).
OWN = {
    "cuda12": [r"cu12[0-8]\d*"],
    "cuda12_9": [r"cu129\d*"],
    "cuda12_arm": [r"cu129\d*"],
    "rocm": [r"rocm[\d.]*", r"gfx\d+"],
    "cpu": [r"cpu"],
    "arm": [r"cpu"],
}

# arch -> foreign flavor tokens. Two deliberate judgements:
# 1) CUDA minor versions are mutually exclusive -- cu126 torch and cu129 torch have
#    different ABIs; one tree can ship only one;
# 2) tokens like `cu12torch2.8...` ("CUDA family without naming a minor") are unknown
#    (neutral, stays published) for every CUDA arch, but hit `cu1\d.*` and get removed
#    for rocm/cpu/arm.
FOREIGN = {
    "cuda12": [r"rocm.*", r"gfx\d+", r"cpu", r"cu13.*", r"cu129.*"],
    "cuda12_9": [r"rocm.*", r"gfx\d+", r"cpu", r"cu13.*", r"cu12[0-8].*"],
    "cuda12_arm": [r"rocm.*", r"gfx\d+", r"cpu", r"cu13.*", r"cu12[0-8].*"],
    "rocm": [r"cu1\d.*", r"cpu"],
    "cpu": [r"cu1\d.*", r"rocm.*", r"gfx\d+"],
    "arm": [r"cu1\d.*", r"rocm.*", r"gfx\d+"],
}


# arch (flavor) -> (lock file, index_view); the single source of truth is deps.json
# profiles. flavor key = profile.index_view minus the "simple-" prefix, which lands
# exactly on the OWN/FOREIGN keys.
def _load_locks():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "deps.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    locks = {}
    for p in manifest["profiles"]:
        flavor = p["index_view"]
        if flavor.startswith("simple-"):
            flavor = flavor[len("simple-") :]
        locks[flavor] = (p["lock"], p["index_view"])
    return locks


LOCKS = _load_locks()


# Explicit exemptions: the index tree skips denylist removal for these
# (arch, package name, full local) triples. Every entry must state a removal condition --
# this is an "on-record opening", not a regular channel.
# fast-safetensors cu121 in cuda12_9: a legacy cross-flavor pin that production installs
# today; the pin must resolve via the simple-cuda12_9 tree, hence the exemption.
# Removal condition: once the owner publishes a +torch*.cu129 build and updates the
# requirements pin, delete this entry and run pip-repo rebuild --flavor cuda12_9.
INDEX_EXEMPT = {
    ("cuda12_9", "fast-safetensors", "torch2.1.2.cu121"),
    ("cuda12_9", "fast_safetensors", "torch2.1.2.cu121"),
}


def segments(local):
    """Split a local version into decidable segments: `torch2.1.2.cu121` -> [torch2, 1, 2, cu121]."""
    return [s for s in re.split(r"[._+-]+", local.lower()) if s]


def _hits(patterns, segs):
    return any(re.fullmatch(p, s) for p in patterns for s in segs)


def classify(arch, local):
    """Affiliation of a local version suffix relative to an arch: allow / deny / unknown.

    unknown is a local naming no arch: consumers must explicitly choose a treatment (see
    the module docstring); defaulting to allow is forbidden.
    """
    segs = segments(local)
    if _hits(FOREIGN[arch], segs):
        return "deny"
    return "allow" if _hits(OWN[arch], segs) else "unknown"


def excluded_from_index(arch, local, name=None):
    """Whether this local version must be removed from the arch's index tree (denylist semantics).

    name optional: when a package name is passed, the INDEX_EXEMPT table is consulted
    first (name/local matching is case- and -/_-insensitive).
    """
    if name is not None:
        key = (arch, name.lower().replace("_", "-"), local.lower())
        if any(
            (a, n.lower().replace("_", "-"), lv.lower()) == key
            for a, n, lv in INDEX_EXEMPT
        ):
            return False
    return classify(arch, local) == "deny"

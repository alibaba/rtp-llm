#!/usr/bin/env python3
"""The open-source tree must not contain internal private repo/rpm names.

The PPU pip hub + private torch/rpm archives are created only on the internal side by the
rtp_non_module_deps extension from ppu.json data; names and credentials all live in
internal_source. Once such names flow back into the open-source declaration/consumption
surface (MODULE.bazel / arch_config/*.bzl / BUILD files under 3rdparty), that is a
private-name leak; this check searches the private-name list word by word and any hit
FAILs. `easy` is a common English word and only counts in the `@easy` reference form.

Usage: check_private_names.py <open-source repo root>
"""

import os
import re
import sys

# Internal private repo/rpm names (should live only in internal_source).
# Hardcoded baseline for public-only clones; when the internal overlay exists, the
# authoritative set is derived from ppu.json and asserted to be a superset of this list.
PRIVATE_NAMES = [
    "pip_ppu_torch",
    "pip_gpu_cuda13_torch",
    "pip_cuda13_arm_torch",
    "torch_2.9_py310_ppu",
    "torch_2.9_py310_cuda-aarch64",
    "torch_2.11_py310_cuda",
    "torch_2.11_py310_cuda-aarch64",
    "accl_ep_rpm",
    "solar",
    "tnet",
    "unicm",
    "u2mm",
    "ali-rdma-core",
    "easy",  # requires @-prefix, see below
    "vipserver",
    "alibaba_rdma",
    "flashinfer_ppu",
    "flashmla_ppu",
    "cutlass3_ppu_flashinfer",
    "cutlass3_ppu_flashmla",
    "cutlass_cu13",
    "cutlass3.6_cu13",
    "flashinfer_cpp_cu13",
]


def _derive_from_overlay(root):
    """When the internal overlay is present, derive the private name set from ppu.json.

    A newly added private artifact that forgets to show up here means check_private_names
    cannot catch its leakage into the open-source surface — so we fail-closed if the overlay
    is present but its names are not a superset of the hardcoded baseline.
    """
    path = os.path.join(root, os.pardir, "internal_source", "deps", "ppu.json")
    if not os.path.isfile(path):
        return None
    import json

    data = json.load(open(path, encoding="utf-8"))
    names = set()
    for kind, records in data.get("artifacts", {}).items():
        if not isinstance(records, list):
            continue
        for art in records:
            if isinstance(art, dict) and art.get("name"):
                names.add(art["name"])
    for prof in data.get("profiles", {}).values():
        if isinstance(prof, dict) and prof.get("hub"):
            names.add(prof["hub"])
    return names


# Token boundary: names contain . and -, so \b would break mid-name; delimit whole words
# by "neither side is a repo-name character".
_BOUND = r"[\w.\-]"


def _pattern(name):
    if name == "easy":
        # easy is a common English word; only count the @easy reference form
        # (e.g. @easy//file:file / @easy:easy).
        return re.compile(r"@easy(?!" + _BOUND + r")")
    return re.compile(
        r"(?<!" + _BOUND + r")" + re.escape(name) + r"(?!" + _BOUND + r")"
    )


_PATTERNS = [(n, _pattern(n)) for n in PRIVATE_NAMES]


def scan(path, rel, hits):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return
    for lineno, line in enumerate(text.split("\n"), 1):
        for name, pat in _PATTERNS:
            if pat.search(line):
                hits.append((rel, lineno, name))


def targets(root):
    # MODULE.bazel + arch_config/*.bzl + 3rdparty/**/BUILD* (word-by-word scan for internal private names).
    out = []
    module = os.path.join(root, "MODULE.bazel")
    if os.path.isfile(module):
        out.append(module)
    arch = os.path.join(root, "arch_config")
    if os.path.isdir(arch):
        for entry in sorted(os.listdir(arch)):
            if entry.endswith(".bzl"):
                out.append(os.path.join(arch, entry))
    third = os.path.join(root, "3rdparty")
    for dirpath, _dirs, files in os.walk(third):
        for fn in sorted(files):
            if fn.startswith("BUILD") or fn.endswith(".BUILD"):
                out.append(os.path.join(dirpath, fn))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ".."
    derived = _derive_from_overlay(root)
    if derived is not None:
        baseline = set(PRIVATE_NAMES)
        missing = sorted(derived - baseline)
        if missing:
            print(
                "FAIL: ppu.json declares %d private names not in PRIVATE_NAMES (hardcoded baseline drifted):"
                % len(missing)
            )
            for n in missing:
                print("  + %s" % n)
            print(
                "  Fix: add the names above to PRIVATE_NAMES in deps/check_private_names.py"
            )
            return 1
        names = sorted(derived | baseline)
    else:
        names = PRIVATE_NAMES
    patterns = [(n, _pattern(n)) for n in names]
    hits, files = [], targets(root)
    for path in files:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        rel = os.path.relpath(path, root)
        for lineno, line in enumerate(text.split("\n"), 1):
            for name, pat in patterns:
                if pat.search(line):
                    hits.append((rel, lineno, name))
    if hits:
        print(
            "FAIL: open-source tree contains %d internal private repo/rpm names (should live only in internal_source):"
            % len(hits)
        )
        for rel, lineno, name in hits:
            print("  %s:%d: %s" % (rel, lineno, name))
        print(
            "  Fix: private torch/rpm/PPU hub names should be created only by the internal defs.bzl payload from ppu.json;"
            " point the open-source consumption surface at a public cc_view repo or @rtp_extension//3rdparty/<n>"
        )
        return 1
    suffix = " (overlay: %d names from ppu.json)" % len(derived) if derived else ""
    print(
        "OK: scanned %d open-source declaration/consumption files, no internal private repo/rpm names%s"
        % (len(files), suffix)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

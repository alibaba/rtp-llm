#!/usr/bin/env python3
"""torch/aiter C++ repos referenced by arch_select must all be defined (prevents dangling build-time references).

A dangling flavor only blows up at analysis phase of that flavor, so it must be caught
offline. Criterion 1: diff "referenced torch_*/aiter repos" against "defined repos".
Definition surface = cc_view in the open-source deps/deps.json + artifacts.torch of the
internal ppu.json (union in an overlay; a standalone open-source clone has only the
former). Reference surface = open-source arch_config/arch_select.bzl + internal
bazel/arch_select.bzl (counted only if present). A non-empty difference is FAIL.
Criterion 2 (divergence): for config branches declared by both arch_selects, the
referenced torch repo must be identical -- otherwise the same flag links different ABIs
in the two trees. Only two divergence classes are allowed: a branch declared by only one
side, and the open side deliberately pointing at the absent_dep stub (no public supply).
"""

import os
import re
import sys

REF_RE = re.compile(r"@(torch[\w.\-]*|aiter)//")
NAME_RE = re.compile(r'name\s*=\s*"(torch[\w.\-]*|aiter)"')
# Open-source torch/aiter C++ coordinates live in deps.json's cc_view ("name": "torch_...");
# internal torch archives are built by the defs.bzl payload from ppu.json, so the definition
# surface = json names in deps.json + artifacts.torch entries in the internal ppu.json.
JSON_NAME_RE = re.compile(r'"name"\s*:\s*"(torch[\w.\-]*|aiter)"')

# torch_deps select branches: config label lines, plus the first @repo//:target
# reference after each. The target must be captured too: the open-source absent stub is
# @arch_config//:torch_cuda12_9_arm_absent -- "absent" is on the target, not in the repo
# name; looking only at repo names would make the exemption never apply.
BRANCH_RE = re.compile(r'"(@rtp_llm//[^"]+|//conditions:default)"\s*:')
ANY_REF_RE = re.compile(r"@([\w.\-]+)//([^\"\s]*)")


def torch_branches(path):
    """torch_deps' {config label: (repo name, full repo//target)}."""
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8").read()
    start = text.find("def torch_deps(")
    if start < 0:
        return {}
    end = text.find("\ndef ", start + 1)
    body = text[start : end if end > 0 else len(text)]
    out = {}
    marks = list(BRANCH_RE.finditer(body))
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        ref = ANY_REF_RE.search(body[m.end() : stop])
        if ref:
            out[m.group(1)] = (ref.group(1), ref.group(1) + "//" + ref.group(2))
    return out


def divergences(open_path, internal_path):
    """Branches declared on both sides that reference different repos -> [(label, open_repo, internal_repo)]."""
    pub, priv = torch_branches(open_path), torch_branches(internal_path)
    bad = []
    for label in sorted(set(pub) & set(priv)):
        (o_repo, o_full), (i_repo, _) = pub[label], priv[label]
        if o_repo == i_repo:
            continue
        # Open side pointing at the absent stub = no public supply, deliberate
        # divergence (marked on the target).
        if "absent" in o_full:
            continue
        bad.append((label, o_repo, i_repo))
    return bad


def refs_in(path):
    if not os.path.isfile(path):
        return set()
    return set(REF_RE.findall(open(path, encoding="utf-8").read()))


def defs_in(path):
    if not os.path.isfile(path):
        return set()
    return set(NAME_RE.findall(open(path, encoding="utf-8").read()))


def json_defs_in(path):
    if not os.path.isfile(path):
        return set()
    return set(JSON_NAME_RE.findall(open(path, encoding="utf-8").read()))


def main():
    root = (
        sys.argv[1] if len(sys.argv) > 1 else ".."
    )  # gate.sh starts from deps/; root is the open-source repo root

    internal = os.path.join(root, "internal_source")

    referenced = refs_in(os.path.join(root, "arch_config", "arch_select.bzl"))
    referenced |= refs_in(os.path.join(internal, "bazel", "arch_select.bzl"))

    defined = json_defs_in(os.path.join(root, "deps", "deps.json"))
    defined |= defs_in(os.path.join(root, "deps", "extensions", "http_deps.bzl"))
    defined |= json_defs_in(os.path.join(internal, "deps", "ppu.json"))

    missing = sorted(referenced - defined)
    if missing:
        print("FAIL: arch_select references undefined torch/aiter repos:")
        for name in missing:
            print(
                "  @%s (referenced by arch_select, defined in neither deps.json nor ppu.json)"
                % name
            )
        print(
            "  Fix: add the definition to cc_view in the open-source deps/deps.json or artifacts.torch in the internal deps/ppu.json, "
            "or point the arch_select reference at an existing repo name"
        )
        sys.exit(1)

    open_sel = os.path.join(root, "arch_config", "arch_select.bzl")
    priv_sel = os.path.join(internal, "bazel", "arch_select.bzl")
    shared = set(torch_branches(open_sel)) & set(torch_branches(priv_sel))
    bad = divergences(open_sel, priv_sel)
    if bad:
        print(
            "FAIL: the same config branch links different torch builds in the two trees:"
        )
        for label, o, i in bad:
            print('  "%s": open-source @%s vs internal @%s' % (label, o, i))
        print(
            "  Fix: converge both sides to the same repo; if the divergence is deliberate (no public supply), "
            "point the open side at the absent_dep stub so absence fails explicitly at analysis phase instead of silently swapping versions"
        )
        sys.exit(1)

    sides = (
        "overlay (both repos)" if os.path.isdir(internal) else "open-source standalone"
    )
    print(
        "OK: all %d torch/aiter repo references in arch_select are defined, %d shared branches consistent across trees (%s)"
        % (len(referenced), len(shared), sides)
    )


if __name__ == "__main__":
    main()

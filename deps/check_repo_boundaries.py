#!/usr/bin/env python3
"""Repo-boundary subtrees must not remain in the main repo's wildcard target surface.

In the WORKSPACE era, dropping a WORKSPACE file in a subdirectory made `bazel build ...`
skip that subtree; Bzlmod's REPO.bazel does [not] have that side effect. So after
swapping WORKSPACE for REPO.bazel, subtrees meant to be "consumed only via @repo" get
their packages rediscovered by the main repo -- shadow packages reference siblings by
apparent name visible only inside that repo, which necessarily fails to resolve in the
main repo:

  * 3rdparty/protobuf:protobuf ships a cc_test requiring @com_google_googletest (a main
    repo Bzlmod dep), falling into the ut job's `bazel test ...` target surface;
  * internal_source/{deps,bazel}: repo roots of @rtp_extension/@arch_config, whose
    shadow packages need @solar and other extension siblings.

Hence: every repo-boundary subtree must either be in .bazelignore, or be registered in
DUAL_ROLE stating "the main repo really has labels pointing into it".
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPER = os.path.dirname(ROOT)

BOUNDARY_FILES = ("REPO.bazel", "WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel")

# Dual role: both a repo root (pointed at by local_repository) and directly consumed by
# the main repo via //<dir>:... . These cannot be ignored -- ignoring them would make
# main-repo labels unable to find the package.
DUAL_ROLE = {
    "deps": "open-source stub root of @rtp_extension; the main repo also uses labels like //deps:gen_wheel_requires",
    "arch_config": "open-source stub root of @arch_config; the main repo also uses //arch_config:... labels",
}


def workspace_rel(path):
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    while rel.startswith("../"):
        rel = rel[3:]
    return rel


def ignored_prefixes():
    path = os.path.join(ROOT, ".bazelignore")
    out = set()
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = re.sub(r"#.*", "", line).strip().rstrip("/")
        if line:
            out.add(line)
    return out


def is_covered(rel, ignored):
    parts = rel.split("/")
    return any("/".join(parts[: i + 1]) in ignored for i in range(len(parts)))


def boundary_dirs():
    out = []
    trees = [ROOT]
    internal = os.path.join(SUPER, "internal_source")
    if os.path.isdir(internal):
        trees.append(internal)
    for tree in trees:
        for base, dirs, files in os.walk(tree):
            dirs[:] = [
                d for d in dirs if not d.startswith("bazel-") and d not in (".git",)
            ]
            if base == ROOT:
                continue  # the root MODULE.bazel is this repo's own boundary
            if any(f in files for f in BOUNDARY_FILES):
                out.append(workspace_rel(base))
    return sorted(set(out))


def main():
    ignored = ignored_prefixes()
    problems = []
    boundaries = boundary_dirs()
    for rel in boundaries:
        if is_covered(rel, ignored) or rel in DUAL_ROLE:
            continue
        problems.append(
            "%s has a repo-boundary file but is neither in .bazelignore nor registered in DUAL_ROLE -- its packages "
            "enter the main repo's wildcard surface (REPO.bazel does not make wildcards skip the subtree like WORKSPACE did)"
            % rel
        )
    if problems:
        print(
            "FAIL: repo-boundary subtrees leak into the main repo's wildcard target surface:"
        )
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(
        "OK: all %d repo-boundary subtrees are contained (.bazelignore or DUAL_ROLE registration, %d cases)"
        % (len(boundaries), len(DUAL_ROLE))
    )


if __name__ == "__main__":
    main()

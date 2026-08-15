#!/usr/bin/env python3
"""Self-test for the flavor_rules criterion table: the rules are regexes, so they need an executable statement of intent.

The local versions in CASES are all taken from real production artifacts (wheels on the
OSS index + the six locks), not constructed examples. To change the table, add a row
here first, then change OWN/FOREIGN.
"""
import sys

from flavor_rules import classify

# (local version, arch, expected affiliation)
CASES = [
    # own CUDA minor allows, foreign minor denies -- cu126 and cu129 torch have different ABIs
    ("cu126", "cuda12", "allow"),
    ("cu126", "cuda12_9", "deny"),
    ("cu129", "cuda12_9", "allow"),
    ("cu129", "cuda12_arm", "allow"),
    ("cu129", "cuda12", "deny"),
    # cross-platform
    ("rocm", "rocm", "allow"),
    ("rocm7.2.0.gitb919bd0c", "rocm", "allow"),
    ("rocm", "cuda12_9", "deny"),
    ("rocm7.2.0.gitb919bd0c", "cuda12_9", "deny"),
    ("cpu", "cpu", "allow"),
    ("cpu", "arm", "allow"),
    ("cpu", "cuda12", "deny"),
    ("gfx942", "cuda12_9", "deny"),
    # token hidden mid-string: prefix matching misses it, segment matching catches it
    # (real fast_safetensors shape in the wild)
    ("torch2.1.2.rocm", "cuda12_9", "deny"),
    ("torch2.1.2.cu121", "cuda12", "allow"),
    ("torch2.1.2.cu121", "cuda12_9", "deny"),
    # CUDA family without naming a minor: neutral for CUDA arches, foreign for rocm/cpu
    ("cu12torch2.8cxx11abitrue", "cuda12_9", "unknown"),
    ("cu12torch2.8cxx11abitrue", "cuda12", "unknown"),
    ("cu12torch2.8cxx11abitrue", "rocm", "deny"),
    ("cu12torch2.8cxx11abitrue", "cpu", "deny"),
    # build stamps / setuptools-scm: names no arch => unknown (lock gate blocks it, index keeps publishing)
    ("gfa35072d0.d20260402", "rocm", "unknown"),
    ("unknown.pai", "cuda12_9", "unknown"),
    ("ali", "cuda12_9", "unknown"),
    ("git7e1940d", "cuda12_9", "unknown"),
    ("9b680f4", "cuda12_9", "unknown"),
    ("local", "cuda12_9", "unknown"),
    ("125c29e5.20260423102158", "cuda12_9", "unknown"),
]


def main():
    bad = []
    for local, arch, want in CASES:
        got = classify(arch, local)
        if got != want:
            bad.append("  %s @ %s: want %s, got %s" % (local, arch, want, got))
    if bad:
        print("FAIL: flavor criterion table has %d cases off expectation:" % len(bad))
        print("\n".join(bad))
        return 1
    print(
        "OK: all %d cases of the flavor criterion table match expectations" % len(CASES)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

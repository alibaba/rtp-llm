#!/usr/bin/env python3
"""Packages in the locks carrying a PEP 440 local version must match the arch flavor.

Explicit pins are exactly the primary check targets (hand-written pins are the easiest
way to nail another arch's flavor in). deny (the local names another arch) always FAILs;
the only exemption channel is the explicit flavor_rules.INDEX_EXEMPT list (exempt hits
print WARN to stay visible). allow passes; unknown (neutral locals like +29d31c0) passes
but is counted, ensuring "checked count > 0" is verifiable and the check can never be
silently always-true.

Usage: check_lock_flavor.py <deps dir>
"""
import os
import sys

from flavor_rules import INDEX_EXEMPT, LOCKS, classify
from pkgname import PIN_RE, norm


def _exempt(arch, name, local):
    key = (arch, norm(name), local.lower())
    return any((a, norm(n), l.lower()) == key for a, n, l in INDEX_EXEMPT)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rc = 0
    stats = {"allow": 0, "unknown": 0, "exempt": 0}
    for arch, (lock, _src) in LOCKS.items():
        path = os.path.join(root, lock)
        if not os.path.exists(path):
            rc = 1
            print(f"FAIL[{arch}]: {lock} missing, cannot complete flavor validation")
            continue
        bad = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = PIN_RE.match(line.strip())
                if not m or "+" not in m.group(2):
                    continue
                name, ver = m.group(1), m.group(2)
                local = ver.split("+", 1)[1]
                verdict = classify(arch, local)
                if verdict == "deny":
                    if _exempt(arch, name, local):
                        stats["exempt"] += 1
                        print(
                            f"WARN[{arch}] {name}=={ver}: cross-flavor but on the INDEX_EXEMPT list (see removal condition)"
                        )
                    else:
                        bad.append(f"{name}=={ver}")
                else:
                    stats[verdict] += 1
        if bad:
            rc = 1
            print(f"FAIL[{arch}] {lock}: {len(bad)} packages have cross-arch flavor:")
            for b in bad:
                print(f"  {b}")
            print(
                "  Fix: change the package's pin in deps.json to this arch's flavor and rerun `scripts/rtpcli deps sync`;"
                " genuine legacy exemptions go into flavor_rules.INDEX_EXEMPT (a removal condition is mandatory)"
            )
    total = sum(stats.values())
    if total == 0:
        rc = 1
        print(
            "FAIL: checked count is 0 -- lock set abnormal, this check must not be always-true"
        )
    elif rc == 0:
        print(
            f"OK: {len(LOCKS)} arch locks, {total} local-version pins checked"
            f" (own {stats['allow']} / neutral {stats['unknown']} / exempt {stats['exempt']})"
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()

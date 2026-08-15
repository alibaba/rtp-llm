#!/usr/bin/env python3
"""Lock freshness: locks must be derived from the current deps.json.

The `# input-hash:` in a lock header is the resolution fingerprint `deps/relock.py`
stamps via `compose.resolution_digest` — sha256 over the generator version, the
canonical uv argv (src name, python version, platform, index urls), the pinned uv
version, and the composed base+src contents. A stamp hit is what lets `relock
--check` skip uv entirely, so this checker and the producer must agree byte for byte.
The same stamp in the absent_map.bzl header covers exceptions + the profile list
(`compose.absent_digest`, same algorithm as deps_sync.absent_input_hash). Fresh only
when `stamp == current digest`; no stamp, stamp mismatch, or missing lock all FAIL.
Usage: python3 deps/check_lock_freshness.py [deps_dir]
"""

import json
import re
import sys
from pathlib import Path

from compose import absent_digest, resolution_digest

DEPS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
STAMP_RE = re.compile(r"^#\s*input-hash:\s*([0-9a-f]{64})", re.M)


def main():
    manifest = json.loads((DEPS / "deps.json").read_text())
    uv_pin = manifest.get("python", {}).get("uv")
    fresh, problems = [], []
    for prof in manifest["profiles"]:
        lock = DEPS / prof["lock"]
        if not lock.exists():
            problems.append(f"{prof['lock']}: file missing (profile {prof['name']})")
            continue
        m = STAMP_RE.search(lock.read_text())
        want = resolution_digest(manifest, prof["name"], uv_pin)
        if not m:
            problems.append(f"{prof['lock']}: no input-hash stamp")
        elif m.group(1) == want:
            fresh.append(prof["lock"])
        else:
            problems.append(
                f"{prof['lock']}: stamp does not match deps.json"
                f" (recorded {m.group(1)[:12]}… actual {want[:12]}…)"
            )
    absent_map = DEPS / "absent_map.bzl"
    if not absent_map.exists():
        problems.append(
            "absent_map.bzl: file missing (run `scripts/rtpcli deps sync --all` to generate)"
        )
    else:
        m = STAMP_RE.search(absent_map.read_text())
        want = absent_digest(manifest)
        if not m:
            problems.append("absent_map.bzl: no input-hash stamp")
        elif m.group(1) == want:
            fresh.append("absent_map.bzl")
        else:
            problems.append(
                "absent_map.bzl: stamp does not match deps.json"
                f" (recorded {m.group(1)[:12]}… actual {want[:12]}…)"
            )
    if problems:
        print("FAIL: locks are not derived from the current deps.json:")
        for p in problems:
            print("  " + p)
        print(
            "  Fix: run `scripts/rtpcli deps sync --profile <name>` for the affected profile, "
            "then rerun `scripts/rtpcli deps check --all`"
        )
        sys.exit(1)
    print(f"OK: {len(fresh)} generated artifacts match the manifest digest (fresh)")


if __name__ == "__main__":
    main()

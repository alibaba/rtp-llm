# deps tools

Maintenance tools for the wheel/lock dependency surface: `deps/requirements_*`,
`deps/requirements_lock_*`, and the `rtp-opensource` PyPI mirror
(`oss://rtp-opensource/rtp_llm/<arch>/` wheels + PEP 503 indexes).

None of these tools handle credentials themselves — they rely on ossutil's own
configuration (`~/.ossutilconfig` or `OSS_*` env vars) or your oss2 setup.

## check_lock_sha.py — lock sha256 guard

Verifies every `deps/requirements_lock_*.txt` still pins the exact sha256 set
frozen at the dependency unification (`lock_sha_baseline.json` — 9 locks,
~1563 packages). Run it from the repo root:

    python3 deps/tools/check_lock_sha.py                    # vs committed baseline
    python3 deps/tools/check_lock_sha.py --internal <repo> --ref <ref>
                                                           # vs internal_source/deps@ref
    python3 deps/tools/check_lock_sha.py --regen            # ONLY after a reviewed
                                                           # re-resolution

URL hosts in the locks may be re-pointed at public mirrors (that is the point
of the unification), but wheel bytes — and therefore every sha256 — must stay
identical; the checker enforces exactly that.

## rebuild_pypi_index.py — PEP 503 index rebuild

Rebuilds the `simple-<platform>/` index trees from the wheels under
`oss://rtp-opensource/rtp_llm/<arch>/` (the arch dirs are the single source of
truth; index pages link directly at those objects). Requires ossutil v2 on
PATH.

    python3 deps/tools/rebuild_pypi_index.py --dry-run      # plan only
    python3 deps/tools/rebuild_pypi_index.py --prefix rtp_llm/simple-rebuild-test/
    python3 deps/tools/rebuild_pypi_index.py                # writes the LIVE indexes
    python3 deps/tools/rebuild_pypi_index.py --recursive    # also index wheels in
                                                            # dated subdirectories

Use `--prune` to also delete stale package pages that are no longer part of
the plan (off by default — stale pages are left in place).

## Uploading a new wheel

1. Obtain the wheel (however your lane's requirements source it) and confirm
   its sha256 equals the pin in the matching lock file.
2. Upload it to `oss://rtp-opensource/rtp_llm/<arch>/<wheel>`.
3. Update the index pages — either rerun `rebuild_pypi_index.py`, or append an
   `<a href="...#sha256=<sha>">` entry to `simple-<platform>/<pkg>/index.html`
   (and to the flat `simple/<pkg>/index.html` if the unified tree lists the
   package).
4. Verify: `curl` the index page and the wheel URL.

**Gotcha:** index pages are `<pkg>/index.html` objects. OSS serves
`GET .../<pkg>/` from the `index.html` inside it — writing an object whose key
ends in the bare trailing slash is invisible to pip. Always write the
`index.html` object.

The flat `rtp_llm/simple/` tree is a convenience aggregation of all platforms;
the per-platform `simple-<platform>/` trees are authoritative and can be
regenerated at any time from the arch dirs.

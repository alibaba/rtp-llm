#!/usr/bin/env python3
"""The single implementation of package-name/marker normalization (previously copied verbatim in 5 places).

The cost of copying is not line count, it is **silence**: these normalizations feed the
absent_map lookup, lock-hit judgement, requirement() shim names, and the wheel
declaration; any drift between them raises no error, it just makes the lookup miss --
the comment in `requirement.bzl` records one "forgot to fold `.`" case, harmless at the
time only because no absent key contained a dot, but the absent branch would silently
land on the real hub instead of the fail stub.

**norm deliberately does not collapse consecutive separators**: the canonical PEP 503
form is `re.sub(r"[-_.]+", "-", name)`, folding `a__b` into `a-b`; here we replace
character by character, yielding `a--b`. The latter is kept because it must match, byte
for byte, the manifest relock generator which generates the `absent_map.bzl` keys, and
`deps/requirement.bzl:norm_dep` (Starlark, cannot import this module; the internal
overlay's arch_select.bzl loads it from there). Verified: all
223 package names currently in use give identical results under both spellings, so this
is a latent divergence, not a live bug. If a package name with consecutive separators
ever appears, all three places must change **together**.
(Note: `_normalize` in `deps/extensions/defs.bzl` is a different matter -- it mirrors
rules_python's `normalize_name` and must fold, because it reconstructs the spoke repo
names rules_python generates.)
"""

import re

__all__ = ["norm", "norm_marker", "PIN_RE"]

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")


def norm(name):
    """Normalize a package name: lowercase + `_`/`.` to `-` (no collapsing of consecutive separators, see module docstring)."""
    return name.lower().replace("_", "-").replace(".", "-")


def norm_marker(mk):
    """Normalize a PEP 508 marker: None/empty -> None; otherwise collapse inner whitespace for equality comparison."""
    if mk is None:
        return None
    out = re.sub(r"\s+", " ", mk).strip()
    return out or None

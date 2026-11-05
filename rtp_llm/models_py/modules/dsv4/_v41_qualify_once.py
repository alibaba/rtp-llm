"""D4: once-per-chunk hoisting of V4.1 prefill qualification.

``DSV41_QUALIFY_PREFILL_ONCE`` — default ON since R9 (decision 1); ``=0``
is the escape hatch back to per-layer qualification.

``_require_qualified_prefill`` (attention_v41.py) runs once per attention
owner per chunk — the broadcast meta build (:1072) plus every layer's
``_prefill_produce`` (:2888): 22-23 full passes per chunk at PP2 21/19 —
each issuing ~25-70 tiny device predicate kernels (lengths, CP metadata,
context bound, per-region block-table coverage/page/alias-sort) and ~1.5 ms
of Python span per layer at 0.12 ms busy (MIGRATION-AUDIT D4: host-bound).
G1-G3 + ``DSV41_ATTN_ASYNC_CHECKS`` already removed the hard host syncs and
batched the assert LAUNCHES (G3); the per-layer predicate kernels and
Python remained.

The hoist caches on the per-chunk ``CPContext`` object (fresh each forward,
so cache lifetime is one chunk; the G2 ``_v41_expected_lengths`` pattern):

- **meta level** — the lengths / CP-metadata / context-bound predicates and
  their ``(lengths, ends)`` results.  Chunk-invariant given tensor identity
  (data_ptr/shape/dtype), the owner's rope limit and ``cp_on``; runs once
  per rank per chunk instead of ~23x.  The first caller's deferred-check
  flush group still gates the chunk.
- **tables level** — the per-region block-table device predicates, keyed by
  ``(region, table ptr+shape, base page count, ends identity,
  single_block)``.  Runs once per distinct fingerprint per chunk: when the
  framework binds one table tensor per region per forward (the broadcast
  build passes a single ``block_tables_by_type``), this collapses 21-23x to
  the ~2-3 distinct region-set fingerprints; per-layer rebinding simply
  misses and revalidates.

Host-only structural checks (lengths dtype/shape, table layout, pool
lookups, L20 provenance, TP/capture guards) ALWAYS run — fail-loud
semantics unchanged, microsecond cost.  Qualification is checks-only, so
outputs are byte-exact by construction (sha-gated regardless).

Never active for: flag off, decode (``is_decode``), non-CP prefill, graph
capture (the eager path raises under capture before any cache use), or
frozen/slotted contexts (setattr fallback → eager every call).
"""

import os

_ATTR = "_v41_qualify_once_cache"
_STATS = {
    "lookups": 0,
    "meta_hit": 0,
    "meta_miss": 0,
    "tables_hit": 0,
    "tables_miss": 0,
}
_STATS_ANNOUNCED = False


def qualify_prefill_once_enabled() -> bool:
    """Default ON since R9 (user decision; byte-exact, A/B median −99.5 ms
    3/3 in the A/B pool). ``=0`` is the escape hatch back to per-layer
    qualification; any non-"1" value also falls back (conservative)."""
    return os.environ.get("DSV41_QUALIFY_PREFILL_ONCE", "1") == "1"


def _stats_enabled() -> bool:
    return os.environ.get("DSV41_QUALIFY_ONCE_STATS", "0") == "1"


def hoist_active(cp, cp_on: bool, is_decode: bool) -> bool:
    """Gate for the hoisted path; everything else runs the eager body."""
    return (
        qualify_prefill_once_enabled()
        and bool(cp_on)
        and not is_decode
        and cp is not None
    )


def _cache(cp):
    cache = getattr(cp, _ATTR, None)
    if cache is None:
        cache = {"meta": {}, "tables": set()}
        try:
            setattr(cp, _ATTR, cache)
        except (AttributeError, TypeError):
            return None  # frozen/slotted context: eager fallback
    return cache


def _bump(name: str) -> None:
    if not _stats_enabled():
        return
    _STATS[name] = _STATS.get(name, 0) + 1
    total = _STATS["lookups"]
    if total and total % 4096 == 0:
        _stats_report()


def _stats_report() -> None:
    import sys

    print(
        "[QPONCE] lookups={lookups} meta_hit={meta_hit} meta_miss={meta_miss} "
        "tables_hit={tables_hit} tables_miss={tables_miss}".format(**_STATS),
        file=sys.stderr,
        flush=True,
    )


if _stats_enabled():
    import atexit as _qp_atexit

    _qp_atexit.register(_stats_report)


def meta_key(input_lengths, prefix_lengths, rope_max):
    return (
        input_lengths.data_ptr(),
        tuple(input_lengths.shape),
        str(input_lengths.dtype),
        str(input_lengths.device),
        prefix_lengths.data_ptr(),
        tuple(prefix_lengths.shape),
        str(prefix_lengths.dtype),
        int(rope_max),
    )


def lookup_meta(cp, key):
    """Return the cached ``(lengths, ends)`` pair or None."""
    _bump("lookups")
    cache = _cache(cp)
    if cache is None:
        _bump("meta_miss")
        return None
    hit = cache["meta"].get(key)
    _bump("meta_hit" if hit is not None else "meta_miss")
    return hit


def store_meta(cp, key, lengths, ends) -> None:
    cache = _cache(cp)
    if cache is not None:
        cache["meta"][key] = (lengths, ends)


def tables_key(region, table, base_pages, ends, single_block) -> tuple:
    return (
        str(region),
        table.data_ptr(),
        tuple(table.shape),
        int(base_pages),
        id(ends),
        bool(single_block),
    )


def tables_seen(cp, fingerprint) -> bool:
    cache = _cache(cp)
    if cache is None:
        _bump("tables_miss")
        return False
    seen = fingerprint in cache["tables"]
    _bump("tables_hit" if seen else "tables_miss")
    return seen


def tables_mark(cp, fingerprint) -> None:
    cache = _cache(cp)
    if cache is not None:
        cache["tables"].add(fingerprint)

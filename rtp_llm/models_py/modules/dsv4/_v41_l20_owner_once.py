"""R9: once-per-chunk hoisting of the native L20 owner device comparisons.

``DSV41_L20_OWNER_ONCE=1`` (default 0).

Stage-1 layers 21-39 import the L20 head source; every layer's
``_require_qualified_prefill`` step-10 calls
``NativeL20PrefillSource.require_owner`` (native_l20_prefill.py), which on
EVERY call rebuilds ``_cp_dispatch`` — 5 blocking device ``tolist()``
(input_lengths_global, prefix_lengths, global_positions 8KB,
req_id_per_token 4KB, local_is_real 1KB) — plus 2 block-table ``tolist()``
comparisons (hca_kv, indexer_kv): **7 DtoH host syncs × 19 layers =
133/chunk on stage 1, zero on stage 0** (the +113 DtoH/chunk delta from
J3.3; the dominant stage-1 host_late_intra source).

All seven comparisons are chunk-invariant given (source, cp context, bound
table tensors): the dispatch values come from the frozen per-chunk CP
metadata, the expected tables from the frozen published source. The hoist
caches a host fingerprint — (source id, kv-cache id, per-tag table
data_ptr+shape) — on the per-chunk CPContext (G2/D4 pattern; fresh ctx per
forward ⇒ chunk lifetime; frozen/slotted ctx ⇒ eager fallback). First
caller per chunk runs the full comparisons and marks; the other 18 layers
skip only the device comparisons. ALL host identity/shape/dtype checks
(_ready, layer mapping, provenance, table layout) still run on every call —
fail-loud semantics unchanged. Checks-only ⇒ byte-exact by construction;
sha-gated regardless.

``DSV41_L20_OWNER_ONCE_STATS=1`` prints cumulative [L20ONCE] hit/miss
counters (engagement proof; periodic every 1024 lookups + atexit).

CP1 (no CP context) always runs eager — nothing to cache on.
"""

import os

_ATTR = "_v41_l20_owner_cache"
_TAGS = ("hca_kv", "indexer_kv")
_STATS = {"lookups": 0, "hit": 0, "miss": 0}


def l20_owner_once_enabled() -> bool:
    return os.environ.get("DSV41_L20_OWNER_ONCE", "0") == "1"


def _stats_enabled() -> bool:
    return os.environ.get("DSV41_L20_OWNER_ONCE_STATS", "0") == "1"


def _cache(cp):
    cache = getattr(cp, _ATTR, None)
    if cache is None:
        cache = set()
        try:
            setattr(cp, _ATTR, cache)
        except (AttributeError, TypeError):
            return None  # frozen/slotted context: eager fallback
    return cache


def _bump(name):
    if not _stats_enabled():
        return
    _STATS[name] = _STATS.get(name, 0) + 1
    if _STATS["lookups"] and _STATS["lookups"] % 1024 == 0:
        _stats_report()


def _stats_report():
    import sys

    print(
        "[L20ONCE] lookups={lookups} hit={hit} miss={miss}".format(**_STATS),
        file=sys.stderr,
        flush=True,
    )


if _stats_enabled():
    import atexit as _l20_atexit

    _l20_atexit.register(_stats_report)


def owner_fingerprint(source, attention):
    """Host-only fingerprint of everything the device comparisons read.

    Returns None when the binding is not fingerprintable (no Mapping, non-
    tensor tables) — callers then run the eager path.
    """
    tables = getattr(attention, "_block_tables_by_type", None)
    get = getattr(tables, "get", None)
    if get is None:
        return None
    fp = [id(source), id(getattr(attention, "_kv_cache", None))]
    for tag in _TAGS:
        value = get(tag)
        if value is None:
            fp.append((tag, None))
            continue
        data_ptr = getattr(value, "data_ptr", None)
        shape = getattr(value, "shape", None)
        if data_ptr is None or shape is None:
            return None
        fp.append((tag, value.data_ptr(), tuple(shape)))
    return tuple(fp)


def owner_seen(cp, fingerprint):
    _bump("lookups")
    cache = _cache(cp)
    if cache is None:
        _bump("miss")
        return False
    seen = fingerprint in cache
    _bump("hit" if seen else "miss")
    return seen


def owner_mark(cp, fingerprint):
    cache = _cache(cp)
    if cache is not None:
        cache.add(fingerprint)

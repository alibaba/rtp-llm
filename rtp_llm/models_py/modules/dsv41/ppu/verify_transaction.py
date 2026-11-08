"""Bounded, call-owned native write journal for shared target verification.

No second cache allocator or full model loop. Reads retain the native alias;
only rows written by this verify call occupy scratch. Rejected rows never
reach native SWA, CSA or compressed pools. Acceptance comes from RTP's sampler.
"""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.verify import VerifyLayout


def accepted_position(tag, position, base, end, ratio):
    if tag in ("global_kv_1", "global_kv_2", "indexer_kv"):
        return base // ratio <= position // ratio < end // ratio
    return base <= position < end


@dataclass(frozen=True)
class SharedCacheVersion:
    key: tuple
    request: int
    positions: object
    payload: object


class SharedVerifyCache(SharedNativeCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = []
        self._native_read = False

    def insert_rows(self, layer, tag, slots, rows):
        if tag != "csa_state":
            return super().insert_rows(layer, tag, slots, rows)
        # Ordinary writes retain only the latest physical CSA slot. Verify
        # needs EVERY raw generation: an earlier even row may be the accepted
        # unfinished pair. Singleton versions avoid duplicate-address writes.
        for index in torch.argsort(slots[:, 1]).tolist():
            rid = int(slots[index, 0])
            self._copy_rows((layer, tag), rid, slots[index:index + 1, 1], rows[index:index + 1])
        self.writes.append({"layer": layer, "tag": tag, "rows": slots.shape[0],
                            "publication": "journal_all_raw_generations"})

    def _copy_rows(self, key, request_row, positions, rows=None):
        positions = positions.to(device=self.device, dtype=torch.int64)
        self._coordinates(key, request_row, positions)
        if rows is not None:
            self.journal.append(SharedCacheVersion(key, int(request_row), positions.clone(), rows.clone()))
            return None
        result = super()._copy_rows(key, request_row, positions)
        if self._native_read:
            return result
        # Logical positions disambiguate ring generations, including every
        # unfinished CSA pair. Never infer rollback from a sequence length.
        for version in self.journal:
            if version.key != key or version.request != int(request_row):
                continue
            match = positions[:, None] == version.positions[None, :]
            present = match.any(-1)
            result[present] = version.payload[match[present].long().argmax(-1)]
        return result

    def native_fingerprint(self, lengths):
        self._native_read = True
        try:
            return super().fingerprint(lengths)
        finally:
            self._native_read = False


class SharedVerifyTransaction:
    def __init__(self, cache, request_ids, layout, windows, tokens, *, call_id, max_backtracking):
        ids = tuple(map(int, request_ids.cpu().reshape(-1).tolist()))
        if len(ids) != len(layout.lengths) or len(set(ids)) != len(ids):
            raise ValueError("verify requires unique scheduler request IDs")
        if max(layout.lengths, default=0) > max_backtracking:
            raise ValueError("actual verify width exceeds frozen rollback capacity")
        if sum(layout.lengths) != layout.rows or tokens.numel() != layout.rows:
            raise ValueError("verify token/row layout is inconsistent")
        self.cache, self.request_ids, self.layout = cache, ids, layout
        self.call_id, self.max_backtracking = call_id, max_backtracking
        self.closed = False
        self.tokens = tuple(map(int, tokens.reshape(-1).cpu().tolist()))
        self.bindings = {
            key: (pool.tensor.data_ptr(), pool.table.clone()) for key, pool in cache.pools.items()
        }
        self.histories = []
        if cache.geometry.engram_enabled():
            if windows is None or tuple(windows.shape) != (layout.rows, cache.geometry.engram_max_ngram_size):
                raise ValueError("verify requires every scheduler Engram history window")
            host, ids_host = windows.cpu().tolist(), tokens.reshape(-1).cpu().tolist()
            for start, length in zip(layout.starts, layout.lengths):
                history = host[start][1:]
                versions = []
                for row in range(start, start + length):
                    expected = [int(ids_host[row]), *history][:cache.geometry.engram_max_ngram_size]
                    if host[row] != expected:
                        raise ValueError("verify Engram window contains future/mismatched tokens")
                    versions.append(tuple(expected))
                    history = expected[:-1]
                self.histories.append(versions)

    def _validate(self, request_ids, prefixes, accepted_rows):
        if self.closed:
            raise RuntimeError("verify transaction already finalized/aborted")
        ids = tuple(map(int, request_ids.cpu().reshape(-1).tolist()))
        bases = tuple(map(int, prefixes.cpu().reshape(-1).tolist()))
        counts = tuple(map(int, accepted_rows.cpu().reshape(-1).tolist()))
        if ids != self.request_ids or bases != self.layout.prefixes:
            raise ValueError("stale acceptance: scheduler request/order/prefix mismatch")
        if len(counts) != len(ids) or any(n < 0 or n > width for n, width in zip(counts, self.layout.lengths)):
            raise ValueError("accepted input rows exceed verified causal scope")
        for key, (pointer, table) in self.bindings.items():
            pool = self.cache.pools[key]
            if pointer != pool.tensor.data_ptr() or not torch.equal(table, pool.table):
                raise RuntimeError("native pool/page ownership changed before acceptance")
        return counts

    def finish(self, request_ids, prefixes, accepted_rows):
        counts = self._validate(request_ids, prefixes, accepted_rows)
        selected = []
        for version in self.cache.journal:
            base = self.layout.prefixes[version.request]
            end = base + counts[version.request]
            ratio = self.cache.geometry.ratio(version.key[0])
            keep = torch.tensor([
                accepted_position(version.key[1], p, base, end, ratio)
                for p in version.positions.tolist()
            ], dtype=torch.bool, device=self.cache.device)
            if version.key[1] == "csa_state":
                keep &= version.positions >= max(base, end - self.cache.pools[version.key].entries)
            if bool(keep.any()):
                selected.append(SharedCacheVersion(version.key, version.request, version.positions[keep], version.payload[keep]))
        # Validate ALL addresses and take physical backups before publishing.
        backups = []
        for v in selected:
            self.cache._coordinates(v.key, v.request, v.positions)
            backups.append(SharedCacheVersion(v.key, v.request, v.positions,
                SharedNativeCache._copy_rows(self.cache, v.key, v.request, v.positions).clone()))
        try:
            for v in selected:
                SharedNativeCache._copy_rows(self.cache, v.key, v.request, v.positions, v.payload)
        except Exception:
            try:
                for v in reversed(backups):
                    SharedNativeCache._copy_rows(self.cache, v.key, v.request, v.positions, v.payload)
            finally:
                self.abort()
            raise
        evidence = {
            "call": self.call_id, "request_ids": self.request_ids,
            "prefixes": self.layout.prefixes, "verified_rows": self.layout.lengths,
            "accepted_input_rows": counts,
            "accepted_ends": [p + n for p, n in zip(self.layout.prefixes, counts)],
            "accepted_input_token_ids": [list(self.tokens[start:start + n]) for start, n in zip(self.layout.starts, counts)],
            "maximum_backtracking": self.max_backtracking,
            "actual_backtracking": [width - n for width, n in zip(self.layout.lengths, counts)],
            "native_rows_published": sum(v.positions.numel() for v in selected),
            "engram_committed_windows": [list(v[n - 1]) if n else None for v, n in zip(self.histories, counts)],
            "native_aliases_preserved": True, "status": "ACCEPTED",
        }
        self.abort()
        return evidence

    def abort(self):
        self.cache.journal.clear()
        self.closed = True

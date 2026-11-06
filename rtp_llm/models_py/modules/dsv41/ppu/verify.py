"""Causal eager target verification with acceptance-only native publication.

Every query stage uses the ordinary decode body. All DP owners execute the
same number of stages; exhausted owners send invalid placeholder rows through
the existing EP/ENP collectives. Candidate cache bytes remain private until the
RTP sampler chooses a prefix. Native SWA/CSA never sees a rejected write.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VerifyLayout:
    lengths: tuple
    prefixes: tuple
    starts: tuple
    rows: int

    @classmethod
    def from_inputs(cls, primary, rows):
        lengths = tuple(int(x) for x in primary.input_lengths.cpu().tolist())
        prefixes = tuple(int(x) for x in primary.prefix_lengths.cpu().tolist())
        if len(prefixes) != len(lengths) or sum(lengths) != rows:
            raise ValueError("verify rows must be request-major dense/packed queries")
        if any(length < 1 for length in lengths) or any(
            prefix < 0 for prefix in prefixes
        ):
            raise ValueError(
                "verify needs an anchor per request and nonnegative prefixes"
            )
        starts, cursor = [], 0
        for length in lengths:
            starts.append(cursor)
            cursor += length
        return cls(lengths, prefixes, tuple(starts), rows)


@dataclass(frozen=True)
class CacheVersion:
    end: int
    payloads: tuple

    @classmethod
    def capture(cls, bridge, cache, request, base, end):
        payloads = []
        for key in cache.pool_keys():
            positions = bridge._positions(key, end).to(cache.device)
            if key[1] in ("global_kv_1", "global_kv_2", "indexer_kv"):
                ratio = cache.geometry.ratio(key[0])
                # An odd base's unfinished pair completes at base//2, too.
                positions = positions[positions // ratio >= base // ratio]
                slots = cache._slots_for_blocks(key, request, end // ratio)
                slots = slots[base // ratio :]
            else:
                slots = cache.slots_at_positions(*key, request, positions)
            payloads.append((key, positions.clone(), cache._gather(key, slots).clone()))
        return cls(end, tuple(payloads))

    def publish(self, bridge, request):
        for key, positions, packed in self.payloads:
            if positions.numel():
                bridge._copy_rows(key, request, positions, packed)


class VerifyTransaction:
    """One synchronous RTP round, bound to request order and absolute prefix."""

    def __init__(self, bridge, request_ids, layout):
        ids = tuple(int(x) for x in request_ids.cpu().reshape(-1).tolist())
        if len(ids) != len(layout.lengths) or len(set(ids)) != len(ids):
            raise ValueError(
                "verify requires one unique scheduler request id per row group"
            )
        self.bridge, self.request_ids, self.layout = bridge, ids, layout
        self.versions = [[] for _ in ids]
        self.closed = False
        self.observer = self.observation = None

    def record(self, cache, request, end):
        if self.closed:
            raise RuntimeError("verify transaction is closed")
        self.versions[request].append(
            CacheVersion.capture(
                self.bridge, cache, request, self.layout.prefixes[request], end
            )
        )

    def finish(self, request_ids, prefixes, accepted_rows):
        if self.closed:
            raise RuntimeError("verify transaction already finalized")
        ids = tuple(int(x) for x in request_ids.cpu().reshape(-1).tolist())
        bases = tuple(int(x) for x in prefixes.cpu().reshape(-1).tolist())
        counts = tuple(int(x) for x in accepted_rows.cpu().reshape(-1).tolist())
        if ids != self.request_ids or bases != self.layout.prefixes:
            raise ValueError(
                "stale verify acceptance: scheduler request/round mismatch"
            )
        if len(counts) != len(ids) or any(
            count < 0 or count > len(versions)
            for count, versions in zip(counts, self.versions)
        ):
            raise ValueError("accepted rows exceed the verified causal prefix")
        # Validate all addresses before the first native write. A cancellation
        # uses zero rows and leaves every native byte untouched.
        selected = [
            (row, versions[count - 1])
            for row, (count, versions) in enumerate(zip(counts, self.versions))
            if count
        ]
        for row, version in selected:
            for key, positions, _ in version.payloads:
                self.bridge._coordinates(key, row, positions)
        backups = [
            (row, key, positions, self.bridge._copy_rows(key, row, positions).clone())
            for row, version in selected
            for key, positions, _ in version.payloads
            if positions.numel()
        ]
        try:
            for row, version in selected:
                version.publish(self.bridge, row)
        except Exception:
            try:
                for row, key, positions, old in backups:
                    self.bridge._copy_rows(key, row, positions, old)
            finally:
                self.abort()
            raise
        if self.observer is not None:
            self.observer.finish(
                self.observation, "accept", counts, versions=self.versions
            )
        self.closed = True
        self.versions.clear()

    def abort(self):
        if not self.closed and self.observer is not None:
            self.observer.finish(self.observation, "abort", [0] * len(self.request_ids))
        self.closed = True
        self.versions.clear()

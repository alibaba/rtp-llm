"""Bounded, call-owned native write journal for shared target verification.

No second cache allocator or full model loop. Reads retain the native alias;
only rows written by this verify call occupy scratch. Rejected rows never
reach native SWA, CSA or compressed pools. Acceptance comes from RTP's sampler.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import enabled
from rtp_llm.models_py.modules.dsv41.ppu.verify import VerifyLayout


@lru_cache(maxsize=1)
def _graph_verify_receipt_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def read_receipt(
        completion,
        lengths,
        prefixes,
        consumed_ids,
        consumed_windows,
        input_ids,
        input_windows,
        receipt,
        input_lengths,
        input_prefixes,
        CAPTURE_BATCH: tl.constexpr,
        BATCH: tl.constexpr,
        ROWS: tl.constexpr,
        WINDOW_WORDS: tl.constexpr,
        BLOCK: tl.constexpr,
        DEVICE_LAYOUT: tl.constexpr,
    ):
        index = tl.arange(0, BLOCK)
        live_id = index < ROWS
        live_window = index < WINDOW_WORDS
        expected_ids = tl.load(input_ids + index, live_id, other=0).to(tl.int64)
        expected_windows = tl.load(
            input_windows + index, live_window, other=0
        ).to(tl.int64)
        replay_ids = tl.load(consumed_ids + index, live_id, other=0).to(tl.int64)
        replay_windows = tl.load(
            consumed_windows + index, live_window, other=0
        ).to(tl.int64)
        mismatch = (expected_ids != replay_ids) | (expected_windows != replay_windows)
        receipt_lengths = tl.load(lengths + index, index < CAPTURE_BATCH, other=0)
        receipt_prefixes = tl.load(prefixes + index, index < BATCH, other=0)
        if DEVICE_LAYOUT:
            scheduler_lengths = tl.load(input_lengths + index, index < BATCH, other=0)
            scheduler_prefixes = tl.load(input_prefixes + index, index < BATCH, other=0)
            mismatch |= (
                (receipt_lengths.to(tl.int64) != scheduler_lengths.to(tl.int64))
                | (receipt_prefixes.to(tl.int64) != scheduler_prefixes.to(tl.int64))
            )
            receipt_lengths = scheduler_lengths
            receipt_prefixes = scheduler_prefixes
            payload = receipt + 2 + CAPTURE_BATCH + BATCH
            tl.store(payload + index, expected_ids, live_id)
            tl.store(payload + ROWS + index, expected_windows, live_window)
        tl.store(receipt, tl.load(completion))
        tl.store(receipt + 1, tl.max(mismatch.to(tl.int32), axis=0))
        tl.store(
            receipt + 2 + index,
            receipt_lengths,
            index < CAPTURE_BATCH,
        )
        tl.store(
            receipt + 2 + CAPTURE_BATCH + index,
            receipt_prefixes,
            index < BATCH,
        )

    return read_receipt, triton.next_power_of_2


def _launch_graph_verify_receipt(impl, inputs, rows, batch, *, device_layout=None):
    if (
        impl.mode != "dense"
        or inputs.is_fake_stream
        or not enabled("RTP_V41_PPU_VERIFY_RECEIPT", default=True)
    ):
        return None

    ids, windows = inputs.input_ids, inputs.engram_token_windows
    if not isinstance(ids, torch.Tensor) or not isinstance(windows, torch.Tensor):
        return None
    capture_batch = impl.batch_size
    window_words = windows.numel()
    if (
        not 0 < batch <= capture_batch <= 64
        or not 0 < rows <= min(128, impl.rows)
        or ids.shape != (rows,)
        or windows.ndim != 2
        or windows.shape[0] != rows
        or not 0 < window_words <= 512
    ):
        return None

    completion = impl.cache.completions
    lengths, prefixes = impl.consumed_lengths, impl.consumed_prefixes
    consumed_ids, consumed_windows = impl.consumed_ids, impl.consumed_windows
    if (
        completion.shape != ()
        or completion.dtype != torch.int64
        or lengths.shape != (capture_batch,)
        or prefixes.shape != (capture_batch,)
        or consumed_ids.shape != (impl.rows,)
        or consumed_windows.shape != (impl.rows, windows.shape[1])
        or not ids.is_cuda
    ):
        return None
    device = ids.device
    buffers = (completion, lengths, prefixes, consumed_ids, consumed_windows, ids, windows)
    layout_buffers = (lengths, prefixes) if device_layout is None else device_layout
    if any(
        value.device != device
        or value.dtype not in (torch.int32, torch.int64)
        or not value.is_contiguous()
        for value in buffers + layout_buffers
    ):
        return None
    try:
        kernel, next_power_of_2 = _graph_verify_receipt_kernel()
    except ImportError:
        return None

    size = 2 + capture_batch + batch
    if device_layout is not None:
        size += rows + window_words
    receipt = torch.empty(size, dtype=torch.int64, device=device)
    kernel[(1,)](
        *buffers,
        receipt,
        *layout_buffers,
        capture_batch,
        batch,
        rows,
        window_words,
        next_power_of_2(max(capture_batch, window_words)),
        device_layout is not None,
        num_warps=4,
    )
    return receipt, buffers + layout_buffers


def try_read_graph_verify_receipt(impl, inputs, layout: VerifyLayout) -> int | None:
    """Read the native dense replay receipt with one blocking device transfer."""
    launch = _launch_graph_verify_receipt(impl, inputs, layout.rows, len(layout.lengths))
    if launch is None:
        return None
    receipt, _ = launch
    capture_batch = impl.batch_size
    batch = len(layout.lengths)
    generation, mismatch, *metadata = receipt.tolist()
    if (
        mismatch
        or tuple(metadata[:capture_batch])
        != layout.lengths + (0,) * (capture_batch - batch)
        or tuple(metadata[capture_batch:]) != layout.prefixes
    ):
        raise ValueError(
            "acceptance input does not identify this completed verify replay"
        )
    return generation


@dataclass(frozen=True)
class _QueuedGraphVerifyReceipt:
    ready: torch.cuda.Event
    host: torch.Tensor
    # Keep the device result and all kernel inputs alive until consumption.
    device_buffers: tuple[torch.Tensor, ...]
    rows: int
    batch: int
    capture_batch: int
    window_width: int

    def read(self) -> tuple[int, VerifyLayout, list[int], list[list[int]]]:
        self.ready.synchronize()
        data = self.host.tolist()
        generation, mismatch = data[:2]
        prefix_start = 2 + self.capture_batch
        payload_start = prefix_start + self.batch
        layout = VerifyLayout.from_lengths_and_prefixes(
            tuple(data[2 : 2 + self.batch]),
            tuple(data[prefix_start:payload_start]),
            self.rows,
        )
        if mismatch:
            raise ValueError(
                "acceptance input does not identify this completed verify replay"
            )
        token_end = payload_start + self.rows
        tokens = data[payload_start:token_end]
        windows = [
            data[start : start + self.window_width]
            for start in range(token_end, len(data), self.window_width)
        ]
        return generation, layout, tokens, windows

    def discard(self) -> int:
        """Drain an abandoned copy and retire its generation."""
        self.ready.synchronize()
        return int(self.host[0])


def try_enqueue_graph_verify_receipt(
    impl, inputs, primary
) -> _QueuedGraphVerifyReceipt | None:
    """Enqueue a dense replay's checked layout and payload on the consumer stream."""
    if impl.mode != "dense" or inputs.is_fake_stream:
        return None
    lengths, prefixes = primary.input_lengths, primary.prefix_lengths
    if (
        not isinstance(lengths, torch.Tensor)
        or not isinstance(prefixes, torch.Tensor)
        or not lengths.is_cuda
        or not prefixes.is_cuda
        or lengths.ndim != 1
        or lengths.shape != prefixes.shape
    ):
        return None
    rows, batch = inputs.input_ids.numel(), lengths.numel()
    launch = _launch_graph_verify_receipt(
        impl, inputs, rows, batch, device_layout=(lengths, prefixes)
    )
    if launch is None:
        return None
    receipt, buffers = launch
    host = torch.empty(receipt.shape, dtype=receipt.dtype, device="cpu", pin_memory=True)
    host.copy_(receipt, non_blocking=True)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(receipt.device))
    return _QueuedGraphVerifyReceipt(
        ready,
        host,
        (receipt, *buffers),
        rows,
        batch,
        impl.batch_size,
        inputs.engram_token_windows.shape[1],
    )


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
    def __init__(
        self, cache, request_ids, layout, windows, tokens, *,
        call_id, max_backtracking, graph_journal=None, host_payload=None,
        batch_binding_checks=False, joint_binding_checks=None,
    ):
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
        host_tokens, host_windows = (
            (tokens.reshape(-1).cpu().tolist(), None)
            if host_payload is None
            else host_payload
        )
        if host_payload is not None and len(host_tokens) != layout.rows:
            raise ValueError("verify token/row layout is inconsistent")
        self.tokens = tuple(map(int, host_tokens))
        self.graph_journal = graph_journal
        self.batch_binding_checks = batch_binding_checks
        self.joint_binding_checks = joint_binding_checks
        self.prepared_joint_publication = None
        self.bindings = {} if graph_journal is not None else {
            key: (pool.tensor.data_ptr(), pool.table.clone()) for key, pool in cache.pools.items()
        }
        self.histories = []
        if cache.geometry.engram_enabled():
            if windows is None or tuple(windows.shape) != (layout.rows, cache.geometry.engram_max_ngram_size):
                raise ValueError("verify requires every scheduler Engram history window")
            host = windows.cpu().tolist() if host_payload is None else host_windows
            if host_payload is not None and (
                host is None
                or len(host) != layout.rows
                or any(len(row) != cache.geometry.engram_max_ngram_size for row in host)
            ):
                raise ValueError("verify requires every scheduler Engram history window")
            for start, length in zip(layout.starts, layout.lengths):
                history = host[start][1:]
                versions = []
                for row in range(start, start + length):
                    expected = [self.tokens[row], *history][:cache.geometry.engram_max_ngram_size]
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

    def _publish_eager(self, counts):
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
        return sum(v.positions.numel() for v in selected)

    def finish(self, request_ids, prefixes, accepted_rows):
        counts = self._validate(request_ids, prefixes, accepted_rows)
        if self.graph_journal is None:
            published = self._publish_eager(counts)
        else:
            bindings = None
            if self.batch_binding_checks:
                bindings = self.graph_journal.try_enqueue_binding_check(self.cache, len(counts))
            if bindings is None:
                self.graph_journal.validate_bindings(self.cache, self.layout)
            else:
                bindings.validate()
            published = self.graph_journal.publish(self.layout, counts)
        evidence = {
            "call": self.call_id, "request_ids": self.request_ids,
            "prefixes": self.layout.prefixes, "verified_rows": self.layout.lengths,
            "accepted_input_rows": counts,
            "accepted_ends": [p + n for p, n in zip(self.layout.prefixes, counts)],
            "accepted_input_token_ids": [list(self.tokens[start:start + n]) for start, n in zip(self.layout.starts, counts)],
            "maximum_backtracking": self.max_backtracking,
            "actual_backtracking": [width - n for width, n in zip(self.layout.lengths, counts)],
            "native_rows_published": published,
            "engram_committed_windows": [list(v[n - 1]) if n else None for v, n in zip(self.histories, counts)],
            "native_aliases_preserved": True, "status": "ACCEPTED",
        }
        self.abort()
        return evidence

    def abort(self):
        self.cache.journal.clear()
        self.closed = True

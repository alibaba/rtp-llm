"""Worker-local inspection receipts. Receipts never own embeddings or media."""

import concurrent.futures
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import CancelledError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict


class InspectionTickets:
    def __init__(self, ttl, capacity=65536):
        self._ttl = ttl
        self._capacity = capacity
        self._tickets = {}
        self._media_count = 0
        self._next_sweep = 0.0
        self._completed = OrderedDict()
        self._lock = threading.Lock()

    def open(self, entries, lifetime_s=0, request_id=0):
        now = time.monotonic()
        with self._lock:
            if now >= self._next_sweep:
                self._tickets = {
                    key: record
                    for key, record in self._tickets.items()
                    if record[0] > now
                }
                self._media_count = sum(
                    len(record[1]) for record in self._tickets.values()
                )
                while self._completed:
                    expires_at, _, _ = next(iter(self._completed.values()))
                    if expires_at > now:
                        break
                    self._completed.popitem(last=False)
                self._next_sweep = now + min(1, self._ttl)
            if self._media_count + len(entries) > self._capacity:
                raise FtRuntimeException(
                    ExceptionType.CONCURRENCY_LIMIT_ERROR, "ViT inspection tickets full"
                )
            ticket = uuid.uuid4().hex
            # None means this generation already had an authoritative approval.
            decisions = {
                item["key"]: (
                    {item["entry_generation"]: None}
                    if item.get("greennet_passed")
                    else {}
                )
                for item in entries
            }
            self._tickets[ticket] = (
                now + max(self._ttl, lifetime_s),
                decisions,
                request_id,
            )
            self._media_count += len(decisions)
            return ticket

    def _lookup(self, ticket, request_id):
        record = self._tickets.get(ticket)
        if record is None or record[0] <= time.monotonic() or record[2] != request_id:
            raise FtRuntimeException(
                ExceptionType.MM_PROCESS_ERROR,
                "Unknown or expired ViT inspection ticket",
            )
        return record[1]

    def validate(self, ticket, keys, request_id):
        with self._lock:
            decisions = self._lookup(ticket, request_id)
            if not all(key in decisions for key in keys):
                raise ValueError("Media does not belong to this inspection ticket")

    def bind(self, ticket, key, generation, decision, request_id=0):
        with self._lock:
            decisions = self._lookup(ticket, request_id)
            if key not in decisions:
                raise ValueError("Media does not belong to this inspection ticket")
            decisions[key].setdefault(generation, decision)

    def _finish(self, ticket, request_id, verdict):
        # Idempotent query retries retain only the small verdict, never futures.
        record = self._tickets.pop(ticket, None)
        if record is not None:
            self._media_count -= len(record[1])
        self._completed[ticket] = (time.monotonic() + self._ttl, verdict, request_id)
        # Concurrent waiters can finish the same receipt more than once.
        # A refreshed expiry belongs at the tail of the sweep order.
        self._completed.move_to_end(ticket)
        if len(self._completed) > self._capacity:
            self._completed.popitem(last=False)
        return verdict

    def wait(self, ticket, deadline, cancellation_event=None, request_id=0):
        observed = set()
        while True:
            with self._lock:
                completed = self._completed.get(ticket)
                if (
                    completed
                    and completed[0] > time.monotonic()
                    and completed[2] == request_id
                ):
                    return completed[1]
                decisions = self._lookup(ticket, request_id)
                if not decisions or not all(decisions.values()):
                    raise FtRuntimeException(
                        ExceptionType.MM_PROCESS_ERROR,
                        "Missing submitted media inspection",
                    )
                pending = {
                    f
                    for generations in decisions.values()
                    for f in generations.values()
                    if f is not None
                } - observed
                if not pending:
                    return self._finish(
                        ticket, request_id, GreenNetVerdict(passed=True)
                    )
            if cancellation_event is not None and cancellation_event.is_set():
                raise CancelledError("ViT inspection query cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ViT inspection deadline exceeded")
            done, _ = concurrent.futures.wait(
                pending,
                timeout=min(0.05, remaining),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                verdict = future.result()
                if not verdict.passed:
                    with self._lock:
                        return self._finish(ticket, request_id, verdict)
                observed.add(future)

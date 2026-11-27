# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run PACE work off the scheduler thread and return core receipts."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Protocol

from block_tree_cache import Operation, Receipt, RemoteObject, Result, Work, WorkKind
from vllm.logger import init_logger

logger = init_logger(__name__)

Key = tuple[int, bytes]


class RemoteStorage(Protocol):
    capacity: int

    def describe(self, keys: list[Key]) -> dict[Key, bytes]: ...

    def read_described(self, items: list[tuple[int, bytes, int, bytes]]) -> bool: ...

    def write(self, group: int, key: bytes, slot: int) -> bool: ...

    def close(self) -> None: ...


@dataclass
class _Pending:
    work: Work
    future: Future[Receipt]
    deadline: float
    timed_out: bool = False


class RemoteExecutor:
    def __init__(self, storage: RemoteStorage, config: dict) -> None:
        self.storage = storage
        self.timeout = float(config.get("lookup_timeout_s", 5))
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("lookup_timeout_s must be finite and positive")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="btc-pace"
        )
        self._pending: dict[Operation, _Pending] = {}
        self._unsafe: set[Operation] = set()

    def submit(self, work: Work) -> bool:
        if work.id in self._pending or work.id in self._unsafe:
            return False
        deadline = (
            monotonic() + self.timeout if work.kind != WorkKind.HOST_TO_REMOTE
            else float("inf")
        )
        self._pending[work.id] = _Pending(
            work, self._executor.submit(self._execute, work), deadline
        )
        return True

    def _execute(self, work: Work) -> Receipt:
        from btc_kvcm import BtcHostOwnershipUnknown, PaceBufferOwnershipUnknown

        (participant,) = work.participants
        try:
            if work.kind is WorkKind.REMOTE_QUERY:
                keys = [(page.position.group, page.key) for page in work.pages]
                proof = self.storage.describe(keys)
                objects = tuple(
                    RemoteObject(page.position, proof[key])
                    for page, key in zip(work.pages, keys, strict=True)
                    if key in proof
                )
                return Receipt(
                    work.id, participant, Result.SUCCESS, True, objects=objects
                )
            if work.kind is WorkKind.REMOTE_TO_HOST:
                ok = self.storage.read_described([
                    (p.position.group, p.key, p.target[0].slot, p.remote)
                    for p in work.pages
                ])
            else:
                assert work.kind is WorkKind.HOST_TO_REMOTE
                ok = all([
                    self.storage.write(p.position.group, p.key, p.source[0].slot)
                    for p in work.pages
                ])
            return Receipt(
                work.id, participant, Result.SUCCESS if ok else Result.FAILED, True,
                committed=ok and work.kind is WorkKind.HOST_TO_REMOTE,
            )
        except (BtcHostOwnershipUnknown, PaceBufferOwnershipUnknown):
            logger.exception("BTC remote operation %s lost buffer ownership", work.id)
            return Receipt(work.id, participant, Result.FAILED, False)
        except Exception:
            logger.exception("BTC remote operation %s failed", work.id)
            return Receipt(work.id, participant, Result.FAILED, True)

    def completed(self) -> list[Receipt]:
        now = monotonic()
        receipts = []
        for operation, pending in tuple(self._pending.items()):
            (participant,) = pending.work.participants
            if pending.future.done():
                receipt = pending.future.result()
                if pending.timed_out:
                    receipt = Receipt(
                        operation, participant, Result.TIMEOUT, receipt.safe
                    )
                if not receipt.safe:
                    self._unsafe.add(operation)
                receipts.append(receipt)
                del self._pending[operation]
            elif not pending.timed_out and now >= pending.deadline:
                cancelled = pending.future.cancel()
                receipts.append(
                    Receipt(operation, participant, Result.TIMEOUT, cancelled)
                )
                if cancelled:
                    del self._pending[operation]
                else:
                    pending.timed_out = True
        return receipts

    def stop(self) -> list[Receipt]:
        self._executor.shutdown(wait=True)
        return self.completed()

    def reject_unsubmitted(self, work: Work) -> list[Receipt]:
        if work.id in self._unsafe:
            return []
        return [
            Receipt(work.id, rank, Result.REJECTED, True) for rank in work.participants
        ]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.storage.close()

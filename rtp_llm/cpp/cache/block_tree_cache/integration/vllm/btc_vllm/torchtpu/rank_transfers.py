# SPDX-License-Identifier: Apache-2.0
"""TorchTPU's transport of native per-rank completion proofs."""

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

from vllm.logger import init_logger

logger = init_logger(__name__)

PageSegments = tuple[int, int, int, Sequence[tuple[int, int, int, int]]]


class RankResult(NamedTuple):
    ok: bool
    safe: bool


class RankTransfers:
    def __init__(self, runtime: Any, rank: int) -> None:
        self._runtime = runtime
        self._rank = rank
        # True means the coordinator already saw an unproven completion.
        self._pending: dict[int, bool] = {}
        self._finished: dict[int, RankResult] = {}

    def submit(self, job: int, load: bool, pages: Sequence[PageSegments]) -> None:
        from _btc_tpu_transfer import TpuDmaAdmissionError

        if not pages or job in self._pending or job in self._finished:
            raise ValueError(f"BTC rank received an empty or duplicate job {job}")
        try:
            self._runtime.submit(job, load, pages)
        except TpuDmaAdmissionError:
            logger.exception("BTC rank %d could not prepare job %d", self._rank, job)
            # Prepare runs before native WorkLoop admission.
            self._finished[job] = RankResult(ok=False, safe=True)
        else:
            self._pending[job] = False

    def wait(self, jobs: Iterable[int]) -> bool:
        success = True
        for job in jobs:
            if job in self._pending:
                result = RankResult(*self._runtime.wait(job))
                if not result.safe:
                    raise RuntimeError(f"BTC rank {self._rank} has no safety proof for {job}")
                self._finished[job] = result
                del self._pending[job]
            success &= self._finished[job].ok
        return success

    def get_finished(self) -> list[tuple[int, RankResult]]:
        for job, reported in tuple(self._pending.items()):
            raw = self._runtime.poll(job)
            if raw is None:
                continue
            result = RankResult(*raw)
            if result.safe:
                self._finished[job] = result
                del self._pending[job]
            elif not reported:
                self._finished[job] = result
                self._pending[job] = True
        results = list(self._finished.items())
        self._finished.clear()
        return results

    def stats(self) -> tuple[int, int, int]:
        return self._runtime.stats()

    def close(self) -> None:
        self._runtime.close()
        self._pending.clear()
        self._finished.clear()

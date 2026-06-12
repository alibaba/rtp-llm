from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.profiler

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalOutputPB
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
from rtp_llm.utils.grpc_util import trans_from_tensor

if TYPE_CHECKING:
    from rtp_llm.multimodal.mm_process_engine import MMWorkItem


def _maybe_tensor_to_list(tensor: Any, dim: int = 2) -> List[Any]:
    """Split a stacked tensor into a per-image list, or wrap a single tensor."""
    if tensor is None:
        return []
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if len(tensor.shape) > dim:
        return list(tensor)
    return [tensor]


def build_multimodal_output_pb(
    embeddings: List[torch.Tensor],
    position_ids: List[torch.Tensor],
    extra_input: List[torch.Tensor],
) -> MultimodalOutputPB:
    """Serialize embedding tensors into a MultimodalOutputPB."""
    if not embeddings:
        return MultimodalOutputPB()
    output_pb = MultimodalOutputPB(
        multimodal_embedding=trans_from_tensor(torch.concat(embeddings)),
        split_size=[e.shape[0] for e in embeddings],
    )
    if position_ids:
        output_pb.multimodal_pos_id.CopyFrom(
            trans_from_tensor(torch.concat(position_ids))
        )
    for extra in extra_input:
        output_pb.multimodal_extra_input.append(trans_from_tensor(extra))
    return output_pb


class _EmbeddingRequest:
    """A single caller's submission to the GPU batch scheduler."""

    __slots__ = ("work_items", "exception", "done", "cancelled")

    def __init__(self, work_items: List[MMWorkItem]):
        self.work_items = work_items
        self.exception: Optional[Exception] = None
        self.done = threading.Event()
        self.cancelled = False


# Sentinel pushed into _waiting to wake the executor thread on close().
_STOP = object()


class MMScheduler:

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        batch_wait_ms: int = 10,
        max_batch_size: int = 8,
        max_batch_tokens: int = 1048576,
    ):
        self._mm_part = mm_part
        self._batch_wait_ms = batch_wait_ms
        self._max_batch_size = max_batch_size
        self._max_batch_tokens = max_batch_tokens

        self._waiting: queue.Queue[_EmbeddingRequest] = queue.Queue()
        self._running: queue.Queue[_EmbeddingRequest] = queue.Queue()
        # A request popped from _waiting that would have overflowed the current
        # batch's work_item budget — carried over as the first request of the
        # next round so it is neither lost nor re-ordered behind newer arrivals.
        self._pending: Optional[_EmbeddingRequest] = None

        # Set by close(); stops the executor loop and rejects new submissions.
        self._stopped = threading.Event()

        self._executor = threading.Thread(
            target=self._executor_loop, daemon=True, name="mm-scheduler"
        )
        self._executor.start()

    def submit_and_wait(self, work_items: List[MMWorkItem]) -> None:
        if self._stopped.is_set():
            raise RuntimeError("MMScheduler is closed, request rejected")

        req = _EmbeddingRequest(work_items)
        req_tokens = self._request_tokens(req)
        if req_tokens > self._max_batch_tokens:
            raise ValueError(
                f"request token cost {req_tokens} exceeds "
                f"gpu_max_batch_tokens {self._max_batch_tokens}, "
                f"request rejected"
            )

        timeout_s = max(wi.mm_timeout_ms for wi in work_items) / 1000.0

        self._waiting.put(req)
        if not req.done.wait(timeout=timeout_s):
            req.cancelled = True
            raise TimeoutError(
                f"MMScheduler: embedding wait timeout after {timeout_s * 1000:.0f}ms"
            )
        if req.exception:
            raise req.exception

    def _executor_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                batch = self._collect_batch()
                if batch is None:
                    break
                for req in batch:
                    self._running.put(req)
                self._execute_batch(batch)
            except Exception as e:
                logging.error(f"MMScheduler: executor loop error: {e}", exc_info=True)
            finally:
                while not self._running.empty():
                    try:
                        self._running.get_nowait()
                    except queue.Empty:
                        break

    def _work_item_cost(self, wi: MMWorkItem) -> int:
        pr = wi.preprocess_result
        if isinstance(pr, (tuple, list)) and pr and isinstance(pr[0], torch.Tensor):
            return max(1, pr[0].shape[0])
        if isinstance(pr, torch.Tensor):
            return max(1, pr.shape[0])
        return 1

    def _request_tokens(self, req: _EmbeddingRequest) -> int:
        """Total token/patch cost of a request (sum over its work_items)."""
        return sum(self._work_item_cost(wi) for wi in req.work_items)

    def _collect_batch(self) -> Optional[List[_EmbeddingRequest]]:
        first = self._pending or self._waiting.get()
        self._pending = None
        if first is _STOP:
            return None
        batch = [first]
        n_tokens = self._request_tokens(first)

        deadline = time.monotonic() + self._batch_wait_ms / 1000.0
        while len(batch) < self._max_batch_size and n_tokens < self._max_batch_tokens:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._waiting.get(timeout=remaining)
            except queue.Empty:
                break
            if req is _STOP:
                # Re-queue so the next _collect_batch sees it; run what we have.
                self._waiting.put(_STOP)
                break
            if req.cancelled:
                continue
            req_tokens = self._request_tokens(req)
            if (
                len(batch) + 1 > self._max_batch_size
                or n_tokens + req_tokens > self._max_batch_tokens
            ):
                self._pending = req
                break
            batch.append(req)
            n_tokens += req_tokens

        return batch

    def _execute_batch(self, batch: List[_EmbeddingRequest]) -> None:
        """Execute the batched GPU forward and write results back.

        _collect_batch has already bounded the whole batch to within both budgets
        (request count <= max_batch_size, total tokens <= max_batch_tokens), and a
        request is never split — so every collected batch fits in a single forward.

        All-or-nothing: the batch is one forward, so on failure every request in
        it fails together. Isolating the culprit (per-request fallback) is not
        attempted — it would break the single-forward contract and failures are rare."""
        batch = [req for req in batch if not req.cancelled]
        if not batch:
            return
        try:
            all_items: List[Tuple[_EmbeddingRequest, MMWorkItem]] = []
            for req in batch:
                for wi in req.work_items:
                    all_items.append((req, wi))

            data_list = [wi.preprocess_result for _, wi in all_items]
            type_list = [wi.mm_type for _, wi in all_items]

            with torch.profiler.record_function("batched_embedding"):
                batch_outputs = self._mm_part.batched_embedding(data_list, type_list)

            for (req, wi), result in zip(all_items, batch_outputs):
                wi.embedding_result = result
                if wi.need_check_cache:
                    vit_emb_cache_.insert_cache(wi.cache_key, result)
        except Exception as e:
            logging.error(f"MMScheduler: batch forward failed: {e}", exc_info=True)
            for req in batch:
                req.exception = e
                req.done.set()
            return

        for req in batch:
            req.done.set()

    def close(self, timeout: float = 10.0) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._waiting.put(_STOP)
        self._executor.join(timeout=timeout)
        if self._executor.is_alive():
            logging.warning("MMScheduler: executor join exceeded %.0fs", timeout)

        exc = RuntimeError("MMScheduler closed before request completed")
        queued = [self._pending] if self._pending else []
        self._pending = None
        while not self._waiting.empty():
            try:
                req = self._waiting.get_nowait()
            except queue.Empty:
                break
            if req is not _STOP:
                queued.append(req)
        for req in queued:
            req.exception = exc
            req.done.set()

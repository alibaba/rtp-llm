"""Request-owned multimodal work, tensor results and cache publication."""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional, Tuple

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
    _current_cuda_streams,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.multimodal.multimodal_util import maybe_tensor_to_list
from rtp_llm.ops import (
    MultimodalInput,
    get_multimodal_feature_hash,
    get_multimodal_feature_hashes,
)
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


def _input_has_url(mm_input: MultimodalInput) -> bool:
    native_check = getattr(mm_input, "has_url", None)
    if callable(native_check):
        return bool(native_check())
    # Keep compatibility with Python-only input fixtures/providers.
    return mm_input.url != ""


def _acquire_input_cache(
    cache: MMEmbeddingCache, mm_input: MultimodalInput, *, require_key: bool = False
):
    if not require_key:
        return cache.try_acquire_lazy(mm_input.cache_key)
    key = mm_input.cache_key()
    state, entry = cache.try_acquire(key)
    return key, state, entry


def _validate_embedding_result(result: Any) -> None:
    """Reject tensors that cannot represent visual tokens before caching."""
    embeddings = result[0]
    if isinstance(embeddings, torch.Tensor):
        # Check stacked tensors before splitting: a zero leading dimension
        # would otherwise become an empty list and bypass per-tensor validation.
        embeddings = [embeddings]
    for index, embedding in enumerate(embeddings or []):
        if isinstance(embedding, torch.Tensor) and embedding.ndim == 0:
            raise ValueError(
                f"ViT returned scalar embedding at index {index}; "
                "expected a visual-token tensor"
            )
        if isinstance(embedding, torch.Tensor) and embedding.numel() == 0:
            raise ValueError(
                f"ViT returned empty embedding at index {index}: "
                f"shape={tuple(embedding.shape)}; expected non-empty visual tokens"
            )


@cuda_graph_gate.operation()
def _feature_hashes_from_result(result: Any) -> List[torch.Tensor]:
    """Build one sidecar hash tensor for every validated embedding tensor."""
    _validate_embedding_result(result)
    embeddings = maybe_tensor_to_list(result[0], ndim_threshold=2)
    return [get_multimodal_feature_hash(embedding) for embedding in embeddings]


@cuda_graph_gate.operation()
def prepare_batch_feature_hashes(work_items, results) -> None:
    """Coalesce eager CUDA hashes before publishing any work-item cache entry.

    Deferred async hashes and already computed hashes retain their old path.
    CPU results keep the single-result implementation. No embedding is copied
    or concatenated, and cache/inspection publication still belongs to each item.
    """
    pending, embeddings = [], []
    for item, result in zip(work_items, results):
        if item.defer_feature_hashes or item.feature_hashes is not None:
            continue
        _validate_embedding_result(result)
        values = maybe_tensor_to_list(result[0], ndim_threshold=2)
        if not values or not all(value.is_cuda for value in values):
            continue
        pending.append((item, len(values)))
        embeddings.extend(values)
    if len(embeddings) < 2:
        return
    hashes = get_multimodal_feature_hashes(embeddings)
    if len(hashes) != len(embeddings):
        raise RuntimeError("Batched feature hash output count mismatch")
    offset = 0
    for item, count in pending:
        item.feature_hashes = hashes[offset : offset + count]
        offset += count


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        extra_input: Optional[List[torch.Tensor]] = None,
        feature_hashes: Optional[List[torch.Tensor]] = None,
    ):
        self.embeddings = embeddings
        self.position_ids = position_ids if position_ids is not None else []
        # Model-specific extra input, one opaque flat 1-D tensor per image (e.g. deepstack).
        self.extra_input = extra_input if extra_input is not None else []
        self.feature_hashes = feature_hashes

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)}, embeddings_shape={[e.shape for e in self.embeddings]}, position_ids_shape={[p.shape for p in self.position_ids] if self.position_ids is not None else []}, extra_input_shape={[d.shape for d in self.extra_input] if self.extra_input is not None else []})"


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self,
        mm_inputs: List[MultimodalInput],
        mm_timeout_ms: Optional[int] = VitConfig.DEFAULT_MM_TIMEOUT_MS,
        embedding_cache: Optional[MMEmbeddingCache] = None,
        cache_claim: Optional[Tuple[str, MMEmbeddingCacheEntry, str]] = None,
        defer_cache_complete: bool = False,
        hash_key_cache: Optional[MMHashKeyCache] = None,
        defer_feature_hashes: bool = False,
        deadline: Optional[float] = None,
        require_cache_key: bool = False,
    ):
        self.deadline = deadline
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        # Resolve each input independently, then use the largest budget for the
        # shared batch so the worker matches the proxy deadline calculation.
        default_timeout_ms = (
            mm_timeout_ms
            if mm_timeout_ms is not None and mm_timeout_ms > 0
            else VitConfig.DEFAULT_MM_TIMEOUT_MS
        )
        self.mm_timeout_ms = max(
            (
                input.mm_preprocess_config.mm_timeout_ms
                if input.mm_preprocess_config.mm_timeout_ms > 0
                else default_timeout_ms
            )
            for input in self.mm_inputs
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.capture_inspection_sources = False
        self.inspection_sources: Optional[List[bytes]] = None
        self.inspection_request = None
        self.inspection_future = None
        self.preprocess_ready = False
        self.preprocess_submitted_at = 0.0
        self.preprocess_received_at = 0.0
        self.preprocess_ready_event = threading.Event()
        self.embedding_result: Optional[Any] = None
        self.feature_hashes: Optional[List[torch.Tensor]] = None
        self.work_estimate: Optional[MMWorkEstimate] = None
        self.need_check_cache = len(mm_inputs) == 1 and _input_has_url(mm_inputs[0])
        self.embedding_cache = embedding_cache
        self.hash_key_cache = hash_key_cache
        self.cache_key: Optional[str] = None
        self.cache_entry: Optional[MMEmbeddingCacheEntry] = None
        self.cache_state: Optional[str] = None
        self.defer_cache_complete = defer_cache_complete
        # Only cache-free async producers defer hashes to get_embedding_result.
        # None distinguishes uncaptured producing streams from CPU-only output.
        self.defer_feature_hashes = defer_feature_hashes
        self.embedding_producer_streams: Optional[List[Any]] = None

        if cache_claim is not None:
            if not self.need_check_cache:
                raise ValueError("cache_claim requires one non-empty multimodal input")
            self.cache_key, self.cache_entry, self.cache_state = cache_claim
            if self.cache_state == "complete":
                self.embedding_result = self.cache_entry.wait()
                if (
                    self.hash_key_cache is not None
                    and self.embedding_cache is not None
                    and self.embedding_cache.peek(self.cache_key) is self.cache_entry
                ):
                    self.feature_hashes = self.hash_key_cache.get(
                        self.cache_key, self.cache_entry.generation
                    )
        elif self.need_check_cache and self.embedding_cache is not None:
            self.cache_key, self.cache_state, self.cache_entry = _acquire_input_cache(
                self.embedding_cache, self.mm_inputs[0], require_key=require_cache_key
            )
            if self.cache_state == "complete":
                self.embedding_result = self.cache_entry.wait()
                if (
                    self.hash_key_cache is not None
                    and self.embedding_cache.peek(self.cache_key) is self.cache_entry
                ):
                    self.feature_hashes = self.hash_key_cache.get(
                        self.cache_key, self.cache_entry.generation
                    )

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None

    def remaining_timeout_s(self) -> float:
        timeout_s = self.mm_timeout_ms / 1000.0
        if self.deadline is not None:
            timeout_s = min(timeout_s, self.deadline - time.monotonic())
            if timeout_s <= 0:
                raise FtRuntimeException(
                    ExceptionType.GENERATE_TIMEOUT, "ViT request deadline exceeded"
                )
        return timeout_s

    @property
    def waiting_for_cache(self) -> bool:
        return self.cache_state == "in_progress" and self.embedding_result is None

    @property
    def should_preprocess(self) -> bool:
        return self.embedding_result is None and not self.waiting_for_cache

    def complete_cache(self, result: Any, force: bool = False) -> None:
        # Async cache-free producers defer hashing, but must fail empty output
        # immediately rather than publish it for a later result consumer.
        _validate_embedding_result(result)
        if self.defer_feature_hashes:
            if self.embedding_producer_streams is None:
                # Capture on the scheduler thread without synchronizing it.
                # Consumers using the same streams need no extra CUDA events.
                self.embedding_producer_streams = _current_cuda_streams(result)
        elif self.feature_hashes is None:
            self.feature_hashes = _feature_hashes_from_result(result)
        if (
            (self.defer_cache_complete and not force)
            or self.embedding_cache is None
            or self.cache_key is None
            or self.cache_entry is None
        ):
            return
        if self.defer_feature_hashes:
            self.cache_entry.producer_streams = self.embedding_producer_streams or []
        self.embedding_cache.complete(
            self.cache_key,
            self.cache_entry,
            result,
            before_publish=self._publish_cache_hashes,
        )

    def _publish_cache_hashes(self) -> None:
        if self.feature_hashes is not None and self.hash_key_cache is not None:
            self.hash_key_cache.put_for_entry(
                self.cache_key,
                self.feature_hashes,
                self.cache_entry,
                self.embedding_cache,
            )

    def fail_cache(self, error: Exception) -> None:
        if (
            self.embedding_cache is None
            or self.cache_key is None
            or self.cache_entry is None
            or self.cache_state != "miss"
        ):
            return
        self.embedding_cache.fail(self.cache_key, self.cache_entry, error)

"""GreenNet (content safety inspection) hook abstraction.

This is the open-source side surface. The real implementation lives under
``internal_source/rtp_llm/multimodal/greennet/`` and is selected at runtime
via ``has_internal_source()``. Without internal source the factory returns
a no-op provider so the call sites stay free of conditionals.

Lifetime contract:

* ``preprocess_and_submit(request, mm_inputs)`` returns a
  :class:`GreenNetHandle` immediately. The handle's ``rewritten_inputs`` are
  ready for downstream consumption (length must equal the input length so the
  C++ ``MultimodalProcessor`` tag-vs-mm-input check passes). The greennet
  side work runs in the background.
* The caller awaits ``handle.wait_result()`` before yielding the first model
  token. On violation the caller cancels the backend stream and raises
  ``FtRuntimeException(UNSAFE_INPUT_CONTENT)``.
* ``handle.cancel()`` is idempotent and must be called from the caller's
  ``finally`` block to drop any pending inspection work.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# Bound at RPC entry; worker processes receive an ordinary dictionary in the
# inspection request and never depend on inheriting the caller's context.
inspection_fields: ContextVar[Optional[dict]] = ContextVar(
    "inspection_fields", default=None
)


@contextmanager
def inspection_context(fields):
    token = inspection_fields.set(fields)
    try:
        yield
    finally:
        inspection_fields.reset(token)


def verdict_scope(fields):
    policy = {k: v for k, v in (fields or {}).items() if k != "x-dashscope-requestid"}
    if not policy:
        return b""
    # Cache entries keep only a digest, never the gateway header values.
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()


@dataclass(frozen=True)
class GreenNetWorkerFactory:
    create: Callable
    max_concurrency: int

    def __call__(self):
        return self.create()


@dataclass
class GreenNetVerdict:
    """Result returned by :meth:`GreenNetHandle.wait_result`.

    ``code`` follows the dashscope safety_result_parser mapping
    (1=Success, 2=DataInspectionFailed, 10/11/12=*ProcessError, ...). See
    ``internal_source/rtp_llm/multimodal/greennet/result.py``.
    """

    passed: bool
    code: int = 1
    message: str = ""


class GreenNetHandle:
    """Token returned by :meth:`GreenNetProvider.preprocess_and_submit`.

    ``rewritten_inputs`` is what the engine should use in place of the original
    ``mm_inputs``. For VIDEO entries the URL is rewritten to a
    ``frames-pack:base64,...`` form; for IMAGE entries the URL is rewritten to
    ``data:image/jpeg;base64,...``. Length is always preserved.
    """

    rewritten_inputs: List[Any]

    async def wait_result(self) -> GreenNetVerdict:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError


class GreenNetProvider:

    def routing_approval(self, key, fields):
        return False

    def remember_routing_approval(self, media, fields):
        pass

    def share_verdict_cache(self) -> Optional[str]:
        """Return a cache handle for ViT child processes when supported."""
        return None

    def attach_verdict_cache(self, handle: str) -> None:
        """Use the parent ViT service's verdict cache when supported."""

    def worker_factory(self):
        """Optional picklable CPU-only factory; never pass a live HTTP client.

        Providers without this capability keep the parent-loop source path.
        """
        return None

    def is_enabled(self) -> bool:
        """Effective enablement: internal source present AND the runtime flag
        on. When False, callers MUST skip greennet entirely (no preprocess,
        no inspect RPC) so the disabled path is byte-identical to pre-greennet
        behavior."""
        return False

    async def preprocess_and_submit(
        self, request: Any, mm_inputs: List[Any]
    ) -> GreenNetHandle:
        raise NotImplementedError

    async def reinspect_cached(self, request, mm_inputs) -> GreenNetHandle:
        """Only the verdict is consumed; providers may reuse a cached decision."""
        return await self.preprocess_and_submit(request, mm_inputs)

    async def submit_sources(self, request, mm_inputs, sources: List[bytes]):
        """Inspect encoded bytes returned by CPU preprocessing, without URLs.

        This internal path must never re-download or rewrite inference input.
        An unsupported provider fails closed instead of silently skipping audit.
        """
        raise NotImplementedError("GreenNet provider cannot inspect CPU sources")

    async def close(self):
        """Release connections owned by the current inspection event loop."""


class NoopGreenNetHandle(GreenNetHandle):
    """No-op handle used when the internal source is unavailable.

    The original mm_inputs are returned verbatim; ``wait_result`` resolves
    immediately to a passing verdict.
    """

    def __init__(self, mm_inputs: List[Any]):
        self.rewritten_inputs = list(mm_inputs)

    async def wait_result(self) -> GreenNetVerdict:
        return GreenNetVerdict(passed=True)

    def cancel(self) -> None:
        return None


class NoopGreenNetProvider(GreenNetProvider):
    def is_enabled(self) -> bool:
        return False

    async def preprocess_and_submit(
        self, request: Any, mm_inputs: List[Any]
    ) -> GreenNetHandle:
        return NoopGreenNetHandle(mm_inputs)


_PROVIDER: Optional[GreenNetProvider] = None


def get_greennet_provider() -> GreenNetProvider:
    """Return the cached singleton greennet provider.

    Resolution order:
      1. Internal source if available (``InternalGreenNetProvider`` under
         ``internal_source.rtp_llm.multimodal.greennet.provider``).
      2. NoopGreenNetProvider.
    """
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    from rtp_llm.utils.import_util import has_internal_source

    if has_internal_source():
        from internal_source.rtp_llm.multimodal.greennet.provider import (
            InternalGreenNetProvider,
        )

        _PROVIDER = InternalGreenNetProvider()
    else:
        _PROVIDER = NoopGreenNetProvider()
    return _PROVIDER


def greennet_enabled() -> bool:
    """Effective greennet enablement for the current process. Cheap to call
    (cached provider). Callers on the LLM/client side use this to skip the
    WaitGreenNetVerdict RPC entirely when greennet is off."""
    return get_greennet_provider().is_enabled()


def _reset_provider_for_testing() -> None:
    global _PROVIDER
    _PROVIDER = None

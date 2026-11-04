"""Versioned integration seam for V4.1 implementations. No accelerator imports.

The provider owns weight descriptors and model orchestration as well as kernels.
See docs/V41_BACKEND_CONTRACT.md for tensor and collective invariants.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from rtp_llm.config.v41_runtime import V41RuntimeConfig

CONTRACT_VERSION = 1
OP_NAMES = (
    "linear",
    "moe",
    "index_score",
    "sparse_attention",
    "encode_kv",
    "decode_kv",
    "engram_lookup",
    "mhc",
)


@dataclass(frozen=True)
class V41StepContext:
    """Token rows are rank-local; positions and request identity remain global."""

    positions: Tensor
    valid_mask: Tensor
    request_ids: Tensor
    owner_ranks: Tensor
    request_offsets: Tensor
    parallel_context: object


@dataclass(frozen=True)
class V41QuantizedTensor:
    values: Tensor
    scales: Tensor | None
    format: str  # bf16 / fp8_32x32_ue8m0 / fp4_ue8m0


# Linear weights and Indexer Q/K share the same logical quantization boundary.
V41Weight = V41QuantizedTensor


@dataclass(frozen=True)
class V41MHCResult:
    hidden: Tensor
    post_mix: Tensor | None = None
    residual_mix: Tensor | None = None


@dataclass(frozen=True)
class V41CacheView:
    pools: Mapping[str, Tensor]
    block_tables: Mapping[str, Tensor]
    # dtype, stride, block size, ratio, source layer, CP placement, SWA range,
    # and compressor phase belong to the canonical A3/A5 descriptors.
    descriptors: Mapping[str, object]


class V41BackendOps(Protocol):
    def linear(
        self, x: Tensor, weight: V41Weight, *, bias: Tensor | None = None
    ) -> Tensor: ...

    def moe(
        self,
        x: Tensor,
        expert_ids: Tensor,
        expert_weights: Tensor,
        *,
        layer_id: int,
        context: V41StepContext,
    ) -> Tensor: ...

    def index_score(
        self,
        q: V41QuantizedTensor,
        k: V41QuantizedTensor,
        head_weights: Tensor,
        *,
        layer_id: int,
        candidate_blocks: Tensor,
        key_positions: Tensor,
        context: V41StepContext,
    ) -> Tensor: ...

    def sparse_attention(
        self,
        q: Tensor,
        selected_blocks: Tensor,
        *,
        cache: V41CacheView,
        layer_id: int,
        context: V41StepContext,
    ) -> Tensor: ...

    def encode_kv(self, kv: Tensor, *, pool_tag: str, descriptor: object) -> Tensor: ...

    def decode_kv(
        self, packed: Tensor, *, pool_tag: str, descriptor: object, out: Tensor
    ) -> Tensor: ...

    def engram_lookup(
        self, rows: Tensor, *, layer_id: int, context: V41StepContext
    ) -> Tensor: ...

    def mhc(
        self,
        residual: Tensor,
        branch: Tensor | None,
        *,
        phase: str,
        parameters: Mapping[str, Tensor],
        context: V41StepContext,
        state: V41MHCResult | None = None,
    ) -> V41MHCResult: ...


@dataclass(frozen=True)
class V41Backend:
    name: str
    strategy: str
    ops: V41BackendOps
    model_cls: type
    weight_cls: type
    check_capabilities: Callable[[V41RuntimeConfig], None]
    contract_version: int = CONTRACT_VERSION

    def validate(self):
        if self.name not in ("ppu", "reference"):
            raise ValueError(f"unsupported V4.1 provider {self.name!r}")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError(f"V4.1 backend contract mismatch: {self.contract_version}")
        missing = [
            name
            for name in OP_NAMES
            if not callable(getattr(self.ops, name, None))
            or getattr(getattr(self.ops, name), "__func__", getattr(self.ops, name))
            is getattr(V41BackendOps, name)
        ]
        if missing or not callable(self.check_capabilities):
            raise ValueError(
                f"incomplete V4.1 backend {self.name}: operations={missing}"
            )
        if not isinstance(self.model_cls, type) or not isinstance(
            self.weight_cls, type
        ):
            raise ValueError(
                "V4.1 backend must provide model and weight descriptor classes"
            )
        if self.name == "ppu":
            from rtp_llm.config.v41_runtime import PPU_STRATEGY

            if self.strategy != PPU_STRATEGY:
                raise ValueError(f"unapproved V4.1 PPU strategy: {self.strategy}")
        return self

    def check_runtime(self, runtime):
        self.validate()
        runtime.validate()
        if self.check_capabilities(runtime) is not None:
            raise RuntimeError(
                "V4.1 capability check must raise on failure and return None on success"
            )


_backends: dict[str, V41Backend] = {}
_backend_lock = threading.RLock()


def register_v41_backend(backend: V41Backend):
    backend.validate()
    with _backend_lock:
        previous = _backends.get(backend.name)
        if previous is not None and previous is not backend:
            raise RuntimeError(f"V4.1 backend {backend.name} is already registered")
        _backends[backend.name] = backend


def get_v41_backend(name: str) -> V41Backend:
    with _backend_lock:
        return _get_v41_backend(name)


def _get_v41_backend(name: str) -> V41Backend:
    if name not in ("ppu", "reference"):
        raise ValueError(f"unsupported V4.1 provider {name!r}")
    if name not in _backends:
        module_name = f"rtp_llm.models_py.modules.dsv41.{name}.backend"
        try:
            provider = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name not in (module_name, module_name.rsplit(".", 1)[0]):
                raise
            raise RuntimeError(
                f"V4.1 {name} backend is not integrated: missing {module_name}. "
                "Required: A1 coordination, A2 Linear/MoE weights, A3 Attention/KV, "
                "A4 Engram/mHC and A5 PD bindings. No CUDA/MegaMoE or CPU fallback."
            ) from error
        backend = provider.get_backend()
        if backend.name != name:
            raise ValueError(
                f"V4.1 provider identity mismatch: {backend.name} != {name}"
            )
        register_v41_backend(backend)
    return _backends[name]

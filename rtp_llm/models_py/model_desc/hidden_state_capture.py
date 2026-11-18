from collections.abc import Callable, Sequence
from typing import Any

import torch

from rtp_llm.ops.compute_ops import PyModelOutputs

CanonicalHiddenHook = Callable[
    [torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]


class CaptureContext:
    """Model-instance capture plan and forward-local capture state.

    A configured model owns one inactive root context. Ordinary forwards reuse
    that object without allocating capture storage. Capture-enabled forwards
    derive an active context whose captured tensors are local to that forward.
    """

    __slots__ = (
        "_ordered_layer_ids",
        "_layer_slots",
        "_canonical_layer",
        "_canonical_final",
        "_captured_hidden_states",
        "_capture_error",
        "_supported",
        "_enabled",
    )

    def __init__(
        self,
        *,
        ordered_layer_ids: tuple[int, ...] = (),
        layer_slots: dict[int, int] | None = None,
        canonical_layer: CanonicalHiddenHook | None = None,
        canonical_final: CanonicalHiddenHook | None = None,
        captured_hidden_states: list[torch.Tensor | None] | None = None,
        supported: bool = False,
        enabled: bool = False,
    ) -> None:
        self._ordered_layer_ids = ordered_layer_ids
        self._layer_slots = layer_slots or {}
        self._canonical_layer = canonical_layer
        self._canonical_final = canonical_final
        self._captured_hidden_states = captured_hidden_states
        self._capture_error: str | None = None
        self._supported = supported
        self._enabled = enabled

    @classmethod
    def unsupported(cls) -> "CaptureContext":
        return _UNSUPPORTED_CAPTURE_CONTEXT

    @classmethod
    def configured(
        cls,
        layer_ids: Sequence[int] | None,
        canonical_layer: CanonicalHiddenHook,
        canonical_final: CanonicalHiddenHook,
    ) -> "CaptureContext":
        ordered_layer_ids = tuple(layer_ids or ())
        return cls(
            ordered_layer_ids=ordered_layer_ids,
            layer_slots={
                layer_id: slot for slot, layer_id in enumerate(ordered_layer_ids)
            },
            canonical_layer=canonical_layer,
            canonical_final=canonical_final,
            supported=True,
        )

    @classmethod
    def passthrough(cls) -> "CaptureContext":
        """Declare capability for a wrapper that transports packed outputs."""
        return cls(supported=True)

    @property
    def supported(self) -> bool:
        return self._supported

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._ordered_layer_ids

    def for_forward(self, requested: bool) -> "CaptureContext":
        if not requested or not self._ordered_layer_ids:
            return self
        if self._canonical_final is None:
            raise RuntimeError(
                "hidden-state capture was requested from a passthrough-only model"
            )
        context = CaptureContext(
            ordered_layer_ids=self._ordered_layer_ids,
            layer_slots=self._layer_slots,
            canonical_layer=self._canonical_layer,
            canonical_final=self._canonical_final,
            captured_hidden_states=[None] * len(self._ordered_layer_ids),
            supported=True,
            enabled=True,
        )
        if self._canonical_layer is None:
            context._record_error(
                RuntimeError("active capture context is missing canonical layer hook")
            )
        return context

    def wants_layer(self, layer_id: int) -> bool:
        return (
            self._enabled
            and self._capture_error is None
            and layer_id in self._layer_slots
        )

    def _record_error(self, error: Exception) -> None:
        # Execution/collective failures can invalidate subsequent model work.
        # Device names and unsupported dtype/layout errors alone do not: packing
        # a CPU auxiliary tensor with a CUDA final is an isolated capture error.
        fatal_types = tuple(
            error_type
            for error_type in (
                MemoryError,
                getattr(torch, "OutOfMemoryError", None),
                getattr(torch, "AcceleratorError", None),
                getattr(torch.distributed, "DistError", None),
            )
            if error_type is not None
        )
        message = str(error).lower()
        if isinstance(error, fatal_types) or any(
            marker in message
            for marker in (
                "out of memory",
                "illegal memory access",
                "illegal address",
                "invalid device pointer",
                "misaligned address",
                "device-side assert",
                "device side assert",
                "launch failure",
                "launch failed",
                "launch timed out",
                "launch_failure",
                "launch_failed",
                "cudaerrorillegaladdress",
                "cuda_error_illegal_address",
                "hiperrorillegaladdress",
                "cudaerrormemoryallocation",
                "hiperroroutofmemory",
                "cublas_status_alloc_failed",
                "cusparse_status_alloc_failed",
                "cudnn_status_alloc_failed",
                "hipblas_status_alloc_failed",
                "rocblas_status_memory_error",
                "cublas_status_execution_failed",
                "cusparse_status_execution_failed",
                "cudnn_status_execution_failed",
                "hipblas_status_execution_failed",
                "rocblas_status_internal_error",
                "nccl error",
                "collective failed",
                "collective failure",
                "collective operation timeout",
                "collective timed out",
            )
        ):
            raise error
        self._capture_error = f"{type(error).__name__}: {error}"
        self._captured_hidden_states = None

    def capture_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> None:
        if not self.wants_layer(layer_id):
            return
        slot = self._layer_slots[layer_id]
        try:
            canonical_layer = self._canonical_layer
            captured_hidden_states = self._captured_hidden_states
            if canonical_layer is None or captured_hidden_states is None:
                raise RuntimeError("active capture context is missing canonical hooks")
            captured_hidden_states[slot] = canonical_layer(hidden_states, residual)
        except Exception as error:
            self._record_error(error)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        fmha_params: Any = None,
    ) -> PyModelOutputs:
        canonical_final = self._canonical_final
        if canonical_final is None:
            if self._enabled:
                raise RuntimeError(
                    "active capture context is missing canonical final hook"
                )
            return PyModelOutputs(hidden_states, fmha_params)
        final_hidden_states = canonical_final(hidden_states, residual)
        if not self._enabled:
            return PyModelOutputs(final_hidden_states, fmha_params)

        # Final computation above is serving work: never catch or replace it.
        outputs = PyModelOutputs(final_hidden_states, fmha_params)
        if self._capture_error is None:
            try:
                captured_hidden_states = self._captured_hidden_states
                if captured_hidden_states is None:
                    raise RuntimeError(
                        "active capture context has no forward-local storage"
                    )
                missing_layer_ids = [
                    layer_id
                    for layer_id, captured in zip(
                        self._ordered_layer_ids, captured_hidden_states
                    )
                    if captured is None
                ]
                if missing_layer_ids:
                    raise RuntimeError(
                        "hidden-state capture did not visit configured layers "
                        f"{missing_layer_ids}"
                    )
                outputs.captured_hidden_states = torch.cat(
                    [
                        captured
                        for captured in captured_hidden_states
                        if captured is not None
                    ]
                    + [final_hidden_states],
                    dim=-1,
                )
            except Exception as error:
                self._record_error(error)
        outputs.capture_error = self._capture_error
        return outputs


_UNSUPPORTED_CAPTURE_CONTEXT = CaptureContext()


__all__ = ["CanonicalHiddenHook", "CaptureContext"]

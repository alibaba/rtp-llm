from collections.abc import Callable, Sequence

import torch

from rtp_llm.models_py.utils.profiler import (
    _NOOP_RECORD_FUNCTION_RANGE,
    record_function_ranges_enabled,
)
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
        "_supported",
        "_enabled",
        "_capture_error",
        "_profile_enabled",
        "_profile_noop",
        "_record_function_fast",
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
        self._supported = supported
        self._enabled = enabled
        self._capture_error = ""
        self._profile_enabled = False
        self._profile_noop = _NOOP_RECORD_FUNCTION_RANGE
        self._record_function_fast = None
        if enabled and record_function_ranges_enabled():
            # Snapshot before DSV4 suppresses nested ranges. Optional private APIs
            # must not prevent capture on torch builds without FUNCTION ranges.
            profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
            if callable(profiler_enabled) and profiler_enabled():
                torch_profiler = getattr(torch._C, "_profiler", None)
                record_function_fast = getattr(
                    torch_profiler, "_RecordFunctionFast", None
                )
                if callable(record_function_fast):
                    self._record_function_fast = record_function_fast
                    self._profile_enabled = True

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
        if self._canonical_layer is None or self._canonical_final is None:
            raise RuntimeError(
                "hidden-state capture was requested from a passthrough-only model"
            )
        return CaptureContext(
            ordered_layer_ids=self._ordered_layer_ids,
            layer_slots=self._layer_slots,
            canonical_layer=self._canonical_layer,
            canonical_final=self._canonical_final,
            captured_hidden_states=[None] * len(self._ordered_layer_ids),
            supported=True,
            enabled=True,
        )

    @property
    def failed(self) -> bool:
        return bool(self._capture_error)

    @property
    def capture_error(self) -> str:
        return self._capture_error

    def wants_layer(self, layer_id: int) -> bool:
        # This is the frozen forward plan, not snapshot health. Kimi SP must
        # still execute every selected-layer collective after a local failure.
        return self._enabled and layer_id in self._layer_slots

    def capture_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> None:
        if not self._enabled or self.failed:
            return
        slot = self._layer_slots.get(layer_id)
        if slot is None:
            return
        try:
            canonical_layer = self._canonical_layer
            captured_hidden_states = self._captured_hidden_states
            if canonical_layer is None or captured_hidden_states is None:
                raise ValueError("active capture context is missing canonical hooks")
            with (
                self._record_function_fast(
                    f"hidden_state_capture.layer(layer={layer_id})"
                )
                if self._record_function_fast is not None
                else self._profile_noop
            ):
                captured_hidden_states[slot] = canonical_layer(hidden_states, residual)
        except (
            AttributeError,
            LookupError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            self._fail(error, hidden_states, residual)

    def _fail(self, error: Exception, *tensors: torch.Tensor | None) -> None:
        # Device RuntimeErrors can report asynchronous kernel/collective faults.
        # Without a typed recoverable device error, fail closed; never infer GPU
        # health from message text or introduce a synchronization to probe it.
        fatal_types = tuple(
            error_type
            for name in ("OutOfMemoryError", "AcceleratorError")
            if isinstance(error_type := getattr(torch, name, None), type)
        )
        if isinstance(error, fatal_types) or (
            isinstance(error, RuntimeError)
            and any(
                tensor is not None and tensor.device.type != "cpu" for tensor in tensors
            )
        ):
            raise error
        if not self._capture_error:
            self._capture_error = f"{type(error).__name__}: {error}"
        self._captured_hidden_states = None

    def finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> PyModelOutputs:
        canonical_final = self._canonical_final
        if canonical_final is None:
            if self._enabled:
                raise RuntimeError(
                    "active capture context is missing canonical final hook"
                )
            return PyModelOutputs(hidden_states)
        final_hidden_states = canonical_final(hidden_states, residual)
        if not self._enabled:
            return PyModelOutputs(final_hidden_states)

        outputs = PyModelOutputs(final_hidden_states)
        if not self.failed:
            try:
                captured_hidden_states = self._captured_hidden_states
                if captured_hidden_states is None or len(captured_hidden_states) != len(
                    self._ordered_layer_ids
                ):
                    raise ValueError(
                        "active capture context has invalid forward-local storage"
                    )
                missing_layer_ids = [
                    layer_id
                    for layer_id, captured in zip(
                        self._ordered_layer_ids, captured_hidden_states
                    )
                    if captured is None
                ]
                if missing_layer_ids:
                    raise ValueError(
                        "hidden-state capture did not visit configured layers "
                        f"{missing_layer_ids}"
                    )
                for captured in captured_hidden_states:
                    if (
                        not isinstance(captured, torch.Tensor)
                        or captured.shape != final_hidden_states.shape
                        or captured.dtype != final_hidden_states.dtype
                        or captured.device != final_hidden_states.device
                    ):
                        raise ValueError(
                            "captured hidden state does not match canonical final tensor"
                        )
                with (
                    self._record_function_fast("hidden_state_capture.pack")
                    if self._record_function_fast is not None
                    else self._profile_noop
                ):
                    outputs.captured_hidden_states = torch.cat(
                        captured_hidden_states + [final_hidden_states], dim=-1
                    )
            except (
                AttributeError,
                LookupError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                self._fail(error, final_hidden_states)
        outputs.capture_error = self._capture_error
        return outputs


_UNSUPPORTED_CAPTURE_CONTEXT = CaptureContext()


__all__ = ["CanonicalHiddenHook", "CaptureContext"]

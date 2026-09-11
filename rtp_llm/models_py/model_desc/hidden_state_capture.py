from collections.abc import Callable, Sequence

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

    def wants_layer(self, layer_id: int) -> bool:
        return self._enabled and layer_id in self._layer_slots

    def capture_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> None:
        if not self._enabled:
            return
        slot = self._layer_slots.get(layer_id)
        if slot is None:
            return
        canonical_layer = self._canonical_layer
        captured_hidden_states = self._captured_hidden_states
        if canonical_layer is None or captured_hidden_states is None:
            raise RuntimeError("active capture context is missing canonical hooks")
        captured_hidden_states[slot] = canonical_layer(hidden_states, residual)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> PyModelOutputs:
        canonical_final = self._canonical_final
        if canonical_final is None:
            raise RuntimeError("capture context has no canonical final hook")
        final_hidden_states = canonical_final(hidden_states, residual)
        if not self._enabled:
            return PyModelOutputs(final_hidden_states)

        captured_hidden_states = self._captured_hidden_states
        if captured_hidden_states is None:
            raise RuntimeError("active capture context has no forward-local storage")
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
        packed_hidden_states = torch.cat(
            [captured for captured in captured_hidden_states if captured is not None]
            + [final_hidden_states],
            dim=-1,
        )
        return PyModelOutputs(packed_hidden_states)


_UNSUPPORTED_CAPTURE_CONTEXT = CaptureContext()


__all__ = ["CanonicalHiddenHook", "CaptureContext"]

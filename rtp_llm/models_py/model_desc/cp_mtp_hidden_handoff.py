from typing import Optional

import torch


class CpMtpHiddenHandoff:
    """Keep GLM's normalized rank-local output before C++ CP gathers last rows."""

    def initialize(self, init_resource) -> bool:
        initialized = super().initialize(init_resource)
        cp = getattr(self.parallelism_config, "prefill_cp_config", None)
        self._cp_mtp_handoff_enabled = bool(
            init_resource.is_speculative
            and not init_resource.is_decode_role
            and cp is not None
            and cp.is_enabled()
        )
        self._cp_mtp_hidden = None
        if self._cp_mtp_handoff_enabled:
            reference = next(iter(self.weight.global_weights.values()))
            # Publish capability during construction, without reserving a full
            # max-sequence buffer. Forward retains its own output tensor.
            self._cp_mtp_hidden = torch.empty(
                (0, int(self.config.hidden_size)),
                dtype=reference.dtype,
                device=reference.device,
            )
        return initialized

    def has_mtp_hidden_buffer(self) -> bool:
        return getattr(self, "_cp_mtp_handoff_enabled", False)

    def capture_cp_mtp_hidden(self, hidden: torch.Tensor) -> None:
        if self.has_mtp_hidden_buffer():
            self._cp_mtp_hidden = hidden

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        if not self.has_mtp_hidden_buffer():
            return super().get_mtp_target_hidden_states(num_tokens)
        hidden = self._cp_mtp_hidden
        requested = int(num_tokens)
        if requested < 0:
            return hidden
        if requested > hidden.shape[0]:
            raise RuntimeError(
                f"requested {requested} CP MTP rows from {hidden.shape[0]} local rows"
            )
        return hidden[:requested]

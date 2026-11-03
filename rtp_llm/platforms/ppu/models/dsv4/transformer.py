"""PPU hidden-dimension shard handling."""

import torch

from rtp_llm.platforms.ppu.models.dsv4.impl.transformer import V4Transformer
from rtp_llm.platforms.ppu.models.dsv4.impl.tp_norm import tp_gather_hidden, tp_rms_norm


class PpuV4Transformer(V4Transformer):
    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and restore the replicated global hidden dimension."""

        hidden = self.embed(input_ids)
        if self.args.tp_size > 1 and hidden.shape[-1] != self.args.dim:
            hidden = tp_gather_hidden(hidden, tp_size=self.args.tp_size)
        if hidden.shape[-1] != self.args.dim:
            raise ValueError(
                f"DSV4 embedding hidden size must be {self.args.dim}, "
                f"got {hidden.shape[-1]}"
            )
        return hidden

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final global RMSNorm and restore the full hidden dimension."""

        normalized = tp_rms_norm(
            self.norm,
            x,
            tp_size=self.args.tp_size,
            tp_rank=self.args.tp_rank,
            global_dim=self.args.dim,
        )
        if normalized.shape[-1] == self.args.dim:
            return normalized
        return tp_gather_hidden(normalized, tp_size=self.args.tp_size)

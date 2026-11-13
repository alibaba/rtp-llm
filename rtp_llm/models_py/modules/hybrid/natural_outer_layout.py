"""Chronological physical-row view of the shared padded Zigzag metadata."""

from types import SimpleNamespace

import torch

from .query_layout import QueryLayout


class NaturalOuterLayout(QueryLayout):
    """Reuse QueryLayout's padding/hole semantics with an identity permutation.

    Natural outer shards split the concatenation of *padded* requests. Compact
    attention rows still contain the original request lengths and valid holes.
    This adapter does not extend end-to-end support to zero-length requests.
    """

    def __init__(self, zigzag_layout: QueryLayout, buffers=None):
        if zigzag_layout.all_valid:
            # Request boundaries and validity are unchanged by the physical
            # row order. Share their read-only storage for this forward round.
            for name in (
                "size",
                "rank",
                "device",
                "lengths",
                "prefixes",
                "physical_rows",
                "local_rows",
                "true_rows",
                "all_valid",
                "natural_all_valid",
                "padding_mask",
                "natural_valid",
                "cu_host",
                "cu_device",
                "lengths_host",
                "lengths_device",
            ):
                setattr(self, name, getattr(zigzag_layout, name))
            self.restore = torch.arange(
                self.physical_rows, dtype=torch.long, device=self.device
            )
            self.natural_to_rankmajor = self.restore
            self.rankmajor_requests = zigzag_layout.rankmajor_requests.index_select(
                0, zigzag_layout.restore
            )
            self.rankmajor_positions = zigzag_layout.rankmajor_positions.index_select(
                0, zigzag_layout.restore
            )
            start = self.rank * self.local_rows
            self.local_valid = self.natural_valid[start : start + self.local_rows]
            self.local_requests = self.rankmajor_requests[
                start : start + self.local_rows
            ]
            self.local_positions = self.rankmajor_positions[
                start : start + self.local_rows
            ]
            self._buffers = {} if buffers is None else buffers
            self._natural_attention_views = {}
            return
        info = SimpleNamespace(
            prefill_actual_input_lengths_cpu=zigzag_layout.lengths_host.clone(),
            prefill_prefix_lengths_cpu=torch.tensor(
                zigzag_layout.prefixes, dtype=torch.int32
            ),
            prefill_qkv_restore_indice=torch.arange(
                zigzag_layout.physical_rows, device=zigzag_layout.device
            ),
            prefill_qkv_padding_mask=zigzag_layout.padding_mask.clone(),
        )
        super().__init__(
            info,
            zigzag_layout.size,
            zigzag_layout.rank,
            zigzag_layout.device,
            valid_mask=(
                None
                if zigzag_layout.natural_all_valid
                else zigzag_layout.natural_valid.clone()
            ),
            buffers=buffers,
        )

    def prepare_kda_buffers(self, projection_width, output_width, *, quantized=False):
        if not self.all_valid:
            super().prepare_kda_buffers(
                projection_width, output_width, quantized=quantized
            )

    def gather_natural(self, projected):
        if self.all_valid:
            return projected
        return super().gather_natural(projected)

    def scatter_rankmajor(self, values, scales=None):
        if self.all_valid:
            return values, scales
        return super().scatter_rankmajor(values, scales)

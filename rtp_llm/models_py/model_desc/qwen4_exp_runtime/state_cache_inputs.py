"""Qwen's FULL state-page views over the graph runner's kernel block tables."""

from collections.abc import Mapping
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_ROPE_POS_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
)

_STATE_TAGS = frozenset(
    (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG, INDEXER_STATE_TAG, INDEXER_ROPE_POS_TAG)
)


class _GraphStateInputs:
    """Model-local metadata view; the original native inputs remain graph-owned."""

    def __init__(self, inputs, table):
        self._inputs = inputs
        self.kv_cache_block_id_device = table
        # Graph replay refreshes the device table, not a physical host mirror.
        self.kv_cache_block_id = None

    def __getattr__(self, name):
        return getattr(self._inputs, name)


def state_page_table(
    kernel_table: torch.Tensor, blocks_per_physical: int, *, out=None
) -> torch.Tensor:
    """Invert FULL block expansion without changing the state pool's geometry."""
    if (
        not isinstance(kernel_table, torch.Tensor)
        or kernel_table.ndim != 2
        or kernel_table.dtype != torch.int32
        or kernel_table.numel() == 0
        or blocks_per_physical <= 0
    ):
        raise RuntimeError(
            "Qwen state cache requires a nonempty int32 kernel block table"
        )
    if blocks_per_physical == 1:
        return kernel_table
    width = kernel_table.shape[1] // blocks_per_physical * blocks_per_physical
    if width == 0:
        raise RuntimeError("Qwen state kernel table cannot hold one physical page")
    # Padding width can exceed this group's complete expanded blocks. Preserve
    # null block 0, and materialize a contiguous table for the QSA writer ABI.
    return torch.div(
        kernel_table[:, :width:blocks_per_physical],
        blocks_per_physical,
        rounding_mode="floor",
        out=out,
    ).contiguous()


def prepare_state_cache_inputs(groups, kv_cache, *, workspace=None) -> Any:
    """Normalize the three model-owned FULL state chains once per forward.

    Their kernel maps are ordinary physical-block expansions: unlike SWA or
    LINEAR execution maps, they never use cyclic or shared chunk remapping.
    Eager/PD inputs retain their native physical tables and host-mirror contract.
    A caller supplying scratch must lease it through all readers; every call
    still projects the current table and binds the current native input owner.
    """
    if kv_cache is None or not isinstance(groups, Mapping):
        return groups
    result = None
    for tag, inputs in groups.items():
        if tag not in _STATE_TAGS or not inputs.is_cuda_graph:
            continue
        physical = int(kv_cache.get_seq_size_per_block(tag))
        kernel = int(kv_cache.get_kernel_seq_size_per_block(tag))
        if kernel <= 0 or physical <= 0 or physical % kernel:
            raise RuntimeError(f"Qwen state cache {tag!r} has invalid block geometry")
        source = inputs.kv_cache_kernel_block_id_device
        factor = physical // kernel
        if not isinstance(source, torch.Tensor):
            state_page_table(source, factor)  # Preserve the table validation error.
        binding = None if workspace is None else workspace.get(tag)
        layout = (source.shape, source.stride(), source.dtype, source.device, factor)
        if binding is not None and binding[0] == layout:
            view = binding[1]
            table = state_page_table(
                source,
                factor,
                out=view.kv_cache_block_id_device if factor > 1 else None,
            )
            view._inputs = inputs
            view.kv_cache_block_id_device = table
        else:
            table = state_page_table(source, factor)
            view = _GraphStateInputs(inputs, table)
            if workspace is not None:
                workspace[tag] = (layout, view)
        if result is None:
            result = dict(groups)
        result[tag] = view
    return groups if result is None else result

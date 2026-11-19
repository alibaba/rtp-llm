# SPDX-License-Identifier: Apache-2.0
"""Fused logical-to-physical expert selection for eager PPU R32 Prefill."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["size", "replica_offset"])
    def _prefill_r32_remap_kernel(
        ids, mapping, counts, output, size, replica_offset,
        SLOTS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        present = index < size
        expert = tl.load(ids + index, present, other=-1)
        valid = present & (expert >= 0) & (expert < 384)
        expert = expert.to(tl.int32)
        replicas = tl.load(counts + expert, valid, other=1).to(tl.uint32)
        row = (index // SLOTS).to(tl.uint32)
        rotation = (row * 23333 + replica_offset.to(tl.uint32)).to(tl.uint32)
        replica = (rotation % replicas).to(tl.int32)
        physical = tl.load(mapping + expert * 33 + replica, valid, other=-1)
        tl.store(output + index, physical.to(tl.int64), present)

    return triton, _prefill_r32_remap_kernel


def try_prefill_r32_remap(ids, mapping, counts, replica_offset):
    """Map authenticated gate IDs using the live, validated EplbLayer tables.

    The caller admits only the static R32 eager Prefill dispatch and supplies
    gate IDs from the logical domain or negative sentinels. All negatives map
    to -1, and every slot of a token uses the same uint32 rotation.
    """
    if (
        not ids.is_cuda
        or ids.dtype != torch.int64
        or ids.ndim != 2
        or ids.shape[1] not in (6, 8)
        or ids.numel() >= 2**31
        or not ids.is_contiguous()
        or ids.is_neg()
        or ids.is_conj()
        or mapping.dtype != torch.int32
        or mapping.shape != (384, 33)
        or counts.dtype != torch.int32
        or counts.shape != (384,)
        or mapping.device != ids.device
        or counts.device != ids.device
        or not mapping.is_contiguous()
        or not counts.is_contiguous()
        or mapping.is_neg()
        or counts.is_neg()
        or not 0 <= replica_offset < 8
        or torch.is_grad_enabled()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None

    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    if not is_ppu() or torch.cuda.get_device_capability(ids.device) != (8, 9):
        return None
    if ids.numel() == 0:
        return torch.empty_like(ids)
    try:
        triton, kernel = _triton_kernel()
    except ImportError:
        return None

    output = torch.empty_like(ids)
    block = 256
    kernel[(triton.cdiv(ids.numel(), block),)](
        ids, mapping, counts, output, ids.numel(), replica_offset,
        SLOTS=ids.shape[1], BLOCK=block, num_warps=4,
    )
    return output

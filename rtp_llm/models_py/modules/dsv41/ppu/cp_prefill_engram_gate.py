"""Independent original-CP admission for the existing masked Engram gate.

The actual block mask remains an operand, including image/padding exclusions.
The one-use PPUSharedEngram prefetch flag changes no lookup rounds or ENP
collectives. A closed prefetch receipt protects the current mask/hash/weight
owners; it does not certify arbitrary CP tensors or omit a known mask.
"""

from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import _source_tensor_key
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value

ENV = "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_GATE"


def _raw(value):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.requires_grad
        and not value.is_neg()
        and not value.is_conj()
    )


def _device_supported(hidden):
    return (
        hidden.is_cuda
        and torch.version.hip is None
        and _quant.is_ppu()
        and torch.cuda.get_device_capability(hidden.device) == (8, 9)
        and not torch.cuda.is_current_stream_capturing()
    )


def _unobserved(block):
    from torch.nn.modules import module as module_impl

    names = (
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
    )
    if any(getattr(module_impl, "_global" + name, {}) for name in names):
        return False
    return isinstance(block, nn.Module) and all(
        not any(getattr(module, name, {}) for name in names)
        for module in block.modules()
    )


def enabled(state, block, hidden):
    """Metadata-only CP admission, separate from the unchanged local/DP guard."""
    if (
        env_value(ENV, "0") != "1"
        or torch.is_grad_enabled()
        or not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or getattr(state, "compacted", True)
        or getattr(block, "layer_id", None) not in (1, 14)
    ):
        return False
    cp = getattr(state, "context", None)
    phase, valid = getattr(state, "_cp_rows_phase", None), getattr(
        state, "_cp_full_rows_valid", None
    )
    if (
        not isinstance(cp, CPContext)
        or cp.cp_size != 8
        or not cp.kv_cache_sharded
        or cp is not getattr(state, "original_context", None)
        or cp.gather_restore_positions is not None
        or not callable(phase)
        or not phase()
        or not callable(valid)
        or not valid()
        or not _unobserved(block)
    ):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedEngram

    engram = getattr(block, "engram", None)
    if (
        type(engram) is not PPUSharedEngram
        or engram.layer_id != block.layer_id
        or getattr(engram, "_graph_state", None) is not None
        or not any(
            owner is block
            for owner in getattr(getattr(state, "v4", None), "layers", ())
        )
    ):
        return False
    mask, hashes = block.engram_token_mask, block.engram_hashes
    q, k, wkv = engram.q_weight, engram.k_weight, getattr(engram.wkv, "weight", None)
    return (
        all(
            _raw(value) and value.device == hidden.device
            for value in (hidden, mask, hashes, q, k, wkv)
        )
        and hidden.dtype == torch.bfloat16
        and hidden.shape == (cp.chunk_length, 4, 5120)
        and hidden.is_contiguous()
        and mask.dtype == torch.bool
        and mask.shape == (cp.chunk_length,)
        and mask.stride(0) > 0
        and hashes.dtype == torch.int64
        and hashes.ndim == 2
        and hashes.shape[0] == cp.chunk_length
        and hashes.is_contiguous()
        and q.dtype in (torch.bfloat16, torch.float32)
        and k.dtype in (torch.bfloat16, torch.float32)
        and q.shape == k.shape == (4, 5120)
        and q.is_contiguous()
        and k.is_contiguous()
        and wkv.dtype == torch.bfloat16
        and wkv.ndim == 2
        and wkv.shape[0] == 25600
        and wkv.shape[1] > 0
        and wkv.is_contiguous()
        and _device_supported(hidden)
    )


def _owners(block, hidden):
    engram = block.engram
    return (
        hidden,
        block.engram_hashes,
        block.engram_token_mask,
        engram.q_weight,
        engram.k_weight,
        engram.wkv.weight,
    )


@dataclass(frozen=True)
class PrefetchReceipt:
    state: object
    block: object
    engram: object
    context: object
    birth: object
    owners: tuple
    keys: tuple


def prefetch_receipt(state, block, hidden):
    """Snapshot the exact operands selected for this one synchronous prefetch."""
    if not enabled(state, block, hidden):
        raise RuntimeError("CP Engram gate lost its admitted original-row operands")
    owners = _owners(block, hidden)
    return PrefetchReceipt(
        state,
        block,
        block.engram,
        state.context,
        state._cp_full_rows_birth,
        owners,
        tuple(_source_tensor_key(value) for value in owners),
    )


def check_prefetch(state, block, hidden, receipt):
    """Reject mutation/rebinding before handing the one-use flag to Block."""
    owners = _owners(block, hidden)
    if (
        not isinstance(receipt, PrefetchReceipt)
        or receipt.state is not state
        or receipt.block is not block
        or receipt.engram is not block.engram
        or receipt.context is not state.context
        or receipt.birth is not state._cp_full_rows_birth
        or any(old is not new for old, new in zip(receipt.owners, owners))
        or receipt.keys != tuple(_source_tensor_key(value) for value in owners)
        or not enabled(state, block, hidden)
    ):
        block.engram.clear()
        raise RuntimeError("CP Engram gate operands changed during prefetch")

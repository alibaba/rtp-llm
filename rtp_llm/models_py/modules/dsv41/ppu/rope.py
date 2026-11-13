"""Common PPU RoPE ownership, frequency lookup and signed-zero contract.

Frequencies are either token-aligned rows, or a full table with explicit
positions whose provenance/range is established by the caller. Permission
to overwrite is explicit; storage shared with any live input stays read-only.
Native provider errors propagate. Unsupported metadata uses fresh storage.
"""

import os

import torch

from rtp_llm.models_py.modules.dsv41.ppu import fused_ops


def can_overwrite(value, *live_inputs):
    """Check disjoint storage, not Python object identity or view status.

    The caller must additionally own the value's lifetime. Unknown aliases
    cannot be discovered here; list every input that must remain readable.
    Even nonoverlapping views of a live storage conservatively decline.
    """
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.is_neg()
        or value.is_conj()
        or value.requires_grad
        or torch.is_grad_enabled()
    ):
        return False
    storage = value.untyped_storage().data_ptr()
    return all(
        not isinstance(other, torch.Tensor)
        or other.device != value.device
        or (
            other.layout == torch.strided
            and other.untyped_storage().data_ptr() != storage
        )
        for other in live_inputs
    )


def _validate(value, frequencies, positions, valid_mask):
    if value.ndim not in (2, 3) or frequencies.ndim != 2:
        raise ValueError("RoPE requires [T,D] or [T,H,D] and a 2D frequency table")
    half = frequencies.shape[-1]
    if not half or value.shape[-1] < half * 2:
        raise ValueError(
            f"rope tail {half * 2} exceeds the tensor's last dim {value.shape[-1]}"
        )
    if positions is None:
        if frequencies.shape[0] != value.shape[0]:
            raise ValueError(
                f"freqs rows {frequencies.shape[0]} != token rows {value.shape[0]}"
            )
    elif (
        positions.shape != (value.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != frequencies.device
    ):
        raise ValueError("RoPE positions must be integer rows on the frequency device")
    if valid_mask is not None and (
        valid_mask.shape != (value.shape[0],)
        or valid_mask.dtype != torch.bool
        or valid_mask.device != value.device
    ):
        raise ValueError("RoPE valid mask must be bool [T] on the input device")


def _native_input(value, frequencies):
    return (
        value.is_cuda
        and not torch.is_grad_enabled()
        and value.layout == frequencies.layout == torch.strided
        and not any(
            tensor.requires_grad or tensor.is_neg() or tensor.is_conj()
            for tensor in (value, frequencies)
        )
    )


def try_rotate_inplace(
    value,
    frequencies,
    *,
    inverse=False,
    valid_mask=None,
    preserve_signed_zero=True,
    positions=None,
    live_inputs=(),
):
    """Try an explicitly owned rotation; decline before any mutation.

    The legacy signed-zero keyword is accepted for caller compatibility;
    common production dispatch always preserves the sign of zero.
    """
    _validate(value, frequencies, positions, valid_mask)
    if not can_overwrite(value, frequencies, positions, valid_mask, *live_inputs):
        return None
    if not _native_input(value, frequencies):
        return None
    from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import try_rope_

    return try_rope_(
        value,
        frequencies,
        inverse=inverse,
        valid_mask=valid_mask,
        preserve_signed_zero=True,
        positions=positions,
    )


def _reference(value, frequencies, inverse):
    """Portable original FP32 complex multiply, one cast and a fresh output."""
    tokens, half = value.shape[0], frequencies.shape[-1]
    rope_dim = half * 2
    # view_as_complex requires adjacent scalar pairs, including for FP32
    # strided inputs whose float() conversion otherwise retains the stride.
    tail = value[..., -rope_dim:].float().contiguous()
    tail = tail.reshape(tokens, *value.shape[1:-1], half, 2)
    pairs = torch.view_as_complex(tail)
    table = frequencies.conj() if inverse else frequencies
    broadcast = (tokens,) + (1,) * (pairs.ndim - 2) + (half,)
    rotated = pairs * table.reshape(broadcast)
    real = torch.view_as_real(rotated).reshape(value.shape[:-1] + (rope_dim,))
    output = value.clone()
    output[..., -rope_dim:] = real.to(value.dtype)
    return output


def try_rotate_fresh(value, frequencies, inverse=False, *, positions=None):
    """Try the nonaliasing provider without changing its eligibility."""
    _validate(value, frequencies, positions, None)
    if (
        value.is_neg()
        or value.is_conj()
        or frequencies.is_neg()
        or frequencies.is_conj()
    ):
        return None
    # Keep the provider's allocation-free CPU admission observable to legacy
    # callers; accelerator values additionally enforce the common grad guard.
    if value.is_cuda and not _native_input(value, frequencies):
        return None
    return fused_ops.try_rope_tail(
        value,
        frequencies,
        inverse,
        **({"positions": positions} if positions is not None else {}),
    )


def rotate(
    value,
    frequencies,
    inverse=False,
    *,
    positions=None,
    inplace=False,
    live_inputs=(),
    valid_mask=None,
    allow_fused=None,
):
    """Rotate an owned value when safe; otherwise return a disjoint result.

    A full-table lookup that the native provider declines is gathered on the
    current device stream, including each Graph replay's dynamic positions.
    No GPU position scan, host synchronization or post-launch retry is used.
    Existing explicit fusion switches remain selection constraints.
    """
    _validate(value, frequencies, positions, valid_mask)
    if inplace:
        output = try_rotate_inplace(
            value,
            frequencies,
            inverse=inverse,
            valid_mask=valid_mask,
            positions=positions,
            live_inputs=live_inputs,
        )
        if output is not None:
            return output
    if valid_mask is not None:
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        value = torch.where(valid_mask.reshape(shape), value, torch.zeros_like(value))
        # This mask result is fresh, even if the original input had live aliases.
        if inplace:
            output = try_rotate_inplace(
                value,
                frequencies,
                inverse=inverse,
                positions=positions,
                live_inputs=live_inputs,
            )
            if output is not None:
                return output
    if allow_fused is None:
        allow_fused = os.environ.get("RTP_V41_PPU_FUSED_BF16_ROPE", "0") == "1"
    if allow_fused:
        output = try_rotate_fresh(value, frequencies, inverse, positions=positions)
        if output is not None:
            return output
    selected = frequencies if positions is None else frequencies[positions.long()]
    if positions is not None:
        # Direct lookup has narrower stream/capture eligibility. A fresh
        # device gather can still use the established row-aligned providers.
        if inplace:
            output = try_rotate_inplace(
                value,
                selected,
                inverse=inverse,
                live_inputs=(frequencies, positions, *live_inputs),
            )
            if output is not None:
                return output
        if allow_fused:
            output = try_rotate_fresh(value, selected, inverse)
            if output is not None:
                return output
    return _reference(value, selected, inverse)


def joint_supported(q, kv, frequencies):
    """Use the same gate for selection receipts and the joint launch."""
    return (
        fused_ops.is_supported(q, kv, frequencies)
        and _native_input(q, frequencies)
        and not kv.is_neg()
        and not kv.is_conj()
    )


def rotate_qk(q, kv, frequencies):
    """Preserve the joint provider's two fresh outputs and read-only inputs."""
    if joint_supported(q, kv, frequencies):
        return fused_ops.qk_rope(q, kv, frequencies)
    return rotate(q, frequencies), rotate(kv, frequencies)

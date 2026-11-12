"""Fused BF16 rotary embedding for shared PPU Attention.

The inputs have already landed in BF16 after Q-b and KV normalization. This
kernel intentionally has no normalization, quantization, or cache side effects.
Its reference is ``ppu.prefill._rope_tail`` on each input independently.
Existing ``_rope_only_triton`` has a single in-place output; the fused public
QKV cache kernel additionally normalizes/quantizes/stores with a different
rounding boundary. Neither supplies this two-output, nonaliasing contract.

Triton imports and JIT construction are lazy. Experimental selection and
model/phase eligibility belong to the shared Attention caller.
"""

from functools import lru_cache

import torch


def is_supported(q, kv, freqs):
    """Check the complete tensor contract without allocation or device work."""
    if not all(isinstance(value, torch.Tensor) for value in (q, kv, freqs)):
        return False
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[-1] == 512
        and q.shape[1] > 0
        and kv.dtype == torch.bfloat16
        and kv.shape == (q.shape[0], 512)
        and freqs.dtype == torch.complex64
        and freqs.ndim == 2
        and freqs.shape[0] == q.shape[0]
        and 0 < freqs.shape[1] <= 256
        and not freqs.is_conj()
        and kv.device == q.device
        and freqs.device == q.device
        and all(value.is_contiguous() for value in (q, kv, freqs))
        and not any(value.requires_grad for value in (q, kv, freqs))
    )


@lru_cache(maxsize=1)
def _complex_multiply():
    import triton
    import triton.language as tl

    @triton.jit
    def multiply(real, imag, cosine, sine):
        # PPU Torch rounds the sine products before the explicit FP32 FMAs.
        rotated_real = tl.fma(real, cosine, -(imag * sine))
        rotated_imag = tl.fma(imag, cosine, real * sine)
        # Preserve the IEEE sign of zero when both products are zero. The PPU
        # lowering of the multiply/add chain can otherwise canonicalize it.
        r = real.to(tl.int32, bitcast=True)
        i = imag.to(tl.int32, bitcast=True)
        c = cosine.to(tl.int32, bitcast=True)
        s = sine.to(tl.int32, bitcast=True)
        real_zero = ((real == 0) | (cosine == 0)) & ((imag == 0) | (sine == 0))
        imag_zero = ((imag == 0) | (cosine == 0)) & ((real == 0) | (sine == 0))
        real_sign = (r ^ c) & ~(i ^ s) & -2147483648
        imag_sign = (i ^ c) & (r ^ s) & -2147483648
        real_bits = tl.where(real_zero & (rotated_real == 0), real_sign, rotated_real.to(tl.int32, bitcast=True))
        imag_bits = tl.where(imag_zero & (rotated_imag == 0), imag_sign, rotated_imag.to(tl.int32, bitcast=True))
        return real_bits.to(tl.float32, bitcast=True), imag_bits.to(tl.float32, bitcast=True)

    return multiply


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    complex_multiply = _complex_multiply()

    @triton.jit
    def rotate_row(
        source,
        destination,
        frequencies,
        TOKEN,
        DIM: tl.constexpr,
        HALF_ROPE: tl.constexpr,
        INVERSE: tl.constexpr,
    ):
        # Copy the NoPE channels along with the rotated tail.
        pair = tl.arange(0, DIM // 2)
        real = tl.load(source + pair * 2).to(tl.float32)
        imag = tl.load(source + pair * 2 + 1).to(tl.float32)
        frequency_pair = pair - (DIM // 2 - HALF_ROPE)
        rotates = frequency_pair >= 0
        address = frequencies + TOKEN * (2 * HALF_ROPE) + frequency_pair * 2
        cosine = tl.load(address, mask=rotates, other=1.0)
        sine = tl.load(address + 1, mask=rotates, other=0.0)
        if INVERSE:
            sine = (sine.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)
        rotated_real, rotated_imag = complex_multiply(real, imag, cosine, sine)
        tl.store(destination + pair * 2, tl.where(rotates, rotated_real, real))
        tl.store(destination + pair * 2 + 1, tl.where(rotates, rotated_imag, imag))

    @triton.jit
    def fused_qk_rope(
        Q,
        KV,
        Frequencies,
        QOut,
        KVOut,
        HEADS: tl.constexpr,
        HALF_ROPE: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1).to(tl.int64)
        if head < HEADS:
            offset = (token * HEADS + head) * 512
            rotate_row(Q + offset, QOut + offset, Frequencies, token, 512, HALF_ROPE, False)
        else:
            offset = token * 512
            rotate_row(KV + offset, KVOut + offset, Frequencies, token, 512, HALF_ROPE, False)

    @triton.jit
    def fused_rope_tail(
        source, frequencies, out,
        HEADS: tl.constexpr, DIM: tl.constexpr, HALF_ROPE: tl.constexpr, INVERSE: tl.constexpr,
        positions=None,
    ):
        row = tl.program_id(0).to(tl.int64)
        frequency_row = row // HEADS
        if positions is not None:
            frequency_row = tl.load(positions + frequency_row).to(tl.int64)
        rotate_row(source + row * DIM, out + row * DIM, frequencies, frequency_row, DIM, HALF_ROPE, INVERSE)

    return fused_qk_rope, fused_rope_tail


def qk_rope(q, kv, freqs):
    """Return new Q and KV tensors after one launch, preserving BF16 landings.

    Unsupported input fails before output allocation or launch. There is no
    post-launch retry. The caller must select its serial reference in advance.
    Frequency rows already correspond to each token's absolute position.
    """
    if not is_supported(q, kv, freqs):
        raise ValueError("shared PPU Q/K RoPE requires contiguous BF16/complex64 inputs")
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    if q.shape[0] == 0:
        return q_out, kv_out
    # view_as_real is a zero-copy view of the existing complex64 frequency
    # rows. Both outputs are fresh and disjoint from all completed inputs.
    qk_kernel, _ = _kernel()
    qk_kernel[(q.shape[0], q.shape[1] + 1)](
        q,
        kv,
        torch.view_as_real(freqs),
        q_out,
        kv_out,
        HEADS=q.shape[1],
        HALF_ROPE=freqs.shape[1],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, kv_out


def _rope_positions_supported(x, freqs, positions):
    """Metadata only: caller proves 0 <= positions[i] < freqs.shape[0]."""
    values = (x, freqs, positions)
    if not (not torch.is_grad_enabled()
            and all(isinstance(value, torch.Tensor) and value.layout == torch.strided
                    and not value.is_neg() and not value.is_conj()
                    and not value.requires_grad for value in values)
            and x.is_cuda and x.dtype == torch.bfloat16 and x.ndim in (2, 3)
            and x.shape[-1] in (128, 512) and x.is_contiguous()
            and freqs.device == x.device and freqs.dtype == torch.complex64
            and freqs.ndim == 2 and freqs.shape[0] > 0 and freqs.shape[1] == 32
            and freqs.is_contiguous() and positions.device == x.device
            and positions.dtype in (torch.int32, torch.int64)
            and positions.shape == (x.shape[0],) and positions.is_contiguous()
            and len({value.untyped_storage().data_ptr() for value in values}) == 3):
        return False
    try:
        return (not torch.cuda.is_current_stream_capturing()
                and torch.cuda.current_device() == x.device.index
                and torch.cuda.current_stream(x.device) == torch.cuda.default_stream(x.device))
    except (AssertionError, RuntimeError):
        return False


def try_rope_tail(x, freqs, inverse, *, positions=None):
    """Return the fused nonaliasing BF16 result for supported row layouts.

    With positions, freqs is a full complex64 [F,32] table. The caller owns
    the original row provenance and numeric position-range proof. This helper
    does not scan GPU values or retry a failed launch.
    """
    if positions is not None:
        supported = _rope_positions_supported(x, freqs, positions)
    else:
        supported = (
        x.is_cuda and x.dtype == torch.bfloat16 and x.ndim in (2, 3)
        and x.shape[-1] in (128, 512) and x.is_contiguous()
        and freqs.dtype == torch.complex64 and freqs.ndim == 2
        and freqs.shape[0] == x.shape[0] and 0 < freqs.shape[1] <= x.shape[-1] // 2
        and freqs.is_contiguous() and not freqs.is_conj() and freqs.device == x.device
        and not x.requires_grad and not freqs.requires_grad
        )
    if not supported:
        return None
    out = torch.empty_like(x)
    if x.numel():
        heads = x.shape[1] if x.ndim == 3 else 1
        _, rope_kernel = _kernel()
        options = {} if positions is None else {"positions": positions}
        rope_kernel[(x.shape[0] * heads,)](
            x, torch.view_as_real(freqs), out,
            heads, x.shape[-1], freqs.shape[1], inverse,
            num_warps=4, enable_fp_fusion=False,
            **options,
        )
    return out

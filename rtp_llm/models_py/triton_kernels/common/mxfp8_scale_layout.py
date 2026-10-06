"""Small MXFP8 producer scale allocation; no DeepGEMM import/JIT dependency."""

import torch
import triton
import triton.language as tl


@triton.jit
def checked_ue8m0_exponent(scale, active):
    """Preserve the old packer's exponent-only contract, including failures.

    A debug-only device_assert is insufficient: DeepGEMM traps for any sign
    or mantissa bit even in release builds. Preserve bits, check active lanes,
    then extract the byte. The side effect must survive compiler DCE.
    """
    bits = scale.to(tl.uint32, bitcast=True)
    invalid = active & ((bits & 0x807FFFFF) != 0)
    checked_bits = tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; @p trap; mov.b32 $0, $2; }",
        "=r,r,r",
        [invalid.to(tl.uint32), bits],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    return ((checked_bits >> 23) & 255).to(tl.uint8)


def allocate_mxfp8_tma_scale(rows: int, width: int, device):
    """Return final int32 TMA scale and its contiguous byte-write view.

    DeepGEMM's (1, 32) scale ABI packs four exponent bytes per int32 and
    aligns the leading column stride to 16 bytes. Each producer group owns
    one byte, at ``(group // 4) * aligned_rows * 4 + row * 4 + group % 4``.
    Both views retain the underlying storage; never view the transpose as u8.
    """
    assert rows >= 0 and width > 0 and width % 128 == 0
    aligned_rows = (rows + 3) // 4 * 4
    storage = torch.empty(
        (width // 128, aligned_rows), dtype=torch.int32, device=device
    )
    return storage.transpose(0, 1)[:rows], storage.view(torch.uint8)

"""Small-batch BF16 target verification for the TP2 Qwen4 vocabulary shard on gfx942."""
import logging

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_verify_head(hidden, weight, output, rows: tl.constexpr, vocab: tl.constexpr,
                      width: tl.constexpr, tile_n: tl.constexpr, tile_k: tl.constexpr):
    tile = tl.program_id(0)
    m = tl.arange(0, 16)
    n = tl.arange(0, tile_n)
    k = tl.arange(0, tile_k)
    acc = tl.full((16, tile_n), 0, tl.float32)
    for step in range(width // tile_k):
        x = tl.load(hidden + m[:, None] * width + step * tile_k + k[None, :], m[:, None] < rows, 0)
        w = tl.load(weight + ((tile * (width // tile_k) + step) * tile_n + n[:, None]) * tile_k + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(output + m[:, None] * vocab + tile * tile_n + n[None, :], acc, m[:, None] < rows)


def pack_bf16_verify_head(weight):
    if torch.cuda.get_device_properties(weight.device).gcnArchName.split(":")[0] != "gfx942":
        return None
    vocab, width = weight.shape
    # Each MFMA tile reads its weights consecutively along K.
    try:
        return weight.reshape(vocab // 128, 128, width // 64, 64).permute(0, 2, 1, 3).contiguous()
    except torch.OutOfMemoryError:
        logging.getLogger(__name__).warning("Insufficient memory for packed BF16 verify head, using torch.mm")
        return None


def project_bf16_verify_head(hidden, packed):
    blocks, _, tile_n, tile_k = packed.shape
    rows, width = hidden.shape
    vocab = blocks * tile_n
    logits = hidden.new_empty((rows, vocab))
    _bf16_verify_head[(blocks,)](
        hidden, packed, logits, rows, vocab, width, tile_n, tile_k,
        num_warps=2, matrix_instr_nonkdim=16, num_stages=1,
    )
    return logits

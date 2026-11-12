"""Thin local-Prefill adapter for pinned SGLang's native selection pipeline.

The caller owns phase/rollout admission and proves device length values are in
range. Dense scores are FP32, sparse scores BF16, and neither path sanitizes,
canonicalizes or finite-filters. In particular the native SG NaN/Inf behavior,
newest-block convention and arbitrary tie order are deliberately retained.
Imports/JIT/launch failures after admission propagate, without a serial retry.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant

TOKEN_K = 512
BLOCK_SIZE = 8


def _raw(value):
    return (isinstance(value, torch.Tensor) and value.layout == torch.strided
            and not value.is_neg() and not value.is_conj()
            and not value.requires_grad)


def _overlap(left, right):
    if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
        return True

    def span(value):
        count = ((value.shape[0] - 1) * value.stride(0) + value.shape[1]
                 if value.ndim == 2 else value.numel())
        return value.data_ptr(), value.data_ptr() + count * value.element_size()

    a, b = span(left), span(right)
    return a[0] < b[1] and b[0] < a[1]


def _device_supported(scores):
    if not scores.is_cuda or torch.version.hip is not None or not _quant.is_ppu():
        return False
    try:
        return (torch.cuda.current_device() == scores.device.index
                and torch.cuda.get_device_capability(scores.device) == (8, 9)
                and not torch.cuda.is_current_stream_capturing()
                and torch.cuda.current_stream(scores.device)
                == torch.cuda.default_stream(scores.device))
    except (AssertionError, RuntimeError):
        return False


def _base(scores, lengths, dtype, maximum):
    return (not torch.is_grad_enabled() and _raw(scores) and _raw(lengths)
            and scores.dtype == dtype and scores.ndim == 2
            and 1 <= scores.shape[0] <= 4096
            and 8 <= scores.shape[1] <= maximum and scores.shape[1] % 8 == 0
            and scores.stride(1) == 1 and scores.stride(0) >= scores.shape[1]
            and scores.stride(0) % 8 == 0
            and scores.data_ptr() % (8 * scores.element_size()) == 0
            and lengths.device == scores.device and lengths.dtype == torch.int32
            and lengths.shape == (scores.shape[0],) and lengths.is_contiguous()
            and not _overlap(scores, lengths))


def _output(value, scores, lengths, width):
    return (value is None or (
        _raw(value) and value.dtype == torch.int32 and value.device == scores.device
        and value.shape == (scores.shape[0], width) and value.is_contiguous()
        and value.data_ptr() % 16 == 0
        and not _overlap(value, scores) and not _overlap(value, lengths)))


def is_supported_dense(scores, ends, out=None, *, block_topk=512, blocks_out=None):
    """Metadata-only gate. Caller proves 0 <= ends[row] <= score width.

    ``blocks_out`` is the actual publication buffer at L20, not an intermediate
    copy. C512 is private selection, C2048 is shared publication and selection.
    Both writable outputs must be exclusive; source scores/ends are read-only.
    """
    if block_topk not in (512, 2048) or not _base(scores, ends, torch.float32, 131072):
        return False
    if not (_output(out, scores, ends, TOKEN_K)
            and _output(blocks_out, scores, ends, block_topk)):
        return False
    if out is not None and blocks_out is not None and _overlap(out, blocks_out):
        return False
    return _device_supported(scores)


def is_supported_sparse(scores, valid_lens, blocks, out=None):
    """Caller proves valid_lens <= width and valid slots' page IDs are >= 0.

    ``blocks`` is the already prepared native SG page table, including its
    original count/padding convention. IDs are translated by the native kernel
    as page*8+lane; this adapter does not add a second remap or finite filter.
    """
    if not _base(scores, valid_lens, torch.bfloat16, 16384):
        return False
    if not (_raw(blocks) and blocks.dtype == torch.int32
            and blocks.device == scores.device
            and blocks.shape == (scores.shape[0], scores.shape[1] // 8)
            and blocks.stride(1) == 1 and blocks.stride(0) >= blocks.shape[1]
            and not _overlap(blocks, scores) and not _overlap(blocks, valid_lens)
            and _output(out, scores, valid_lens, TOKEN_K)):
        return False
    if out is not None and _overlap(out, blocks):
        return False
    return _device_supported(scores)


@lru_cache(maxsize=None)
def _module(kind):
    from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel

    header, symbol = {
        "amax": ("csrc/deepseek_v4/block_amax.cuh",
                 "sglang::BlockAmaxKernel<false, float, float>::amax8_varlen"),
        "select": ("compat/prefill_topk_entry.cuh", "sglang::PrefillTopKEntry::forward"),
        "small": ("csrc/deepseek_v4/topk_bf16_small.cuh",
                  "sglang::TopKBF16Kernel<false>::transform"),
    }[kind]
    return load_sglang_kernel("prefill_topk_" + kind, header, symbol, (8, 9),
                             bundle="prefill_topk")


def select_dense(scores, ends, out=None, *, block_topk=512, blocks_out=None):
    """Return (token512, blocksC), or None before unsupported GPU work.

    One fresh native amax is shared by token selection and block publication.
    The SG selector may modify its private keys; original scores stay valid.
    Scratch is per-call and has no cross-forward or cross-stream cache.
    """
    if not is_supported_dense(scores, ends, out, block_topk=block_topk,
                              blocks_out=blocks_out):
        return None
    amax, selector = _module("amax"), _module("select")
    rows, width = scores.shape
    if out is None:
        out = torch.empty((rows, TOKEN_K), dtype=torch.int32, device=scores.device)
    if blocks_out is None:
        blocks_out = torch.empty((rows, block_topk), dtype=torch.int32, device=scores.device)
    key_width = ((width // BLOCK_SIZE + 3) // 4) * 4
    keys = torch.empty((rows, key_width), dtype=torch.float32, device=scores.device)
    block_ends = torch.div(ends + 7, 8, rounding_mode="floor")
    offsets = torch.zeros_like(ends)
    amax.forward(scores, ends, keys, 0)
    selector.forward(keys, block_ends, None, offsets, blocks_out)
    selector.forward(scores, ends, blocks_out, offsets, out)
    return out, blocks_out


def select_sparse(scores, valid_lens, blocks, out=None):
    """Return native SG translated token512, or None before unsupported work."""
    if not is_supported_sparse(scores, valid_lens, blocks, out):
        return None
    small = _module("small")
    if out is None:
        out = torch.empty((scores.shape[0], TOKEN_K), dtype=torch.int32, device=scores.device)
    small.forward(scores, valid_lens, blocks, out, BLOCK_SIZE)
    return out

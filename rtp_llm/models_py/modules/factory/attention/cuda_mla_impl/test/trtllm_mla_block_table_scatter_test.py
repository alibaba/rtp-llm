"""UT for the trtllm-gen MLA decode block-table scatter kernel.

_scatter_block_table_kernel replaced the aten chain (searchsorted + clamp +
where + index_put + zeros) that builds the padded [bs, table_width] block
table from the flat page ids and the per-request cum indptr. The kernel's
contract under test:

  1. row mapping via in-kernel binary search over the cum indptr,
  2. only live entries are written (stale unused region tolerated),
  3. per-request page runs longer than table_width are clamped (col <
     table_width guard),
  4. flat pages beyond the indptr coverage (the one-beat numel/coverage
     diff seen during MTP verify capture) map to row == bs and are
     dropped, not written out of bounds,
  5. zero-length requests (empty indptr spans) write nothing.
"""

import math

import torch
import triton

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    _scatter_block_table_kernel,
)

SENTINEL = -1


def _launch(flat, indptr, bs, width):
    total = flat.numel()
    padded = torch.full((bs, width), SENTINEL, dtype=torch.int32, device=flat.device)
    _scatter_block_table_kernel[(triton.cdiv(total, 256),)](
        flat,
        indptr,
        padded,
        total,
        width,
        bs,
        LOG2_ITERS=8,
        BLOCK=256,
        num_warps=4,
    )
    return padded


def _reference(flat, indptr, bs, width):
    """Live-region expectation: row r holds flat[indptr[r] + c] for
    c < min(len_r, width); everything else keeps the sentinel."""
    out = torch.full((bs, width), SENTINEL, dtype=torch.int32)
    for r in range(bs):
        n = min(int(indptr[r + 1] - indptr[r]), width)
        if n > 0:
            out[r, :n] = flat[int(indptr[r]) : int(indptr[r]) + n]
    return out


def _run_case(dev, lens, width, tail_pages=0):
    bs = len(lens)
    pages = torch.arange(1, sum(lens) + 1, dtype=torch.int32, device=dev)
    indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    indptr[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32, device=dev), 0)
    if tail_pages:
        extra = torch.arange(9000, 9000 + tail_pages, dtype=torch.int32, device=dev)
        flat = torch.cat([pages, extra])
    else:
        flat = pages
    got = _launch(flat, indptr, bs, width)
    want = _reference(flat, indptr, bs, width)
    torch.testing.assert_close(got.cpu(), want)


def test_uniform_requests():
    dev = torch.device("cuda:0")
    _run_case(dev, [8, 8, 8, 8], 16)


def test_skewed_and_empty_requests():
    dev = torch.device("cuda:0")
    # includes zero-length spans and a heavy request
    _run_case(dev, [0, 20, 0, 3, 12], 24)


def test_width_clamp_when_run_exceeds_table_width():
    dev = torch.device("cuda:0")
    # request 1 owns 40 pages but the table is only 16 wide
    _run_case(dev, [4, 40, 6], 16)


def test_one_beat_coverage_tail_is_dropped():
    dev = torch.device("cuda:0")
    # flat numel exceeds indptr[-1] by 5: tail pages must not be written
    _run_case(dev, [5, 7, 9], 32, tail_pages=5)


def test_large_bs_binary_search_boundaries():
    dev = torch.device("cuda:0")
    # bs at the LOG2_ITERS=8 boundary (256) with edge-heavy lens
    lens = [1] * 254 + [0, 64]
    _run_case(dev, lens, 64)


def test_single_request_covers_full_width():
    dev = torch.device("cuda:0")
    _run_case(dev, [16], 16)


if __name__ == "__main__":
    test_uniform_requests()
    test_skewed_and_empty_requests()
    test_width_clamp_when_run_exceeds_table_width()
    test_one_beat_coverage_tail_is_dropped()
    test_large_bs_binary_search_boundaries()
    test_single_request_covers_full_width()
    print("ALL SCATTER TESTS PASSED")

"""Exercise real kernels across signed-int32 element-offset boundaries.

Large-stride cases allocate sparse test buffers but touch only small tiles.
The allocation prefix also keeps the old wrapped addresses in bounds, so a
regression gives an output mismatch rather than poisoning the CUDA context.
"""

from itertools import product
from unittest import SkipTest, TestCase, main, skipUnless

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    _convert_req_index_to_global_index_kernel,
    triton_convert_req_index_to_global_index,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
    _fused_zero_scatter_kernel,
    _gather_restore_fused_kernel,
    _gather_restore_separate_kernel,
)


def _require_free_memory(gib):
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < gib * 2**30:
        raise SkipTest(f"Address-boundary test requires {gib} GiB free")


def _byte_buffer(last_offset, bias):
    storage = torch.empty(bias + last_offset + 16, dtype=torch.uint8, device="cuda")
    # Old int32 wraparound lands in this prefix in the large-stride cases.
    storage[:128].zero_()
    return storage[bias:]


def _check_indexer_source_and_destination_offsets(kind, index_dtype, wide):
    if wide:
        _require_free_memory(12)
    stride = (2**30 + 16) if wide else 128
    bias = 2**31 if wide else 0
    scale_width, tile = 4, 16
    source_stride = stride + scale_width if kind == "packed" else stride
    last_offset = 2 * source_stride + (stride if kind == "packed" else 0)
    source = _byte_buffer(last_offset, bias)
    output = _byte_buffer(2 * stride, bias)
    source_scale = torch.arange(12, dtype=torch.uint8, device="cuda").view(3, 4)
    output_scale = torch.full_like(source_scale, 231)
    for row in range(3):
        source[row * source_stride : row * source_stride + tile].fill_(11 + row)
        output[row * stride : row * stride + tile].fill_(231)
        if kind == "packed":
            start = row * source_stride + stride
            source[start : start + scale_width].copy_(source_scale[row])

    ids = [2, -1, 0] if kind == "scatter" else [2, 0, 1]
    indices = torch.tensor(ids, dtype=index_dtype, device="cuda")
    if kind == "scatter":
        _fused_zero_scatter_kernel[(3,)](
            source,
            source_scale,
            output,
            output_scale,
            indices,
            stride,
            scale_width,
            3,
            tile,
            scale_width,
        )
    elif kind == "packed":
        _gather_restore_fused_kernel[(3,)](
            source,
            output,
            output_scale,
            indices,
            source_stride,
            stride,
            scale_width,
            3,
            tile,
            scale_width,
        )
    else:
        _gather_restore_separate_kernel[(3,)](
            source,
            source_scale,
            output,
            output_scale,
            indices,
            stride,
            scale_width,
            3,
            tile,
            scale_width,
        )

    for row, src in enumerate(ids):
        expected = torch.full(
            (tile,), 0 if src < 0 else 11 + src, dtype=torch.uint8, device="cuda"
        )
        torch.testing.assert_close(
            output[row * stride : row * stride + tile], expected, rtol=0, atol=0
        )
        expected_scale = (
            torch.zeros_like(source_scale[0]) if src < 0 else source_scale[src]
        )
        torch.testing.assert_close(output_scale[row], expected_scale, rtol=0, atol=0)


def _check_block_table_offsets_exceed_int32(axis):
    _require_free_memory(18)
    stride = 2**30 + 16  # Each stride fits int32; 2 * stride does not.
    bias = 2**31
    storage = torch.empty(bias + 2 * stride + 1, dtype=torch.int32, device="cuda")
    storage[:64].zero_()  # Safe old wrapped address: bias + int32(2 * stride).
    table = storage[bias:]
    table[2 * stride] = 123
    req = torch.tensor(
        [2 if axis == "request" else 0], dtype=torch.int32, device="cuda"
    )
    local = torch.tensor(
        [[5 if axis == "request" else 133]], dtype=torch.int32, device="cuda"
    )
    out = torch.empty_like(local)
    _convert_req_index_to_global_index_kernel[(1, 1)](
        req,
        table,
        local,
        out,
        None,
        None,
        3,
        64,
        64,
        1,
        False,
        stride if axis == "request" else 1,
        stride if axis == "block" else 1,
        1,
        1,
        1,
        1,
    )
    assert out.item() == 123 * 64 + 5
    assert out.dtype == torch.int32


def _check_mapping_padding_and_prefill_semantics(prefill):
    req = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    table = torch.tensor([[3, 0, 5], [7, 8, 9]], dtype=torch.int32, device="cuda")
    local = torch.tensor(
        [[0, 63, 64, -1], [128, 129, -1, 192]], dtype=torch.int32, device="cuda"
    )
    kwargs = {}
    if prefill:
        kwargs = dict(
            HAS_PREFILL_WORKSPACE=True,
            prefill_workspace_request_ids=torch.tensor(
                [-1, 0], dtype=torch.int32, device="cuda"
            ),
            prefill_workspace_starts=torch.tensor(
                [1000], dtype=torch.int32, device="cuda"
            ),
        )
    out = triton_convert_req_index_to_global_index(
        req, table, local, 64, 64, NUM_TOPK_TOKENS=4, BLOCK_N=4, **kwargs
    )
    expected = torch.tensor(
        [[192, 255, -1, -1], [1128, 1129, -1, -1] if prefill else [576, 577, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@skipUnless(torch.cuda.is_available(), "CUDA required")
class SparseMlaAddressingTest(TestCase):
    def test_indexer_source_and_destination_offsets(self):
        for wide, dtype, kind in product(
            (False, True), (torch.int32, torch.int64), ("scatter", "packed", "separate")
        ):
            with self.subTest(wide=wide, dtype=dtype, kind=kind):
                _check_indexer_source_and_destination_offsets(kind, dtype, wide)

    def test_block_table_offsets_exceed_int32(self):
        for axis in ("request", "block"):
            with self.subTest(axis=axis):
                _check_block_table_offsets_exceed_int32(axis)

    def test_mapping_padding_and_prefill_semantics(self):
        for prefill in (False, True):
            with self.subTest(prefill=prefill):
                _check_mapping_padding_and_prefill_semantics(prefill)


if __name__ == "__main__":
    main()

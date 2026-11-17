import torch

from rtp_llm.models_py.kernels.cuda.sm120.sparse.entry import (
    trtllm_batch_decode_sparse_mla_dsv4,
)
from rtp_llm.models_py.kernels.cuda.sm120.sparse.jit import gen_sparse_mla_sm120_module


def test_single_token_attention_preserves_caller_output():
    device = "cuda"
    query = torch.zeros((1, 1, 16, 512), device=device, dtype=torch.bfloat16)
    # Packed page layout stores 448 quantized values and 64 BF16 values
    # per slot followed by the eight UE8M0 scale bytes for each slot.
    page = torch.zeros((1, 1, 64, 584), device=device, dtype=torch.uint8)
    page.view(-1)[448:576].view(torch.bfloat16).fill_(2)
    page.view(-1)[64 * 576 :].fill_(127)
    indices = torch.full((1, 1, 128), -1, device=device, dtype=torch.int32)
    indices[..., 0] = 0
    lengths = torch.ones((1,), device=device, dtype=torch.int32)
    workspace = torch.empty((32 << 20,), device=device, dtype=torch.uint8)
    out = torch.empty_like(query)

    result = trtllm_batch_decode_sparse_mla_dsv4(
        query, page, workspace, indices, swa_topk_lens=lengths, out=out
    )

    assert result.data_ptr() == out.data_ptr()
    assert torch.count_nonzero(out[..., :448]).item() == 0
    assert torch.all(out[..., 448:] == 2)


def test_native_prefill_rejects_non_64_primary_page():
    module = gen_sparse_mla_sm120_module().build_and_load()
    device = "cuda"
    query = torch.zeros((1, 16, 512), device=device, dtype=torch.bfloat16)
    # Oversized pages keep the native page-64 reads in bounds if the check
    # regresses. The extra page uses an independently supported page size.
    primary = torch.zeros((1, 128, 584), device=device, dtype=torch.uint8)
    extra = torch.zeros((1, 2, 584), device=device, dtype=torch.uint8)
    indices = torch.zeros((1, 128), device=device, dtype=torch.int32)
    lengths = torch.ones((1,), device=device, dtype=torch.int32)

    for extra_cache in (None, extra):
        out = torch.full_like(query, 7)
        lse = torch.full((1, 16), 7.0, device=device, dtype=torch.float32)
        try:
            module.sparse_mla_sm120_paged_attention(
                query,
                primary,
                indices,
                out,
                lse,
                1.0,
                -1,  # infer DSv4 from the query head dimension
                lengths,
                None,
                extra_cache,
                indices if extra_cache is not None else None,
                lengths if extra_cache is not None else None,
            )
        except Exception as error:
            assert "Unsupported sparse-MLA prefill configuration" in str(error)
            assert "page_block_size=128" in str(error)
        else:
            raise AssertionError("native prefill accepted primary page size 128")
        assert torch.all(out == 7)
        assert torch.all(lse == 7)


def test_native_prefill_rejects_unaligned_secondary_indices():
    module = gen_sparse_mla_sm120_module().build_and_load()
    device = "cuda"
    query = torch.zeros((1, 16, 512), device=device, dtype=torch.bfloat16)
    primary = torch.zeros((1, 64, 584), device=device, dtype=torch.uint8)
    extra = torch.zeros((1, 2, 584), device=device, dtype=torch.uint8)
    indices = torch.zeros((1, 128), device=device, dtype=torch.int32)
    lengths = torch.ones((1,), device=device, dtype=torch.int32)
    # Keep both possible 64-entry tile reads within the backing allocation if
    # validation regresses. The view still advertises the unaligned row width.
    backing = torch.full((128,), -1, device=device, dtype=torch.int32)
    backing[0] = 0
    for width in (2, 65):
        extra_indices = backing[:width].view(1, width)
        extra_lengths = torch.full_like(lengths, width)
        out = torch.full_like(query, 7)
        lse = torch.full((1, 16), 7.0, device=device, dtype=torch.float32)
        try:
            module.sparse_mla_sm120_paged_attention(
                query,
                primary,
                indices,
                out,
                lse,
                1.0,
                -1,  # infer DSv4 from the query head dimension
                lengths,
                None,
                extra,
                extra_indices,
                extra_lengths,
            )
        except Exception as error:
            assert (
                f"prefill extra_indices width must be a multiple of 64, got {width}"
                in str(error)
            )
        else:
            raise AssertionError(f"native prefill accepted extra_indices width {width}")
        assert torch.all(out == 7)
        assert torch.all(lse == 7)
        print(f"PASS native prefill rejected extra_indices width {width}")


if __name__ == "__main__":
    test_single_token_attention_preserves_caller_output()
    print("PASS test_single_token_attention_preserves_caller_output")
    test_native_prefill_rejects_non_64_primary_page()
    print("PASS test_native_prefill_rejects_non_64_primary_page")
    test_native_prefill_rejects_unaligned_secondary_indices()

"""Vary request extents without changing compiled kernels or numerical results."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import triton

from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton as compressor
from rtp_llm.models_py.modules.dsv4.fp8._indexer_hadamard_quant_triton import (
    indexer_hadamard_quant_fold,
    indexer_hadamard_quant_fold_kernel,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    _causal_conv1d_fwd_kernel,
    causal_conv1d_fn,
)
from rtp_llm.models_py.triton_kernels.common.glm53_reduce_scatter import (
    publish_glm53_partials,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    hadamard_transform_128,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@contextmanager
def compiled_variants(kernel):
    """Observe the actual CompiledKernel returned by each launch."""
    hashes = set()
    run = kernel.run

    def record(*args, **kwargs):
        compiled = run(*args, **kwargs)
        hashes.add(compiled.hash)
        return compiled

    with patch.object(kernel, "run", side_effect=record):
        yield hashes


def test_quant_dynamic_rows_reuse_kernel_and_preserve_fp8_bytes():
    torch.manual_seed(531601)
    with compiled_variants(indexer_hadamard_quant_fold_kernel) as hashes:
        for rows in (1, 17, 32, 129, 257):
            q = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(rows, device="cuda", dtype=torch.float32)
            expected = indexer_q_fp8_quant_fold(
                hadamard_transform_128(q.reshape(1, rows, 1, 128)),
                w.reshape(1, rows, 1),
            )
            actual = indexer_hadamard_quant_fold(q, w)
            assert torch.equal(
                actual[0].view(torch.uint8).reshape(-1),
                expected[0].view(torch.uint8).reshape(-1),
            )
            torch.testing.assert_close(
                actual[1].reshape(-1), expected[1].reshape(-1), rtol=0, atol=0
            )
    assert len(hashes) == 1, "changing row count must not compile another kernel"


@pytest.mark.parametrize("head_dim", (128, 512))
def test_compressor_dynamic_prefix_state_capacity_reuses_kernel(head_dim):
    torch.manual_seed(531602)
    key = torch.randn(4, head_dim, device="cuda")
    score = torch.randn_like(key)
    ape = torch.randn(4, head_dim, device="cuda")
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    state_slots = 8 + positions
    # Only the last position completes a compressed entry. Read state block
    # one from the prefix pool, so the dynamic bounds check is exercised.
    kv_slots = torch.tensor([-1, -1, -1, 32], device="cuda", dtype=torch.int64)
    expected = None
    kernel = (
        compressor._fused_kv_compress_norm_rope_insert_indexer_attn
        if head_dim == 128
        else compressor._fused_kv_compress_norm_rope_insert_sparse_attn
    )
    rope_dim = 0 if head_dim == 128 else 64
    cos_sin = torch.zeros(4, rope_dim, device="cuda")
    cos_sin[:, : rope_dim // 2] = 1
    with compiled_variants(kernel) as hashes:
        for blocks in (2, 3, 16, 33):
            state = torch.zeros(blocks, 8, 2 * head_dim, device="cuda")
            entry_bytes = 132 if head_dim == 128 else 584
            pool = torch.zeros(2, 32, entry_bytes, dtype=torch.uint8, device="cuda")
            compressor.run_save_partial_states(
                key, score, ape, positions, state, state_slots, compress_ratio=4
            )
            compressor.run_fused_compress_kv_write(
                state,
                torch.zeros_like(positions, dtype=torch.int32),
                positions,
                state_slots,
                torch.tensor([[1]], device="cuda", dtype=torch.int32),
                torch.ones(head_dim, device="cuda", dtype=torch.bfloat16),
                1e-6,
                cos_sin,
                pool,
                kv_slots,
                key,
                score,
                ape,
                0,
                disable_raw_path=True,
                head_dim=head_dim,
                rope_head_dim=rope_dim,
                compress_ratio=4,
                overlap=False,
                state_tokens_per_block=128,
                kpool_mode=head_dim == 128,
            )
            if expected is None:
                expected = pool.clone()
                assert torch.count_nonzero(expected).item() > 0
            else:
                assert torch.equal(pool, expected)
    assert len(hashes) == 1, "prefix read-cache capacity must remain a runtime bound"


def test_grouped_conv_dynamic_stride_preserves_output_and_state():
    torch.manual_seed(531603)
    dim, width = 384, 4
    weight = torch.randn(dim, width, device="cuda")
    bias = torch.randn(dim, device="cuda")
    hashes = set()
    for length in (17, 64, 129, 257):
        lengths = (length, length + 3)
        x = torch.randn(sum(lengths), dim, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor(
            [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
        )
        block_map = torch.arange(8, device="cuda", dtype=torch.int32).reshape(2, 4)
        prefixes = torch.full((2,), 128, device="cuda", dtype=torch.int32)
        state = torch.randn(8, width - 1, dim, device="cuda", dtype=torch.bfloat16)
        reference_state, actual_state = state.clone(), state.clone()
        kwargs = dict(
            x=x.T,
            weight=weight,
            bias=bias,
            query_start_loc=cu,
            block_map=block_map,
            prefix_lengths=prefixes,
            seq_size_per_block=128,
            activation="silu",
        )
        expected = causal_conv1d_fn(
            conv_states=reference_state.transpose(1, 2), **kwargs
        ).T
        with compiled_variants(_causal_conv1d_fwd_kernel) as variants:
            actual = causal_conv1d_fn(
                conv_states=actual_state.transpose(1, 2), output_groups=3, **kwargs
            )
        hashes.update(variants)
        for got, want in zip(actual.unbind(0), expected.split(dim // 3, -1)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)
    assert len(hashes) == 1, "grouped output stride must not specialize by token count"


def test_publish_dynamic_extent_preserves_rank_offsets_and_guards():
    torch.manual_seed(531604)
    ranks, source_rank, guard = 8, 7, 16
    with compiled_variants(publish_glm53_partials) as hashes:
        for numel in (1, 17, 1024, 2051):
            source = torch.randn(ranks, numel, device="cuda", dtype=torch.bfloat16)
            peers = [
                torch.full(
                    (guard + ranks * numel + guard,),
                    -7,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for _ in range(ranks)
            ]
            pointers = torch.tensor(
                [p.data_ptr() for p in peers], device="cuda", dtype=torch.int64
            )
            publish_glm53_partials[(triton.cdiv(numel, 1024) * ranks,)](
                source,
                pointers,
                numel,
                NUM_RANKS=ranks,
                SOURCE_RANK=source_rank,
                DATA_OFFSET_BYTES=guard * 2,
                BLOCK=1024,
            )
            for destination, peer in enumerate(peers):
                expected = torch.full_like(peer, -7)
                start = guard + source_rank * numel
                expected[start : start + numel] = source[destination]
                torch.testing.assert_close(peer, expected, rtol=0, atol=0)
    assert len(hashes) == 1, "publish offsets must not specialize by batch extent"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

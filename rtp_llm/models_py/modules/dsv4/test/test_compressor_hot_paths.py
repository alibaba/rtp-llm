from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.cp import CPContext
from rtp_llm.models_py.modules.dsv4.indexer import Indexer


def _make_compressor(dim: int = 8, head_dim: int = 4, ratio: int = 128) -> Compressor:
    weights = {
        "ape": torch.zeros(ratio, head_dim, dtype=torch.float32),
        "wkv": torch.randn(head_dim, dim, dtype=torch.float32),
        "wgate": torch.randn(head_dim, dim, dtype=torch.float32),
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    comp = Compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=0,
        compress_ratio=ratio,
        max_batch_size=2,
        compressor_weights=weights,
    )
    comp.configure_kv_cache_shape(4)
    comp.freqs_cis = torch.empty(4096, 0, dtype=torch.complex64)
    comp._rmsnorm = lambda x: x
    return comp


def test_prefill_linear_uses_full_sequence_not_abs256_chunks():
    comp = _make_compressor()
    x = torch.randn(1, 300, comp.dim, dtype=torch.bfloat16)

    real_linear = torch.nn.functional.linear
    linear_input_shapes = []

    def counted_linear(input, weight, bias=None):
        linear_input_shapes.append(tuple(input.shape))
        return real_linear(input, weight, bias)

    with patch("torch.nn.functional.linear", side_effect=counted_linear):
        out = comp.forward(x, start_pos=0)

    assert out is not None
    assert tuple(out.shape) == (1, 2, comp.head_dim)
    assert linear_input_shapes == [(1, 300, comp.dim), (1, 300, comp.dim)]
    assert not hasattr(comp, "_linear_abs_blocked")


def test_indexer_weights_proj_uses_full_sequence_not_abs256_chunks():
    forward_names = Indexer.forward.__code__.co_names

    assert not hasattr(Indexer, "_weights_proj_abs_blocked")
    assert "_weights_proj_abs_blocked" not in forward_names
    assert "linear" in forward_names


def test_bind_kv_cache_from_pool_gathers_valid_slots_and_zero_fills_invalid():
    comp = _make_compressor(ratio=4)
    pool = torch.arange(12 * comp.head_dim, dtype=torch.bfloat16).reshape(
        12, comp.head_dim
    )
    block_table = torch.tensor([[3, 4], [5, 0]], dtype=torch.int32)
    comp.configure_kv_cache_shape(5)
    comp.set_pool_context(
        kv_pool_view=pool,
        kv_block_table=block_table,
        kv_eb=2,
        state_pool_view=None,
        state_block_table=None,
        state_eb=0,
    )

    comp._bind_kv_cache_from_pool(
        bsz=2,
        is_fresh_prefill=False,
        device=pool.device,
        dtype=torch.bfloat16,
    )

    zero = torch.zeros(comp.head_dim, dtype=torch.bfloat16)
    expected = torch.stack(
        [
            torch.stack([pool[6], pool[7], pool[8], pool[9], zero]),
            torch.stack([pool[10], pool[11], zero, zero, zero]),
        ]
    )
    torch.testing.assert_close(comp.kv_cache, expected, rtol=0, atol=0)
    assert comp.kv_cache.dtype == torch.bfloat16
    assert comp._kv_write_mask.dtype == torch.bool


def test_indexer_bind_kv_cache_from_pool_gathers_valid_slots_and_zero_fills_invalid():
    idx = Indexer.__new__(Indexer)
    torch.nn.Module.__init__(idx)
    idx._kv_cache_t = 5
    idx._kv_cache_d = 4
    idx._kv_eb = 2
    idx._dbg_prefix = None

    pool = torch.arange(12 * idx._kv_cache_d, dtype=torch.bfloat16).reshape(
        12, idx._kv_cache_d
    )
    idx._kv_pool_view = pool
    idx._kv_block_table = torch.tensor([[3, 4], [5, 0]], dtype=torch.int32)

    idx._bind_kv_cache_from_pool(
        bsz=2,
        is_fresh_prefill=False,
        device=pool.device,
        dtype=torch.bfloat16,
    )

    zero = torch.zeros(idx._kv_cache_d, dtype=torch.bfloat16)
    expected = torch.stack(
        [
            torch.stack([pool[6], pool[7], pool[8], pool[9], zero]),
            torch.stack([pool[10], pool[11], zero, zero, zero]),
        ]
    )
    torch.testing.assert_close(idx.kv_cache, expected, rtol=0, atol=0)
    assert idx.kv_cache.dtype == torch.bfloat16


def test_cp_continuation_state_uses_global_prefix_for_bind_and_scatter():
    comp = _make_compressor(ratio=4)
    T = comp._state_rows
    D = comp._state_dim
    state_eb = T
    kv_eb = 8
    block_tokens = kv_eb * comp.compress_ratio
    prefix_length = block_tokens * 15 + 5
    seq_len_full = block_tokens + 3
    read_logical_block = (prefix_length - 1) // block_tokens
    write_logical_block = (prefix_length + seq_len_full - 1) // block_tokens

    state_block_table = torch.zeros(1, write_logical_block + 1, dtype=torch.int32)
    state_block_table[0, read_logical_block] = 3
    state_block_table[0, write_logical_block] = 4
    state_pool = torch.zeros(64, 2 * D, dtype=torch.float32)
    read_slots = torch.arange(3 * state_eb, 3 * state_eb + T)
    write_slots = torch.arange(4 * state_eb, 4 * state_eb + T)
    read_rows = torch.arange(T * 2 * D, dtype=torch.float32).reshape(T, 2 * D)
    state_pool[read_slots] = read_rows

    comp.set_pool_context(
        kv_pool_view=None,
        kv_block_table=None,
        kv_eb=kv_eb,
        state_pool_view=state_pool,
        state_block_table=state_block_table,
        state_eb=state_eb,
    )
    comp.set_cp_ctx(
        CPContext(
            cp_size=2,
            cp_rank=1,
            chunk_length=4,
            padded_seq_len=8,
            seq_len_full=seq_len_full,
            relative_positions=torch.tensor([2, 3, 4, 5], dtype=torch.long),
            prefix_length=prefix_length,
            global_positions=torch.tensor(
                [prefix_length + 2, prefix_length + 3, prefix_length + 4, prefix_length + 5],
                dtype=torch.long,
            ),
            local_is_real=torch.ones(4, dtype=torch.bool),
            unpad_restore=torch.arange(seq_len_full, dtype=torch.long),
            seq_len_total=prefix_length + seq_len_full,
            cp_info=object(),
        )
    )

    expected_bound_kv = read_rows[:, :D].view(1, T, D)
    expected_bound_score = read_rows[:, D:].view(1, T, D)
    new_kv = torch.full((1, T, D), 17.0, dtype=torch.float32)
    new_score = torch.full((1, T, D), 23.0, dtype=torch.float32)

    def fake_scalar_body(x, start_pos, sequence_lengths=None):
        torch.testing.assert_close(comp.kv_state, expected_bound_kv, rtol=0, atol=0)
        torch.testing.assert_close(
            comp.score_state, expected_bound_score, rtol=0, atol=0
        )
        comp.kv_state = new_kv.clone()
        comp.score_state = new_score.clone()
        return torch.ones((1, 1, comp.head_dim), dtype=torch.float32)

    with patch.object(comp, "_bind_kv_cache_from_pool", lambda *args, **kwargs: None):
        with patch.object(comp, "_scatter_kv_cache_to_pool", lambda *args, **kwargs: None):
            with patch.object(comp, "_forward_scalar_impl", side_effect=fake_scalar_body):
                out = comp.forward(
                    torch.zeros(1, 4, comp.dim, dtype=torch.bfloat16),
                    start_pos=torch.tensor(12, dtype=torch.long),
                )

    assert out is not None
    expected_written = torch.cat([new_kv.squeeze(0), new_score.squeeze(0)], dim=-1)
    torch.testing.assert_close(state_pool[write_slots], expected_written, rtol=0, atol=0)
    # The rank-local start_pos block must not receive the continuation state.
    wrong_block = (12 + 4 - 1) // block_tokens
    if wrong_block not in (read_logical_block, write_logical_block):
        wrong_block_id = int(state_block_table[0, wrong_block])
        if wrong_block_id > 0:
            wrong_slots = torch.arange(
                wrong_block_id * state_eb, wrong_block_id * state_eb + T
            )
            assert not torch.equal(state_pool[wrong_slots], expected_written)

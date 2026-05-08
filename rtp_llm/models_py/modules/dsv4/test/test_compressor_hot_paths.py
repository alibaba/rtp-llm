import os
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import compressor as compressor_mod
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.compressor_vllm import _flatten_token_major_2d
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


def test_vllm_flatten_preserves_regular_strided_linear_slices():
    fused = torch.arange(2 * 5 * 9, dtype=torch.float32).reshape(2, 5, 9)
    kv = fused[..., :4]
    score = fused[..., 4:]

    kv_flat = _flatten_token_major_2d(kv, rows=10)
    score_flat = _flatten_token_major_2d(score, rows=10)

    assert tuple(kv_flat.shape) == (10, 4)
    assert tuple(score_flat.shape) == (10, 5)
    assert kv_flat.stride() == (9, 1)
    assert score_flat.stride() == (9, 1)
    assert kv_flat.data_ptr() == kv.data_ptr()
    assert score_flat.data_ptr() == score.data_ptr()
    torch.testing.assert_close(kv_flat, kv.reshape(10, 4), rtol=0, atol=0)
    torch.testing.assert_close(score_flat, score.reshape(10, 5), rtol=0, atol=0)


def _reference_metadata(positions, b_idx, state_bt, state_eb, kv_bt, kv_eb, ratio):
    state_bt_long = state_bt.to(torch.long)
    state_block_in_seq = (positions // state_eb) % state_bt_long.shape[1]
    state_in_block = positions % state_eb
    state_block_id = state_bt_long[b_idx, state_block_in_seq]
    state_slot = state_block_id * state_eb + state_in_block
    state_slot = torch.where(
        state_block_id > 0, state_slot, torch.full_like(state_slot, -1)
    )

    kv_bt_long = kv_bt.to(torch.long)
    tokens_per_block = kv_eb * ratio
    boundary = ((positions + 1) % ratio) == 0
    kv_block_in_seq = positions // tokens_per_block
    kv_in_block = (positions % tokens_per_block) // ratio
    kv_in_capacity = kv_block_in_seq < kv_bt_long.shape[1]
    kv_safe_block = kv_block_in_seq.clamp(min=0, max=kv_bt_long.shape[1] - 1)
    kv_block_id = kv_bt_long[b_idx, kv_safe_block]
    kv_valid = boundary & kv_in_capacity & (kv_block_id > 0)
    kv_slot = kv_block_id * kv_eb + kv_in_block
    kv_slot = torch.where(kv_valid, kv_slot, torch.full_like(kv_slot, -1))
    token_to_req = b_idx.to(torch.int32)
    return state_slot, kv_slot, token_to_req


def test_compressor_metadata_triton_matches_torch_reference():
    if not torch.cuda.is_available():
        return
    from rtp_llm.models_py.modules.dsv4._compressor_metadata_triton import (
        build_prefill_compressor_metadata,
        map_compressor_metadata,
    )

    state_eb = 16
    kv_eb = 8
    ratio = 4
    state_bt = torch.tensor(
        [[3, 4, 0, 6, 7], [8, 0, 10, 11, 12]], device="cuda", dtype=torch.int32
    )
    kv_bt = torch.tensor(
        [[13, 14, 0, 16], [17, 0, 19, 20]], device="cuda", dtype=torch.int32
    )
    positions = torch.tensor(
        [0, 3, 4, 15, 16, 31, 32, 63, 64, 95, 96, 127],
        device="cuda",
        dtype=torch.long,
    )
    b_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], device="cuda")

    got = map_compressor_metadata(
        positions, b_idx, state_bt, state_eb, kv_bt, kv_eb, ratio
    )
    assert got is not None
    expected = _reference_metadata(positions, b_idx, state_bt, state_eb, kv_bt, kv_eb, ratio)
    for actual, ref in zip(got, expected):
        torch.testing.assert_close(actual.cpu(), ref.cpu(), rtol=0, atol=0)

    prefill = build_prefill_compressor_metadata(
        5,
        2,
        10,
        torch.device("cuda"),
        state_bt,
        state_eb,
        kv_bt,
        kv_eb,
        ratio,
    )
    assert prefill is not None
    p_pos, p_bidx, p_state, p_kv, p_req = prefill
    ref_pos = (
        (torch.arange(10, device="cuda", dtype=torch.long).unsqueeze(0) + 5)
        .expand(2, -1)
        .reshape(-1)
        .contiguous()
    )
    ref_bidx = (
        torch.arange(2, device="cuda", dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, 10)
        .reshape(-1)
        .contiguous()
    )
    torch.testing.assert_close(p_pos.cpu(), ref_pos.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(p_bidx.cpu(), ref_bidx.cpu(), rtol=0, atol=0)
    expected = _reference_metadata(ref_pos, ref_bidx, state_bt, state_eb, kv_bt, kv_eb, ratio)
    for actual, ref in zip((p_state, p_kv, p_req), expected):
        torch.testing.assert_close(actual.cpu(), ref.cpu(), rtol=0, atol=0)


def test_compressor_fast_path_is_explicit_opt_in():
    with patch.dict(os.environ, {}, clear=True):
        assert not compressor_mod._use_compressor_fast()

    with patch.dict(os.environ, {"DSV4_COMPRESSOR_FAST": "0"}, clear=True):
        assert not compressor_mod._use_compressor_fast()

    with patch.dict(os.environ, {"DSV4_COMPRESSOR_FAST": "1"}, clear=True):
        with patch.object(compressor_mod, "_COMPRESSOR_FAST_OK", True):
            assert compressor_mod._use_compressor_fast()


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

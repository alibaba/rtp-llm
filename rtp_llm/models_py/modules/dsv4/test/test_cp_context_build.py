"""CPU-runnable unit tests for ``rtp_llm.models_py.modules.dsv4.cp.build_cp_context``.

Covers Phase B of the DSV4 prefill CP-on-FP8 migration: framework metadata
(``prefill_qkv_padding_mask`` / ``prefill_qkv_restore_indice``) is consumed
correctly to derive ``CPContext`` for both CP=1 (no-op) and CP>1 (real
zigzag split). Pool-write / attention CP behavior is exercised by the
follow-up phases.

Loads ``cp.py`` via importlib with a stubbed ``rtp_llm`` package chain so
the test stays runnable on machines without a built ``libth_transformer_config.so``
(matches the standalone pattern used by ``test_cp_topk_idxs_align`` /
``test_cp_via_concat_meta`` / ``test_cp_indexer_seq_total``).
"""

import importlib.util
import os
import sys
import types

import torch


def _load_cp_module():
    if "rtp_llm.models_py.modules.dsv4.cp" in sys.modules:
        return sys.modules["rtp_llm.models_py.modules.dsv4.cp"]

    class _Group:
        TP = "TP"

    class _CT:
        Group = _Group

        @staticmethod
        def all_gather(*a, **k):
            return None

    for name in [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["rtp_llm.models_py.distributed.collective_torch"] = _CT

    here = os.path.dirname(os.path.abspath(__file__))
    cp_path = os.path.normpath(os.path.join(here, os.pardir, "cp.py"))
    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.dsv4.cp", cp_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["rtp_llm.models_py.modules.dsv4.cp"] = mod
    return mod


_CP = _load_cp_module()
build_cp_context = _CP.build_cp_context
build_cp_full_prefill_positions = _CP.build_cp_full_prefill_positions
cp_gather_last_by_request = _CP.cp_gather_last_by_request


class _CpInfo:
    """Minimal stand-in for ``PyContextParallelParams`` carrying the
    fields ``build_cp_context`` reads. Mirrors what
    ``ZigZagProcessor::handleInputs`` populates.
    """

    def __init__(
        self,
        padding_mask: torch.Tensor,
        restore_indice: torch.Tensor,
        prefill_actual_input_lengths_cpu: torch.Tensor = None,
        prefill_cp_chunk_lengths: torch.Tensor = None,
    ):
        self.prefill_qkv_padding_mask = padding_mask
        self.prefill_qkv_restore_indice = restore_indice
        # Optional Phase-C field: per-request global full lengths. When
        # absent, build_cp_context leaves the global write-side view None
        # (rank-local tensors are used by the SWA write meta).
        if prefill_actual_input_lengths_cpu is None:
            prefill_actual_input_lengths_cpu = torch.empty(0, dtype=torch.int32)
        self.prefill_actual_input_lengths_cpu = prefill_actual_input_lengths_cpu
        if prefill_cp_chunk_lengths is not None:
            self.prefill_cp_chunk_lengths = prefill_cp_chunk_lengths


def _zigzag_restore(seq_len_full: int, cp_size: int) -> torch.Tensor:
    """Reproduce ZigZagProcessor's ``prefill_qkv_restore_indice`` for a
    single non-padded prefill stream of length ``seq_len_full`` evenly
    splittable into ``cp_size * 2`` zigzag pairs.
    """
    assert seq_len_full % (cp_size * 2) == 0
    pair = seq_len_full // (cp_size * 2)
    chunk = seq_len_full // cp_size
    restore = torch.empty(seq_len_full, dtype=torch.int32)
    for r in range(cp_size):
        # rank r owns local rows [r*chunk, (r+1)*chunk):
        #   first half  -> global positions [r*pair, (r+1)*pair)
        #   second half -> global positions [seq_len_full - (r+1)*pair, seq_len_full - r*pair)
        first_global = torch.arange(r * pair, (r + 1) * pair, dtype=torch.int32)
        second_global = torch.arange(
            seq_len_full - (r + 1) * pair, seq_len_full - r * pair, dtype=torch.int32
        )
        first_local = torch.arange(r * chunk, r * chunk + pair, dtype=torch.int32)
        second_local = torch.arange(
            r * chunk + pair, (r + 1) * chunk, dtype=torch.int32
        )
        restore[first_global] = first_local
        restore[second_global] = second_local
    return restore


def _zigzag_restore_multi(chunk_lengths, cp_size: int) -> torch.Tensor:
    total_chunk = sum(chunk_lengths)
    restore = torch.empty(cp_size * total_chunk, dtype=torch.int32)
    chunk_offset = 0
    seq_offset = 0
    for chunk in chunk_lengths:
        pair = chunk // 2
        padded = chunk * cp_size
        for rank in range(cp_size):
            dst_base = rank * total_chunk + chunk_offset
            even = torch.arange(
                seq_offset + rank * pair,
                seq_offset + rank * pair + pair,
                dtype=torch.long,
            )
            odd = torch.arange(
                seq_offset + padded - (rank + 1) * pair,
                seq_offset + padded - rank * pair,
                dtype=torch.long,
            )
            restore[even] = torch.arange(dst_base, dst_base + pair, dtype=torch.int32)
            restore[odd] = torch.arange(
                dst_base + pair, dst_base + 2 * pair, dtype=torch.int32
            )
        chunk_offset += chunk
        seq_offset += padded
    return restore


def _padding_mask_multi(chunk_lengths, actual_lengths, cp_size: int) -> torch.Tensor:
    parts = []
    for chunk, actual in zip(chunk_lengths, actual_lengths):
        padded = chunk * cp_size
        part = torch.zeros(padded, dtype=torch.int32)
        part[:actual] = 1
        parts.append(part)
    return torch.cat(parts)


def test_cp1_noop_collapses_to_local() -> None:
    """CP=1: chunk_length == padded_seq_len == seq_len_full, no zigzag,
    no padding, ``unpad_restore`` is the identity, ``global_positions``
    == ``relative_positions`` (when prefix_length=0)."""
    seq_len = 16
    padding_mask = torch.ones(seq_len, dtype=torch.int32)
    restore_indice = torch.arange(seq_len, dtype=torch.int32)
    ctx = build_cp_context(
        _CpInfo(padding_mask, restore_indice),
        cp_size=1,
        cp_rank=0,
        chunk_length=seq_len,
        device=torch.device("cpu"),
        position_offset=0,
    )
    assert ctx.cp_size == 1
    assert ctx.chunk_length == seq_len
    assert ctx.padded_seq_len == seq_len
    assert ctx.seq_len_full == seq_len
    assert ctx.prefix_length == 0
    assert ctx.seq_len_total == seq_len
    assert torch.equal(ctx.relative_positions, torch.arange(seq_len, dtype=torch.long))
    assert torch.equal(ctx.global_positions, ctx.relative_positions)
    assert ctx.local_is_real.all()
    assert torch.equal(ctx.unpad_restore, torch.arange(seq_len, dtype=torch.long))


def test_cp2_zigzag_two_ranks_cover_full_sequence() -> None:
    """CP=2: rank 0 + rank 1 ``relative_positions`` together cover every
    global position exactly once; ``unpad_restore`` is a permutation of
    [0, padded_seq_len)."""
    seq_len_full = 16
    cp_size = 2
    chunk_length = seq_len_full // cp_size
    padding_mask = torch.ones(seq_len_full, dtype=torch.int32)
    restore_indice = _zigzag_restore(seq_len_full, cp_size)

    rank_positions = []
    for r in range(cp_size):
        ctx = build_cp_context(
            _CpInfo(padding_mask, restore_indice),
            cp_size=cp_size,
            cp_rank=r,
            chunk_length=chunk_length,
            device=torch.device("cpu"),
            position_offset=0,
        )
        assert ctx.chunk_length == chunk_length
        assert ctx.seq_len_full == seq_len_full
        assert ctx.local_is_real.all()
        rank_positions.append(ctx.relative_positions)

    union = torch.cat(rank_positions).sort().values
    assert torch.equal(union, torch.arange(seq_len_full, dtype=torch.long))


def test_cp2_zigzag_pair_layout_matches_processor() -> None:
    """Spot-check the explicit zigzag formula: rank ``r`` owns the
    first-half pair ``[r*pair, (r+1)*pair)`` and the mirrored
    second-half pair ``[padded - (r+1)*pair, padded - r*pair)``."""
    seq_len_full = 16
    cp_size = 2
    chunk_length = seq_len_full // cp_size
    pair = chunk_length // 2
    padding_mask = torch.ones(seq_len_full, dtype=torch.int32)
    restore_indice = _zigzag_restore(seq_len_full, cp_size)

    ctx0 = build_cp_context(
        _CpInfo(padding_mask, restore_indice),
        cp_size,
        0,
        chunk_length,
        torch.device("cpu"),
    )
    expected0 = torch.cat(
        [
            torch.arange(0, pair, dtype=torch.long),
            torch.arange(seq_len_full - pair, seq_len_full, dtype=torch.long),
        ]
    )
    assert torch.equal(ctx0.relative_positions, expected0)

    ctx1 = build_cp_context(
        _CpInfo(padding_mask, restore_indice),
        cp_size,
        1,
        chunk_length,
        torch.device("cpu"),
    )
    expected1 = torch.cat(
        [
            torch.arange(pair, 2 * pair, dtype=torch.long),
            torch.arange(
                seq_len_full - 2 * pair, seq_len_full - pair, dtype=torch.long
            ),
        ]
    )
    assert torch.equal(ctx1.relative_positions, expected1)


def test_cp2_gather_last_by_request_handles_split_owners() -> None:
    """Last-token rows can be owned by different CP ranks across requests.
    The small MTP gather should return [B, H] in request order without a
    full-sequence gather.
    """
    cp_size = 2
    actual_lengths = [5, 7]
    chunk_lengths = [4, 4]
    chunk_length = sum(chunk_lengths)
    padding_mask = _padding_mask_multi(chunk_lengths, actual_lengths, cp_size)
    restore_indice = _zigzag_restore_multi(chunk_lengths, cp_size)
    cp_info = _CpInfo(
        padding_mask,
        restore_indice,
        prefill_actual_input_lengths_cpu=torch.tensor(actual_lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
    )

    ctx0 = build_cp_context(cp_info, cp_size, 0, chunk_length, torch.device("cpu"))
    ctx1 = build_cp_context(cp_info, cp_size, 1, chunk_length, torch.device("cpu"))
    local0 = torch.arange(chunk_length * 2, dtype=torch.float32).reshape(chunk_length, 2)
    local1 = (100 + torch.arange(chunk_length * 2, dtype=torch.float32)).reshape(chunk_length, 2)

    rank0_last = torch.zeros(2, 2)
    rank0_last[1] = local0[6]
    rank1_last = torch.zeros(2, 2)
    rank1_last[0] = local1[2]

    old_all_gather = _CP.all_gather
    try:
        _CP.all_gather = lambda local, group=None: torch.cat([local, rank1_last], dim=0)
        out0 = cp_gather_last_by_request(local0, ctx0)
        assert torch.equal(out0, torch.stack([local1[2], local0[6]]))

        _CP.all_gather = lambda local, group=None: torch.cat([rank0_last, local], dim=0)
        out1 = cp_gather_last_by_request(local1, ctx1)
        assert torch.equal(out1, torch.stack([local1[2], local0[6]]))
    finally:
        _CP.all_gather = old_all_gather


def test_cp2_continuation_prefix_offset_applied_to_global_positions() -> None:
    """``position_offset > 0`` (continuation prefill) shifts
    ``global_positions`` and ``seq_len_total`` accordingly; rank-local
    ``relative_positions`` and ``unpad_restore`` are unchanged because
    they describe the current input shard only."""
    seq_len_full = 8
    cp_size = 2
    chunk_length = seq_len_full // cp_size
    prefix = 32
    padding_mask = torch.ones(seq_len_full, dtype=torch.int32)
    restore_indice = _zigzag_restore(seq_len_full, cp_size)

    ctx = build_cp_context(
        _CpInfo(padding_mask, restore_indice),
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=chunk_length,
        device=torch.device("cpu"),
        position_offset=prefix,
    )
    assert ctx.prefix_length == prefix
    assert ctx.seq_len_total == prefix + seq_len_full
    assert torch.equal(ctx.global_positions, ctx.relative_positions + prefix)


def test_padding_tokens_marked_not_real_and_excluded_from_unpad_restore() -> None:
    """Real prefill length 12 with padded_seq_len 16 (cp_size=2): the
    last 4 global positions are padding. ``local_is_real`` flags them
    out and ``unpad_restore`` has length 12 (only the real tokens)."""
    real_len = 12
    padded = 16
    cp_size = 2
    chunk_length = padded // cp_size

    padding_mask = torch.zeros(padded, dtype=torch.int32)
    padding_mask[:real_len] = 1
    # restore index entries are only ever consulted at positions where
    # padding_mask == 1 (build_cp_context uses ``restore_indices[mask==1]``),
    # so padding rows can hold sentinel -1 without affecting the result.
    restore_indice = torch.full((padded,), -1, dtype=torch.int32)
    restore_indice[:real_len] = _zigzag_restore(padded, cp_size)[:real_len]

    seen_real = 0
    for r in range(cp_size):
        ctx = build_cp_context(
            _CpInfo(padding_mask, restore_indice),
            cp_size=cp_size,
            cp_rank=r,
            chunk_length=chunk_length,
            device=torch.device("cpu"),
        )
        assert ctx.seq_len_full == real_len
        seen_real += int(ctx.local_is_real.sum().item())
    assert seen_real == real_len


def test_global_write_view_absent_when_actual_input_lengths_empty() -> None:
    """No ``prefill_actual_input_lengths_cpu`` (CP-1 / framework didn't
    populate it) → ``input_lengths_global`` and ``cu_seqlens_global`` are
    None. SWA write meta then falls back to rank-local cu_seqlens."""
    seq_len = 8
    padding_mask = torch.ones(seq_len, dtype=torch.int32)
    restore_indice = torch.arange(seq_len, dtype=torch.int32)
    ctx = build_cp_context(
        _CpInfo(padding_mask, restore_indice),
        cp_size=1,
        cp_rank=0,
        chunk_length=seq_len,
        device=torch.device("cpu"),
    )
    assert ctx.input_lengths_global is None
    assert ctx.cu_seqlens_global is None


def test_global_write_view_b1_collapses_to_full_seq() -> None:
    """B=1 CP=2 prefill of 16 real tokens: ``input_lengths_global`` is
    ``[16]`` and ``cu_seqlens_global`` is ``[0, 16]`` — independent of
    cp_rank, since the gather'd write target is rank-invariant."""
    seq_len_full = 16
    cp_size = 2
    chunk_length = seq_len_full // cp_size
    padding_mask = torch.ones(seq_len_full, dtype=torch.int32)
    restore_indice = _zigzag_restore(seq_len_full, cp_size)
    actual = torch.tensor([seq_len_full], dtype=torch.int32)

    for r in range(cp_size):
        ctx = build_cp_context(
            _CpInfo(padding_mask, restore_indice, actual),
            cp_size=cp_size,
            cp_rank=r,
            chunk_length=chunk_length,
            device=torch.device("cpu"),
        )
        assert ctx.input_lengths_global is not None
        assert torch.equal(
            ctx.input_lengths_global, torch.tensor([seq_len_full], dtype=torch.int32)
        )
        assert torch.equal(
            ctx.cu_seqlens_global, torch.tensor([0, seq_len_full], dtype=torch.int32)
        )


def test_global_write_view_b_gt_1_cumsum_matches() -> None:
    """B>1 (Phase-2 forward-compat): ``cu_seqlens_global`` is the
    cumulative prefix sum over per-request global lengths."""
    actual = torch.tensor([12, 8, 20], dtype=torch.int32)
    cp_size = 2
    chunk_lengths = [6, 4, 10]
    padding_mask = _padding_mask_multi(chunk_lengths, actual.tolist(), cp_size)
    restore_indice = _zigzag_restore_multi(chunk_lengths, cp_size)

    ctx = build_cp_context(
        _CpInfo(
            padding_mask,
            restore_indice,
            actual,
            torch.tensor(chunk_lengths, dtype=torch.int32),
        ),
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=sum(chunk_lengths),
        device=torch.device("cpu"),
    )
    assert torch.equal(ctx.input_lengths_global, actual.to(torch.int32))
    assert torch.equal(
        ctx.cu_seqlens_global, torch.tensor([0, 12, 20, 40], dtype=torch.int32)
    )


def test_b_gt_1_positions_match_cpp_per_stream_zigzag() -> None:
    """C++ ZigZagProcessor plans each request independently.  Python must
    not treat the batch as one long virtual sequence."""
    chunk_lengths = [2, 4, 8, 2]
    actual = torch.tensor([8, 16, 32, 8], dtype=torch.int32)
    cp_size = 4
    restore_indice = _zigzag_restore_multi(chunk_lengths, cp_size)
    padding_mask = _padding_mask_multi(chunk_lengths, actual.tolist(), cp_size)
    cp_info = _CpInfo(
        padding_mask,
        restore_indice,
        actual,
        torch.tensor(chunk_lengths, dtype=torch.int32),
    )

    expected_rank0_padded = torch.tensor(
        [0, 7, 8, 9, 22, 23, 24, 25, 26, 27, 52, 53, 54, 55, 56, 63],
        dtype=torch.long,
    )
    ctx = build_cp_context(
        cp_info,
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=sum(chunk_lengths),
        device=torch.device("cpu"),
    )
    assert torch.equal(ctx.relative_positions, expected_rank0_padded)
    assert torch.equal(
        ctx.req_id_per_token,
        torch.tensor(
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3], dtype=torch.int32
        ),
    )


def test_b_gt_1_global_positions_are_per_request_not_padded_concat() -> None:
    chunk_lengths = [2, 4]
    actual = torch.tensor([8, 14], dtype=torch.int32)
    prefixes = torch.tensor([10, 100], dtype=torch.int64)
    cp_size = 4
    padding_mask = _padding_mask_multi(chunk_lengths, actual.tolist(), cp_size)
    restore_indice = _zigzag_restore_multi(chunk_lengths, cp_size)
    ctx = build_cp_context(
        _CpInfo(
            padding_mask,
            restore_indice,
            actual,
            torch.tensor(chunk_lengths, dtype=torch.int32),
        ),
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=sum(chunk_lengths),
        device=torch.device("cpu"),
        position_offset=prefixes,
    )
    # req0 rank0 owns local positions [0, 7]; req1 rank0 owns [0, 1, 14, 14].
    # The final req1 slot is padding (padded pos 15), clamped to last real pos 13.
    assert torch.equal(
        ctx.global_positions,
        torch.tensor([10, 17, 100, 101, 113, 113], dtype=torch.long),
    )


def test_cp_full_prefill_positions_preserve_request_ids() -> None:
    chunk_lengths = [2, 4]
    actual = torch.tensor([8, 14], dtype=torch.int32)
    prefixes = torch.tensor([10, 100], dtype=torch.int64)
    cp_size = 4
    ctx = build_cp_context(
        _CpInfo(
            _padding_mask_multi(chunk_lengths, actual.tolist(), cp_size),
            _zigzag_restore_multi(chunk_lengths, cp_size),
            actual,
            torch.tensor(chunk_lengths, dtype=torch.int32),
        ),
        cp_size=cp_size,
        cp_rank=0,
        chunk_length=sum(chunk_lengths),
        device=torch.device("cpu"),
        position_offset=prefixes,
    )
    positions, b_idx, seq_start, cu_seq = build_cp_full_prefill_positions(
        ctx, torch.device("cpu")
    )
    assert torch.equal(
        positions,
        torch.cat(
            [
                torch.arange(10, 18, dtype=torch.long),
                torch.arange(100, 114, dtype=torch.long),
            ]
        ),
    )
    assert torch.equal(
        b_idx,
        torch.cat(
            [
                torch.zeros(8, dtype=torch.long),
                torch.ones(14, dtype=torch.long),
            ]
        ),
    )
    assert torch.equal(seq_start, torch.tensor([10, 100], dtype=torch.long))
    assert torch.equal(cu_seq, torch.tensor([0, 8, 22], dtype=torch.long))


if __name__ == "__main__":
    test_cp1_noop_collapses_to_local()
    test_cp2_zigzag_two_ranks_cover_full_sequence()
    test_cp2_zigzag_pair_layout_matches_processor()
    test_cp2_continuation_prefix_offset_applied_to_global_positions()
    test_padding_tokens_marked_not_real_and_excluded_from_unpad_restore()
    test_global_write_view_absent_when_actual_input_lengths_empty()
    test_global_write_view_b1_collapses_to_full_seq()
    test_global_write_view_b_gt_1_cumsum_matches()
    test_b_gt_1_positions_match_cpp_per_stream_zigzag()
    test_b_gt_1_global_positions_are_per_request_not_padded_concat()
    test_cp_full_prefill_positions_preserve_request_ids()
    print("OK")

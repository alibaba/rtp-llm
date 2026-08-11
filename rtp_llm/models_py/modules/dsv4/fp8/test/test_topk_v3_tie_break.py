"""Correctness tests for the stable ragged GLM5 prefill TopK kernel."""

from __future__ import annotations

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _run(
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    output = torch.full(
        (scores.shape[0], k), -7, dtype=torch.int32, device=scores.device
    )
    rtp_llm_ops.topk_v3_tie_break(
        scores,
        row_starts,
        row_ends,
        output,
        k,
        scores.shape[1] if max_seq_len is None else max_seq_len,
    )
    return output


def _clamped_window(start: int, end: int, width: int, max_seq_len: int) -> tuple[int, int]:
    start = min(max(start, 0), width)
    end = min(max(end, start), width)
    end = min(end, start + max_seq_len)
    return start, end


def _stable_reference_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    # Stable descending sort preserves the original relative-index order for
    # numerically equal finite scores. Signed-zero ordering is checked directly
    # against dsv4_top_k_per_row_prefill below because its radix key ranks +0
    # above -0.
    return torch.argsort(values, descending=True, stable=True)[:k]


def _run_per_row(
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
) -> torch.Tensor:
    output = torch.full(
        (scores.shape[0], k), -7, dtype=torch.int32, device=scores.device
    )
    rtp_llm_ops.dsv4_top_k_per_row_prefill(
        scores,
        row_starts,
        row_ends,
        output,
        scores.shape[0],
        scores.stride(0),
        scores.stride(1),
        k,
        True,
    )
    return output


def _assert_per_row_selected_set(
    output: torch.Tensor,
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    tag: str,
) -> None:
    expected = _run_per_row(scores, row_starts, row_ends, k)
    actual_sorted = output.sort(dim=1).values
    expected_sorted = expected.sort(dim=1).values

    def failure_message(msg: str) -> str:
        mismatch = (actual_sorted != expected_sorted).nonzero()
        if mismatch.numel() == 0:
            detail = "no mismatched index"
        else:
            row, column = mismatch[0].tolist()
            stop = min(column + 8, k)
            detail = (
                f"first mismatch row={row}, column={column}, "
                f"actual={actual_sorted[row, column:stop].tolist()}, "
                f"per_row={expected_sorted[row, column:stop].tolist()}"
            )
        return f"{tag}: selected set differs from per-row radix; {detail}\n{msg}"

    torch.testing.assert_close(
        actual_sorted,
        expected_sorted,
        atol=0,
        rtol=0,
        msg=failure_message,
    )


def _assert_stable_equiv(
    output: torch.Tensor,
    scores: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    max_seq_len: int,
    tag: str,
) -> None:
    width = scores.shape[1]
    output_host = output.cpu()
    starts = row_starts.cpu().tolist()
    ends = row_ends.cpu().tolist()
    for row, (requested_start, requested_end) in enumerate(zip(starts, ends)):
        start, end = _clamped_window(
            int(requested_start), int(requested_end), width, max_seq_len
        )
        length = end - start
        keep = min(k, length)
        actual = output_host[row, :keep].to(torch.int64)
        padding = output_host[row, keep:]
        assert (padding == -1).all(), f"{tag}: row {row} padding is not -1"
        assert ((actual >= 0) & (actual < length)).all(), (
            f"{tag}: row {row} returned an out-of-window relative index"
        )
        assert torch.unique(actual).numel() == keep, (
            f"{tag}: row {row} returned duplicate indices"
        )
        expected = _stable_reference_indices(scores[row, start:end], keep).cpu()
        assert torch.equal(actual.sort().values, expected.sort().values), (
            f"{tag}: row {row} stable index set mismatch\n"
            f"actual-only={sorted(set(actual.tolist()) - set(expected.tolist()))[:16]}\n"
            f"expected-only={sorted(set(expected.tolist()) - set(actual.tolist()))[:16]}"
        )


def test_random_ragged_all_k() -> None:
    rows, width = 8, 16387
    generator = torch.Generator(device="cuda").manual_seed(2026081001)
    scores = torch.randn(rows, width, device="cuda", generator=generator)
    starts = torch.tensor(
        [0, 1, 3, 17, 511, 1025, 4097, 8191],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor(
        [1, 511, 512, 1023, 2049, 4097, 8193, 8196],
        dtype=torch.int32,
        device="cuda",
    )
    ends = starts + lengths
    for k in (512, 1024, 2048):
        output = _run(scores, starts, ends, k, width)
        _assert_stable_equiv(
            output, scores, starts, ends, k, width, f"random ragged K={k}"
        )


def _make_signed_zero_threshold(
    length: int, start: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = length + start + 5
    scores = torch.full((1, width), torch.inf, device="cuda", dtype=torch.float32)
    window = torch.full((length,), -1.0, device="cuda", dtype=torch.float32)
    above = k // 2
    window[:above] = torch.linspace(2.0, 1.0, above, device="cuda")
    zero_count = min(length - above, k * 2)
    zero_bits = torch.zeros(zero_count, dtype=torch.int32, device="cuda")
    zero_bits[1::2] = torch.iinfo(torch.int32).min
    window[above : above + zero_count] = zero_bits.view(torch.float32)
    scores[0, start : start + length] = window
    starts = torch.tensor([start], dtype=torch.int32, device="cuda")
    ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")
    return scores, starts, ends


def test_signed_zero_stable_tie_register() -> None:
    # The threshold is +0. Per-row ranks +0 above -0, then uses the smaller
    # index among the bitwise-identical +0 candidates.
    length, start, k = 8192, 3, 512
    scores, starts, ends = _make_signed_zero_threshold(length, start, k)
    output = _run(scores, starts, ends, k, length)
    _assert_per_row_selected_set(
        output, scores, starts, ends, k, "signed zero register"
    )


def test_signed_zero_stable_tie_streaming_overflow() -> None:
    # More than kMaxNumTie bitwise-identical +0 candidates force the full-row
    # overflow path without conflating +0 and -0.
    length, start, k = 65537, 5, 2048
    width = length + start + 5
    scores = torch.full((1, width), torch.inf, device="cuda", dtype=torch.float32)
    window = torch.full((length,), -1.0, device="cuda", dtype=torch.float32)
    above = k // 2
    window[:above] = torch.linspace(2.0, 1.0, above, device="cuda")
    window[above : above + 8192] = 0.0
    scores[0, start : start + length] = window
    starts = torch.tensor([start], dtype=torch.int32, device="cuda")
    ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")
    output = _run(scores, starts, ends, k, length)
    _assert_per_row_selected_set(
        output, scores, starts, ends, k, "signed zero streaming overflow"
    )


def test_negative_zero_coarse_threshold_matches_per_row() -> None:
    # Place -0 before +0 so the old numeric-boundary collect path reliably
    # exposes its bug: with a -0 threshold, `-0 >= +0` classifies both signs as
    # above and bypasses stable_handle_tie(). The histogram-key path must retain
    # every later +0 and select only the required smaller-index -0 candidates.
    k = 512
    for length, start, path in ((8192, 9, "register"), (32771, 13, "streaming")):
        width = length + start + 7
        scores = torch.full(
            (1, width), torch.inf, device="cuda", dtype=torch.float32
        )
        window = torch.full((length,), -1.0, device="cuda", dtype=torch.float32)
        above = k // 2
        negative_zero_count = k
        positive_zero_count = k // 4
        window[:above] = torch.linspace(2.0, 1.0, above, device="cuda")
        negative_zero_bits = torch.full(
            (negative_zero_count,),
            torch.iinfo(torch.int32).min,
            dtype=torch.int32,
            device="cuda",
        )
        window[above : above + negative_zero_count] = negative_zero_bits.view(
            torch.float32
        )
        positive_start = above + negative_zero_count
        window[positive_start : positive_start + positive_zero_count] = 0.0
        scores[0, start : start + length] = window
        starts = torch.tensor([start], dtype=torch.int32, device="cuda")
        ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")

        output = _run(scores, starts, ends, k, length)
        _assert_per_row_selected_set(
            output,
            scores,
            starts,
            ends,
            k,
            f"negative-zero coarse threshold {path}",
        )


def test_signed_nan_order_matches_per_row() -> None:
    # The first per-row coarse pass canonicalizes all NaNs into its highest
    # FP16 bin. Keep that production behavior even though its later exact key
    # distinguishes positive and negative NaNs.
    positive_nan_bits = 0x7FC00001
    negative_nan_bits = -0x003FFFFF  # int32 bit pattern 0xffc00001
    for length, start, path in ((8192, 7, "register"), (32771, 11, "streaming")):
        k = 512
        width = length + start + 5
        scores = torch.full((1, width), -1000.0, device="cuda", dtype=torch.float32)
        window = torch.linspace(4.0, -4.0, length, device="cuda")
        positive_nan = torch.tensor(
            [positive_nan_bits], dtype=torch.int32, device="cuda"
        ).view(torch.float32)
        negative_nan = torch.tensor(
            [negative_nan_bits], dtype=torch.int32, device="cuda"
        ).view(torch.float32)
        window[:192] = positive_nan
        window[192:384] = negative_nan
        scores[0, start : start + length] = window
        starts = torch.tensor([start], dtype=torch.int32, device="cuda")
        ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")

        output = _run(scores, starts, ends, k, length)
        _assert_per_row_selected_set(
            output, scores, starts, ends, k, f"signed NaN ordering {path}"
        )


def test_signed_nan_threshold_bins_match_per_row() -> None:
    positive_nan_bits = 0x7FC00001
    negative_nan_bits = -0x003FFFFF  # int32 bit pattern 0xffc00001
    lower_negative_nan_bits = -1  # int32 bit pattern 0xffffffff
    positive_nan = torch.tensor(
        [positive_nan_bits], dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    negative_nan = torch.tensor(
        [negative_nan_bits], dtype=torch.int32, device="cuda"
    ).view(torch.float32)
    lower_negative_nan = torch.tensor(
        [lower_negative_nan_bits], dtype=torch.int32, device="cuda"
    ).view(torch.float32)

    # Register path: the selected threshold is inside a positive-NaN tie.
    length, start, k = 8192, 7, 512
    scores = torch.full(
        (1, length + start + 5), -1000.0, device="cuda", dtype=torch.float32
    )
    scores[0, start : start + 1024] = positive_nan
    starts = torch.tensor([start], dtype=torch.int32, device="cuda")
    ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")
    output = _run(scores, starts, ends, k, length)
    _assert_per_row_selected_set(
        output, scores, starts, ends, k, "positive NaN threshold register"
    )

    # Streaming path: the negative-NaN threshold interval overflows the
    # candidate buffer and exercises the exact rescan before index tie-break.
    length, start, k = 32771, 11, 512
    scores = torch.empty(
        (1, length + start + 5), device="cuda", dtype=torch.float32
    )
    window = scores[0, start : start + length]
    window[:] = lower_negative_nan
    window[:192] = positive_nan
    window[192:448] = torch.linspace(4.0, -4.0, 256, device="cuda")
    window[448:1472] = negative_nan
    starts = torch.tensor([start], dtype=torch.int32, device="cuda")
    ends = torch.tensor([start + length], dtype=torch.int32, device="cuda")
    output = _run(scores, starts, ends, k, length)
    _assert_per_row_selected_set(
        output, scores, starts, ends, k, "negative NaN threshold streaming"
    )


def test_dense_discrete_ties_register_and_streaming() -> None:
    generator = torch.Generator(device="cuda").manual_seed(2026081002)
    for length, start, k in ((4097, 7, 512), (32771, 11, 2048)):
        width = length + start + 13
        scores = torch.full((4, width), 1000.0, device="cuda")
        values = torch.randn(4, length, device="cuda", generator=generator)
        values = torch.round(values * 4.0) / 4.0
        scores[:, start : start + length] = values
        starts = torch.full((4,), start, dtype=torch.int32, device="cuda")
        ends = torch.full((4,), start + length, dtype=torch.int32, device="cuda")
        output = _run(scores, starts, ends, k, length)
        _assert_stable_equiv(
            output,
            scores,
            starts,
            ends,
            k,
            length,
            f"discrete ties L={length} K={k}",
        )


def test_multi_request_nonzero_starts_ignore_poison() -> None:
    rows, width, k = 5, 65539, 2048
    generator = torch.Generator(device="cuda").manual_seed(2026081003)
    scores = torch.full((rows, width), torch.inf, device="cuda")
    starts = torch.tensor(
        [0, 4097, 8195, 16387, 32771], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(
        [4096, 4097, 8193, 16385, 32768], dtype=torch.int32, device="cuda"
    )
    ends = starts + lengths
    for row, (start, length) in enumerate(
        zip(starts.cpu().tolist(), lengths.cpu().tolist())
    ):
        scores[row, start : start + length] = torch.randn(
            length, device="cuda", generator=generator
        )
    output = _run(scores, starts, ends, k, width)
    _assert_stable_equiv(
        output, scores, starts, ends, k, width, "multi-request poison"
    )


def test_unaligned_score_view_and_row_starts() -> None:
    rows, width, k = 4, 8193, 1024
    generator = torch.Generator(device="cuda").manual_seed(2026081004)
    storage = torch.randn(rows, width + 3, device="cuda", generator=generator)
    scores = storage[:, 1 : width + 1]
    assert scores.stride(1) == 1 and scores.data_ptr() % 16 != 0
    starts = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    ends = torch.tensor([8193, 8192, 8191, 8190], dtype=torch.int32, device="cuda")
    output = _run(scores, starts, ends, k, width)
    _assert_stable_equiv(
        output, scores, starts, ends, k, width, "unaligned scores and starts"
    )


def test_bounds_clamp_empty_rows_and_padding() -> None:
    rows, width, k = 5, 4096, 512
    generator = torch.Generator(device="cuda").manual_seed(2026081005)
    scores = torch.randn(rows, width, device="cuda", generator=generator)
    starts = torch.tensor(
        [-17, 100, 2048, width, width + 7], dtype=torch.int32, device="cuda"
    )
    ends = torch.tensor(
        [100, 99, torch.iinfo(torch.int32).max, width, width + 11],
        dtype=torch.int32,
        device="cuda",
    )
    max_seq_len = 1024
    output = _run(scores, starts, ends, k, max_seq_len)
    _assert_stable_equiv(
        output,
        scores,
        starts,
        ends,
        k,
        max_seq_len,
        "bounds clamp and padding",
    )


def test_stable_selected_set_across_replays() -> None:
    length, start, k = 32771, 3, 2048
    scores, starts, ends = _make_signed_zero_threshold(length, start, k)
    expected_set = None
    for replay in range(20):
        output = _run(scores, starts, ends, k, length)
        selected = output[0].sort().values.cpu()
        if expected_set is None:
            expected_set = selected
        else:
            assert torch.equal(selected, expected_set), (
                f"stable tie-break changed selected set on replay {replay}"
            )


def test_empty_batch_is_noop() -> None:
    scores = torch.empty((0, 4096), dtype=torch.float32, device="cuda")
    bounds = torch.empty((0,), dtype=torch.int32, device="cuda")
    output = torch.empty((0, 512), dtype=torch.int32, device="cuda")
    rtp_llm_ops.topk_v3_tie_break(scores, bounds, bounds, output, 512, 4096)
    torch.cuda.synchronize()
    assert output.numel() == 0


if __name__ == "__main__":
    if not hasattr(rtp_llm_ops, "topk_v3_tie_break"):
        print("SKIP: topk_v3_tie_break binding is not built")
        raise SystemExit(0)
    test_random_ragged_all_k()
    test_signed_zero_stable_tie_register()
    test_signed_zero_stable_tie_streaming_overflow()
    test_negative_zero_coarse_threshold_matches_per_row()
    test_signed_nan_order_matches_per_row()
    test_signed_nan_threshold_bins_match_per_row()
    test_dense_discrete_ties_register_and_streaming()
    test_multi_request_nonzero_starts_ignore_poison()
    test_unaligned_score_view_and_row_starts()
    test_bounds_clamp_empty_rows_and_padding()
    test_stable_selected_set_across_replays()
    test_empty_batch_is_noop()
    print("topk_v3_tie_break correctness: PASS")

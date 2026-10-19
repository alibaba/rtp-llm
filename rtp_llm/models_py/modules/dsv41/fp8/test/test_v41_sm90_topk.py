"""Exact SM90 decode TopK contracts and opt-in clean graph timing.

The op is the existing SGLang-derived RTP topk_v3, not a new sorter. These
fixtures cover concentrated histograms without granting an all-zero-only
performance contract. Equal-score membership may differ from torch.topk;
the selected value multiset, valid index uniqueness and padding must match.
"""

import json
import os
import struct
import unittest

import torch

# Reuse the upstream BF16/finite-epilogue oracle on the SM90 runner.
from rtp_llm.models_py.modules.dsv4.fp8.test.test_topk_v3 import (
    test_bf16_cuda_graph_replay_updates_scores_and_lengths,
    test_bf16_dispatch_boundaries_match_fp32_cast,
    test_bf16_ragged_padded_stride_and_short_rows,
    test_bf16_special_values_ties_and_finite_epilogue,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class SignedZeroKeySourceProof(unittest.TestCase):
    def test_fp32_zero_and_half_rounded_negative_tiny_are_distinct(self):
        # CPU-only arithmetic proof of the old coarse/collect inconsistency.
        # Old -0 bins are511/2047, but collect's value >= upper(+0) is true.
        for hist_bits in (10, 12):
            positive_bin = 1 << (hist_bits - 1)
            for value in (0.0, -0.0, 1e-20, -1e-20):
                bits32 = struct.unpack("<I", struct.pack("<f", value))[0]
                bits16 = struct.unpack("<H", struct.pack("<e", value))[0]
                old_key16 = (~bits16 & 65535) if bits16 & 32768 else bits16 | 32768
                real_zero = bits32 & 0x7FFFFFFF == 0
                fixed_bits16 = 0 if real_zero else bits16
                fixed_key16 = (
                    (~fixed_bits16 & 65535)
                    if fixed_bits16 & 32768
                    else fixed_bits16 | 32768
                )
                if real_zero:
                    self.assertTrue(value >= 0.0)
                    self.assertEqual(fixed_key16 >> (16 - hist_bits), positive_bin)
                    if bits32 >> 31:
                        self.assertEqual(
                            old_key16 >> (16 - hist_bits), positive_bin - 1
                        )
                elif value < 0:
                    self.assertEqual(bits16, 0x8000)
                    self.assertEqual(fixed_key16, old_key16)
                    self.assertEqual(fixed_key16 >> (16 - hist_bits), positive_bin - 1)


def _scores(rows, capacity, distribution, *, strided=False):
    storage = torch.empty((rows, capacity + 5 if strided else capacity), device="cuda")
    scores = storage[:, 1 : capacity + 1] if strided else storage
    if distribution == "normal":
        scores.normal_()
    elif distribution == "zeros":
        scores.zero_()
    elif distribution == "negative_inf":
        scores.fill_(-float("inf"))
    elif distribution == "masked_zeros":
        scores.fill_(-float("inf"))
        scores[:, ::13] = 0.0
    elif distribution == "duplicates":
        scores.copy_(torch.randint(-4, 5, scores.shape, device="cuda").float())
    elif distribution == "coarse_collision":
        # Far more than 2048 genuinely different FP32 values share one coarse
        # FP16 histogram bin. Truncating the tie buffer is not an exact answer.
        scores.copy_(1.0 + torch.rand(scores.shape, device="cuda") * 0.001)
        scores[:, capacity // 2 : capacity // 2 + 37] = 2.0
    elif distribution == "late_collision":
        # All boundary candidates in one non-primary cluster CTA; exact local
        # counts must survive cross-CTA staging despite its fixed tie buffer.
        scores.fill_(-float("inf"))
        start = capacity * 6 // 8
        stop = capacity * 7 // 8
        scores[:, start:stop] = (
            1.0 + torch.rand((rows, stop - start), device="cuda") * 0.001
        )
    elif distribution == "nonfinite":
        scores.normal_()
        scores[:, ::17] = -float("inf")
        scores[:, ::31] = float("inf")
        scores[:, ::257] = float("nan")
    else:
        raise ValueError(distribution)
    return scores


def _workspace(scores, k):
    out = torch.empty((scores.shape[0], k), device=scores.device, dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device=scores.device, dtype=torch.uint8)
    return out, workspace


def _check(scores, lengths, out, k, capacity):
    """Independent torch.topk oracle with legal equal-score membership."""
    for row, requested in enumerate(lengths.cpu().tolist()):
        length = max(0, min(requested, capacity))
        keep = min(length, k)
        selected = out[row, :keep].long()
        assert bool((out[row, keep:] == -1).all()), "invalid padding"
        assert bool(((selected >= 0) & (selected < length)).all()), "out-of-range index"
        assert torch.unique(selected).numel() == keep, "duplicate selected indices"
        if keep:
            actual = scores[row].index_select(0, selected).sort().values
            expected = (
                scores[row, :length].topk(keep, sorted=False).values.sort().values
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


class SM90DecodeTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("Dedicated target requires a parent-locked SM90 GPU")
        if not hasattr(rtp_llm_ops, "topk_v3"):
            raise RuntimeError("topk_v3 CUDA binding is required")

    def setUp(self):
        torch.manual_seed(240041)

    def test_decode_distributions_exact_and_repeatable(self):
        for capacity in (66048, 132096):
            for distribution in (
                "normal",
                "zeros",
                "negative_inf",
                "masked_zeros",
                "duplicates",
                "coarse_collision",
                "late_collision",
                "nonfinite",
            ):
                with self.subTest(capacity=capacity, distribution=distribution):
                    scores = _scores(1, capacity, distribution)
                    lengths = torch.tensor(
                        [capacity - 512], device="cuda", dtype=torch.int32
                    )
                    out, workspace = _workspace(scores, 512)
                    first = None
                    for _ in range(3):
                        out.fill_(-12345)
                        rtp_llm_ops.topk_v3(
                            scores, lengths, out, workspace, 512, capacity
                        )
                        _check(scores, lengths, out, 512, capacity)
                        if first is None:
                            first = out.clone()
                        else:
                            torch.testing.assert_close(out, first, rtol=0, atol=0)

    def test_ragged_strided_short_lengths_and_all_supported_k(self):
        for rows, capacity in ((1, 513), (7, 32769), (19, 66048), (145, 4097)):
            for k in (512, 1024, 2048):
                with self.subTest(rows=rows, capacity=capacity, k=k):
                    scores = _scores(rows, capacity, "coarse_collision", strided=True)
                    values = [-17, 0, 1, k - 1, k, k + 1, capacity + 127]
                    lengths = torch.tensor(
                        [values[row % len(values)] for row in range(rows)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    out, workspace = _workspace(scores, k)
                    out.fill_(-12345)
                    rtp_llm_ops.topk_v3(scores, lengths, out, workspace, k, capacity)
                    _check(scores, lengths, out, k, capacity)

    def test_graph_changed_scores_and_lengths(self):
        for rows, capacity in ((1, 66048), (1, 132096), (7, 132096)):
            with self.subTest(rows=rows, capacity=capacity):
                scores = _scores(rows, capacity, "normal", strided=rows > 1)
                lengths = torch.full(
                    (rows,), capacity, device="cuda", dtype=torch.int32
                )
                out, workspace = _workspace(scores, 512)

                def invoke():
                    rtp_llm_ops.topk_v3(scores, lengths, out, workspace, 512, capacity)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        invoke()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke()
                for iteration, (distribution, length) in enumerate(
                    (
                        ("zeros", capacity - 512),
                        ("coarse_collision", capacity),
                        ("masked_zeros", 511),
                        ("negative_inf", 0),
                        ("normal", capacity + 32),
                        ("duplicates", 513),
                        ("nonfinite", capacity),
                    )
                ):
                    scores.copy_(_scores(rows, capacity, distribution))
                    lengths.fill_(length)
                    if rows > 1:
                        lengths[iteration % rows] = -1
                    graph.replay()
                    _check(scores, lengths, out, 512, capacity)
                    captured = out.clone()
                    invoke()
                    torch.testing.assert_close(out, captured, rtol=0, atol=0)
                    graph.replay()
                    torch.testing.assert_close(out, captured, rtol=0, atol=0)

    def test_nan_and_positive_inf_thresholds_across_dispatches(self):
        for rows, capacity in ((1, 4097), (1, 12289), (19, 66048), (145, 32769)):
            for nan_step in (7, 1021):
                with self.subTest(rows=rows, capacity=capacity, nan_step=nan_step):
                    scores = _scores(rows, capacity, "normal")
                    scores[:, ::2] = float("inf")
                    scores[:, ::nan_step] = float("nan")
                    lengths = torch.full(
                        (rows,), capacity, device="cuda", dtype=torch.int32
                    )
                    out, workspace = _workspace(scores, 512)
                    out.fill_(-12345)
                    rtp_llm_ops.topk_v3(scores, lengths, out, workspace, 512, capacity)
                    _check(scores, lengths, out, 512, capacity)

    def test_uniform_exact_boundary_with_strictly_greater_prefix(self):
        for capacity in (66048, 132096):
            for pivot in (
                0.0,
                -0.0,
                1.0,
                -1.0,
                1e-20,
                -1e-20,
                7.25,
                -9.75,
                float("inf"),
                -float("inf"),
                float("nan"),
            ):
                with self.subTest(capacity=capacity, pivot=pivot):
                    scores = torch.full(
                        (3, capacity), pivot, device="cuda", dtype=torch.float32
                    )
                    lengths = torch.tensor(
                        [capacity, capacity - 513, capacity - 1025],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    if pivot == pivot and pivot != float("inf"):
                        # Rows1/2 retain many exactly equal boundary keys, but
                        # contain 257 strict-above items spread over all CTAs.
                        # The fast path must neither overwrite this prefix nor
                        # count these items among the remaining tie slots.
                        higher = 10.0 if pivot == -float("inf") else 2 * abs(pivot) + 10
                        locations = torch.linspace(
                            0, capacity - 2048, 257, device="cuda"
                        ).long()
                        scores[1:, locations] = higher
                        if pivot != -float("inf"):
                            scores[2, 1::3] = -float("inf")
                            scores[2, locations] = higher
                    out, workspace = _workspace(scores, 512)
                    previous = None
                    for _ in range(3):
                        out.fill_(-12345)
                        rtp_llm_ops.topk_v3(
                            scores, lengths, out, workspace, 512, capacity
                        )
                        _check(scores, lengths, out, 512, capacity)
                        if previous is not None:
                            torch.testing.assert_close(out, previous, rtol=0, atol=0)
                        previous = out.clone()

    def test_equal_staged_sample_does_not_prove_uniform_boundary(self):
        for capacity in (66048, 132096):
            for k in (512, 1024, 2048):
                with self.subTest(capacity=capacity, k=k):
                    scores = torch.ones((4, capacity), device="cuda")
                    # Rank0 has >2048 equal ties, so its staged prefix (and
                    # therefore the 256-key rejection sample) contains only1.
                    # Both outliers are in the final rank and share1's coarse
                    # FP16 bin. A sample must never certify exact uniformity.
                    outlier = (capacity // 1024 - 1) * 1024
                    scores[0::2, outlier] = 1.0 + 2**-20
                    scores[1::2, outlier] = 1.0 - 2**-20
                    # The outlier index is visited by thread0 of the full-row
                    # exact emitter, so a false uniform shortcut would wrongly
                    # include the smaller value (and miss the larger value
                    # if it used only the staged prefix).
                    scores[2:, -257:] = 2.0
                    lengths = torch.full(
                        (4,), capacity, device="cuda", dtype=torch.int32
                    )
                    out, workspace = _workspace(scores, k)
                    previous = None
                    for _ in range(3):
                        out.fill_(-12345)
                        rtp_llm_ops.topk_v3(
                            scores, lengths, out, workspace, k, capacity
                        )
                        _check(scores, lengths, out, k, capacity)
                        if previous is not None:
                            torch.testing.assert_close(out, previous, rtol=0, atol=0)
                        previous = out.clone()

    def test_mixed_signed_zero_and_half_zero_tiny_across_dispatches(self):
        for rows, capacity in ((3, 4097), (3, 12289), (3, 66048), (145, 32769)):
            with self.subTest(rows=rows, capacity=capacity):
                scores = torch.full((rows, capacity), -0.0, device="cuda")
                scores[0::3, ::2] = 0.0
                scores[1::3, :64] = 0.0
                scores[2::3, :] = -1e-20  # FP16 -0, but not FP32 zero.
                scores[2::3, :64] = -0.0
                scores[:, -257:] = 10.0
                lengths = torch.full(
                    (rows,), capacity, device="cuda", dtype=torch.int32
                )
                out, workspace = _workspace(scores, 512)
                previous = None
                for _ in range(2):
                    out.fill_(-12345)
                    rtp_llm_ops.topk_v3(scores, lengths, out, workspace, 512, capacity)
                    _check(scores, lengths, out, 512, capacity)
                    if previous is not None:
                        torch.testing.assert_close(out, previous, rtol=0, atol=0)
                    previous = out.clone()

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_TOPK_BENCH") == "1", "opt-in timing"
    )
    def test_clean_graph_latency(self):
        for capacity in (66048, 132096):
            for distribution in ("normal", "zeros", "masked_zeros", "coarse_collision"):
                with self.subTest(capacity=capacity, distribution=distribution):
                    scores = _scores(1, capacity, distribution)
                    lengths = torch.tensor(
                        [capacity - 512], device="cuda", dtype=torch.int32
                    )
                    out, workspace = _workspace(scores, 512)

                    def invoke():
                        rtp_llm_ops.topk_v3(
                            scores, lengths, out, workspace, 512, capacity
                        )

                    invoke()
                    _check(scores, lengths, out, 512, capacity)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(5):
                            invoke()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(50):
                            invoke()
                    for _ in range(3):
                        graph.replay()
                    samples = []
                    for _ in range(5):
                        begin, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(10):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end) * 1000 / 500)
                    _check(scores, lengths, out, 512, capacity)
                    print(
                        "SM90_TOPK_BENCH "
                        + json.dumps(
                            dict(
                                rows=1,
                                capacity=capacity,
                                valid_length=capacity - 512,
                                k=512,
                                distribution=distribution,
                                microseconds=samples,
                                scope="topk_v3 only; no profiler; no score producer or input copies",
                            )
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()

"""Exact candidate-only paged scoring and full-chain benchmarks on H20."""

import json
import os
import statistics
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as decode_indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_decode_candidate as candidate
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    quantize_indexer_k_fp4,
    quantize_rows_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_indexer import paged_score


def make_case(b=1, s=1, capacity=2053, physical=128, ratio=1, blocks=128):
    torch.manual_seed(2441 + b + s + ratio)
    logical = physical // ratio
    pages = (capacity + logical - 1) // logical
    table = torch.randperm(b * pages, device="cuda").add_(1).int().view(b, pages)
    pool = torch.full(
        (b * pages + 1, physical, 68), 127, device="cuda", dtype=torch.uint8
    )
    ids = torch.arange(capacity, device="cuda")
    slots = table[:, ids // logical].long() * physical + ids % logical
    keys = torch.randn(b * capacity, 128, device="cuda", dtype=torch.bfloat16) * 0.25
    quantize_indexer_k_fp4(keys, slots.flatten(), pool)
    raw_q = torch.randn(b, s, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.25
    q, qs = quantize_rows_fp4(raw_q)
    q, qs = q.view(b, s, 32, 64), qs.view(b, s, 32)
    weights = torch.rand(b, s, 32, device="cuda") / 64
    lengths = torch.full((b * s,), capacity, device="cuda", dtype=torch.int32)
    candidates = torch.randint(
        0, pages * logical // 8 + 2, (b * s, blocks), device="cuda", dtype=torch.int32
    )
    return (
        dict(
            q=q,
            q_scale=qs,
            pool=pool,
            weights=weights,
            table=table,
            lengths=lengths,
            candidates=candidates,
            capacity=capacity,
            logical_entries=logical,
        ),
        raw_q,
    )


def flags_for(case):
    """Independent set-membership oracle, never used inside timed candidate calls."""
    width = (case["capacity"] + 7) // 8
    ids = case["candidates"].long()
    valid = (ids >= 0) & (ids < width)
    safe = torch.where(valid, ids, width)
    flags = torch.zeros(ids.shape[0], width + 1, device=ids.device, dtype=torch.uint8)
    flags.scatter_(1, safe, 1)
    return flags[:, :width].contiguous()


def baseline(case, flags=None):
    result = paged_score(
        case["q"],
        case["q_scale"],
        case["pool"],
        case["weights"],
        case["table"],
        case["lengths"],
        case["capacity"],
        case["logical_entries"],
    )
    topk.mask_candidates(result, flags_for(case) if flags is None else flags, 8)
    return result


def selected(logits, case):
    lengths = case["lengths"].int().clamp(0, case["capacity"]).contiguous()
    return topk.select_tokens(logits, lengths, 512)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90DecodeCandidateTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires SM90")

    def check(self, case):
        before = {
            key: value.clone()
            for key, value in case.items()
            if isinstance(value, torch.Tensor)
        }
        old = baseline(case)
        new = candidate.try_score(**case)
        self.assertIsNotNone(new)
        torch.testing.assert_close(
            new.view(torch.int32), old.view(torch.int32), rtol=0, atol=0
        )
        actual = selected(new, case)
        torch.testing.assert_close(actual, selected(old, case), rtol=0, atol=0)
        torch.testing.assert_close(
            actual, selected(candidate.try_score(**case), case), rtol=0, atol=0
        )
        for key, value in before.items():
            torch.testing.assert_close(
                case[key].view(torch.uint8), value.view(torch.uint8), rtol=0, atol=0
            )
        return actual

    def test_paged_layout_ratio_sentinel_and_exact_ties(self):
        for physical in (64, 128):
            for ratio in (1, 2):
                with self.subTest(physical=physical, ratio=ratio):
                    case, _ = make_case(b=3, s=5, physical=physical, ratio=ratio)
                    case["table"][0, 0] = 0
                    case["table"][1, 1] = -1
                    case["weights"][2].zero_()
                    case["lengths"].copy_(
                        torch.tensor(
                            [
                                -3,
                                0,
                                1,
                                511,
                                512,
                                513,
                                777,
                                2047,
                                2053,
                                9999,
                                512,
                                513,
                                1031,
                                2052,
                                2053,
                            ],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    case["candidates"][:, :7] = torch.tensor(
                        [-1, -7, 0, 0, 1, 256, 999999], device="cuda", dtype=torch.int32
                    )
                    self.check(case)

    def test_short_rows_empty_and_fewer_than_topk(self):
        case, _ = make_case(b=1, s=6)
        case["lengths"].copy_(
            torch.tensor(
                [0, 511, 512, 513, 2053, 2053], device="cuda", dtype=torch.int32
            )
        )
        case["candidates"].fill_(-1)
        case["candidates"][4, :12] = torch.arange(12, device="cuda")
        case["candidates"][5, :12] = torch.arange(12, device="cuda")
        case["weights"][0, 5].zero_()
        out = self.check(case)
        prefix = torch.arange(512, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(
            out[1], torch.where(prefix < 511, prefix, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(out[2], prefix, rtol=0, atol=0)
        self.assertTrue(bool((out[3] == -1).all()))
        for row in (4, 5):
            self.assertEqual(int((out[row] >= 0).sum()), 96)
            self.assertEqual(out[row][out[row] >= 0].unique().numel(), 96)

    def test_unsupported_layout_returns_none(self):
        case, _ = make_case()
        for patch in (
            {"capacity": 0},
            {"logical_entries": 17},
            {"table": case["table"].long()},
            {"candidates": case["candidates"][:, :7]},
        ):
            with self.subTest(patch=tuple(patch)):
                self.assertIsNone(candidate.try_score(**(case | patch)))

    def test_public_raw_q_positions_candidate_hook(self):
        # Exercise the actual production entry point, including its unchanged
        # SM90 gather/RoPE/FP4 preparation and raw-weight normalization.
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                case, raw_q = make_case(b=2, s=3, ratio=ratio)
                case["table"][0, 0] = 0
                case["table"][1, 1] = -1
                case["candidates"][:, :5] = torch.tensor(
                    [-1, 0, 0, 1, 999999], device="cuda", dtype=torch.int32
                )
                positions = torch.tensor(
                    [0, 1021, 1023, 1025, 3001, 4105],
                    device="cuda",
                    dtype=torch.int64,
                )
                case["lengths"] = ((positions + 1) // 2).int()
                angles = torch.randn(4106, 32, device="cuda")
                frequencies = torch.polar(torch.ones_like(angles), angles)
                raw_weights = (case["weights"] * 64).to(torch.bfloat16)
                raw_weights[1, 2].zero_()  # Exact-score ties at a long row.
                args = (
                    raw_q,
                    raw_weights,
                    frequencies,
                    case["pool"],
                    case["table"],
                    case["lengths"].view(2, 3),
                )
                kwargs = dict(
                    max_ctx_len=case["capacity"],
                    logical_entries_per_block=case["logical_entries"],
                    positions=positions,
                )
                old = decode_indexer.score_decode_indexer(*args, **kwargs)
                topk.mask_candidates(old, flags_for(case), 8)
                new = decode_indexer.score_decode_indexer(
                    *args, **kwargs, candidate_blocks=case["candidates"]
                )
                self.assertIsNotNone(new)
                # Production keeps its original downstream mask, even though
                # the candidate scorer already writes full-width masked logits.
                torch.testing.assert_close(
                    new.view(torch.int32), old.view(torch.int32), rtol=0, atol=0
                )
                topk.mask_candidates(new, flags_for(case), 8)
                torch.testing.assert_close(
                    new.view(torch.int32), old.view(torch.int32), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    selected(new, case), selected(old, case), rtol=0, atol=0
                )

    def test_graph_changes_all_live_inputs_and_metadata(self):
        for sequence in (3, 4):
            with self.subTest(sequence=sequence):
                self._check_graph_changes_all_live_inputs_and_metadata(sequence)

    def _check_graph_changes_all_live_inputs_and_metadata(self, sequence):
        case, raw_q = make_case(b=2, s=sequence, ratio=2)
        positions = torch.full((2 * sequence,), 4000, device="cuda", dtype=torch.int64)

        def run():
            q, sf = quantize_rows_fp4(raw_q)
            current = case | {
                "q": q.view(2, sequence, 32, 64),
                "q_scale": sf.view(2, sequence, 32),
                "lengths": ((positions + 1) // 2).int(),
            }
            logits = candidate.try_score(**current)
            return logits, selected(logits, current)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_logits, captured_indices = run()
        for iteration in range(3):
            raw_q.copy_(torch.randn_like(raw_q) * (iteration + 1) / 4)
            case["weights"].copy_(torch.rand_like(case["weights"]) / 64)
            case["table"].copy_(case["table"].flip(-1))
            # Change existing payload bytes without touching planar SF bytes.
            physical = case["pool"].shape[1]
            case["pool"].view(-1, physical * 68)[:, : physical * 64].bitwise_xor_(17)
            positions.copy_(
                torch.tensor(
                    [0, 1021, 1023, 1025, 3001, 4105, 511, 4097][: 2 * sequence],
                    device="cuda",
                )
                + iteration
            )
            case["candidates"].copy_(
                torch.randint(
                    -3, 260, case["candidates"].shape, device="cuda", dtype=torch.int32
                )
            )
            if iteration == 0:
                case["candidates"].fill_(-1)
            graph.replay()
            eager_logits, eager_indices = run()
            torch.testing.assert_close(
                captured_logits.view(torch.int32),
                eager_logits.view(torch.int32),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(captured_indices, eager_indices, rtol=0, atol=0)
            q, sf = quantize_rows_fp4(raw_q)
            current = case | {
                "q": q.view(2, sequence, 32, 64),
                "q_scale": sf.view(2, sequence, 32),
                "lengths": ((positions + 1) // 2).int(),
            }
            self.check(current)

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_DECODE_CANDIDATE_BENCH") == "1",
        "explicit benchmark opt-in",
    )
    def test_full_chain_128k_benchmark(self):
        for rows in (1, 4):
            for zero in (False, True):
                case, _ = make_case(b=rows, capacity=132096, blocks=2048)
                case["lengths"].fill_(131072)
                case["candidates"].copy_(
                    torch.randperm(16384, device="cuda")[:2048].int().expand(rows, -1)
                )
                if zero:
                    case["pool"].zero_()
                self.check(case)
                flags = flags_for(case)  # L20 has already produced this bitmap.
                functions = {
                    "full_score_mask_topk": lambda: selected(
                        baseline(case, flags), case
                    ),
                    "prepare_plan_candidate_score_topk": lambda: selected(
                        candidate.try_score(**case), case
                    ),
                }
                graphs = {}
                outputs = {}
                for name, fn in functions.items():
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs[name] = fn()
                    graphs[name] = graph
                samples = {name: [] for name in functions}
                for repeat in range(6):
                    order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
                    for name in order:
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 10)
                torch.testing.assert_close(*outputs.values(), rtol=0, atol=0)
                print(
                    "SM90_DECODE_CANDIDATE_BENCH "
                    + json.dumps(
                        {
                            "rows": rows,
                            "capacity": 132096,
                            "visible": 131072,
                            "candidate_blocks": 2048,
                            "zero_kv": zero,
                            "samples_us": samples,
                            "median_us": {
                                name: statistics.median(values)
                                for name, values in samples.items()
                            },
                            "timing": "CUDA events / graph complete plan+score+topk; q preparation excluded equally",
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()

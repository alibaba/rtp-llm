"""V4.1 paged decode scoring: real packed layout and changing graph inputs."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as fused
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_indexer as prefill
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    dequantize_indexer_k_fp4,
    quantize_indexer_k_fp4,
)


def make_case(b, s, capacity, physical, logical, ratio, seed=42):
    """Owner pages are permuted; unused rows/scales contain NaNs, not zeros."""
    torch.manual_seed(seed)
    device = "cuda"
    blocks = (capacity + logical - 1) // logical
    table = torch.randperm(b * blocks, device=device).add_(1).view(b, blocks).int()
    pool = torch.full(
        (b * blocks + 1, physical, 68), 127, dtype=torch.uint8, device=device
    )
    ids = torch.arange(capacity, device=device)
    slots = table[:, ids // logical].long() * physical + ids.remainder(logical)
    keys = torch.randn(b, capacity, 128, dtype=torch.bfloat16, device=device) * 0.4
    quantize_indexer_k_fp4(keys.flatten(0, 1), slots.flatten(), pool)
    # Interpret the documented physical bytes independently of the dequant
    # kernel. Re-quantizing with PyTorch would change rare rounding ties; both
    # old and new production scorers consume these existing cache bytes.
    raw_blocks = pool.view(-1, physical * 68)
    packed_q = raw_blocks[:, : physical * 64].view(-1, physical, 64).view(torch.int8)
    packed_sf = raw_blocks[:, physical * 64 :].view(torch.int32)
    physical_ids, offsets = slots // physical, slots % physical
    keys_ref = _decode_fp4(
        packed_q[physical_ids, offsets].reshape(-1, 64),
        packed_sf[physical_ids, offsets].reshape(-1),
    ).view(b, capacity, 128)
    restored = dequantize_indexer_k_fp4(pool, slots.flatten()).view_as(keys_ref)
    torch.testing.assert_close(restored, keys_ref, rtol=0, atol=0)
    q = torch.randn(b, s, 32, 128, dtype=torch.bfloat16, device=device) * 0.4
    weights = torch.randn(b, s, 32, dtype=torch.float32, device=device) / 64
    angles = torch.randn(b * s, 32, dtype=torch.float32, device=device)
    freqs = torch.polar(torch.ones_like(angles), angles)
    starts = torch.linspace(0, capacity * ratio - s, b, device=device).long()
    if b == 1:
        starts.fill_(capacity * ratio - s)
    positions = starts[:, None] + torch.arange(s, device=device)[None]
    lengths = ((positions + 1) // ratio).int()
    return {
        "q": q,
        "weights": weights,
        "freqs_cis": freqs,
        "pool": pool,
        "block_table": table,
        "context_lens": lengths,
        "max_ctx_len": capacity,
        "logical_entries_per_block": logical,
    }, keys_ref


def _decode_fp4(payload, sf):
    """Decode packed int8 e2m1 payload + packed-UE8M0 int32 scales to fp32."""
    rows = payload.shape[0]
    flat = payload.view(torch.uint8)
    low = flat & 15
    high = flat >> 4
    codes = torch.stack((low, high), dim=-1).reshape(rows, -1)
    magnitude = codes & 7
    normal = torch.exp2((magnitude >> 1).float() - 1.0) * (
        1.0 + (magnitude & 1).float() * 0.5
    )
    values = torch.where(magnitude < 2, magnitude.float() * 0.5, normal)
    values = torch.where(codes.ge(8), -values, values)
    exponent = sf.view(torch.uint8).view(rows, 4).float() - 127.0
    scale = torch.exp2(exponent)
    return values.reshape(rows, 4, 32) * scale[:, :, None]


def reference_q(q, freqs):
    value = q.clone()
    b, s, h, _ = q.shape
    pairs = torch.view_as_complex(value[..., -64:].float().reshape(b * s, h, 32, 2))
    rotated = torch.view_as_real(pairs * freqs[:, None]).flatten(-2)
    value[..., -64:] = rotated.reshape(b, s, h, 64).to(torch.bfloat16)
    _, _, values = prefill._fp4_rows_torch(value)
    return values.view(b, s, h, 128)


def reference(case, keys):
    q = reference_q(case["q"], case["freqs_cis"])
    dots = (
        torch.einsum("bshd,bkd->bshk", q.double(), keys.double())
        .relu_()
        .bfloat16()
        .double()
    )
    weights = case["weights"].bfloat16().double()
    lanes = [torch.zeros_like(dots[:, :, 0]) for _ in range(4)]
    for head in range(dots.shape[2]):
        lane = head % 4
        lanes[lane] = (
            (dots[:, :, head] * weights[:, :, head, None] + lanes[lane])
            .bfloat16()
            .double()
        )
    even = (lanes[0] + lanes[2]).bfloat16().double()
    odd = (lanes[1] + lanes[3]).bfloat16().double()
    logits = (even + odd).bfloat16().float()
    return mask(logits.flatten(0, 1), case)


def mask(logits, case):
    return logits.masked_fill(
        torch.arange(case["max_ctx_len"], device=logits.device)[None]
        >= case["context_lens"].flatten()[:, None],
        -torch.inf,
    )


def assert_logits_and_topk(actual, expected, case):
    visible = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), visible)
    assert torch.isneginf(actual[~visible]).all()
    if not visible.any():
        return
    error = (actual[visible] - expected[visible]).float()
    denom = expected[visible].square().mean().sqrt().clamp_min(1e-6)
    relative_rms = error.square().mean().sqrt() / denom
    print(
        f"relative_rms={relative_rms.item():.7g} max_abs={error.abs().max().item():.7g}"
    )
    assert relative_rms.item() < 0.0001
    for row, length in enumerate(case["context_lens"].flatten().tolist()):
        count = min(512, length)
        if count == 0:
            continue
        old = expected[row].topk(count).indices
        new = actual[row].topk(count).indices
        matches = torch.isin(old, new).sum().item()
        print(f"row={row} live={length} topk_overlap={matches}/{count}")
        if length <= count:
            assert matches == count
        else:
            ordered = expected[row].topk(count + 1).values
            gap = ordered[-2] - ordered[-1]
            row_error = (actual[row, :length] - expected[row, :length]).abs().max()
            if gap > 2 * row_error:
                # With a resolved boundary, membership must be exactly equal.
                assert matches == count
            else:
                # Only a numerically unresolved boundary may swap near ties.
                assert matches >= count - max(1, count // 100)


def _check_paged_owner_layout(physical, logical, ratio, b, s):
    if not torch.cuda.is_available() or not fused.is_supported(
        torch.device("cuda"), physical
    ):
        raise unittest.SkipTest("paged FP8 indexer unsupported on this device")
    torch.backends.cuda.matmul.allow_tf32 = False
    case, keys = make_case(b, s, 2053, physical, logical, ratio)
    actual = fused.score_decode_indexer(**case)
    assert actual is not None
    assert_logits_and_topk(actual, reference(case, keys), case)


def _check_graph_replay(logical, ratio):
    if not torch.cuda.is_available() or not fused.is_supported(
        torch.device("cuda"), 128
    ):
        raise unittest.SkipTest("paged FP8 indexer unsupported on this device")
    torch.backends.cuda.matmul.allow_tf32 = False
    case, keys = make_case(2, 6, 2053, 128, logical, ratio)
    # Compile all kernels and initialize dependency metadata before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.score_decode_indexer(**case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fused.score_decode_indexer(**case)
    for step in range(4):
        updated, updated_keys = make_case(
            2, 6, 2053, 128, logical, ratio, seed=71 + step
        )
        if step % 2:
            updated["block_table"][0, 0] = 0
            updated["block_table"][1, 2] = -7
            updated_keys[0, :logical] = 0
            updated_keys[1, 2 * logical : 3 * logical] = 0
        # Change Q, frequencies, FP32 weights, physical pages and the table, as
        # the engine does when replay admits different requests into a batch.
        for key in ("q", "weights", "freqs_cis", "pool", "block_table"):
            case[key].copy_(updated[key])
        starts = torch.tensor([step * 31, 901 + step * 27], device="cuda")
        positions = starts[:, None] * ratio + torch.arange(6, device="cuda")[None]
        case["context_lens"].copy_(((positions + 1) // ratio).int())
        if step == 3:
            case["context_lens"].zero_()
        graph.replay()
        assert_logits_and_topk(output, reference(case, updated_keys), case)


def _check_unallocated_table_rows(logical, ratio):
    if not torch.cuda.is_available() or not fused.is_supported(
        torch.device("cuda"), 128
    ):
        raise unittest.SkipTest("paged FP8 indexer unsupported on this device")
    case, keys = make_case(2, 6, 2053, 128, logical, ratio)
    case["context_lens"].fill_(1031)
    case["block_table"][0, 0] = 0
    case["block_table"][1, 2] = -99
    keys[0, :logical] = 0
    keys[1, 2 * logical : 3 * logical] = 0
    output = fused.score_decode_indexer(**case)
    assert_logits_and_topk(output, reference(case, keys), case)
    assert torch.count_nonzero(output[:6, :logical]).item() == 0
    assert torch.count_nonzero(output[6:, 2 * logical : 3 * logical]).item() == 0


def _selector_or_skip():
    from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as selector

    if not selector._TOPK_V3_OK:
        raise unittest.SkipTest("topk_v3 unavailable; bounded readers not testable")
    return selector


def _visible_mask(case, capacity):
    lengths = case["context_lens"].flatten()
    return torch.arange(capacity, device="cuda")[None] < lengths[:, None]


def _check_elided_padding(logical, ratio, capacity, preserve, poison_table):
    """pad_invisible=False keeps visible columns bitwise identical and every
    bounded decode selector output-identical. The elision itself (padding left
    untouched) is proven deterministically by the graph-replay staleness
    check, not by allocator contents."""
    if not torch.cuda.is_available() or not fused.is_supported(
        torch.device("cuda"), 128
    ):
        raise unittest.SkipTest("paged FP8 indexer unsupported on this device")
    selector = _selector_or_skip()
    torch.backends.cuda.matmul.allow_tf32 = False
    b, s = 2, 6
    case, keys = make_case(b, s, capacity, 128, logical, ratio)
    if poison_table:
        case["block_table"][0, 0] = 0
        case["block_table"][1, 2] = -7
        keys[0, :logical] = 0
        keys[1, 2 * logical : 3 * logical] = 0
    kwargs = dict(case, preserve_score_dtype=preserve)
    padded = fused.score_decode_indexer(**kwargs)
    elided = fused.score_decode_indexer(**kwargs, pad_invisible=False)
    visible = _visible_mask(case, capacity)
    assert torch.equal(elided[visible], padded[visible])
    assert torch.isneginf(padded[~visible]).all()
    lengths = case["context_lens"].flatten().contiguous()
    assert selector.is_supported(elided, lengths, 512)
    assert torch.equal(
        selector.select_tokens(elided, lengths, 512),
        selector.select_tokens(padded, lengths, 512),
    )
    for block_size, blocks in ((8, 64), (16, 32)):
        elided_c, elided_f = selector.select_candidates(
            elided, lengths, block_size, blocks
        )
        padded_c, padded_f = selector.select_candidates(
            padded, lengths, block_size, blocks
        )
        assert torch.equal(elided_c, padded_c)
        assert torch.equal(elided_f, padded_f)
    del padded, elided


def _check_graph_replay_elided(logical, ratio):
    """Replayed elided scoring with growing/shrinking/zero lengths keeps the
    bounded selection identical to freshly padded scoring, while the padding
    stays undefined (never -inf-filled) across replays."""
    if not torch.cuda.is_available() or not fused.is_supported(
        torch.device("cuda"), 128
    ):
        raise unittest.SkipTest("paged FP8 indexer unsupported on this device")
    selector = _selector_or_skip()
    torch.backends.cuda.matmul.allow_tf32 = False
    case, _ = make_case(2, 6, 2053, 128, logical, ratio)
    kwargs = dict(case, preserve_score_dtype=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused.score_decode_indexer(**kwargs, pad_invisible=False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fused.score_decode_indexer(**kwargs, pad_invisible=False)
    # Step 0 grows lengths, step 1 shrinks them (undefined padding),
    # step 2 regrows with an unallocated table row, step 3 zeroes all rows.
    starts_by_step = [(0, 901), (11, 40), (300, 1801), (0, 0)]
    prev_lengths = None
    checked_stale = 0
    for step, starts in enumerate(starts_by_step):
        updated, _ = make_case(2, 6, 2053, 128, logical, ratio, seed=71 + step)
        if step == 2:
            updated["block_table"][0, 0] = 0
        for key in ("q", "weights", "freqs_cis", "pool", "block_table"):
            case[key].copy_(updated[key])
        positions = (
            torch.tensor(starts, device="cuda")[:, None] * ratio
            + torch.arange(6, device="cuda")[None]
        )
        case["context_lens"].copy_(((positions + 1) // ratio).int())
        graph.replay()
        lengths = case["context_lens"].flatten().contiguous()
        if prev_lengths is not None:
            # Elision proof: after a shrink, the columns that just became
            # padding are NOT -inf-filled. Their contents are whatever the
            # graph pool's overlapping capture-time intermediates leave
            # behind (bounded readers must ignore them); only a regressed
            # PAD_INVISIBLE=True kernel would write -inf there.
            for row in range(lengths.numel()):
                new_len = int(lengths[row])
                if new_len < int(prev_lengths[row]):
                    assert not torch.isneginf(output[row, new_len:]).all()
                    checked_stale += 1
        reference = fused.score_decode_indexer(**kwargs)
        visible = _visible_mask(case, 2053)
        assert torch.equal(output[visible], reference[visible])
        assert torch.equal(
            selector.select_tokens(output, lengths, 512),
            selector.select_tokens(reference, lengths, 512),
        )
        prev_lengths = lengths.clone()
    assert checked_stale > 0, "no shrinking row: elision proof never ran"
    del output, graph


class V41DecodeIndexerTest(unittest.TestCase):
    def test_pad_invisible_elision_keeps_visible_columns_and_bounded_selectors(self):
        for logical, ratio, capacity, preserve, poison in [
            (64, 2, 2053, False, False),
            (64, 2, 2053, True, False),
            (64, 2, 2048, True, True),
            (128, 1, 2053, True, False),
            (128, 1, 2048, False, True),
        ]:
            with self.subTest(
                logical=logical,
                ratio=ratio,
                capacity=capacity,
                preserve=preserve,
                poison=poison,
            ):
                _check_elided_padding(logical, ratio, capacity, preserve, poison)

    def test_pad_invisible_elision_graph_replay_length_changes(self):
        for logical, ratio in [(64, 2), (128, 1)]:
            with self.subTest(logical=logical, ratio=ratio):
                _check_graph_replay_elided(logical, ratio)

    def test_fused_metadata_rejects_unsupported_inputs_before_launch(self):
        keys = fused.DecodeIndexerKeys(
            torch.empty(2, 128, 68, dtype=torch.uint8),
            torch.ones(2, 4, dtype=torch.int32),
            512,
            128,
        )
        shared = {}
        with patch.object(fused, "_fused_decode_metadata_kernel") as kernel:
            self.assertIsNone(
                fused.get_decode_metadata(
                    shared,
                    keys,
                    torch.zeros(12, dtype=torch.int64),
                    batch_size=2,
                    query_len=6,
                    compress_ratio=1,
                )
            )
            kernel.__getitem__.assert_not_called()
        self.assertEqual(shared, {})

    def test_disabled_rope_reuse_bypasses_metadata_before_tensor_access(self):
        shared = {}
        keys = fused.DecodeIndexerKeys(object(), object(), 512, 128)
        with patch.dict(
            os.environ,
            {"DSV41_FUSE_INDEXER_METADATA": "1", "DSV41_REUSE_DECODE_ROPE": "0"},
        ), patch.object(fused, "_fused_decode_metadata_kernel") as kernel:
            for _ in range(4):
                self.assertIsNone(
                    fused.get_decode_metadata(
                        shared,
                        keys,
                        object(),
                        batch_size=2,
                        query_len=6,
                        compress_ratio=1,
                    )
                )
            kernel.__getitem__.assert_not_called()
        self.assertEqual(shared, {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_metadata_matches_original_fields_and_isolates_groups(self):
        if not fused.is_supported(torch.device("cuda"), 128):
            self.skipTest("SM10x paged indexer required")
        with patch.dict(os.environ, {"DSV41_FUSE_INDEXER_METADATA": "1"}):
            for batch, span, context in ((1, 1, 32768), (3, 6, 32768), (64, 6, 131072)):
                for physical, logical, ratio in (
                    (128, 128, 1),
                    (128, 64, 2),
                    (64, 64, 2),
                ):
                    for dtype in (torch.int32, torch.int64):
                        with self.subTest(
                            batch=batch,
                            span=span,
                            ratio=ratio,
                            physical=physical,
                            dtype=dtype,
                        ):
                            capacity = context // ratio
                            cols = (capacity + logical - 1) // logical
                            storage = torch.arange(
                                batch * (cols * 2 + 3), device="cuda", dtype=dtype
                            ).reshape(batch, -1)
                            table = storage[:, 1 : 1 + 2 * cols : 2]
                            table[:, 0] = -7
                            table[:, -1] = 0
                            positions = torch.arange(
                                batch * span, device="cuda", dtype=dtype
                            ) % (context - 1)
                            positions[-1] = context - 1
                            positions[0] = -1
                            keys = fused.DecodeIndexerKeys(
                                torch.empty(
                                    2, physical, 68, dtype=torch.uint8, device="cuda"
                                ),
                                table,
                                capacity,
                                logical,
                            )
                            shared = {}
                            plan = fused.get_decode_metadata(
                                shared,
                                keys,
                                positions,
                                batch_size=batch,
                                query_len=span,
                                compress_ratio=ratio,
                            )
                            lengths = ((positions.long() + 1) // ratio).int()
                            last = (lengths - 1).clamp_min(0)
                            expected_physical = torch.where(
                                lengths > 0,
                                last // logical * physical + last % logical + 1,
                                0,
                            )
                            for actual, expected in (
                                (plan.lengths, lengths),
                                (plan.paged.lengths.flatten(), expected_physical),
                                (plan.safe_table, table.clamp_min(0).int()),
                                (
                                    plan.paged.tables,
                                    table.clamp_min(0)
                                    .int()
                                    .repeat_interleave(span, dim=0),
                                ),
                                (
                                    plan.paged.request_ids,
                                    torch.arange(
                                        batch, device="cuda", dtype=torch.int32
                                    ).repeat_interleave(span),
                                ),
                            ):
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0
                                )
                            # Owners with identical table views share metadata, never KV pools.
                            view_keys = replace(
                                keys, pool=keys.pool.clone(), block_table=table[:]
                            )
                            self.assertIs(
                                fused.get_decode_metadata(
                                    shared,
                                    view_keys,
                                    positions[:],
                                    batch_size=batch,
                                    query_len=span,
                                    compress_ratio=ratio,
                                ),
                                plan,
                            )
                            for other, pos, other_ratio in (
                                (
                                    replace(keys, block_table=table.clone()),
                                    positions,
                                    ratio,
                                ),
                                (
                                    replace(
                                        keys,
                                        block_table=table.as_strided(
                                            table.shape, (table.stride(0), 1)
                                        ),
                                    ),
                                    positions,
                                    ratio,
                                ),
                                (
                                    replace(keys, capacity=capacity - 1),
                                    positions,
                                    ratio,
                                ),
                                (keys, positions.clone(), ratio),
                                (keys, positions, 3 - ratio),
                            ):
                                self.assertIsNot(
                                    fused.get_decode_metadata(
                                        shared,
                                        other,
                                        pos,
                                        batch_size=batch,
                                        query_len=span,
                                        compress_ratio=other_ratio,
                                    ),
                                    plan,
                                )
            for disabled in ("DSV41_FUSE_INDEXER_METADATA", "DSV41_REUSE_DECODE_ROPE"):
                shared = {}
                with patch.dict(os.environ, {disabled: "0"}), patch.object(
                    fused, "_fused_decode_metadata_kernel"
                ) as kernel:
                    self.assertIsNone(
                        fused.get_decode_metadata(
                            shared,
                            keys,
                            positions,
                            batch_size=batch,
                            query_len=span,
                            compress_ratio=ratio,
                        )
                    )
                    kernel.__getitem__.assert_not_called()
                self.assertEqual(shared, {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shared_metadata_read_only_across_owners_and_graph_replays(self):
        if not fused.is_supported(torch.device("cuda"), 128):
            self.skipTest("SM10x paged indexer required")
        with patch.dict(
            os.environ,
            {"DSV41_FUSE_INDEXER_METADATA": "1", "DSV41_REUSE_DECODE_ROPE": "1"},
        ):
            for logical, ratio in ((128, 1), (64, 2)):
                cases = [
                    make_case(2, 6, 2053, 128, logical, ratio, seed=seed)[0]
                    for seed in (42, 43)
                ]
                cases[1]["block_table"] = cases[0]["block_table"][:]
                cases[1]["context_lens"] = cases[0]["context_lens"]
                positions = cases[0]["context_lens"].flatten().long() * ratio - 1

                def prepared():
                    shared, outputs = {}, []
                    first = None
                    # Distinct pools consume one schedule repeatedly, without rebuilds.
                    for case in cases * 2:
                        keys = fused.DecodeIndexerKeys(
                            case["pool"], case["block_table"], 2053, logical
                        )
                        plan = fused.get_decode_metadata(
                            shared,
                            keys,
                            positions,
                            batch_size=2,
                            query_len=6,
                            compress_ratio=ratio,
                        )
                        if first is None:
                            first = plan
                            tensors = (
                                plan.lengths,
                                plan.safe_table,
                                plan.paged.lengths,
                                plan.paged.tables,
                                plan.paged.request_ids,
                                plan.paged.schedule,
                            )
                            before = tuple(t.clone() for t in tensors)
                        else:
                            self.assertIs(plan, first)
                        outputs.append(
                            fused.score_decode_indexer(
                                **case, metadata=plan, preserve_score_dtype=True
                            )
                        )
                    return outputs, tensors, before

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        prepared()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = prepared()
                for step in range(3):
                    for owner, case in enumerate(cases):
                        update, _ = make_case(
                            2, 6, 2053, 128, logical, ratio, seed=90 + step * 2 + owner
                        )
                        for name in ("q", "weights", "freqs_cis", "pool"):
                            case[name].copy_(update[name])
                        if owner == 0:
                            case["block_table"].copy_(update["block_table"])
                    lengths = torch.tensor(
                        [0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 1024, 2053],
                        device="cuda",
                        dtype=torch.int32,
                    ).view(2, 6)
                    if step == 2:
                        lengths.zero_()
                    cases[0]["context_lens"].copy_(lengths)
                    positions.copy_(lengths.flatten().long() * ratio - 1)
                    graph.replay()
                    expected = [
                        fused.score_decode_indexer(**case, preserve_score_dtype=True)
                        for case in cases
                    ]
                    for actual, tensors, before in (captured, prepared()):
                        for output, reference in zip(actual, expected * 2):
                            torch.testing.assert_close(
                                output, reference, rtol=0, atol=0
                            )
                        for tensor, snapshot in zip(tensors, before):
                            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_prepared_score_graph_replay_and_forward_cache_lifetime(self):
        if not fused.is_supported(torch.device("cuda"), 128):
            self.skipTest("SM10x paged indexer required")
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

        with patch.dict(os.environ, {"DSV41_FUSE_INDEXER_METADATA": "1"}):
            for logical, ratio in ((128, 1), (64, 2)):
                case, _ = make_case(2, 6, 2053, 128, logical, ratio)
                keys = fused.DecodeIndexerKeys(
                    case["pool"], case["block_table"], 2053, logical
                )
                positions = case["context_lens"].flatten().long() * ratio - 1
                shared = {"layers": {0: None, 2: None}}
                first = SimpleNamespace(layer_id=0, _shared_attention=shared)
                later = SimpleNamespace(layer_id=2, _shared_attention=shared)

                def prepared():
                    AttentionV41FP8._begin_forward(first)
                    self.assertNotIn("decode_indexer_metadata", shared)
                    plan = fused.get_decode_metadata(
                        shared,
                        keys,
                        positions,
                        batch_size=2,
                        query_len=6,
                        compress_ratio=ratio,
                    )
                    AttentionV41FP8._begin_forward(later)
                    reused = fused.get_decode_metadata(
                        shared,
                        keys,
                        positions,
                        batch_size=2,
                        query_len=6,
                        compress_ratio=ratio,
                    )
                    self.assertIs(reused, plan)
                    return fused.score_decode_indexer(
                        **case, metadata=plan, preserve_score_dtype=True
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        prepared()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = prepared()
                for step in range(4):
                    update, _ = make_case(
                        2, 6, 2053, 128, logical, ratio, seed=91 + step
                    )
                    if step % 2:
                        update["block_table"][0, 0] = 0
                        update["block_table"][1, 2] = -7
                    for name in ("q", "weights", "freqs_cis", "pool", "block_table"):
                        case[name].copy_(update[name])
                    lengths = torch.tensor(
                        [0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 1024, 2053],
                        device="cuda",
                        dtype=torch.int32,
                    ).view(2, 6)
                    if step == 3:
                        lengths.zero_()
                    case["context_lens"].copy_(lengths)
                    positions.copy_(lengths.flatten().long() * ratio - 1)
                    graph.replay()
                    expected = fused.score_decode_indexer(
                        **case, preserve_score_dtype=True
                    )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    torch.testing.assert_close(prepared(), expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_full_layer_selector_uses_prepared_metadata_with_exact_fallback(self):
        if not fused.is_supported(torch.device("cuda"), 128):
            self.skipTest("SM10x paged indexer required")
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

        for logical, ratio, layer_id in ((128, 1, 20), (64, 2, 2)):
            case, _ = make_case(2, 6, 2053, 128, logical, ratio)
            positions = case["context_lens"].flatten().long() * ratio - 1
            # Real requests have nonnegative positions; keep one short visible row.
            positions.clamp_min_(0)
            freqs = torch.ones(2053 * ratio, 32, dtype=torch.complex64, device="cuda")
            owner = SimpleNamespace(
                layer_id=layer_id,
                kv_source_layer_id=layer_id,
                is_index_source=True,
                index_n_heads=32,
                index_head_dim=128,
                index_topk=512,
                compress_ratio=ratio,
                rope_head_dim=64,
                index_weights=torch.randn(32, 16, dtype=torch.bfloat16, device="cuda"),
                freqs_cis=freqs,
                v41_config={},
                _shared_attention={
                    "global": {
                        layer_id: fused.DecodeIndexerKeys(
                            case["pool"], case["block_table"], 2053, logical
                        )
                    }
                },
            )
            x = torch.randn(2, 6, 16, dtype=torch.bfloat16, device="cuda")

            def run(enabled):
                with patch.dict(os.environ, {"DSV41_FUSE_INDEXER_METADATA": enabled}):
                    return AttentionV41FP8._select_indices_decode(
                        owner, x, None, positions, index_q=case["q"].flatten(0, 1)
                    )

            expected = run("0")
            with patch.object(
                fused, "get_decode_metadata", wraps=fused.get_decode_metadata
            ) as prepare:
                actual = run("1")
                self.assertEqual(prepare.call_count, 1)
            self.assertTrue(owner._shared_attention["decode_indexer_metadata"])
            torch.testing.assert_close(
                actual.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
            )

            owner._shared_attention.pop("decode_indexer_metadata")
            with patch.dict(os.environ, {"DSV41_REUSE_DECODE_ROPE": "0"}), patch.object(
                fused, "_fused_decode_metadata_kernel"
            ) as kernel:
                for _ in range(4):
                    actual = run("1")
                    torch.testing.assert_close(
                        actual.sort(dim=-1).values,
                        expected.sort(dim=-1).values,
                        rtol=0,
                        atol=0,
                    )
                    self.assertNotIn("decode_indexer_metadata", owner._shared_attention)
                kernel.__getitem__.assert_not_called()

    def test_paged_owner_layout_and_speculative_lengths(self):
        for physical, logical, ratio, b, s in [
            (64, 64, 1, 1, 1),
            (128, 128, 1, 2, 6),
            (128, 64, 2, 2, 6),
            (128, 64, 2, 3, 1),
            (64, 64, 2, 2, 6),
        ]:
            with self.subTest(
                physical=physical, logical=logical, ratio=ratio, b=b, s=s
            ):
                _check_paged_owner_layout(physical, logical, ratio, b, s)

    def test_graph_replay_updates_live_lengths_and_owner_table(self):
        for logical, ratio in [(128, 1), (64, 2)]:
            with self.subTest(logical=logical, ratio=ratio):
                _check_graph_replay(logical, ratio)

    def test_unallocated_and_negative_table_rows_ignore_poison_block_zero(self):
        for logical, ratio in [(128, 1), (64, 2)]:
            with self.subTest(logical=logical, ratio=ratio):
                _check_unallocated_table_rows(logical, ratio)

    def test_missing_backend_and_device_gates(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score

        with patch.object(
            _indexer_score, "has_fp8_fp4_paged_mqa_logits", return_value=False
        ):
            self.assertFalse(fused.is_supported(torch.device("cuda"), 128))
        self.assertFalse(fused.is_supported(torch.device("cpu"), 128))
        self.assertFalse(fused.is_supported(torch.device("cuda"), 32))
        self.assertFalse(fused.is_supported(torch.device("cuda"), 128, num_heads=64))
        with patch.object(
            _indexer_score, "has_fp8_fp4_paged_mqa_logits", return_value=True
        ):
            with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                self.assertFalse(fused.is_supported(torch.device("cuda"), 64))
                self.assertFalse(fused.is_supported(torch.device("cuda"), 128))
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ):
                self.assertTrue(fused.is_supported(torch.device("cuda"), 64))
                self.assertTrue(fused.is_supported(torch.device("cuda"), 128))


if __name__ == "__main__":
    unittest.main()

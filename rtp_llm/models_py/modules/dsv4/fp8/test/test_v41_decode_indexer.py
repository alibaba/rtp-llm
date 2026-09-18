"""V4.1 paged decode scoring: real packed layout and changing graph inputs."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_indexer as fused
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    dequantize_indexer_k,
    quantize_indexer_k,
)


def make_case(b, s, capacity, physical, logical, ratio, seed=42):
    """Owner pages are permuted; unused rows/scales contain NaNs, not zeros."""
    torch.manual_seed(seed)
    device = "cuda"
    blocks = (capacity + logical - 1) // logical
    table = torch.randperm(b * blocks, device=device).add_(1).view(b, blocks).int()
    pool = torch.full(
        (b * blocks + 1, physical, 132), 127, dtype=torch.uint8, device=device
    )
    ids = torch.arange(capacity, device=device)
    slots = table[:, ids // logical].long() * physical + ids.remainder(logical)
    keys = torch.randn(b, capacity, 128, dtype=torch.bfloat16, device=device) * 0.4
    quantize_indexer_k(keys.flatten(0, 1), slots.flatten(), pool)
    # Interpret the documented physical bytes independently of the dequant
    # kernel. Re-quantizing with PyTorch would change rare FP8 subnormal ties;
    # both old and new production scorers consume these existing cache bytes.
    raw_blocks = pool.view(-1, physical * 132)
    packed_q = (
        raw_blocks[:, : physical * 128]
        .view(-1, physical, 128)
        .view(torch.float8_e4m3fn)
    )
    packed_scales = raw_blocks[:, physical * 128 :].view(torch.float32)
    physical_ids, offsets = slots // physical, slots % physical
    keys_ref = (
        packed_q[physical_ids, offsets].float()
        * packed_scales[physical_ids, offsets, None]
    )
    restored = dequantize_indexer_k(pool, slots.flatten()).view_as(keys_ref)
    torch.testing.assert_close(restored, keys_ref, rtol=2e-6, atol=2e-6)
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


def reference_q(q, freqs):
    value = q.clone()
    b, s, h, _ = q.shape
    pairs = torch.view_as_complex(value[..., -64:].float().reshape(b * s, h, 32, 2))
    rotated = torch.view_as_real(pairs * freqs[:, None]).flatten(-2)
    value[..., -64:] = rotated.reshape(b, s, h, 64).to(torch.bfloat16)
    value = value.float()
    scale = (value.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (value / scale).to(torch.float8_e4m3fn), scale


def reference(case, keys):
    q, scale = reference_q(case["q"], case["freqs_cis"])
    q = q.float() * scale
    logits = torch.einsum("bshd,bkd->bshk", q, keys).relu_()
    logits = (logits * case["weights"][..., None]).sum(2)
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


class V41DecodeIndexerTest(unittest.TestCase):
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

    def test_disabled_and_device_gates(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score

        with patch.dict(os.environ, {"DSV41_FUSED_DECODE_INDEXER": "0"}):
            self.assertFalse(fused.is_supported(torch.device("cuda"), 128))
        with patch.dict(os.environ, {"DSV41_FUSED_DECODE_INDEXER": "1"}):
            self.assertFalse(fused.is_supported(torch.device("cpu"), 128))
            self.assertFalse(fused.is_supported(torch.device("cuda"), 32))
            self.assertFalse(
                fused.is_supported(torch.device("cuda"), 128, num_heads=64)
            )
            with patch.object(
                _indexer_score, "has_fp8_paged_mqa_logits", return_value=True
            ):
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=(9, 0)
                ):
                    self.assertTrue(fused.is_supported(torch.device("cuda"), 64))
                    self.assertFalse(fused.is_supported(torch.device("cuda"), 128))
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 3)
                ):
                    self.assertTrue(fused.is_supported(torch.device("cuda"), 64))
                    self.assertTrue(fused.is_supported(torch.device("cuda"), 128))

    def test_rejects_padded_cache_and_fp32_weight_loss(self):
        q = torch.empty(1, 6, 32, 128, dtype=torch.bfloat16)
        weights = torch.empty(1, 6, 32, dtype=torch.bfloat16)
        freqs = torch.empty(6, 32, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "remain FP32"):
            fused.prepare_indexer_q(q, weights, freqs)
        pool = torch.empty(2, 128, 136, dtype=torch.uint8)[..., :132]
        with self.assertRaisesRegex(ValueError, "no block padding"):
            fused.score_decode_indexer(
                q,
                weights,
                freqs,
                pool,
                torch.zeros(1, 1, dtype=torch.int32),
                torch.ones(1, 6, dtype=torch.int32),
                max_ctx_len=128,
            )


if __name__ == "__main__":
    unittest.main()

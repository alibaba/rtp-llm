"""Independent checks of the E2M1/UE8M0 indexer storage contract on SM120."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.dsv4_indexer_cache import (
    indexer_entry_bytes,
    validate_fp4_indexer_config,
)
from rtp_llm.models_py.modules.dsv4.fp4.quantization import (
    dequantize_rows,
    quantize_q_rope,
    quantize_q_rope_packed,
    quantize_rows,
)
from rtp_llm.models_py.modules.dsv4.fp4.score import (
    gather_cache,
    paged_score,
    prefill_score,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)
from rtp_llm.ops import SpeculativeType


def reference_hadamard(x):
    y = x.float()
    width = 1
    while width < 128:
        grouped = y.reshape(*x.shape[:-1], -1, 2, width)
        low, high = grouped.unbind(-2)
        y = torch.stack((low + high, low - high), dim=-2).reshape_as(x)
        width *= 2
    return y * (128**-0.5)


def reference_query_rope(q, freqs):
    """Emulate the FP4 query's FP32 product/FMA order with FP64 intermediates."""
    batch, seq, heads, _ = q.shape
    rotated = q.float().clone()
    pairs = q[..., 64:].double().reshape(batch * seq, heads, 32, 2)
    x, y = pairs[..., 0], pairs[..., 1]
    real, imag = freqs.real[:, None].double(), freqs.imag[:, None].double()
    rope = torch.stack(
        (
            (x * real - (y * imag).float().double()).float(),
            (x * imag + (y * real).float().double()).float(),
        ),
        dim=-1,
    )
    rotated[..., 64:] = rope.reshape(batch, seq, heads, 64)
    return rotated


def reference_quantize(x):
    grouped = x.float().reshape(*x.shape[:-1], 4, 32)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 6))
    )
    normalized = grouped / scale.unsqueeze(-1)
    thresholds = x.new_tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
    )
    codes = torch.bucketize(normalized.abs().contiguous(), thresholds).to(torch.uint8)
    codes |= ((normalized < 0) & (codes != 0)).to(torch.uint8) * 8
    codes = codes.reshape_as(x)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    scales = (torch.log2(scale) + 127).to(torch.uint8)
    levels = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    restored = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    restored *= scale.repeat_interleave(32, dim=-1)
    return packed, scales, restored


def write_reference_pages(packed, scales, entries=64):
    blocks = (packed.shape[0] + entries - 1) // entries
    pool = torch.full(
        (blocks + 1, entries, 68), 0xAB, dtype=torch.uint8, device=packed.device
    )
    for block in range(blocks):
        lo, hi = block * entries, min((block + 1) * entries, packed.shape[0])
        page = pool[block + 1].flatten()
        page[: entries * 64].view(entries, 64)[: hi - lo] = packed[lo:hi]
        page[entries * 64 :].view(entries, 4)[: hi - lo] = scales[lo:hi]
    return pool


class Fp4ConfigTest(unittest.TestCase):
    def test_opt_in_and_guards(self):
        valid = dict(
            model_type="deepseek_v4",
            speculative_type=SpeculativeType.NONE,
            generation_slots=0,
            reuse_cache=False,
            tokens_per_block=256,
            kernel_tokens_per_block=256,
        )
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(indexer_entry_bytes(), 132)
            validate_fp4_indexer_config(**valid)
        with patch.dict(os.environ, {"DSV4_INDEXER_CACHE_MODE": "fp4"}, clear=True):
            self.assertEqual(indexer_entry_bytes(), 68)
            validate_fp4_indexer_config(**valid)
            for key, value in (
                ("speculative_type", SpeculativeType.DSPARK),
                ("generation_slots", 1),
                ("reuse_cache", True),
                ("tokens_per_block", 128),
                ("kernel_tokens_per_block", 128),
                ("model_type", "other"),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_fp4_indexer_config(**{**valid, key: value})
            with patch.dict(
                os.environ, {"RTP_LLM_RELEASE_PROFILE": "qualified"}
            ), self.assertRaises(ValueError):
                validate_fp4_indexer_config(**valid)
        with patch.dict(
            os.environ, {"DSV4_INDEXER_CACHE_MODE": "invalid"}
        ), self.assertRaises(ValueError):
            indexer_entry_bytes()

    def test_real_speculation_and_generation_slots(self):
        valid = dict(
            model_type="deepseek_v4",
            speculative_type=SpeculativeType.DSPARK,
            generation_slots=3,
            reuse_cache=False,
            tokens_per_block=256,
            kernel_tokens_per_block=256,
        )
        with patch.dict(os.environ, {"DSV4_INDEXER_CACHE_MODE": "fp4"}, clear=True):
            validate_fp4_indexer_config(**valid)
            for mode in (
                SpeculativeType.NONE,
                SpeculativeType.DSPARK,
                SpeculativeType.MTP,
            ):
                for slots in (0, 1, 2, 3, 4):
                    config = dict(valid, speculative_type=mode, generation_slots=slots)
                    with self.subTest(mode=mode, slots=slots):
                        if (mode, slots) in (
                            (SpeculativeType.NONE, 0),
                            (SpeculativeType.DSPARK, 3),
                        ):
                            validate_fp4_indexer_config(**config)
                        else:
                            with self.assertRaises(ValueError):
                                validate_fp4_indexer_config(**config)
            for key, value in (
                ("reuse_cache", True),
                ("tokens_per_block", 128),
                ("kernel_tokens_per_block", 128),
                ("model_type", "other"),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_fp4_indexer_config(**dict(valid, **{key: value}))
            with patch.dict(
                os.environ, {"RTP_LLM_RELEASE_PROFILE": "qualified"}
            ), self.assertRaises(ValueError):
                validate_fp4_indexer_config(**valid)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Fp4IndexerContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)
        self.device = torch.device("cuda")

    def assert_quantized(self, values, *, rotate=False, round_bf16=False):
        reference = reference_hadamard(values) if rotate else values.float()
        if round_bf16:
            reference = reference.bfloat16().float()
        expected, expected_scales, restored = reference_quantize(reference)
        actual, scales = quantize_rows(values, rotate=rotate, round_bf16=round_bf16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            dequantize_rows(actual, scales, dtype=torch.float32),
            restored,
            rtol=0,
            atol=0,
        )

    def test_midpoints_sign_zero_and_scale_ceil(self):
        thresholds = torch.tensor(
            [0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6], device=self.device
        )
        values = torch.cat((thresholds, -thresholds, thresholds + 1e-5))
        rows = torch.zeros((5, 4, 32), device=self.device)
        rows[:, :, :30] = values
        rows[:, :, 31] = 6
        rows *= torch.tensor([0.25, 0.5, 1, 2, 1.00001], device=self.device)[
            :, None, None
        ]
        self.assert_quantized(rows.reshape(-1, 128))
        self.assert_quantized(torch.zeros((1, 128), device=self.device))
        self.assert_quantized(torch.randn((257, 128), device=self.device) * 0.001)

    def test_hadamard_basis_and_random(self):
        self.assert_quantized(torch.eye(128, device=self.device), rotate=True)
        self.assert_quantized(torch.randn((131, 128), device=self.device), rotate=True)
        self.assert_quantized(
            torch.randn((129, 128), device=self.device, dtype=torch.bfloat16),
            rotate=True,
            round_bf16=True,
        )

    def test_query_rope_rounding_and_weight_scale(self):
        for batch, seq in ((1, 17), (3, 1), (2, 3)):
            with self.subTest(batch=batch, seq=seq):
                q = torch.randn(
                    (batch, seq, 64, 128), device=self.device, dtype=torch.bfloat16
                )
                w = torch.randn(q.shape[:-1], device=self.device, dtype=torch.bfloat16)
                angles = torch.randn((batch * seq, 32), device=self.device)
                freqs = torch.polar(torch.ones_like(angles), angles)
                rotated = reference_query_rope(q, freqs)
                transformed = reference_hadamard(rotated.bfloat16()).bfloat16().float()
                expected, expected_scales, restored = reference_quantize(transformed)
                actual, folded, packed, scales = quantize_q_rope(
                    q,
                    w,
                    freqs,
                    64,
                    weight_scale=128**-0.5 * 64**-0.5,
                    return_packed=True,
                )
                torch.testing.assert_close(
                    packed, expected.reshape(-1, 64), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales, expected_scales.reshape(-1, 4), rtol=0, atol=0
                )
                torch.testing.assert_close(actual, restored.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(
                    folded, w.float() * (128**-0.5 * 64**-0.5), rtol=0, atol=0
                )
                plain = quantize_q_rope(
                    q, w, freqs, 64, weight_scale=128**-0.5 * 64**-0.5
                )
                torch.testing.assert_close(plain[0], actual, rtol=0, atol=0)
                native_q, native_weights, native_scales = quantize_q_rope_packed(
                    q, w, freqs, 64, weight_scale=128**-0.5 * 64**-0.5
                )
                torch.testing.assert_close(native_q, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    native_scales, expected_scales, rtol=0, atol=0
                )
                torch.testing.assert_close(native_weights, folded, rtol=0, atol=0)

    def test_gather_multiple_requests_and_page_boundaries(self):
        x = torch.randn((320, 128), device=self.device)
        packed, scales, _ = reference_quantize(x)
        pool = write_reference_pages(packed, scales)
        table = torch.tensor(
            [[2, 1, 3], [4, 5, 0]], dtype=torch.int32, device=self.device
        )
        lengths = torch.tensor([0, 129, 194], dtype=torch.int32, device=self.device)
        out, out_scales = torch.empty(
            (194, 64), dtype=torch.uint8, device=self.device
        ), torch.empty((194, 4), dtype=torch.uint8, device=self.device)
        gather_cache(pool, out, out_scales, table, lengths)
        ids = torch.cat(
            (
                torch.arange(64, 128),
                torch.arange(64),
                torch.arange(128, 129),
                torch.arange(192, 257),
            )
        ).to(self.device)
        torch.testing.assert_close(out, packed[ids], rtol=0, atol=0)
        torch.testing.assert_close(out_scales, scales[ids], rtol=0, atol=0)
        self.assertTrue(torch.all(pool[0] == 0xAB).item())

    def test_query_rope_contraction_and_output_modes(self):
        for seed in (20261014, 20261015):
            with self.subTest(seed=seed):
                generator = torch.Generator(device=self.device).manual_seed(seed)
                q = torch.randn(
                    (1, 257, 64, 128), device=self.device, generator=generator
                ).bfloat16()
                weights = torch.randn(
                    q.shape[:-1], device=self.device, generator=generator
                ).bfloat16()
                angles = (
                    torch.randn((257, 32), device=self.device, generator=generator)
                    * 10000
                )
                freqs = torch.polar(torch.ones_like(angles), angles)
                rotated = reference_query_rope(q.cpu(), freqs.cpu())
                transformed = reference_hadamard(rotated.bfloat16()).bfloat16()
                expected, expected_scales, restored = reference_quantize(transformed)
                plain, _ = quantize_q_rope(q, weights, freqs, 64)
                both, _, packed, scales = quantize_q_rope(
                    q, weights, freqs, 64, return_packed=True
                )
                direct, _, direct_scales = quantize_q_rope_packed(q, weights, freqs, 64)
                for actual in (plain, both, dequantize_rows(direct, direct_scales)):
                    torch.testing.assert_close(
                        actual.cpu(), restored.bfloat16(), rtol=0, atol=0
                    )
                for actual in (packed.reshape_as(expected), direct):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                for actual in (scales.reshape_as(expected_scales), direct_scales):
                    torch.testing.assert_close(
                        actual.cpu(), expected_scales, rtol=0, atol=0
                    )

    def test_paged_and_prefill_scores(self):
        packed, scales, keys = reference_quantize(
            torch.randn((257, 128), device=self.device)
        )
        pool = write_reference_pages(packed, scales)
        q_p, q_s, q_ref = reference_quantize(
            torch.randn((2, 3, 64, 128), device=self.device)
        )
        q = dequantize_rows(q_p, q_s)
        weights = torch.randn((6, 64), device=self.device) * 0.01
        table = torch.arange(1, 6, dtype=torch.int32, device=self.device).repeat(2, 1)
        lengths = torch.tensor(
            [[0, 1, 63], [64, 129, 257]], dtype=torch.int32, device=self.device
        )
        reference = torch.einsum("rhd,td->rht", q_ref.reshape(6, 64, 128), keys)
        reference = (reference.relu() * weights[:, :, None]).sum(1)
        for cache in (pool, pool.flatten(0, 1)):
            actual = paged_score(
                q, weights, cache, table, lengths, block_size=64, max_ctx_len=257
            )
            expected = reference.masked_fill(
                torch.arange(257, device=self.device)[None, :]
                >= lengths.flatten()[:, None],
                -torch.inf,
            )
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        starts = torch.tensor([0, 0, 3, 17, 64, 129], device=self.device)
        actual = prefill_score(
            q.reshape(6, 64, 128),
            weights,
            packed,
            scales,
            starts,
            lengths.flatten(),
            clean_logits=True,
        )
        cols = torch.arange(257, device=self.device)[None, :]
        expected = reference.masked_fill(
            (cols < starts[:, None]) | (cols >= lengths.flatten()[:, None]), -torch.inf
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_raw_state_and_fp8_default(self):
        kv = torch.randn((8, 256), device=self.device)
        score = torch.randn_like(kv)
        ape = torch.randn((4, 256), device=self.device)
        pos = torch.arange(8, dtype=torch.int64, device=self.device)
        slots = pos + 8
        for raw in (False, True):
            state = torch.zeros((2, 8, 512), device=self.device)
            run_save_partial_states(
                kv, score, ape, pos, state, slots, 4, raw_scores=raw
            )
            torch.testing.assert_close(state[1, :, :256], kv, rtol=0, atol=0)
            torch.testing.assert_close(
                state[1, :, 256:],
                score if raw else score + ape[pos % 4],
                rtol=0,
                atol=0,
            )
            self.assertEqual(state[0].count_nonzero().item(), 0)

    def test_compact_transport_preserves_split_page_bytes(self):
        from rtp_llm.models_py.modules.dsv4.fp8._compact_cp_runtime import split_pool
        from rtp_llm.models_py.modules.dsv4.fp8._compact_cp_transport_triton import (
            pack_compact_bytes,
            scatter_compact_bytes,
        )

        for entry_bytes, head_dim in ((68, 128), (132, 128), (584, 512)):
            with self.subTest(entry_bytes=entry_bytes):
                source = torch.randint(
                    256, (3, 64, entry_bytes), dtype=torch.uint8, device=self.device
                )
                target = torch.full_like(source, 0xAB)
                data, scales, layout = split_pool(source, head_dim)
                target_data, target_scales, _ = split_pool(target, head_dim)
                slots = torch.tensor(
                    [64, 127, 128, 130], dtype=torch.int64, device=self.device
                )
                logical = torch.tensor(
                    [3, 255, 259, 267], dtype=torch.int64, device=self.device
                )
                packet = pack_compact_bytes(data, scales, slots, logical, 4096)
                torch.testing.assert_close(
                    packet[:, :8].contiguous().view(torch.int64).flatten(),
                    logical + 4096,
                )
                torch.testing.assert_close(
                    packet[:, 8 : 8 + layout.data_bytes],
                    data[slots // 64, slots % 64],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    packet[:, 8 + layout.data_bytes :],
                    scales[slots // 64, slots % 64],
                    rtol=0,
                    atol=0,
                )
                scatter_compact_bytes(target_data, target_scales, packet, slots)
                torch.testing.assert_close(
                    target_data[slots // 64, slots % 64],
                    data[slots // 64, slots % 64],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    target_scales[slots // 64, slots % 64],
                    scales[slots // 64, slots % 64],
                    rtol=0,
                    atol=0,
                )
                self.assertTrue(torch.all(target[0] == 0xAB).item())

    def test_graph_replay_uses_updated_query_and_lengths(self):
        packed, scales, _ = reference_quantize(
            torch.randn((131, 128), device=self.device)
        )
        pool = write_reference_pages(packed, scales)
        q = torch.randn((2, 1, 64, 128), device=self.device, dtype=torch.bfloat16)
        weights = torch.randn((2, 1, 64), device=self.device, dtype=torch.bfloat16)
        freqs = torch.ones((2, 32), device=self.device, dtype=torch.complex64)
        table = torch.tensor(
            [[1, 2, 3], [1, 2, 3]], device=self.device, dtype=torch.int32
        )
        lengths = torch.tensor([131, 65], device=self.device, dtype=torch.int32)

        def run():
            reconstructed, folded = quantize_q_rope(q, weights, freqs, 64)
            return paged_score(
                reconstructed,
                folded.view(2, 64),
                pool,
                table,
                lengths,
                block_size=64,
                max_ctx_len=131,
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        q.mul_(2)
        lengths.copy_(torch.tensor([63, 129], device=self.device, dtype=torch.int32))
        graph.replay()
        actual = result.clone()
        torch.testing.assert_close(actual, run(), rtol=0, atol=0)

    def test_compressor_continuation_matches_full_prefill(self):
        count = 272
        kv = torch.randn((count, 256), device=self.device)
        score = torch.randn_like(kv)
        ape = torch.randn((4, 256), device=self.device)
        norm = torch.randn(128, device=self.device)
        angles = torch.randn((count, 32), device=self.device)
        freq = torch.polar(torch.ones_like(angles), angles)
        cos_sin, _ = build_cos_sin_cache(freq)
        table = torch.tensor([[1, 2]], device=self.device, dtype=torch.int32)

        def compress(state, cache, begin, end, cached):
            positions = torch.arange(begin, end, device=self.device, dtype=torch.int64)
            slots = (positions // 256 + 1) * 8 + positions % 8
            target = torch.where((positions + 1) % 4 == 0, 64 + positions // 4, -1)
            run_fused_compress_kv_write(
                state,
                torch.zeros_like(positions, dtype=torch.int32),
                positions,
                slots,
                table,
                norm,
                1e-6,
                cos_sin,
                cache,
                target,
                kv[begin:end],
                score[begin:end],
                ape,
                begin,
                cached,
                head_dim=128,
                rope_head_dim=64,
                compress_ratio=4,
                overlap=True,
                state_tokens_per_block=256,
                fp4_indexer=True,
            )

        expected = torch.full((3, 64, 68), 0xAB, dtype=torch.uint8, device=self.device)
        compress(
            torch.zeros((3, 8, 512), device=self.device), expected, 0, count, False
        )
        for boundary in range(3, count, 4):
            begin = boundary - 7
            positions = torch.arange(max(begin, 0), boundary + 1, device=self.device)
            columns = torch.arange(128, device=self.device)[None, :]
            columns = columns + ((positions - begin >= 4).long() * 128)[:, None]
            values = kv[positions[:, None], columns]
            logits = (
                score[positions[:, None], columns]
                + ape[(positions % 4)[:, None], columns]
            )
            compressed = (values * logits.softmax(0)).sum(0)
            normalized = (
                compressed * torch.rsqrt(compressed.square().mean() + 1e-6) * norm
            )
            rotated = normalized.clone()
            rotated[64:] = torch.view_as_real(
                torch.view_as_complex(normalized[64:].reshape(32, 2))
                * freq[boundary - 3]
            ).flatten()
            reference, reference_scale, _ = reference_quantize(
                reference_hadamard(rotated)
            )
            block, offset = divmod(boundary // 4, 64)
            page = expected[block + 1].flatten()
            torch.testing.assert_close(
                page[offset * 64 : (offset + 1) * 64], reference, rtol=0, atol=0
            )
            torch.testing.assert_close(
                page[4096 + offset * 4 : 4100 + offset * 4],
                reference_scale,
                rtol=0,
                atol=0,
            )
        for split in (3, 4, 7, 127, 128, 129, 255, 256, 257):
            with self.subTest(split=split):
                state = torch.zeros((3, 8, 512), device=self.device)
                cache = torch.full_like(expected, 0xAB)
                compress(state, cache, 0, split, False)
                tail = torch.arange(max(0, split - 8), split, device=self.device)
                slots = (tail // 256 + 1) * 8 + tail % 8
                run_save_partial_states(
                    kv[tail], score[tail], ape, tail, state, slots, 4, raw_scores=True
                )
                end = min(count, ((split + 3) // 4) * 4 + 8)
                for pos in range(split, end):
                    position = torch.tensor(
                        [pos], dtype=torch.int64, device=self.device
                    )
                    slot = (position // 256 + 1) * 8 + position % 8
                    run_save_partial_states(
                        kv[pos : pos + 1],
                        score[pos : pos + 1],
                        ape,
                        position,
                        state,
                        slot,
                        4,
                        raw_scores=True,
                    )
                    compress(state, cache, pos, pos + 1, True)
                for token in range(end // 4):
                    block, offset = divmod(token, 64)
                    actual_page, expected_page = (
                        cache[block + 1].flatten(),
                        expected[block + 1].flatten(),
                    )
                    torch.testing.assert_close(
                        actual_page[offset * 64 : (offset + 1) * 64],
                        expected_page[offset * 64 : (offset + 1) * 64],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        actual_page[4096 + offset * 4 : 4100 + offset * 4],
                        expected_page[4096 + offset * 4 : 4100 + offset * 4],
                        rtol=0,
                        atol=0,
                    )
                self.assertTrue(torch.all(cache[0] == 0xAB).item())


if __name__ == "__main__":
    unittest.main()

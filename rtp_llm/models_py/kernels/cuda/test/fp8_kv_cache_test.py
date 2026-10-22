import unittest
from unittest import SkipTest

import torch

import rtp_llm.ops  # isort:skip
from rtp_llm.ops import RopeConfig, RopeStyle  # isort:skip
from rtp_llm.ops.compute_ops import (  # isort:skip
    gather_and_dequantize_fp8_kv_cache,
    quantize_and_write_fp8_kv_cache,
    rtp_llm_ops,
)


class Fp8KvCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    @staticmethod
    def _quantize_row(row: torch.Tensor) -> tuple[torch.Tensor, float]:
        row = row.float()
        max_abs = row.abs().max().item()
        scale = 1.0 if max_abs == 0.0 else max_abs / 448.0
        quantized = (row / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return quantized, scale

    def _run_layout(self, subdivision: int, layout: str) -> None:
        torch.manual_seed(2026 + subdivision)
        physical_pages = 3
        physical_page_size = 8
        kernel_page_size = physical_page_size // subdivision
        heads = 3
        head_dim = 17
        input_dtype = torch.float16 if layout == "physical" else torch.bfloat16
        output_dtype = torch.bfloat16 if layout == "physical" else torch.float16

        target_pages = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32)
        token_offsets = torch.tensor([7, 0, 3, 5, 2], dtype=torch.int32)
        token_count = target_pages.numel()

        # K and V intentionally use different magnitudes. Explicit zeros and
        # outliers exercise independent row scales and the zero-row contract.
        k_cpu = (torch.randn(token_count, heads, head_dim) * 0.125).to(input_dtype)
        v_cpu = (torch.randn(token_count, heads, head_dim) * 24.0).to(input_dtype)
        k_cpu[1, 1].zero_()
        v_cpu[3, 2].zero_()
        k_cpu[2, 0, 5] = 1000.0
        v_cpu[0, 2, 9] = -2000.0

        if layout == "physical":
            cache_shape = (physical_pages, 2, heads, physical_page_size, head_dim)
            scale_shape = (physical_pages, 2 * heads * physical_page_size)
        else:
            kernel_pages = physical_pages * subdivision
            cache_shape = (kernel_pages, 2, heads, kernel_page_size, head_dim)
            scale_shape = (kernel_pages, 2 * heads * kernel_page_size)

        sentinel_q = 3.0
        sentinel_scale = 13.0
        cache = torch.full(cache_shape, sentinel_q, dtype=torch.float32).to(
            device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.full(
            scale_shape, sentinel_scale, dtype=torch.float32, device="cuda"
        )

        quantize_and_write_fp8_kv_cache(
            k_cpu.cuda(),
            v_cpu.cuda(),
            cache,
            scales,
            target_pages.cuda(),
            token_offsets.cuda(),
            physical_page_size,
            kernel_page_size,
            subdivision,
        )

        expected_scales = torch.full(scale_shape, sentinel_scale, dtype=torch.float32)
        for n in range(token_count):
            physical_page = int(target_pages[n])
            physical_token = int(token_offsets[n])
            if layout == "physical":
                storage_page = physical_page
                storage_token = physical_token
            else:
                storage_page = (
                    physical_page * subdivision + physical_token // kernel_page_size
                )
                storage_token = physical_token % kernel_page_size
            scale_view = expected_scales.view(cache_shape[0], 2, heads, cache_shape[3])
            for kv, source in enumerate((k_cpu, v_cpu)):
                for head in range(heads):
                    _, scale = self._quantize_row(source[n, head])
                    scale_view[storage_page, kv, head, storage_token] = scale

        cache_cpu = cache.cpu()
        scales_cpu = scales.cpu()
        torch.testing.assert_close(scales_cpu, expected_scales, rtol=1e-6, atol=1e-7)

        # Gather every kernel page in a deliberately non-monotonic order.
        source_kernel_pages = torch.tensor(
            list(reversed(range(physical_pages * subdivision))),
            dtype=torch.int64,
            device="cuda",
        )
        output = torch.empty(
            source_kernel_pages.numel(),
            2,
            heads,
            kernel_page_size,
            head_dim,
            dtype=output_dtype,
            device="cuda",
        )
        gather_and_dequantize_fp8_kv_cache(
            cache,
            scales,
            source_kernel_pages,
            output,
            physical_page_size,
            kernel_page_size,
            subdivision,
        )

        expected_output = torch.empty_like(output, device="cpu")
        scale_view = scales_cpu.view(cache_shape[0], 2, heads, cache_shape[3])
        for r, kernel_page_tensor in enumerate(source_kernel_pages.cpu()):
            kernel_page = int(kernel_page_tensor)
            if layout == "physical":
                storage_page = kernel_page // subdivision
                token_start = (kernel_page % subdivision) * kernel_page_size
            else:
                storage_page = kernel_page
                token_start = 0
            q = cache_cpu[
                storage_page, :, :, token_start : token_start + kernel_page_size, :
            ].float()
            s = scale_view[
                storage_page, :, :, token_start : token_start + kernel_page_size
            ].unsqueeze(-1)
            expected_output[r] = (q * s).to(output_dtype)
        tolerance = 8e-3 if output_dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(
            output.cpu(), expected_output, rtol=tolerance, atol=tolerance
        )

        # Most rows were never written: both the FP8 sentinel and its scale must
        # survive, and gather must dequantize them rather than clearing them.
        self.assertEqual(cache[0, 0, 0, 1, 0].float().item(), sentinel_q)
        self.assertEqual(
            scales.view(cache_shape[0], 2, heads, cache_shape[3])[0, 0, 0, 1].item(),
            sentinel_scale,
        )

        # Explicit zero rows use scale=1 and round-trip to zero.
        if layout == "physical":
            zero_page, zero_token = 0, 0
        else:
            zero_page, zero_token = 0, 0
        self.assertEqual(
            scales.view(cache_shape[0], 2, heads, cache_shape[3])[
                zero_page, 0, 1, zero_token
            ].item(),
            1.0,
        )
        self.assertTrue(
            torch.count_nonzero(cache[zero_page, 0, 1, zero_token].float()).item() == 0
        )

        # Validate quantize/dequantize round-trip error at every written row.
        scales_cpu = scales_cpu.view(cache_shape[0], 2, heads, cache_shape[3])
        for n in range(token_count):
            physical_page = int(target_pages[n])
            physical_token = int(token_offsets[n])
            if layout == "physical":
                storage_page, storage_token = physical_page, physical_token
            else:
                storage_page = (
                    physical_page * subdivision + physical_token // kernel_page_size
                )
                storage_token = physical_token % kernel_page_size
            for kv, source in enumerate((k_cpu, v_cpu)):
                restored = cache_cpu[
                    storage_page, kv, :, storage_token
                ].float() * scales_cpu[storage_page, kv, :, storage_token].unsqueeze(-1)
                torch.testing.assert_close(
                    restored, source[n].float(), rtol=0.13, atol=0.02
                )

    def test_nonfinite_values_are_sanitized(self) -> None:
        k = torch.tensor(
            [[[float("nan"), float("inf"), -float("inf"), 2.0]]],
            device="cuda",
            dtype=torch.float16,
        )
        v = torch.tensor(
            [[[float("nan"), float("inf"), -float("inf"), 4.0]]],
            device="cuda",
            dtype=torch.float16,
        )
        cache = torch.empty((1, 2, 1, 1, 4), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((1, 2), device="cuda", dtype=torch.float32)
        page_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0], device="cuda", dtype=torch.int32)

        quantize_and_write_fp8_kv_cache(k, v, cache, scales, page_ids, offsets, 1, 1, 1)
        output = torch.empty((1, 2, 1, 1, 4), device="cuda", dtype=torch.float16)
        gather_and_dequantize_fp8_kv_cache(cache, scales, page_ids, output, 1, 1, 1)

        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(
            output.cpu().float(),
            torch.tensor([[[[[0.0, 2.0, -2.0, 2.0]]], [[[0.0, 4.0, -4.0, 4.0]]]]]),
            rtol=0,
            atol=0,
        )

    def test_physical_and_existing_kernel_page_layouts(self) -> None:
        for subdivision in (1, 2, 4):
            for layout in ("physical", "kernel"):
                with self.subTest(subdivision=subdivision, layout=layout):
                    self._run_layout(subdivision, layout)

    @staticmethod
    def _rope_reference(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = tensor.size(-1)
        inv_freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=tensor.device))
            * torch.arange(0, dim, 2, device=tensor.device, dtype=torch.float32)
            / dim
        )
        angles = positions.float()[:, None] * inv_freq[None, :]
        cos = angles.cos()[:, None, :]
        sin = angles.sin()[:, None, :]
        first = tensor[..., : dim // 2].float()
        second = tensor[..., dim // 2 :].float()
        return torch.cat(
            [first * cos - second * sin, second * cos + first * sin], dim=-1
        ).to(tensor.dtype)

    def test_fused_decode_prepare_matches_standalone_oracle(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 128):
                for rope_style in (RopeStyle.No, RopeStyle.Base):
                    with self.subTest(
                        dtype=dtype, head_dim=head_dim, rope_style=rope_style
                    ):
                        torch.manual_seed(17 + head_dim)
                        num_tokens, q_heads, kv_heads, page_size = 3, 4, 2, 4
                        qkv = torch.randn(
                            num_tokens,
                            (q_heads + 2 * kv_heads) * head_dim,
                            device="cuda",
                            dtype=dtype,
                        )
                        q_raw, k_raw, v_raw = torch.split(
                            qkv,
                            [
                                q_heads * head_dim,
                                kv_heads * head_dim,
                                kv_heads * head_dim,
                            ],
                            dim=-1,
                        )
                        q_raw = q_raw.reshape(num_tokens, q_heads, head_dim)
                        k_raw = k_raw.reshape(num_tokens, kv_heads, head_dim)
                        v_raw = v_raw.reshape(num_tokens, kv_heads, head_dim)
                        k_raw[0, 0].zero_()
                        v_raw[1, 1] *= 32
                        qkv = torch.cat(
                            [q_raw.flatten(1), k_raw.flatten(1), v_raw.flatten(1)],
                            dim=-1,
                        )
                        batch_indices = torch.tensor(
                            [0, 1, 1], device="cuda", dtype=torch.int32
                        )
                        positions = torch.tensor(
                            [0, 2, 11], device="cuda", dtype=torch.int32
                        )
                        page_indptr = torch.tensor(
                            [0, 2, 5], device="cuda", dtype=torch.int32
                        )
                        page_indices = torch.tensor(
                            [6, 1, 5, 2, 6], device="cuda", dtype=torch.int32
                        )
                        cache = torch.full(
                            (7, 2, kv_heads, page_size, head_dim),
                            3.0,
                            device="cuda",
                            dtype=torch.float8_e4m3fn,
                        )
                        scales = torch.full(
                            (7, 2 * kv_heads * page_size),
                            13.0,
                            device="cuda",
                            dtype=torch.float32,
                        )
                        rope_config = RopeConfig()
                        rope_config.style = rope_style
                        rope_config.dim = head_dim
                        rope_config.base = 10000
                        q = rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                            qkv,
                            cache,
                            scales,
                            batch_indices,
                            positions,
                            page_indptr,
                            page_indices,
                            q_heads,
                            kv_heads,
                            page_size,
                            rope_config,
                            None,
                        )
                        expected_q = (
                            q_raw
                            if rope_style == RopeStyle.No
                            else self._rope_reference(q_raw, positions)
                        )
                        expected_k = (
                            k_raw
                            if rope_style == RopeStyle.No
                            else self._rope_reference(k_raw, positions)
                        )
                        torch.testing.assert_close(q, expected_q, rtol=2e-3, atol=2e-3)

                        target_pages = torch.tensor(
                            [6, 5, 6], device="cuda", dtype=torch.int32
                        )
                        target_offsets = torch.tensor(
                            [0, 2, 3], device="cuda", dtype=torch.int32
                        )
                        oracle_cache = torch.full_like(cache, 3.0)
                        oracle_scales = torch.full_like(scales, 13.0)
                        quantize_and_write_fp8_kv_cache(
                            expected_k.contiguous(),
                            v_raw.contiguous(),
                            oracle_cache,
                            oracle_scales,
                            target_pages,
                            target_offsets,
                            page_size,
                            page_size,
                            1,
                        )
                        torch.testing.assert_close(
                            cache.float(), oracle_cache.float(), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            scales, oracle_scales, rtol=1e-6, atol=1e-7
                        )

    def test_dynamic_ntk_uses_request_lengths_for_qk_and_cache(self) -> None:
        torch.manual_seed(43)
        qkv = torch.randn(4, 4 * 64, device="cuda", dtype=torch.float16)
        batch = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
        positions = torch.tensor([6, 7, 7, 8], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([8, 9], device="cuda", dtype=torch.int32)
        indptr = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        pages = torch.tensor([4, 0, 3, 1, 2], device="cuda", dtype=torch.int32)
        rope = RopeConfig()
        rope.dim, rope.max_pos, rope.scale = 64, 8, 2.0
        for style in (RopeStyle.DynamicNTK, RopeStyle.QwenDynamicNTK):
            with self.subTest(style=style):
                rope.style = style
                cache = torch.zeros(
                    (5, 2, 1, 4, 64), device="cuda", dtype=torch.float8_e4m3fn
                )
                scales = torch.ones((5, 8), device="cuda")
                output = rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                    qkv,
                    cache,
                    scales,
                    batch,
                    positions,
                    indptr,
                    pages,
                    2,
                    1,
                    4,
                    rope,
                    output_qkv=True,
                    kv_lengths=lengths,
                ).view(4, 4, 64)
                expected = qkv.view(4, 4, 64).clone()
                token_lengths = lengths[batch.long()].float()
                if style == RopeStyle.DynamicNTK:
                    factor = rope.scale * token_lengths / rope.max_pos - (
                        rope.scale - 1
                    )
                else:
                    factor = (
                        2 ** torch.ceil(torch.log2(token_lengths / rope.max_pos) + 1)
                        - 1
                    )
                bases = torch.where(
                    token_lengths > rope.max_pos,
                    rope.base * factor ** (64 / 62),
                    rope.base,
                )
                # Match apply_rope's integer conversion of the adjusted base.
                bases = bases.to(torch.int32).float()
                frequency = torch.arange(32, device="cuda").float() / 32
                angles = positions.float()[:, None] * bases[:, None] ** -frequency
                cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
                first, second = (
                    expected[:, :3, :32].float(),
                    expected[:, :3, 32:].float(),
                )
                expected[:, :3] = torch.cat(
                    (first * cosine - second * sine, second * cosine + first * sine), -1
                ).to(qkv.dtype)
                torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
                restored = cache.float() * scales.view(5, 2, 1, 4, 1)
                for token, (page, offset) in enumerate(
                    ((0, 2), (0, 3), (1, 3), (2, 0))
                ):
                    torch.testing.assert_close(
                        restored[page, :, 0, offset],
                        expected[token, 2:].float(),
                        rtol=0.13,
                        atol=0.02,
                    )

    def test_dynamic_ntk_requires_request_length_metadata(self) -> None:
        qkv = torch.zeros(2, 3 * 64, device="cuda", dtype=torch.float16)
        batch = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        positions = torch.zeros_like(batch)
        indptr = torch.tensor([0, 0, 0], device="cuda", dtype=torch.int32)
        pages = torch.empty(0, device="cuda", dtype=torch.int32)
        rope = RopeConfig()
        rope.dim, rope.max_pos = 64, 8
        for style in (RopeStyle.DynamicNTK, RopeStyle.QwenDynamicNTK):
            rope.style = style
            for lengths in (None, torch.ones(1, device="cuda", dtype=torch.int32)):
                with self.subTest(style=style, missing=lengths is None):
                    with self.assertRaisesRegex(RuntimeError, "kv_lengths"):
                        rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                            qkv,
                            None,
                            None,
                            batch,
                            positions,
                            indptr,
                            pages,
                            1,
                            1,
                            4,
                            rope,
                            output_qkv=True,
                            kv_lengths=lengths,
                        )

    def test_fused_mrope_prefill_rotates_and_writes_only_live_tokens(self) -> None:
        torch.manual_seed(42)
        heads, kv_heads, head_dim, rotary_dim, page_size = 4, 2, 64, 32, 4
        live_tokens, capacity = 3, 5
        positions = torch.tensor([3, 4, 1], device="cuda", dtype=torch.int32)
        logical_positions = torch.tensor(
            [[0, 0, 0], [7, 3, 1], [8, 4, 2]], device="cuda", dtype=torch.int32
        )
        batch_indices = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
        token_indptr = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
        page_indptr = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
        page_indices = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
        rope = RopeConfig()
        rope.style = RopeStyle.Mrope
        rope.dim = rotary_dim
        rope.index_factor = 3
        rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3 = 6, 5, 5
        rope.scale = 2.0

        for dtype, interleaved in ((torch.float16, False), (torch.bfloat16, True)):
            with self.subTest(dtype=dtype, interleaved=interleaved):
                rope.mrope_interleaved = interleaved
                qkv = torch.randn(
                    capacity,
                    (heads + 2 * kv_heads) * head_dim,
                    device="cuda",
                    dtype=dtype,
                )
                raw = qkv[:live_tokens].view(
                    live_tokens, heads + 2 * kv_heads, head_dim
                )
                expected = raw.clone()
                frequency = torch.arange(rotary_dim // 2, device="cuda")
                if interleaved:
                    axes = frequency % 3
                    axes[frequency >= 15] = 0
                else:
                    axes = torch.repeat_interleave(
                        torch.arange(3, device="cuda"),
                        torch.tensor([6, 5, 5], device="cuda"),
                    )
                angles = (
                    logical_positions[:, axes].float()
                    / rope.scale
                    * (rope.base ** (-2.0 * frequency.float() / rotary_dim))
                )
                cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
                first = raw[:, : heads + kv_heads, : rotary_dim // 2].float()
                second = raw[
                    :, : heads + kv_heads, rotary_dim // 2 : rotary_dim
                ].float()
                expected[:, : heads + kv_heads, :rotary_dim] = torch.cat(
                    (first * cosine - second * sine, second * cosine + first * sine), -1
                ).to(dtype)
                cache = torch.full(
                    (6, 2, kv_heads, page_size, head_dim),
                    3.0,
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
                scales = torch.full((6, 2 * kv_heads * page_size), 13.0, device="cuda")

                def run(payload, scale):
                    return rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                        qkv,
                        payload,
                        scale,
                        batch_indices,
                        positions,
                        page_indptr,
                        page_indices,
                        heads,
                        kv_heads,
                        page_size,
                        rope,
                        rope_position_ids=logical_positions,
                        token_indptr=token_indptr,
                        output_qkv=True,
                    )

                output = run(cache, scales).view_as(qkv)
                torch.testing.assert_close(
                    output[:live_tokens], expected.flatten(1), rtol=2e-3, atol=2e-3
                )
                torch.testing.assert_close(
                    output[live_tokens:], torch.zeros_like(output[live_tokens:])
                )
                torch.testing.assert_close(run(None, None), output, rtol=0, atol=0)
                # Partial RoPE must leave the remaining channels and all V unchanged.
                torch.testing.assert_close(
                    output[:live_tokens].view_as(raw)[:, :, rotary_dim:],
                    raw[:, :, rotary_dim:],
                    rtol=0,
                    atol=0,
                )
                expected_cache = torch.full_like(cache, 3.0)
                expected_scales = torch.full_like(scales, 13.0)
                quantize_and_write_fp8_kv_cache(
                    expected[:, heads : heads + kv_heads].contiguous(),
                    expected[:, heads + kv_heads :].contiguous(),
                    expected_cache,
                    expected_scales,
                    torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32),
                    torch.tensor([3, 0, 1], device="cuda", dtype=torch.int32),
                    page_size,
                    page_size,
                    1,
                )
                torch.testing.assert_close(
                    cache.float(), expected_cache.float(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales, expected_scales, rtol=1e-6, atol=1e-7
                )

    def test_fused_decode_padding_does_not_overwrite_live_cache(self) -> None:
        qkv = torch.randn(2, 3 * 64, device="cuda", dtype=torch.bfloat16)
        cache = torch.full(
            (1, 2, 1, 4, 64), 3.0, device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = torch.full((1, 8), 13.0, device="cuda")
        rope = RopeConfig()
        rope.style = RopeStyle.No
        batch = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        positions = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        indptr = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        pages = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

        def run():
            return rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                qkv,
                cache,
                scales,
                batch,
                positions,
                indptr,
                pages,
                1,
                1,
                4,
                rope,
                decode_input_lengths=lengths,
            )

        output = run()
        torch.testing.assert_close(output[0].flatten(), qkv[0, :64])
        torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
        restored = cache[0, :, 0, 0].float() * scales.view(2, 4)[:, :1]
        torch.testing.assert_close(
            restored, qkv[0, 64:].view(2, 64).float(), rtol=0.13, atol=0.02
        )
        payload_before, scales_before = cache.clone(), scales.clone()
        lengths.zero_()
        torch.testing.assert_close(run(), torch.zeros_like(output))
        torch.testing.assert_close(
            cache.float(), payload_before.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, scales_before, rtol=0, atol=0)

    def test_fused_decode_prepare_sanitizes_nonfinite_kv(self) -> None:
        qkv = torch.tensor(
            [
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    2.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    4.0,
                ]
            ],
            device="cuda",
            dtype=torch.float16,
        )
        cache = torch.empty((1, 2, 1, 1, 4), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((1, 2), device="cuda", dtype=torch.float32)
        zero = torch.tensor([0], device="cuda", dtype=torch.int32)
        page_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        rope_config = RopeConfig()
        rope_config.style = RopeStyle.No

        q = rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
            qkv,
            cache,
            scales,
            zero,
            zero,
            page_indptr,
            zero,
            1,
            1,
            1,
            rope_config,
            None,
        )

        torch.testing.assert_close(
            q.cpu().flatten().float(), torch.arange(4, dtype=torch.float32)
        )
        torch.testing.assert_close(
            scales.cpu(), torch.tensor([[2.0 / 448.0, 4.0 / 448.0]])
        )
        restored = cache.float() * scales.view(1, 2, 1, 1, 1)
        torch.testing.assert_close(
            restored.cpu().flatten(),
            torch.tensor([0.0, 2.0, -2.0, 2.0, 0.0, 4.0, -4.0, 4.0]),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()

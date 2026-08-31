import unittest

import torch

from rtp_llm.ops import compute_ops


class TestConcatAndCacheGlm53Fp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.device = torch.device("cuda:0")

    def test_zero_rope_528_byte_layout(self) -> None:
        torch.manual_seed(20260830)
        num_tokens = 5
        block_size = 8
        kv_lora_rank = 512
        slot_mapping = torch.tensor(
            [0, 7, 8, 13, 22], dtype=torch.int64, device=self.device
        )

        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                kv_c = torch.randn(
                    num_tokens,
                    kv_lora_rank,
                    dtype=dtype,
                    device=self.device,
                )
                k_pe = torch.empty(num_tokens, 0, dtype=dtype, device=self.device)
                kv_cache = torch.zeros(
                    3,
                    block_size,
                    528,
                    dtype=torch.uint8,
                    device=self.device,
                )
                scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

                compute_ops.concat_and_cache_mla(
                    kv_c,
                    k_pe,
                    kv_cache,
                    slot_mapping,
                    "fp8_glm53_mla",
                    scale,
                )

                for token_idx, slot_idx in enumerate(slot_mapping.tolist()):
                    entry = kv_cache[slot_idx // block_size, slot_idx % block_size]
                    actual_scales = entry[512:528].view(torch.float32)
                    expected_scales = (
                        kv_c[token_idx]
                        .float()
                        .reshape(4, 128)
                        .abs()
                        .amax(dim=1)
                        .div(448.0)
                        .clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    torch.testing.assert_close(
                        actual_scales, expected_scales, rtol=1e-6, atol=0
                    )

                    quantized = entry[:512].view(torch.float8_e4m3fn).float()
                    restored = quantized * actual_scales.repeat_interleave(128)
                    torch.testing.assert_close(
                        restored,
                        kv_c[token_idx].float(),
                        rtol=0.13,
                        atol=0.03,
                    )

    def test_full_and_selected_gather(self) -> None:
        torch.manual_seed(20260830)
        block_size = 8
        num_entries = 24
        kv_c = torch.randn(num_entries, 512, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.zeros(
            3, block_size, 528, dtype=torch.uint8, device=self.device
        )
        compute_ops.concat_and_cache_mla(
            kv_c,
            torch.empty(num_entries, 0, dtype=torch.bfloat16, device=self.device),
            kv_cache,
            torch.arange(num_entries, dtype=torch.int64, device=self.device),
            "fp8_glm53_mla",
            torch.tensor(1.0, dtype=torch.float32, device=self.device),
        )

        block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device=self.device)
        full = torch.empty(20, 512, dtype=torch.bfloat16, device=self.device)
        compute_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
            kv_cache,
            full,
            block_table,
            torch.tensor([20], dtype=torch.int32, device=self.device),
            torch.tensor([0], dtype=torch.int32, device=self.device),
            1,
            20,
        )
        full_expected = torch.cat((kv_c[16:24], kv_c[0:8], kv_c[8:12]))
        torch.testing.assert_close(
            full.float(), full_expected.float(), rtol=0.13, atol=0.03
        )

        physical_indices = torch.tensor(
            [0, 7, 8, 23, -1], dtype=torch.int32, device=self.device
        )
        selected = torch.empty(
            physical_indices.numel(),
            512,
            dtype=torch.bfloat16,
            device=self.device,
        )
        compute_ops.gather_selected_glm53_fp8_mla_kv(
            kv_cache, selected, physical_indices
        )
        torch.testing.assert_close(
            selected[:-1].float(),
            kv_c[[0, 7, 8, 23]].float(),
            rtol=0.13,
            atol=0.03,
        )
        self.assertEqual(torch.count_nonzero(selected[-1]).item(), 0)

    def test_bf16_gather_widths_with_variable_batch_lengths(self) -> None:
        torch.manual_seed(20260901)
        block_size = 8
        block_table = torch.tensor(
            [[2, 0], [4, 1]], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor([10, 5], dtype=torch.int32, device=self.device)
        workspace_starts = torch.tensor([0, 10], dtype=torch.int32, device=self.device)
        for fused_width in (512, 576):
            with self.subTest(fused_width=fused_width):
                kv_cache = torch.randn(
                    5,
                    block_size,
                    fused_width,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                actual = torch.empty(
                    15, fused_width, dtype=torch.bfloat16, device=self.device
                )

                compute_ops.cp_gather_mla_kv_cache_v2(
                    kv_cache,
                    actual,
                    block_table,
                    seq_lens,
                    workspace_starts,
                    2,
                    15,
                )

                expected = torch.cat(
                    (kv_cache[2], kv_cache[0, :2], kv_cache[4, :5]), dim=0
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_gather_skips_metadata_gaps_and_tail(self) -> None:
        block_size = 4
        fused_width = 512
        kv_cache = (
            torch.arange(
                3 * block_size * fused_width,
                dtype=torch.float32,
                device=self.device,
            )
            .reshape(3, block_size, fused_width)
            .to(torch.bfloat16)
        )
        block_table = torch.tensor([[1], [2]], dtype=torch.int32, device=self.device)
        seq_lens = torch.tensor([2, 2], dtype=torch.int32, device=self.device)
        workspace_starts = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
        actual = torch.full(
            (6, fused_width), -7.0, dtype=torch.bfloat16, device=self.device
        )

        compute_ops.cp_gather_mla_kv_cache_v2(
            kv_cache,
            actual,
            block_table,
            seq_lens,
            workspace_starts,
            2,
            6,
        )

        torch.testing.assert_close(actual[0:2], kv_cache[1, 0:2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3:5], kv_cache[2, 0:2], rtol=0, atol=0)
        self.assertTrue(torch.equal(actual[2], torch.full_like(actual[2], -7.0)))
        self.assertTrue(torch.equal(actual[5], torch.full_like(actual[5], -7.0)))

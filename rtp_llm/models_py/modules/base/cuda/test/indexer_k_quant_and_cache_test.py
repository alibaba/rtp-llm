from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class IndexerKQuantAndCacheTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def test_rejects_short_slot_mapping_before_kernel_launch(self):
        k = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.empty(1, 64, 132, dtype=torch.uint8, device=self.device)
        slot_mapping = torch.empty(0, dtype=torch.int64, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "slot_mapping size"):
            rtp_llm_ops.indexer_k_quant_and_cache(
                k, kv_cache, slot_mapping, 128, "ue8m0"
            )

    def test_valid_single_token_write(self):
        k = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.zeros(1, 64, 132, dtype=torch.uint8, device=self.device)
        slot_mapping = torch.tensor([0], dtype=torch.int64, device=self.device)

        rtp_llm_ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, 128, "ue8m0")
        torch.cuda.synchronize()
        self.assertGreater(int(kv_cache.sum().item()), 0)

    def test_cache_bytes_and_gather_match_reference(self):
        torch.manual_seed(42)
        block_size = 64
        slots = [65, 0, -1, 63, 64, 127]
        k = torch.randn(len(slots), 128, dtype=torch.bfloat16, device=self.device)
        group_magnitudes = torch.tensor([0.001, 0.25, 16.0, 1024.0], device=self.device)
        k = (
            (k.view(-1, 4, 32) * group_magnitudes.view(1, 4, 1))
            .to(torch.bfloat16)
            .view(-1, 128)
        )
        k[1].zero_()
        # Exercise scales immediately below, at, and above a power-of-two boundary.
        k[3] = torch.tensor(
            [223.0, 224.0, 225.0, -448.0],
            dtype=torch.bfloat16,
            device=self.device,
        ).repeat_interleave(32)
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=self.device)
        block_table = torch.tensor([[1, 0]], dtype=torch.int32, device=self.device)
        cu_seq_lens = torch.tensor([0, 128], dtype=torch.int32, device=self.device)

        for scale_fmt in ("mxfp8", "ue8m0", "fp32"):
            with self.subTest(scale_fmt=scale_fmt):
                kv_cache = torch.full(
                    (2, block_size, 132), 0xA5, dtype=torch.uint8, device=self.device
                )
                expected = kv_cache.clone().view(2, -1)
                group_size = 32 if scale_fmt == "mxfp8" else 128
                values = k.float().view(len(slots), -1, group_size)
                scales = values.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
                if scale_fmt == "mxfp8":
                    bits = scales.view(torch.int32)
                    exponents = (
                        ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
                    ).clamp(1, 254)
                    scales = (exponents << 23).view(torch.float32)
                    scale_bytes = exponents.to(torch.uint8)
                else:
                    if scale_fmt == "ue8m0":
                        scales = torch.exp2(torch.ceil(torch.log2(scales)))
                    scale_bytes = scales.contiguous().view(torch.uint8)
                quantized = (values / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
                quantized = quantized.view(torch.uint8).view(len(slots), 128)
                for token, slot in enumerate(slots):
                    if slot < 0:
                        continue
                    block, offset = divmod(slot, block_size)
                    expected[block, offset * 128 : (offset + 1) * 128] = quantized[
                        token
                    ]
                    start = block_size * 128 + offset * 4
                    expected[block, start : start + 4] = scale_bytes[token]

                rtp_llm_ops.indexer_k_quant_and_cache(
                    k, kv_cache, slot_mapping, 128, scale_fmt
                )
                torch.testing.assert_close(
                    kv_cache.view(2, -1), expected, rtol=0, atol=0
                )

                dst_k = torch.empty(128, 128, dtype=torch.uint8, device=self.device)
                dst_scale = torch.empty(128, 4, dtype=torch.uint8, device=self.device)
                rtp_llm_ops.cp_gather_indexer_k_quant_cache(
                    kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
                )
                ordered = expected.flip(0)
                torch.testing.assert_close(
                    dst_k,
                    ordered[:, : block_size * 128].reshape(128, 128),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    dst_scale,
                    ordered[:, block_size * 128 :].reshape(128, 4),
                    rtol=0,
                    atol=0,
                )

    def test_mxfp8_empty_and_all_negative_slots_preserve_cache(self):
        for slots in ([], [-1, -2]):
            with self.subTest(slots=slots):
                k = torch.zeros(
                    len(slots), 128, dtype=torch.bfloat16, device=self.device
                )
                slot_mapping = torch.tensor(
                    slots, dtype=torch.int64, device=self.device
                )
                kv_cache = torch.full(
                    (1, 64, 132), 0xA5, dtype=torch.uint8, device=self.device
                )
                expected = kv_cache.clone()
                rtp_llm_ops.indexer_k_quant_and_cache(
                    k, kv_cache, slot_mapping, 128, "mxfp8"
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(kv_cache, expected, rtol=0, atol=0)

    def test_mxfp8_rejects_unsupported_shape_and_storage_group(self):
        for head_dim, quant_block_size in ((64, 128), (256, 128), (128, 32)):
            with self.subTest(head_dim=head_dim, quant_block_size=quant_block_size):
                k = torch.zeros(1, head_dim, dtype=torch.bfloat16, device=self.device)
                kv_cache = torch.empty(
                    1, 64, head_dim + 16, dtype=torch.uint8, device=self.device
                )
                slot_mapping = torch.tensor([0], dtype=torch.int64, device=self.device)
                with self.assertRaisesRegex(RuntimeError, "mxfp8 requires"):
                    rtp_llm_ops.indexer_k_quant_and_cache(
                        k, kv_cache, slot_mapping, quant_block_size, "mxfp8"
                    )


if __name__ == "__main__":
    main()

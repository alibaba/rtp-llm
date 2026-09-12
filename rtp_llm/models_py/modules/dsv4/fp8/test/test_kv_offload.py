"""Mapped-host MODEL1 transfers, graph replay and real FlashMLA parity."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_kv_quant_decode_op import (
    read_model1_kv_slot_bytes,
    reference_quantize_v4_kv_decode,
)
from rtp_llm.models_py.modules.dsv4.fp8.kv_offload import CsaKvStaging


def _pool(blocks, entries, *, device="cpu", pinned=False):
    stride = ((entries * 584 + 575) // 576) * 576
    storage = torch.zeros(
        (blocks, stride), dtype=torch.uint8, device=device, pin_memory=pinned
    )
    return storage.as_strided((blocks, entries, 584), (stride, 584, 1))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CsaKvOffloadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)

    def _assert_gather(self, source, selected, staging):
        pool, remapped = staging.gather(selected)
        torch.cuda.synchronize()
        source_cpu = source.cpu()
        pool_cpu = pool.cpu()
        choices = selected.cpu()
        mapped = remapped.cpu()
        for row in range(choices.shape[0]):
            for col in range(choices.shape[1]):
                slot = int(choices[row, col])
                valid = 0 <= slot < source.shape[0] * source.shape[1]
                self.assertEqual(
                    int(mapped[row, 0, col]), row * staging.topk + col if valid else -1
                )
                actual = read_model1_kv_slot_bytes(pool_cpu, row, col)
                if valid:
                    expected = read_model1_kv_slot_bytes(
                        source_cpu, slot // source.shape[1], slot % source.shape[1]
                    )
                else:
                    expected = torch.zeros(584, dtype=torch.uint8)
                self.assertTrue(torch.equal(actual, expected), (row, col, slot))

    def test_split_payload_scales_padding_and_request_rows(self):
        for entries in (2, 32, 64):
            for device in ("cpu", "cuda"):
                with self.subTest(entries=entries, device=device):
                    source = _pool(3, entries, device=device, pinned=device == "cpu")
                    source.random_(0, 256)
                    choices = torch.tensor(
                        [
                            [0, entries - 1, entries, 3 * entries - 1, -1, entries],
                            [2 * entries, 1, -1, 0, 3 * entries, -8],
                        ],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    staging = CsaKvStaging(source, max_batch_size=2, topk=6, max_ctas=3)
                    self.assertEqual(staging.pool.stride(0) % 576, 0)
                    self._assert_gather(source, choices, staging)

    def test_batch_32_and_512_selected_entries(self):
        source = _pool(4, 64, pinned=True)
        source.random_(0, 256)
        choices = torch.randint(0, 256, (32, 512), dtype=torch.int64, device="cuda")
        staging = CsaKvStaging(source, max_batch_size=32, topk=512)
        pool, indices = staging.gather(choices)
        torch.cuda.synchronize()
        for row, col in ((0, 0), (7, 511), (16, 123), (31, 511)):
            slot = int(choices[row, col])
            expected = read_model1_kv_slot_bytes(source, slot // 64, slot % 64)
            actual = read_model1_kv_slot_bytes(pool, row, col).cpu()
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(int(indices[row, 0, col]), row * 512 + col)

    def test_graph_replay_reads_new_indices_and_reused_source_pages(self):
        source = _pool(3, 32, pinned=True)
        source.random_(0, 256)
        staging = CsaKvStaging(source, max_batch_size=2, topk=4)
        choices = torch.tensor(
            [[1, 33, -1, 70], [95, 0, 3, 1]], dtype=torch.int32, device="cuda"
        )
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            staging.gather(choices)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            staging.gather(choices)
        addresses = (staging.pool.data_ptr(), staging.indices.data_ptr())
        for step in range(3):
            source.random_(0, 256)
            choices.copy_(
                torch.tensor(
                    [[step, 64 + step, 4, -1], [35, step, -1, 94 - step]],
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            for row, col in ((0, 0), (0, 1), (1, 0), (1, 3)):
                slot = int(choices[row, col])
                expected = read_model1_kv_slot_bytes(source, slot // 32, slot % 32)
                actual = read_model1_kv_slot_bytes(staging.pool, row, col).cpu()
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(
                addresses, (staging.pool.data_ptr(), staging.indices.data_ptr())
            )

    def test_rejects_pageable_source_and_invalid_descriptors(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            CsaKvStaging(_pool(1, 32), max_batch_size=1, topk=4)
        source = _pool(1, 32, pinned=True)
        staging = CsaKvStaging(source, max_batch_size=1, topk=4)
        for selected in (
            torch.zeros((1, 4), dtype=torch.int32),
            torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
            torch.zeros((1, 4), dtype=torch.float32, device="cuda"),
        ):
            with self.assertRaisesRegex(ValueError, "selected"):
                staging.gather(selected)

    def test_dual_pool_flashmla_matches_resident_compressed_kv(self):
        from rtp_llm.models_py.modules.dsv4.flash_mla_compat import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        source = _pool(4, 32, pinned=True)
        swa_cpu = _pool(1, 128)
        for pool in (source, swa_cpu):
            count = pool.shape[0] * pool.shape[1]
            values = torch.randn(count, 512, dtype=torch.bfloat16) * 0.5
            reference_quantize_v4_kv_decode(
                values, torch.arange(count), pool, pool.shape[1]
            )
        # Preserve physical padding, not just the logical tensor elements.
        resident = _pool(4, 32, device="cuda")
        resident.copy_(source)
        swa = _pool(1, 128, device="cuda")
        swa.copy_(swa_cpu)
        choices = torch.arange(127, -1, -1, dtype=torch.int32, device="cuda").repeat(
            2, 1
        )
        choices[1, -8:] = -1
        staging = CsaKvStaging(source, max_batch_size=2, topk=128)
        pool, indices = staging.gather(choices)
        q = torch.randn(2, 1, 32, 512, dtype=torch.bfloat16, device="cuda") * 0.1
        swa_indices = (
            torch.arange(128, dtype=torch.int32, device="cuda")
            .view(1, 1, 128)
            .repeat(2, 1, 1)
        )
        sink = torch.zeros(32, dtype=torch.float32, device="cuda")

        def attend(compressed, selected):
            meta, _ = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=128,
                topk=128,
                num_heads_q=32,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            output, _ = flash_mla_with_kvcache(
                q=q,
                k_cache=swa.unsqueeze(-2),
                block_table=None,
                head_dim_v=512,
                cache_seqlens=None,
                tile_scheduler_metadata=meta,
                num_splits=None,
                is_fp8_kvcache=True,
                indices=swa_indices,
                softmax_scale=512**-0.5,
                attn_sink=sink,
                extra_k_cache=compressed.unsqueeze(-2),
                extra_indices_in_kvcache=selected,
            )
            return output

        expected = attend(resident, choices.unsqueeze(1))
        actual = attend(pool, indices)
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertGreater(actual.float().abs().max().item(), 0)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

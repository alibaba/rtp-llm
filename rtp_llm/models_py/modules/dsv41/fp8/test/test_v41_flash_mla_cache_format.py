"""RTP producers consumed by upstream FlashMLA, against independent FP32 attention."""

import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    quantize_and_insert_k_cache_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    quantize_and_insert_k_cache_cp_byte_sliced,
    quantize_and_insert_swa_k_cache,
)


def _pool(entries, width, padding):
    stride = entries * width + padding
    storage = torch.zeros((3, stride), dtype=torch.uint8, device="cuda")
    return storage.as_strided((3, entries, width), (stride, width, 1))


def _read_rows(pool, slots):
    """Decode the documented upstream token format, independent of RTP readers."""
    rows = pool[slots // pool.shape[1], slots % pool.shape[1]].contiguous()
    if pool.shape[-1] == 528:
        payload = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = torch.exp2(rows[:, 512:].float() - 127)
        return (payload.reshape(-1, 16, 32) * scales[..., None]).flatten(1)
    packed = rows[:, :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=pool.device,
    )
    scales = rows[:, 256:].contiguous().view(torch.float8_e4m3fn).float()
    return (lut[codes].reshape(-1, 32, 16) * scales[..., None]).flatten(1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FlashMLACacheFormatTest(unittest.TestCase):
    def test_native_swa_and_dual_pool_with_page_padding_and_graph(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("V4.1 FlashMLA requires SM100")
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        torch.manual_seed(410528288)
        for padding_tokens in (0, 32):
            for dual in (False, True):
                with self.subTest(padding_tokens=padding_tokens, dual=dual):
                    swa = _pool(128, 528, padding_tokens * 528)
                    glob = _pool(64, 288, padding_tokens * 288)
                    swa_slots = torch.arange(128, 256, device="cuda")
                    global_slots = torch.arange(64, 128, device="cuda")
                    quantize_and_insert_swa_k_cache(
                        torch.randn(128, 512, device="cuda", dtype=torch.bfloat16),
                        swa,
                        swa_slots,
                    )
                    quantize_and_insert_k_cache_fp4(
                        torch.randn(64, 512, device="cuda", dtype=torch.bfloat16),
                        glob,
                        global_slots,
                    )
                    q = torch.randn(2, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
                    sink = torch.linspace(-1, 1, 64, device="cuda")
                    indices = swa_slots.int().reshape(1, 1, -1).repeat(2, 1, 1)
                    extra = global_slots.int().reshape(1, 1, -1).repeat(2, 1, 1)
                    metadata, _ = get_mla_metadata()

                    def run():
                        return flash_mla_with_kvcache(
                            q,
                            swa.unsqueeze(2),
                            None,
                            None,
                            512,
                            metadata,
                            softmax_scale=512**-0.5,
                            indices=indices,
                            attn_sink=sink,
                            extra_k_cache=glob.unsqueeze(2) if dual else None,
                            extra_indices_in_kvcache=extra if dual else None,
                        )

                    actual, _ = run()
                    kv = _read_rows(swa, swa_slots)
                    if dual:
                        kv = torch.cat((kv, _read_rows(glob, global_slots)))
                    scores = q.float() @ kv.T * 512**-0.5
                    denominator = torch.logaddexp(torch.logsumexp(scores, -1), sink)
                    expected = torch.exp(scores - denominator[..., None]) @ kv
                    torch.testing.assert_close(
                        actual.float(), expected, rtol=0.015, atol=0.004
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured, _ = run()
                    graph.replay()
                    torch.testing.assert_close(captured, actual, rtol=0, atol=0)

    def test_cp5_byte_sliced_writes_rebuild_native_cache(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("V4.1 FlashMLA requires SM100")
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        torch.manual_seed(41528)
        cp_size = 5
        alignment = math.lcm(512, 528, cp_size)
        for ring_entries in (128, 136):
            # Main rounds logical entries for CP, independently of the physical
            # stride alignment. Padding capacity must never become a slot divisor.
            entries = math.ceil(ring_entries / cp_size) * cp_size
            stride = math.ceil(entries * 528 / alignment) * alignment
            with self.subTest(ring_entries=ring_entries, logical_entries=entries):
                self.assertIn(entries, (130, 140))
                self.assertEqual(stride // 528, 160)
                local_bytes = stride // cp_size
                slots = torch.arange(entries, 3 * entries, device="cuda")
                values = torch.randn(
                    slots.numel(), 512, device="cuda", dtype=torch.bfloat16
                )
                metadata = SimpleNamespace(
                    unique_blocks=torch.tensor([1, 2], device="cuda"),
                    compact_slots=slots - entries,
                    contiguous_block_start=1,
                )
                slices = [
                    torch.full((3, local_bytes), 0x5A, dtype=torch.uint8, device="cuda")
                    for _ in range(cp_size)
                ]
                for rank, raw in enumerate(slices):
                    quantize_and_insert_k_cache_cp_byte_sliced(
                        values, raw, slots, entries, rank, cp_size, metadata
                    )
                # Rebuild bytes exactly as the five-rank collective does. The
                # test runs on one GPU; no distributed transport is exercised.
                rebuilt = torch.stack(slices).permute(1, 0, 2).reshape(3, stride)
                self.assertTrue(rebuilt[0].eq(0x5A).all())
                self.assertTrue(rebuilt[:, entries * 528 :].eq(0x5A).all())
                pool = rebuilt.as_strided((3, entries, 528), (stride, 528, 1))
                q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
                sink = torch.linspace(-1, 1, 64, device="cuda")
                indices = torch.full(
                    (1, 1, math.ceil(slots.numel() / 64) * 64),
                    -1,
                    dtype=torch.int32,
                    device="cuda",
                )
                indices[..., : slots.numel()] = slots.int()
                schedule, _ = get_mla_metadata()

                def run():
                    return flash_mla_with_kvcache(
                        q,
                        pool.unsqueeze(2),
                        None,
                        None,
                        512,
                        schedule,
                        softmax_scale=512**-0.5,
                        indices=indices,
                        attn_sink=sink,
                    )

                actual, _ = run()
                kv = _read_rows(pool, slots)
                scores = q.float() @ kv.T * 512**-0.5
                denominator = torch.logaddexp(torch.logsumexp(scores, -1), sink)
                expected = torch.exp(scores - denominator[..., None]) @ kv
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.015, atol=0.004
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, _ = run()
                graph.replay()
                torch.testing.assert_close(captured, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

"""Check exact cache bytes and ownership of asynchronous MLA metadata."""

import gc
import unittest
from itertools import accumulate
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.ops import KvCacheDataType
from page_rr_jit_warmup_test import source_table, PAGE, KERNEL


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaAsyncMetadataTest(unittest.TestCase):
    def test_unit_scale_can_be_created_in_a_cuda_graph(self):
        MlaKVCacheWriteOp(KvCacheDataType.BASE)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation = MlaKVCacheWriteOp(KvCacheDataType.BASE)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(operation.scale.shape, torch.Size([]))
        self.assertEqual(operation.scale.dtype, torch.float32)
        self.assertEqual(operation.scale.item(), 1.0)

    def test_cache_writes_match_the_original_unit_scale_bytes(self):
        torch.manual_seed(3101)
        rows = 145
        latent = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        suffix = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
        slots_host = torch.arange(rows, dtype=torch.int64).mul(5).remainder(256)
        slots_host[::7] = -1
        params = SimpleNamespace(slot_mapping=slots_host.cuda())
        cases = [
            (KvCacheDataType.BASE, False, False, torch.bfloat16, 576),
            (KvCacheDataType.FP8, False, False, torch.uint8, 656),
            (KvCacheDataType.FP8, False, True, torch.uint8, 656),
            (KvCacheDataType.FP8, True, False, torch.uint8, 576),
        ]
        for cache_type, fp8_compute, ue8m0, dtype, width in cases:
            with self.subTest(cache_type=cache_type, fp8_compute=fp8_compute, ue8m0=ue8m0):
                reference = torch.zeros((4, 64, width), device="cuda", dtype=dtype)
                actual = reference.clone()
                old = MlaKVCacheWriteOp(cache_type, fp8_compute)
                old.scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
                new = MlaKVCacheWriteOp(cache_type, fp8_compute)
                old.use_ue8m0 = new.use_ue8m0 = ue8m0
                old.forward(latent, suffix, SimpleNamespace(kv_cache_base=reference), params)
                new.forward(latent, suffix, SimpleNamespace(kv_cache_base=actual), params)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)))
                if cache_type == KvCacheDataType.BASE:
                    expected = torch.zeros((256, 576), dtype=torch.bfloat16)
                    values = torch.cat((latent.cpu(), suffix.cpu()), dim=1)
                    for token, slot in enumerate(slots_host.tolist()):
                        if slot >= 0:
                            expected[slot] = values[token]
                    self.assertTrue(torch.equal(actual.cpu().view(256, 576), expected))

    def test_pinned_indptr_survives_eager_plan_release(self):
        lengths, prefixes = [0, 2, 7], [PAGE - 2, PAGE + 1, PAGE + 4]
        total = sum(lengths)
        table, maps = source_table(len(lengths), 4, 2, 1)
        padding = [
            r * max(lengths) + j - row
            for row, (r, j) in enumerate(
                (r, j) for r, n in enumerate(lengths) for j in range(n)
            )
        ]
        inputs = SimpleNamespace(
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            input_lengths_device=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            prefix_lengths_device=torch.tensor(prefixes, device="cuda", dtype=torch.int32),
            cu_seqlens_device=torch.tensor(
                list(accumulate([0] + lengths)), device="cuda", dtype=torch.int32,
            ),
            kv_cache_kernel_block_id_device=table,
            logical_token_count=total - 1, physical_token_count=total,
            padding_offset=torch.tensor(padding, device="cuda", dtype=torch.int32),
            # The eager plan must own its copy rather than reading this stale value.
            cu_kv_seqlens_device=torch.full((4,), -313, device="cuda", dtype=torch.int32),
        )
        warm = PageRRMlaMetadata(PAGE, KERNEL, 2, 1)
        warm.prepare_prefill(inputs)
        warm_pinned = [torch.empty((4,), dtype=torch.int32, pin_memory=True) for _ in range(128)]
        del warm_pinned
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            metadata = PageRRMlaMetadata(PAGE, KERNEL, 2, 1)
            metadata.prepare_prefill(inputs)
            self.assertTrue(metadata._prefill_kv_indptr_host.is_pinned())
            indptr = metadata.prefill_ragged_kv_len_indptr_d
            kv_lens = metadata.kv_lens_d
            slots = metadata.slot_mapping
            del metadata
        churn = [torch.empty((4,), dtype=torch.int32, pin_memory=True).fill_(-999) for _ in range(128)]
        gc.collect()
        stream.synchronize()
        expected_lengths = [q + prefix for q, prefix in zip(lengths, prefixes)]
        self.assertEqual(indptr.cpu().tolist(), list(accumulate([0] + expected_lengths)))
        self.assertEqual(kv_lens.cpu().tolist(), expected_lengths)
        expected_slots = [
            maps[r].get(prefixes[r] + j, -1)
            for r, n in enumerate(lengths) for j in range(n)
        ]
        expected_slots[-1] = -1
        self.assertEqual(slots.cpu().tolist(), expected_slots)
        self.assertEqual(len(churn), 128)


if __name__ == "__main__":
    unittest.main()

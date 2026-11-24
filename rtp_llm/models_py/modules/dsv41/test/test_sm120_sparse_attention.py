"""SM120 fallback versus dense attention over the real V4.1 cache codecs."""

import unittest

import torch
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    dequantize_k_cache_slots_fp4,
    quantize_and_insert_k_cache_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    dequantize_swa_k_cache,
    quantize_and_insert_swa_k_cache,
)
from rtp_llm.models_py.modules.dsv41.fp8.sm120_sparse_attention import (
    sparse_decode,
    sparse_prefill,
)


def dense_reference(query, keys, mask, sink, scale):
    keys = torch.where(mask[..., None], keys, 0.0).float()
    logits = torch.einsum("rhd,rkd->rhk", query.float(), keys) * scale
    logits.masked_fill_(~mask[:, None], -torch.inf)
    all_logits = torch.cat(
        (logits, sink[None, :, None].expand(query.shape[0], -1, -1)), -1
    )
    probabilities = torch.softmax(all_logits, -1)[..., :-1]
    result = torch.einsum("rhk,rkd->rhd", probabilities, keys)
    lse = torch.logsumexp(logits, -1)
    lse = torch.where(mask.any(-1)[:, None], lse, torch.inf)
    return result.to(query.dtype), lse


class Sm120SparseAttentionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "A real SM120 GPU is required")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(318)

    def test_joint_swa_fp4_decode_and_graph_replay(self):
        device = torch.device("cuda")
        swa = torch.full((2, 128, 528), 255, device=device, dtype=torch.uint8)
        global_pool = torch.full((2, 128, 288), 255, device=device, dtype=torch.uint8)
        slots = torch.arange(1, 240, device=device, dtype=torch.int64)
        source = torch.randn(239, 512, device=device, dtype=torch.bfloat16)
        quantize_and_insert_swa_k_cache(source, swa, slots)
        quantize_and_insert_k_cache_fp4(source * 0.7, global_pool, slots)
        query = torch.randn(2, 2, 8, 512, device=device, dtype=torch.bfloat16)
        sink = torch.randn(8, device=device)
        ids = (torch.arange(160, device=device).expand(4, -1) + 1).clone()
        extra = ids[:, :140].clone()
        ids[:, 3::7] = -1
        extra[:, 5::9] = -1
        ids[3] = -1
        extra[3] = -1
        lengths = torch.tensor([153, 147, 141, 160], device=device)
        extra_lengths = torch.tensor([131, 127, 133, 140], device=device)
        scale = 512**-0.5

        def run():
            return sparse_decode(
                query,
                sink,
                swa,
                ids.view(2, 2, -1),
                scale,
                lengths,
                global_pool,
                extra.view(2, 2, -1),
                extra_lengths,
            )

        def reference():
            first = dequantize_swa_k_cache(swa, ids.reshape(-1)).view(4, 160, 512)
            second = dequantize_k_cache_slots_fp4(global_pool, extra.reshape(-1)).view(
                4, 140, 512
            )
            mask = (ids >= 0) & (
                torch.arange(160, device=device)[None] < lengths[:, None]
            )
            extra_mask = (extra >= 0) & (
                torch.arange(140, device=device)[None] < extra_lengths[:, None]
            )
            return dense_reference(
                query.view(4, 8, 512),
                torch.cat((first, second), 1),
                torch.cat((mask, extra_mask), 1),
                sink,
                scale,
            )

        expected, expected_lse = reference()
        actual, lse = run()
        torch.testing.assert_close(
            actual.view_as(expected), expected, atol=0.016, rtol=0.016
        )
        torch.testing.assert_close(
            lse.transpose(1, 2).reshape(4, 8), expected_lse, atol=2e-5, rtol=2e-5
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual.reshape(4, 8, 512)[3].count_nonzero().item(), 0)
        # Capture after JIT and allocator warmup, then change graph inputs.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_lse = run()
        query.mul_(0.5)
        graph.replay()
        expected, expected_lse = reference()
        torch.testing.assert_close(
            captured.view_as(expected), expected, atol=0.016, rtol=0.016
        )
        torch.testing.assert_close(
            captured_lse.transpose(1, 2).reshape(4, 8),
            expected_lse,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_prefill_masked_nan_slot_and_empty_row(self):
        device = torch.device("cuda")
        query = torch.randn(3, 8, 512, device=device, dtype=torch.bfloat16)
        keys = torch.randn(180, 1, 512, device=device, dtype=torch.bfloat16)
        keys[0] = torch.nan
        ids = (torch.arange(160, device=device).expand(3, -1) + 1).clone()
        ids[:, 2::5] = -1
        ids[2] = -1
        sink = torch.randn(8, device=device)
        lengths = torch.tensor([155, 149, 160], device=device)
        mask = (ids >= 0) & (torch.arange(160, device=device)[None] < lengths[:, None])
        expected, expected_lse = dense_reference(
            query, keys.reshape(180, 512)[ids.clamp_min(0)], mask, sink, 512**-0.5
        )
        actual, _, lse = sparse_prefill(query, keys, ids, 512**-0.5, 512, sink, lengths)
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()

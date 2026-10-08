"""Upstream decode interface over native V4.1 packed pools on SM120."""

import unittest
from unittest.mock import Mock

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8._v41_decode_attention_sm120 import (
    sparse_decode_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_decode_attn import (
    fp4_dual_decode_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    quantize_and_insert_k_cache_fp4,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
    quantize_and_insert_swa_k_cache,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def pool(tag, gen):
    layout = dsv41_page_layouts(gen)[tag]
    backing = torch.full(
        (4, layout.stride_bytes + 128), 0xAD, device="cuda", dtype=torch.uint8
    )
    raw = backing[:, : layout.stride_bytes]
    width = layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
    packed = raw.as_strided((4, layout.entries, width), (backing.stride(0), width, 1))
    values = torch.randn(4 * layout.entries, 512, device="cuda", dtype=torch.bfloat16)
    write_slots = torch.arange(values.shape[0], device="cuda", dtype=torch.int64)
    if tag == "swa_kv":
        quantize_and_insert_swa_k_cache(values, packed, write_slots)
    else:
        quantize_and_insert_k_cache_fp4(values, packed, write_slots)
    return V41PagedCache(raw, tag, gen), packed, backing


def oracle(query, sinks, sources):
    output = torch.empty_like(query, device="cpu")
    views = [
        (
            V41PagedCache(cache.raw.cpu(), cache.tag, cache.gen_num_per_cycle),
            slots.cpu(),
        )
        for cache, slots in sources
    ]
    for row in range(query.shape[0]):
        values = []
        for cache, slots in views:
            selected = slots[row, 0]
            values.append(
                cache.read(
                    selected[selected >= 0].contiguous(), backend="eager"
                ).double()
            )
        kv = torch.cat(values)
        q = query[row, 0].cpu().double()
        logits = q @ kv.T * 512**-0.5
        probabilities = torch.cat((logits, sinks.cpu().double()[:, None]), -1).softmax(
            -1
        )
        output[row, 0] = (probabilities[:, :-1] @ kv).bfloat16()
    return output


class DecodeAttentionSM120(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41991)

    @torch.inference_mode()
    def test_joint_softmax_native_pages_and_padding(self):
        for gen in (0, 1):
            for tag in (None, "csa_kv", "hca_kv"):
                for heads in (1, 64):
                    with self.subTest(gen=gen, tag=tag, heads=heads):
                        swa, swa_pool, swa_raw = pool("swa_kv", gen)
                        sw = torch.randint(
                            128,
                            4 * swa.layout.entries,
                            (3, 1, 128),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        sw[0].fill_(-1)
                        sw[1, :, ::3] = -1
                        sources = [(swa, sw)]
                        originals = [(swa_raw, swa_raw.clone())]
                        gp = gs = None
                        if tag is not None:
                            global_cache, gp, raw = pool(tag, gen)
                            gs = torch.randint(
                                global_cache.layout.entries,
                                4 * global_cache.layout.entries,
                                (3, 1, 512),
                                device="cuda",
                                dtype=torch.int32,
                            )
                            gs[0].fill_(-1)
                            gs[1, :, ::5] = -1
                            sources.append((global_cache, gs))
                            originals.append((raw, raw.clone()))
                        q = torch.randn(
                            3, 1, heads, 512, device="cuda", dtype=torch.bfloat16
                        )
                        sinks = torch.linspace(-1, 2, heads, device="cuda")
                        forbidden = Mock()
                        forbidden.forward.side_effect = AssertionError(
                            "SM100 provider called on SM120"
                        )
                        if tag is None:
                            actual = sparse_decode_attention(q, swa_pool, sw, sinks)
                        else:
                            actual = fp4_dual_decode_attention(
                                q=q,
                                swa_pool_3d=swa_pool,
                                global_pool_3d=gp,
                                attn_sink=sinks,
                                swa_topk_3d=sw,
                                global_topk_3d=gs,
                                swa_block_table=None,
                                sched_meta=None,
                                fp8_op=forbidden,
                            )
                        torch.testing.assert_close(
                            actual.cpu(),
                            oracle(q, sinks, sources),
                            rtol=0.01,
                            atol=0.004,
                        )
                        self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
                        for raw, before in originals:
                            torch.testing.assert_close(raw, before, rtol=0, atol=0)
                        forbidden.forward.assert_not_called()

    @torch.inference_mode()
    def test_changed_input_graph_replay(self):
        for gen in (0, 1):
            with self.subTest(gen=gen):
                swa, sp, _ = pool("swa_kv", gen)
                global_cache, gp, _ = pool("csa_kv", gen)
                sw = torch.full((2, 1, 128), -1, device="cuda", dtype=torch.int32)
                gs = torch.full((2, 1, 512), -1, device="cuda", dtype=torch.int32)
                q = torch.randn(2, 1, 4, 512, device="cuda", dtype=torch.bfloat16)
                sinks = torch.linspace(-2, 1, 4, device="cuda")

                def run():
                    return sparse_decode_attention(q, sp, sw, sinks, gp, gs)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()
                for iteration in range(3):
                    q.normal_()
                    sinks.add_(0.25)
                    sw.random_(swa.layout.entries, 4 * swa.layout.entries)
                    gs.random_(128, 512)
                    sw[iteration % 2, :, ::3] = -1
                    gs[iteration % 2, :, ::5] = -1
                    for cache, packed, encode in (
                        (swa, sp, quantize_and_insert_swa_k_cache),
                        (global_cache, gp, quantize_and_insert_k_cache_fp4),
                    ):
                        slots = torch.arange(
                            cache.layout.entries,
                            4 * cache.layout.entries,
                            device="cuda",
                        )
                        values = torch.randn(
                            len(slots), 512, device="cuda", dtype=torch.bfloat16
                        )
                        encode(values, packed, slots)
                    graph.replay()
                    torch.testing.assert_close(
                        output.cpu(),
                        oracle(q, sinks, [(swa, sw), (global_cache, gs)]),
                        rtol=0.01,
                        atol=0.004,
                    )

    def test_reject_unsupported_span_and_pool_geometry(self):
        _, packed, raw = pool("swa_kv", 0)
        before = raw.clone()
        sinks = torch.zeros(4, device="cuda")
        for span in (0, 2):
            q = torch.empty(1, span, 4, 512, device="cuda", dtype=torch.bfloat16)
            slots = torch.empty(1, span, 128, device="cuda", dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "single-token"):
                sparse_decode_attention(q, packed, slots, sinks)
        q = torch.empty(1, 1, 4, 512, device="cuda", dtype=torch.bfloat16)
        slots = torch.full((1, 1, 128), -1, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "pool geometry"):
            sparse_decode_attention(q, packed[:, :-1], slots, sinks)
        torch.testing.assert_close(raw, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

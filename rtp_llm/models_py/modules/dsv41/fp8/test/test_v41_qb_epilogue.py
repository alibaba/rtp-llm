"""Native QB epilogue, shared projection reload, Graph replay and decode wiring."""
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8._v41_decode_indexer import (
    prepare_indexer_q_and_weights,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_merged_qb import MergedQB
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_merged_qb import fixture, linear
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QBEpilogueTest(unittest.TestCase):
    @torch.inference_mode()
    def test_reindex_projection_chain_reload_graph(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_front_gemm as front
        from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )
        from rtp_llm.models_py.modules.dsv41.test.test_v41_front_gemm import (
            FrontGemmTest,
        )

        with patch.dict(os.environ, {
            "DSV41_FUSED_FRONT_PROJECTION": "1",
            "DSV41_FUSED_QB_POSTPROCESS": "1",
            "DSV41_FUSED_QUERY_QUANT": "1",
            "DSV4_FP8_QUANT_KERNEL": "legacy",
        }):
            attn, state, owner = fixture()
            projection = FrontGemmTest().make_attention()
            for name in ("wq_a_wkv", "index_weights", "q_norm", "q_lora_rank", "eps"):
                setattr(attn, name, getattr(projection, name))
            attn.is_kv_source = False
            attn.rope_head_dim, attn.head_dim = 64, 512
            attn.skip_post_q_norm = True
            attn.kv_norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
            state.merged_qb = MergedQB.create(attn, state, owner)
            front.prepare(attn)
            x = torch.randn(64, 6, 5120, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(384, device="cuda", dtype=torch.int32) % 128

            def call():
                return decode_compute_qkv(attn, x, positions, mega_attention=state)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            self.assertIsNotNone(captured.head_weights)
            self.assertIsNotNone(captured.index_prepared)
            self.assertIsNone(captured.index_q)
            pointers = [t.data_ptr() for t in attn._front_weights.tensors]
            for _ in range(3):
                torch.cuda.synchronize()
                attn.index_weights = -attn.index_weights
                attn.wq_a_wkv.weight = (-attn.wq_a_wkv.weight.float()).to(torch.float8_e4m3fn)
                attn.wq_b.weight.copy_((-attn.wq_b.weight.float()).to(torch.float8_e4m3fn))
                owner[W.v4_indexer_wq_b_w].copy_(
                    (-owner[W.v4_indexer_wq_b_w].float()).to(torch.float8_e4m3fn)
                )
                state.reload_weights(attn)
                front.prepare(attn)
                x.normal_()
                positions.add_(1)
                graph.replay()
                expected = call()
                for name in ("q", "qr", "kv", "head_weights"):
                    torch.testing.assert_close(getattr(captured, name), getattr(expected, name), atol=0, rtol=0)
                for actual, reference in zip(captured.index_prepared, expected.index_prepared):
                    self.assertTrue(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)))
                self.assertEqual(pointers, [t.data_ptr() for t in attn._front_weights.tensors])

    @torch.inference_mode()
    def test_prepared_consumers_and_fallback(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_attention_fusion import (
            _fixture,
        )

        for mode in ("dense", "sparse", "nonpaged"):
            attn, keys, x, qr, positions = _fixture("cuda", 1)
            q = attn._lin(attn.index_wq, qr.reshape(-1, 16)).view(2, 6, 32, 128)
            weights = torch.nn.functional.linear(x, attn.index_weights)
            prepared = prepare_indexer_q_and_weights(q, weights, attn.freqs_cis, positions,
                                                     weights_out_dtype=torch.bfloat16)
            if mode == "sparse":
                attn.layer_id = 24
                attn.decode_sparse_mqa = True
                attn.v41_config.update(candidate_source_layer_id=20, candidate_block_size=8,
                                       candidate_topk_blocks=512)
                candidates = torch.full((12,512), -1, dtype=torch.int32, device="cuda")
                candidates[:,:96] = torch.arange(96, device="cuda", dtype=torch.int32)
                attn._shared_attention["candidates"] = candidates
            elif mode == "nonpaged":
                attn._shared_attention["global"][20] = torch.randn(2,768,128,device="cuda",dtype=torch.float32)
            expected = AttentionV41FP8._select_indices_decode(attn,x,qr,positions)
            attn._shared_attention.pop("decode_sparse_plan", None)
            actual = AttentionV41FP8._select_indices_decode(
                attn,x,qr,positions,head_weights=weights.reshape(-1,32),index_prepared=prepared)
            torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values,atol=0,rtol=0)
            print(f"PASS prepared consumer {mode}",flush=True)

    @torch.inference_mode()
    def test_reload_and_graph(self):
        torch.manual_seed(1921)
        with patch.dict(os.environ):
            os.environ.pop("DSV41_FUSED_QB_POSTPROCESS", None)
            for m in (256, 384, 512):
                for pos_dtype in (torch.int32, torch.int64):
                    attn, state, owner = fixture()
                    merged = state.merged_qb = MergedQB.create(attn, state, owner)
                    import deep_gemm

                    self.assertIs(merged._postprocess_op, deep_gemm.dsv41_qb_indexer)
                    data = torch.randn(m // 2, 2, 1280, device="cuda", dtype=torch.bfloat16)
                    quantized = attn.wq_b._quantize_input(data)
                    head = torch.randn(m // 2, 2, 32, device="cuda", dtype=torch.bfloat16)
                    angles = torch.randn(1024, 32, device="cuda")
                    freqs = torch.polar(torch.ones_like(angles), angles)
                    positions = torch.arange(m, device="cuda", dtype=pos_dtype)
                    pointers = [t.data_ptr() for t in (merged.weight, merged.weight_scales,
                                                       state.query_weight, attn.index_wq.weight)]
                    def call():
                        return merged.try_forward_prepared(*quantized, head, freqs, positions)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            call()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = call()
                    self.assertIsNotNone(captured)
                    for phase in range(3):
                        new_main, main_sf = linear(32768)
                        new_index, index_sf = linear(4096)
                        main_sf.copy_(torch.full_like(main_sf.float(), 2 ** -phase).to(main_sf.dtype))
                        index_sf.copy_(torch.full_like(index_sf.float(), 2 ** -phase).to(index_sf.dtype))
                        attn.wq_b.weight.copy_(new_main.weight)
                        attn.wq_b.weight_scales.copy_(new_main.pack_scales(main_sf))
                        owner[W.v4_indexer_wq_b_w].copy_(new_index.weight)
                        owner[W.v4_indexer_wq_b_s].copy_(index_sf)
                        state.reload_weights(attn)
                        head.normal_()
                        positions.copy_(torch.arange(m, device="cuda", dtype=pos_dtype) - phase)
                        freqs.copy_(torch.polar(torch.ones_like(angles), angles + phase))
                        fresh = attn.wq_b._quantize_input(torch.randn_like(data))
                        for dst, src in zip(quantized, fresh):
                            dst.copy_(src)
                        graph.replay()
                        expected_q, index = merged.forward_quantized(*quantized)
                        expected = prepare_indexer_q_and_weights(
                            index.view(m // 2, 2, 32, 128), head, freqs, positions,
                            weights_out_dtype=torch.bfloat16)
                        for actual in (captured, call()):
                            torch.testing.assert_close(actual[0], expected_q, atol=0, rtol=0)
                            for value, ref in zip(actual[1], expected):
                                self.assertTrue(torch.equal(value.view(torch.uint8), ref.view(torch.uint8)))
                        self.assertEqual(pointers, [t.data_ptr() for t in (merged.weight,
                            merged.weight_scales, state.query_weight, attn.index_wq.weight)])
                        self.assertIs(owner[W.v4_indexer_wq_b_w], attn.index_wq.weight)
                    # Opt-out/unsupported dtype and small M preserve the old path.
                    with patch.dict(os.environ, DSV41_FUSED_QB_POSTPROCESS="0"):
                        self.assertIsNone(call())
                    with patch.dict(os.environ, DSV41_FUSED_QB_POSTPROCESS="1"):
                        explicit = call()
                        torch.testing.assert_close(explicit[0], captured[0], atol=0, rtol=0)
                        for value, ref in zip(explicit[1], captured[1]):
                            self.assertTrue(torch.equal(value.view(torch.uint8), ref.view(torch.uint8)))
                    self.assertIsNone(merged.try_forward_prepared(*quantized, head.float(), freqs, positions))
                    small = attn.wq_b._quantize_input(data[:64])
                    self.assertIsNone(merged.try_forward_prepared(*small, head[:64], freqs, positions[:128]))
                    print(f"PASS M={m} {pos_dtype}: native exact, 3 reloads, Graph, shared pointers", flush=True)


if __name__ == "__main__":
    unittest.main()

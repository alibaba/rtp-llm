"""GLM uses UE8M0 query scales while retaining ordinary K cache scales."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc import glm5
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.hybrid import indexer
from rtp_llm.utils.model_weight import W


class Glm5IndexerQueryScaleTest(TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(Path(__file__).with_name("test_indexer_query_ue8_sgl_golden.json").read_text())
        def tensor(name, dtype):
            return torch.tensor(data[name], dtype=dtype, device="cuda")
        cls.q = tensor("q_u16", torch.uint16).view(torch.bfloat16)
        cls.k = tensor("k_u16", torch.uint16).view(torch.bfloat16)
        cls.qh = tensor("query_hadamard_u16", torch.uint16).view(torch.bfloat16)
        cls.kh = tensor("expected_k_u16", torch.uint16).view(torch.bfloat16)
        cls.expected_q = tensor("expected_q_u8", torch.uint8)
        cls.expected_s = tensor("expected_scale_u32", torch.uint32).view(torch.float32)
        cls.expected_cache = tensor("expected_key_cache_u8", torch.uint8)
        cls.positions = torch.zeros(len(cls.q), dtype=torch.int64, device="cuda")
        cls.cache = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), dim=1).cuda()

    def operator(self, **kwargs):
        return IndexerOp(1, 128, 16, 64, cos_sin_cache=self.cache,
                         scale_fmt="none", query_scale_fmt="ue8m0",
                         is_neox_style=False, **kwargs)

    def test_all_query_paths_match_sgl_and_key_cache_bytes_are_unchanged(self):
        for mode in ("legacy", "v2"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}):
                operator = self.operator()
                q, s, k = operator.fused_rope_quant_qk(self.q, self.k, self.positions)
                self.assertTrue(torch.equal(q.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s, self.expected_s))
                self.assertTrue(torch.equal(k, self.kh))
                q, s = operator.fused_rope_quant_q(self.q, self.positions)
                self.assertTrue(torch.equal(q.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s, self.expected_s))
                q, s = operator.quant_q_only(self.qh)
                self.assertTrue(torch.equal(q.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s, self.expected_s))
                cache = torch.full_like(self.expected_cache, 137)
                kv = SimpleNamespace(kv_cache_base=cache, seq_size_per_block=64)
                slots = torch.arange(len(self.q), device="cuda", dtype=torch.int64)
                q, s = operator.quant_q_k(self.qh, self.kh, kv, slots)
                self.assertTrue(torch.equal(q.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s, self.expected_s))
                self.assertTrue(torch.equal(cache, self.expected_cache))

    def test_nondefault_stream_graph_reads_dynamic_query_and_key_values(self):
        operator = self.operator()
        q, k = self.q[:4].clone(), self.k[:4].clone()
        positions = self.positions[:4]
        for _ in range(3): operator.fused_rope_quant_qk(q, k, positions)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream): outputs = operator.fused_rope_quant_qk(q, k, positions)
        for multiplier in (2, 4):
            q.copy_(self.q[:4] * multiplier)
            k.copy_(self.k[:4] * multiplier)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(outputs[0].view(torch.uint8), self.expected_q[:4]))
            self.assertTrue(torch.equal(outputs[1], self.expected_s[:4] * multiplier))
            self.assertTrue(torch.equal(outputs[2], self.kh[:4] * multiplier))

    def test_existing_callers_keep_their_coupled_scale_formats(self):
        for fmt in ("none", "ue8m0"):
            operator = IndexerOp(1, 128, 16, 64, cos_sin_cache=self.cache, scale_fmt=fmt)
            self.assertEqual(operator.scale_fmt, fmt)
            self.assertEqual(operator.query_scale_fmt, fmt)
        operator = self.operator()
        self.assertEqual(operator.scale_fmt, "none")
        self.assertEqual(operator.query_scale_fmt, "ue8m0")
        q, s, k = operator.fused_rope_quant_qk(self.q[:0], self.k[:0], self.positions[:0])
        self.assertEqual(q.shape, (0, 1, 128))
        self.assertEqual(s.shape, (0, 1, 1))
        self.assertEqual(k.shape, (0, 128))

    def test_glm_selects_query_format_only_on_ppu_and_indexer_forwards_it(self):
        observed = []

        class Attention(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                observed.append(kwargs)

        config = SimpleNamespace(hidden_size=6144, quant_config=None, attn_config=SimpleNamespace(use_mla=True),
                                 moe_layer_index=[3], layernorm_eps=1e-5)
        parallel = SimpleNamespace(get_attn_tp_size=lambda: 1)
        weights = {W.pre_ln_gamma: torch.ones(6144, device="cuda", dtype=torch.bfloat16),
                   W.post_ln_gamma: torch.ones(6144, device="cuda", dtype=torch.bfloat16)}
        for ppu in (True, False):
            with patch.object(glm5, "is_ppu", return_value=ppu), \
                 patch.object(glm5, "MlaAttention", Attention), \
                 patch.object(glm5, "GenericMoeLayer", nn.Identity), \
                 patch.object(glm5, "dsa_layer_has_indexer", return_value=True), \
                 patch.object(glm5, "dsa_layer_skips_topk", return_value=False), \
                 patch.dict(os.environ, {"RTP_LLM_GLM5_CMP": "0"}):
                glm5.Glm5MoeDecoderLayer(config, parallel, weights, {}, 3,
                    SimpleNamespace(moe_strategy="auto"),
                    hw_kernel_config=SimpleNamespace(enable_fuse_kernels=False))
            self.assertEqual(observed[-1]["indexer_query_scale_fmt"], "ue8m0" if ppu else None)

        attn = SimpleNamespace(indexer_head_num=1, indexer_head_dim=128, indexer_topk=16,
                               rope_head_dim=64, kernel_tokens_per_block=64, tokens_per_block=64,
                               rope_config=SimpleNamespace(indexer_is_neox_style=False))
        weights = {W.mla_indexer_k_norm_w: torch.ones(128, device="cuda", dtype=torch.bfloat16),
                   W.mla_indexer_k_norm_b: torch.zeros(128, device="cuda", dtype=torch.bfloat16)}
        with patch.object(indexer.LinearFactory, "create_linear_from_weights", return_value=nn.Identity()):
            module = indexer.Indexer(attn, weights, {W.rope_cos_sin_cache: self.cache}, 0, 1e-5, None,
                                     hw_kernel_config=SimpleNamespace(enable_fuse_kernels=False),
                                     query_scale_fmt="ue8m0")
        self.assertEqual(module.indexer_op.query_scale_fmt, "ue8m0")
        self.assertEqual(module.indexer_op.scale_fmt, "none")


if __name__ == "__main__":
    main()

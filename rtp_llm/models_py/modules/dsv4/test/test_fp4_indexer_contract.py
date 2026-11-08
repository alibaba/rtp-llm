import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_save_partial_states,
)


class Fp4ConfigTest(unittest.TestCase):
    def test_real_cache_descriptor_uses_fp8_entry_geometry(self):
        from rtp_llm.models.dsv4_kv_cache import build_dsv4_kv_cache_spec_descs

        layers = build_dsv4_kv_cache_spec_descs(
            layer_num=1,
            layer_compress_ratios=[4],
            fp8_kv=True,
            head_dim=512,
            indexer_head_dim=128,
        )
        self.assertEqual(len(layers), 1)
        by_tag = {desc.tag: desc for desc in layers[0]}
        self.assertEqual(by_tag["indexer_kv"].entry_elems, 132)
        self.assertEqual(by_tag["indexer_kv"].compression_ratio, 4)
        self.assertEqual(by_tag["indexer_state"].entry_elems, 512)
        self.assertEqual(by_tag["csa_kv"].entry_elems, 584)


class Fp4IndexerContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(421)
        self.device = torch.device("cuda")

    def test_raw_state_and_fp8_default(self):
        kv = torch.randn((8, 256), device=self.device)
        score = torch.randn_like(kv)
        ape = torch.randn((4, 256), device=self.device)
        pos = torch.arange(8, dtype=torch.int64, device=self.device)
        slots = pos + 8
        state = torch.zeros((2, 8, 512), device=self.device)
        run_save_partial_states(kv, score, ape, pos, state, slots, 4)
        torch.testing.assert_close(state[1, :, :256], kv, rtol=0, atol=0)
        torch.testing.assert_close(
            state[1, :, 256:], score + ape[pos % 4], rtol=0, atol=0
        )
        self.assertEqual(state[0].count_nonzero().item(), 0)

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.kimi_k3.fp8_weight import KimiK3LoadFp8Weight
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.utils.model_weight import W, identity


class KimiK3Fp8WeightTest(unittest.TestCase):
    def test_attention_projection_set_includes_kda_and_mla(self):
        names = KimiK3LoadFp8Weight.w8a8_weight_list
        for name in (K3W.KDA_INPUT, W.linear_attn_f_b_w, W.linear_attn_out_w,
                     W.mla_fusedqkrope_w, W.mla_q_b_w, W.mla_kv_b_w,
                     W.attn_o_w):
            self.assertIn(name, names)

    def test_bounded_quantizer_matches_reference(self):
        torch.manual_seed(19)
        raw = torch.randn(2112, 256, dtype=torch.bfloat16)
        for ue8m0 in (False, True):
            actual = KimiK3LoadFp8Weight._quantize_matrix(raw, use_ue8m0=ue8m0)
            expected = per_block_cast_to_fp8(raw, use_ue8m0=ue8m0)
            self.assertTrue(torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8)))
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_kda_fused_input_preserves_replicated_tail_at_tp8(self):
        cfg = SimpleNamespace(linear_num_key_heads=96,
                              linear_num_value_heads=96,
                              linear_key_head_dim=128,
                              linear_value_head_dim=128)
        source = CustomAtomicWeight(K3W.KDA_INPUT, [], process_fun=identity)
        source.config = cfg
        wrapper = KimiK3LoadFp8Weight(source, Fp8BlockWiseQuantConfig())
        width = 4 * 12288 + 128 + 96
        weight = torch.arange(width, dtype=torch.float32)[:, None].expand(width, 128)
        scales = torch.arange((width + 127) // 128, dtype=torch.float32)[:, None]
        result = wrapper._split(
            {wrapper.kernel.name: weight, wrapper.scale.name: scales},
            SimpleNamespace(tp_size=8, tp_rank=7),
        )
        shard = result[wrapper.kernel.name]
        logical_rows = 4 * 1536 + 128 + 12
        self.assertEqual(shard.shape, (6400, 128))
        torch.testing.assert_close(shard[4 * 1536:4 * 1536 + 128],
                                   weight[4 * 12288:4 * 12288 + 128], rtol=0, atol=0)
        beta_begin = 4 * 12288 + 128 + 7 * 12
        torch.testing.assert_close(shard[logical_rows - 12:logical_rows],
                                   weight[beta_begin:beta_begin + 12], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(shard[logical_rows:]).item(), 0)
        torch.testing.assert_close(result[wrapper.scale.name][-2:], scales[-2:], rtol=0, atol=0)
        for part in range(4):
            begin = part * 12288 + 7 * 1536
            torch.testing.assert_close(shard[part * 1536:(part + 1) * 1536],
                                       weight[begin:begin + 1536], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

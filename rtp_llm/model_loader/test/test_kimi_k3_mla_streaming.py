"""CPU regression using the real manifest, collector and TP transforms."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3Weight
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import W


class KimiK3MlaStreamingTest(unittest.TestCase):
    def test_shared_source_produces_all_tp_views_without_database_reads(self):
        for tp in (1, 2, 4, 8):
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    self.check_views(tp, rank)

    def check_views(self, tp, rank):
        heads, key_dim, value_dim, latent = 16, 8, 4, 6
        manifest = KimiK3Weight.__new__(KimiK3Weight)
        manifest._head_num = heads
        manifest.nope_head_dim = key_dim
        manifest.v_head_dim = value_dim
        manifest.rope_head_dim = 0
        manifest.kv_lora_rank = latent
        manifest.tp_size, manifest.tp_rank = tp, rank
        manifest.model_config = SimpleNamespace(
            attn_config=SimpleNamespace(use_mla=True, q_lora_rank=8),
            mla_ops_type=MlaOpsType.AUTO,
        )
        config = SimpleNamespace(
            compute_dtype=torch.bfloat16, merge_lora=False,
            tp_size=tp, tp_rank=rank, ep_size=1, ep_rank=0,
            dp_size=1, dp_rank=0, ffn_tp_size=tp, ffn_tp_rank=rank,
            hidden_size=16, head_num=heads, head_num_kv=heads,
            size_per_head=key_dim, moe_pure_tp_mode=False, bit=16,
            use_swizzleA=False,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda name, tensor, **kwargs: tensor
            ),
        )
        key = "language_model.model.layers.3.self_attn.kv_b_proj.weight"
        consumers = [
            component
            for weight in manifest._mla_weights()
            for component in weight.get_components()
            if key in component.get_tensor_names(3, config)
        ]
        self.assertEqual(len(consumers), 1)
        collector = TensorCollector({key}, None)
        # No database is available: all three outputs must use the streamed tensor.
        source = torch.arange(heads * (key_dim + value_dim) * latent).reshape(
            heads * (key_dim + value_dim), latent
        ).to(torch.bfloat16)
        self.assertTrue(collector.store_tensor(key, source))
        result = consumers[0].load(collector, 3, "cpu", config)
        self.assertEqual(set(result), {W.mla_kv_b_w, W.mla_kc, W.mla_vc})
        local = source.reshape(heads, key_dim + value_dim, latent).chunk(tp, 0)[rank]
        expected = {
            W.mla_kv_b_w: local.reshape(-1, latent).T.contiguous(),
            W.mla_kc: local[:, :key_dim, :].contiguous(),
            W.mla_vc: local[:, key_dim:, :].transpose(1, 2).contiguous(),
        }
        for name, reference in expected.items():
            self.assertEqual(result[name].dtype, torch.bfloat16)
            torch.testing.assert_close(result[name], reference, rtol=0, atol=0)
        collector.clear()
        self.assertTrue(collector.is_collection_complete())


if __name__ == "__main__":
    unittest.main()

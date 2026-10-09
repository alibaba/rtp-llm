import unittest
from types import SimpleNamespace

import torch
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.utils.model_weight import W, sp_head, sp_id


def load_config(tp, rank):
    return SimpleNamespace(
        hidden_size=7,
        head_num=4,
        head_num_kv=4,
        size_per_head=2,
        tp_size=tp,
        tp_rank=rank,
        ffn_tp_size=tp,
        ffn_tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        lm_head_tp_size=tp,
        lm_head_tp_rank=rank,
        moe_pure_tp_mode=False,
        bit=16,
    )


class PerLayerLoadConfigTest(unittest.TestCase):
    def test_non_square_marker_tp_and_lora_use_local_geometry(self):
        marker = torch.arange(7 * 72, dtype=torch.float32).reshape(7, 72)
        geometry = AttnConfig(hidden_size=7, head_num=8, head_num_kv=2, size_per_head=6)
        weight = AttnAtomicWeight(
            W.attn_qkv_w,
            [],
            config=geometry,
            lora_a_split_func=sp_id,
            lora_b_split_func=sp_head,
        )
        for tp in (2, 4):
            for rank in range(tp):
                config = load_config(tp, rank)
                before = vars(config).copy()
                actual = weight._split(marker, config)[weight.name]
                # Independent slices in the [Q | K | V] checkpoint layout.
                q = marker[:, rank * (48 // tp) : (rank + 1) * (48 // tp)]
                kv_rank = rank if tp == 2 else rank // 2
                k = marker[:, 48 + kv_rank * 6 : 54 + kv_rank * 6]
                v = marker[:, 60 + kv_rank * 6 : 66 + kv_rank * 6]
                self.assertTrue(torch.equal(actual, torch.cat([q, k, v], dim=1)))
                lora = weight._split_lora(
                    {weight.lora_a_name: marker, weight.lora_b_name: marker}, config
                )
                self.assertTrue(torch.equal(lora[weight.lora_b_name], actual))
                self.assertEqual(vars(config), before)

    def test_default_and_two_layer_instances(self):
        config = load_config(2, 0)
        default = AttnAtomicWeight(W.attn_qkv_w, [])
        self.assertIs(default._layer_load_config(config), config)
        a = AttnAtomicWeight(
            W.attn_qkv_w,
            [],
            config=AttnConfig(
                hidden_size=7, head_num=8, head_num_kv=2, size_per_head=6
            ),
        )
        b = AttnAtomicWeight(
            W.attn_qkv_w,
            [],
            config=AttnConfig(
                hidden_size=7, head_num=4, head_num_kv=4, size_per_head=2
            ),
        )
        self.assertEqual(a._layer_load_config(config).size_per_head, 6)
        self.assertEqual(b._layer_load_config(config).size_per_head, 2)
        self.assertEqual(config.size_per_head, 2)


if __name__ == "__main__":
    unittest.main()

"""Load KDA control tensors through real collectors and TP transforms."""
import unittest
from types import SimpleNamespace
import torch
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3Weight, KimiK3MtpWeight
from rtp_llm.utils.model_weight import W


class KimiK3KdaPrecisionTest(unittest.TestCase):
    def test_vllm_precision_for_target_and_native_draft(self):
        heads, dim, width = 8, 4, 4
        conv = [torch.linspace(-0.31 + i, 0.41 + i, heads*dim*width).reshape(heads*dim,1,width)
                for i in range(3)]
        norm = torch.tensor([0.01561231, 0.107321, 0.982734, 1.000031])
        alog = torch.cat([torch.linspace(-0.42, 0.13, heads), torch.zeros(128-heads)])
        bias = torch.linspace(-2.98731, 0.65437, heads*dim)
        tensors = dict(zip(('self_attn.q_conv1d.weight', 'self_attn.k_conv1d.weight',
                            'self_attn.v_conv1d.weight'), conv))
        tensors.update({'self_attn.o_norm.weight': norm, 'self_attn.A_log': alog,
                        'self_attn.dt_bias': bias})
        wanted = {W.linear_attn_conv1d_w, W.linear_attn_norm_w,
                  W.linear_attn_alog, W.linear_attn_dt_b_kda}
        for cls in (KimiK3Weight, KimiK3MtpWeight):
            for tp in (1,2,8):
                for rank in range(tp):
                    with self.subTest(model=cls.__name__, tp=tp, rank=rank):
                        manifest = cls.__new__(cls)
                        manifest.model_config = SimpleNamespace(linear_attention_config=SimpleNamespace(
                            linear_num_key_heads=heads, linear_num_value_heads=heads,
                            linear_key_head_dim=dim, linear_value_head_dim=dim))
                        cfg = SimpleNamespace(compute_dtype=torch.bfloat16, merge_lora=False,
                            tp_size=tp, tp_rank=rank, ep_size=tp, ep_rank=rank,
                            dp_size=1, dp_rank=0, ffn_tp_size=1, ffn_tp_rank=0,
                            hidden_size=16, head_num=heads, head_num_kv=heads, size_per_head=dim,
                            moe_pure_tp_mode=False, bit=16, use_swizzleA=False,
                            exported_device=SimpleNamespace(maybe_rewrite_weight_by_key=lambda name,tensor,**kw:tensor))
                        loaded = {}
                        for weight in manifest._kda_weights():
                            if weight.name not in wanted:
                                continue
                            names = weight.get_tensor_names(0,cfg)
                            collector = TensorCollector(set(names),None)
                            for name in names:
                                collector.store_tensor(name,tensors[name.removeprefix('language_model.model.layers.0.')])
                            loaded.update(weight.load(collector,0,'cpu',cfg))
                        expected = {
                            W.linear_attn_conv1d_w: torch.cat([x.chunk(tp,0)[rank] for x in conv]),
                            W.linear_attn_norm_w: norm.bfloat16(),
                            W.linear_attn_alog: alog[:heads].chunk(tp)[rank],
                            W.linear_attn_dt_b_kda: bias.chunk(tp)[rank],
                        }
                        self.assertEqual(set(loaded),wanted)
                        for name,value in expected.items():
                            self.assertEqual(loaded[name].dtype,value.dtype,name)
                            torch.testing.assert_close(loaded[name],value,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()

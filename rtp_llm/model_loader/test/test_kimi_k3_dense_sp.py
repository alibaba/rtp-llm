"""SP uses replicated dense weights and computes each token on one rank."""
import unittest
from types import SimpleNamespace
import torch
from rtp_llm.model_loader.tensor_source import TensorCollector
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3Weight
from rtp_llm.utils.model_weight import W

class KimiK3DenseSPTest(unittest.TestCase):
    def test_sp_loader_replicates_dense_mlp_for_local_tokens(self):
        gen = torch.Generator().manual_seed(39)
        x = torch.randn(16, 8, generator=gen)
        tensors = {
            "mlp.gate_proj.weight": torch.randn(32, 8, generator=gen),
            "mlp.up_proj.weight": torch.randn(32, 8, generator=gen),
            "mlp.down_proj.weight": torch.randn(8, 32, generator=gen),
        }
        def activation(g, u):
            return 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)
        expected = activation(x @ tensors["mlp.gate_proj.weight"].T,
                              x @ tensors["mlp.up_proj.weight"].T) @ tensors["mlp.down_proj.weight"].T
        for tp in (1, 2, 8):
            with self.subTest(tp=tp):
                outputs = []
                for rank in range(tp):
                    manifest = KimiK3Weight.__new__(KimiK3Weight)
                    config = SimpleNamespace(
                        compute_dtype=torch.float32, merge_lora=False,
                        tp_size=tp, tp_rank=rank, ep_size=tp, ep_rank=rank,
                        dp_size=1, dp_rank=0, ffn_tp_size=1, ffn_tp_rank=0,
                        hidden_size=8, head_num=8, head_num_kv=8, size_per_head=1,
                        moe_pure_tp_mode=False, bit=16, use_swizzleA=False,
                        exported_device=SimpleNamespace(maybe_rewrite_weight_by_key=lambda name,tensor,**kw:tensor),
                    )
                    loaded = {}
                    for weight in manifest._dense_weights():
                        names = weight.get_tensor_names(0, config)
                        collector = TensorCollector(set(names), None)
                        for name in names:
                            suffix = name.removeprefix("language_model.model.layers.0.")
                            collector.store_tensor(name, tensors[suffix])
                        loaded.update(weight.load(collector, 0, "cpu", config))
                    self.assertEqual(tuple(loaded[W.ffn_w1].shape), (8, 32))
                    self.assertEqual(tuple(loaded[W.ffn_w2].shape), (32, 8))
                    # Different ranks own different token slices. Concatenation
                    # restores logical order; no TP sum or duplicated tokens.
                    local_x = x.chunk(tp)[rank]
                    outputs.append(activation(local_x @ loaded[W.ffn_w1],
                                              local_x @ loaded[W.ffn_w3]) @ loaded[W.ffn_w2])
                actual = torch.cat(outputs)
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

if __name__ == "__main__":
    unittest.main()

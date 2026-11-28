"""Real checkpoint loading for interleaved gates, KV replication and raw norms."""
import os
import tempfile
import unittest
from unittest import mock
import torch
from safetensors.torch import save_file
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.attention_weights import Qwen4AttentionWeights
from rtp_llm.models_py.registry import register_model


@register_model("_qwen4_attention_fixture")
class AttentionFixture(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.self_attn = Qwen4AttentionWeights(8, 4, config["kv_heads"], 4,
            tp_size=load.tp_size, tp_rank=load.tp_rank)


class AttentionWeightsTest(unittest.TestCase):
    def test_tp2_partition_and_replication(self):
        torch.manual_seed(82)
        for kv_heads in (1, 2, 4):
            shapes = {"q_proj.weight": (32,8), "k_proj.weight": (kv_heads*4,8),
                "v_proj.weight": (kv_heads*4,8), "o_proj.weight": (8,16),
                "q_norm.weight": (4,), "k_norm.weight": (4,)}
            weights = {"self_attn."+k: torch.randn(v).bfloat16() for k,v in shapes.items()}
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LOAD_METHOD": ""}):
                save_file(weights, os.path.join(directory,"model.safetensors"))
                for rank in (0,1):
                    with self.subTest(kv_heads=kv_heads, rank=rank):
                        model = NewModelLoader({"model_type":"_qwen4_attention_fixture", "kv_heads":kv_heads},
                            NewLoaderConfig(device="cpu", compute_dtype=torch.bfloat16,
                                tp_size=2,tp_rank=rank,load_method="scratch"),model_path=directory).load()
                        attn=model.self_attn
                        q,g=weights["self_attn.q_proj.weight"].reshape(4,8,8).split(4,dim=1)
                        q=q[rank*2:(rank+1)*2].reshape(8,8)
                        g=g[rank*2:(rank+1)*2].reshape(8,8)
                        start=(0 if kv_heads==1 else rank*(kv_heads//2))*4
                        count=max(1,kv_heads//2)*4
                        expected=torch.cat([q,weights["self_attn.k_proj.weight"][start:start+count],
                            weights["self_attn.v_proj.weight"][start:start+count]])
                        torch.testing.assert_close(attn.qkv_proj.weight,expected,rtol=0,atol=0)
                        torch.testing.assert_close(attn.gate.weight,g,rtol=0,atol=0)
                        torch.testing.assert_close(attn.o_proj.weight,weights["self_attn.o_proj.weight"][:,rank*8:(rank+1)*8],rtol=0,atol=0)
                        fresh = -weights["self_attn.o_proj.weight"]
                        update, = attn.o_proj.prepare_weight_updates({"weight": fresh})
                        self.assertIs(update.target, attn.o_proj.weight)
                        torch.testing.assert_close(update.source, fresh[:,rank*8:(rank+1)*8],rtol=0,atol=0)
                        torch.testing.assert_close(attn.q_norm.weight,weights["self_attn.q_norm.weight"],rtol=0,atol=0)

    def test_missing_component_fails_loading(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LOAD_METHOD":""}):
            save_file({"self_attn.q_proj.weight":torch.zeros(32,8,dtype=torch.bfloat16)},os.path.join(directory,"model.safetensors"))
            with self.assertRaises(RuntimeError):
                NewModelLoader({"model_type":"_qwen4_attention_fixture","kv_heads":1},
                    NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16,load_method="scratch"),model_path=directory).load()


if __name__ == "__main__":
    unittest.main()

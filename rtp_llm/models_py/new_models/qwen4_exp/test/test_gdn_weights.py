"""GDN source-to-direct-parameter staging; no inference backend replacement."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.gdn_weights import Qwen4GDNWeights
from rtp_llm.models_py.registry import register_model


@register_model("_qwen4_gdn_weights_fixture")
class _GDNFixture(RtpModule):
    def __init__(self, cfg, load):
        super().__init__()
        la = SimpleNamespace(linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=3)
        self.linear_attn = Qwen4GDNWeights(la, 8, 1e-6, load.tp_size, load.tp_rank,
            torch.bfloat16, None, "linear_attn")


class GDNWeightsTest(unittest.TestCase):
    def test_fused_per_head_checkpoint_layout(self):
        shapes = {"in_proj_qkvz.weight":(48,8), "in_proj_ba.weight":(8,8),
            "conv1d.weight":(32,1,3), "dt_bias":(4,), "A_log":(4,),
            "norm.weight":(4,), "out_proj.weight":(8,16)}
        weights = {"linear_attn."+name:torch.arange(torch.Size(shape).numel()).reshape(shape).bfloat16()
            for name,shape in shapes.items()}
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{"LOAD_METHOD":""}):
            save_file(weights,os.path.join(directory,"model.safetensors"))
            for tp in (1,2):
                for rank in range(tp):
                    methods = ["scratch"] + (["fastsafetensors"] if tp == 1 and os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1" else [])
                    for method in methods:
                        with self.subTest(tp=tp,rank=rank,method=method):
                            model = NewModelLoader({"model_type":"_qwen4_gdn_weights_fixture"},
                                NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16,
                                    tp_size=tp,tp_rank=rank,load_method=method),model_path=directory).load()
                            source = weights["linear_attn.in_proj_qkvz.weight"].reshape(2,24,8)
                            local = source[rank*(2//tp):(rank+1)*(2//tp)]
                            expected = torch.cat([local[:,start:end].reshape(-1,8)
                                for start,end in ((0,4),(4,8),(8,16),(16,24))])
                            torch.testing.assert_close(model.linear_attn.in_proj_qkvz.weight,expected,rtol=0,atol=0)
                            source_ba = weights["linear_attn.in_proj_ba.weight"].reshape(2,4,8)
                            local_ba = source_ba[rank*(2//tp):(rank+1)*(2//tp)]
                            expected_ba = torch.cat((local_ba[:,:2].reshape(-1,8),local_ba[:,2:].reshape(-1,8))).t()
                            torch.testing.assert_close(model.linear_attn.in_proj_ba_w,expected_ba,rtol=0,atol=0)

    def test_split_checkpoint_tp_parts_and_raw_norm(self):
        torch.manual_seed(66)
        shapes = {"in_proj_qkv.weight": (32, 8), "in_proj_z.weight": (16, 8),
            "in_proj_b.weight": (4, 8), "in_proj_a.weight": (4, 8),
            "conv1d.weight": (32, 1, 3), "dt_bias": (4,), "A_log": (4,),
            "norm.weight": (4,), "out_proj.weight": (8, 16)}
        weights = {f"linear_attn.{k}": torch.randn(v).bfloat16() for k, v in shapes.items()}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LOAD_METHOD": ""}):
            save_file(weights, os.path.join(directory, "model.safetensors"))
            for rank in (0, 1):
                model = NewModelLoader({"model_type": "_qwen4_gdn_weights_fixture"},
                    NewLoaderConfig(device="cpu", compute_dtype=torch.bfloat16,
                        tp_size=2, tp_rank=rank, load_method="scratch"), model_path=directory).load()
                gdn = model.linear_attn
                expected = torch.cat([weights["linear_attn.in_proj_qkv.weight"][i+rank*n:i+(rank+1)*n]
                    for i,n in ((0,4),(8,4),(16,8))] + [weights["linear_attn.in_proj_z.weight"][rank*8:(rank+1)*8]])
                torch.testing.assert_close(gdn.in_proj_qkvz.weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(gdn.out_proj.weight, weights["linear_attn.out_proj.weight"][:,rank*8:(rank+1)*8], rtol=0, atol=0)
                torch.testing.assert_close(gdn.norm_w, weights["linear_attn.norm.weight"], rtol=0, atol=0)
                torch.testing.assert_close(gdn.in_proj_ba_w, torch.cat([weights["linear_attn.in_proj_b.weight"][rank*2:(rank+1)*2],weights["linear_attn.in_proj_a.weight"][rank*2:(rank+1)*2]]).t(), rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()

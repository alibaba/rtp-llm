"""Whole target checkpoint ownership; CPU loading is not GPU inference proof."""
import json
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import torch
from safetensors.torch import save_file
from safetensors.torch import load_file
from rtp_llm.models_py.model_loader import NewLoaderConfig,NewModelLoader
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4TargetWeights
from rtp_llm.models_py.registry import register_model

register_model("_qwen4_whole_target_checkpoint")(Qwen4TargetWeights)


def _checkpoint(root):
    torch.manual_seed(89)
    text = dict(num_hidden_layers=2,hidden_size=128,hc_count=4,vocab_size=32,
        num_experts=2,moe_intermediate_size=128,num_attention_heads=4,num_key_value_heads=1,head_dim=32,
        linear_key_head_dim=32,linear_value_head_dim=32,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_conv_kernel_dim=3,layer_types=["linear_attention","full_attention"],rms_norm_eps=1e-6,
        ple_layer_ids=[1],ngram_size=3,heads_per_ngram=2,split_ngram_parts=8,ple_embed_dim=16,
        ple_conv_kernel_size=3,eos_token_id=2,indexer_head_dim=4,indexer_n_heads=2,
        indexer_budget=8,indexer_compress_ratio=4,indexer_kv_heads=1)
    config = dict(text_config=text,quantization_config=dict(quant_method="modelopt",quant_algo="NVFP4",group_size=16,with_input_scale=True))
    values = {}
    prefix="model.language_model."
    def add(name,shape):
        values[prefix+name]=(torch.randn(shape)*0.02).bfloat16()
    add("embed_tokens.weight",(32,128))
    values["lm_head.weight"]=(torch.randn(32,128)*0.02).bfloat16()
    for unit in ["hyper_connection_mixer"]+[f"layers.{i}.{part}_hyper_connection" for i in range(2) for part in ("attn","mlp")]:
        for name,shape in (("hc_norm",(512,)),("input_mix_weight_down",(8,512)),("input_mix_weight_up",(512,8))):
            add(unit+"."+name+".weight",shape)
        if unit!="hyper_connection_mixer": add(unit+".block_inject_weight.weight",(4,512))
    gdn_shapes={"in_proj_qkv.weight":(256,128),"in_proj_z.weight":(128,128),
        "in_proj_b.weight":(4,128),"in_proj_a.weight":(4,128),"conv1d.weight":(256,1,3),
        "dt_bias":(4,),"A_log":(4,),"norm.weight":(32,),"out_proj.weight":(128,128)}
    for name,shape in gdn_shapes.items():add("layers.0.linear_attn."+name,shape)
    for name,shape in {"q_proj.weight":(256,128),"k_proj.weight":(32,128),"v_proj.weight":(32,128),
        "o_proj.weight":(128,128),"q_norm.weight":(32,),"k_norm.weight":(32,),
        "index_qk_proj.weight":(12,128),"q_layernorm.weight":(4,),"k_layernorm.weight":(4,)}.items():
        nested = "indexer." if name.split(".")[0] in ("index_qk_proj", "q_layernorm", "k_layernorm") else ""
        add("layers.1.self_attn."+nested+name,shape)
    for i in range(2):
        base=f"layers.{i}.mlp."
        add(base+"gate.weight",(2,128))
        add(base+"shared_expert_gate.weight",(1,128))
        for name in ("gate_proj","up_proj","down_proj"):add(base+"shared_expert."+name+".weight",(128,128))
        for e in range(2):
            for p in ("up","gate","down"):
                name=prefix+base+f"experts.{e}.{p}_proj."
                values[name+"weight"]=torch.full((128,64),e+1,dtype=torch.uint8)
                values[name+"weight_scale"]=torch.ones(128,8).to(torch.float8_e4m3fn)
                values[name+"weight_scale_2"]=torch.ones(1)
                values[name+"input_scale"]=torch.ones(1)
    ple="layers.0.ple."
    for name,shape in {"key_proj":(512,16),"value_proj":(128,16),"conv1d":(512,1,3),
        "norm_key":(512,),"norm_query":(512,),"norm_conv":(512,)}.items():add(ple+name+".weight",shape)
    metadata={"layer_multipliers":torch.tensor([2**44+123,2**43+7,2**42+13]),
        "ngram_heads_offsets":torch.tensor([0,3,8,15]),"ngram_heads_vocab_sizes":torch.tensor([3,5,7,11])}
    for name,value in metadata.items():values[prefix+ple+"ple_embedding."+name]=value
    for i in range(8):add(ple+f"ple_embedding.ngram_embedding.shard_{i}.weight",(4,4))
    # Draft keys are intentionally unrelated and must not enter target owners.
    values["mtp.fc_hidden.weight"]=torch.ones(128,128,dtype=torch.bfloat16)
    for name,value in list(values.items()):
        if name.startswith(prefix+"layers.0.mlp.experts."):
            values["mtp."+name[len(prefix):]]=value.clone()
    save_file(values,str(root/"model.safetensors"))
    (root/"config.json").write_text(json.dumps(config))
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{name:"model.safetensors" for name in values}}))
    return values


class TargetWeightsTest(unittest.TestCase):
    def test_prefill_role_and_ple_document_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint(root)
            path = root / "config.json"
            raw = json.loads(path.read_text())
            del raw["text_config"]["eos_token_id"]
            raw["eos_token_id"] = 7
            path.write_text(json.dumps(raw))
            load = NewLoaderConfig(device="cpu", parallelism_config=SimpleNamespace(role_type="PREFILL"))
            model = Qwen4TargetWeights({"ckpt_path": directory}, load)
            self.assertEqual(model.layers[0].ple.eos_token_id, 7)
            hcs = [model.hyper_connection_mixer]
            for layer in model.layers:
                hcs.extend((layer.attn_hyper_connection, layer.mlp_hyper_connection))
            self.assertTrue(all(hc.role == "PREFILL" for hc in hcs))
            del raw["eos_token_id"]
            path.write_text(json.dumps(raw))
            config = SimpleNamespace(ckpt_path=directory, special_tokens=SimpleNamespace(eos_token_id=5))
            model = Qwen4TargetWeights(config, load)
            self.assertEqual(model.layers[0].ple.eos_token_id, 5)
            config.special_tokens.eos_token_id = None
            with self.assertRaisesRegex(ValueError, "resolved eos_token_id"):
                Qwen4TargetWeights(config, load)

    def test_owners_follow_explicit_attention_and_head_shards(self):
        # Ownership contract only; production admission remains main's resolved
        # config. Physical EP ownership must not force attention/head sharding.
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{"LOAD_METHOD":""}):
            values=_checkpoint(Path(directory))
            load=NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16,load_method="scratch",
                tp_size=2,tp_rank=1,ep_size=2,ep_rank=1,attn_tp_size=1,attn_tp_rank=0,
                ffn_tp_size=1,ffn_tp_rank=0,lm_head_tp_size=1,lm_head_tp_rank=0)
            model=NewModelLoader({"model_type":"_qwen4_whole_target_checkpoint","ckpt_path":directory,"num_experts":2,"moe_k":1,"qwen4_ple_prefetch_mb":64},load,model_path=directory).load()
            try:
                torch.testing.assert_close(model.embed_tokens.weight,values["model.language_model.embed_tokens.weight"],rtol=0,atol=0)
                torch.testing.assert_close(model.lm_head.weight,values["lm_head.weight"],rtol=0,atol=0)
                self.assertEqual(model.layers[0].linear_attn.in_proj_qkvz.weight.shape,(384,128))
                self.assertEqual(model.layers[1].self_attn.qkv_proj.weight.shape,(192,128))
                self.assertEqual(len(list(model.layers[0].ple.ple_embedding.ngram_embedding.parameters())),8)
                self.assertEqual(model.layers[0].ple.prefetch_budget_mb,64)
                self.assertEqual(model.layers[0].mlp.experts.w13.unique().item(),2)
            finally:
                model.close()

    def test_complete_target_scratch_tp2_ep2(self):
        self._check_complete_target("scratch")

    def test_complete_target_native_fst_tp2_ep2(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _checkpoint(root)
            torch.multiprocessing.spawn(_fst_rank,args=(directory,str(root/"gloo")),nprocs=2,join=True)

    def _check_complete_target(self, method):
        with tempfile.TemporaryDirectory() as directory,mock.patch.dict(os.environ,{"LOAD_METHOD":""}):
            values=_checkpoint(Path(directory))
            for rank in (0,1):
                with self.subTest(rank=rank):
                    self._check_rank(directory,values,rank,method)

    def _check_rank(self,directory,values,rank,method):
        loader=NewModelLoader({"model_type":"_qwen4_whole_target_checkpoint","ckpt_path":directory,"num_experts":2,"moe_k":1},
            NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16,load_method=method,tp_size=2,tp_rank=rank,ep_size=2,ep_rank=rank),model_path=directory)
        model=loader.load()
        try:
            torch.testing.assert_close(model.embed_tokens.weight,values["model.language_model.embed_tokens.weight"][:,rank*64:(rank+1)*64],rtol=0,atol=0)
            torch.testing.assert_close(model.lm_head.weight,values["lm_head.weight"][rank*16:(rank+1)*16],rtol=0,atol=0)
            self.assertEqual(model.layers[0].mlp.experts.w13.shape,(1,256,64))
            self.assertEqual(model.layers[0].mlp.experts.w13.unique().item(),rank+1)
            self.assertEqual(model.layers[1].self_attn.qkv_proj.weight.shape,(128,128))
            self.assertEqual(model.layers[1].self_attn.qsa_indexer.index_qk_proj.weight.shape,(12,128))
            ple=model.layers[0].ple.ple_embedding
            self.assertEqual(ple.layer_multipliers.dtype,torch.int64)
            self.assertGreater(ple.layer_multipliers.max().item(),2**43)
            self.assertTrue(all(p.device.type=="cpu" for p in ple.ngram_embedding.parameters()))
            self.assertEqual(len(list(ple.ngram_embedding.parameters())),4)
            self.assertEqual(loader._selected_load_method.value,method)
        finally:
            model.close()

    def test_manifest_missing_expert_scale_fails_before_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_checkpoint(root)
            index=json.loads((root/"model.safetensors.index.json").read_text())
            del index["weight_map"]["model.language_model.layers.1.mlp.experts.1.down_proj.weight_scale"]
            (root/"model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                Qwen4TargetWeights({"ckpt_path":directory},NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16))

    def test_fp8_dense_source_is_not_cast_as_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            values=_checkpoint(root)
            values["model.language_model.layers.1.self_attn.q_proj.weight"]=torch.zeros(256,128).to(torch.float8_e4m3fn)
            save_file(values,str(root/"model.safetensors"))
            with self.assertRaisesRegex(ValueError,"non-expert weights"):
                Qwen4TargetWeights({"ckpt_path":directory},NewLoaderConfig(device="cpu",compute_dtype=torch.bfloat16))

def _fst_rank(rank,directory,rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo",init_method="file://"+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        with mock.patch.dict(os.environ,{"LOAD_METHOD":""}):
            TargetWeightsTest()._check_rank(directory,load_file(str(Path(directory)/"model.safetensors")),rank,"fastsafetensors")
    finally:
        torch.distributed.destroy_process_group()


if __name__=="__main__":unittest.main()

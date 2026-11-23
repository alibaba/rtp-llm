"""Actual Qwen4 release source metadata and precision-role coverage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
from datetime import timedelta

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4CheckpointPlan
from rtp_llm.models_py.new_models.qwen4_exp.test.test_native_service import native_config
from rtp_llm.models_py.new_models.qwen4_exp.language import Qwen4ForCausalLM
from rtp_llm.models_py.registry import MODEL_REGISTRY
from rtp_llm.ops import ParallelismConfig, HWKernelConfig, MoeConfig
from rtp_llm.models_py.model_loader import NewModelLoader, NewLoaderConfig
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4TargetWeights
from rtp_llm.config.qwen4_exp_config import resolve_qwen4_parallelism


def source_checkpoint(directory, kind):
    """Generate the same typed tiny checkpoint in each original source recipe."""
    native_config(directory)
    root = Path(directory)
    config_json = json.loads((root / "config.json").read_text())
    # Own fixture bytes before replacing the source file: safetensors may keep
    # mmap-backed views, whose offsets change when auxiliary tensors are removed.
    values = {name:value.clone() for name,value in load_file(str(root / "model.safetensors")).items()}
    if kind != "nvfp4":
        for name in list(values):
            if ".mlp.experts." not in name:
                continue
            old = values.pop(name)
            if not name.endswith(".weight"):
                continue
            tensor = torch.randn(old.shape[0], old.shape[1] * 2).bfloat16()
            if kind == "fp8":
                from rtp_llm.models_py.quant_methods.fp8 import per_block_quant_like_legacy
                tensor, scale = per_block_quant_like_legacy(tensor,128)
                values[name+"_scale_inv"] = scale
            values[name] = tensor
        if kind == "bf16":
            config_json.pop("quantization_config")
        elif kind == "fp8":
            config_json["quantization_config"] = dict(quant_method="fp8", activation_scheme="dynamic",weight_block_size=[128,128])
            for name in list(values):
                if ".ple_embedding.ngram_embedding.shard_" in name:
                    values[name] = values[name].to(torch.float8_e4m3fn)
                    values[name.rsplit(".shard_",1)[0]+".weight_scale"] = torch.ones(1)
        else:
            raise ValueError(kind)
    save_file(values,str(root/"model.safetensors"))
    (root/"config.json").write_text(json.dumps(config_json))
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{name:"model.safetensors" for name in values}}))
    config = Qwen4Exp._create_config_from_json(directory,config_json)
    config.data_type = "bf16"
    config.generate_env_config = None
    config.enable_fp32_lm_head = False
    return config, values


class PrecisionPlanTest(unittest.TestCase):
    def test_bf16_eplb_reloads_mapped_expert_prefix(self):
        from rtp_llm.utils.model_weight import W

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            MODEL_REGISTRY, {"qwen4_exp": Qwen4TargetWeights}
        ):
            config, _ = source_checkpoint(directory, "bf16")
            pc = ParallelismConfig()
            pc.world_size = pc.tp_size = pc.ep_size = 1
            pc = resolve_qwen4_parallelism(config, pc)
            load = NewLoaderConfig.from_runtime(
                pc, device="cpu", compute_dtype=torch.bfloat16, load_method="scratch",
                moe_config=MoeConfig(), py_hw_kernel_config=HWKernelConfig(),
            )
            loader = NewModelLoader(config, load, model_path=directory)
            owner = loader.load()
            try:
                leaf = owner.layers[0].mlp.experts
                self.assertTrue(leaf.has_shared_expert_gate)
                remap = list(reversed(range(leaf.num_experts)))
                staged = loader.load_moe_layer_weights(
                    owner, layer_id=0, physical_to_logical=remap, ep_size=1, ep_rank=0
                )
                torch.testing.assert_close(staged[W.moe_w1], leaf.w13[remap])
                torch.testing.assert_close(staged[W.moe_w2], leaf.w2[remap])
            finally:
                owner.close()

    @unittest.skipUnless(os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "1", "native FST opt-in")
    def test_two_rank_whole_bf16_and_mixed_fst(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(_whole_worker,
                args=(str(Path(directory)/"gloo"),),nprocs=2,join=True,start_method="spawn")

    def test_actual_source_recipes_and_role_policy(self):
        for kind in ("bf16","fp8","nvfp4"):
            with self.subTest(source=kind), tempfile.TemporaryDirectory() as directory:
                config, _ = source_checkpoint(directory,kind)
                plan = Qwen4CheckpointPlan(directory,config)
                self.assertEqual(plan.source_kind,kind)
                self.assertIs(plan.model_config,config)
                self.assertEqual(plan.dense_fp8,kind == "fp8")
                self.assertEqual(plan.vocab_fp8,kind == "fp8")
                if kind == "fp8":
                    config._qwen4_dense_storage = "source_bf16"
                    config._qwen4_vocab_storage = "fp8"
                    plan = Qwen4CheckpointPlan(directory,config)
                    self.assertFalse(plan.dense_fp8)
                    self.assertTrue(plan.vocab_fp8)
                    self.assertTrue(plan.source_fp8)

    def test_unindexed_bf16_safetensors_are_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            config, values = source_checkpoint(directory,"bf16")
            (Path(directory)/"model.safetensors.index.json").unlink()
            plan = Qwen4CheckpointPlan(directory,config)
            self.assertEqual(set(plan.shapes),set(values))
            save_file({"lm_head.weight":values["lm_head.weight"]},str(Path(directory)/"duplicate.safetensors"))
            with self.assertRaisesRegex(ValueError,"Duplicate"):
                Qwen4CheckpointPlan(directory,config)


def _whole_worker(rank, rendezvous):
    torch.distributed.init_process_group("gloo",init_method="file://"+rendezvous,
        rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        for source in ("bf16","fp8"):
            for ep in (1,2):
                with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{"LOAD_METHOD":""}),patch.dict(MODEL_REGISTRY,{"qwen4_exp":Qwen4TargetWeights}):
                    config,_ = source_checkpoint(directory,source)
                    pc = ParallelismConfig()
                    pc.world_size = pc.tp_size = 2
                    pc.world_rank = pc.tp_rank = rank
                    pc.ep_size,pc.ep_rank = ep,rank if ep == 2 else 0
                    pc.local_rank = rank
                    pc = resolve_qwen4_parallelism(config,pc)
                    snapshots = []
                    for method in ("scratch","fastsafetensors"):
                        load = NewLoaderConfig.from_runtime(pc,device="cpu",compute_dtype=torch.bfloat16,
                            load_method=method,moe_config=MoeConfig(),py_hw_kernel_config=HWKernelConfig())
                        loader = NewModelLoader(config,load,model_path=directory)
                        owner = loader.load()
                        try:
                            assert loader.loaded_method.value == method
                            assert owner.checkpoint_plan.source_kind == source
                            assert owner.layers[0].mlp.experts.runtime is None  # CPU staging only.
                            assert owner.embed_tokens.weight.dtype == (torch.float8_e4m3fn if source == "fp8" else torch.bfloat16)
                            exports = Qwen4ForCausalLM.runtime_weight_view(owner)
                            if source == "fp8":
                                assert exports["embedding.qwen4_fp8_storage_scale"] is owner.embed_tokens.storage_scale
                                assert exports["lm_head.weight_only_quant_scale"] is owner.lm_head.storage_scale
                                assert exports["embedding"] is owner.embed_tokens.weight
                                assert exports["lm_head"] is owner.lm_head.weight
                            snapshots.append({key:value.detach().clone() for key,value in owner.state_dict().items()})
                        finally:
                            owner.close()
                    assert snapshots[0].keys() == snapshots[1].keys()
                    for key in snapshots[0]:
                        left,right = snapshots[0][key],snapshots[1][key]
                        if left.dtype == torch.float8_e4m3fn:
                            left,right = left.view(torch.uint8),right.view(torch.uint8)
                        torch.testing.assert_close(left,right,rtol=0,atol=0,msg=f"{source}/EP{ep}/rank{rank}/{key}")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()

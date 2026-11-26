"""Exercise the actual Qwen4 service class and native config, without GPU use.

Production registry routes are exercised without test-only owner substitution.
CPU publication does not establish device inference or golden accuracy.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.new_models.qwen4_exp.language import Qwen4ForCausalLM
from rtp_llm.models_py.new_models.qwen4_exp.test.test_target_weights import _checkpoint
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig


class CheckpointOnlyQwen4(Qwen4Exp):
    # Only external tokenizer/download and execution-device selection change.
    # load(), configuration validation, ownership and publication remain real.
    def load_tokenizer(self):
        pass

    def _get_device_str(self):
        return "cpu"

    def _init_custom_module(self):
        return None


def native_config(directory):
    root = Path(directory)
    _checkpoint(root)
    config_json = json.loads((root / "config.json").read_text())
    config_json["language_model_only"] = True
    config_json["text_config"].update(
        max_position_embeddings=128,
        num_experts_per_tok=1,
        shared_expert_intermediate_size=128,
        full_attention_interval=2,
        ngram_vocab_size_base=3,
        rope_parameters={"rope_theta":10000,"partial_rotary_factor":1.0,
                         "mrope_section":[8,4,4],"mrope_interleaved":True},
    )
    (root / "config.json").write_text(json.dumps(config_json))
    config = Qwen4Exp._create_config_from_json(directory, config_json)
    config.data_type = "bf16"
    config.generate_env_config = None
    config.enable_fp32_lm_head = False
    return config


class NativeQwen4ServiceTest(unittest.TestCase):
    def test_mixed_cpu_staging_cannot_publish_a_fake_gpu_runtime(self):
        # Runtime publication imports the real attention/cache native bindings.
        # This is not a portable checkpoint-only test, even with CPU tensors.
        from rtp_llm.models_py.new_models.qwen4_exp.test.test_precision_profiles import source_checkpoint

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOAD_METHOD": ""}):
            config, _ = source_checkpoint(directory, "fp8")
            pc = ParallelismConfig()
            pc.world_size = pc.tp_size = pc.ep_size = 1
            pc.world_rank = pc.tp_rank = pc.ep_rank = pc.local_rank = 0
            model = CheckpointOnlyQwen4(
                model_config=config, parallelism_config=pc,
                hw_kernel_config=HWKernelConfig(), kv_cache_config=KVCacheConfig(),
                fmha_config=None, moe_config=MoeConfig(), load_method="scratch",
                max_generate_batch_size=2, vit_config=None, merge_lora=False,
            )
            with self.assertRaisesRegex(ValueError, "FP8 vocabulary requires contiguous CUDA"):
                model.load()
            self.assertIsNone(model.py_model)
            self.assertIsNone(model.weight)

    def test_bf16_actual_service_scratch_and_fst(self):
        from rtp_llm.models_py.new_models.qwen4_exp.test.test_precision_profiles import source_checkpoint

        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOAD_METHOD": ""}):
                config, values = source_checkpoint(directory, "bf16")
                parallel = ParallelismConfig()
                parallel.world_size = parallel.tp_size = parallel.ep_size = 1
                parallel.world_rank = parallel.tp_rank = parallel.ep_rank = parallel.local_rank = 0
                hw = HWKernelConfig()
                hw.enable_cuda_graph = hw.use_swizzleA = False
                model = CheckpointOnlyQwen4.from_config(
                    model_config=config, parallelism_config=parallel,
                    hw_kernel_config=hw, kv_cache_config=KVCacheConfig(),
                    fmha_config=None, moe_config=MoeConfig(), load_method=method,
                    max_generate_batch_size=2, vit_config=None, merge_lora=False,
                )
                self.addCleanup(model.py_model.close)
                self.assertEqual(model.py_model.checkpoint_plan.source_kind, "bf16")
                self.assertEqual(model.model_weights_loader.loaded_method.value, method)
                self.assertEqual(model.py_model.layers[0].mlp.experts.w13.dtype, torch.bfloat16)
                self.assertIs(model.weight.get_global_weight("embedding"), model.py_model.embed_tokens.weight)
                embedding = model.py_model.embed_tokens.weight
                head = model.weight.get_global_weight("lm_head")
                old_embedding = embedding.clone()
                update = old_embedding + 1
                with self.assertRaises((ValueError, NotImplementedError, PermissionError)):
                    model.weight_manager.update_tensors({
                        "embedding": update,
                        "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight": update,
                    })
                torch.testing.assert_close(embedding, old_embedding)
                model.weight_manager.update_tensors({"embedding": update, "lm_head": head.clone() + 2})
                torch.testing.assert_close(model.py_model._runtime.embed_tokens.weight, update)
                torch.testing.assert_close(head, values["lm_head.weight"] + 2)
                qsa = model.py_model.layers[1].self_attn.qsa_indexer
                self.assertIs(qsa.rope_config, config.attn_config.rope_config)
                torch.testing.assert_close(
                    qsa.index_qk_proj.weight,
                    values["model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight"],
                    rtol=0, atol=0,
                )

    def test_actual_service_publishes_complete_target_owner(self):
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method):
                self._check_service(method)

    def _check_service(self, method):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{"LOAD_METHOD":""}):
            self.assertIs(get_model_class("qwen4_exp"), Qwen4ForCausalLM)
            config = native_config(directory)
            parallel = ParallelismConfig()
            parallel.world_size = parallel.tp_size = parallel.ep_size = 1
            parallel.world_rank = parallel.tp_rank = parallel.ep_rank = 0
            parallel.local_rank = 0
            hardware = HWKernelConfig()
            hardware.enable_cuda_graph = False
            hardware.use_swizzleA = False
            model = CheckpointOnlyQwen4.from_config(
                model_config=config,parallelism_config=parallel,
                hw_kernel_config=hardware,kv_cache_config=KVCacheConfig(),
                fmha_config=None,moe_config=MoeConfig(),load_method=method,
                max_generate_batch_size=2,vit_config=None,merge_lora=False)
            self.addCleanup(model.py_model.close)
            self.assertIsInstance(model.py_model,Qwen4ForCausalLM)
            self.assertEqual(model.parallelism_config.get_ffn_tp_size(),1)
            self.assertEqual(model.py_model.load_config.parallelism_config.get_ffn_tp_size(),1)
            self.assertIs(model.weight.get_global_weight("embedding"),model.py_model.embed_tokens.weight)
            self.assertIs(model.weight.get_global_weight("lm_head"),model.py_model.lm_head.weight)
            self.assertEqual(model.model_weights_loader.loaded_method.value,method)
            runtime = model.py_model._runtime
            from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel, Qwen4ExpAttention
            self.assertIs(type(runtime), Qwen4ExpModel)
            self.assertIs(type(runtime.layers[1].self_attn), Qwen4ExpAttention)
            # Standalone callers assign the public cache directly; do not let
            # that shadow the real execution object's cache field.
            model.py_model.kv_cache = None
            self.assertNotIn("kv_cache",model.py_model.__dict__)
            self.assertIsNone(runtime.kv_cache)
            self.assertEqual(len(runtime.layers),2)
            self.assertEqual(len(runtime.ple_layers),1)
            self.assertIsNotNone(runtime.layers[1].self_attn.qsa_indexer)
            self.assertTrue(runtime.layers[0].self_attn._qkvz_ba_fused)
            self.assertIsNotNone(model.weight_manager)
            x = torch.randn(3,128,dtype=torch.bfloat16)
            full_owner = model.py_model.layers[1].self_attn
            full_runtime = runtime.layers[1].self_attn
            torch.testing.assert_close(full_runtime.qkv_proj(x),torch.nn.functional.linear(x,full_owner.qkv_proj.weight),rtol=0,atol=0)
            torch.testing.assert_close(full_runtime.gate(x),torch.nn.functional.linear(x,full_owner.gate.weight),rtol=0,atol=0)
            gdn_owner = model.py_model.layers[0].linear_attn
            self._check_target_verify(runtime.layers[0].self_attn.decode_gdn, gdn_owner)
            qkvz, ba = runtime.layers[0].self_attn._input_project(x)
            torch.testing.assert_close(qkvz,torch.nn.functional.linear(x,gdn_owner.in_proj_qkvz.weight),rtol=0,atol=0)
            torch.testing.assert_close(ba,x @ gdn_owner.in_proj_ba_w,rtol=0,atol=0)
            self.assertEqual(gdn_owner.in_proj_qkvz.weight.untyped_storage().data_ptr(),
                             gdn_owner.in_proj_ba_w.untyped_storage().data_ptr())
            shared_owner = model.py_model.layers[0].mlp.shared_expert
            shared_runtime = runtime.layers[0].mlp.shared_expert
            gate_up = torch.cat((shared_owner.gate_proj.weight,shared_owner.up_proj.weight))
            torch.testing.assert_close(shared_runtime.up_proj(x),torch.nn.functional.linear(x,gate_up),rtol=0,atol=0)
            torch.testing.assert_close(shared_runtime.down_proj(x),torch.nn.functional.linear(x,shared_owner.down_proj.weight),rtol=0,atol=0)
            before = {name:(id(value),value.data_ptr()) for name,value in model.py_model.named_parameters()}
            model.py_model.process_weights_after_loading()
            self.assertEqual(before,{name:(id(value),value.data_ptr()) for name,value in model.py_model.named_parameters()})

    def _check_target_verify(self, decode, owner):
        # Exercise the inherited target-verify forward with a real loaded
        # owner. Only the final device kernel is replaced in this CPU test.
        h, hv = owner.local_num_k_heads, owner.local_num_v_heads
        k, v = owner.head_k_dim, owner.head_v_dim
        channels = owner.qkv_size
        cache = SimpleNamespace(
            kv_cache_base=torch.empty(
                2, decode.linear_cache_converter.block_size_bytes // 4, dtype=torch.float32
            ),
            group_id=0,
            linear_replay=SimpleNamespace(
                k=torch.empty(2, 2, h, k),
                u=torch.empty(2, 2, hv, v),
                g=torch.empty(2, 2, hv, 1),
                conv_inputs=torch.empty(2, 2, channels, dtype=decode.conv_state_dtype),
            ),
        )
        inputs = SimpleNamespace(prefix_lengths=torch.ones(1), linear_replay=object())
        metadata = SimpleNamespace(is_target_verify=True)
        qkv = torch.zeros(2, channels)
        b, a = torch.zeros(2, hv), torch.zeros(2, hv)
        with patch("rtp_llm.models_py.triton_kernels.linear_replay._linear_serial_replay") as launch:
            self.assertIs(decode(qkv, b, a, inputs, cache, metadata), launch.return_value)
            first = decode._linear_replay
            self.assertIs(launch.call_args.kwargs["conv_weights"], owner.conv1d_w)
            self.assertIs(launch.call_args.kwargs["dt_bias"], owner.dt_bias)
            self.assertIs(launch.call_args.kwargs["a_log"], owner.a_log)
            self.assertIs(launch.call_args.kwargs["replay_inputs"], inputs.linear_replay)
            decode(qkv, b, a, inputs, cache, metadata)
            self.assertIs(decode._linear_replay, first)
            cache.kv_cache_base = cache.kv_cache_base.clone()
            decode(qkv, b, a, inputs, cache, metadata)
            self.assertIsNot(decode._linear_replay, first)
            self.assertIs(decode._linear_replay.base, cache.kv_cache_base)

    def test_manifest_failure_does_not_publish_service_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = native_config(directory)
            index_path = Path(directory)/"model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"]["model.language_model.layers.1.mlp.experts.1.down_proj.weight_scale"]
            index_path.write_text(json.dumps(index))
            parallel = ParallelismConfig()
            parallel.world_size = parallel.tp_size = parallel.ep_size = 1
            parallel.world_rank = parallel.tp_rank = parallel.ep_rank = parallel.local_rank = 0
            model = CheckpointOnlyQwen4(model_config=config,parallelism_config=parallel,
                hw_kernel_config=HWKernelConfig(),kv_cache_config=KVCacheConfig(),
                fmha_config=None,moe_config=MoeConfig(),load_method="scratch",
                max_generate_batch_size=2,vit_config=None,merge_lora=False)
            with self.assertRaises(ValueError):
                model.load()
            self.assertIsNone(model.py_model)
            self.assertIsNone(model.weight)


if __name__ == "__main__":
    unittest.main()

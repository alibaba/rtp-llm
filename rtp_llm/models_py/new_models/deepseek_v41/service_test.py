"""Real V4.1 service config/runtime construction and CPU checkpoint staging.

No attention/MoE CUDA execution is substituted. Full load on CPU must reject
the native-only executor before the service publishes a model.
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v41 import DeepSeekV41
from rtp_llm.models.direct_checkpoint import (
    RuntimeWeightView,
    _runtime_view,
    load_owned_model,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41.backbone_test import checkpoint
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import KvCacheDataType, ParallelismConfig


def raw_config():
    return dict(
        num_hidden_layers=3,
        hidden_size=64,
        vocab_size=8,
        num_attention_heads=2,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=32,
        compress_ratios=[0, 2, 1],
        sliding_window=8,
        index_head_dim=32,
        index_n_heads=2,
        index_topk=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=64,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        swiglu_limit=10.0,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        engram_layer_ids=[],
        engram_num_embeddings=[],
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_compressed_vocab_size=8,
        engram_pad_token_id=0,
        engram_vocab_size=7,
    )


def config_and_load(folder, rank=0, *, engram=False, method="scratch"):
    raw = raw_config()
    if engram:
        raw.update(engram_layer_ids=[1], engram_num_embeddings=[8])
    with open(os.path.join(folder, "config.json"), "w") as stream:
        json.dump(raw, stream)
    config = DeepSeekV41._create_config(folder)
    config.model_type = "deepseek_v41"
    config.ckpt_path = folder
    config.data_type = "bf16"
    config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.ep_size = parallel.world_size = 2
    parallel.tp_rank = parallel.ep_rank = parallel.world_rank = rank
    load = NewLoaderConfig.from_runtime(
        parallel,
        device="cpu",
        compute_dtype=torch.bfloat16,
        load_method=method,
        max_generate_batch_size=2,
    )
    return config, load


class V41ServiceOwnerTest(unittest.TestCase):
    def test_reader_cannot_replace_the_service_attention_or_expert_rank(self):
        with tempfile.TemporaryDirectory() as root:
            for group in ("attention", "expert"):
                with self.subTest(group=group):
                    config, load = config_and_load(root, rank=0)
                    # Reader ranks are a snapshot. Simulate a stale snapshot
                    # by changing the serving group before owner construction.
                    if group == "attention":
                        load.parallelism_config.tp_rank = 1
                    else:
                        load.parallelism_config.ep_rank = 1
                    with self.assertRaisesRegex(ValueError, group + " rank mismatch"):
                        get_model_class(config.model_type)(config, load)

    def test_accelerator_preflight_requires_the_serving_binding_before_allocation(self):
        with tempfile.TemporaryDirectory() as root:
            model, _, _, _ = self.staged(root)
            with patch.dict(sys.modules, {"rtp_llm.ops.compute_ops": None}):
                with self.assertRaisesRegex(ImportError, "rtp_llm.ops.compute_ops"):
                    NewModelLoader._validate_runtime_backends(model, "cuda")

    def staged(self, root, rank=0, *, engram=False, method="scratch"):
        values = checkpoint()
        path = os.path.join(root, "model.safetensors")
        save_file(values, path)
        config, load = config_and_load(root, rank, engram=engram, method=method)
        cls = get_model_class(config.model_type)
        model = cls(config, load)
        loader = NewModelLoader(config, load, model_path=root)
        predicate = model.checkpoint_weight_name_filter()
        plan = {}
        resolver = loader._checkpoint_read_spec_resolver(model, predicate, None, plan)
        model._rtp_checkpoint_load_plan = plan
        weights = (
            loader._fastsafetensors_weights_iterator(
                [path], "cpu", predicate, None, resolver
            )
            if method == "fastsafetensors"
            else get_all_weights(
                [path], name_filter=predicate, read_spec_resolver=resolver
            )
        )
        model.load_weights(weights)
        loader._validate_loaded_weights(model)
        return model, config, loader, values

    def test_real_target_config_runtime_classes_and_checkpoint(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
        from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as root:
                model, config, loader, values = self.staged(root, rank)
                self.assertIsInstance(model.v4, V4Transformer)
                self.assertIsInstance(model.v4.layers[1].attn, AttentionV41FP8)
                self.assertIsInstance(model.v4.layers[1].attn_hc, DelayedHCUnit)
                self.assertEqual(model.v4.layers[2].attn.kv_source_layer_id, 1)
                torch.testing.assert_close(
                    model.v4.embed.weight,
                    values["embed.weight"][:, rank * 32 : (rank + 1) * 32],
                )
                model.weight = _runtime_view(model, config, "cpu")
                exports = dict(model.weight.global_weights)
                model._release_runtime_weight_source()
                self.assertFalse(hasattr(model, "weight"))
                self.assertIs(model.v4.embed.weight, exports["embedding"])
                self.assertIs(model.v4.embed.owner, model)
                self.assertEqual(model._engram_layers, ())

    def test_nonowning_runtime_view_cannot_replace_owned_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            model, config, _, _ = self.staged(root)
            exports = model.runtime_weight_view()
            exports["embedding"] = exports["embedding"].clone()
            model.weight = RuntimeWeightView(
                [{}, {}, {}], exports, torch.bfloat16, "cpu"
            )
            with self.assertRaisesRegex(RuntimeError, "replace owned parameters"):
                model._release_runtime_weight_source()

    def test_full_loader_rejects_cpu_executor_not_false_success(self):
        with tempfile.TemporaryDirectory() as root:
            save_file(checkpoint(), os.path.join(root, "model.safetensors"))
            config, load = config_and_load(root)
            loader = NewModelLoader(config, load, model_path=root)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                loader.load()
            self.assertIsNone(loader.loaded_method)
            service = SimpleNamespace(
                model_config=config,
                parallelism_config=load.parallelism_config,
                load_method=load.load_method,
                _weight_alias_names=(),
                _get_device_str=lambda: "cpu",
                hw_kernel_config=None,
                moe_config=None,
                fmha_config=None,
                max_generate_batch_size=2,
                custom_module=None,
                merge_lora=False,
                py_model=None,
                checkpoint_owner=None,
                weight=None,
            )
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                load_owned_model(service)
            self.assertIsNone(service.py_model)
            self.assertIsNone(service.checkpoint_owner)

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires native FST"
    )
    def test_target_extra_loads_host_engram_with_selected_source(self):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                tokenizer = Tokenizer(
                    models.WordLevel(
                        {"[UNK]": 0, **{chr(96 + i): i for i in range(1, 8)}},
                        unk_token="[UNK]",
                    )
                )
                PreTrainedTokenizerFast(
                    tokenizer_object=tokenizer, unk_token="[UNK]"
                ).save_pretrained(root)
                host = {
                    "embed.weight": torch.ones(8, 32).to(torch.float8_e4m3fn),
                    "embed.scale": torch.full((8, 1), 126, dtype=torch.uint8).view(
                        torch.float8_e8m0fnu
                    ),
                    "wkv.weight": torch.ones(192, 32).to(torch.float8_e4m3fn),
                    "wkv.scale": torch.full((6, 1), 126, dtype=torch.uint8).view(
                        torch.float8_e8m0fnu
                    ),
                    "q_weight": torch.ones(2, 64, dtype=torch.bfloat16),
                    "k_weight": torch.ones(2, 64, dtype=torch.bfloat16),
                }
                save_file(
                    {"layers.1.engram." + name: value for name, value in host.items()},
                    os.path.join(root, "engram.safetensors"),
                )
                index = {name: "model.safetensors" for name in checkpoint()}
                index.update(
                    {"layers.1.engram." + name: "engram.safetensors" for name in host}
                )
                with open(
                    os.path.join(root, "model.safetensors.index.json"), "w"
                ) as stream:
                    json.dump({"weight_map": index}, stream)
                model, config, _, _ = self.staged(root, engram=True, method=method)
                model.weight = _runtime_view(model, config, "cpu")
                model._release_runtime_weight_source()
                self.assertEqual(model._engram_layers, ((1, 0),))
                owner = model.v4.layers[1].engram
                self.assertEqual(len(list(owner.parameters())), 6)
                self.assertEqual(owner.embed.weight.device.type, "cpu")
                self.assertTrue(
                    torch.equal(
                        owner.embed.weight.view(torch.uint8),
                        host["embed.weight"].view(torch.uint8),
                    )
                )
                hashes = model._engram_hash_state.hash_token_windows(
                    torch.tensor([[1, 2], [3, -1]])
                )
                self.assertEqual(tuple(hashes.shape), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()

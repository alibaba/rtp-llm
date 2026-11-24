"""Public NextN draft loading, from separate and combined DeepSeek checkpoints."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV3Mtp
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import DeepseekV3YarnRotaryEmbedding
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.deepseek_v3.language import DeepSeekV32ForCausalLM
from rtp_llm.models_py.new_models.deepseek_v3.test.test_score_route import _score_weights
from rtp_llm.models_py.new_models.deepseek_v3_mtp.language import DeepSeekV32MTPForCausalLM
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


class _CpuCheckpoint:
    # The production BaseModel.from_config selects and loads the owner. Only
    # the tokenizer, unrelated custom handlers, and target device are replaced.
    def load_tokenizer(self):
        pass

    def _get_device_str(self):
        return "cpu"

    def _init_custom_module(self):
        return None


class _Draft(_CpuCheckpoint, DeepSeekV3Mtp):
    pass


class _Score(_CpuCheckpoint, DeepSeekV2):
    pass


def fixture(combined):
    score = _score_weights()
    idx = 2 if combined else 0
    prefix = f"model.layers.{idx}."
    draft = {
        prefix + k[len("model.layers.1."):]: v + 2.0
        for k, v in score.items() if k.startswith("model.layers.1.")
    }
    draft.update({
        prefix+"enorm.weight": torch.full((32,), 0.5),
        prefix+"hnorm.weight": torch.full((32,), 3.0),
        prefix+"eh_proj.weight": torch.arange(2048).reshape(32, 64).float() / 2048,
        prefix+"embed_tokens.weight": score["model.embed_tokens.weight"] + 5,
        prefix+"shared_head.norm.weight": torch.full((32,), 7.0),
        prefix+"shared_head.head.weight": score["lm_head.weight"] + 9,
    })
    cfg = dict(
        architectures=["DeepseekV3ForCausalLM" if combined else "DeepseekV3ForCausalLMNextN"],
        model_type="deepseek_v3", hidden_size=32, intermediate_size=32,
        max_position_embeddings=128, moe_intermediate_size=32, topk_method="noaux_tc",
        scoring_func="sigmoid", norm_topk_prob=True, num_attention_heads=4, n_group=1,
        topk_group=1, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        num_hidden_layers=2 if combined else 1, num_nextn_predict_layers=1,
        first_k_dense_replace=1 if combined else 0, moe_layer_freq=1, q_lora_rank=32,
        kv_lora_rank=32, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
        vocab_size=32, routed_scaling_factor=2.5, rms_norm_eps=1e-6,
        rope_theta=10000, torch_dtype="bfloat16",
    )
    return cfg, (dict(score, **draft) if combined else draft), prefix


def load_service(root, method, cls=_Draft, parallel=None, tag=None, override=None):
    cfg = cls._create_config(root)
    cfg.ckpt_path, cfg.model_type, cfg.max_seq_len = root, tag or ("deepseek-v3-mtp" if cls is _Draft else "deepseek3"), 128
    cfg.attn_config.tokens_per_block = cfg.attn_config.kernel_tokens_per_block = 64
    if override is not None:
        cfg.apply_override_args(json.dumps(override))
    cls._post_build_model_config(cfg)
    cfg.data_type, cfg.enable_fp32_lm_head, cfg.lora_infos = "bf16", False, {}
    return cls.from_config(
        model_config=cfg, parallelism_config=parallel or ParallelismConfig(),
        hw_kernel_config=SimpleNamespace(use_swizzleA=False, enable_cuda_graph=False),
        kv_cache_config=SimpleNamespace(multi_task_prompt="", multi_task_prompt_str=""),
        fmha_config=None, moe_config=None, max_generate_batch_size=1,
        load_method=method, vit_config=None, merge_lora=False,
    )


def _rank_load(rank, root, init):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        pc = ParallelismConfig()
        pc.world_size = pc.local_world_size = pc.tp_size = pc.ep_size = 2
        pc.world_rank = pc.local_rank = pc.tp_rank = pc.ep_rank = rank
        pc.ffn_tp_size, pc.ffn_tp_rank = 2, rank
        _, data, prefix = fixture(True)
        for method in ("scratch", "fastsafetensors"):
            service = load_service(root, method, parallel=pc)
            assert service.model_weights_loader.loaded_method == NewLoaderLoadMethod(method)
            model = service.py_model
            experts = model.layers[0].mlp.experts
            # EP splits the expert axis, not the experts' intermediate dimension.
            expected = torch.stack([data[prefix+f"mlp.experts.{i}.down_proj.weight"]
                                    for i in range(rank*2, rank*2+2)])
            torch.testing.assert_close(experts.w2, expected.to(experts.w2))
            # Embedding is vocabulary-sharded, while the EH projection is replicated.
            torch.testing.assert_close(model.embed_tokens.weight,
                                       data[prefix+"embed_tokens.weight"].chunk(2)[rank].to(model.embed_tokens.weight))
            torch.testing.assert_close(model.mtp_block.fc.weight,
                                       data[prefix+"eh_proj.weight"].to(model.mtp_block.fc.weight))
            # EPLB callbacks address runtime layer 0, not the appended
            # checkpoint layer 2. The new physical slots must read draft experts.
            placement = [3, 2, 1, 0]
            reload = service.model_weights_loader.load_moe_layer_weights(
                model, layer_id=0, physical_to_logical=placement, ep_rank=rank, ep_size=2,
            )
            changed = torch.stack([data[prefix+f"mlp.experts.{i}.down_proj.weight"]
                                   for i in placement[rank*2:rank*2+2]])
            torch.testing.assert_close(reload[W.moe_w2], changed.to(reload[W.moe_w2]))
    finally:
        dist.destroy_process_group()


class NextNRouteTest(unittest.TestCase):
    def test_yarn_override_reaches_score_and_nextn_runtime_cache(self):
        cfg, weights, _ = fixture(True)
        override = {"rope_scaling": dict(type="yarn", factor=8.0,
                    original_max_position_embeddings=16, beta_fast=2.0, beta_slow=0.5)}
        for disk_yarn in (False, True):
            # Plain checkpoints may gain YaRN; scaled checkpoints must keep
            # their DeepSeek magnitude coefficients when the frequencies change.
            mscale, all_dim = (0.5, 0.0) if disk_yarn else (1.0, 1.0)
            if disk_yarn:
                cfg["rope_scaling"] = dict(type="yarn", factor=2.0,
                    original_max_position_embeddings=64, mscale=mscale, mscale_all_dim=all_dim)
            legacy = DeepseekV3YarnRotaryEmbedding(4, 128, 10000,
                scaling_factor=8.0, original_max_position_embeddings=16,
                beta_fast=2.0, beta_slow=0.5, mscale=mscale, mscale_all_dim=all_dim)
            expected = torch.cat((legacy.cos_cached[:, :2], legacy.sin_cached[:, :2]), -1)
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / "config.json").write_text(json.dumps(cfg))
                save_file(weights, str(Path(root) / "model.safetensors"))
                for cls in (_Score, _Draft):
                    with self.subTest(disk_yarn=disk_yarn, service=cls.__name__):
                        model = load_service(root, "scratch", cls, override=override).py_model
                        self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None)))
                        actual = model.weight.get_global_weight_or_none(W.rope_cos_sin_cache)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_public_registry_distinguishes_nextn_from_target(self):
        for architecture, service_tag, owner in (
            ("DeepseekV3ForCausalLMNextN", "deepseek-v3-mtp", DeepSeekV32MTPForCausalLM),
            ("DeepseekV3ForCausalLM", "deepseek3", DeepSeekV32ForCausalLM),
        ):
            with self.subTest(architecture=architecture):
                resolved = ModelDict.get_ft_model_type_by_config({"architectures": [architecture]})
                self.assertEqual(resolved, service_tag)
                self.assertIs(get_model_class(resolved), owner)

    def test_kimi_documented_target_with_separate_draft(self):
        # docs/references/kimi/deploy.md selects the deepseek3 service with a
        # separately supplied deepseek-v3-mtp checkpoint. K2's score config
        # declares no appended draft. Use reduced dimensions and BF16 here;
        # this is not a validation of the published full-size FP8 checkpoint.
        score_cfg, _, _ = fixture(True)
        score_cfg.update(model_type="kimi_k2", num_nextn_predict_layers=0,
                         routed_scaling_factor=2.827, rope_theta=50000.0,
                         rope_scaling=dict(type="yarn", factor=32.0, beta_fast=1.0,
                                           beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0,
                                           original_max_position_embeddings=4096))
        draft_cfg, draft_data, _ = fixture(False)
        with tempfile.TemporaryDirectory() as parent:
            score_dir, draft_dir = Path(parent)/"score", Path(parent)/"draft"
            for folder, cfg, weights in ((score_dir, score_cfg, _score_weights()),
                                         (draft_dir, draft_cfg, draft_data)):
                folder.mkdir()
                (folder/"config.json").write_text(json.dumps(cfg))
                save_file(weights, str(folder/"model.safetensors"))
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    score = load_service(str(score_dir), method, _Score)
                    draft = load_service(str(draft_dir), method)
                    self.assertEqual(draft.speculative_weight_alias_names(score, draft.model_config), ())
                    torch.testing.assert_close(score.py_model.embed_tokens.weight,
                                               _score_weights()["model.embed_tokens.weight"].to(score.py_model.embed_tokens.weight))
                    self.assertNotEqual(score.py_model.lm_head.weight.data_ptr(), draft.py_model.lm_head.weight.data_ptr())
                    with self.assertRaisesRegex(ValueError, "appended"):
                        load_service(str(score_dir), method)

    def test_explicit_standalone_without_architecture_marker(self):
        cfg, data, _ = fixture(False)
        cfg.pop("architectures")
        with tempfile.TemporaryDirectory() as root:
            save_file(data, str(Path(root)/"model.safetensors"))
            for declared in (0, 1):
                cfg["num_nextn_predict_layers"] = declared
                (Path(root)/"config.json").write_text(json.dumps(cfg))
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(declared=declared, method=method):
                        draft = load_service(root, method)
                        self.assertEqual(draft.py_model._checkpoint_layer, 0)

    def test_ambiguous_depth_uses_named_projection_not_iteration_order(self):
        cfg, original, prefix = fixture(True)
        cfg["num_hidden_layers"] = 1
        cfg["first_k_dense_replace"] = 1
        data = {key.replace(prefix, "model.layers.1."): tensor for key, tensor in original.items()
                if not key.startswith("model.layers.1.")}
        for suffix, method in (("safetensors", "fastsafetensors"), ("bin", "scratch")):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as root:
                (Path(root)/"config.json").write_text(json.dumps(cfg))
                name = str(Path(root)/("model."+suffix))
                write = save_file if suffix == "safetensors" else torch.save
                write(data, name)
                self.assertEqual(load_service(root, method).py_model._checkpoint_layer, 1)
                write(dict(data, **{"model.layers.0.eh_proj.weight": torch.ones(32, 64)}), name)
                with self.assertRaisesRegex(ValueError, "two EH projections"):
                    load_service(root, method)

    def test_combined_two_rank_gloo(self):
        cfg, data, _ = fixture(True)
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/"config.json").write_text(json.dumps(cfg))
            save_file(data, str(Path(root)/"model.safetensors"))
            torch.multiprocessing.spawn(_rank_load, args=(root, (Path(root)/"dist").as_uri()), nprocs=2)

    def test_separate_and_combined(self):
        for combined in (True, False):
            cfg, data, prefix = fixture(combined)
            data["model.rotary_emb.inv_freq"] = torch.ones(2)
            with tempfile.TemporaryDirectory() as root:
                (Path(root)/"config.json").write_text(json.dumps(cfg))
                save_file(data, str(Path(root)/"model.safetensors"))
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(combined=combined, method=method):
                        score = load_service(root, method, _Score) if combined else None
                        draft = load_service(root, method)
                        model = draft.py_model
                        self.assertIsInstance(model, DeepSeekV32MTPForCausalLM)
                        self.assertEqual(draft.model_weights_loader.loaded_method, NewLoaderLoadMethod(method))
                        self.assertEqual(draft.model_config.num_layers, 1)
                        self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None)))
                        self.assertEqual(len(model.weight.weights), 1)
                        self.assertEqual(draft._weight_alias_names, ())
                        for dest, key in ((model.embed_tokens.weight, "embed_tokens.weight"),
                                          (model.norm.weight, "shared_head.norm.weight"),
                                          (model.lm_head.weight, "shared_head.head.weight"),
                                          (model.mtp_block.fc.weight, "eh_proj.weight")):
                            torch.testing.assert_close(dest, data[prefix+key].to(dest))
                        if score is not None:
                            self.assertIsNot(score.py_model.embed_tokens.weight, model.embed_tokens.weight)
                            self.assertNotEqual(score.py_model.lm_head.weight.data_ptr(), model.lm_head.weight.data_ptr())
                        ids = torch.tensor([3, 13])
                        embeds = model.embed_tokens.weight[ids]
                        hidden = torch.arange(64).reshape(2,32).to(embeds) / 7
                        def norm(x, key):
                            y = x.float()
                            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + cfg["rms_norm_eps"])
                            return (y * data[prefix+key]).to(x)
                        expected = torch.nn.functional.linear(
                            torch.cat((norm(hidden,"hnorm.weight"), norm(embeds,"enorm.weight")), -1),
                            data[prefix+"eh_proj.weight"].to(embeds),
                        )
                        torch.testing.assert_close(model.embedding(SimpleNamespace(input_ids=ids, input_hiddens=hidden)), expected)

    def test_missing_draft_head_does_not_borrow_main_head(self):
        cfg, data, prefix = fixture(True)
        del data[prefix+"shared_head.head.weight"]
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/"config.json").write_text(json.dumps(cfg))
            save_file(data, str(Path(root)/"model.safetensors"))
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method), self.assertRaisesRegex((ValueError, RuntimeError), "lm_head"):
                    load_service(root, method)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen2_mtp import Qwen2MTPForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class


def _load_tp_pair(rank, root):
    from datetime import timedelta

    import torch.distributed as dist

    from rtp_llm.models_py.model_loader import NewLoaderLoadMethod

    dist.init_process_group(
        "gloo",
        init_method="file://" + os.path.join(root, "rendezvous"),
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=45),
    )
    try:
        case = Qwen2MTPLoadTest()
        case.setUp()
        pc = types.SimpleNamespace(
            tp_size=2,
            tp_rank=rank,
            ep_size=1,
            ep_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
            world_size=2,
            world_rank=rank,
            local_rank=rank,
            get_attn_tp_size=lambda: 2,
            get_attn_tp_rank=lambda: rank,
            get_ffn_tp_size=lambda: 2,
            get_ffn_tp_rank=lambda: rank,
        )
        for method in ("scratch", "fastsafetensors"):
            cfg = NewLoaderConfig.from_runtime(
                pc,
                device="cpu",
                compute_dtype=torch.float32,
                quant_config=QuantizationConfig("none"),
                load_method=method,
            )
            loader = NewModelLoader(case.config, cfg, model_path=root)
            draft = loader.load()
            case.assertEqual(loader.loaded_method, NewLoaderLoadMethod(method))
            source = case.weights()
            torch.testing.assert_close(
                draft.embed_tokens.weight,
                source["model.embed_tokens.weight"].chunk(2)[rank],
            )
            torch.testing.assert_close(draft.lm_head.weight, draft.embed_tokens.weight)
            torch.testing.assert_close(
                draft.mtp_block.fc.weight, source["model.layers.0.eh_proj.weight"]
            )
    finally:
        dist.destroy_process_group()


class _MTPCheckpointFixture(unittest.TestCase):
    def setUp(self):
        self.config = types.SimpleNamespace(
            model_type="qwen_2-mtp",
            is_mtp=True,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[],
            num_layers=1,
            vocab_size=8,
            hidden_size=4,
            inter_size=4,
            attn_config=types.SimpleNamespace(
                head_num=2, kv_head_num=1, size_per_head=2
            ),
            layernorm_eps=1e-6,
            enable_fp32_lm_head=False,
            tie_word_embeddings=True,
        )

    def weights(self):
        weights = {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
                8, 4
            )
        }
        for name, shape in {
            "input_layernorm.weight": (4,),
            "self_attn.q_proj.weight": (4, 4),
            "self_attn.k_proj.weight": (2, 4),
            "self_attn.v_proj.weight": (2, 4),
            "self_attn.q_proj.bias": (4,),
            "self_attn.k_proj.bias": (2,),
            "self_attn.v_proj.bias": (2,),
            "self_attn.o_proj.weight": (4, 4),
            "post_attention_layernorm.weight": (4,),
            "mlp.gate_proj.weight": (4, 4),
            "mlp.up_proj.weight": (4, 4),
            "mlp.down_proj.weight": (4, 4),
        }.items():
            weights[f"model.layers.0.{name}"] = torch.ones(shape)
        weights.update(
            {
                "model.layers.0.e_norm.weight": torch.full((4,), 2.0),
                "model.layers.0.h_norm.weight": torch.full((4,), 3.0),
                "model.layers.0.eh_proj.weight": torch.cat(
                    (torch.eye(4), 2 * torch.eye(4)), -1
                ),
                "model.layers.0.final_head.norm.weight": torch.full((4,), 4.0),
            }
        )
        return weights

    def load(self, weights, *, rank=0, tp=1):
        parallelism = types.SimpleNamespace(
            tp_size=tp,
            tp_rank=rank,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            pp_size=1,
            pp_rank=0,
            prefill_cp_config=None,
            ffn_disaggregate_config=None,
        )
        load_config = NewLoaderConfig(
            tp_size=tp,
            tp_rank=rank,
            attn_tp_size=tp,
            attn_tp_rank=rank,
            ffn_tp_size=tp,
            ffn_tp_rank=rank,
            lm_head_tp_size=tp,
            lm_head_tp_rank=rank,
            compute_dtype=torch.float32,
            device="cpu",
            load_method="scratch",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, f"{directory}/model.safetensors")
            return NewModelLoader(
                model_config=self.config,
                load_config=load_config,
                model_path=directory,
            ).load()


class Qwen2MTPLoadTest(_MTPCheckpointFixture):
    def test_actual_two_rank_checkpoint_transport(self):
        with tempfile.TemporaryDirectory() as root:
            save_file(self.weights(), os.path.join(root, "model.safetensors"))
            torch.multiprocessing.start_processes(
                _load_tp_pair, args=(root,), nprocs=2, join=True, start_method="fork"
            )

    def test_registry_and_real_safetensors_tp_rank_loading(self):
        self.assertIs(get_model_class("qwen_2-mtp"), Qwen2MTPForCausalLM)
        weights = self.weights()
        for rank in (0, 1):
            with self.subTest(rank=rank):
                model = self.load(weights, rank=rank, tp=2)
                torch.testing.assert_close(
                    model.mtp_block.fc.weight, weights["model.layers.0.eh_proj.weight"]
                )
                torch.testing.assert_close(
                    model.norm.weight, weights["model.layers.0.final_head.norm.weight"]
                )
                torch.testing.assert_close(
                    model.embed_tokens.weight,
                    weights["model.embed_tokens.weight"][rank * 4 : (rank + 1) * 4],
                )
                torch.testing.assert_close(
                    model.lm_head.weight, model.embed_tokens.weight
                )
                self.assertIs(
                    model.runtime_weight_view()["embedding"], model.embed_tokens.weight
                )
                self.assertIs(
                    model.runtime_weight_view()["lm_head"], model.lm_head.weight
                )

    def test_hidden_first_projection_matches_legacy_order(self):
        model = self.load(self.weights())
        embedding = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        hidden = torch.tensor([[4.0, 1.0, 2.0, 3.0]])
        # CPU normalization here exercises the actual MTP module, not an
        # attention backend or mocked projection.
        e = (
            embedding
            * torch.rsqrt(embedding.square().mean(-1, keepdim=True) + 1e-6)
            * 2
        )
        h = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6) * 3
        torch.testing.assert_close(model.mtp_block(embedding, hidden), h + 2 * e)

    def test_optional_sampler_maps_remain_int64_and_alias_runtime_view(self):
        weights = self.weights()
        weights["model.d2t"] = torch.arange(8, dtype=torch.int64)
        weights["model.t2d"] = torch.arange(8, dtype=torch.int64)
        model = self.load(weights)
        self.assertIs(
            model.runtime_weight_view()["multi_tokens_predict_d2t_map"], model.d2t
        )
        self.assertEqual(model.d2t.dtype, torch.int64)
        torch.testing.assert_close(model.t2d, weights["model.t2d"])

    def test_missing_fusion_parameter_fails_real_load(self):
        weights = self.weights()
        del weights["model.layers.0.h_norm.weight"]
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            self.load(weights)

    def test_multiple_decoder_layers_keep_indices_and_layer_zero_fusion(self):
        weights = self.weights()
        for name, tensor in tuple(weights.items()):
            if name.startswith("model.layers.0."):
                weights[name.replace("model.layers.0.", "model.layers.1.")] = (
                    tensor + 10
                )
        self.config.num_layers = 2
        model = self.load(weights)
        self.assertEqual([layer.self_attn.layer_idx for layer in model.layers], [0, 1])
        torch.testing.assert_close(model.norm.weight, torch.full((4,), 4.0))
        torch.testing.assert_close(
            model.layers[1].self_attn.o_proj.weight, torch.full((4, 4), 11.0)
        )

    def test_incomplete_or_wrong_dtype_maps_fail(self):
        weights = self.weights()
        weights["model.d2t"] = torch.arange(8, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "both d2t and t2d"):
            self.load(weights)
        weights["model.d2t"] = torch.arange(8, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "integer vocabulary map"):
            self.load(weights)

    def test_forward_requires_matching_target_hidden_states(self):
        model = self.load(self.weights())
        with self.assertRaisesRegex(ValueError, "requires input_hiddens"):
            model(types.SimpleNamespace(input_hiddens=None))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            model(
                types.SimpleNamespace(
                    input_ids=torch.tensor([1]), input_hiddens=torch.zeros(2, 4)
                )
            )


class Qwen2MTPNativeBoundaryTest(_MTPCheckpointFixture):
    def test_target_and_draft_public_factory_own_distinct_vocabularies(self):
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.models.base_model import BaseModel
        from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
        from rtp_llm.ops import SpeculativeType, TaskType

        ns = types.SimpleNamespace
        pc = ns(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
            world_size=1,
            world_rank=0,
            local_rank=0,
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
            get_ffn_tp_size=lambda: 1,
            get_ffn_tp_rank=lambda: 0,
        )

        def config(path, draft):
            cfg = ns(**vars(self.config))
            cfg.model_type = "qwen_2-mtp" if draft else "qwen_2"
            cfg.is_mtp = draft
            cfg.tie_word_embeddings = False
            cfg.task_type = TaskType.LANGUAGE_MODEL
            cfg.model_name = cfg.model_type
            cfg.ckpt_path, cfg.quant_config, cfg.lora_infos = path, None, {}
            cfg.generate_env_config, cfg.compute_dtype = None, torch.float32
            cfg.enable_output_vocab_pruning = False
            cfg.eplb_config = ns(enable_eplb=lambda: False)
            cfg.max_seq_len, cfg.gen_num_per_cycle = 32, 3
            return cfg

        for method in ("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as target_path, tempfile.TemporaryDirectory() as draft_path, patch.object(
                BaseModel, "load_tokenizer"
            ), patch.object(
                BaseModel, "_get_device_str", return_value="cpu"
            ):
                engine = ns(
                    parallelism_config=pc,
                    runtime_config=ns(
                        warm_up=False, model_warm_up=False, max_generate_batch_size=2
                    ),
                    sp_config=ns(type=SpeculativeType.EAGLE, gen_num_per_cycle=3),
                    load_config=ns(
                        load_method=method,
                        force_cpu_load_weights=False,
                        loader_recycle_handles=False,
                        moe_pure_tp_preshard=False,
                    ),
                    kv_cache_config=ns(multi_task_prompt="", multi_task_prompt_str=""),
                    fmha_config=None,
                    moe_config=None,
                    hw_kernel_config=ns(enable_cuda_graph=False, use_swizzleA=False),
                )
                cfg, draft_cfg = config(target_path, False), config(draft_path, True)
                draft = self.weights()
                target = {
                    name: value
                    for name, value in draft.items()
                    if not any(
                        k in name
                        for k in ("e_norm.", "h_norm.", "eh_proj.", "final_head.")
                    )
                }
                target["model.norm.weight"] = torch.ones(4)
                target["lm_head.weight"] = torch.full((8, 4), 17.0)
                draft["model.embed_tokens.weight"] = (
                    torch.arange(32).reshape(8, 4).float() + 47
                )
                draft["lm_head.weight"] = torch.full((8, 4), 53.0)
                save_file(target, os.path.join(target_path, "model.safetensors"))
                draft_file = os.path.join(draft_path, "model.safetensors")
                save_file(draft, draft_file)
                score = ModelFactory._create_model(cfg, engine)
                proposed = ModelFactory.get_sp_model(
                    cfg, draft_cfg, engine, target_model=score
                )
                self.assertIsInstance(proposed.model.py_model, Qwen2MTPForCausalLM)
                self.assertNotIsInstance(score.py_model, Qwen2MTPForCausalLM)
                for obj in (score, proposed.model):
                    self.assertEqual(
                        obj.model_weights_loader.loaded_method,
                        NewLoaderLoadMethod(method),
                    )
                for role, key in (
                    ("embedding", "model.embed_tokens.weight"),
                    ("lm_head", "lm_head.weight"),
                ):
                    param = proposed.model.weight.get_global_weight(role)
                    self.assertNotEqual(
                        param.data_ptr(),
                        score.weight.get_global_weight(role).data_ptr(),
                    )
                    torch.testing.assert_close(param, draft[key])
                    torch.testing.assert_close(
                        score.weight.get_global_weight(role), target[key]
                    )
                del draft["lm_head.weight"]
                save_file(draft, draft_file)
                with self.assertRaisesRegex(RuntimeError, "missing required"):
                    ModelFactory.get_sp_model(
                        cfg, draft_cfg, engine, target_model=score
                    )

    def test_full_single_token_draft_forward_matches_dense_reference(self):
        # For one token softmax(QK) is exactly one: both query heads return
        # the one KV head's V. Exercise real loaded projections, not CUDA FMHA.
        class OneTokenAttention:
            def forward(self, packed, cache, layer):
                if packed.shape != (1, 8) or cache is not None or layer != 0:
                    raise AssertionError("unexpected GQA geometry")
                return packed[:, -2:].repeat(1, 2)

        weights = self.weights()
        rng = torch.Generator().manual_seed(9430)
        for name, prototype in weights.items():
            weights[name] = torch.randn(prototype.shape, generator=rng) * 0.3
        owner = self.load(weights)
        ids = torch.tensor([3])
        h = torch.tensor([[0.2, 0.7, -1.0, 0.4]])
        output = owner(
            types.SimpleNamespace(input_ids=ids, input_hiddens=h), OneTokenAttention()
        )
        p = "model.layers.0."

        def norm(x, name):
            return (
                x
                * torch.rsqrt(
                    x.square().mean(-1, keepdim=True) + self.config.layernorm_eps
                )
                * weights[p + name + ".weight"]
            )

        def linear(x, name, bias=False):
            return torch.nn.functional.linear(
                x,
                weights[p + name + ".weight"],
                weights[p + name + ".bias"] if bias else None,
            )

        e = weights["model.embed_tokens.weight"][ids]
        x = linear(torch.cat([norm(h, "h_norm"), norm(e, "e_norm")], dim=-1), "eh_proj")
        v = linear(norm(x, "input_layernorm"), "self_attn.v_proj", bias=True)
        x = x + linear(v.repeat(1, 2), "self_attn.o_proj")
        hidden = norm(x, "post_attention_layernorm")
        mlp = torch.nn.functional.silu(linear(hidden, "mlp.gate_proj")) * linear(
            hidden, "mlp.up_proj"
        )
        expected = norm(x + linear(mlp, "mlp.down_proj"), "final_head.norm")
        torch.testing.assert_close(output.hidden_states, expected)


if __name__ == "__main__":
    unittest.main()

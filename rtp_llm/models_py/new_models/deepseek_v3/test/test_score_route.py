"""Dense and sparse DeepSeek-V3 family reach the public direct serving owner."""

import json
import math
import tempfile
import unittest
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.deepseek_v3.language import DeepSeekV32ForCausalLM
from rtp_llm.models_py.new_models.deepseek_v3.moe import DeepSeekV32MoEBlock
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


def _deterministic_values(shape, offset):
    count = math.prod(shape)
    return (
        torch.arange(offset, offset + count, dtype=torch.float32).reshape(shape)
        / max(count, 1)
        - 0.5
    )


def _score_weights():
    values = _deterministic_values
    weights = {
        "model.embed_tokens.weight": values((32, 32), 0),
        "model.norm.weight": values((32,), 3) + 1.0,
        "lm_head.weight": values((32, 32), 5),
    }
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}."
        offset = 11 + layer_idx * 100
        weights.update(
            {
                prefix + "input_layernorm.weight": values((32,), offset) + 1.0,
                prefix + "self_attn.q_a_proj.weight": values((32, 32), offset + 3),
                prefix
                + "self_attn.q_a_layernorm.weight": values((32,), offset + 5)
                + 1.0,
                prefix + "self_attn.q_b_proj.weight": values((32, 32), offset + 7),
                prefix
                + "self_attn.kv_a_proj_with_mqa.weight": values((36, 32), offset + 11),
                prefix
                + "self_attn.kv_a_layernorm.weight": values((32,), offset + 13)
                + 1.0,
                prefix + "self_attn.kv_b_proj.weight": values((32, 32), offset + 17),
                prefix + "self_attn.o_proj.weight": values((32, 16), offset + 19),
                prefix
                + "post_attention_layernorm.weight": values((32,), offset + 23)
                + 1.0,
            }
        )
    weights.update(
        {
            "model.layers.0.mlp.gate_proj.weight": values((32, 32), 31),
            "model.layers.0.mlp.up_proj.weight": values((32, 32), 37),
            "model.layers.0.mlp.down_proj.weight": values((32, 32), 41),
            "model.layers.1.mlp.gate.weight": values((4, 32), 131),
            "model.layers.1.mlp.gate.e_score_correction_bias": values((4,), 137),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": values((32, 32), 139),
            "model.layers.1.mlp.shared_experts.up_proj.weight": values((32, 32), 149),
            "model.layers.1.mlp.shared_experts.down_proj.weight": values((32, 32), 151),
        }
    )
    for expert_id in range(4):
        prefix = f"model.layers.1.mlp.experts.{expert_id}."
        offset = 157 + expert_id * 17
        weights[prefix + "gate_proj.weight"] = values((32, 32), offset)
        weights[prefix + "up_proj.weight"] = values((32, 32), offset + 3)
        weights[prefix + "down_proj.weight"] = values((32, 32), offset + 5)
    return weights


def _check_expert_reduction(rank, rendezvous):
    import torch.distributed as dist

    from rtp_llm.models_py.distributed.collective_torch import Group

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:

        def tp_reduce(value, group):
            assert group == Group.TP
            dist.all_reduce(value)
            return value

        # Identical real BF16 collectives but deliberately different rounding
        # when branches are reduced independently versus combined on each rank.
        for ep_size, can_skip in ((1, True), (1, False), (2, True)):
            expected_skip = ep_size == 1 and can_skip

            class Routed:
                fused_moe = SimpleNamespace(
                    router=SimpleNamespace(
                        tp_collective_size=2, supports_skip_tp_allreduce=can_skip
                    )
                )

                def __call__(self, x, weights, ids, *, skip_tp_allreduce=False):
                    assert skip_tp_allreduce == expected_skip
                    routed = torch.tensor(
                        [256.0 if rank == 0 else -256.0], dtype=torch.bfloat16
                    )
                    return routed if skip_tp_allreduce else tp_reduce(routed, Group.TP)

            owner = SimpleNamespace(
                ffn_tp_size=2,
                ep_size=ep_size,
                experts=Routed(),
                shared_experts=lambda x: torch.ones(1, dtype=torch.bfloat16),
            )
            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce", tp_reduce
            ):
                result = DeepSeekV32MoEBlock._merge_experts(owner, None, None, None)
            torch.testing.assert_close(
                result,
                torch.tensor([1.0 if expected_skip else 2.0], dtype=torch.bfloat16),
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group()


class ScoreRouteTest(unittest.TestCase):
    def test_shared_experts_round_after_the_correct_collective(self):
        with tempfile.TemporaryDirectory() as root:
            torch.multiprocessing.spawn(
                _check_expert_reduction,
                args=((Path(root) / "rendezvous").as_uri(),),
                nprocs=2,
            )

    def test_main_config_load_and_sparse_runtime_ownership(self):
        self._check_score_routes(yarn=False)

    def test_legacy_and_modern_yarn_load_cache_and_attention_scale(self):
        self._check_score_routes(yarn=True)

    def _check_score_routes(self, *, yarn):
        for tag, hf, sparse, interleave in (
            ("deepseek2", "deepseek_v2", False, False),
            ("deepseek3", "deepseek_v3", False, False),
            ("deepseek_v32", "deepseek_v32", True, False),
            ("glm_5", "glm_moe_dsa", True, True),
        ):
            if yarn and tag not in ("deepseek2", "deepseek3"):
                continue
            data = _score_weights()
            cfg = dict(
                model_type=hf,
                hidden_size=32,
                intermediate_size=32,
                max_position_embeddings=128,
                moe_intermediate_size=32,
                topk_method="noaux_tc",
                scoring_func="sigmoid",
                norm_topk_prob=True,
                num_attention_heads=4,
                n_group=1,
                topk_group=1,
                n_routed_experts=4,
                n_shared_experts=1,
                num_experts_per_tok=2,
                num_hidden_layers=2,
                num_nextn_predict_layers=1,
                first_k_dense_replace=1,
                moe_layer_freq=1,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                vocab_size=32,
                routed_scaling_factor=2.5,
                rms_norm_eps=1e-6,
                rope_theta=10000,
                torch_dtype="bfloat16",
            )
            # Two equivalent YaRN schemas must reach both the model-owned
            # cache and the native attention configuration, not just one.
            if yarn:
                schema = "rope_scaling" if tag == "deepseek2" else "rope_parameters"
                cfg[schema] = dict(
                    rope_type="yarn",
                    factor=4,
                    original_max_position_embeddings=32,
                    mscale=0.75,
                    mscale_all_dim=1.25,
                )
            if tag == "deepseek2":
                # V2-Lite: the checkpoint has a direct Q projection, not Q-LoRA.
                cfg["q_lora_rank"] = None
                for layer in range(2):
                    prefix = f"model.layers.{layer}.self_attn."
                    data[prefix + "q_proj.weight"] = data.pop(
                        prefix + "q_a_proj.weight"
                    )
                    del (
                        data[prefix + "q_b_proj.weight"],
                        data[prefix + "q_a_layernorm.weight"],
                    )
            if sparse:
                cfg.update(
                    index_n_heads=2,
                    index_head_dim=128,
                    index_topk=4,
                    indexer_rope_interleave=interleave,
                )
                for layer in range(2):
                    if tag == "glm_5" and layer == 1:
                        # A shared layer has no indexer checkpoint or cache slot.
                        continue
                    prefix = f"model.layers.{layer}.self_attn.indexer."
                    for key, shape in {
                        "wq_b.weight": (256, 32),
                        "wk.weight": (128, 32),
                        "k_norm.weight": (128,),
                        "k_norm.bias": (128,),
                        "weights_proj.weight": (2, 32),
                    }.items():
                        data[prefix + key] = _deterministic_values(shape, 17 + layer)
            data["model.layers.2.embed_tokens.weight"] = data[
                "model.embed_tokens.weight"
            ].clone()
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / "config.json").write_text(json.dumps(cfg))
                save_file(data, str(Path(root) / "model.safetensors"))
                config = DeepSeekV2._create_config(root)
                config.model_type, config.ckpt_path, config.max_seq_len = tag, root, 128
                if tag == "glm_5":
                    config.index_topk_pattern = "FS"
                # Main fills attention page geometry after parsing HF config.
                config.attn_config.tokens_per_block = 64
                config.attn_config.kernel_tokens_per_block = 64
                DeepSeekV2._post_build_model_config(config)
                config.data_type, config.enable_fp32_lm_head, config.lora_infos = (
                    "bf16",
                    False,
                    {},
                )
                for method in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(model=tag, method=method):
                        service = SimpleNamespace(
                            model_config=config,
                            _weight_alias_names=(),
                            _weight_alias_owner=None,
                            _get_device_str=lambda: "cpu",
                            parallelism_config=ParallelismConfig(),
                            hw_kernel_config=None,
                            moe_config=None,
                            fmha_config=None,
                            max_generate_batch_size=1,
                            load_method=method,
                            custom_module=None,
                        )
                        load_owned_model(service)
                        model = service.py_model
                        self.assertIsInstance(model, DeepSeekV32ForCausalLM)
                        if yarn:
                            from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
                                DeepseekV3YarnRotaryEmbedding,
                            )

                            ref = DeepseekV3YarnRotaryEmbedding(
                                4, 128, 10000, scaling_factor=4,
                                original_max_position_embeddings=32,
                                mscale=0.75, mscale_all_dim=1.25,
                            )
                            expected = torch.cat(
                                [ref.cos_cached[:, :2], ref.sin_cached[:, :2]], dim=-1
                            )
                            torch.testing.assert_close(
                                model.cos_sin_cache, expected, rtol=0, atol=0
                            )
                            self.assertAlmostEqual(
                                config.attn_config.softmax_extra_scale,
                                (1 + 0.1 * 1.25 * math.log(4)) ** 2, places=6,
                            )
                        self.assertEqual(
                            service.model_weights_loader.loaded_method,
                            NewLoaderLoadMethod(method),
                        )
                        q_source = weakref.ref(
                            model.layers[0].self_attn.q_a_proj.weight
                        )
                        self.assertTrue(
                            model.initialize(SimpleNamespace(kv_cache=None))
                        )
                        self.assertIs(
                            model.weight.weights[1][W.mla_kc],
                            model.layers[1].self_attn._kc_w,
                        )
                        self.assertEqual(
                            model._get_fmha_group_tags(),
                            ["default", "indexer_kv"] if sparse else None,
                        )
                        if sparse:
                            full_layer = 0 if tag == "glm_5" else 1
                            idx = model.layers[full_layer].indexer
                            if tag == "glm_5":
                                shared = model.layers[1].self_attn
                                self.assertIsNone(shared.indexer)
                                prior = torch.tensor([[2, 0]])
                                self.assertIs(
                                    shared._run_sparse_indexer(
                                        None,
                                        None,
                                        None,
                                        None,
                                        SimpleNamespace(is_sparse=lambda: True),
                                        prior,
                                    ),
                                    prior,
                                )
                            self.assertIs(
                                idx.indexer_op.cos_sin_cache, model.cos_sin_cache
                            )
                            self.assertEqual(
                                idx.indexer_op.is_neox_style, not interleave
                            )
                            torch.testing.assert_close(
                                idx.weights_proj.weight,
                                data[
                                    f"model.layers.{full_layer}.self_attn.indexer.weights_proj.weight"
                                ],
                            )
                            model.to(dtype=torch.bfloat16)
                            x = torch.ones(1, 32, dtype=torch.bfloat16)
                            scales = torch.ones(1, 2, 1)
                            expected = torch.nn.functional.linear(
                                x.float(),
                                data[
                                    f"model.layers.{full_layer}.self_attn.indexer.weights_proj.weight"
                                ],
                            ).unsqueeze(-1) * (idx.softmax_scale * idx.weights_scale)
                            torch.testing.assert_close(
                                idx._get_logits_head_gate(x, scales), expected
                            )
                        else:
                            self.assertIsNone(model.layers[1].indexer)
                        if tag == "deepseek2":
                            manager = service.weight_manager
                            norm = model.layers[0].input_layernorm
                            name = "model.layers.0.input_layernorm.weight"
                            updated = torch.full_like(norm.weight, 2.5)
                            before_ptr = norm.weight.data_ptr()
                            manager.update_tensors({name: updated})
                            self.assertEqual(manager.version, 1)

                            self.assertEqual(norm.weight.data_ptr(), before_ptr)
                            torch.testing.assert_close(
                                norm.weight, updated, rtol=0, atol=0
                            )
                            self.assertIsNone(q_source())
                            # Writing an expired projection would not update the
                            # fused tensor that inference actually consumes.
                            for source in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
                                key = f"model.layers.0.self_attn.{source}.weight"
                                with self.assertRaises(NotImplementedError):
                                    manager.update_tensors({key: data[key]})
                            # The global check still catches changes to a live
                            # parameter even when the RPC targets another one.
                            other = model.layers[0].post_attention_layernorm
                            other.weight = torch.nn.Parameter(
                                other.weight.clone(), requires_grad=False
                            )
                            with self.assertRaisesRegex(
                                RuntimeError,
                                "parameter/storage changed.*post_attention_layernorm",
                            ):
                                manager.update_tensors({name: updated})
                            self.assertEqual(manager.version, 1)

                # The same service configuration can publish a normal classifier
                # instead of a vocabulary projection (including GLM5's mapper).
                from rtp_llm.models.downstream_modules.classifier.classifier import ClassifierHandler
                from rtp_llm.models.downstream_modules.custom_module import CustomModule
                from rtp_llm.ops import TaskType

                cfg["num_labels"] = 3
                (Path(root) / "config.json").write_text(json.dumps(cfg))
                data["classifier.weight"] = _deterministic_values((3, 32), 7).bfloat16()
                data["classifier.bias"] = torch.arange(3).bfloat16()
                save_file(data, str(Path(root) / "model.safetensors"))
                config.task_type = TaskType.SEQ_CLASSIFICATION
                service.custom_module = CustomModule(config, None)
                service.custom_module.handler = ClassifierHandler(config)
                service.custom_module.handler.device = torch.device("cpu")
                load_owned_model(service)
                self.assertIsNone(service.py_model.lm_head)
                linear = service.custom_module.handler.linear
                torch.testing.assert_close(linear.weight, data["classifier.weight"])
                torch.testing.assert_close(linear.bias, data["classifier.bias"])


if __name__ == "__main__":
    unittest.main()

"""V4 draft typed-config/checkpoint/borrow protocol; not GPU execution."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models.deepseek_v4 import DeepSeekV4DSpark
from rtp_llm.models.direct_checkpoint import _runtime_view, load_owned_model
from rtp_llm.models_py.borrowed_parameters import (
    BorrowedParameter,
    bind_borrowed_parameters,
    exclude_borrowed_checkpoint_weights,
    validate_borrowed_parameters,
)
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4_dspark.language import (
    DeepSeekV4DSparkForCausalLM,
)
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType


class Globals(RtpModule):
    def __init__(self):
        super().__init__()
        self.embed = VocabParallelEmbedding(16, 128, params_dtype=torch.bfloat16)
        self.head = ParallelLMHead(16, 128, params_dtype=torch.bfloat16)
        self.load_weights(
            {
                "embed.weight": torch.ones(16, 128, dtype=torch.bfloat16),
                "head.weight": torch.full((16, 128), 2, dtype=torch.bfloat16),
            }
        )


def build(root, commit_only=True):
    raw = dict(
        num_hidden_layers=3,
        hidden_size=128,
        vocab_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=128,
        compress_ratios=[0, 4, 0, 0, 0, 0],
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=2,
        index_topk=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        num_experts_per_tok=2,
        n_routed_experts=4,
        moe_intermediate_size=128,
        n_shared_experts=1,
        hc_mult=2,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        num_hash_layers=0,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=[0, 1],
        dspark_markov_rank=2,
    )
    (root / "config.json").write_text(json.dumps(raw))
    config = DeepSeekV4DSpark._create_config(str(root))
    config.model_type = "deepseek_v4_dspark"
    config.ckpt_path = str(root)
    config.max_seq_len = 128
    config.gen_num_per_cycle = 3
    config.enable_fp32_lm_head = False
    config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
    parallel = ParallelismConfig()
    parallel.role_type = RoleType.PREFILL if commit_only else RoleType.DECODE
    load_config = NewLoaderConfig(
        device="cpu",
        compute_dtype=torch.bfloat16,
        parallelism_config=parallel,
        max_generate_batch_size=2,
    )
    return config, load_config, get_model_class(config.model_type)(config, load_config)


def weights():
    result = {
        "mtp.0.main_proj.weight": torch.ones(128, 256, dtype=torch.float8_e4m3fn),
        "mtp.0.main_proj.scale": torch.full((1, 2), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "mtp.0.main_norm.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    for i in range(3):
        result[f"mtp.{i}.attn.wkv.weight"] = torch.full(
            (512, 128), i + 1, dtype=torch.float8_e4m3fn
        )
        result[f"mtp.{i}.attn.wkv.scale"] = torch.full(
            (4, 1), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        result[f"mtp.{i}.attn.kv_norm.weight"] = torch.full(
            (512,), i + 1, dtype=torch.bfloat16
        )
    # These are excluded before source materialization for commit-only workers.
    result["mtp.0.attn.wq_a.weight"] = torch.ones(1)
    result["mtp.2.markov_head.markov_w1.weight"] = torch.ones(1)
    result["embed.weight"] = torch.full((16, 128), -900, dtype=torch.bfloat16)
    result["head.weight"] = torch.full((16, 128), -900, dtype=torch.bfloat16)
    return result


class V4DraftOwnerTest(unittest.TestCase):
    def test_proposal_dispatches_complete_three_stage_checkpoint(self):
        from rtp_llm.models_py.new_models.deepseek_v4.backbone_test import (
            Dsv4FullBackboneCheckpointTest,
        )

        with tempfile.TemporaryDirectory() as folder:
            cfg, load_cfg, model = build(Path(folder), commit_only=False)
            self.assertIs(type(model), DeepSeekV4DSparkForCausalLM)
            tensors = {}
            for name, value in Dsv4FullBackboneCheckpointTest().weights().items():
                if "compressor" in name or "indexer" in name or "tid2eid" in name:
                    continue
                if name.startswith("layers."):
                    name = "mtp." + name[len("layers.") :]
                elif name in ("embed.weight", "head.weight"):
                    value = value[:16].contiguous()
                else:
                    name = "mtp.2." + name
                tensors[name] = value
            tensors["mtp.0.ffn.gate.bias"] = torch.zeros(4)
            tensors["mtp.2.confidence_head.proj.weight"] = torch.ones(1, 128)
            tensors.update(
                {
                    key: value
                    for key, value in weights().items()
                    if key.startswith("mtp.0.main_")
                }
            )
            for i in (1, 2):
                tensors[f"mtp.2.markov_head.markov_w{i}.weight"] = torch.full(
                    (16, 2), i, dtype=torch.bfloat16
                )
            path = Path(folder) / "pytorch_model.bin"
            torch.save(tensors, path)
            loader = NewModelLoader(cfg, load_cfg, model_path=folder)
            include = model.checkpoint_weight_name_filter()
            resolve = loader._checkpoint_read_spec_resolver(model, include, None)
            model.load_weights(
                get_all_weights(
                    [str(path)], name_filter=include, read_spec_resolver=resolve
                )
            )
            NewModelLoader._validate_loaded_weights(model)
            for name in (
                "mtp.1.attn.wkv.scale",
                "mtp.2.norm.weight",
                "mtp.0.main_proj.weight",
                "mtp.2.markov_head.markov_w2.weight",
            ):
                value = model.get_parameter(model.WEIGHTS_MAPPER.map_name(name))
                self.assertTrue(
                    torch.equal(
                        value.view(torch.uint8), tensors[name].view(torch.uint8)
                    )
                )

    def test_proposal_role_keeps_main_draft_forward_and_owns_three_swa_blocks(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
            DeepSeekV4DSparkModel,
        )

        with tempfile.TemporaryDirectory() as folder:
            cfg, load_cfg, model = build(Path(folder), commit_only=False)
            self.assertIs(type(model).forward, DeepSeekV4DSparkModel.forward)
            self.assertIs(
                type(model)._forward_device, DeepSeekV4DSparkModel._forward_device
            )
            self.assertEqual(len(model.v4.layers), 3)
            self.assertTrue(
                all(block.attn.compress_ratio == 0 for block in model.v4.layers)
            )
            self.assertTrue(all(block.ffn is not None for block in model.v4.layers))
            self.assertTrue(
                all(block.ffn._runtime is None for block in model.v4.layers)
            )
            self.assertIs(
                model.runtime_weight_view()["dspark_markov_w1.weight"],
                model.auxiliary.markov_head.markov_w1.weight,
            )
            self.assertIsNone(model.weight)

    def test_real_commit_checkpoint_and_target_loans(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg, load_cfg, model = build(root)
            self.assertTrue(model.v4.commit_only)
            self.assertIsNone(model.auxiliary.markov_head)
            self.assertIsNone(model.v4.norm)
            self.assertTrue(model.v4.embed.weight.is_meta)
            self.assertTrue(model.v4.head.weight.is_meta)
            target = Globals()
            before = target.head.weight.clone()
            aliases = {
                "v4.embed.weight": BorrowedParameter(target, "embed.weight"),
                "v4.head.weight": BorrowedParameter(target, "head.weight"),
            }
            bind_borrowed_parameters(model, aliases, "cpu")
            torch.save(weights(), root / "pytorch_model.bin")
            loader = NewModelLoader(cfg, load_cfg, model_path=folder)
            include = exclude_borrowed_checkpoint_weights(
                model, model.checkpoint_weight_name_filter()
            )
            resolver = loader._checkpoint_read_spec_resolver(model, include, None)
            model.load_weights(
                get_all_weights(
                    [str(root / "pytorch_model.bin")],
                    name_filter=include,
                    read_spec_resolver=resolver,
                )
            )
            NewModelLoader._validate_loaded_weights(model)
            validate_borrowed_parameters(model, "cpu")
            self.assertIs(model.v4.embed.weight, target.embed.weight)
            self.assertIs(model.v4.head.weight, target.head.weight)
            torch.testing.assert_close(target.head.weight, before)
            self.assertEqual(len(list(model.named_parameters())), 14)
            self.assertIs(model.main_proj, model.auxiliary.main_proj)
            self.assertIs(model.main_norm, model.auxiliary.main_norm)
            self.assertEqual(set(model.runtime_weight_view()), {"embedding", "lm_head"})
            for block in model.v4.layers:
                self.assertIsNone(block.ffn)
                self.assertIsNone(block.attn.wq_a)
            self.assertIsNone(model.weight)

            _, _, broken = build(root)
            bind_borrowed_parameters(broken, aliases, "cpu")
            pointer = target.head.weight.data_ptr()
            NewModelLoader._prepare_model_for_fastsafetensors(broken, "cpu")
            validate_borrowed_parameters(broken, "cpu")
            incomplete = weights()
            del incomplete["mtp.2.attn.wkv.scale"]
            torch.save(incomplete, root / "pytorch_model.bin")
            include = exclude_borrowed_checkpoint_weights(
                broken, broken.checkpoint_weight_name_filter()
            )
            resolver = loader._checkpoint_read_spec_resolver(broken, include, None)
            broken.load_weights(
                get_all_weights(
                    [str(root / "pytorch_model.bin")],
                    name_filter=include,
                    read_spec_resolver=resolver,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "missing|required|Missing"):
                NewModelLoader._validate_loaded_weights(broken)
            self.assertIs(broken.v4.head.weight, target.head.weight)
            self.assertEqual(target.head.weight.data_ptr(), pointer)
            torch.testing.assert_close(target.head.weight, before)
            published = _runtime_view(model, cfg, "cpu")
            model.weight = published
            model._release_runtime_weight_source()
            self.assertIs(model.weight.get_global_weight("lm_head"), target.head.weight)

    def test_explicit_service_backends_do_not_publish_a_cpu_v4(self):
        # Service -> real registry/import -> real role-specific owner -> device
        # preflight. CPU cannot execute these checkpoints; neither explicit
        # method may turn a preflight failure into success or publish a model.
        from rtp_llm.models.deepseek_v4 import DeepSeekV4

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            draft, load_cfg, _ = build(root, commit_only=False)
            from safetensors.torch import save_file

            save_file({"unused": torch.ones(1)}, str(root / "model.safetensors"))
            target = DeepSeekV4._create_config(folder)
            target.model_type = "deepseek_v4"
            target.ckpt_path = folder
            target.max_seq_len = 128
            target.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            for config in (draft, target):
                config.data_type = "bf16"
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(model=config.model_type, method=method):
                        instance = service(folder, method)
                        instance.model_config = config
                        instance.parallelism_config = load_cfg.parallelism_config
                        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                            load_owned_model(instance)
                        self.assertIsNone(instance.py_model)
                        self.assertIsNone(instance.weight)
                        self.assertIsNone(instance.model_weights_loader)

            # The wrong placement can otherwise look complete (e.g. a
            # permutation within one rank). Check the actual public FST
            # entry fails for the placement before reaching CUDA admission.
            placement_path = root / "phy2log.json"
            placement_path.write_text(
                json.dumps([[0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 2, 3]])
            )
            target.phy2log_path = str(placement_path)
            instance = service(folder, "fastsafetensors")
            instance.model_config = target
            instance.parallelism_config = load_cfg.parallelism_config
            with self.assertRaisesRegex(
                ValueError, "layer 1 requires identity phy2log"
            ):
                load_owned_model(instance)
            self.assertIsNone(instance.py_model)
            self.assertIsNone(instance.weight)

    def test_commit_requires_both_loans_before_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, model = build(Path(folder))
            with self.assertRaisesRegex(ValueError, "before dispatch"):
                model.validate_runtime_device("cpu")
            with self.assertRaisesRegex(ValueError, "before dispatch"):
                model.load_weights(weights())
            target = Globals()
            with self.assertRaisesRegex(ValueError, "both target globals"):
                bind_borrowed_parameters(
                    model,
                    {"v4.embed.weight": BorrowedParameter(target, "embed.weight")},
                    "cpu",
                )


if __name__ == "__main__":
    unittest.main()

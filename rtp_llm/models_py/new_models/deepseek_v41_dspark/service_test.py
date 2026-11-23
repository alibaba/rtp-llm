"""Real typed-config/main-runtime draft staging; cold interfaces, not GPU graph."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v41 import DeepSeekV41DSpark
from rtp_llm.models.direct_checkpoint import RuntimeWeightView, _runtime_view
from rtp_llm.models_py.borrowed_parameters import (
    BorrowedParameter,
    aliases_from_runtime_exports,
    bind_borrowed_parameters,
)
from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
    DeepSeekV41DSparkModel,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41_commit import (
    CommitOnlyAttentionV41FP8,
)
from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.new_models.deepseek_v41 import service_test as target_fixture
from rtp_llm.models_py.new_models.deepseek_v41_dspark.checkpoint_test import (
    _TargetGlobals,
    load_into,
    values,
)
from rtp_llm.models_py.new_models.deepseek_v41_dspark.language import (
    DeepSeekV41DSparkForCausalLM,
)
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType


def config_and_load(directory, rank, commit):
    raw = target_fixture.raw_config()
    raw.update(
        num_nextn_predict_layers=3,
        compress_ratios=[0, 2, 1, 0, 0, 0],
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
        dspark_noise_token_id=1,
    )
    (Path(directory) / "config.json").write_text(json.dumps(raw))
    config = DeepSeekV41DSpark._create_config(directory)
    config.model_type, config.ckpt_path, config.data_type = (
        "deepseek_v41_dspark",
        directory,
        "bf16",
    )
    config.gen_num_per_cycle = 3
    config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.ep_size = parallel.world_size = 2
    parallel.tp_rank = parallel.ep_rank = parallel.world_rank = rank
    parallel.role_type = RoleType.PREFILL if commit else RoleType.DECODE
    load = NewLoaderConfig.from_runtime(
        parallel,
        device="cpu",
        compute_dtype=torch.bfloat16,
        load_method="scratch",
        max_generate_batch_size=2,
    )
    return config, load


class V41DraftServiceTest(unittest.TestCase):
    def test_stats_owner_matches_commit_and_decode_roles(self):
        from rtp_llm.models_py.eplb_controller import create_stage_expert_balancer

        for commit in (False, True):
            with self.subTest(commit=commit), tempfile.TemporaryDirectory() as directory:
                save_file(values(commit), str(Path(directory) / "model.safetensors"))
                config, load = config_and_load(directory, 0, commit)
                config.eplb_config.eplb_mode = "STATS"
                model = DeepSeekV41DSparkForCausalLM(config, load)
                controller = create_stage_expert_balancer(SimpleNamespace(load_config=load), model, config)
                self.assertTrue(controller.eplb_stats_only)
                logical = torch.zeros(config.num_layers, config.expert_num, dtype=torch.int32)
                gpu = torch.zeros(config.num_layers, load.ep_size, dtype=torch.int32)
                model._newloader_eplb_bind_stats(logical, gpu)
                for i, block in enumerate(model.v4.layers):
                    if commit:
                        self.assertIsNone(block.ffn)
                    else:
                        self.assertEqual(block.ffn._eplb_stats.logical.data_ptr(), logical[i].data_ptr())

    def test_actual_target_image_wrapper_exports_bind_draft_without_copy(self):
        # Use the actual target owner, including its runtime image wrapper,
        # instead of assuming export keys equal named_parameter paths.
        for commit in (False, True):
            with self.subTest(
                commit=commit
            ), tempfile.TemporaryDirectory() as directory:
                target, target_config, _, _ = (
                    target_fixture.V41ServiceOwnerTest().staged(directory)
                )
                target_view = _runtime_view(target, target_config, "cpu")
                target.weight = target_view
                target._release_runtime_weight_source()
                self.assertIs(target.v4.embed.owner, target)
                before = target.v4.head.weight.clone()
                path = Path(directory) / "model.safetensors"
                save_file(values(commit), str(path))
                config, load = config_and_load(directory, 0, commit)
                loans = aliases_from_runtime_exports(
                    target,
                    {
                        name: target_view.get_global_weight(name)
                        for name in ("embedding", "lm_head")
                    },
                    DeepSeekV41DSparkForCausalLM.runtime_parameter_paths,
                )
                draft = NewModelLoader(
                    config, load, model_path=directory, parameter_aliases=loans
                )._create_model()
                self.assertTrue(draft.v4.embed.weight.is_meta)
                self.assertTrue(draft.v4.head.weight.is_meta)
                bind_borrowed_parameters(draft, loans, "cpu")
                load_into(draft, path, 0, native_fst=True)
                self.assertIs(draft.v4.embed.weight, target.v4.embed.base.weight)
                self.assertIs(draft.v4.head.weight, target.v4.head.weight)
                torch.testing.assert_close(target.v4.head.weight, before)

    def staged(self, directory, rank=0, commit=False, fst=False):
        path = Path(directory) / "model.safetensors"
        save_file(values(commit), str(path))
        config, load = config_and_load(directory, rank, commit)
        model = DeepSeekV41DSparkForCausalLM(config, load)
        target = _TargetGlobals(rank)
        target.load_weights(
            get_all_weights(
                [str(path)],
                name_filter=lambda name: name in ("embed.weight", "head.weight"),
            )
        )
        bind_borrowed_parameters(
            model,
            {
                "v4.embed.weight": BorrowedParameter(target, "embed.weight"),
                "v4.head.weight": BorrowedParameter(target, "head.weight"),
            },
            "cpu",
        )
        load_into(model, path, rank, native_fst=fst)
        return model, target, config

    def test_real_config_draft_runtime_and_target_borrowed_parameters(self):
        self.assertIs(
            DeepSeekV41DSparkForCausalLM.forward, DeepSeekV41DSparkModel.forward
        )
        self.assertIs(
            DeepSeekV41DSparkForCausalLM.forward_commit,
            DeepSeekV41DSparkModel.forward_commit,
        )
        self.assertIs(
            DeepSeekV41DSparkForCausalLM.forward_propose,
            DeepSeekV41DSparkModel.forward_propose,
        )
        for commit in (False, True):
            for rank in (0, 1):
                with self.subTest(
                    commit=commit, rank=rank
                ), tempfile.TemporaryDirectory() as directory:
                    model, target, config = self.staged(
                        directory, rank, commit, fst=True
                    )
                    self.assertIsInstance(
                        model.v4.layers[0].attn,
                        CommitOnlyAttentionV41FP8 if commit else AttentionV41FP8,
                    )
                    if not commit:
                        self.assertIsInstance(model.v4.layers[0].attn_hc, DelayedHCUnit)
                    self.assertFalse(model._captures_aux_hidden)
                    self.assertIs(model.v4.embed.weight, target.embed.weight)
                    self.assertIs(model.v4.head.weight, target.head.weight)
                    self.assertEqual(config.deepseek_v41_config["engram_layer_ids"], [])
                    if not commit:
                        self.assertTrue(
                            all(
                                block.attn.kv_source_layer_id is None
                                for block in model.v4.layers
                            )
                        )
                    self.assertEqual(
                        len(
                            {
                                block.attn.wkv.weight.data_ptr()
                                for block in model.v4.layers
                            }
                        ),
                        3,
                    )
                    model.weight = _runtime_view(model, config, "cpu")
                    exports = dict(model.weight.global_weights)
                    model._release_runtime_weight_source()
                    self.assertFalse(hasattr(model, "weight"))
                    for key, value in model.runtime_weight_view().items():
                        self.assertIs(value, exports[key])
                    self.assertFalse(hasattr(model, "_engram_hash_state"))

    def test_runtime_view_cannot_replace_borrowed_head_or_markov(self):
        with tempfile.TemporaryDirectory() as directory:
            model, target, config = self.staged(directory)
            before = target.head.weight.clone()
            exports = model.runtime_weight_view()
            exports["dspark_markov_w1.weight"] = exports[
                "dspark_markov_w1.weight"
            ].clone()
            model.weight = RuntimeWeightView(
                [{}, {}, {}], exports, config.compute_dtype, "cpu"
            )
            with self.assertRaisesRegex(RuntimeError, "replace owned parameters"):
                model._release_runtime_weight_source()
            torch.testing.assert_close(target.head.weight, before)

    def test_real_gpu_backend_not_misreported_as_cpu_runtime_support(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _, _ = self.staged(directory)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                NewModelLoader._validate_runtime_backends(model, "cpu")


if __name__ == "__main__":
    unittest.main()

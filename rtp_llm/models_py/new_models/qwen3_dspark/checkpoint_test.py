"""Real tiny checkpoints and CPU parameter/runtime math, not a GPU smoke."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.borrowed_parameters import (
    BorrowedParameter,
    validate_borrowed_parameters,
)
from rtp_llm.models_py.model_desc.qwen3_dspark_runtime import (
    _apply_non_interleaved_rope,
    _write_rocm_paged_kv_cache,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen3_dspark import Qwen3DSparkForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class


def config(draft=True):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import RopeStyle

    # Let the production config derive rank-local heads, RoPE cache length
    # and kernel block geometry; CUDA initializes the real context RoPE on load.
    cfg = ModelConfig()
    cfg.model_type = "qwen_3_dspark" if draft else "qwen_3"
    cfg.num_layers = 2
    cfg.is_mtp = draft
    cfg.vocab_size = 4 if draft else 8
    cfg.input_vocab_size = 8
    cfg.hidden_size = cfg.inter_size = 4
    cfg.max_seq_len = 32
    cfg.layernorm_eps = 1e-6
    cfg.enable_fp32_lm_head = False
    cfg.dspark_target_layer_ids = [0, 3]
    cfg.gen_num_per_cycle = 3
    cfg.dspark_sample_from_anchor = False
    cfg.dspark_noise_token_id = 7
    cfg.dspark_markov_rank = 2
    attn = cfg.attn_config
    attn.head_num, attn.kv_head_num, attn.size_per_head = 2, 1, 2
    attn.is_causal = not draft
    attn.tokens_per_block = 4
    attn.rope_config.style = RopeStyle.Base
    attn.rope_config.dim = attn.size_per_head
    attn.rope_config.base = 10000
    return cfg


def load_config(tp=1, rank=0):
    parallel = SimpleNamespace(
        tp_size=tp,
        tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        pp_stage_layer_counts=[],
        dp_size=1,
        dp_rank=0,
        world_size=tp,
        world_rank=rank,
        get_attn_tp_size=lambda: tp,
        get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp,
        get_ffn_tp_rank=lambda: rank,
    )
    return NewLoaderConfig(
        tp_size=tp,
        tp_rank=rank,
        compute_dtype=torch.float32,
        device="cpu",
        load_method="scratch",
        quant_config=QuantizationConfig("none"),
        parallelism_config=parallel,
    )


def checkpoint(draft=True):
    weights = {
        "model.embed_tokens.weight": torch.arange(32.0).reshape(8, 4),
        "lm_head.weight": torch.arange(16.0 if draft else 32.0).reshape(-1, 4),
        "model.norm.weight": torch.ones(4),
    }
    for i in range(2):
        prefix = f"model.layers.{i}."
        for name, rows in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2)):
            weights[prefix + f"self_attn.{name}.weight"] = (
                torch.arange(rows * 4.0).reshape(rows, 4) / 10 + i
            )
        weights.update(
            {
                prefix + "self_attn.o_proj.weight": torch.eye(4),
                prefix + "self_attn.q_norm.weight": torch.ones(2),
                prefix + "self_attn.k_norm.weight": torch.ones(2),
                prefix + "input_layernorm.weight": torch.ones(4),
                prefix + "post_attention_layernorm.weight": torch.ones(4),
                prefix + "mlp.gate_proj.weight": torch.ones(4, 4),
                prefix + "mlp.up_proj.weight": torch.ones(4, 4),
                prefix + "mlp.down_proj.weight": torch.ones(4, 4),
            }
        )
    if draft:
        weights.update(
            {
                "fc.weight": torch.arange(32.0).reshape(4, 8) / 10,
                "hidden_norm.weight": torch.ones(4),
                "markov_head.markov_w1.weight": torch.arange(16.0).reshape(8, 2),
                "markov_head.markov_w2.weight": torch.arange(8.0).reshape(4, 2),
                "d2t": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            }
        )
    return weights


class DSparkLoadTest(unittest.TestCase):
    def test_pipeline_checkpoint_geometry_without_compute_extension(self):
        # No C++ graph/attention objects are constructed for CPU checkpoint IO.
        attn = SimpleNamespace(
            head_num=2, kv_head_num=1, size_per_head=2, is_causal=False
        )
        cfg = SimpleNamespace(
            model_type="qwen_3_dspark",
            num_layers=2,
            is_mtp=True,
            vocab_size=4,
            input_vocab_size=8,
            hidden_size=4,
            inter_size=4,
            attn_config=attn,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[],
            enable_fp32_lm_head=False,
            layernorm_eps=1e-6,
            dspark_target_layer_ids=[0, 3],
            gen_num_per_cycle=3,
            dspark_noise_token_id=7,
            dspark_markov_rank=2,
            getAttentionConfigs=lambda _: attn,
        )
        self._check_last_stage(cfg)

    def test_last_pipeline_stage_loads_the_entire_self_contained_draft(self):
        self._check_last_stage(config())

    def _check_last_stage(self, cfg):
        weights = checkpoint()
        with tempfile.TemporaryDirectory() as root:
            save_file(weights, str(Path(root) / "model.safetensors"))

            def worker(rank):
                from torch import distributed as dist

                dist.init_process_group(
                    "gloo",
                    init_method=Path(root, "rdv").as_uri(),
                    rank=rank,
                    world_size=2,
                    timeout=timedelta(seconds=30),
                )
                try:
                    tail = dist.new_group([1], backend="gloo")
                    if rank == 1:

                        class Loader(NewModelLoader):
                            def _runtime_checkpoint_group(self, group):
                                from rtp_llm.models_py.distributed.rank_layout import (
                                    Group,
                                )

                                assert group == Group.STAGE
                                return tail

                        self._check_tail_methods(root, weights, cfg, Loader)
                    dist.barrier()
                finally:
                    dist.destroy_process_group()

            torch.multiprocessing.start_processes(worker, nprocs=2, start_method="fork")

    def _check_tail_methods(self, root, weights, cfg, loader_cls):
        for method in ("scratch", "fastsafetensors"):
            load = replace(load_config(), load_method=method)
            pc = load.parallelism_config
            pc.pp_size, pc.pp_rank = 2, 1
            pc.world_size, pc.world_rank = 2, 1
            pc.pp_stage_layer_counts = [1, 1]
            model = loader_cls(cfg, load, model_path=root).load()
            self.assertEqual(model._pipeline_partitioned, False)
            self.assertEqual(tuple(model.layer_ids()), (0, 1))
            self.assertEqual(len(model.layers), 2)
            torch.testing.assert_close(
                model.embed_tokens.weight, weights["model.embed_tokens.weight"]
            )
            torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])
            torch.testing.assert_close(model.fc.weight, weights["fc.weight"])

    def load(self, weights, cfg=None, aliases=None, tp=1, rank=0):
        with tempfile.TemporaryDirectory() as root:
            save_file(weights, str(Path(root) / "model.safetensors"))
            return NewModelLoader(
                cfg or config(),
                load_config(tp, rank),
                model_path=root,
                parameter_aliases=aliases,
            ).load()

    def test_production_registration_and_real_scratch_checkpoint(self):
        from rtp_llm.ops import RopeStyle

        self.assertIs(get_model_class("qwen_3_dspark"), Qwen3DSparkForCausalLM)
        model = self.load(checkpoint())
        # The CUDA graph target loads this same checkpoint and uses these
        # native fields when it constructs MhaRotaryEmbeddingOp.
        attn = model.attn_configs
        self.assertEqual(attn.rope_config.style, RopeStyle.Base)
        self.assertEqual(attn.rope_config.dim, attn.size_per_head)
        self.assertEqual(attn.max_seq_len + attn.gen_num_per_cycle + 1, 36)
        self.assertEqual(attn.kernel_tokens_per_block, 4)
        self.assertEqual(model.d2t.tolist(), [0, 2, 4, 6])
        self.assertEqual(tuple(model.embed_tokens.weight.shape), (8, 4))
        self.assertEqual(tuple(model.lm_head.weight.shape), (4, 4))
        exports = model.runtime_weight_view()
        self.assertIs(
            exports["dspark_markov_w1.weight"], model.markov_head.markov_w1.weight
        )
        self.assertIs(exports["multi_tokens_predict_d2t_map"], model.d2t)
        self.assertEqual(model._dspark_width, 3)
        self.assertEqual(model._dspark_query_width, 4)
        features = torch.arange(16.0).reshape(2, 8)
        torch.testing.assert_close(
            model.combine_hidden_states(features),
            features @ checkpoint()["fc.weight"].T,
        )
        self.assertFalse(
            any("context_kv_projection" in name for name, _ in model.named_parameters())
        )

    def test_tp2_each_rank_parameter_slices(self):
        weights = checkpoint()
        for rank in range(2):
            with self.subTest(rank=rank):
                model = self.load(weights, tp=2, rank=rank)
                torch.testing.assert_close(
                    model.embed_tokens.weight,
                    weights["model.embed_tokens.weight"][rank * 4 : (rank + 1) * 4],
                )
                torch.testing.assert_close(
                    model.lm_head.weight,
                    weights["lm_head.weight"][rank * 2 : (rank + 1) * 2],
                )
                torch.testing.assert_close(model.fc.weight, weights["fc.weight"])
                torch.testing.assert_close(
                    model.markov_head.markov_w1.weight.float(),
                    weights["markov_head.markov_w1.weight"],
                )

    def test_explicit_fst_after_close_matches_scratch(self):
        weights = checkpoint()
        expected = self.load(weights)
        with tempfile.TemporaryDirectory() as root:
            save_file(weights, str(Path(root) / "model.safetensors"))
            load = replace(load_config(), load_method="fastsafetensors")
            loader = NewModelLoader(config(), load, model_path=root)
            actual = loader.load()
            self.assertEqual(loader._selected_load_method, "fastsafetensors")
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], tensor)

    def test_absolute_mapping_and_missing_mapping(self):
        weights = checkpoint()
        weights["model.d2t"] = weights.pop("d2t")
        model = self.load(weights)
        self.assertEqual(model.d2t.tolist(), [0, 1, 2, 3])
        del weights["model.d2t"]
        with self.assertRaisesRegex(ValueError, "requires d2t"):
            self.load(weights)

    def test_borrowed_embedding_survives_success_and_failed_load(self):
        owner = self.load(checkpoint(False), config(False))
        original = owner.embed_tokens.weight
        pointer = original.data_ptr()
        before = original.clone()
        weights = checkpoint()
        weights["model.embed_tokens.weight"].fill_(-900)
        alias = {"embed_tokens.weight": BorrowedParameter(owner, "embed_tokens.weight")}
        draft = self.load(weights, aliases=alias)
        self.assertIs(draft.embed_tokens.weight, original)
        self.assertEqual(draft.embed_tokens.weight.data_ptr(), pointer)
        torch.testing.assert_close(original, before)
        validate_borrowed_parameters(draft, "cpu")
        weights["fc.weight"] = torch.zeros(2, 2)
        with self.assertRaises((ValueError, RuntimeError)):
            self.load(weights, aliases=alias)
        torch.testing.assert_close(original, before)
        with torch.inference_mode():
            owner.embed_tokens.weight.add_(2)
        torch.testing.assert_close(draft.embed_tokens.weight, before + 2)

    def test_rope_and_rocm_cache_layout_cpu_math(self):
        q = torch.arange(8.0).reshape(1, 1, 8)
        k = q.clone()
        _apply_non_interleaved_rope(
            q, k, torch.tensor([[0.0] * 4 + [1.0] * 4]), torch.tensor([0]), 8
        )
        torch.testing.assert_close(
            q, torch.tensor([[[-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]]])
        )
        for vectorized in (False, True):
            cache = torch.zeros(2, 2, 1, 8, 8, dtype=torch.float32)
            value = torch.arange(8.0).reshape(1, 1, 8) + 10
            _write_rocm_paged_kv_cache(
                cache,
                torch.tensor([1]),
                torch.tensor([5]),
                k,
                value,
                vectorized_value=vectorized,
            )
            key_physical = cache.view(2, 2, 1, 2, 8, 4)[:, 0]
            torch.testing.assert_close(key_physical[1, :, :, 5, :].reshape(1, 1, 8), k)
            if vectorized:
                read_value = cache.view(2, 2, 1, 2, 8, 4)[1, 1, :, 1, :, 1]
            else:
                read_value = cache.view(2, 2, 1, 8, 8)[1, 1, :, :, 5]
            torch.testing.assert_close(read_value, value[0])

    def test_borrowed_head_is_not_postprocessed_twice(self):
        target_config = config(False)
        target_config.normalize_lm_head_weight = True
        target_config.logit_scale = 0.25
        owner = self.load(checkpoint(False), target_config)
        before = owner.lm_head.weight.clone()
        draft_config = config()
        draft_config.vocab_size = 8
        weights = checkpoint()
        weights["lm_head.weight"] = torch.full((8, 4), -900.0)
        weights["markov_head.markov_w2.weight"] = torch.zeros(8, 2)
        weights["d2t"] = torch.zeros(8, dtype=torch.int64)
        aliases = {
            "embed_tokens.weight": BorrowedParameter(owner, "embed_tokens.weight"),
            "lm_head.weight": BorrowedParameter(owner, "lm_head.weight"),
        }
        draft = self.load(weights, draft_config, aliases=aliases)
        self.assertIs(draft.lm_head.weight, owner.lm_head.weight)
        torch.testing.assert_close(draft.lm_head.weight, before)
        draft_config.logit_scale = 2.0
        with self.assertRaisesRegex(ValueError, "another postprocess"):
            self.load(weights, draft_config, aliases=aliases)
        torch.testing.assert_close(owner.lm_head.weight, before)


if __name__ == "__main__":
    unittest.main()
